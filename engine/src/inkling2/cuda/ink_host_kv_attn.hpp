#pragma once
#include <cstdint>
#include <cstddef>

namespace strata::ink {

// Deduplicated physical host-RAM KV block shared by 1..16 concurrent decoding slots!
// Each physical block is read from DDR4 ONCE and dotted against every active slot in `slot_mask`!
struct InkHostKvUniqueBlock {
    int32_t  slab;         // index into slab_raw_k / slab_raw_v
    int32_t  boff;         // block offset inside slab (0..255)
    int32_t  pos_start;    // global token position of first token in this block (blk * 256)
    int32_t  max_pos_len;  // max valid token count across slots in slot_mask (1..256)
    uint32_t slot_mask;    // bitmask of m in 0..M-1 (M <= 16) that attend this block
};

// Exact AVX2+FMA+F16C multi-threaded GQA-3 (24 Q : 8 KV, head_dim=128) attention over deduplicated FP16 Host KV slabs.
// Writes contiguous partial softmax outputs:
//   out_buf layout per (m in 0..M-1, h in 0..23): [128 floats o, 1 float m, 1 float l] = 130 floats!
void ink_host_kv_attn_fp16(
    const float* q,                       // [M, 24, 128]
    const float* rel_table,               // [M, 24, 1025] (may be nullptr if no host block has qp - pos < 1024)
    const void* const* slab_raw_k,        // [n_slabs] raw host __half* [256, 256, 8, 128]
    const void* const* slab_raw_v,        // [n_slabs] raw host __half* [256, 256, 8, 128]
    const int32_t* slot_qp,               // [M] query position for each active batch slot
    const InkHostKvUniqueBlock* ublocks,  // [n_ublocks] deduplicated physical blocks
    int n_ublocks,
    int M,
    float* out_buf,                       // [M, 24, 130]: 128 floats o, float m, float l
    int n_threads = 0);

}  // namespace strata::ink

// ---- Stage F: FreeKV (ICLR 2026) + HERALD (Nov 2026) Speculative Sparse 1M CPU Lookahead ----
namespace strata::ink {

// Per-block Quest/FreeKV min-max key summary: [8 KV heads][128 dims] FP16 min + [8][128] FP16 max = 4 KiB per 256-tok block (128x smaller than K+V!)
struct InkBlockSummary {
    uint16_t k_min[8 * 128];
    uint16_t k_max[8 * 128];
    uint8_t  valid;
};

// Compute min/max FP16 key summary for one finalized 256-token host block using AVX2+F16C (< 5 us!)
void ink_host_kv_build_summary_fp16(const void* slab_raw_k, int boff, int pos_len, InkBlockSummary* out_sum);

// Speculatively score all host blocks using query q[M, 24, 128] (either Q(t-1) during SWA visits or real Q(t) on correction)
// and return indices into `ublocks` of the Top-K critical blocks (+ sink + recent blocks), sorted by block order!
int ink_host_kv_select_topk_fp16(
    const float* q,                       // [M, 24, 128]
    const void* const* slab_raw_k,        // [n_slabs] raw host __half*
    InkBlockSummary* const* slab_sums,    // [n_slabs] array of 256 InkBlockSummary per slab (lazily populated!)
    const InkHostKvUniqueBlock* ublocks,  // [n_ublocks] candidate past host blocks
    int n_ublocks,
    int M,
    int top_k,                            // max blocks to select (<= 44)
    int* out_indices);                    // [top_k] selected indices into ublocks

// FreeKV Section 3.3: min group-pooled cosine similarity between Q(t) and Q(t-1) across all active slots & 8 KV groups!
float ink_host_kv_min_q_cosine(const float* q_cur, const float* q_prev, int M);

}  // namespace strata::ink

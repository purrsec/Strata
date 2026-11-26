// engine/src/inkling2/cuda/ink_dflash2_cpu.hpp
// AVX2+FMA+F16C CPU DFlash2 Drafter + DF2P Pack Loader + Zero-VRAM RAM Selector for Inkling-2-Mini.
// Supports both full CPU 0-VRAM drafting (with GPU lm_head) and zero-VRAM RAM selector for Hybrid/Stream GPU modes.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::ink {

enum class Df2Dtype : int {
    F32  = 0,
    BF16 = 1,
    Q8_0 = 2,
    Q4_0 = 3,
};

struct Df2Tensor {
    std::string name;
    Df2Dtype dtype = Df2Dtype::F32;
    int rows = 0;
    int cols = 0;
    size_t offset = 0;
    size_t bytes = 0;
    const uint8_t* ptr = nullptr;
};

struct Df2Pack {
    int hidden = 2304;
    int layers = 5;
    int heads = 32;
    int kv_heads = 8;
    int head_dim = 128;
    int inter = 6912;
    int window = 2048;
    float rope_theta = 10000.0f;
    float eps = 1e-6f;
    int block = 8;
    int conv_group = 16;
    int conv_kernel = 2;
    int mask_token = 200056;
    int rank = 256;
    int top_k = 16;
    int vocab = 201024;
    float softcap = 30.0f;
    std::vector<int> taps;

    std::vector<uint8_t> raw_storage; // 64-byte aligned via padded allocation
    uint8_t* data_base = nullptr;
    size_t data_bytes = 0;
    std::vector<Df2Tensor> tensors;
    std::unordered_map<std::string, int> name_to_idx;

    bool load(const std::string& path, std::string& err);
    const Df2Tensor* find(const std::string& name) const;
    const Df2Tensor& require(const std::string& name) const;
    bool is_q4_backbone() const;
};

// Persistent 16-thread AVX2+FMA+F16C CPU DFlash2 Drafter + Zero-VRAM Selector
class InkDFlash2Cpu {
public:
    static constexpr int kRingCap = 2064;
    static constexpr int kMaxLayers = 8;

    InkDFlash2Cpu();
    ~InkDFlash2Cpu();
    InkDFlash2Cpu(const InkDFlash2Cpu&) = delete;
    InkDFlash2Cpu& operator=(const InkDFlash2Cpu&) = delete;

    // Bind to an already-loaded Df2Pack (must outlive InkDFlash2Cpu).
    // If `alloc_rings` is false, only the zero-VRAM selector is initialized (for Hybrid/Stream GPU modes!).
    bool init(const Df2Pack* pack, int max_slots, bool alloc_rings, int num_threads = 0);
    void free_all();

    void reset_slot(int slot);

    // Context ingest on CPU: taps4 = [4] pointers each [n, 2304] F32 for positions [p0 .. p0 + n - 1].
    void ingest_taps(int slot, const float* const taps4[4], int p0, int n);

    // Block-8 forward on CPU:
    //   embed_in: [8, 2304] F32 (anchor + 7 mask embeddings from target)
    //   anchor_pos: position of anchor token (s >= 1)
    //   out_hidden7: [7, 2304] F32 output (rows 1..7 after final RMSNorm) ready for target lm_head!
    void forward_block(int slot, const float* embed_in, int anchor_pos, float* out_hidden7);

    // Zero-VRAM RAM Selector walk (used by CPU, Hybrid GPU, and PCIe-Stream GPU modes! < 0.05 ms):
    //   out_hidden7: [7, 2304] F32 (rows 1..7 after final RMSNorm)
    //   unary16:     [7, 16] F32 raw lm_head logits (before softcap)
    //   cand16:      [7, 16] int32 token ids
    //   anchor_tok:  int32 anchor token id
    //   out_ids7:    [7] selected draft token ids
    //   out_conf7:   [7] softmax confidence of each pick after softcap(30.0)
    void select_path(
        const float* out_hidden7,
        const float* unary16,
        const int32_t* cand16,
        int32_t anchor_tok,
        int32_t* out_ids7,
        float* out_conf7) const;

    // Standalone matmul helper exposed for unit/parity tests
    void matmul(const Df2Tensor& w, const float* x, float* y, int M) const;

    int num_threads() const;
    bool rings_ready() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::ink

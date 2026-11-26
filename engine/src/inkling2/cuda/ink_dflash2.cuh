// engine/src/inkling2/cuda/ink_dflash2.cuh
// Unified DFlash2 Block-8 Speculative Drafter for Inkling-2-Mini on CUDA.
// Supports 4 execution modes from the same DF2P pack file (tools/dflash2_pack.py):
//   - HYBRID (~275 MB VRAM): GPU Q4_0/Q8_0 backbone + GPU SWA-2048 ring + ZERO-VRAM CPU RAM selector + shared target lm_head
//   - GPU    (~333 MB VRAM): Same + GPU Q4_0 selector walk
//   - STREAM (~130 MB VRAM): Zero persistent weight VRAM; 2-layer ping-pong async PCIe Gen4 DMA + CPU RAM selector
//   - CPU    (~0.5 MB VRAM): Full 16-thread AVX2+FMA+F16C CPU backbone & SWA-2048 ring + shared target GPU lm_head
#pragma once

#include "ink_dflash2_cpu.hpp"
#include "ink_runner.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace strata::ink {

enum class Df2Mode : int {
    HYBRID = 0, // GPU backbone + GPU ring + CPU RAM selector (~275 MB VRAM in Q4_0)
    GPU    = 1, // GPU backbone + GPU ring + GPU selector (~333 MB VRAM in Q4_0)
    STREAM = 2, // Ping-pong 2-layer PCIe DMA backbone + GPU ring + CPU RAM selector (~130 MB VRAM)
    CPU    = 3, // Full AVX2 CPU backbone & ring + GPU shared lm_head (~0.5 MB VRAM)
};

inline const char* df2_mode_name(Df2Mode m) {
    switch (m) {
        case Df2Mode::HYBRID: return "hybrid";
        case Df2Mode::GPU:    return "gpu";
        case Df2Mode::STREAM: return "stream";
        case Df2Mode::CPU:    return "cpu";
    }
    return "hybrid";
}

inline Df2Mode parse_df2_mode(const char* s) {
    if (!s || !s[0]) return Df2Mode::HYBRID;
    std::string v(s);
    for (auto& c : v) c = (char)std::tolower((unsigned char)c);
    if (v == "cpu") return Df2Mode::CPU;
    if (v == "gpu") return Df2Mode::GPU;
    if (v == "stream" || v == "pcie") return Df2Mode::STREAM;
    return Df2Mode::HYBRID;
}

struct alignas(8) Df2GpuActQ8 {
    int8_t q[32];
    float d;
    float s; // d * sum(q) for Q4_0 zero-point correction (-8 * s)
};

// Quantize X[M, K] F32 -> Df2GpuActQ8[M, K/32]; Grid = (K/32, M), Block = 32
__global__ void __launch_bounds__(32) k_df2_quant_x_q8(
    const float* __restrict__ X,
    Df2GpuActQ8* __restrict__ out,
    int K) {
    const int b = blockIdx.x;
    const int m = blockIdx.y;
    const int lane = threadIdx.x;
    const int nb = K >> 5;
    const float v = X[(size_t)m * K + b * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
    const float d = amax * (1.0f / 127.0f);
    const float inv = (d > 0.0f) ? (127.0f / amax) : 0.0f;
    int qi = __float2int_rn(v * inv);
    qi = max(-127, min(127, qi));
    int sum = qi;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
    Df2GpuActQ8& dst = out[(size_t)m * nb + b];
    dst.q[lane] = (int8_t)qi;
    if (lane == 0) {
        dst.d = d;
        dst.s = d * (float)sum;
    }
}

// Multi-row Q4_0 GEMM (M <= 8): reads each 18-byte Q4_0 block ONCE for all M rows!
// Alignment-safe (18 bytes/block -> uint16_t loads -> zero misaligned address traps!).
// Grid = ((N + 3) / 4), Block = 128 (4 warps, 1 output row per warp)
__global__ void __launch_bounds__(128) k_df2_gemm_q4_0(
    const uint8_t* __restrict__ W,
    const Df2GpuActQ8* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    if (n >= N) return;

    const int nb = K >> 5;
    const uint8_t* wr = W + (size_t)n * nb * 18;
    float acc[8] = {};

    for (int b = lane; b < nb; b += 32) {
        const uint16_t* u16 = reinterpret_cast<const uint16_t*>(wr + (size_t)b * 18);
        __half dh;
        const uint16_t d_bits = u16[0];
        memcpy(&dh, &d_bits, 2);
        const float dw = __half2float(dh);

        const uint32_t w0 = (uint32_t)u16[1] | ((uint32_t)u16[2] << 16);
        const uint32_t w1 = (uint32_t)u16[3] | ((uint32_t)u16[4] << 16);
        const uint32_t w2 = (uint32_t)u16[5] | ((uint32_t)u16[6] << 16);
        const uint32_t w3 = (uint32_t)u16[7] | ((uint32_t)u16[8] << 16);

        const uint32_t l0 = w0 & 0x0F0F0F0Fu, l1 = w1 & 0x0F0F0F0Fu, l2 = w2 & 0x0F0F0F0Fu, l3 = w3 & 0x0F0F0F0Fu;
        const uint32_t h0 = (w0 >> 4) & 0x0F0F0F0Fu, h1 = (w1 >> 4) & 0x0F0F0F0Fu;
        const uint32_t h2 = (w2 >> 4) & 0x0F0F0F0Fu, h3 = (w3 >> 4) & 0x0F0F0F0Fu;

        for (int m = 0; m < M; ++m) {
            const Df2GpuActQ8& xb = Xq[(size_t)m * nb + b];
            const int32_t* xw = reinterpret_cast<const int32_t*>(xb.q);
            int dot = 0;
            dot = __dp4a((int32_t)l0, xw[0], dot);
            dot = __dp4a((int32_t)l1, xw[1], dot);
            dot = __dp4a((int32_t)l2, xw[2], dot);
            dot = __dp4a((int32_t)l3, xw[3], dot);
            dot = __dp4a((int32_t)h0, xw[4], dot);
            dot = __dp4a((int32_t)h1, xw[5], dot);
            dot = __dp4a((int32_t)h2, xw[6], dot);
            dot = __dp4a((int32_t)h3, xw[7], dot);
            acc[m] = fmaf(dw, fmaf(xb.d, (float)dot, -8.0f * xb.s), acc[m]);
        }
    }

    for (int m = 0; m < M; ++m) {
        float v = acc[m];
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
        if (lane == 0) Y[(size_t)m * N + n] = v;
    }
}

// Multi-row Q8_0 GEMM (M <= 8): reads each 34-byte Q8_0 block ONCE for all M rows!
__global__ void __launch_bounds__(128) k_df2_gemm_q8_0(
    const uint8_t* __restrict__ W,
    const Df2GpuActQ8* __restrict__ Xq,
    float* __restrict__ Y,
    int M, int N, int K) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n = blockIdx.x * 4 + warp;
    if (n >= N) return;

    const int nb = K >> 5;
    const uint8_t* wr = W + (size_t)n * nb * 34;
    float acc[8] = {};

    for (int b = lane; b < nb; b += 32) {
        const uint16_t* u16 = reinterpret_cast<const uint16_t*>(wr + (size_t)b * 34);
        __half dh;
        const uint16_t d_bits = u16[0];
        memcpy(&dh, &d_bits, 2);
        const float dw = __half2float(dh);

        int32_t wv[8];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            wv[j] = (int32_t)((uint32_t)u16[1 + 2 * j] | ((uint32_t)u16[2 + 2 * j] << 16));
        }

        for (int m = 0; m < M; ++m) {
            const Df2GpuActQ8& xb = Xq[(size_t)m * nb + b];
            const int32_t* xw = reinterpret_cast<const int32_t*>(xb.q);
            int dot = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) dot = __dp4a(wv[j], xw[j], dot);
            acc[m] = fmaf(dw * xb.d, (float)dot, acc[m]);
        }
    }

    for (int m = 0; m < M; ++m) {
        float v = acc[m];
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, o);
        if (lane == 0) Y[(size_t)m * N + n] = v;
    }
}

// Concatenate 4 taps [4][M, D] -> feats [M, 4*D]
__global__ void __launch_bounds__(256) k_df2_concat_taps(
    const float* __restrict__ t0, const float* __restrict__ t1,
    const float* __restrict__ t2, const float* __restrict__ t3,
    float* __restrict__ out, int M, int D) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const float* ts[4] = {t0, t1, t2, t3};
    float* dst = out + (size_t)m * 4 * D;
    for (int t = 0; t < 4; ++t) {
        const float* src = ts[t] + (size_t)m * D;
        for (int d = threadIdx.x; d < D; d += 256) dst[(size_t)t * D + d] = src[d];
    }
}

// RMSNorm [M, D]
__global__ void __launch_bounds__(256) k_df2_rms_norm(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ out, int M, int D, float eps) {
    const int m = blockIdx.x;
    if (m >= M) return;
    const float* xr = x + (size_t)m * D;
    float* yr = out + (size_t)m * D;
    __shared__ float sh[32];
    float ss = 0.0f;
    for (int d = threadIdx.x; d < D; d += 256) {
        float v = xr[d];
        ss += v * v;
    }
    ss = block_sum(ss, sh);
    const float inv = rsqrtf(ss / (float)D + eps);
    for (int d = threadIdx.x; d < D; d += 256) yr[d] = xr[d] * inv * w[d];
}

// Grouped 2-tap dynamic convolution: Grid = (B,), Block = 256
// phase = 0 (prepare) or 1 (finish); if add_res != nullptr, writes add_res[r, c] + conv_out!
__global__ void __launch_bounds__(256) k_df2_dyn_conv(
    const float* __restrict__ x,
    const float* __restrict__ dyn,       // [B, 2, 2, groups]
    const float* __restrict__ base_kern, // [2, 2, H]
    const float* __restrict__ add_res,   // optional [B, H]
    float* __restrict__ out,             // [B, H]
    int B, int H, int gs, int phase) {
    const int r = blockIdx.x;
    if (r >= B) return;
    const int groups = H / gs;
    const int dyn_stride = 4 * groups;
    const float* d0 = dyn + (size_t)r * dyn_stride + (size_t)phase * 2 * groups;
    const float* d1 = d0 + groups;
    const float* b0 = base_kern + (size_t)phase * 2 * H;
    const float* b1 = b0 + H;
    const float* x0 = x + (size_t)r * H;
    const float* x1 = (r >= 1) ? (x + (size_t)(r - 1) * H) : nullptr;
    float* dst = out + (size_t)r * H;
    const float* res = add_res ? (add_res + (size_t)r * H) : nullptr;
    for (int c = threadIdx.x; c < H; c += 256) {
        const int g = c / gs;
        float v = (b0[c] + d0[g]) * x0[c];
        if (x1) v = fmaf(b1[c] + d1[g], x1[c], v);
        dst[c] = res ? (res[c] + v) : v;
    }
}

// Per-head RMSNorm + NeoX RoPE (rotate_half) on [M, n_heads, 128]
// Grid = (n_heads, M), Block = 64 (1 thread per pair (d, d+64)!)
__global__ void __launch_bounds__(64) k_df2_rms_rope_128(
    float* __restrict__ x,
    const float* __restrict__ norm_w,
    int p0, int n_heads, float theta, float eps) {
    const int h = blockIdx.x;
    const int m = blockIdx.y;
    const int d = threadIdx.x; // 0..63
    float* v = x + ((size_t)m * n_heads + h) * 128;
    const float x0 = v[d];
    const float x1 = v[d + 64];
    float ss = x0 * x0 + x1 * x1;
    __shared__ float ws[2];
    const int warp = d >> 5;
    const int lane = d & 31;
#pragma unroll
    for (int o = 16; o >= 1; o >>= 1) ss += __shfl_xor_sync(0xFFFFFFFFu, ss, o);
    if (lane == 0) ws[warp] = ss;
    __syncthreads();
    const float tot = ws[0] + ws[1];
    const float inv = rsqrtf(tot * (1.0f / 128.0f) + eps);

    const float nx0 = x0 * inv * norm_w[d];
    const float nx1 = x1 * inv * norm_w[d + 64];
    const float ang = (float)(p0 + m) / powf(theta, (float)(2 * d) * (1.0f / 128.0f));
    float sn, cs;
    __sincosf(ang, &sn, &cs);
    v[d]      = nx0 * cs - nx1 * sn;
    v[d + 64] = nx1 * cs + nx0 * sn;
}

// Write [n, 8, 128] K, V into FP16 ring [kRingCap, 8, 128] at positions (p0 + i) % ring_cap
__global__ void __launch_bounds__(128) k_df2_write_ring(
    const float* __restrict__ K_in, const float* __restrict__ V_in,
    __half* __restrict__ K_ring, __half* __restrict__ V_ring,
    int p0, int n, int ring_cap) {
    const int h = blockIdx.x; // 0..7
    const int i = blockIdx.y; // 0..n-1
    const int d = threadIdx.x; // 0..127
    const int rslot = (p0 + i) % ring_cap;
    const size_t sidx = ((size_t)i * 8 + h) * 128 + d;
    const size_t didx = ((size_t)rslot * 8 + h) * 128 + d;
    K_ring[didx] = __float2half_rn(K_in[sidx]);
    V_ring[didx] = __float2half_rn(V_in[sidx]);
}

// Fast 8-Warp Cooperative Fused-Block SWA-2048 + Bidirectional Block-8 GQA Attention (32Q / 8KV, HD=128)
// Grid = (32 Q heads,), Block = 256 (8 warps = 1 warp per block row qi=0..7; reads each context KV ONCE across all 8 rows!)
__global__ void __launch_bounds__(256) k_df2_block_attn(
    const float* __restrict__ Q,       // [B, 32, 128]
    const float* __restrict__ Kb,      // [B, 8, 128]
    const float* __restrict__ Vb,      // [B, 8, 128]
    const __half* __restrict__ K_ring, // [ring_cap, 8, 128]
    const __half* __restrict__ V_ring, // [ring_cap, 8, 128]
    float* __restrict__ out,           // [B, 32, 128]
    int anchor_pos, int B, int window, int ring_cap) {
    const int qh   = blockIdx.x;
    const int qi   = threadIdx.x >> 5;  // 0..7 = block row!
    const int lane = threadIdx.x & 31;
    const int kvh  = qh >> 2;
    const int d4   = lane * 4;
    if (qi >= B) return;

    const float4 q4 = *reinterpret_cast<const float4*>(Q + ((size_t)qi * 32 + qh) * 128 + d4);
    const float scale = 0.08838834764831845f;

    const int qpos = anchor_pos + qi;
    const int lo = max(0, anchor_pos - window + 1);
    const int c_total = max(0, anchor_pos - lo);

    float m_w = -1e30f;
    float l_w = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

    auto attend_one = [&](float4 k4, float4 v4) {
        float dot = q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) dot += __shfl_xor_sync(0xFFFFFFFFu, dot, o);
        const float sc = dot * scale;
        const float m_new = fmaxf(m_w, sc);
        const float alpha = __expf(m_w - m_new);
        const float beta  = __expf(sc - m_new);
        l_w = l_w * alpha + beta;
        o0 = fmaf(beta, v4.x, o0 * alpha);
        o1 = fmaf(beta, v4.y, o1 * alpha);
        o2 = fmaf(beta, v4.z, o2 * alpha);
        o3 = fmaf(beta, v4.w, o3 * alpha);
        m_w = m_new;
    };

    __shared__ float4 sk[8][32];
    __shared__ float4 sv[8][32];
    __shared__ int    skp[8];

    // 1. Context ring [lo .. anchor_pos - 1]: all 8 warps cooperatively load 8 FP16 keys ONCE!
    for (int ci_base = 0; ci_base < c_total; ci_base += 8) {
        const int ci = ci_base + qi;
        int kpos = -1;
        if (ci < c_total) {
            kpos = lo + ci;
            const int rslot = kpos % ring_cap;
            const size_t base = ((size_t)rslot * 8 + kvh) * 128 + d4;
            const __half2* kh2 = reinterpret_cast<const __half2*>(K_ring + base);
            const __half2* vh2 = reinterpret_cast<const __half2*>(V_ring + base);
            const float2 kf0 = __half22float2(kh2[0]), kf1 = __half22float2(kh2[1]);
            const float2 vf0 = __half22float2(vh2[0]), vf1 = __half22float2(vh2[1]);
            sk[qi][lane] = make_float4(kf0.x, kf0.y, kf1.x, kf1.y);
            sv[qi][lane] = make_float4(vf0.x, vf0.y, vf1.x, vf1.y);
        }
        if (lane == 0) skp[qi] = kpos;
        __syncthreads();

#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int kp = skp[j];
            if (kp >= 0 && (qpos - kp) < window) {
                attend_one(sk[j][lane], sv[j][lane]);
            }
        }
        __syncthreads();
    }

    // 2. Bidirectional block keys [0 .. B-1]: load cooperatively into SMEM once!
    {
        const size_t base = ((size_t)qi * 8 + kvh) * 128 + d4;
        sk[qi][lane] = *reinterpret_cast<const float4*>(Kb + base);
        sv[qi][lane] = *reinterpret_cast<const float4*>(Vb + base);
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            if (j < B) attend_one(sk[j][lane], sv[j][lane]);
        }
    }

    const float inv_l = 1.0f / (l_w + 1e-12f);
    *reinterpret_cast<float4*>(out + ((size_t)qi * 32 + qh) * 128 + d4) =
        make_float4(o0 * inv_l, o1 * inv_l, o2 * inv_l, o3 * inv_l);
}

__global__ void __launch_bounds__(256) k_df2_swiglu(
    float* __restrict__ gate, const float* __restrict__ up, int n) {
    const int i = blockIdx.x * 256 + threadIdx.x;
    if (i < n) {
        const float g = gate[i];
        gate[i] = (g / (1.0f + __expf(-g))) * up[i];
    }
}

// Stage 1 Top-16 over V logits for 7 rows: Grid = (128, 7), Block = 256 (8 warps, each warp finds Top-2 -> 16/CTA!)
__global__ void __launch_bounds__(256) k_df2_top16_s1(
    const float* __restrict__ logits, // [7, V]
    int V,
    float* __restrict__ tmp_val,      // [7, 128 * 16 = 2048]
    int32_t* __restrict__ tmp_id) {
    const int ch = blockIdx.x;  // 0..127
    const int row = blockIdx.y; // 0..6
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const float* lg = logits + (size_t)row * V;

    float v1 = -1e30f, v2 = -1e30f;
    int32_t i1 = 0, i2 = 0;
    for (int v = ch * 256 + tid; v < V; v += 128 * 256) {
        const float val = lg[v];
        if (val > v1) { v2 = v1; i2 = i1; v1 = val; i1 = v; }
        else if (val > v2) { v2 = val; i2 = v; }
    }
    // Warp top-2 reduction
    auto wmax = [](float& bv, int32_t& bi) {
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
            int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
        }
    };
    float best1 = v1; int32_t bid1 = i1;
    wmax(best1, bid1);
    float cand2 = (i1 == bid1) ? v2 : v1;
    int32_t cid2 = (i1 == bid1) ? i2 : i1;
    wmax(cand2, cid2);

    if (lane == 0) {
        const size_t ob = (size_t)row * 2048 + ch * 16 + warp * 2;
        tmp_val[ob + 0] = best1; tmp_id[ob + 0] = bid1;
        tmp_val[ob + 1] = cand2; tmp_id[ob + 1] = cid2;
    }
}

// Stage 2 exact Top-16 from 2048 pairs per row: Grid = (7,), Block = 256
__global__ void __launch_bounds__(256) k_df2_top16_s2(
    const float* __restrict__ tmp_val,
    const int32_t* __restrict__ tmp_id,
    float* __restrict__ out_val,   // [7, 16]
    int32_t* __restrict__ out_id) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float sv[2048];
    __shared__ int32_t si[2048];
    __shared__ float wv[8];
    __shared__ int32_t wi[8];
    __shared__ int32_t wp[8];

    const size_t base = (size_t)row * 2048;
    for (int i = tid; i < 2048; i += 256) {
        sv[i] = tmp_val[base + i];
        si[i] = tmp_id[base + i];
    }
    __syncthreads();

    const int warp = tid >> 5;
    const int lane = tid & 31;
    for (int k = 0; k < 16; ++k) {
        float bv = -1e30f; int32_t bi = 0; int32_t bp = -1;
        for (int i = tid; i < 2048; i += 256) {
            const float v = sv[i];
            const int32_t id = si[i];
            if (v > bv || (v == bv && id < bi)) { bv = v; bi = id; bp = i; }
        }
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            float ov = __shfl_xor_sync(0xFFFFFFFFu, bv, o);
            int32_t oi = __shfl_xor_sync(0xFFFFFFFFu, bi, o);
            int32_t op = __shfl_xor_sync(0xFFFFFFFFu, bp, o);
            if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; bp = op; }
        }
        if (lane == 0) { wv[warp] = bv; wi[warp] = bi; wp[warp] = bp; }
        __syncthreads();
        if (tid == 0) {
            float fv = -1e30f; int32_t fi = 0; int32_t fp = -1;
            for (int w = 0; w < 8; ++w) {
                if (wv[w] > fv || (wv[w] == fv && wi[w] < fi)) { fv = wv[w]; fi = wi[w]; fp = wp[w]; }
            }
            out_val[(size_t)row * 16 + k] = fv;
            out_id[(size_t)row * 16 + k] = fi;
            if (fp >= 0) sv[fp] = -1e30f;
        }
        __syncthreads();
    }
}

// Optional GPU Q4_0 selector walk (used in Df2Mode::GPU): Grid = (1,), Block = 256 (1 thread per rank dim!)
__global__ void __launch_bounds__(256) k_df2_selector_walk_q4(
    const float* __restrict__ hp,          // [7, 256]
    const float* __restrict__ unary16,     // [7, 16]
    const int32_t* __restrict__ cand16,    // [7, 16]
    const uint8_t* __restrict__ pred_q4,   // [vocab, 8 * 18]
    const uint8_t* __restrict__ succ_q4,   // [vocab, 8 * 18]
    int32_t anchor_tok,
    float softcap,
    int32_t* __restrict__ out_ids7,
    float* __restrict__ out_conf7) {
    const int r = threadIdx.x; // 0..255
    const int blk = r >> 5;    // 0..7
    const int in_b = r & 31;   // 0..31
    const int warp = r >> 5;
    const int lane = r & 31;
    __shared__ float wsum[8];
    __shared__ float s_sc[16];
    __shared__ int32_t s_pred;
    if (r == 0) s_pred = anchor_tok;
    __syncthreads();

    auto decode_q4 = [&](const uint8_t* cb, int32_t row) -> float {
        const uint8_t* bp = cb + ((size_t)row * 8 + blk) * 18;
        uint16_t db;
        memcpy(&db, bp, 2);
        __half dh;
        memcpy(&dh, &db, 2);
        const float d = __half2float(dh);
        const uint8_t byte = bp[2 + (in_b & 15)];
        const int q = (in_b < 16) ? (byte & 0x0F) : (byte >> 4);
        return ((float)q - 8.0f) * d;
    };

    const float inv_cap = (softcap > 0.0f) ? (1.0f / softcap) : 0.0f;
    for (int i = 0; i < 7; ++i) {
        const float ap = decode_q4(pred_q4, s_pred) * hp[(size_t)i * 256 + r];
        for (int k = 0; k < 16; ++k) {
            const int32_t cid = cand16[(size_t)i * 16 + k];
            float prod = decode_q4(succ_q4, cid) * ap;
#pragma unroll
            for (int o = 16; o >= 1; o >>= 1) prod += __shfl_xor_sync(0xFFFFFFFFu, prod, o);
            if (lane == 0) wsum[warp] = prod;
            __syncthreads();
            if (r == 0) {
                float corr = wsum[0] + wsum[1] + wsum[2] + wsum[3] + wsum[4] + wsum[5] + wsum[6] + wsum[7];
                float val = unary16[(size_t)i * 16 + k] + corr;
                if (softcap > 0.0f) val = softcap * tanhf(val * inv_cap);
                s_sc[k] = val;
            }
            __syncthreads();
        }
        if (r == 0) {
            int best_k = 0; float best_v = s_sc[0];
            for (int k = 1; k < 16; ++k) {
                if (s_sc[k] > best_v) { best_v = s_sc[k]; best_k = k; }
            }
            float sum_e = 0.0f;
            for (int k = 0; k < 16; ++k) sum_e += __expf(s_sc[k] - best_v);
            const int32_t pick = cand16[(size_t)i * 16 + best_k];
            out_ids7[i] = pick;
            out_conf7[i] = 1.0f / sum_e;
            s_pred = pick;
        }
        __syncthreads();
    }
}

// ============================================================================
// Unified InkDFlash2 Engine (HYBRID / GPU / STREAM / CPU)
// ============================================================================
struct InkDFlash2 {
    static constexpr int kRingCap = 2064;
    static constexpr int kMaxLayers = 8;

    InkCudaRunner& r;
    InkCudaModel& m;
    Df2Pack pack;
    InkDFlash2Cpu cpu;
    Df2Mode mode = Df2Mode::HYBRID;
    bool ready = false;
    bool use_emb_norm = true;
    size_t vram_bytes = 0;

    // GPU weight storage (HYBRID & GPU modes)
    uint8_t* d_blob = nullptr;
    std::vector<const uint8_t*> d_tensor_ptr; // indexed by pack.tensors index

    // STREAM mode: pinned host mirror of per-layer weights + 2 ping-pong VRAM layer buffers
    struct StreamLayerLayout {
        size_t total_bytes = 0;
        size_t off_in_ln, off_post_ln, off_q, off_k, off_v, off_o, off_qn, off_kn;
        size_t off_gate, off_up, off_down, off_ac_kp, off_ac_bk, off_mc_kp, off_mc_bk;
    };
    StreamLayerLayout sl_lay{};
    uint8_t* h_layer_pin[kMaxLayers] = {};
    uint8_t* d_layer_pp[2] = {};
    uint8_t* d_fc_gpu = nullptr;      // small 12 MB Q4_0 fc + norms kept resident
    float* d_hnorm_gpu = nullptr;
    float* d_norm_gpu = nullptr;
    uint8_t* d_st_kproj[kMaxLayers] = {}; // 13.4 MB total across 5 layers -> 0.13 ms ingest_taps in STREAM mode!
    uint8_t* d_st_vproj[kMaxLayers] = {};
    float*   d_st_knorm[kMaxLayers] = {};
    cudaStream_t st_dma = nullptr;
    cudaEvent_t ev_dma_done[2] = {};
    cudaEvent_t ev_comp_done[2] = {};

    // Active GPU ring (HYBRID / GPU / STREAM)
    __half* d_ring_k[kMaxLayers] = {};
    __half* d_ring_v[kMaxLayers] = {};

    // Reusable GPU scratch buffers (all tiny: ~12 MB total)
    int32_t* d_block_ids = nullptr;
    Df2GpuActQ8* d_xq = nullptr;
    float *d_feats = nullptr, *d_fcy = nullptr;
    float *d_h = nullptr, *d_xn = nullptr, *d_xc = nullptr, *d_dyn = nullptr;
    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_ao = nullptr, *d_o = nullptr;
    float *d_gate = nullptr, *d_up = nullptr, *d_mlp = nullptr, *d_out7 = nullptr;
    float *d_logits7 = nullptr, *d_top_val = nullptr, *d_unary16 = nullptr, *d_hp = nullptr;
    int32_t *d_top_id = nullptr, *d_cand16 = nullptr, *d_out_ids7 = nullptr;
    float *d_out_conf7 = nullptr;

    // Pinned host staging buffers
    float* h_taps_pin[4] = {};
    int h_taps_cap = 0;
    float* h_embed_pin = nullptr;
    float* h_out7_pin = nullptr;
    float* h_unary16_pin = nullptr;
    int32_t* h_cand16_pin = nullptr;
    int32_t* h_ids7_pin = nullptr;
    float* h_conf7_pin = nullptr;

    // Profiling stats
    double prof_ingest_ms = 0.0, prof_draft_ms = 0.0, prof_head_ms = 0.0, prof_sel_ms = 0.0;
    long prof_drafts = 0;

    explicit InkDFlash2(InkCudaRunner& runner) : r(runner), m(runner.m) {}
    ~InkDFlash2() { free_all(); }

    template <typename T>
    T* dalloc_tracked(size_t n) {
        T* p = nullptr;
        INK_CK(cudaMalloc(&p, n * sizeof(T)));
        vram_bytes += n * sizeof(T);
        return p;
    }

    const uint8_t* gptr(const std::string& name) const {
        auto it = pack.name_to_idx.find(name);
        if (it == pack.name_to_idx.end()) return nullptr;
        return d_tensor_ptr[it->second];
    }

    bool init(const std::string& pack_path, Df2Mode req_mode, int max_slots, std::string& err) {
        free_all();
        mode = req_mode;
        use_emb_norm = true;
        if (const char* e = std::getenv("STRATA_INK_DF2_EMBNORM")) use_emb_norm = (e[0] != '0');
        if (!pack.load(pack_path, err)) return false;

        // Verify pack dimensions match Inkling-2-Mini
        if (pack.hidden != kInkHidden || pack.block != 8) {
            err = "DF2P pack dimensions mismatch Inkling-2-Mini (expected hidden=2304, block=8)";
            return false;
        }
        // Always initialize CPU engine: full rings for CPU mode, selector-only for HYBRID/STREAM/GPU!
        const bool need_cpu_rings = (mode == Df2Mode::CPU);
        if (!cpu.init(&pack, max_slots, need_cpu_rings)) {
            err = "InkDFlash2Cpu init failed";
            return false;
        }

        const int B = pack.block;
        const int H = pack.hidden;
        const int Q = pack.heads * pack.head_dim;
        const int KV = pack.kv_heads * pack.head_dim;
        const int I = pack.inter;
        const int V = m.man.c.vocab_unpadded;

        // Pinned host buffers (tiny, ~1 MB)
        INK_CK(cudaHostAlloc(&h_embed_pin,   (size_t)B * H * sizeof(float), cudaHostAllocDefault));
        INK_CK(cudaHostAlloc(&h_out7_pin,    (size_t)(B - 1) * H * sizeof(float), cudaHostAllocDefault));
        INK_CK(cudaHostAlloc(&h_unary16_pin, (size_t)(B - 1) * 16 * sizeof(float), cudaHostAllocDefault));
        INK_CK(cudaHostAlloc(&h_cand16_pin,  (size_t)(B - 1) * 16 * sizeof(int32_t), cudaHostAllocDefault));
        INK_CK(cudaHostAlloc(&h_ids7_pin,    8 * sizeof(int32_t), cudaHostAllocDefault));
        INK_CK(cudaHostAlloc(&h_conf7_pin,   8 * sizeof(float), cudaHostAllocDefault));

        // Shared minimal GPU scratch needed by ALL modes (including CPU mode for embed + lm_head + top-16)
        d_block_ids = dalloc_tracked<int32_t>(8);
        d_h         = dalloc_tracked<float>((size_t)B * H);
        d_out7      = dalloc_tracked<float>((size_t)(B - 1) * H);
        d_logits7   = dalloc_tracked<float>((size_t)(B - 1) * V);
        d_top_val   = dalloc_tracked<float>((size_t)(B - 1) * 2048);
        d_top_id    = dalloc_tracked<int32_t>((size_t)(B - 1) * 2048);
        d_unary16   = dalloc_tracked<float>((size_t)(B - 1) * 16);
        d_cand16    = dalloc_tracked<int32_t>((size_t)(B - 1) * 16);

        if (mode == Df2Mode::CPU) {
            ready = true;
            std::fprintf(stderr, "[InkDFlash2] Ready (mode=cpu, pack=%s, threads=%d, vram=%.1f MiB, emb_norm=%d)\n",
                         pack.is_q4_backbone() ? "q4_0" : "q8_0", cpu.num_threads(),
                         (double)vram_bytes / 1048576.0, (int)use_emb_norm);
            return true;
        }

        // GPU scratch for backbone (HYBRID, GPU, STREAM)
        const int max_k = std::max(4 * H, std::max(Q, I));
        d_xq    = dalloc_tracked<Df2GpuActQ8>((size_t)B * (max_k >> 5));
        d_feats = dalloc_tracked<float>((size_t)B * 4 * H);
        d_fcy   = dalloc_tracked<float>((size_t)B * H);
        d_xn    = dalloc_tracked<float>((size_t)B * H);
        d_xc    = dalloc_tracked<float>((size_t)B * H);
        d_dyn   = dalloc_tracked<float>((size_t)B * 4 * (H / pack.conv_group));
        d_q     = dalloc_tracked<float>((size_t)B * Q);
        d_k     = dalloc_tracked<float>((size_t)B * KV);
        d_v     = dalloc_tracked<float>((size_t)B * KV);
        d_ao    = dalloc_tracked<float>((size_t)B * Q);
        d_o     = dalloc_tracked<float>((size_t)B * H);
        d_gate  = dalloc_tracked<float>((size_t)B * I);
        d_up    = dalloc_tracked<float>((size_t)B * I);
        d_mlp   = dalloc_tracked<float>((size_t)B * H);

        // GPU FP16 SWA-2048 KV ring (1 active slot = 42.3 MB!)
        const size_t ring_elems = (size_t)kRingCap * KV;
        for (int l = 0; l < pack.layers; ++l) {
            d_ring_k[l] = dalloc_tracked<__half>(ring_elems);
            d_ring_v[l] = dalloc_tracked<__half>(ring_elems);
            INK_CK(cudaMemset(d_ring_k[l], 0, ring_elems * sizeof(__half)));
            INK_CK(cudaMemset(d_ring_v[l], 0, ring_elems * sizeof(__half)));
        }

        if (mode == Df2Mode::HYBRID || mode == Df2Mode::GPU) {
            const bool upload_sel = (mode == Df2Mode::GPU) &&
                (pack.require("candidate_selector.predecessor_codebook").dtype == Df2Dtype::Q4_0);
            if (mode == Df2Mode::GPU && !upload_sel) {
                std::fprintf(stderr, "[InkDFlash2] GPU selector requires Q4_0 selector pack; falling back to CPU RAM selector (hybrid)\n");
                mode = Df2Mode::HYBRID;
            }
            size_t tot = 0;
            std::vector<size_t> offs(pack.tensors.size(), (size_t)-1);
            for (size_t i = 0; i < pack.tensors.size(); ++i) {
                const auto& t = pack.tensors[i];
                const bool is_sel = (t.name.rfind("candidate_selector.", 0) == 0);
                if (is_sel && !upload_sel) continue;
                tot = (tot + 255ull) & ~255ull;
                offs[i] = tot;
                tot += t.bytes;
            }
            d_blob = dalloc_tracked<uint8_t>(tot);
            d_tensor_ptr.assign(pack.tensors.size(), nullptr);
            for (size_t i = 0; i < pack.tensors.size(); ++i) {
                if (offs[i] == (size_t)-1) continue;
                INK_CK(cudaMemcpy(d_blob + offs[i], pack.tensors[i].ptr, pack.tensors[i].bytes, cudaMemcpyHostToDevice));
                d_tensor_ptr[i] = d_blob + offs[i];
            }
            if (mode == Df2Mode::GPU) {
                d_hp        = dalloc_tracked<float>((size_t)(B - 1) * pack.rank);
                d_out_ids7  = dalloc_tracked<int32_t>(8);
                d_out_conf7 = dalloc_tracked<float>(8);
            }
        } else if (mode == Df2Mode::STREAM) {
            // Pack each layer's weights contiguously in pinned host RAM for single-DMA streaming!
            auto align256 = [](size_t x) { return (x + 255ull) & ~255ull; };
            auto build_layout = [&](int l, StreamLayerLayout& lay, uint8_t* dst) {
                const std::string p = "layers." + std::to_string(l) + ".";
                size_t cur = 0;
                auto put = [&](const std::string& nm, size_t& out_off) {
                    const auto& t = pack.require(p + nm);
                    cur = align256(cur);
                    out_off = cur;
                    if (dst) std::memcpy(dst + cur, t.ptr, t.bytes);
                    cur += t.bytes;
                };
                put("input_layernorm.weight", lay.off_in_ln);
                put("post_attention_layernorm.weight", lay.off_post_ln);
                put("self_attn.q_proj.weight", lay.off_q);
                put("self_attn.k_proj.weight", lay.off_k);
                put("self_attn.v_proj.weight", lay.off_v);
                put("self_attn.o_proj.weight", lay.off_o);
                put("self_attn.q_norm.weight", lay.off_qn);
                put("self_attn.k_norm.weight", lay.off_kn);
                put("mlp.gate_proj.weight", lay.off_gate);
                put("mlp.up_proj.weight", lay.off_up);
                put("mlp.down_proj.weight", lay.off_down);
                put("attention_conv.kernel_projection.weight", lay.off_ac_kp);
                put("attention_conv.base_kernel", lay.off_ac_bk);
                put("mlp_conv.kernel_projection.weight", lay.off_mc_kp);
                put("mlp_conv.base_kernel", lay.off_mc_bk);
                lay.total_bytes = align256(cur);
            };
            build_layout(0, sl_lay, nullptr);
            for (int l = 0; l < pack.layers; ++l) {
                INK_CK(cudaHostAlloc(&h_layer_pin[l], sl_lay.total_bytes, cudaHostAllocDefault));
                StreamLayerLayout tmp{};
                build_layout(l, tmp, h_layer_pin[l]);
            }
            for (int b = 0; b < 2; ++b) {
                d_layer_pp[b] = dalloc_tracked<uint8_t>(sl_lay.total_bytes);
                INK_CK(cudaEventCreateWithFlags(&ev_dma_done[b], cudaEventDisableTiming));
                INK_CK(cudaEventCreateWithFlags(&ev_comp_done[b], cudaEventDisableTiming));
                INK_CK(cudaEventRecord(ev_comp_done[b], r.st));
            }
            INK_CK(cudaStreamCreateWithFlags(&st_dma, cudaStreamNonBlocking));
            // Keep fc (12 MB) + norms (18 KB) resident on GPU for zero-latency ingest
            const auto& tfc = pack.require("fc.weight");
            const auto& thn = pack.require("hidden_norm.weight");
            const auto& tn  = pack.require("norm.weight");
            d_fc_gpu    = dalloc_tracked<uint8_t>(tfc.bytes);
            d_hnorm_gpu = dalloc_tracked<float>(thn.bytes / 4);
            d_norm_gpu  = dalloc_tracked<float>(tn.bytes / 4);
            INK_CK(cudaMemcpy(d_fc_gpu, tfc.ptr, tfc.bytes, cudaMemcpyHostToDevice));
            INK_CK(cudaMemcpy(d_hnorm_gpu, thn.ptr, thn.bytes, cudaMemcpyHostToDevice));
            INK_CK(cudaMemcpy(d_norm_gpu, tn.ptr, tn.bytes, cudaMemcpyHostToDevice));
            // Also keep tiny k_proj + v_proj + k_norm (~13.4 MB total in Q4_0) resident for 0.13 ms ingest_taps!
            for (int l = 0; l < pack.layers; ++l) {
                const std::string p = "layers." + std::to_string(l) + ".self_attn.";
                const auto& tk = pack.require(p + "k_proj.weight");
                const auto& tv = pack.require(p + "v_proj.weight");
                const auto& tkn = pack.require(p + "k_norm.weight");
                d_st_kproj[l] = dalloc_tracked<uint8_t>(tk.bytes);
                d_st_vproj[l] = dalloc_tracked<uint8_t>(tv.bytes);
                d_st_knorm[l] = dalloc_tracked<float>(tkn.bytes / 4);
                INK_CK(cudaMemcpy(d_st_kproj[l], tk.ptr, tk.bytes, cudaMemcpyHostToDevice));
                INK_CK(cudaMemcpy(d_st_vproj[l], tv.ptr, tv.bytes, cudaMemcpyHostToDevice));
                INK_CK(cudaMemcpy(d_st_knorm[l], tkn.ptr, tkn.bytes, cudaMemcpyHostToDevice));
            }
        }

        ready = true;
        std::fprintf(stderr, "[InkDFlash2] Ready (mode=%s, pack=%s, vram=%.1f MiB, emb_norm=%d)\n",
                     df2_mode_name(mode), pack.is_q4_backbone() ? "q4_0" : "q8_0",
                     (double)vram_bytes / 1048576.0, (int)use_emb_norm);
        return true;
    }

    void free_all() {
        auto fr = [](auto*& p) { if (p) { cudaFree(p); p = nullptr; } };
        auto frh = [](auto*& p) { if (p) { cudaFreeHost(p); p = nullptr; } };
        fr(d_blob);
        for (int b = 0; b < 2; ++b) {
            fr(d_layer_pp[b]);
            if (ev_dma_done[b]) { cudaEventDestroy(ev_dma_done[b]); ev_dma_done[b] = nullptr; }
            if (ev_comp_done[b]) { cudaEventDestroy(ev_comp_done[b]); ev_comp_done[b] = nullptr; }
        }
        if (st_dma) { cudaStreamDestroy(st_dma); st_dma = nullptr; }
        for (int l = 0; l < kMaxLayers; ++l) {
            frh(h_layer_pin[l]);
            fr(d_ring_k[l]);
            fr(d_ring_v[l]);
            fr(d_st_kproj[l]);
            fr(d_st_vproj[l]);
            fr(d_st_knorm[l]);
        }
        fr(d_fc_gpu); fr(d_hnorm_gpu); fr(d_norm_gpu);
        fr(d_block_ids); fr(d_xq); fr(d_feats); fr(d_fcy);
        fr(d_h); fr(d_xn); fr(d_xc); fr(d_dyn);
        fr(d_q); fr(d_k); fr(d_v); fr(d_ao); fr(d_o);
        fr(d_gate); fr(d_up); fr(d_mlp); fr(d_out7);
        fr(d_logits7); fr(d_top_val); fr(d_top_id); fr(d_unary16); fr(d_cand16);
        fr(d_hp); fr(d_out_ids7); fr(d_out_conf7);
        for (int i = 0; i < 4; ++i) frh(h_taps_pin[i]);
        h_taps_cap = 0;
        frh(h_embed_pin); frh(h_out7_pin); frh(h_unary16_pin); frh(h_cand16_pin); frh(h_ids7_pin); frh(h_conf7_pin);
        cpu.free_all();
        d_tensor_ptr.clear();
        vram_bytes = 0;
        ready = false;
    }

    void gpu_matmul(const uint8_t* w_ptr, Df2Dtype dt, const float* x, float* y, int M, int N, int K, bool reuse_xq = false) {
        const int nb = K >> 5;
        if (!reuse_xq) k_df2_quant_x_q8<<<dim3(nb, M), 32, 0, r.st>>>(x, d_xq, K);
        const int grid_n = (N + 3) / 4;
        if (dt == Df2Dtype::Q4_0) {
            k_df2_gemm_q4_0<<<grid_n, 128, 0, r.st>>>(w_ptr, d_xq, y, M, N, K);
        } else {
            k_df2_gemm_q8_0<<<grid_n, 128, 0, r.st>>>(w_ptr, d_xq, y, M, N, K);
        }
    }

    void embed_block_to_gpu(int32_t anchor_tok) {
        h_ids7_pin[0] = anchor_tok;
        for (int i = 1; i < 8; ++i) h_ids7_pin[i] = pack.mask_token;
        INK_CK(cudaMemcpyAsync(d_block_ids, h_ids7_pin, 8 * sizeof(int32_t), cudaMemcpyHostToDevice, r.st));
        const auto& ei = m.tinfo("model.llm.embed.weight");
        const float* norm_w = use_emb_norm ? (const float*)m.trunk_ptr("model.llm.embed_norm.weight") : nullptr;
        k_ink_embed_norm<<<8, 256, 0, r.st>>>(
            m.d_trunk + ei.off, ei.fmt, norm_w, d_block_ids, nullptr, nullptr, d_h, 8, kInkHidden);
    }

    // Ingest target taps [4][M, 2304] at positions [p0 .. p0 + M - 1]
    void ingest_taps_from(float* const taps[4], int p0, int M, int slot = 0) {
        if (!ready || M <= 0) return;
        const int H = pack.hidden;
        if (mode == Df2Mode::CPU) {
            if (M > h_taps_cap) {
                for (int i = 0; i < 4; ++i) {
                    if (h_taps_pin[i]) cudaFreeHost(h_taps_pin[i]);
                    INK_CK(cudaHostAlloc(&h_taps_pin[i], (size_t)M * H * sizeof(float), cudaHostAllocDefault));
                }
                h_taps_cap = M;
            }
            for (int i = 0; i < 4; ++i) {
                INK_CK(cudaMemcpyAsync(h_taps_pin[i], taps[i], (size_t)M * H * sizeof(float), cudaMemcpyDeviceToHost, r.st));
            }
            INK_CK(cudaStreamSynchronize(r.st));
            const float* ctaps[4] = {h_taps_pin[0], h_taps_pin[1], h_taps_pin[2], h_taps_pin[3]};
            cpu.ingest_taps(slot, ctaps, p0, M);
            return;
        }

        const Df2Dtype mdt = pack.require("fc.weight").dtype;
        const uint8_t* w_fc = (mode == Df2Mode::STREAM) ? d_fc_gpu : gptr("fc.weight");
        const float* w_hn   = (mode == Df2Mode::STREAM) ? d_hnorm_gpu : reinterpret_cast<const float*>(gptr("hidden_norm.weight"));
        const int KV = pack.kv_heads * pack.head_dim;

        // Only need to ingest the last `window` positions of a long prefill!
        int start_off = 0;
        if (M > pack.window) start_off = M - pack.window;
        for (int off = start_off; off < M; off += 8) {
            const int C = std::min(8, M - off);
            k_df2_concat_taps<<<C, 256, 0, r.st>>>(
                taps[0] + (size_t)off * H, taps[1] + (size_t)off * H,
                taps[2] + (size_t)off * H, taps[3] + (size_t)off * H,
                d_feats, C, H);
            gpu_matmul(w_fc, mdt, d_feats, d_fcy, C, H, 4 * H);
            k_df2_rms_norm<<<C, 256, 0, r.st>>>(d_fcy, w_hn, d_fcy, C, H, pack.eps);

            if (mode == Df2Mode::STREAM) {
                for (int l = 0; l < pack.layers; ++l) {
                    gpu_matmul(d_st_kproj[l], mdt, d_fcy, d_k, C, KV, H, l > 0);
                    gpu_matmul(d_st_vproj[l], mdt, d_fcy, d_v, C, KV, H, true);
                    k_df2_rms_rope_128<<<dim3(pack.kv_heads, C), 64, 0, r.st>>>(
                        d_k, d_st_knorm[l], p0 + off, pack.kv_heads, pack.rope_theta, pack.eps);
                    k_df2_write_ring<<<dim3(pack.kv_heads, C), 128, 0, r.st>>>(
                        d_k, d_v, d_ring_k[l], d_ring_v[l], p0 + off, C, kRingCap);
                }
            } else {
                for (int l = 0; l < pack.layers; ++l) {
                    const std::string p = "layers." + std::to_string(l) + ".self_attn.";
                    gpu_matmul(gptr(p + "k_proj.weight"), mdt, d_fcy, d_k, C, KV, H, l > 0);
                    gpu_matmul(gptr(p + "v_proj.weight"), mdt, d_fcy, d_v, C, KV, H, true);
                    k_df2_rms_rope_128<<<dim3(pack.kv_heads, C), 64, 0, r.st>>>(
                        d_k, reinterpret_cast<const float*>(gptr(p + "k_norm.weight")), p0 + off, pack.kv_heads, pack.rope_theta, pack.eps);
                    k_df2_write_ring<<<dim3(pack.kv_heads, C), 128, 0, r.st>>>(
                        d_k, d_v, d_ring_k[l], d_ring_v[l], p0 + off, C, kRingCap);
                }
            }
        }
    }

    void ingest_taps(int p0, int M, int slot = 0) {
        ingest_taps_from(r.d_taps, p0, M, slot);
    }

    // Run 1 Block-8 draft step -> outputs 7 draft token IDs + 7 selector confidences!
    void draft_block8(int32_t anchor_tok, int anchor_pos, int32_t out_ids7[7], float out_conf7[7], int slot = 0) {
        const int B = pack.block; // 8
        const int H = pack.hidden;
        const int Q = pack.heads * pack.head_dim;
        const int KV = pack.kv_heads * pack.head_dim;
        const int I = pack.inter;
        const int conv_out = 4 * (H / pack.conv_group);
        const int V = m.man.c.vocab_unpadded;
        const Df2Dtype mdt = pack.require("fc.weight").dtype;

        // 1. Embed [anchor, mask x 7] on GPU
        embed_block_to_gpu(anchor_tok);

        if (mode == Df2Mode::CPU) {
            INK_CK(cudaMemcpyAsync(h_embed_pin, d_h, (size_t)B * H * sizeof(float), cudaMemcpyDeviceToHost, r.st));
            INK_CK(cudaStreamSynchronize(r.st));
            // Full 5-layer AVX2 CPU forward pass!
            cpu.forward_block(slot, h_embed_pin, anchor_pos, h_out7_pin);
            // Upload 7 hidden vectors (64 KB) to GPU for shared target lm_head + Top-16!
            INK_CK(cudaMemcpyAsync(d_out7, h_out7_pin, (size_t)(B - 1) * H * sizeof(float), cudaMemcpyHostToDevice, r.st));
            r.linear("model.llm.unembed.weight", d_out7, d_logits7, B - 1, V);
            k_df2_top16_s1<<<dim3(128, B - 1), 256, 0, r.st>>>(d_logits7, V, d_top_val, d_top_id);
            k_df2_top16_s2<<<B - 1, 256, 0, r.st>>>(d_top_val, d_top_id, d_unary16, d_cand16);
            INK_CK(cudaMemcpyAsync(h_unary16_pin, d_unary16, (size_t)(B - 1) * 16 * sizeof(float), cudaMemcpyDeviceToHost, r.st));
            INK_CK(cudaMemcpyAsync(h_cand16_pin,  d_cand16,  (size_t)(B - 1) * 16 * sizeof(int32_t), cudaMemcpyDeviceToHost, r.st));
            INK_CK(cudaStreamSynchronize(r.st));
            cpu.select_path(h_out7_pin, h_unary16_pin, h_cand16_pin, anchor_tok, out_ids7, out_conf7);
            return;
        }

        // GPU / HYBRID / STREAM 5-layer backbone
        auto run_layer = [&](int l,
            const float* in_ln, const float* post_ln,
            const uint8_t* wq, const uint8_t* wk, const uint8_t* wv, const uint8_t* wo,
            const float* qn, const float* kn,
            const uint8_t* wg, const uint8_t* wu, const uint8_t* wd,
            const uint8_t* w_ackp, const float* w_acbk,
            const uint8_t* w_mckp, const float* w_mcbk) {
            // Attention sublayer
            k_df2_rms_norm<<<B, 256, 0, r.st>>>(d_h, in_ln, d_xn, B, H, pack.eps);
            gpu_matmul(w_ackp, mdt, d_xn, d_dyn, B, conv_out, H);
            k_df2_dyn_conv<<<B, 256, 0, r.st>>>(d_xn, d_dyn, w_acbk, nullptr, d_xc, B, H, pack.conv_group, 0);
            gpu_matmul(wq, mdt, d_xc, d_q, B, Q, H, false);
            gpu_matmul(wk, mdt, d_xc, d_k, B, KV, H, true);
            gpu_matmul(wv, mdt, d_xc, d_v, B, KV, H, true);
            k_df2_rms_rope_128<<<dim3(pack.heads, B), 64, 0, r.st>>>(d_q, qn, anchor_pos, pack.heads, pack.rope_theta, pack.eps);
            k_df2_rms_rope_128<<<dim3(pack.kv_heads, B), 64, 0, r.st>>>(d_k, kn, anchor_pos, pack.kv_heads, pack.rope_theta, pack.eps);
            k_df2_block_attn<<<pack.heads, 256, 0, r.st>>>(
                d_q, d_k, d_v, d_ring_k[l], d_ring_v[l], d_ao, anchor_pos, B, pack.window, kRingCap);
            gpu_matmul(wo, mdt, d_ao, d_o, B, H, Q, false);
            k_df2_dyn_conv<<<B, 256, 0, r.st>>>(d_o, d_dyn, w_acbk, d_h, d_h, B, H, pack.conv_group, 1);

            // MLP sublayer
            k_df2_rms_norm<<<B, 256, 0, r.st>>>(d_h, post_ln, d_xn, B, H, pack.eps);
            gpu_matmul(w_mckp, mdt, d_xn, d_dyn, B, conv_out, H, false);
            k_df2_dyn_conv<<<B, 256, 0, r.st>>>(d_xn, d_dyn, w_mcbk, nullptr, d_xc, B, H, pack.conv_group, 0);
            gpu_matmul(wg, mdt, d_xc, d_gate, B, I, H, false);
            gpu_matmul(wu, mdt, d_xc, d_up,   B, I, H, true);
            k_df2_swiglu<<<(B * I + 255) / 256, 256, 0, r.st>>>(d_gate, d_up, B * I);
            gpu_matmul(wd, mdt, d_gate, d_mlp, B, H, I);
            k_df2_dyn_conv<<<B, 256, 0, r.st>>>(d_mlp, d_dyn, w_mcbk, d_h, d_h, B, H, pack.conv_group, 1);
        };

        if (mode == Df2Mode::STREAM) {
            for (int l = 0; l < pack.layers; ++l) {
                const int b = l & 1;
                if (l == 0) {
                    INK_CK(cudaStreamWaitEvent(st_dma, ev_comp_done[0], 0));
                    INK_CK(cudaMemcpyAsync(d_layer_pp[0], h_layer_pin[0], sl_lay.total_bytes, cudaMemcpyHostToDevice, st_dma));
                    INK_CK(cudaEventRecord(ev_dma_done[0], st_dma));
                }
                if (l + 1 < pack.layers) {
                    const int nb = (l + 1) & 1;
                    INK_CK(cudaStreamWaitEvent(st_dma, ev_comp_done[nb], 0));
                    INK_CK(cudaMemcpyAsync(d_layer_pp[nb], h_layer_pin[l + 1], sl_lay.total_bytes, cudaMemcpyHostToDevice, st_dma));
                    INK_CK(cudaEventRecord(ev_dma_done[nb], st_dma));
                }
                INK_CK(cudaStreamWaitEvent(r.st, ev_dma_done[b], 0));
                const uint8_t* lb = d_layer_pp[b];
                run_layer(l,
                    reinterpret_cast<const float*>(lb + sl_lay.off_in_ln),
                    reinterpret_cast<const float*>(lb + sl_lay.off_post_ln),
                    lb + sl_lay.off_q, lb + sl_lay.off_k, lb + sl_lay.off_v, lb + sl_lay.off_o,
                    reinterpret_cast<const float*>(lb + sl_lay.off_qn),
                    reinterpret_cast<const float*>(lb + sl_lay.off_kn),
                    lb + sl_lay.off_gate, lb + sl_lay.off_up, lb + sl_lay.off_down,
                    lb + sl_lay.off_ac_kp, reinterpret_cast<const float*>(lb + sl_lay.off_ac_bk),
                    lb + sl_lay.off_mc_kp, reinterpret_cast<const float*>(lb + sl_lay.off_mc_bk));
                INK_CK(cudaEventRecord(ev_comp_done[b], r.st));
            }
            k_df2_rms_norm<<<B - 1, 256, 0, r.st>>>(d_h + H, d_norm_gpu, d_out7, B - 1, H, pack.eps);
        } else {
            for (int l = 0; l < pack.layers; ++l) {
                const std::string p = "layers." + std::to_string(l) + ".";
                run_layer(l,
                    reinterpret_cast<const float*>(gptr(p + "input_layernorm.weight")),
                    reinterpret_cast<const float*>(gptr(p + "post_attention_layernorm.weight")),
                    gptr(p + "self_attn.q_proj.weight"), gptr(p + "self_attn.k_proj.weight"),
                    gptr(p + "self_attn.v_proj.weight"), gptr(p + "self_attn.o_proj.weight"),
                    reinterpret_cast<const float*>(gptr(p + "self_attn.q_norm.weight")),
                    reinterpret_cast<const float*>(gptr(p + "self_attn.k_norm.weight")),
                    gptr(p + "mlp.gate_proj.weight"), gptr(p + "mlp.up_proj.weight"), gptr(p + "mlp.down_proj.weight"),
                    gptr(p + "attention_conv.kernel_projection.weight"),
                    reinterpret_cast<const float*>(gptr(p + "attention_conv.base_kernel")),
                    gptr(p + "mlp_conv.kernel_projection.weight"),
                    reinterpret_cast<const float*>(gptr(p + "mlp_conv.base_kernel")));
            }
            k_df2_rms_norm<<<B - 1, 256, 0, r.st>>>(d_h + H, reinterpret_cast<const float*>(gptr("norm.weight")), d_out7, B - 1, H, pack.eps);
        }

        // Shared target lm_head (raw logits, NO softcap!) + exact Top-16
        r.linear("model.llm.unembed.weight", d_out7, d_logits7, B - 1, V);
        k_df2_top16_s1<<<dim3(128, B - 1), 256, 0, r.st>>>(d_logits7, V, d_top_val, d_top_id);
        k_df2_top16_s2<<<B - 1, 256, 0, r.st>>>(d_top_val, d_top_id, d_unary16, d_cand16);

        if (mode == Df2Mode::GPU) {
            gpu_matmul(gptr("candidate_selector.hidden_projection.weight"), mdt, d_out7, d_hp, B - 1, pack.rank, H);
            k_df2_selector_walk_q4<<<1, 256, 0, r.st>>>(
                d_hp, d_unary16, d_cand16,
                gptr("candidate_selector.predecessor_codebook"),
                gptr("candidate_selector.successor_codebook"),
                anchor_tok, pack.softcap, d_out_ids7, d_out_conf7);
            INK_CK(cudaMemcpyAsync(h_ids7_pin,  d_out_ids7,  7 * sizeof(int32_t), cudaMemcpyDeviceToHost, r.st));
            INK_CK(cudaMemcpyAsync(h_conf7_pin, d_out_conf7, 7 * sizeof(float),   cudaMemcpyDeviceToHost, r.st));
            INK_CK(cudaStreamSynchronize(r.st));
            std::memcpy(out_ids7,  h_ids7_pin,  7 * sizeof(int32_t));
            std::memcpy(out_conf7, h_conf7_pin, 7 * sizeof(float));
        } else {
            // HYBRID / STREAM: Zero-VRAM CPU RAM selector (< 0.05 ms!)
            INK_CK(cudaMemcpyAsync(h_out7_pin,    d_out7,    (size_t)(B - 1) * H * sizeof(float),  cudaMemcpyDeviceToHost, r.st));
            INK_CK(cudaMemcpyAsync(h_unary16_pin, d_unary16, (size_t)(B - 1) * 16 * sizeof(float), cudaMemcpyDeviceToHost, r.st));
            INK_CK(cudaMemcpyAsync(h_cand16_pin,  d_cand16,  (size_t)(B - 1) * 16 * sizeof(int32_t), cudaMemcpyDeviceToHost, r.st));
            INK_CK(cudaStreamSynchronize(r.st));
            cpu.select_path(h_out7_pin, h_unary16_pin, h_cand16_pin, anchor_tok, out_ids7, out_conf7);
        }
    }
};

} // namespace strata::ink

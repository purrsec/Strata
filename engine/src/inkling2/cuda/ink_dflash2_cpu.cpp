// engine/src/inkling2/cuda/ink_dflash2_cpu.cpp
// Persistent multi-thread AVX2+FMA+F16C CPU DFlash2 Drafter + DF2P Loader + Zero-VRAM Selector.
#include "ink_dflash2_cpu.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <thread>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::ink {

namespace {

inline float fp16_bits_to_f32(uint16_t h) {
#if defined(__F16C__)
    return _cvtsh_ss(h);
#else
    uint32_t s = (uint32_t)(h >> 15) << 31;
    uint32_t e = (h >> 10) & 0x1F;
    uint32_t m = h & 0x3FF;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = s;
        else {
            float v = std::ldexp((float)m, -24);
            return s ? -v : v;
        }
    } else if (e == 31) {
        bits = s | 0x7F800000u | (m << 13);
    } else {
        bits = s | ((e + 112) << 23) | (m << 13);
    }
    float out;
    std::memcpy(&out, &bits, 4);
    return out;
#endif
}

inline uint16_t f32_to_fp16_bits(float v) {
#if defined(__F16C__)
    return _cvtss_sh(v, _MM_FROUND_TO_NEAREST_INT);
#else
    uint32_t bits;
    std::memcpy(&bits, &v, 4);
    uint32_t s = (bits >> 16) & 0x8000u;
    int e = (int)((bits >> 23) & 0xFF) - 112;
    uint32_t m = bits & 0x7FFFFFu;
    if (e <= 0) return (uint16_t)s;
    if (e >= 31) return (uint16_t)(s | 0x7BFFu);
    return (uint16_t)(s | ((uint32_t)e << 10) | (m >> 13));
#endif
}

inline float bf16_bits_to_f32(uint16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Quantized activation block (32 values) for Q4_0 and Q8_0 AVX2 dot products
struct alignas(32) ActQ8Block {
    int8_t q[32];
    float d;
    float s; // d * sum(q[0..31]) for Q4_0 zero-point offset (-8 * s)
};

inline void quantize_rows_q8(const float* x, int M, int K, ActQ8Block* out) {
    const int nb = K >> 5;
    for (int m = 0; m < M; ++m) {
        const float* row = x + (size_t)m * K;
        ActQ8Block* dst = out + (size_t)m * nb;
        for (int b = 0; b < nb; ++b) {
            const float* p = row + (size_t)b * 32;
            float amax = 0.0f;
#if defined(__AVX2__)
            __m256 vabs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));
            __m256 vmax = _mm256_setzero_ps();
            for (int i = 0; i < 32; i += 8) {
                vmax = _mm256_max_ps(vmax, _mm256_and_ps(_mm256_loadu_ps(p + i), vabs_mask));
            }
            __m128 hi = _mm256_extractf128_ps(vmax, 1);
            __m128 lo = _mm256_castps256_ps128(vmax);
            __m128 m4 = _mm_max_ps(lo, hi);
            m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
            m4 = _mm_max_ss(m4, _mm_shuffle_ps(m4, m4, 1));
            amax = _mm_cvtss_f32(m4);
#else
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(p[i]));
#endif
            const float d = amax * (1.0f / 127.0f);
            const float inv = (d > 0.0f) ? (127.0f / amax) : 0.0f;
            int sum = 0;
#if defined(__AVX2__)
            __m256 vinv = _mm256_set1_ps(inv);
            __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + 0), vinv));
            __m256i i1 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + 8), vinv));
            __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + 16), vinv));
            __m256i i3 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(p + 24), vinv));
            __m256i s01 = _mm256_add_epi32(i0, i1);
            __m256i s23 = _mm256_add_epi32(i2, i3);
            __m256i sall = _mm256_add_epi32(s01, s23);
            __m128i slo = _mm_add_epi32(_mm256_castsi256_si128(sall), _mm256_extracti128_si256(sall, 1));
            slo = _mm_add_epi32(slo, _mm_srli_si128(slo, 8));
            slo = _mm_add_epi32(slo, _mm_srli_si128(slo, 4));
            sum = _mm_cvtsi128_si32(slo);

            // Pack 4x8 int32 -> 32 int8 in linear lane order [0..31]
            __m128i p0 = _mm_packs_epi32(_mm256_castsi256_si128(i0), _mm256_extracti128_si256(i0, 1));
            __m128i p1 = _mm_packs_epi32(_mm256_castsi256_si128(i1), _mm256_extracti128_si256(i1, 1));
            __m128i p2 = _mm_packs_epi32(_mm256_castsi256_si128(i2), _mm256_extracti128_si256(i2, 1));
            __m128i p3 = _mm_packs_epi32(_mm256_castsi256_si128(i3), _mm256_extracti128_si256(i3, 1));
            __m128i b01 = _mm_packs_epi16(p0, p1);
            __m128i b23 = _mm_packs_epi16(p2, p3);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst[b].q), b01);
            _mm_storeu_si128(reinterpret_cast<__m128i*>(dst[b].q + 16), b23);
#else
            for (int i = 0; i < 32; ++i) {
                int v = (int)std::lrintf(p[i] * inv);
                v = std::max(-127, std::min(127, v));
                dst[b].q[i] = (int8_t)v;
                sum += v;
            }
#endif
            dst[b].d = d;
            dst[b].s = d * (float)sum;
        }
    }
}

#if defined(__AVX2__)
inline float hsum256_ps(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 s = _mm_add_ps(lo, hi);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
#endif

// Compute rows [r0 .. r1) of Y[M, N] = X_q8[M, K] @ W[N, K]^T for Q4_0 weights.
// Loads each 18-byte Q4_0 weight block ONCE for all M <= 8 rows!
void gemm_q4_0_rows(
    const uint8_t* W, int N, int K,
    const ActQ8Block* Xq, int M,
    float* Y, int r0, int r1) {
    const int nb = K >> 5;
    const size_t row_bytes = (size_t)nb * 18;
#if defined(__AVX2__)
    const __m128i mask_0f = _mm_set1_epi8(0x0F);
    const __m256i ones16  = _mm256_set1_epi16(1);
#endif
    for (int r = r0; r < r1; ++r) {
        const uint8_t* wr = W + (size_t)r * row_bytes;
#if defined(__AVX2__)
        if (M == 8) {
            __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps(), acc3 = _mm256_setzero_ps();
            __m256 acc4 = _mm256_setzero_ps(), acc5 = _mm256_setzero_ps();
            __m256 acc6 = _mm256_setzero_ps(), acc7 = _mm256_setzero_ps();
            float bs0 = 0, bs1 = 0, bs2 = 0, bs3 = 0, bs4 = 0, bs5 = 0, bs6 = 0, bs7 = 0;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* bp = wr + (size_t)b * 18;
                uint16_t dh;
                std::memcpy(&dh, bp, 2);
                const float dw = fp16_bits_to_f32(dh);
                __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bp + 2));
                __m128i lo  = _mm_and_si128(raw, mask_0f);
                __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);
                __m256i q32 = _mm256_set_m128i(hi, lo);

                auto step = [&](int m, __m256& acc, float& bs) {
                    const ActQ8Block& xb = Xq[(size_t)m * nb + b];
                    __m256i xv = _mm256_load_si256(reinterpret_cast<const __m256i*>(xb.q));
                    __m256i p16 = _mm256_maddubs_epi16(q32, xv);
                    __m256i p32 = _mm256_madd_epi16(p16, ones16);
                    acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(dw * xb.d), acc);
                    bs = std::fma(dw, xb.s, bs);
                };
                step(0, acc0, bs0); step(1, acc1, bs1); step(2, acc2, bs2); step(3, acc3, bs3);
                step(4, acc4, bs4); step(5, acc5, bs5); step(6, acc6, bs6); step(7, acc7, bs7);
            }
            Y[0 * (size_t)N + r] = hsum256_ps(acc0) - 8.0f * bs0;
            Y[1 * (size_t)N + r] = hsum256_ps(acc1) - 8.0f * bs1;
            Y[2 * (size_t)N + r] = hsum256_ps(acc2) - 8.0f * bs2;
            Y[3 * (size_t)N + r] = hsum256_ps(acc3) - 8.0f * bs3;
            Y[4 * (size_t)N + r] = hsum256_ps(acc4) - 8.0f * bs4;
            Y[5 * (size_t)N + r] = hsum256_ps(acc5) - 8.0f * bs5;
            Y[6 * (size_t)N + r] = hsum256_ps(acc6) - 8.0f * bs6;
            Y[7 * (size_t)N + r] = hsum256_ps(acc7) - 8.0f * bs7;
            continue;
        }
        __m256 acc[8];
        float bs[8];
        for (int m = 0; m < M; ++m) { acc[m] = _mm256_setzero_ps(); bs[m] = 0.0f; }
        for (int b = 0; b < nb; ++b) {
            const uint8_t* bp = wr + (size_t)b * 18;
            uint16_t dh;
            std::memcpy(&dh, bp, 2);
            const float dw = fp16_bits_to_f32(dh);
            __m128i raw = _mm_loadu_si128(reinterpret_cast<const __m128i*>(bp + 2));
            __m128i lo  = _mm_and_si128(raw, mask_0f);
            __m128i hi  = _mm_and_si128(_mm_srli_epi16(raw, 4), mask_0f);
            __m256i q32 = _mm256_set_m128i(hi, lo);
            for (int m = 0; m < M; ++m) {
                const ActQ8Block& xb = Xq[(size_t)m * nb + b];
                __m256i xv = _mm256_load_si256(reinterpret_cast<const __m256i*>(xb.q));
                __m256i p16 = _mm256_maddubs_epi16(q32, xv);
                __m256i p32 = _mm256_madd_epi16(p16, ones16);
                acc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(dw * xb.d), acc[m]);
                bs[m] = std::fma(dw, xb.s, bs[m]);
            }
        }
        for (int m = 0; m < M; ++m) Y[(size_t)m * N + r] = hsum256_ps(acc[m]) - 8.0f * bs[m];
#else
        float acc[8] = {};
        for (int b = 0; b < nb; ++b) {
            const uint8_t* bp = wr + (size_t)b * 18;
            uint16_t dh;
            std::memcpy(&dh, bp, 2);
            const float dw = fp16_bits_to_f32(dh);
            for (int m = 0; m < M; ++m) {
                const ActQ8Block& xb = Xq[(size_t)m * nb + b];
                int dot = 0;
                for (int i = 0; i < 16; ++i) {
                    int ql = (int)(bp[2 + i] & 0x0F) - 8;
                    int qh = (int)(bp[2 + i] >> 4) - 8;
                    dot += ql * xb.q[i] + qh * xb.q[i + 16];
                }
                acc[m] += dw * xb.d * (float)dot;
            }
        }
        for (int m = 0; m < M; ++m) Y[(size_t)m * N + r] = acc[m];
#endif
    }
}

void gemm_q8_0_rows(
    const uint8_t* W, int N, int K,
    const ActQ8Block* Xq, int M,
    float* Y, int r0, int r1) {
    const int nb = K >> 5;
    const size_t row_bytes = (size_t)nb * 34;
#if defined(__AVX2__)
    const __m256i ones16 = _mm256_set1_epi16(1);
#endif
    for (int r = r0; r < r1; ++r) {
        const uint8_t* wr = W + (size_t)r * row_bytes;
#if defined(__AVX2__)
        __m256 acc[8];
        for (int m = 0; m < M; ++m) acc[m] = _mm256_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            const uint8_t* bp = wr + (size_t)b * 34;
            uint16_t dh;
            std::memcpy(&dh, bp, 2);
            const float dw = fp16_bits_to_f32(dh);
            __m256i wv = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(bp + 2));
            __m256i aw = _mm256_sign_epi8(wv, wv); // abs(w) in [0..127] (unsigned!)
            for (int m = 0; m < M; ++m) {
                const ActQ8Block& xb = Xq[(size_t)m * nb + b];
                __m256i xv = _mm256_load_si256(reinterpret_cast<const __m256i*>(xb.q));
                __m256i sx = _mm256_sign_epi8(xv, wv);
                __m256i p16 = _mm256_maddubs_epi16(aw, sx);
                __m256i p32 = _mm256_madd_epi16(p16, ones16);
                acc[m] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p32), _mm256_set1_ps(dw * xb.d), acc[m]);
            }
        }
        for (int m = 0; m < M; ++m) Y[(size_t)m * N + r] = hsum256_ps(acc[m]);
#else
        float acc[8] = {};
        for (int b = 0; b < nb; ++b) {
            const uint8_t* bp = wr + (size_t)b * 34;
            uint16_t dh;
            std::memcpy(&dh, bp, 2);
            const float dw = fp16_bits_to_f32(dh);
            const int8_t* wq = reinterpret_cast<const int8_t*>(bp + 2);
            for (int m = 0; m < M; ++m) {
                const ActQ8Block& xb = Xq[(size_t)m * nb + b];
                int dot = 0;
                for (int i = 0; i < 32; ++i) dot += (int)wq[i] * (int)xb.q[i];
                acc[m] += dw * xb.d * (float)dot;
            }
        }
        for (int m = 0; m < M; ++m) Y[(size_t)m * N + r] = acc[m];
#endif
    }
}

void rms_norm_rows(const float* x, const float* w, float* out, int M, int D, float eps) {
    for (int m = 0; m < M; ++m) {
        const float* xr = x + (size_t)m * D;
        float* yr = out + (size_t)m * D;
        float ss = 0.0f;
#if defined(__AVX2__)
        __m256 vs = _mm256_setzero_ps();
        int i = 0;
        for (; i + 8 <= D; i += 8) {
            __m256 vx = _mm256_loadu_ps(xr + i);
            vs = _mm256_fmadd_ps(vx, vx, vs);
        }
        ss = hsum256_ps(vs);
        for (; i < D; ++i) ss += xr[i] * xr[i];
#else
        for (int i = 0; i < D; ++i) ss += xr[i] * xr[i];
#endif
        const float inv = 1.0f / std::sqrt(ss / (float)D + eps);
#if defined(__AVX2__)
        __m256 vinv = _mm256_set1_ps(inv);
        int j = 0;
        for (; j + 8 <= D; j += 8) {
            __m256 vx = _mm256_loadu_ps(xr + j);
            __m256 vw = _mm256_loadu_ps(w + j);
            _mm256_storeu_ps(yr + j, _mm256_mul_ps(_mm256_mul_ps(vx, vinv), vw));
        }
        for (; j < D; ++j) yr[j] = xr[j] * inv * w[j];
#else
        for (int j = 0; j < D; ++j) yr[j] = xr[j] * inv * w[j];
#endif
    }
}

// NeoX RoPE (rotate_half) on [M, n_heads, HD] in-place after per-head RMSNorm
void rms_rope_heads(
    float* x, const float* norm_w, const float* inv_freq,
    const int* pos, int M, int n_heads, int HD, float eps) {
    const int half = HD >> 1;
    for (int m = 0; m < M; ++m) {
        const float pf = (float)pos[m];
        float cs[64], sn[64];
        for (int d = 0; d < half; ++d) {
            float ang = pf * inv_freq[d];
            cs[d] = std::cos(ang);
            sn[d] = std::sin(ang);
        }
        for (int h = 0; h < n_heads; ++h) {
            float* v = x + ((size_t)m * n_heads + h) * HD;
            float ss = 0.0f;
            for (int d = 0; d < HD; ++d) ss += v[d] * v[d];
            const float inv = 1.0f / std::sqrt(ss / (float)HD + eps);
            for (int d = 0; d < half; ++d) {
                const float x0 = v[d] * inv * norm_w[d];
                const float x1 = v[d + half] * inv * norm_w[d + half];
                v[d]        = x0 * cs[d] - x1 * sn[d];
                v[d + half] = x1 * cs[d] + x0 * sn[d];
            }
        }
    }
}

// Grouped 2-tap dynamic convolution: x [b, H], dyn [b, ks, groups], base [ks, H]
void apply_dyn_conv(
    const float* x, const float* dyn_sub, const float* base_sub,
    float* out, int b, int H, int gs, int ks, int dyn_row_stride) {
    const int groups = H / gs;
    for (int r = 0; r < b; ++r) {
        float* dst = out + (size_t)r * H;
        const float* x0 = x + (size_t)r * H;
        const float* d0 = dyn_sub + (size_t)r * dyn_row_stride;
        for (int g = 0; g < groups; ++g) {
            const float dg0 = d0[g];
            const int c0 = g * gs;
            for (int e = 0; e < gs; ++e) {
                const int c = c0 + e;
                dst[c] = (base_sub[c] + dg0) * x0[c];
            }
        }
        if (ks >= 2 && r >= 1) {
            const float* x1 = x + (size_t)(r - 1) * H;
            const float* d1 = d0 + groups;
            const float* b1 = base_sub + H;
            for (int g = 0; g < groups; ++g) {
                const float dg1 = d1[g];
                const int c0 = g * gs;
                for (int e = 0; e < gs; ++e) {
                    const int c = c0 + e;
                    dst[c] += (b1[c] + dg1) * x1[c];
                }
            }
        }
    }
}

// Persistent lock-free thread pool (same spin/yield pattern as ink_host_kv_attn.cpp)
class Df2ThreadPool {
public:
    explicit Df2ThreadPool(int n_threads) {
        n_ = std::max(1, n_threads);
        workers_.reserve((size_t)(n_ - 1));
        for (int t = 1; t < n_; ++t) {
            workers_.emplace_back([this, t]() { loop(t); });
        }
    }
    ~Df2ThreadPool() {
        stop_.store(true, std::memory_order_release);
        gen_.fetch_add(1, std::memory_order_release);
        for (auto& th : workers_) if (th.joinable()) th.join();
    }

    int size() const { return n_; }

    void run(const std::function<void(int, int)>& fn) {
        if (n_ <= 1) { fn(0, 1); return; }
        fn_ = &fn;
        done_.store(0, std::memory_order_relaxed);
        gen_.fetch_add(1, std::memory_order_release);
        fn(0, n_);
        while (done_.load(std::memory_order_acquire) < n_ - 1) {
#if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
#endif
        }
    }

private:
    void loop(int tid) {
        uint64_t my_gen = 0;
        while (true) {
            uint64_t g = gen_.load(std::memory_order_acquire);
            int spins = 0;
            while (g == my_gen) {
                if (++spins < 4000) {
#if defined(__x86_64__) || defined(_M_X64)
                    _mm_pause();
#endif
                } else {
                    std::this_thread::yield();
                }
                g = gen_.load(std::memory_order_acquire);
            }
            my_gen = g;
            if (stop_.load(std::memory_order_acquire)) return;
            (*fn_)(tid, n_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    int n_ = 1;
    std::vector<std::thread> workers_;
    std::atomic<uint64_t> gen_{0};
    std::atomic<int> done_{0};
    std::atomic<bool> stop_{false};
    const std::function<void(int, int)>* fn_ = nullptr;
};

} // namespace

bool Df2Pack::load(const std::string& path, std::string& err) {
    std::ifstream fh(path, std::ios::binary | std::ios::ate);
    if (!fh) { err = "cannot open DF2P pack: " + path; return false; }
    const size_t file_sz = (size_t)fh.tellg();
    if (file_sz < 16) { err = "DF2P pack too small"; return false; }
    fh.seekg(0, std::ios::beg);
    char magic[4];
    uint32_t ver = 0, idx_len = 0;
    fh.read(magic, 4);
    fh.read(reinterpret_cast<char*>(&ver), 4);
    fh.read(reinterpret_cast<char*>(&idx_len), 4);
    if (std::memcmp(magic, "DF2P", 4) != 0 || ver != 1) {
        err = "invalid DF2P magic/version in " + path;
        return false;
    }
    const size_t base = 12 + (size_t)idx_len;
    if (base > file_sz) { err = "DF2P index overflows file"; return false; }
    std::string index(idx_len, '\0');
    fh.read(index.data(), idx_len);

    data_bytes = file_sz - base;
    raw_storage.resize(data_bytes + 64);
    uintptr_t addr = reinterpret_cast<uintptr_t>(raw_storage.data());
    uintptr_t aligned = (addr + 63ull) & ~63ull;
    data_base = reinterpret_cast<uint8_t*>(aligned);
    fh.read(reinterpret_cast<char*>(data_base), (std::streamsize)data_bytes);
    if (!fh) { err = "failed reading DF2P tensor payload"; return false; }

    tensors.clear();
    name_to_idx.clear();
    taps.clear();

    std::istringstream iss(index);
    std::string line;
    while (std::getline(iss, line)) {
        if (line.empty()) continue;
        std::istringstream ls(line);
        std::string kind;
        ls >> kind;
        if (kind == "end") break;
        if (kind == "cfg") {
            std::string k, v;
            ls >> k >> v;
            if (k == "hidden") hidden = std::atoi(v.c_str());
            else if (k == "layers") layers = std::atoi(v.c_str());
            else if (k == "heads") heads = std::atoi(v.c_str());
            else if (k == "kv_heads") kv_heads = std::atoi(v.c_str());
            else if (k == "head_dim") head_dim = std::atoi(v.c_str());
            else if (k == "inter") inter = std::atoi(v.c_str());
            else if (k == "window") window = std::atoi(v.c_str());
            else if (k == "rope_theta") rope_theta = (float)std::atof(v.c_str());
            else if (k == "eps") eps = (float)std::atof(v.c_str());
            else if (k == "block") block = std::atoi(v.c_str());
            else if (k == "conv_group") conv_group = std::atoi(v.c_str());
            else if (k == "conv_kernel") conv_kernel = std::atoi(v.c_str());
            else if (k == "mask_token") mask_token = std::atoi(v.c_str());
            else if (k == "rank") rank = std::atoi(v.c_str());
            else if (k == "top_k") top_k = std::atoi(v.c_str());
            else if (k == "vocab") vocab = std::atoi(v.c_str());
            else if (k == "softcap") softcap = (float)std::atof(v.c_str());
            else if (k == "taps") {
                std::istringstream ts(v);
                std::string tok;
                while (std::getline(ts, tok, ',')) taps.push_back(std::atoi(tok.c_str()));
            }
        } else if (kind == "t") {
            Df2Tensor t;
            std::string dt;
            ls >> t.name >> dt >> t.rows >> t.cols >> t.offset >> t.bytes;
            if (dt == "f32") t.dtype = Df2Dtype::F32;
            else if (dt == "bf16") t.dtype = Df2Dtype::BF16;
            else if (dt == "q8_0") t.dtype = Df2Dtype::Q8_0;
            else if (dt == "q4_0") t.dtype = Df2Dtype::Q4_0;
            else { err = "unsupported DF2P dtype: " + dt; return false; }
            if (t.offset + t.bytes > data_bytes) {
                err = "DF2P tensor out of bounds: " + t.name;
                return false;
            }
            t.ptr = data_base + t.offset;
            name_to_idx[t.name] = (int)tensors.size();
            tensors.push_back(std::move(t));
        }
    }
    if (tensors.empty()) { err = "DF2P pack has no tensors"; return false; }
    return true;
}

const Df2Tensor* Df2Pack::find(const std::string& name) const {
    auto it = name_to_idx.find(name);
    if (it == name_to_idx.end()) return nullptr;
    return &tensors[it->second];
}

const Df2Tensor& Df2Pack::require(const std::string& name) const {
    const Df2Tensor* t = find(name);
    if (!t) throw std::runtime_error("missing DF2P tensor: " + name);
    return *t;
}

bool Df2Pack::is_q4_backbone() const {
    const Df2Tensor* t = find("fc.weight");
    return t && t->dtype == Df2Dtype::Q4_0;
}

struct InkDFlash2Cpu::Impl {
    const Df2Pack* pack = nullptr;
    std::unique_ptr<Df2ThreadPool> pool;
    int max_slots = 1;
    bool rings_on = false;

    // Precomputed pointers
    const Df2Tensor* t_fc = nullptr;
    const Df2Tensor* t_hnorm = nullptr;
    const Df2Tensor* t_norm = nullptr;
    const Df2Tensor* t_sel_hp = nullptr;
    const Df2Tensor* t_sel_pred = nullptr;
    const Df2Tensor* t_sel_succ = nullptr;

    struct LayerPtrs {
        const Df2Tensor *in_ln, *post_ln, *q_proj, *k_proj, *v_proj, *o_proj, *q_norm, *k_norm;
        const Df2Tensor *gate, *up, *down;
        const Df2Tensor *ac_kp, *ac_bk, *mc_kp, *mc_bk;
    };
    std::vector<LayerPtrs> L;
    std::vector<float> inv_freq;

    // Per-slot FP16 SWA-2048 KV rings in RAM: lazily allocated on first use!
    struct SlotRing {
        bool allocated = false;
        std::vector<uint16_t> k[kMaxLayers]; // [kRingCap * kv_heads * head_dim]
        std::vector<uint16_t> v[kMaxLayers];
    };
    std::vector<SlotRing> rings;

    // Reusable scratch buffers
    std::vector<ActQ8Block> xq_buf;
    std::vector<float> h, xn, xc, dyn, q, k, v, ao, o, gate, up, mlp, fcy, feats;

    void ensure_slot_ring(int slot) {
        if (slot < 0 || slot >= (int)rings.size()) return;
        auto& sr = rings[slot];
        if (sr.allocated) return;
        const size_t elems = (size_t)kRingCap * pack->kv_heads * pack->head_dim;
        for (int l = 0; l < pack->layers; ++l) {
            sr.k[l].assign(elems, 0);
            sr.v[l].assign(elems, 0);
        }
        sr.allocated = true;
    }

    void run_matmul(const Df2Tensor& w, const float* x, float* y, int M) {
        const int N = w.rows;
        const int K = w.cols;
        const int nb = K >> 5;
        if ((int)xq_buf.size() < M * nb) xq_buf.resize((size_t)M * nb);
        quantize_rows_q8(x, M, K, xq_buf.data());
        const ActQ8Block* xqp = xq_buf.data();
        if (w.dtype == Df2Dtype::Q4_0) {
            pool->run([&](int tid, int nt) {
                const int r0 = (N * tid) / nt;
                const int r1 = (N * (tid + 1)) / nt;
                if (r0 < r1) gemm_q4_0_rows(w.ptr, N, K, xqp, M, y, r0, r1);
            });
        } else if (w.dtype == Df2Dtype::Q8_0) {
            pool->run([&](int tid, int nt) {
                const int r0 = (N * tid) / nt;
                const int r1 = (N * (tid + 1)) / nt;
                if (r0 < r1) gemm_q8_0_rows(w.ptr, N, K, xqp, M, y, r0, r1);
            });
        } else {
            throw std::runtime_error("InkDFlash2Cpu: unsupported matmul weight dtype");
        }
    }

    // Small single-threaded matmul (for selector hp [7, 256] = out[7, 2304] @ W^T, < 20 us!)
    void run_matmul_small(const Df2Tensor& w, const float* x, float* y, int M) const {
        const int N = w.rows;
        const int K = w.cols;
        const int nb = K >> 5;
        std::vector<ActQ8Block> xq((size_t)M * nb);
        quantize_rows_q8(x, M, K, xq.data());
        if (w.dtype == Df2Dtype::Q4_0) gemm_q4_0_rows(w.ptr, N, K, xq.data(), M, y, 0, N);
        else if (w.dtype == Df2Dtype::Q8_0) gemm_q8_0_rows(w.ptr, N, K, xq.data(), M, y, 0, N);
        else throw std::runtime_error("InkDFlash2Cpu: unsupported small matmul dtype");
    }

    static void decode_codebook_row(const Df2Tensor& cb, int row, float* out) {
        const int cols = cb.cols;
        if (row < 0 || row >= cb.rows) { std::memset(out, 0, (size_t)cols * sizeof(float)); return; }
        if (cb.dtype == Df2Dtype::BF16) {
            const uint16_t* src = reinterpret_cast<const uint16_t*>(cb.ptr) + (size_t)row * cols;
            for (int i = 0; i < cols; ++i) out[i] = bf16_bits_to_f32(src[i]);
        } else if (cb.dtype == Df2Dtype::Q4_0) {
            const int nb = cols >> 5;
            const uint8_t* src = cb.ptr + (size_t)row * nb * 18;
            for (int b = 0; b < nb; ++b) {
                const uint8_t* bp = src + (size_t)b * 18;
                uint16_t dh;
                std::memcpy(&dh, bp, 2);
                const float d = fp16_bits_to_f32(dh);
                float* dst = out + (size_t)b * 32;
                for (int i = 0; i < 16; ++i) {
                    dst[i]      = ((float)(bp[2 + i] & 0x0F) - 8.0f) * d;
                    dst[i + 16] = ((float)(bp[2 + i] >> 4)   - 8.0f) * d;
                }
            }
        } else {
            throw std::runtime_error("InkDFlash2Cpu: unsupported codebook dtype");
        }
    }
};

InkDFlash2Cpu::InkDFlash2Cpu() = default;
InkDFlash2Cpu::~InkDFlash2Cpu() { free_all(); }

bool InkDFlash2Cpu::init(const Df2Pack* pack, int max_slots, bool alloc_rings, int num_threads) {
    if (!pack) return false;
    impl_ = std::make_unique<Impl>();
    impl_->pack = pack;
    impl_->max_slots = std::max(1, max_slots);
    impl_->rings_on = alloc_rings;

    if (num_threads <= 0) {
        if (const char* e = std::getenv("STRATA_INK_DF2_CPU_THREADS")) num_threads = std::atoi(e);
        if (num_threads <= 0) {
            int hw = (int)std::thread::hardware_concurrency();
            num_threads = std::max(1, std::min(16, hw));
        }
    }
    if (alloc_rings) {
        impl_->pool = std::make_unique<Df2ThreadPool>(num_threads);
    }

    impl_->t_fc       = &pack->require("fc.weight");
    impl_->t_hnorm    = &pack->require("hidden_norm.weight");
    impl_->t_norm     = &pack->require("norm.weight");
    impl_->t_sel_hp   = &pack->require("candidate_selector.hidden_projection.weight");
    impl_->t_sel_pred = &pack->require("candidate_selector.predecessor_codebook");
    impl_->t_sel_succ = &pack->require("candidate_selector.successor_codebook");

    impl_->L.resize(pack->layers);
    for (int l = 0; l < pack->layers; ++l) {
        const std::string p = "layers." + std::to_string(l) + ".";
        auto& lp = impl_->L[l];
        lp.in_ln   = &pack->require(p + "input_layernorm.weight");
        lp.post_ln = &pack->require(p + "post_attention_layernorm.weight");
        lp.q_proj  = &pack->require(p + "self_attn.q_proj.weight");
        lp.k_proj  = &pack->require(p + "self_attn.k_proj.weight");
        lp.v_proj  = &pack->require(p + "self_attn.v_proj.weight");
        lp.o_proj  = &pack->require(p + "self_attn.o_proj.weight");
        lp.q_norm  = &pack->require(p + "self_attn.q_norm.weight");
        lp.k_norm  = &pack->require(p + "self_attn.k_norm.weight");
        lp.gate    = &pack->require(p + "mlp.gate_proj.weight");
        lp.up      = &pack->require(p + "mlp.up_proj.weight");
        lp.down    = &pack->require(p + "mlp.down_proj.weight");
        lp.ac_kp   = &pack->require(p + "attention_conv.kernel_projection.weight");
        lp.ac_bk   = &pack->require(p + "attention_conv.base_kernel");
        lp.mc_kp   = &pack->require(p + "mlp_conv.kernel_projection.weight");
        lp.mc_bk   = &pack->require(p + "mlp_conv.base_kernel");
    }

    const int half = pack->head_dim >> 1;
    impl_->inv_freq.resize(half);
    for (int d = 0; d < half; ++d) {
        impl_->inv_freq[d] = 1.0f / std::pow(pack->rope_theta, (float)(2 * d) / (float)pack->head_dim);
    }

    if (alloc_rings) {
        impl_->rings.resize(impl_->max_slots);
        const int B = pack->block;
        const int H = pack->hidden;
        const int Q = pack->heads * pack->head_dim;
        const int KV = pack->kv_heads * pack->head_dim;
        const int I = pack->inter;
        const int conv_out = 2 * pack->conv_kernel * (H / pack->conv_group);
        impl_->h.resize((size_t)B * H);
        impl_->xn.resize((size_t)B * H);
        impl_->xc.resize((size_t)B * H);
        impl_->dyn.resize((size_t)B * conv_out);
        impl_->q.resize((size_t)B * Q);
        impl_->k.resize((size_t)B * KV);
        impl_->v.resize((size_t)B * KV);
        impl_->ao.resize((size_t)B * Q);
        impl_->o.resize((size_t)B * H);
        impl_->gate.resize((size_t)B * I);
        impl_->up.resize((size_t)B * I);
        impl_->mlp.resize((size_t)B * H);
        impl_->fcy.resize((size_t)8 * H);
        impl_->feats.resize((size_t)8 * pack->taps.size() * H);
    }
    return true;
}

void InkDFlash2Cpu::free_all() { impl_.reset(); }

void InkDFlash2Cpu::reset_slot(int slot) {
    if (!impl_ || !impl_->rings_on || slot < 0 || slot >= (int)impl_->rings.size()) return;
    // Positions are tracked externally; zeroing is unnecessary because attention only reads < s.
}

void InkDFlash2Cpu::ingest_taps(int slot, const float* const taps4[4], int p0, int n) {
    if (!impl_ || !impl_->rings_on || n <= 0) return;
    auto& d = *impl_;
    const auto& pk = *d.pack;
    d.ensure_slot_ring(slot);
    auto& sr = d.rings[slot];
    const int H = pk.hidden;
    const int ntaps = (int)pk.taps.size();
    const int KV = pk.kv_heads * pk.head_dim;

    // Process in chunks of up to 8 rows (fits optimal AVX2 multi-row GEMM!)
    for (int off = 0; off < n; off += 8) {
        const int C = std::min(8, n - off);
        for (int i = 0; i < C; ++i) {
            float* dst = d.feats.data() + (size_t)i * ntaps * H;
            for (int t = 0; t < ntaps; ++t) {
                std::memcpy(dst + (size_t)t * H, taps4[t] + (size_t)(off + i) * H, (size_t)H * sizeof(float));
            }
        }
        d.run_matmul(*d.t_fc, d.feats.data(), d.fcy.data(), C);
        rms_norm_rows(d.fcy.data(), reinterpret_cast<const float*>(d.t_hnorm->ptr), d.fcy.data(), C, H, pk.eps);

        int pos[8];
        for (int i = 0; i < C; ++i) pos[i] = p0 + off + i;

        for (int l = 0; l < pk.layers; ++l) {
            const auto& lp = d.L[l];
            d.run_matmul(*lp.k_proj, d.fcy.data(), d.k.data(), C);
            d.run_matmul(*lp.v_proj, d.fcy.data(), d.v.data(), C);
            rms_rope_heads(d.k.data(), reinterpret_cast<const float*>(lp.k_norm->ptr),
                           d.inv_freq.data(), pos, C, pk.kv_heads, pk.head_dim, pk.eps);
            for (int i = 0; i < C; ++i) {
                const int rslot = (pos[i] % kRingCap + kRingCap) % kRingCap;
                uint16_t* kd = sr.k[l].data() + (size_t)rslot * KV;
                uint16_t* vd = sr.v[l].data() + (size_t)rslot * KV;
                const float* ks = d.k.data() + (size_t)i * KV;
                const float* vs = d.v.data() + (size_t)i * KV;
                for (int j = 0; j < KV; ++j) {
                    kd[j] = f32_to_fp16_bits(ks[j]);
                    vd[j] = f32_to_fp16_bits(vs[j]);
                }
            }
        }
    }
}

void InkDFlash2Cpu::forward_block(int slot, const float* embed_in, int anchor_pos, float* out_hidden7) {
    auto& d = *impl_;
    const auto& pk = *d.pack;
    d.ensure_slot_ring(slot);
    auto& sr = d.rings[slot];
    const int B = pk.block; // 8
    const int H = pk.hidden;
    const int NH = pk.heads;
    const int NKV = pk.kv_heads;
    const int HD = pk.head_dim;
    const int I = pk.inter;
    const int rep = NH / NKV;
    const int groups = H / pk.conv_group;
    const int dyn_stride = 2 * pk.conv_kernel * groups;
    const float attn_scale = 1.0f / std::sqrt((float)HD);

    std::memcpy(d.h.data(), embed_in, (size_t)B * H * sizeof(float));
    int pos[8];
    for (int i = 0; i < B; ++i) pos[i] = anchor_pos + i;
    const int lo = std::max(0, anchor_pos - pk.window + 1);

    for (int l = 0; l < pk.layers; ++l) {
        const auto& lp = d.L[l];

        // 1. Attention sublayer: RMSNorm -> dynamic conv prepare -> Q/K/V -> norm+RoPE -> SWA+block attn -> O -> conv finish + residual
        rms_norm_rows(d.h.data(), reinterpret_cast<const float*>(lp.in_ln->ptr), d.xn.data(), B, H, pk.eps);
        d.run_matmul(*lp.ac_kp, d.xn.data(), d.dyn.data(), B);
        const float* ac_bk = reinterpret_cast<const float*>(lp.ac_bk->ptr);
        apply_dyn_conv(d.xn.data(), d.dyn.data(), ac_bk, d.xc.data(), B, H, pk.conv_group, pk.conv_kernel, dyn_stride);

        d.run_matmul(*lp.q_proj, d.xc.data(), d.q.data(), B);
        d.run_matmul(*lp.k_proj, d.xc.data(), d.k.data(), B);
        d.run_matmul(*lp.v_proj, d.xc.data(), d.v.data(), B);

        rms_rope_heads(d.q.data(), reinterpret_cast<const float*>(lp.q_norm->ptr), d.inv_freq.data(), pos, B, NH, HD, pk.eps);
        rms_rope_heads(d.k.data(), reinterpret_cast<const float*>(lp.k_norm->ptr), d.inv_freq.data(), pos, B, NKV, HD, pk.eps);

        // KV-Outer Fused SWA-2048 + Block-8 GQA Attention (32x less DDR4 traffic: reads each FP16 KV row ONCE for all 4 Q heads x 8 block rows!)
        const uint16_t* rk_l = sr.k[l].data();
        const uint16_t* rv_l = sr.v[l].data();
        const int c_total = anchor_pos - lo; // context positions [lo .. anchor_pos - 1]
        // 16 tasks = 8 KV heads x 2 K-splits (for full 16-core saturation even at c_total = 2048!)
        constexpr int kSplits = 2;
        struct SplitOut {
            float m[4][8];
            float l[4][8];
            float o[4][8][128];
        };
        SplitOut sp_buf[8 * kSplits];
        d.pool->run([&](int tid, int nt) {
            const int n_tasks = NKV * kSplits;
            for (int task = tid; task < n_tasks; task += nt) {
                const int kvh = task / kSplits;
                const int sp  = task % kSplits;
                auto& so = sp_buf[task];
                for (int r = 0; r < rep; ++r) {
                    for (int qi = 0; qi < B; ++qi) {
                        so.m[r][qi] = -1e30f;
                        so.l[r][qi] = 0.0f;
                        std::memset(so.o[r][qi], 0, (size_t)HD * sizeof(float));
                    }
                }
                const float* qptrs[4][8];
                for (int r = 0; r < rep; ++r) {
                    const int qh = kvh * rep + r;
                    for (int qi = 0; qi < B; ++qi) {
                        qptrs[r][qi] = d.q.data() + ((size_t)qi * NH + qh) * HD;
                    }
                }
                const int ci_beg = (c_total * sp) / kSplits;
                const int ci_end = (c_total * (sp + 1)) / kSplits;

                auto attend_one_kv = [&](const float* kf, const float* vf, int kpos, bool is_blk) {
                    for (int r = 0; r < rep; ++r) {
                        for (int qi = 0; qi < B; ++qi) {
                            if (!is_blk && (anchor_pos + qi) - kpos >= pk.window) continue;
                            const float* qp = qptrs[r][qi];
                            float dot = 0.0f;
#if defined(__AVX2__)
                            __m256 vsum0 = _mm256_setzero_ps(), vsum1 = _mm256_setzero_ps();
                            for (int dd = 0; dd < HD; dd += 16) {
                                vsum0 = _mm256_fmadd_ps(_mm256_loadu_ps(qp + dd),     _mm256_loadu_ps(kf + dd),     vsum0);
                                vsum1 = _mm256_fmadd_ps(_mm256_loadu_ps(qp + dd + 8), _mm256_loadu_ps(kf + dd + 8), vsum1);
                            }
                            dot = hsum256_ps(_mm256_add_ps(vsum0, vsum1));
#else
                            for (int dd = 0; dd < HD; ++dd) dot += qp[dd] * kf[dd];
#endif
                            const float sc = dot * attn_scale;
                            const float mn = std::max(so.m[r][qi], sc);
                            const float al = std::exp(so.m[r][qi] - mn);
                            const float p  = std::exp(sc - mn);
                            so.l[r][qi] = so.l[r][qi] * al + p;
                            so.m[r][qi] = mn;
                            float* op = so.o[r][qi];
#if defined(__AVX2__)
                            __m256 val = _mm256_set1_ps(al);
                            __m256 vp  = _mm256_set1_ps(p);
                            for (int dd = 0; dd < HD; dd += 8) {
                                __m256 vo = _mm256_mul_ps(_mm256_loadu_ps(op + dd), val);
                                _mm256_storeu_ps(op + dd, _mm256_fmadd_ps(vp, _mm256_loadu_ps(vf + dd), vo));
                            }
#else
                            for (int dd = 0; dd < HD; ++dd) op[dd] = op[dd] * al + p * vf[dd];
#endif
                        }
                    }
                };

                alignas(32) float kf[128], vf[128];
                for (int ci = ci_beg; ci < ci_end; ++ci) {
                    const int kpos = lo + ci;
                    const int rslot = kpos % kRingCap;
                    const uint16_t* khp = rk_l + ((size_t)rslot * NKV + kvh) * HD;
                    const uint16_t* vhp = rv_l + ((size_t)rslot * NKV + kvh) * HD;
#if defined(__AVX2__) && defined(__F16C__)
                    for (int dd = 0; dd < HD; dd += 8) {
                        _mm256_store_ps(kf + dd, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(khp + dd))));
                        _mm256_store_ps(vf + dd, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vhp + dd))));
                    }
#else
                    for (int dd = 0; dd < HD; ++dd) { kf[dd] = fp16_bits_to_f32(khp[dd]); vf[dd] = fp16_bits_to_f32(vhp[dd]); }
#endif
                    attend_one_kv(kf, vf, kpos, false);
                }
                if (sp == 0) {
                    for (int bi = 0; bi < B; ++bi) {
                        const float* kp = d.k.data() + ((size_t)bi * NKV + kvh) * HD;
                        const float* vp = d.v.data() + ((size_t)bi * NKV + kvh) * HD;
                        attend_one_kv(kp, vp, anchor_pos + bi, true);
                    }
                }
            }
        });
        // Merge the 2 splits per KV head
        for (int kvh = 0; kvh < NKV; ++kvh) {
            const auto& s0 = sp_buf[kvh * kSplits + 0];
            const auto& s1 = sp_buf[kvh * kSplits + 1];
            for (int r = 0; r < rep; ++r) {
                const int qh = kvh * rep + r;
                for (int qi = 0; qi < B; ++qi) {
                    const float mg = std::max(s0.m[r][qi], s1.m[r][qi]);
                    const float w0 = (s0.l[r][qi] > 0.0f) ? std::exp(s0.m[r][qi] - mg) : 0.0f;
                    const float w1 = (s1.l[r][qi] > 0.0f) ? std::exp(s1.m[r][qi] - mg) : 0.0f;
                    const float inv_l = 1.0f / (s0.l[r][qi] * w0 + s1.l[r][qi] * w1 + 1e-12f);
                    float* dst = d.ao.data() + ((size_t)qi * NH + qh) * HD;
                    for (int dd = 0; dd < HD; ++dd) {
                        dst[dd] = (s0.o[r][qi][dd] * w0 + s1.o[r][qi][dd] * w1) * inv_l;
                    }
                }
            }
        }

        d.run_matmul(*lp.o_proj, d.ao.data(), d.o.data(), B);
        apply_dyn_conv(d.o.data(), d.dyn.data() + pk.conv_kernel * groups, ac_bk + pk.conv_kernel * H,
                       d.xc.data(), B, H, pk.conv_group, pk.conv_kernel, dyn_stride);
        for (int i = 0; i < B * H; ++i) d.h[i] += d.xc[i];

        // 2. MLP sublayer: RMSNorm -> dynamic conv prepare -> SwiGLU -> down -> conv finish + residual
        rms_norm_rows(d.h.data(), reinterpret_cast<const float*>(lp.post_ln->ptr), d.xn.data(), B, H, pk.eps);
        d.run_matmul(*lp.mc_kp, d.xn.data(), d.dyn.data(), B);
        const float* mc_bk = reinterpret_cast<const float*>(lp.mc_bk->ptr);
        apply_dyn_conv(d.xn.data(), d.dyn.data(), mc_bk, d.xc.data(), B, H, pk.conv_group, pk.conv_kernel, dyn_stride);

        d.run_matmul(*lp.gate, d.xc.data(), d.gate.data(), B);
        d.run_matmul(*lp.up, d.xc.data(), d.up.data(), B);
        for (int i = 0; i < B * I; ++i) {
            const float g = d.gate[i];
            d.gate[i] = (g / (1.0f + std::exp(-g))) * d.up[i];
        }
        d.run_matmul(*lp.down, d.gate.data(), d.mlp.data(), B);
        apply_dyn_conv(d.mlp.data(), d.dyn.data() + pk.conv_kernel * groups, mc_bk + pk.conv_kernel * H,
                       d.xc.data(), B, H, pk.conv_group, pk.conv_kernel, dyn_stride);
        for (int i = 0; i < B * H; ++i) d.h[i] += d.xc[i];
    }

    // Final norm on rows 1..B-1 -> out_hidden7 [7, H]
    rms_norm_rows(d.h.data() + H, reinterpret_cast<const float*>(d.t_norm->ptr), out_hidden7, B - 1, H, pk.eps);
}

void InkDFlash2Cpu::select_path(
    const float* out_hidden7,
    const float* unary16,
    const int32_t* cand16,
    int32_t anchor_tok,
    int32_t* out_ids7,
    float* out_conf7) const {
    const auto& d = *impl_;
    const auto& pk = *d.pack;
    const int n_draft = pk.block - 1; // 7
    const int R = pk.rank;            // 256
    const int K = pk.top_k;           // 16
    const float cap = pk.softcap;     // 30.0f
    const float inv_cap = (cap > 0.0f) ? (1.0f / cap) : 0.0f;

    float hp[7 * 256];
    d.run_matmul_small(*d.t_sel_hp, out_hidden7, hp, n_draft);

    float a_vec[256], b_vec[256], ap[256], sc[32];
    int32_t pred = anchor_tok;
    for (int i = 0; i < n_draft; ++i) {
        Impl::decode_codebook_row(*d.t_sel_pred, pred, a_vec);
        const float* hpi = hp + (size_t)i * R;
        for (int r = 0; r < R; ++r) ap[r] = a_vec[r] * hpi[r];

        int best_k = 0;
        float best_sc = -1e30f;
        for (int k = 0; k < K; ++k) {
            const int32_t cid = cand16[(size_t)i * K + k];
            Impl::decode_codebook_row(*d.t_sel_succ, cid, b_vec);
            float corr = 0.0f;
#if defined(__AVX2__)
            __m256 vs = _mm256_setzero_ps();
            for (int r = 0; r < R; r += 8) {
                vs = _mm256_fmadd_ps(_mm256_loadu_ps(b_vec + r), _mm256_loadu_ps(ap + r), vs);
            }
            corr = hsum256_ps(vs);
#else
            for (int r = 0; r < R; ++r) corr += b_vec[r] * ap[r];
#endif
            float val = unary16[(size_t)i * K + k] + corr;
            if (cap > 0.0f) val = cap * std::tanh(val * inv_cap);
            sc[k] = val;
            if (val > best_sc) { best_sc = val; best_k = k; }
        }
        float sum_e = 0.0f;
        for (int k = 0; k < K; ++k) sum_e += std::exp(sc[k] - best_sc);
        out_ids7[i] = cand16[(size_t)i * K + best_k];
        out_conf7[i] = 1.0f / sum_e;
        pred = out_ids7[i];
    }
}

void InkDFlash2Cpu::matmul(const Df2Tensor& w, const float* x, float* y, int M) const {
    impl_->run_matmul(w, x, y, M);
}

int InkDFlash2Cpu::num_threads() const { return (impl_ && impl_->pool) ? impl_->pool->size() : 0; }
bool InkDFlash2Cpu::rings_ready() const { return impl_ && impl_->rings_on; }

} // namespace strata::ink

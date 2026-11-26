#include "ink_host_kv_attn.hpp"

#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::ink {

namespace {

constexpr int kHeads    = 24;
constexpr int kKVHeads  = 8;
constexpr int kHeadDim  = 128;
constexpr int kBlockTok = 256;
constexpr int kRelExt   = 1024;
constexpr int kMaxM     = 16;
constexpr float kScale  = 0.08838834764831845f;

static inline float hsum256(__m256 v) noexcept {
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 s4 = _mm_add_ps(lo, hi);
    __m128 s2 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
    __m128 s1 = _mm_add_ss(s2, _mm_shuffle_ps(s2, s2, 1));
    return _mm_cvtss_f32(s1);
}

struct HostAttnPool {
    static constexpr int kMaxThr = 32;
    struct Job {
        const float* q = nullptr;
        const float* rel_table = nullptr;
        const void* const* slab_k = nullptr;
        const void* const* slab_v = nullptr;
        const int32_t* slot_qp = nullptr;
        const InkHostKvUniqueBlock* ublocks = nullptr;
        int n_ublocks = 0;
        int M = 0;
        int n_thr = 1;
        float* thr_out = nullptr;
    };

    std::vector<std::thread> workers;
    std::mutex mu;
    std::condition_variable cv_go;
    std::atomic<uint64_t> gen{0};
    std::atomic<int> done_cnt{0};
    bool stop = false;
    int num_workers = 0;
    Job cur_job{};
    std::vector<float> scratch;

    HostAttnPool() {
        int hw = (int)std::thread::hardware_concurrency();
        if (hw < 1) hw = 1;
        int want = std::min(hw, 16);
        if (const char* e = std::getenv("STRATA_INK_HOST_KV_THREADS")) {
            int v = std::atoi(e);
            if (v > 0) want = std::min(v, kMaxThr);
        }
        num_workers = std::max(0, want - 1);
        workers.reserve((size_t)num_workers);
        for (int t = 1; t <= num_workers; ++t) {
            workers.emplace_back([this, t]() { worker_loop(t); });
        }
    }

    ~HostAttnPool() {
        {
            std::lock_guard<std::mutex> lk(mu);
            stop = true;
            gen.fetch_add(1, std::memory_order_release);
        }
        cv_go.notify_all();
        for (auto& w : workers) if (w.joinable()) w.join();
    }

    static void run_slice(const Job& job, int tid) noexcept {
        const int M = job.M;
        const int n_thr = job.n_thr;
        float* my_base = job.thr_out + (size_t)tid * M * kHeads * 130;

        for (int m = 0; m < M; ++m) {
            for (int h = 0; h < kHeads; ++h) {
                float* dst = my_base + ((size_t)m * kHeads + h) * 130;
                std::memset(dst, 0, kHeadDim * sizeof(float));
                dst[128] = -1e30f;
                dst[129] = 0.0f;
            }
        }

        const int ub_per = (job.n_ublocks + n_thr - 1) / n_thr;
        const int ub_beg = tid * ub_per;
        const int ub_end = std::min(job.n_ublocks, ub_beg + ub_per);
        if (ub_beg >= ub_end) return;

        // Pre-scale Q for all M active slots (hot in L1/L2 cache: <= 192 KB even for M=16!)
        alignas(32) float qs[kMaxM][kHeads][kHeadDim];
        const __m256 vsc = _mm256_set1_ps(kScale);
        for (int m = 0; m < M; ++m) {
            const float* qm = job.q + (size_t)m * kHeads * kHeadDim;
            for (int h = 0; h < kHeads; ++h) {
                const float* qh = qm + (size_t)h * kHeadDim;
                float* qsh = qs[m][h];
                for (int d = 0; d < kHeadDim; d += 8) {
                    _mm256_store_ps(qsh + d, _mm256_mul_ps(_mm256_loadu_ps(qh + d), vsc));
                }
            }
        }

        // Thread-local accumulators for all M slots (for M=8: 8*24*128*4 = 96 KB -> fits in Zen2 512 KB L2!)
        alignas(32) float o_loc[kMaxM][kHeads][kHeadDim];
        float m_loc[kMaxM][kHeads];
        float l_loc[kMaxM][kHeads];
        for (int m = 0; m < M; ++m) {
            std::memset(o_loc[m], 0, sizeof(o_loc[m]));
            for (int h = 0; h < kHeads; ++h) {
                m_loc[m][h] = -1e30f;
                l_loc[m][h] = 0.0f;
            }
        }

        alignas(32) float kf[kHeadDim];
        alignas(32) float vf[kHeadDim];

        // Scan each assigned unique physical block ONCE from DDR4 and fan out to all slots in slot_mask!
        for (int bi = ub_beg; bi < ub_end; ++bi) {
            const InkHostKvUniqueBlock ub = job.ublocks[bi];
            const uint32_t mask = ub.slot_mask;
            if (mask == 0u) continue;

            int ms[kMaxM];
            int nm = 0;
            uint32_t tmp = mask;
            while (tmp) {
                int m = __builtin_ctz(tmp);
                tmp &= (tmp - 1u);
                if (m < M) ms[nm++] = m;
            }
            if (nm == 0) continue;

            const uint16_t* sk = reinterpret_cast<const uint16_t*>(job.slab_k[ub.slab])
                               + (size_t)ub.boff * kBlockTok * kKVHeads * kHeadDim;
            const uint16_t* sv = reinterpret_cast<const uint16_t*>(job.slab_v[ub.slab])
                               + (size_t)ub.boff * kBlockTok * kKVHeads * kHeadDim;

            if (nm == 1) {
                // Fast register-resident single-slot path (zero stack kf/vf spill!)
                const int m = ms[0];
                const int qp = job.slot_qp[m];
                const int plen = std::min(ub.max_pos_len, qp - ub.pos_start + 1);
                if (plen <= 0) continue;
                const float* rtm = job.rel_table ? (job.rel_table + (size_t)m * kHeads * (kRelExt + 1)) : nullptr;
                const bool need_rel = (rtm != nullptr) && ((qp - (ub.pos_start + plen - 1)) < kRelExt);

                for (int t = 0; t < plen; ++t) {
                    const int kp = ub.pos_start + t;
                    const int dist = need_rel ? std::min(std::max(qp - kp, 0), kRelExt) : kRelExt;
                    const uint16_t* kt_base = sk + (size_t)t * kKVHeads * kHeadDim;
                    const uint16_t* vt_base = sv + (size_t)t * kKVHeads * kHeadDim;

                    if (t + 2 < plen) {
                        _mm_prefetch(reinterpret_cast<const char*>(sk + (size_t)(t + 2) * kKVHeads * kHeadDim), _MM_HINT_T0);
                    }

                    for (int kh = 0; kh < kKVHeads; ++kh) {
                        const int h0 = 3 * kh;
                        const uint16_t* kp_h = kt_base + (size_t)kh * kHeadDim;
                        const float* q0 = qs[m][h0 + 0];
                        const float* q1 = qs[m][h0 + 1];
                        const float* q2 = qs[m][h0 + 2];

                        __m256 d0a = _mm256_setzero_ps(), d0b = _mm256_setzero_ps();
                        __m256 d1a = _mm256_setzero_ps(), d1b = _mm256_setzero_ps();
                        __m256 d2a = _mm256_setzero_ps(), d2b = _mm256_setzero_ps();
                        for (int d = 0; d < kHeadDim; d += 16) {
                            __m256 k0 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kp_h + d)));
                            __m256 k1 = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kp_h + d + 8)));
                            d0a = _mm256_fmadd_ps(_mm256_load_ps(q0 + d),     k0, d0a);
                            d0b = _mm256_fmadd_ps(_mm256_load_ps(q0 + d + 8), k1, d0b);
                            d1a = _mm256_fmadd_ps(_mm256_load_ps(q1 + d),     k0, d1a);
                            d1b = _mm256_fmadd_ps(_mm256_load_ps(q1 + d + 8), k1, d1b);
                            d2a = _mm256_fmadd_ps(_mm256_load_ps(q2 + d),     k0, d2a);
                            d2b = _mm256_fmadd_ps(_mm256_load_ps(q2 + d + 8), k1, d2b);
                        }
                        float sc[3] = {
                            hsum256(_mm256_add_ps(d0a, d0b)),
                            hsum256(_mm256_add_ps(d1a, d1b)),
                            hsum256(_mm256_add_ps(d2a, d2b))
                        };
                        if (need_rel) {
                            sc[0] += rtm[(size_t)(h0 + 0) * (kRelExt + 1) + dist];
                            sc[1] += rtm[(size_t)(h0 + 1) * (kRelExt + 1) + dist];
                            sc[2] += rtm[(size_t)(h0 + 2) * (kRelExt + 1) + dist];
                        }

                        float al[3], pv[3];
                        for (int r = 0; r < 3; ++r) {
                            const int h = h0 + r;
                            const float mn = (sc[r] > m_loc[m][h]) ? sc[r] : m_loc[m][h];
                            al[r] = std::exp(m_loc[m][h] - mn);
                            pv[r] = std::exp(sc[r] - mn);
                            l_loc[m][h] = l_loc[m][h] * al[r] + pv[r];
                            m_loc[m][h] = mn;
                        }

                        const bool act0 = (al[0] != 1.0f) || (pv[0] > l_loc[m][h0 + 0] * 1e-10f);
                        const bool act1 = (al[1] != 1.0f) || (pv[1] > l_loc[m][h0 + 1] * 1e-10f);
                        const bool act2 = (al[2] != 1.0f) || (pv[2] > l_loc[m][h0 + 2] * 1e-10f);
                        if (!act0 && !act1 && !act2) continue;

                        const uint16_t* vp_h = vt_base + (size_t)kh * kHeadDim;
                        const __m256 va0 = _mm256_set1_ps(al[0]), vp0 = _mm256_set1_ps(pv[0]);
                        const __m256 va1 = _mm256_set1_ps(al[1]), vp1 = _mm256_set1_ps(pv[1]);
                        const __m256 va2 = _mm256_set1_ps(al[2]), vp2 = _mm256_set1_ps(pv[2]);
                        float* o0 = o_loc[m][h0 + 0];
                        float* o1 = o_loc[m][h0 + 1];
                        float* o2 = o_loc[m][h0 + 2];
                        for (int d = 0; d < kHeadDim; d += 8) {
                            __m256 vv = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vp_h + d)));
                            if (act0) _mm256_store_ps(o0 + d, _mm256_fmadd_ps(vp0, vv, _mm256_mul_ps(_mm256_load_ps(o0 + d), va0)));
                            if (act1) _mm256_store_ps(o1 + d, _mm256_fmadd_ps(vp1, vv, _mm256_mul_ps(_mm256_load_ps(o1 + d), va1)));
                            if (act2) _mm256_store_ps(o2 + d, _mm256_fmadd_ps(vp2, vv, _mm256_mul_ps(_mm256_load_ps(o2 + d), va2)));
                        }
                    }
                }
            } else {
                // SHARED MULTI-AGENT PATH (nm = 2..16): load each FP16 K and V from DDR4 ONCE, reuse across all nm slots!
                bool any_rel = false;
                if (job.rel_table != nullptr) {
                    for (int k = 0; k < nm; ++k) {
                        const int qp = job.slot_qp[ms[k]];
                        if ((qp - (ub.pos_start + ub.max_pos_len - 1)) < kRelExt) { any_rel = true; break; }
                    }
                }

                for (int t = 0; t < ub.max_pos_len; ++t) {
                    const int kp = ub.pos_start + t;
                    const uint16_t* kt_base = sk + (size_t)t * kKVHeads * kHeadDim;
                    const uint16_t* vt_base = sv + (size_t)t * kKVHeads * kHeadDim;

                    if (t + 2 < ub.max_pos_len) {
                        _mm_prefetch(reinterpret_cast<const char*>(sk + (size_t)(t + 2) * kKVHeads * kHeadDim), _MM_HINT_T0);
                    }

                    for (int kh = 0; kh < kKVHeads; ++kh) {
                        const int h0 = 3 * kh;
                        // 1. Dequantize FP16 K[128] -> L1 kf[128] ONCE for all nm agents!
                        const uint16_t* kp_h = kt_base + (size_t)kh * kHeadDim;
                        for (int d = 0; d < kHeadDim; d += 8) {
                            _mm256_store_ps(kf + d, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kp_h + d))));
                        }

                        float al_arr[kMaxM][3];
                        float pv_arr[kMaxM][3];
                        uint8_t act_arr[kMaxM];
                        bool any_v = false;

                        // 2. Score against all nm agents sharing this block (100% L1/L2 cache!)
                        for (int k = 0; k < nm; ++k) {
                            const int m = ms[k];
                            const int qp = job.slot_qp[m];
                            if (kp > qp) { act_arr[k] = 0; continue; }

                            const float* q0 = qs[m][h0 + 0];
                            const float* q1 = qs[m][h0 + 1];
                            const float* q2 = qs[m][h0 + 2];
                            __m256 d0a = _mm256_setzero_ps(), d0b = _mm256_setzero_ps();
                            __m256 d1a = _mm256_setzero_ps(), d1b = _mm256_setzero_ps();
                            __m256 d2a = _mm256_setzero_ps(), d2b = _mm256_setzero_ps();
                            for (int d = 0; d < kHeadDim; d += 16) {
                                __m256 k0 = _mm256_load_ps(kf + d);
                                __m256 k1 = _mm256_load_ps(kf + d + 8);
                                d0a = _mm256_fmadd_ps(_mm256_load_ps(q0 + d),     k0, d0a);
                                d0b = _mm256_fmadd_ps(_mm256_load_ps(q0 + d + 8), k1, d0b);
                                d1a = _mm256_fmadd_ps(_mm256_load_ps(q1 + d),     k0, d1a);
                                d1b = _mm256_fmadd_ps(_mm256_load_ps(q1 + d + 8), k1, d1b);
                                d2a = _mm256_fmadd_ps(_mm256_load_ps(q2 + d),     k0, d2a);
                                d2b = _mm256_fmadd_ps(_mm256_load_ps(q2 + d + 8), k1, d2b);
                            }
                            float sc[3] = {
                                hsum256(_mm256_add_ps(d0a, d0b)),
                                hsum256(_mm256_add_ps(d1a, d1b)),
                                hsum256(_mm256_add_ps(d2a, d2b))
                            };
                            if (any_rel && (qp - kp) < kRelExt) {
                                const int dist = qp - kp;
                                const float* rtm = job.rel_table + (size_t)m * kHeads * (kRelExt + 1);
                                sc[0] += rtm[(size_t)(h0 + 0) * (kRelExt + 1) + dist];
                                sc[1] += rtm[(size_t)(h0 + 1) * (kRelExt + 1) + dist];
                                sc[2] += rtm[(size_t)(h0 + 2) * (kRelExt + 1) + dist];
                            }

                            uint8_t am = 0;
                            for (int r = 0; r < 3; ++r) {
                                const int h = h0 + r;
                                const float mn = (sc[r] > m_loc[m][h]) ? sc[r] : m_loc[m][h];
                                const float al = std::exp(m_loc[m][h] - mn);
                                const float pv = std::exp(sc[r] - mn);
                                l_loc[m][h] = l_loc[m][h] * al + pv;
                                m_loc[m][h] = mn;
                                al_arr[k][r] = al;
                                pv_arr[k][r] = pv;
                                if ((al != 1.0f) || (pv > l_loc[m][h] * 1e-10f)) am |= (uint8_t)(1u << r);
                            }
                            act_arr[k] = am;
                            if (am) any_v = true;
                        }

                        // 3. Load FP16 V[128] -> L1 vf[128] ONCE if ANY agent needs it!
                        if (!any_v) continue;
                        const uint16_t* vp_h = vt_base + (size_t)kh * kHeadDim;
                        for (int d = 0; d < kHeadDim; d += 8) {
                            _mm256_store_ps(vf + d, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(vp_h + d))));
                        }

                        for (int k = 0; k < nm; ++k) {
                            const uint8_t am = act_arr[k];
                            if (!am) continue;
                            const int m = ms[k];
                            for (int r = 0; r < 3; ++r) {
                                if (!(am & (1u << r))) continue;
                                const __m256 va = _mm256_set1_ps(al_arr[k][r]);
                                const __m256 vp = _mm256_set1_ps(pv_arr[k][r]);
                                float* om = o_loc[m][h0 + r];
                                for (int d = 0; d < kHeadDim; d += 8) {
                                    __m256 vv = _mm256_load_ps(vf + d);
                                    _mm256_store_ps(om + d, _mm256_fmadd_ps(vp, vv, _mm256_mul_ps(_mm256_load_ps(om + d), va)));
                                }
                            }
                        }
                    }
                }
            }
        }

        for (int m = 0; m < M; ++m) {
            for (int h = 0; h < kHeads; ++h) {
                float* dst = my_base + ((size_t)m * kHeads + h) * 130;
                std::memcpy(dst, o_loc[m][h], kHeadDim * sizeof(float));
                dst[128] = m_loc[m][h];
                dst[129] = l_loc[m][h];
            }
        }
    }

    void worker_loop(int tid) {
        uint64_t my_gen = 0;
        while (true) {
            uint64_t g = gen.load(std::memory_order_acquire);
            if (g == my_gen) {
                for (int sp = 0; sp < 4000; ++sp) {
                    _mm_pause();
                    g = gen.load(std::memory_order_acquire);
                    if (g != my_gen) break;
                }
            }
            if (g == my_gen) {
                std::unique_lock<std::mutex> lk(mu);
                cv_go.wait(lk, [&]() { return stop || gen.load(std::memory_order_acquire) != my_gen; });
                if (stop) return;
                g = gen.load(std::memory_order_acquire);
            }
            if (stop) return;
            my_gen = g;
            if (tid < cur_job.n_thr) run_slice(cur_job, tid);
            done_cnt.fetch_add(1, std::memory_order_release);
        }
    }

    void execute(const float* q, const float* rel_table,
                 const void* const* slab_k, const void* const* slab_v,
                 const int32_t* slot_qp,
                 const InkHostKvUniqueBlock* ublocks, int n_ublocks,
                 int M, float* out_buf, int req_threads) {
        const int avail_thr = num_workers + 1;
        int n_thr = (req_threads > 0) ? std::min(req_threads, avail_thr) : avail_thr;
        n_thr = std::max(1, std::min(n_thr, std::max(1, n_ublocks)));

        const size_t per_thr = (size_t)M * kHeads * 130;
        if (scratch.size() < (size_t)n_thr * per_thr) scratch.resize((size_t)n_thr * per_thr);

        cur_job.q = q;
        cur_job.rel_table = rel_table;
        cur_job.slab_k = slab_k;
        cur_job.slab_v = slab_v;
        cur_job.slot_qp = slot_qp;
        cur_job.ublocks = ublocks;
        cur_job.n_ublocks = n_ublocks;
        cur_job.M = M;
        cur_job.n_thr = n_thr;
        cur_job.thr_out = scratch.data();

        if (n_thr > 1 && num_workers > 0) {
            done_cnt.store(0, std::memory_order_relaxed);
            {
                std::lock_guard<std::mutex> lk(mu);
                gen.fetch_add(1, std::memory_order_release);
            }
            cv_go.notify_all();
            run_slice(cur_job, 0);
            while (done_cnt.load(std::memory_order_acquire) < num_workers) _mm_pause();
        } else {
            run_slice(cur_job, 0);
        }

        for (int m = 0; m < M; ++m) {
            for (int h = 0; h < kHeads; ++h) {
                const size_t off = ((size_t)m * kHeads + h) * 130;
                float mg = -1e30f;
                for (int t = 0; t < n_thr; ++t) {
                    const float* s = scratch.data() + (size_t)t * per_thr + off;
                    if (s[129] > 0.0f && s[128] > mg) mg = s[128];
                }
                float* dst = out_buf + off;
                if (mg <= -1e29f) {
                    std::memset(dst, 0, kHeadDim * sizeof(float));
                    dst[128] = -1e30f;
                    dst[129] = 0.0f;
                    continue;
                }
                float lt = 0.0f;
                std::memset(dst, 0, kHeadDim * sizeof(float));
                for (int t = 0; t < n_thr; ++t) {
                    const float* s = scratch.data() + (size_t)t * per_thr + off;
                    if (s[129] <= 0.0f) continue;
                    const float w = std::exp(s[128] - mg);
                    lt += s[129] * w;
                    const __m256 vw = _mm256_set1_ps(w);
                    for (int d = 0; d < kHeadDim; d += 8) {
                        __m256 cur = _mm256_loadu_ps(dst + d);
                        __m256 add = _mm256_loadu_ps(s + d);
                        _mm256_storeu_ps(dst + d, _mm256_fmadd_ps(add, vw, cur));
                    }
                }
                dst[128] = mg;
                dst[129] = lt;
            }
        }
    }
};

HostAttnPool& pool_instance() {
    static HostAttnPool pool;
    return pool;
}

}  // namespace

void ink_host_kv_attn_fp16(
    const float* q,
    const float* rel_table,
    const void* const* slab_raw_k,
    const void* const* slab_raw_v,
    const int32_t* slot_qp,
    const InkHostKvUniqueBlock* ublocks,
    int n_ublocks,
    int M,
    float* out_buf,
    int n_threads) {
    if (M <= 0 || n_ublocks <= 0) return;
    pool_instance().execute(q, rel_table, slab_raw_k, slab_raw_v, slot_qp, ublocks, n_ublocks, M, out_buf, n_threads);
}

}  // namespace strata::ink

// mmid_det: whole-graph MUL_MAT_ID determinism, exactness and timing bench on Vulkan0, with an optional fused epilogue.
// usage: mmid_det <type> <m> <k> <n_tokens> <bcast_b 0|1> <fused 0|1> [time_reps=20]
//   type      weight type (ggml type name, e.g. q4_K); weights are [k x m] per expert
//   bcast_b   1 = one B row per token shared by every selected expert (gate / up form), 0 = one B row per selected expert (down form)
//   fused     1 = MUL_MAT_ID -> MUL by a per-expert router weight -> CAST f16 (the chain a MoE down projection runs), so a backend
//             that fuses the chain is exercised as one graph
// env: MMID_EXPERTS (default 512) experts in the stack, MMID_USED (default 10) experts selected per token, MMID_CPU_THREADS
//      (default: hardware threads) for the CPU reference.
// Runs the graph twice on Vulkan0 (bit-compare), then compares with an f32 CPU reference whose weights are the
// same quantized weights dequantized to f32 (so only the GPU path's arithmetic and activation handling differ), then times
// the whole graph (median of time_reps). Use it for fused chains: test-backend-ops perf times only the out node.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <thread>
#include <vector>

static int env_int(const char * name, int def) { const char * v = getenv(name); return v && *v ? atoi(v) : def; }
static const int N_MATS = env_int("MMID_EXPERTS", 512), N_USED = env_int("MMID_USED", 10);

static ggml_type type_by_name(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && strcmp(n, s) == 0) return (ggml_type) t;
    }
    fprintf(stderr, "unknown type %s\n", s); exit(2);
}

struct inputs { std::vector<uint8_t> a; std::vector<float> a_f32, b, w; std::vector<int32_t> ids; };

// build and run the graph on one backend; type_a = the weight type that backend sees
static std::vector<float> run(ggml_backend_t be, ggml_type type_a, const inputs & in, int64_t m, int64_t k, int64_t n,
        bool bcast, bool fused, int reps, std::vector<std::vector<uint8_t>> * raw, int time_reps = 0, double * us_med = nullptr) {
    ggml_init_params ip = { 64*ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * as  = ggml_new_tensor_3d(ctx, type_a, k, m, N_MATS);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, N_USED, n);
    ggml_tensor * b   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k, bcast ? 1 : N_USED, n);
    ggml_tensor * out = ggml_mul_mat_id(ctx, as, b, ids);
    ggml_tensor * w   = nullptr;
    if (fused) {
        w   = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, N_USED, n);
        out = ggml_mul(ctx, out, w);
        out = ggml_cast(ctx, out, GGML_TYPE_F16);
    }
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (type_a == GGML_TYPE_F32) ggml_backend_tensor_set(as, in.a_f32.data(), 0, ggml_nbytes(as));
    else                         ggml_backend_tensor_set(as, in.a.data(), 0, ggml_nbytes(as));
    ggml_backend_tensor_set(ids, in.ids.data(), 0, ggml_nbytes(ids));
    ggml_backend_tensor_set(b, in.b.data(), 0, ggml_nbytes(b));
    if (w) ggml_backend_tensor_set(w, in.w.data(), 0, ggml_nbytes(w));
    std::vector<float> res(ggml_nelements(out));
    for (int r = 0; r < reps; ++r) {
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(3); }
        std::vector<uint8_t> bytes(ggml_nbytes(out));
        ggml_backend_tensor_get(out, bytes.data(), 0, bytes.size());
        if (raw) raw->push_back(bytes);
        if (r == reps - 1) {
            if (out->type == GGML_TYPE_F16) {
                const ggml_fp16_t * h = (const ggml_fp16_t *) bytes.data();
                for (size_t i = 0; i < res.size(); ++i) res[i] = ggml_fp16_to_fp32(h[i]);
            } else {
                memcpy(res.data(), bytes.data(), bytes.size());
            }
        }
    }
    // whole-graph wall time (the fused chain included), median of time_reps after the checked runs
    std::vector<double> ts;
    for (int r = 0; r < time_reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); exit(3); }
        ggml_backend_synchronize(be);
        ts.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    if (us_med && !ts.empty()) { std::sort(ts.begin(), ts.end()); *us_med = ts[ts.size() / 2]; }
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return res;
}

int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s <type> <m> <k> <n_tokens> <bcast_b 0|1> <fused 0|1> [time_reps=20]\n", argv[0]); return 2; }
    const ggml_type t = type_by_name(argv[1]);
    const int64_t m = atoll(argv[2]), k = atoll(argv[3]), n = atoll(argv[4]);
    const bool bcast = atoi(argv[5]) != 0, fused = atoi(argv[6]) != 0;
    const int time_reps = argc > 7 ? atoi(argv[7]) : 20;
    ggml_backend_load_all();
    ggml_backend_t gpu = ggml_backend_init_by_name("Vulkan0", nullptr);
    ggml_backend_t cpu = ggml_backend_init_by_name("CPU", nullptr);
    if (!gpu || !cpu) { fprintf(stderr, "no backend\n"); return 4; }
    if (N_USED < 1 || N_USED > N_MATS) { fprintf(stderr, "need 1 <= MMID_USED <= MMID_EXPERTS\n"); return 2; }
    const int hw = (int) std::thread::hardware_concurrency();
    ggml_backend_cpu_set_n_threads(cpu, env_int("MMID_CPU_THREADS", hw > 0 ? hw : 4));

    inputs in;
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    const size_t na = (size_t) k * m * N_MATS;
    std::vector<float> a(na);
    for (auto & x : a) x = nd(rng) * 0.05f;
    in.a.resize(ggml_row_size(t, k) * m * N_MATS);
    ggml_quantize_chunk(t, a.data(), in.a.data(), 0, m * N_MATS, k, nullptr);
    in.a_f32.resize(na);
    ggml_get_type_traits(t)->to_float(in.a.data(), in.a_f32.data(), (int64_t) na);
    in.b.resize((size_t) k * (bcast ? 1 : N_USED) * n);
    for (auto & x : in.b) x = nd(rng);
    in.w.resize((size_t) N_USED * n);
    for (auto & x : in.w) x = std::uniform_real_distribution<float>(0.0f, 0.3f)(rng);
    in.ids.resize((size_t) N_USED * n);
    for (int64_t i = 0; i < n; ++i) {   // distinct experts per token, uniform over the stack
        std::vector<int> pick;
        while ((int) pick.size() < N_USED) {
            int e = std::uniform_int_distribution<int>(0, N_MATS - 1)(rng);
            bool dup = false; for (int p : pick) dup |= p == e;
            if (!dup) pick.push_back(e);
        }
        for (int j = 0; j < N_USED; ++j) in.ids[i * N_USED + j] = pick[j];
    }

    std::vector<std::vector<uint8_t>> raw;
    double us = 0;
    std::vector<float> g = run(gpu, t, in, m, k, n, bcast, fused, 2, &raw, time_reps, &us);
    const bool same = raw.size() == 2 && raw[0] == raw[1];
    std::vector<float> ref = run(cpu, GGML_TYPE_F32, in, m, k, n, bcast, fused, 1, nullptr);
    double num = 0, den = 0, maxabs = 0;
    for (size_t i = 0; i < g.size(); ++i) {
        const double d = (double) g[i] - ref[i];
        num += d * d; den += (double) ref[i] * ref[i]; maxabs = std::max(maxabs, std::fabs(d));
    }
    printf("MMID_DET type=%s m=%lld k=%lld n=%lld bcast=%d fused=%d repeat_bit_identical=%s nmse_vs_f32=%.3e max_abs=%.3e us_graph_med=%.1f\n",
            argv[1], (long long) m, (long long) k, (long long) n, (int) bcast, (int) fused, same ? "yes" : "NO", num / den, maxabs, us);
    ggml_backend_free(gpu);
    ggml_backend_free(cpu);
    return 0;
}

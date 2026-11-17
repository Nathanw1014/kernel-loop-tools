// dram-mm <type_a> <type_b> <m> <n> <k> <secs> [reps_per_graph=20]: one ggml MUL_MAT (A type_a [k x m], B type_b [k x n] -> f32 [m x n])
// on Vulkan0, repeated back to back for <secs> seconds so slow firmware-filtered DRAM counters (mem/dram-sample.py) reach steady
// state. Same graph form as test-backend-ops perf (the node added reps times), so the per-call time is comparable with its perf rows.
// Prints us/call and the minimum bytes per call (A + B read once, C written once): known-traffic streams for calibrating the DRAM
// sampler (mem/DRAM-MONITOR.md) and per-arm bytes per call. Backend env knobs (GGML_VK_*) select the arm.
// Build: build-dram-mm.sh <ggml source tree> <build dir>.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static ggml_type parse_type(const char * s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && strcmp(n, s) == 0) return (ggml_type) t;
    }
    fprintf(stderr, "unknown type %s\n", s); exit(2);
}

int main(int argc, char ** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s <type_a> <type_b> <m> <n> <k> <secs> [reps_per_graph=20]\n", argv[0]); return 2; }
    const ggml_type ta = parse_type(argv[1]), tb = parse_type(argv[2]);
    const int64_t m = atoll(argv[3]), n = atoll(argv[4]), k = atoll(argv[5]);
    const double secs = atof(argv[6]); const int reps = argc > 7 ? atoi(argv[7]) : 20;
    ggml_backend_load_all();
    ggml_backend_t be = ggml_backend_init_by_name("Vulkan0", nullptr);
    if (!be) { fprintf(stderr, "no Vulkan0\n"); return 1; }
    ggml_init_params ip = { ggml_tensor_overhead() * 8 + ggml_graph_overhead_custom(reps + 8, false), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * A = ggml_new_tensor_2d(ctx, ta, k, m);
    ggml_tensor * B = ggml_new_tensor_2d(ctx, tb, k, n);
    ggml_tensor * C = ggml_mul_mat(ctx, A, B);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!buf) { fprintf(stderr, "alloc failed\n"); return 1; }
    // A: random quant bytes with every f16 block scale forced to 1/64 (q8_0 / q4_0-style layouts put the f16 d first in the block);
    // B: f32 / f16 values in [-1, 1]. Content does not change the memory traffic; finite values keep the run representative.
    std::mt19937 rng(42);
    {
        std::vector<uint8_t> h(ggml_nbytes(A)); for (auto & x : h) x = rng() & 0xff;
        const size_t bs = ggml_type_size(ta); const int64_t bk = ggml_blck_size(ta);
        if (bk > 1 && (ta == GGML_TYPE_Q8_0 || ta == GGML_TYPE_Q4_0 || ta == GGML_TYPE_Q5_0))
            for (size_t o = 0; o + bs <= h.size(); o += bs) { h[o] = 0x00; h[o + 1] = 0x24; }
        ggml_backend_tensor_set(A, h.data(), 0, h.size());
    }
    {
        std::uniform_real_distribution<float> u(-1.f, 1.f);
        std::vector<float> f(ggml_nelements(B)); for (auto & x : f) x = u(rng);
        if (tb == GGML_TYPE_F32) ggml_backend_tensor_set(B, f.data(), 0, ggml_nbytes(B));
        else if (tb == GGML_TYPE_F16) { std::vector<ggml_fp16_t> g(f.size()); ggml_fp32_to_fp16_row(f.data(), g.data(), f.size()); ggml_backend_tensor_set(B, g.data(), 0, ggml_nbytes(B)); }
        else { fprintf(stderr, "type_b must be f32 or f16\n"); return 2; }
    }
    ggml_cgraph * gf = ggml_new_graph_custom(ctx, reps + 8, false);
    ggml_build_forward_expand(gf, C);
    for (int i = 1; i < reps; ++i) ggml_graph_add_node(gf, C);
    if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }   // warm-up / compile
    const double a_b = ggml_nbytes(A), b_b = ggml_nbytes(B), c_b = ggml_nbytes(C);
    printf("dram-mm %s x %s m=%lld n=%lld k=%lld reps/graph=%d: A %.2f MB B %.2f MB C %.2f MB, min bytes/call %.2f MB\n",
           argv[1], argv[2], (long long) m, (long long) n, (long long) k, reps, a_b / 1e6, b_b / 1e6, c_b / 1e6, (a_b + b_b + c_b) / 1e6);
    fflush(stdout);
    const int64_t t0 = ggml_time_us(); int64_t calls = 0, tl = t0;
    while (ggml_time_us() - t0 < (int64_t) (secs * 1e6)) {
        if (ggml_backend_graph_compute(be, gf) != GGML_STATUS_SUCCESS) { fprintf(stderr, "compute failed\n"); return 1; }
        calls += reps;
        const int64_t t = ggml_time_us();
        if (t - tl >= 1000000) { printf("t %.1f s calls %lld us/call %.1f\n", (t - t0) / 1e6, (long long) calls, (t - t0) / (double) calls); fflush(stdout); tl = t; }
    }
    const double us = (ggml_time_us() - t0) / (double) calls;
    printf("RESULT calls %lld us/call %.2f min-bytes GB/s %.1f (A-only %.1f) TFLOP/s %.2f\n", (long long) calls, us,
           (a_b + b_b + c_b) / us / 1e3, a_b / us / 1e3, 2.0 * m * n * k / us / 1e6);
    ggml_backend_buffer_free(buf); ggml_free(ctx); ggml_backend_free(be);
    return 0;
}

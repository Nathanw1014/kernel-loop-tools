// fa_masked_leak: does GGML_OP_FLASH_ATTN_EXT on a backend read the K/V content of MASKED-OUT cells?
// Runs the same FA op twice; the only change between the runs is the K/V data in cells the mask hides (-inf). A correct
// kernel gives bit-identical outputs (masked cells contribute exp(-inf) = 0 whatever they hold). Four kinds of hidden
// change localise a leak: K+V random, K only, V magnitude, V sign flip. Positive control per case: change one VISIBLE
// cell instead; it must differ, else the case proves nothing ("control broken").
// Why: a kernel that reads masked cells and relies on a zero weight is exact on sane data and wrong as soon as a reused
// KV cell holds inf / NaN (0 * NaN = NaN) or the arithmetic is not exactly zero. test-backend-ops compares against the CPU
// with sane random data in every cell, so it passes such a kernel; this probe changes only what the mask hides.
//
// usage: fa_masked_leak <backend-name-substring> [hsk n_head n_head_kv n_q n_kv n_visible]...
//   backend: substring of a ggml backend device name (ggml_backend_dev_name, e.g. "Vulkan0", "CPU")
//   cases: groups of six ints; none = a built-in set (head sizes 128/256/512, GQA ratios 2..8, n_q 1..20, n_kv 256..3072)
//   FA_PREC_DEFAULT=1: leave the op at default precision (default: GGML_PREC_F32)
// Output: one block per case and a RESULT line; exit 1 if any case leaks under "K+V random", 3 if a positive control
// came out identical (the run proves nothing), 2 on setup failure, else 0.
// Run CPU as the reference too: `fa_masked_leak CPU` must be bit-identical everywhere.
// build: lib/build-ggml-tools.sh <llama.cpp source> <build dir>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static ggml_backend_t pick(const char * want) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strstr(ggml_backend_dev_name(d), want)) return ggml_backend_dev_init(d, nullptr);
    }
    return nullptr;
}

struct Case { int hsk, nh, nhkv, nq, nkv, nvis; };

// returns output bytes for given K/V host buffers
static std::vector<float> run(ggml_backend_t be, const Case & c, const std::vector<ggml_fp16_t> & q,
                              const std::vector<ggml_fp16_t> & k, const std::vector<ggml_fp16_t> & v,
                              const std::vector<ggml_fp16_t> & m) {
    ggml_init_params ip = { 16 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    // ggml FA layout: q [hsk, nq, nh, 1], k [hsk, nkv, nhkv, 1], v [hsv, nkv, nhkv, 1], mask [nkv, nq_pad, 1, 1]
    ggml_tensor * tq = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.hsk, c.nq, c.nh, 1);
    ggml_tensor * tk = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.hsk, c.nkv, c.nhkv, 1);
    ggml_tensor * tv = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.hsk, c.nkv, c.nhkv, 1);
    ggml_tensor * tm = ggml_new_tensor_4d(ctx, GGML_TYPE_F16, c.nkv, c.nq, 1, 1);
    ggml_tensor * out = ggml_flash_attn_ext(ctx, tq, tk, tv, tm, 1.0f / sqrtf((float) c.hsk), 0.0f, 0.0f);
    if (!getenv("FA_PREC_DEFAULT")) ggml_flash_attn_ext_set_prec(out, GGML_PREC_F32);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    std::vector<float> qf(q.size());
    for (size_t i = 0; i < q.size(); i++) qf[i] = ggml_fp16_to_fp32(q[i]);
    ggml_backend_tensor_set(tq, qf.data(), 0, ggml_nbytes(tq));
    ggml_backend_tensor_set(tk, k.data(), 0, ggml_nbytes(tk));
    ggml_backend_tensor_set(tv, v.data(), 0, ggml_nbytes(tv));
    ggml_backend_tensor_set(tm, m.data(), 0, ggml_nbytes(tm));
    ggml_backend_graph_compute(be, gf);
    std::vector<float> o(ggml_nelements(out));
    ggml_backend_tensor_get(out, o.data(), 0, ggml_nbytes(out));
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    return o;
}

int main(int argc, char ** argv) {
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        puts("usage: fa_masked_leak <backend-name-substring> [hsk n_head n_head_kv n_q n_kv n_visible]...");
        return 0;
    }
    if (argc > 2 && (argc - 2) % 6 != 0) { fprintf(stderr, "cases are groups of 6 ints (hsk n_head n_head_kv n_q n_kv n_visible)\n"); return 2; }
    ggml_backend_load_all();
    const char * want = argc > 1 ? argv[1] : "Vulkan";
    ggml_backend_t be = pick(want);
    if (!be) { fprintf(stderr, "no backend matching %s\n", want); return 2; }
    std::vector<Case> cases;
    for (int i = 2; i + 5 < argc; i += 6)
        cases.push_back({atoi(argv[i]), atoi(argv[i+1]), atoi(argv[i+2]), atoi(argv[i+3]), atoi(argv[i+4]), atoi(argv[i+5])});
    if (cases.empty()) cases = {
        {512, 16, 2, 1, 1024, 875}, {512, 16, 2, 20, 1024, 874}, {256, 16, 8, 1, 3072, 875}, {256, 16, 8, 20, 3072, 874},
        {512, 16, 2, 1, 256, 200},  {128, 32, 8, 1, 1024, 875},  {512, 16, 2, 4, 1024, 875}, {256, 16, 8, 1, 1024, 875} };
    int bad = 0, ctl_bad = 0;
    for (const Case & c : cases) {
        std::mt19937 rng(1234);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        auto fill = [&](std::vector<ggml_fp16_t> & x) { for (auto & e : x) e = ggml_fp32_to_fp16(nd(rng)); };
        std::vector<ggml_fp16_t> q((size_t) c.hsk * c.nq * c.nh), k((size_t) c.hsk * c.nkv * c.nhkv), v(k.size()), m((size_t) c.nkv * c.nq);
        fill(q); fill(k); fill(v);
        // causal-style visibility: query row r sees cells [0, nvis - nq + r]; everything past nvis stays hidden
        for (int r = 0; r < c.nq; r++)
            for (int j = 0; j < c.nkv; j++)
                m[(size_t) r * c.nkv + j] = ggml_fp32_to_fp16(j <= c.nvis - c.nq + r ? 0.0f : -INFINITY);
        auto o1 = run(be, c, q, k, v, m);
        // change ONLY hidden cells (j >= nvis) in every kv head; four kinds, to localise the mechanism
        auto hidden = [&](int mode, std::vector<ggml_fp16_t> & kk, std::vector<ggml_fp16_t> & vv) {
            for (int h = 0; h < c.nhkv; h++)
                for (int j = c.nvis; j < c.nkv; j++)
                    for (int d = 0; d < c.hsk; d++) {
                        size_t idx = ((size_t) h * c.nkv + j) * c.hsk + d;
                        float kv = ggml_fp16_to_fp32(kk[idx]), vvv = ggml_fp16_to_fp32(vv[idx]);
                        if (mode == 0) { kk[idx] = ggml_fp32_to_fp16(nd(rng) * 4.0f); vv[idx] = ggml_fp32_to_fp16(nd(rng) * 4.0f); }
                        if (mode == 1) { kk[idx] = ggml_fp32_to_fp16(-kv * 3.0f); }                 // K only
                        if (mode == 2) { vv[idx] = ggml_fp32_to_fp16(vvv * 3.0f); }                 // V magnitude, same sign
                        if (mode == 3) { vv[idx] = ggml_fp32_to_fp16(-vvv); }                       // V sign flip only
                    }
        };
        const char * names[4] = {"K+V random", "K only", "V magnitude", "V sign flip"};
        int leaks[4]; size_t nd4[4]; double md4[4];
        for (int mode = 0; mode < 4; mode++) {
            auto k2 = k, v2 = v; hidden(mode, k2, v2);
            auto o2 = run(be, c, q, k2, v2, m);
            nd4[mode] = 0; md4[mode] = 0;
            for (size_t i = 0; i < o1.size(); i++) if (memcmp(&o1[i], &o2[i], 4)) { nd4[mode]++; md4[mode] = std::max(md4[mode], (double) fabsf(o1[i] - o2[i])); }
            leaks[mode] = nd4[mode] != 0;
        }
        // positive control: change one VISIBLE cell (must differ)
        auto k3 = k; k3[(size_t) (c.nvis / 2) * c.hsk] = ggml_fp32_to_fp16(5.0f);
        auto o3 = run(be, c, q, k3, v, m);
        bool ctl = memcmp(o1.data(), o3.data(), o1.size() * 4) != 0;
        ctl_bad += ctl ? 0 : 1;
        printf("hsk %3d nh %2d nhkv %d nq %3d nkv %5d nvis %5d | control %s\n", c.hsk, c.nh, c.nhkv, c.nq, c.nkv, c.nvis,
               ctl ? "differs (ok)" : "IDENTICAL (control broken)");
        for (int mode = 0; mode < 4; mode++)
            printf("    hidden %-12s -> %s (%zu/%zu elems, max %.3g)\n", names[mode], leaks[mode] ? "LEAK" : "bit-identical", nd4[mode], o1.size(), md4[mode]);
        bad += leaks[0] ? 1 : 0;
    }
    ggml_backend_free(be);
    printf("RESULT %s: %d case(s) leak, %d broken control(s)\n", want, bad, ctl_bad);
    return bad ? 1 : ctl_bad ? 3 : 0;
}

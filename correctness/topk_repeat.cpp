// topk_repeat: is GGML_OP_TOP_K on a backend repeatable run to run? Same input every run, one process, one graph.
// Scores are expanded per 4-cell block (every value appears 4 times in a row), so the selection boundary falls inside a
// block of 4 tied cells: an implementation whose tie-break depends on scheduling (atomics, workgroup order) picks a
// different subset run to run. Reports, per run vs run 0: rows whose index ORDER differs and rows whose selected SET
// differs (a consumer that only uses the set, e.g. a sparse-attention index, cares about the second).
// Positive control (always run, last): one row's input is changed so its top-k set must change; the comparison has to see
// it, else the run proves nothing.
// Why: a single test-backend-ops pass compares one run against the CPU with few ties; non-repeatable selection shows up
// only as request-to-request drift in a served model.
//
// usage: topk_repeat <backend-name-substring> [ncols k nrows runs]     defaults 8192 2051 64 20
//   k past a backend's small-k path exercises its large-k (e.g. radix-select) path; try several k.
// Exit: 0 repeatable and control detected, 1 NOT REPEATABLE, 3 control not detected, 2 setup failure / unsupported op.
// build: lib/build-ggml-tools.sh <llama.cpp source> <build dir>
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

static ggml_backend_t pick(const char * want) {
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (strstr(ggml_backend_dev_name(d), want)) return ggml_backend_dev_init(d, nullptr);
    }
    return nullptr;
}

int main(int argc, char ** argv) {
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) {
        puts("usage: topk_repeat <backend-name-substring> [ncols k nrows runs]   (defaults 8192 2051 64 20)");
        return 0;
    }
    ggml_backend_load_all();
    const char * want = argc > 1 ? argv[1] : "Vulkan";
    const int ncols = argc > 2 ? atoi(argv[2]) : 8192;
    const int k     = argc > 3 ? atoi(argv[3]) : 2051;
    const int nrows = argc > 4 ? atoi(argv[4]) : 64;
    const int runs  = argc > 5 ? atoi(argv[5]) : 20;
    ggml_backend_t be = pick(want);
    if (!be) { fprintf(stderr, "no backend matching %s\n", want); return 2; }
    if (ncols < 8 || k < 1 || k >= ncols || nrows < 1 || runs < 2) { fprintf(stderr, "need ncols >= 8, 1 <= k < ncols, nrows >= 1, runs >= 2\n"); return 2; }

    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) ncols * nrows);
    for (int r = 0; r < nrows; r++)
        for (int b = 0; b < ncols / 4; b++) {
            const float s = nd(rng);
            for (int j = 0; j < 4; j++) x[(size_t) r * ncols + b * 4 + j] = s;
        }

    ggml_init_params ip = { 8 * ggml_tensor_overhead() + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);
    ggml_tensor * a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, ncols, nrows);
    ggml_tensor * out = ggml_top_k(ctx, a, k);
    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, be);
    if (!ggml_backend_supports_op(be, out)) { fprintf(stderr, "backend does not support TOP_K at k=%d\n", k); return 2; }

    std::vector<std::vector<int32_t>> res(runs, std::vector<int32_t>((size_t) k * nrows));
    for (int i = 0; i < runs; i++) {
        ggml_backend_tensor_set(a, x.data(), 0, ggml_nbytes(a));
        ggml_backend_graph_compute(be, gf);
        // out may be a strided view: read row by row
        for (int r = 0; r < nrows; r++)
            ggml_backend_tensor_get(out, res[i].data() + (size_t) r * k, (size_t) r * out->nb[1], (size_t) k * sizeof(int32_t));
    }

    auto set_differs = [&](const int32_t * p0, const int32_t * pi) {
        std::vector<int32_t> s0(p0, p0 + k), si(pi, pi + k);
        std::sort(s0.begin(), s0.end()); std::sort(si.begin(), si.end());
        return s0 != si;
    };
    int runs_order = 0, runs_set = 0;
    for (int i = 1; i < runs; i++) {
        int rows_order = 0, rows_set = 0;
        for (int r = 0; r < nrows; r++) {
            const int32_t * p0 = res[0].data() + (size_t) r * k;
            const int32_t * pi = res[i].data() + (size_t) r * k;
            if (memcmp(p0, pi, k * sizeof(int32_t)) != 0) rows_order++;
            if (set_differs(p0, pi)) rows_set++;
        }
        if (rows_order) runs_order++;
        if (rows_set) runs_set++;
        printf("run %2d vs 0: rows with different order %d/%d, different set %d/%d\n", i, rows_order, nrows, rows_set, nrows);
    }
    // positive control: lift the lowest selected cell of row 0 out of the set (its whole 4-block to the minimum) and the
    // highest unselected cell's block above the maximum; the selected set of row 0 must change.
    std::vector<float> xc = x;
    {
        float mx = x[0], mn = x[0];
        for (int c = 0; c < ncols; c++) { mx = std::max(mx, x[c]); mn = std::min(mn, x[c]); }
        std::vector<char> in(ncols, 0);
        for (int j = 0; j < k; j++) in[res[0][j]] = 1;
        int lo = -1, hi = -1;
        for (int c = 0; c < ncols; c++) {
            if (in[c] && (lo < 0 || x[c] < x[lo])) lo = c;
            if (!in[c] && (hi < 0 || x[c] > x[hi])) hi = c;
        }
        for (int j = 0; j < 4; j++) { xc[(lo / 4) * 4 + j] = mn - 1.0f; xc[(hi / 4) * 4 + j] = mx + 1.0f; }
    }
    std::vector<int32_t> rc((size_t) k);
    ggml_backend_tensor_set(a, xc.data(), 0, ggml_nbytes(a));
    ggml_backend_graph_compute(be, gf);
    ggml_backend_tensor_get(out, rc.data(), 0, (size_t) k * sizeof(int32_t));
    const bool ctl = set_differs(res[0].data(), rc.data());
    printf("control (row 0 input changed): selected set %s\n", ctl ? "differs (ok)" : "IDENTICAL (control broken)");
    printf("SUMMARY ncols %d k %d nrows %d runs %d: runs differing in order %d/%d, in set %d/%d -> %s\n",
           ncols, k, nrows, runs, runs_order, runs - 1, runs_set, runs - 1, (runs_order || runs_set) ? "NOT REPEATABLE" : "repeatable");
    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
    ggml_backend_free(be);
    return (runs_order || runs_set) ? 1 : ctl ? 0 : 3;
}

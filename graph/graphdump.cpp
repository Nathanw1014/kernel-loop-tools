// graphdump: dump every node of ONE prefill ubatch graph, after filling <depth> tokens of context, with type, shape, strides
// (nb), view source / offset and buffer usage (W weights, C compute, A other: KV cache, state, inputs) for the node and each
// of its srcs. Uses only the public eval callback: it answers ask=true calls and returns false, so the scheduler still runs
// each split in one call and the graph is the one production runs. Feed the TSV to graph/camp.py (stride / channel-camping
// census) or read it directly (what does node X really consume, and at what stride?).
//
// usage: graphdump <model.gguf> <depth> <ubatch> <out.tsv>
//   all layers on the GPU (n_gpu_layers 999), flash attention on, n_ctx = depth + ubatch + 256, random tokens (fixed seed).
//   Only the graph of the ubatch that starts at position <depth> is written; earlier ubatches only fill the cache.
// Output: "# depth D ubatch U pos P" then one line per node: idx, then name|op|type|ne0..3|nb0..3|view_src|view_offs|usage
// for the node, then "S<j>" + the same fields for each src j.
// build: graph/build-tools.sh <llama.cpp build dir>
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>
#include <string>

static FILE * g_out = nullptr;
static bool g_on = false, g_done = false;
static int g_idx = 0;

static const char * usage(const ggml_tensor * t) {
    const ggml_tensor * b = t->view_src ? t->view_src : t;
    if (!b->buffer) return "none";
    switch (ggml_backend_buffer_get_usage(b->buffer)) {
        case GGML_BACKEND_BUFFER_USAGE_WEIGHTS: return "W";
        case GGML_BACKEND_BUFFER_USAGE_COMPUTE: return "C";
        default: return "A";
    }
}

static void tdesc(const ggml_tensor * t) {
    fprintf(g_out, "\t%s|%s|%s|%lld,%lld,%lld,%lld|%zu,%zu,%zu,%zu|%s|%zu|%s", t->name, ggml_op_desc(t), ggml_type_name(t->type),
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
            t->nb[0], t->nb[1], t->nb[2], t->nb[3], t->view_src ? t->view_src->name : "-", t->view_offs, usage(t));
}

static bool cb(ggml_tensor * t, bool ask, void *) {
    if (!ask || !g_on) return false;
    fprintf(g_out, "%d", g_idx++);
    tdesc(t);
    for (int j = 0; j < GGML_MAX_SRC; j++) {
        if (t->src[j]) { fprintf(g_out, "\tS%d", j); tdesc(t->src[j]); }
    }
    fputc('\n', g_out);
    return false;
}

int main(int argc, char ** argv) {
    const char * U = "usage: graphdump <model.gguf> <depth> <ubatch> <out.tsv>";
    if (argc > 1 && (!strcmp(argv[1], "-h") || !strcmp(argv[1], "--help"))) { puts(U); return 0; }
    if (argc < 5) { fprintf(stderr, "%s\n", U); return 1; }
    const int depth = atoi(argv[2]), ub = atoi(argv[3]);
    if (depth < 0 || ub <= 0) { fprintf(stderr, "%s\n", U); return 1; }
    g_out = fopen(argv[4], "w");
    if (!g_out) { perror(argv[4]); return 1; }
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 999;
    mp.load_mode = LLAMA_LOAD_MODE_NONE;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 2;
    auto cp = llama_context_default_params();
    cp.n_ctx = depth + ub + 256; cp.n_batch = ub; cp.n_ubatch = ub; cp.n_seq_max = 1;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.cb_eval = cb; cp.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 3;
    const int nv = llama_vocab_n_tokens(llama_model_get_vocab(model));
    std::mt19937 rng(1234);
    std::vector<llama_token> tok(ub);
    int pos = 0;
    while (pos < depth + ub) {
        for (auto & x : tok) x = (llama_token) (rng() % (nv - 1000)) + 100;
        g_on = (pos >= depth);
        if (g_on) fprintf(g_out, "# depth %d ubatch %d pos %d\n", depth, ub, pos);
        llama_batch b = llama_batch_get_one(tok.data(), ub);
        if (llama_decode(ctx, b) != 0) { fprintf(stderr, "decode failed at %d\n", pos); return 4; }
        pos += ub;
    }
    fclose(g_out);
    fprintf(stderr, "dumped %d nodes\n", g_idx);
    llama_free(ctx); llama_model_free(model);
    return 0;
}

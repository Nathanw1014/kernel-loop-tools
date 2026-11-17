// topkdump: MoE routing dump. Runs one prompt through the model (one llama_decode of the whole prompt, split into ubatches
// by -b / -ub as usual) and writes every ffn_moe_topk-* tensor (the selected expert ids, i32 [n_expert_used, n_tokens]) to
// TOPKDUMP_OUT in graph order, one record per MoE layer per ubatch. Upstream llama.cpp names this tensor ffn_moe_topk-<il>
// in build_moe_ffn; a model that names it differently dumps nothing (the summary line then says 0 records).
// Analyse with graph/moe-route.py: rows per expert per ubatch (tile fill of the MUL_MAT_ID path), expert-set agreement
// between two builds or settings, and expert overlap within small batches.
//
// usage: TOPKDUMP_OUT=route.bin topkdump -m model.gguf -f prompt.txt -c 4096 -b 2048 -ub 512 [-ngl 99 ...]   (common flags)
//   the prompt is truncated to n_ctx - 1 tokens; no warm-up run.
// Record: 64-byte NUL-padded name, int64 ne0 (k), int64 ne1 (tokens), then ne0*ne1 int32, token-major.
// build: lib/build-ggml-tools.sh <llama.cpp source> <build dir>   (links libllama-common)
#include "arg.h"
#include "common.h"
#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <cstdio>
#include <cstring>
#include <vector>

struct state { FILE * f = nullptr; long n = 0; };

static bool cb(struct ggml_tensor * t, bool ask, void * ud) {
    const bool want = strncmp(t->name, "ffn_moe_topk", 12) == 0;
    if (ask) return want;
    if (!want) return true;
    state * s = (state *) ud;
    if (t->type != GGML_TYPE_I32) { fprintf(stderr, "topkdump: %s type %s\n", t->name, ggml_type_name(t->type)); return true; }
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1] * t->ne[2] * t->ne[3];
    std::vector<int32_t> buf(ne0 * ne1);
    // rows may be strided (a view of the argsort): copy row by row
    for (int64_t r = 0; r < ne1; r++) {
        const int64_t i1 = r % t->ne[1], i2 = (r / t->ne[1]) % t->ne[2], i3 = r / (t->ne[1] * t->ne[2]);
        const size_t off = i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        if (t->nb[0] == sizeof(int32_t)) {
            ggml_backend_tensor_get(t, buf.data() + r * ne0, off, ne0 * sizeof(int32_t));
        } else {
            for (int64_t c = 0; c < ne0; c++) ggml_backend_tensor_get(t, buf.data() + r * ne0 + c, off + c * t->nb[0], sizeof(int32_t));
        }
    }
    char name[64] = {0};
    strncpy(name, t->name, 63);
    fwrite(name, 1, 64, s->f);
    fwrite(&ne0, 8, 1, s->f);
    fwrite(&ne1, 8, 1, s->f);
    fwrite(buf.data(), 4, buf.size(), s->f);
    s->n++;
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_DEBUG)) return 1;
    const char * out = getenv("TOPKDUMP_OUT");
    if (!out) { fprintf(stderr, "set TOPKDUMP_OUT=<file>\n"); return 1; }
    state s;
    s.f = fopen(out, "wb");
    if (!s.f) { perror(out); return 1; }
    params.cb_eval = cb;
    params.cb_eval_user_data = &s;
    params.warmup = false;
    llama_backend_init();
    auto init = common_init_from_params(params);
    llama_context * ctx = init->context();
    if (!ctx) return 1;
    std::vector<llama_token> toks = common_tokenize(ctx, params.prompt, true);
    if ((int) toks.size() > params.n_ctx - 1) toks.resize(params.n_ctx - 1);
    common_batch batch = common_batch_get_one(ctx, toks);
    if (llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch.get())) { fprintf(stderr, "decode failed\n"); return 1; }
    fclose(s.f);
    printf("topkdump: %zu tokens, %ld topk records -> %s\n", toks.size(), s.n, out);
    llama_backend_free();
    return 0;
}

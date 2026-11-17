#!/usr/bin/env python3
# needs: a llama-eval-callback patch that reads EVCB_ASK_ONLY=<comma-separated node names> (answer the scheduler's "ask" only for those nodes), EVCB_ASK_NONE=1 (ask for no node) and EVCB_LOGITS_DUMP=<file> (write the f32 logits of every prompt position, n_tokens x n_vocab, to <file>).
"""split-bisect.py <eval-callback-bin> <model.gguf> <prompt.txt> <node-list.txt> <reference-logits.bin> <n_vocab> <out-dir>

Prefix-isolation bisect over graph node order. Asking the eval callback for a node makes ggml_backend_sched split the graph
there, which re-allocates around it. If the whole-graph output is wrong but every per-node output is right, the smallest prefix of
isolated nodes that restores parity brackets the offending op (or the reallocation around it).

  node-list.txt         node names in graph execution order, one per line (e.g. from an eval-callback run that asks for all)
  reference-logits.bin  f32 all-position logits of a known-good run of the same prompt (same EVCB_LOGITS_DUMP format)
  out-dir               per-step logits and logs (logits-k<K>.bin, run-k<K>.log)
env:
  SPLIT_BISECT_ARGS     llama args after -m / -f (default "-ngl 99 -fa on -c 512 -b 512 -ub 512 -n 1"; -b / -ub must cover
                        the prompt so it is one graph)
  SPLIT_BISECT_PARITY   same-argmax % that counts as parity (default 99.99)
Driver env (VK_ICD_FILENAMES etc.) passes through. Runs the GPU: hold your GPU lock around it."""
import os, shlex, subprocess, sys

if len(sys.argv) != 8 or sys.argv[1] in ('-h', '--help'): print(__doc__); sys.exit(2)
import numpy as np
B, M, P, NODES, REF, V, D = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6]), sys.argv[7]
ARGS = shlex.split(os.environ.get('SPLIT_BISECT_ARGS', '-ngl 99 -fa on -c 512 -b 512 -ub 512 -n 1'))
PARITY = float(os.environ.get('SPLIT_BISECT_PARITY', '99.99'))
nodes = [l.rstrip("\n") for l in open(NODES) if l.strip()]
ref = np.fromfile(REF, dtype=np.float32).reshape(-1, V)
os.makedirs(D, exist_ok=True)

def run(k):
    """isolate the first k nodes (ask=true for them), whole-graph for the rest; return same-argmax %"""
    out = f"{D}/logits-k{k}.bin"
    env = dict(os.environ, EVCB_ASK_ONLY=",".join(nodes[:k]) if k > 0 else "__none__", EVCB_LOGITS_DUMP=out)
    if k == 0:
        env["EVCB_ASK_NONE"] = "1"
    with open(f"{D}/run-k{k}.log", "w") as lf:
        subprocess.run([B, "-m", M, "-f", P] + ARGS, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
    got = np.fromfile(out, dtype=np.float32).reshape(-1, V)
    same = float((got.argmax(1) == ref.argmax(1)).mean() * 100)
    maxd = float(np.abs(got - ref).max())
    print(f"k={k:5d} isolate[{nodes[0] if k else '-'} .. {nodes[k-1] if k else '-'}]  same-argmax={same:.3f}%  max|dlogit|={maxd:.4f}", flush=True)
    return same

lo, hi = 0, len(nodes)          # invariant: run(lo) BAD (whole graph), run(hi) GOOD (all isolated)
s_lo = run(lo); s_hi = run(hi)
print(f"bracket: k=0 -> {s_lo:.3f}%, k={hi} -> {s_hi:.3f}%")
if s_lo >= PARITY or s_hi < PARITY:
    print("no clean bracket (whole-graph already good, or full isolation not good): stop"); sys.exit(1)
while hi - lo > 1:
    mid = (lo + hi) // 2
    if run(mid) >= PARITY:
        hi = mid
    else:
        lo = mid
print(f"FIRST node whose isolation restores parity: index {hi-1} = {nodes[hi-1]}  (prefix through index {lo-1} = {nodes[lo-1] if lo else '-'} still BAD)")

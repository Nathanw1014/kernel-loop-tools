#!/usr/bin/env python3
"""model-census.py <model.gguf> [more.gguf ...] [--json OUT] [--gguf-py DIR]

Dense MUL_MAT census per model: every (M, K, type) weight shape with its multiplicity and its share of the dense
weight elements, from the gguf tensor list (no weights are read), so op benches cover the shapes that carry the time.
GGUF shape is (ne0=K, ne1=M) for a weight, so M = shape[1], K = shape[0]. MUL_MAT_ID tensors (3-D expert stacks) are excluded:
they take the MUL_MAT_ID path, not this one. Norms, biases, the token embedding (a row lookup) and a few small SSM tensors are
skipped, as are f32 tensors.
Needs gguf-py: an installed `gguf` package, or the gguf-py directory of a llama.cpp checkout via --gguf-py DIR or $GGUF_PY.
--json OUT also writes the census as JSON."""
import collections, json, os, sys

args = sys.argv[1:]
def opt(name):
    if name in args:
        i = args.index(name); v = args[i + 1]; del args[i:i + 2]; return v
    return None
out_json = opt('--json'); gguf_py = opt('--gguf-py') or os.environ.get('GGUF_PY')
if not args or args[0] in ('-h', '--help'): print(__doc__); sys.exit(2)
if gguf_py: sys.path.insert(0, gguf_py)
try:
    from gguf import GGUFReader
except ImportError:
    sys.exit('model-census: cannot import gguf; pip install gguf, or pass --gguf-py <llama.cpp>/gguf-py (or set GGUF_PY)')

SKIP = ("norm", "bias", "ssm_a", "ssm_dt", "ssm_conv1d", "token_embd")
out = {}
for path in args:
    r = GGUFReader(path)
    c = collections.Counter()
    for t in r.tensors:
        if any(s in t.name for s in SKIP): continue
        sh = [int(x) for x in t.shape]
        if len(sh) != 2 or sh[0] == 1: continue
        tt = t.tensor_type.name
        if tt == "F32": continue
        c[(sh[1], sh[0], tt)] += 1
    out[os.path.basename(path)] = [{"m": m, "k": k, "type": tt, "n": n} for (m, k, tt), n in
                                   sorted(c.items(), key=lambda kv: -kv[1])]
if out_json:
    with open(out_json, "w") as f: json.dump(out, f, indent=1)
for mod, rows in out.items():
    print("=====", mod)
    tot = sum(r["n"]*r["m"]*r["k"] for r in rows)
    for r in rows:
        w = r["n"]*r["m"]*r["k"]
        print(f"  {r['n']:4d} x m={r['m']:7d} k={r['k']:7d} {r['type']:8s}  {100*w/tot:5.1f}% of dense weight-elements")

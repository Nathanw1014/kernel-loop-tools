#!/usr/bin/env python3
# arms-compare.py <arms-a.json> <arms-b.json> [label]: compare the greedy token streams of two rollback_gate.py runs arm by
# arm (A1/A2/B/L1/L2/C and the three diag runs); prints PASS when every arm is token-identical, else the first difference.
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2])); lab = sys.argv[3] if len(sys.argv) > 3 else "streams"
ok = True; out = []
def cmp(name, x, y):
    global ok
    tx, ty = x["tokens"], y["tokens"]
    n = min(len(tx), len(ty))
    i = next((k for k in range(n) if tx[k] != ty[k]), None)
    if i is None and len(tx) == len(ty):
        out.append(f"{name}:identical({len(tx)})")
    else:
        ok = False
        out.append(f"{name}:DIFF@{i if i is not None else n} ({len(tx)} vs {len(ty)} tokens)")
for k in ("A1", "A2", "B", "L1", "L2", "C"):
    if k in a and k in b: cmp(k, a[k], b[k])
for j, (x, y) in enumerate(zip(a.get("diag", []), b.get("diag", []))):
    cmp(f"diag{j+1}", x, y)
print(f"[{lab}] {'PASS' if ok else 'FAIL'} - " + " ".join(out))
sys.exit(0 if ok else 1)

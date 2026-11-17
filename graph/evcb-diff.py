#!/usr/bin/env python3
"""evcb-diff.py: first divergent graph node between two llama-eval-callback dumps (upstream tool, no patch).

usage: evcb-diff.py <A.log> <B.log> [--limit N] [--window W]
       evcb-diff.py --selftest                     classifier controls, no input files

Make the dumps with the same model, prompt and flags on the two builds (or knob settings) under test:
    llama-eval-callback -m M.gguf -p "<prompt>" -n 1 --no-warmup [-ngl 99 ...] > A.log 2>&1
Each node prints "common_debug_cb_eval: <name> = (<type>) <OP>(...) = {shape}" and, a few lines later, "sum = <value>".
Nodes are aligned by NAME, not by position: node_<N> and bare "(view)" / "(reshaped)" names shift between builds, so only
identifier-like names count and node_* is skipped; same-named nodes can be reordered by the scheduler, so each A node is
matched within a sliding window of the next W (default 64) unmatched B nodes. Prints up to N (default 12) SHAPE / VALUE
differences in A order, then the names only one side has. The first VALUE line is where to look; later ones are usually
downstream of it.
Limits: "sum" is printed with 6 decimals, so a difference below that resolution is invisible here (use a KLD or logits
comparison for small drift); quantised tensors print no sum and are not compared.
Exit: 0 no difference found, 1 differences found, 2 usage / parse error (no named nodes parsed).
Guards against: bisecting a numerical difference by hand across hundreds of nodes, and against positional alignment that
reports the first fused / reordered node instead of the first wrong one.
"""
import re, sys
from collections import defaultdict, deque

NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.\-]*$")
NODE_RE = re.compile(r"common_debug_cb_eval:\s+(\S+)\s*=\s*\((\w+)\)\s+(\w+)\(.*=\s*\{([^}]*)\}")
SUM_RE = re.compile(r"^\s*sum\s*=\s*([-\d.eE+na]+)")


def parse(lines):
    seq, pending = [], None
    for line in lines:
        m = NODE_RE.search(line)
        if m:
            name, typ, op, shape = m.groups()
            pending = (name, op, shape.replace(" ", ""))
            continue
        s = SUM_RE.match(line)
        if s and pending:
            name, op, shape = pending
            if NAME_RE.match(name) and not name.startswith("node_"):
                seq.append((name, op, shape, s.group(1)))
            pending = None
    return seq


def diff(a, b, limit, window):
    out = []
    bwin, bi = deque(), 0
    for idx, (name, op, shape, s) in enumerate(a):
        while bi < len(b) and len(bwin) < window:
            bwin.append(b[bi]); bi += 1
        hit = None
        for j, (bn, bop, bshape, bs) in enumerate(bwin):
            if bn == name:
                hit = (j, bop, bshape, bs)
                if bs == s and bshape == shape:
                    break
        if hit is None:
            continue
        j, bop, bshape, bs = hit
        if bshape != shape:
            out.append(f"[{idx}] SHAPE {name}: A {op}{shape}  B {bop}{bshape}")
        elif bs != s:
            try:
                rel = abs(float(s) - float(bs)) / max(abs(float(bs)), 1e-30)
            except ValueError:
                rel = float("nan")
            out.append(f"[{idx}] VALUE {name} ({op}): A={s} B={bs} rel={rel:.2e}")
        else:
            del bwin[j]; continue
        if len(out) >= limit:
            break
    return out


def selftest():
    def dump(nodes):
        lines = []
        for name, op, sh, sm in nodes:
            lines += [f"common_debug_cb_eval: {name:>24} = (f32) {op:>10}(x{{1}}, }}) = {{{sh}}}", "  [ ... ]", f"  sum = {sm}"]
        return lines
    base = [("attn_norm-0", "RMS_NORM", "64, 4, 1, 1", "1.000000"), ("Qcur-0", "MUL_MAT", "64, 4, 1, 1", "2.000000"),
            ("node_7", "ADD", "64, 4, 1, 1", "3.000000"), ("ffn_out-0", "MUL_MAT", "64, 4, 1, 1", "4.000000")]
    swapped = [base[1], base[0], base[2], base[3]]                           # scheduler reorder only
    wrong = [base[0], ("Qcur-0", "MUL_MAT", "64, 4, 1, 1", "2.500000"), base[2], base[3]]
    renum = [base[0], base[1], ("node_9", "ADD", "64, 4, 1, 1", "9.000000"), base[3]]   # node_N differs: ignored
    cases = [("identical", base, base, 0), ("reordered same values", base, swapped, 0),
             ("one wrong value", base, wrong, 1), ("node_N renumbered", base, renum, 0)]
    bad = 0
    for name, x, y, want in cases:
        got = len(diff(parse(dump(x)), parse(dump(y)), 12, 64))
        ok = (got > 0) == (want > 0)
        bad += not ok
        print(f"  {'ok  ' if ok else 'BAD '} {name}: want {'diff' if want else 'none'}, got {got} difference(s)")
    print(f"SELFTEST: {'PASS' if not bad else 'FAIL'}")
    return 1 if bad else 0


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__); return 0 if a else 2
    if a[0] == "--selftest":
        return selftest()
    limit, window = 12, 64
    for flag in ("--limit", "--window"):
        if flag in a:
            i = a.index(flag); v = int(a[i + 1]); del a[i:i + 2]
            if flag == "--limit": limit = v
            else: window = v
    if len(a) != 2:
        print("usage: evcb-diff.py <A.log> <B.log> [--limit N] [--window W]", file=sys.stderr); return 2
    A = parse(open(a[0], errors="replace")); B = parse(open(a[1], errors="replace"))
    print(f"parsed named nodes: A={len(A)} B={len(B)}")
    if not A or not B:
        print("no named nodes parsed from one side (not a llama-eval-callback dump?)"); return 2
    out = diff(A, B, limit, window)
    for line in out:
        print(line)
    an, bn = defaultdict(int), defaultdict(int)
    for n, *_ in A: an[n] += 1
    for n, *_ in B: bn[n] += 1
    oa = [n for n in an if n not in bn]; ob = [n for n in bn if n not in an]
    print(f"names only in A ({len(oa)}): {oa[:12]}")
    print(f"names only in B ({len(ob)}): {ob[:12]}")
    print(f"RESULT: {len(out)} difference(s) shown" if out else "RESULT: no SHAPE / VALUE difference at sum resolution")
    return 1 if out else 0


if __name__ == "__main__":
    sys.exit(main())

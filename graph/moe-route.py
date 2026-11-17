#!/usr/bin/env python3
"""moe-route.py: MoE routing census from graph/topkdump files (records of ffn_moe_topk, one per MoE layer per ubatch).

usage:
  moe-route.py stats <route.bin> [--tile 16,128] [--experts E]
      per ubatch record: active experts, rows (tokens) per active expert, the share of experts / rows below each tile size,
      tile fill (rows / (tiles * tile)) and a log2 histogram of rows per expert. This is the shape the MUL_MAT_ID kernel
      sees: a tile or path tuned for 128 rows per expert is mostly empty if most experts get 2 rows at your -ub.
  moe-route.py cmp <A.bin> <B.bin>
      per-token expert-set agreement between two dumps of the SAME prompt and ubatch split (two builds, a quant change, a
      kernel knob): same set %, mean overlap %, and every layer whose agreement is below 100%. Routing is a hard top-k, so
      a small numerical change flips experts near ties; a large drop means the router inputs changed, not just rounding.
  moe-route.py overlap <route.bin> [--max-n 8] [--experts E] [--pool OUT]
      for records with 2..max-n tokens (decode / speculative / small batches): unique experts per record U / (N*k), the share
      of expert slots on an expert that 2+ tokens share, and the share of experts with 2+ tokens; the same numbers for
      uniform random routing over E experts (E = largest id + 1 unless --experts). How much a multi-token MUL_MAT_ID could
      reuse each expert's weights beyond chance. --pool OUT writes "N k ids..." lines of those records.
Exit 0 (cmp: 1 when any token's set differs), 2 on bad input.
Pure Python (no numpy).
"""
import collections, random, statistics as st, struct, sys
from array import array


def load(path):
    recs = []
    with open(path, 'rb') as f:
        while True:
            h = f.read(80)
            if not h:
                break
            if len(h) < 80:
                sys.exit(f"moe-route: {path}: truncated record header")
            name = h[:64].split(b'\0')[0].decode(errors='replace')
            ne0, ne1 = struct.unpack('<qq', h[64:])
            a = array('i')
            a.frombytes(f.read(4 * ne0 * ne1))
            if len(a) != ne0 * ne1:
                sys.exit(f"moe-route: {path}: truncated record {name}")
            recs.append((name, ne0, ne1, a))
    if not recs:
        sys.exit(f"moe-route: {path}: no records (did the model name its router output ffn_moe_topk?)")
    return recs


def nonempty(recs):
    # the last layer of a ubatch that needs no output rows routes 0 tokens (llama.cpp gathers only the output rows there)
    keep = [r for r in recs if r[1] * r[2] > 0]
    return keep, len(recs) - len(keep)


def stats(path, tiles, experts):
    recs, empty = nonempty(load(path))
    tiles = sorted(tiles)
    n_calls = active = rows = 0
    below = {t: [0, 0] for t in tiles}    # experts, rows below tile t
    tcount = {t: 0 for t in tiles}
    hist = collections.Counter(); max_rows = 0; emax = 0; ntok = collections.Counter()
    for name, k, n, ids in recs:
        cnt = collections.Counter(ids)
        n_calls += 1; ntok[n] += 1
        emax = max(emax, max(ids) + 1)
        for e, c in cnt.items():
            active += 1; rows += c; max_rows = max(max_rows, c)
            for t in tiles:
                if c < t:
                    below[t][0] += 1; below[t][1] += c
                tcount[t] += (c + t - 1) // t
            hist[c.bit_length() - 1] += 1
    E = experts or emax
    print(f"{path}: {n_calls} records (MoE layer x ubatch), tokens per record {dict(sorted(ntok.items()))}, experts seen {emax}"
          f"{'' if not experts else f' (E={E})'}, k {recs[0][1]}"
          + (f"; {empty} empty records skipped (0 tokens: last layer, no output rows)" if empty else ""))
    print(f"  active experts per record {active / n_calls:.1f} of {E}, rows per active expert {rows / active:.1f} (max {max_rows})")
    for t in tiles:
        print(f"  < {t} rows: {100.0 * below[t][0] / active:.1f}% of active experts, {100.0 * below[t][1] / rows:.1f}% of rows; "
              f"{t}-row tile fill {100.0 * rows / (t * tcount[t]):.1f}%")
    print("  rows per expert (log2 bucket: count): " + " ".join(f"[{1 << b},{2 << b}):{v}" for b, v in sorted(hist.items())))
    return 0


def cmp(pa, pb):
    a, b = load(pa), load(pb)
    if len(a) != len(b):
        print(f"record counts differ: {len(a)} vs {len(b)} (different prompt, ubatch split or model?)"); return 2
    tot = same = 0; ov = 0.0; per = {}
    for (na, k, n, xa), (nb, kb, nnb, xb) in zip(a, b):
        if na != nb or (k, n) != (kb, nnb):
            print(f"record mismatch: {na} {k}x{n} vs {nb} {kb}x{nnb}"); return 2
        d = per.setdefault(na, [0, 0])
        for t in range(n):
            ra = set(xa[t * k:(t + 1) * k]); rb = set(xb[t * k:(t + 1) * k])
            tot += 1; d[0] += 1
            if ra == rb:
                same += 1; d[1] += 1
            ov += len(ra & rb) / k
    for nm, (t, s) in per.items():
        if s != t:
            print(f"  {nm}: {s}/{t} token rows with the same expert set")
    print(f"records {len(a)} token-rows {tot}: same expert set {same}/{tot} = {100.0 * same / tot:.4f}%, "
          f"mean overlap {100.0 * ov / tot:.4f}%")
    return 0 if same == tot else 1


def overlap(path, max_n, experts, pool):
    recs = [(n, k, list(ids)) for _, k, n, ids in nonempty(load(path))[0]]
    emax = max(max(ids) for _, _, ids in recs) + 1
    E = experts or emax
    def m(n, k, ids):
        c = collections.Counter(ids); U = len(c)
        return U / (n * k), sum(v for v in c.values() if v >= 2) / (n * k), sum(1 for v in c.values() if v >= 2) / U
    byn = collections.defaultdict(list)
    for n, k, ids in recs:
        if 2 <= n <= max_n:
            byn[n].append(m(n, k, ids))
    print(f"{path}: {len(recs)} records, E={E}; records with 2..{max_n} tokens: {sum(len(v) for v in byn.values())}")
    if not byn:
        print("  no small-batch records: dump a decode / speculative run, or use a small -ub"); return 0
    rng = random.Random(1); k = recs[0][1]
    print("| N | records | U/(N*k) | slots on shared experts | experts with 2+ tokens | random U/(N*k) | random shared slots |")
    print("|---|---|---|---|---|---|---|")
    for n in sorted(byn):
        v = byn[n]
        rnd = [m(n, k, sum((rng.sample(range(E), k) for _ in range(n)), [])) for _ in range(2000)]
        print(f"| {n} | {len(v)} | {st.mean(x[0] for x in v):.3f} | {st.mean(x[1] for x in v):.3f} | {st.mean(x[2] for x in v):.3f} | "
              f"{st.mean(x[0] for x in rnd):.3f} | {st.mean(x[1] for x in rnd):.3f} |")
    if pool:
        with open(pool, "w") as f:
            for n, k_, ids in recs:
                if 2 <= n <= max_n:
                    f.write(f"{n} {k_} {' '.join(map(str, ids))}\n")
    return 0


def main():
    a = sys.argv[1:]
    if not a or a[0] in ('-h', '--help'):
        print(__doc__); return 0 if a else 2
    cmd, rest = a[0], a[1:]
    def opt(name, default, conv=int):
        if name in rest:
            i = rest.index(name); v = conv(rest[i + 1]); del rest[i:i + 2]; return v
        return default
    if cmd == 'stats' and rest:
        tiles = opt('--tile', [16, 128], lambda s: [int(x) for x in s.split(',')])
        return stats(rest[0], tiles, opt('--experts', 0))
    if cmd == 'cmp' and len(rest) == 2:
        return cmp(*rest)
    if cmd == 'overlap' and rest:
        max_n = opt('--max-n', 8); E = opt('--experts', 0); pool = opt('--pool', None, str)
        return overlap(rest[0], max_n, E, pool)
    print(__doc__, file=sys.stderr); return 2


if __name__ == '__main__':
    sys.exit(main())

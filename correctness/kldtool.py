#!/usr/bin/env python3
"""kldtool.py: read llama-perplexity KL-divergence files and logs (upstream format, no patch). Needs numpy for peek / diff.

usage:
  kldtool.py peek <base.kld> [--rows N]
      header (n_ctx, n_vocab, n_chunk, rows per chunk) and, for N rows spread over the file, the stored log-prob window:
      scale, min, entries inside the window (within the stored range of the max), max log-prob.
  kldtool.py diff <A.kld> <B.kld> [more.kld ...] --ub N [--batch N] [--show 40]
      row-by-row comparison of files written with the SAME text, -c and model (e.g. two builds' --kl-divergence-base dumps,
      or two runs of one build): how many rows differ, at which positions, the position within -ub and the (batch, ubatch)
      each lands in, header-only differences, the largest quantised delta. Use it to see WHERE two runs differ: a band at
      ubatch tails, one ubatch, the first chunk only, everything.
  kldtool.py log <A.log> [<B.log>]
      parse the "KL divergence statistics" block of llama-perplexity --kl-divergence output (labels are padded:
      "Mean    KLD:"). With two logs: which mean KLD is lower, the ratio, and whether the difference is outside
      2 x (sum of the two standard errors).

File layout (tools/perplexity/perplexity.cpp): "_logits_", n_ctx, n_vocab, n_chunk (int32), the tokens (int32 x n_chunk*n_ctx),
then per chunk n_ctx - 1 - n_ctx/2 rows of nv = 2*((n_vocab+1)/2) + 4 uint16: 2 floats (scale, min log-prob) then n_vocab
quantised log-probs. Row i of a chunk holds the logits at position n_ctx/2 + i.
Exit: diff 1 if any row differs, else 0; log 2 if a log has no KLD block; peek 0.
Guards against: a single mean KLD hiding WHERE the difference is (one bad ubatch shape averages out), and against a log
parser that misses the padded labels and reports a good run as "incomplete".
"""
import os, re, sys


def np_or_die():
    try:
        import numpy as np
        return np
    except ImportError:
        sys.exit("kldtool: peek / diff need numpy (pip install numpy)")


def header(fh, np):
    if fh.read(8) != b'_logits_':
        sys.exit(f"kldtool: {fh.name} is not a llama-perplexity logits file (no _logits_ magic)")
    n_ctx, n_vocab, n_chunk = (int(x) for x in np.frombuffer(fh.read(12), dtype=np.int32))
    nv = 2 * ((n_vocab + 1) // 2) + 4
    off = 8 + 12 + 4 * n_chunk * n_ctx
    return n_ctx, n_vocab, n_chunk, nv, off


def peek(path, nrows):
    np = np_or_die()
    with open(path, 'rb') as fh:
        n_ctx, n_vocab, n_chunk, nv, off = header(fh, np)
        per = n_ctx - 1 - n_ctx // 2
        total = (os.path.getsize(path) - off) // (2 * nv)
        print(f"n_ctx {n_ctx} n_vocab {n_vocab} n_chunk {n_chunk} nv {nv} rows/chunk {per} rows in file {total} "
              f"(expected {per * n_chunk}) data_off {off}")
        for r in np.linspace(0, max(total - 1, 0), nrows).astype(int):
            fh.seek(off + int(r) * nv * 2)
            row = np.frombuffer(fh.read(nv * 2), dtype=np.uint16)
            sc, mn = np.frombuffer(row[:4].tobytes(), dtype=np.float32)
            q = row[4:4 + n_vocab]
            print(f"row {r} (chunk {r // per}, pos {n_ctx // 2 + r % per}): scale {sc:.3g} min_lp {mn:.2f} "
                  f"inside {np.count_nonzero(q)} max_lp {mn + sc * q.max():.3f}")
    return 0


def load(path, np):
    with open(path, 'rb') as fh:
        n_ctx, n_vocab, n_chunk, nv, off = header(fh, np)
        fh.seek(off)
        a = np.frombuffer(fh.read(), dtype=np.uint16)
    if a.size % nv:
        sys.exit(f"kldtool: {path}: data is not a whole number of rows (truncated file?)")
    return (n_ctx, n_vocab, n_chunk), a.reshape(-1, nv)


def diff(files, ub, batch, show):
    np = np_or_die()
    (meta0, base) = load(files[0], np)
    n_ctx = meta0[0]; per = n_ctx - 1 - n_ctx // 2
    any_diff = False
    for f in files[1:]:
        meta, a = load(f, np)
        name = f"{os.path.basename(files[0])} vs {os.path.basename(f)}"
        if meta != meta0 or a.shape != base.shape:
            print(f"{name}: NOT COMPARABLE (n_ctx, n_vocab, n_chunk {meta} vs {meta0}; rows {a.shape[0]} vs {base.shape[0]})")
            any_diff = True; continue
        d = np.any(a != base, axis=1)
        rows = np.nonzero(d)[0]
        print(f"{name}: {len(rows)} of {len(base)} rows differ")
        if not len(rows):
            continue
        any_diff = True
        chunk = rows // per; pos = n_ctx // 2 + rows % per
        hdr = np.any(a[:, :4] != base[:, :4], axis=1)
        body_diff = np.any(a[:, 4:] != base[:, 4:], axis=1)
        body = np.abs(a[rows, 4:].astype(np.int32) - base[rows, 4:].astype(np.int32))
        print(f"  chunks with a difference: {sorted(set(chunk.tolist()))[:show]}")
        print(f"  first/last pos {pos[0]} {pos[-1]} | pos % ub (first {show}): {(pos % ub)[:show].tolist()}")
        print(f"  pos (first {show}): {pos[:show].tolist()}")
        print(f"  header-only rows: {int(np.sum(hdr & ~body_diff))} | max uint16 delta {int(body.max())} | "
              f"mean elements changed per differing row {float(np.mean(np.sum(a[rows, 4:] != base[rows, 4:], axis=1))):.1f}")
        bi = pos // batch; wi = (pos % batch) // ub
        print(f"  (chunk, batch, ubatch) (first {show}): {sorted(set(zip(chunk.tolist(), bi.tolist(), wi.tolist())))[:show]}")
    return 1 if any_diff else 0


def kld_block(path):
    txt = re.sub(r'[ \t]+', ' ', open(path, errors='ignore').read())
    def last(pat):
        m = re.findall(pat, txt)
        return float(m[-1]) if m else None
    return {'mean': last(r'Mean KLD: (-?[0-9.]+)'), 'se': last(r'Mean KLD: -?[0-9.]+ . ([0-9.]+)'),
            'p99': last(r'99\.0% KLD: (-?[0-9.]+)'), 'max': last(r'Maximum KLD: (-?[0-9.]+)'),
            'top': last(r'Same top p: ([0-9.]+)'), 'ppl': last(r'Mean PPL\(Q\) ?: ([0-9.]+)')}


def log(paths):
    ks = []
    for p in paths:
        k = kld_block(p)
        if k['mean'] is None:
            print(f"{p}: no 'KL divergence statistics' block (run unfinished, or not a --kl-divergence log)"); return 2
        print(f"{os.path.basename(p)}: mean KLD {k['mean']:.6f} +- {k['se'] or 0:.6f}  99% {k['p99']}  max {k['max']}  "
              f"same-top {k['top']}%  PPL(Q) {k['ppl']}")
        ks.append(k)
    if len(ks) == 2:
        a, b = ks
        lo = "A" if a['mean'] < b['mean'] else "B" if b['mean'] < a['mean'] else "neither"
        ratio = (f"{max(a['mean'], b['mean']) / min(a['mean'], b['mean']):.2f}x" if min(a['mean'], b['mean']) > 0 else "n/a")
        sig = abs(a['mean'] - b['mean']) > 2 * ((a['se'] or 0) + (b['se'] or 0))
        print(f"A={paths[0]} B={paths[1]}: {lo} has the lower mean KLD, ratio {ratio}, "
              f"{'outside' if sig else 'inside'} 2 x (sum of standard errors)")
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
    if cmd == 'peek' and rest:
        n = opt('--rows', 6)
        return peek(rest[0], n)
    if cmd == 'diff':
        ub = opt('--ub', None); batch = opt('--batch', 2048); show = opt('--show', 40)
        if ub is None or len(rest) < 2:
            print("usage: kldtool.py diff <A.kld> <B.kld> [more ...] --ub N [--batch N] [--show N]", file=sys.stderr); return 2
        return diff(rest, ub, batch, show)
    if cmd == 'log' and 1 <= len(rest) <= 2:
        return log(rest)
    print(__doc__, file=sys.stderr); return 2


if __name__ == '__main__':
    sys.exit(main())

#!/usr/bin/env python3
"""perflog.py: parse ggml-vulkan perf-logger output (stdlib only).

Input: stderr of any llama.cpp tool run with GGML_VK_PERF_LOGGER=1 (serialising: a barrier after
every node, exclusive per-op times) or with GGML_VK_PERF_LOGGER=1 GGML_VK_PERF_LOGGER_CONCURRENT=1
(true totals; one row per group of nodes that ran between two timestamps, names joined by ", ").
The logger prints one block per graph (per split, per ubatch):

    Vulkan Timings:
    <label>: <calls> x <avg> us = <total> us[ (<rate> GFLOPS/s)]
    Total time: <sum> us.

Rows are sorted by label, not by execution order. Values use 6 significant digits and switch to
scientific notation (1.23457e+06) above 1e6 us. GGML_VK_PERF_LOGGER_FREQUENCY=N prints one block
per N graphs, which breaks ubatch cycle detection: keep it at 1.

Rules this tool encodes:
  - default selection is the LAST block (the first graph carries pipeline compiles and buffer
    growth), never the largest; --ubatch takes the last detected cycle of blocks, because one
    ubatch can span several blocks (one per graph split).
  - every derived number is printed next to a cross-check: the logger's own "Total time" against
    the sum of rows, and with a llama-bench jsonl, op time against wall per ubatch (1000 / t/s).
    A mismatch over a few percent is a parsing bug until proven otherwise.
  - rank from the serialising logger; quote totals from the concurrent one. Per-op attribution in
    a concurrent log is smeared across the group.
  - a fused chain whose first node is "not enqueued" is charged to the next logged node: a
    surprising row right after a chain start is suspect.
  - count the ops in a chained row before computing a rate (`shapes` skips multi-op rows).

Subcommands (run `perflog.py <cmd> -h` for options):
  blocks LOG                 list blocks: index, rows, calls, row sum vs logged total, ubatch cycle
  top LOG                    top rows of the selection, --by row|kind|class
  sum LOG [LOG2]             totals per row label over ALL blocks (+ per-block diff for two logs)
  diff A B                   per class, per op kind and per dense GEMM shape, two logs
  shapes LOG [LOG2]          dense GEMM per shape: calls, us/call, TF/s, logger GFLOPS/s
  ubatch LOG [LOG2]          timed ubatch: op time vs wall (llama-bench jsonl) + residual
  xbackend --vk LOG --hip CSV   per-class time, Vulkan block vs rocprofv3 kernel_stats.csv

Classes: --classes FILE, either JSON ([["class", "regex"], ...] or {"class": "regex", ...} in
order) or text lines "class regex" (# comments). A row goes to the first class whose regex
re.search-es its label (Vulkan) or kernel name (HIP); no match = "other". A concurrent row goes
to the first class matching any member.
"""
import argparse, collections, csv, json, math, os, re, sys

NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?|nan|inf'
ROW = re.compile(r'^\s*(?P<label>.+?): (?P<n>\d+) x (?P<avg>' + NUM + r') us = (?P<tot>' + NUM +
                 r') us(?: \((?P<rate>' + NUM + r') GFLOPS/s\))?\s*$')
TOTAL = re.compile(r'^\s*Total time: (?P<tot>' + NUM + r') us')
# concurrent groups join member labels with ", "; FLASH_ATTN_EXT / CONV_2D labels contain commas
# themselves, so split only where an op token (3+ caps, then space, "(", "," or end) follows
GROUP_SPLIT = re.compile(r', (?=[A-Z][A-Z0-9_]{2,}(?:[ (,]|$))')
GEMM = re.compile(r'\bMUL_MAT(?P<vec>_VEC)? (?P<type>\S+) m=(?P<m>\d+) n=(?P<n>\d+) k=(?P<k>\d+)'
                  r'(?: batch=(?P<batch>\d+))?')

# Generic default classes: upstream op names (Vulkan) and ggml-cuda/HIP kernel names; first match
# wins. HIP: upstream mul_mat_q is one kernel for dense and expert GEMM, so splitting them needs a
# --classes file matched to your build's kernel names. Vulkan runs the B quantise/convert inside
# the MUL_MAT interval while HIP runs separate kernels (gemm_prep): add it to the GEMM classes
# when comparing backends.
DEFAULT_CLASSES = [
    ('moe_gemm', r'\bMUL_MAT_ID(_VEC)?\b|mm_ids_helper'),
    ('dense_mm', r'\bMUL_MAT(_VEC)?\b|\bOUT_PROD\b|mul_mat|Cijk_|gemm'),
    ('gemm_prep', r'quantize_mmq|quantize_q8_1|convert_unary'),
    ('routing', r'TOPK_MOE|\bTOP_K\b|\bARGSORT\b|argsort|topk'),
    ('attn', r'FLASH_ATTN|SOFT_MAX|flash_attn|soft_max'),
    ('recurrent', r'SSM_CONV|SSM_SCAN|GATED_DELTA|RWKV_WKV|GATED_LINEAR_ATTN|ssm_conv|ssm_scan|gated_delta|rwkv'),
    ('norm', r'RMS_NORM|\bNORM\b|L2_NORM|GROUP_NORM|rms_norm|norm_f32|l2_norm'),
    ('glue', r'.'),
]


def fnum(s):
    return float(s)


class Block:
    def __init__(self, index, line):
        self.index, self.line, self.rows, self.logged = index, line, [], None

    @property
    def total(self):
        return sum(r[2] for r in self.rows)

    @property
    def calls(self):
        return sum(r[1] for r in self.rows)


def parse(path):
    """-> list of Block; rows are (label, calls, total_us, avg_us, gflops_or_None)."""
    blocks, cur, loose = [], None, []
    with open(path, errors='replace') as f:
        for ln, line in enumerate(f, 1):
            if 'Vulkan Timings:' in line:
                cur = Block(len(blocks), ln)
                blocks.append(cur)
                continue
            m = TOTAL.match(line)
            if m:
                if cur is not None:
                    cur.logged = fnum(m['tot'])
                cur = None
                continue
            m = ROW.match(line)
            if not m:
                continue
            row = (m['label'].strip(), int(m['n']), fnum(m['tot']), fnum(m['avg']),
                   fnum(m['rate']) if m['rate'] else None)
            if cur is not None:
                cur.rows.append(row)
            elif not blocks:
                loose.append(row)
    if not blocks and loose:   # an already-extracted block without its header: one block
        b = Block(0, 1)
        b.rows = loose
        blocks.append(b)
    return blocks


def members(label):
    return GROUP_SPLIT.split(label)


def kind(label):
    """dispatch kind: fusion label or op name (first token) of each member."""
    return ', '.join(re.split(r'[ (]', p, 1)[0] for p in members(label))


def is_concurrent(blocks):
    return any(len(members(r[0])) > 1 for b in blocks for r in b.rows)


def signature(blocks):
    return [b.calls for b in blocks]


def cycle(blocks, kmax=8, tol=0.03):
    """smallest k whose last k blocks repeat the k before them (op-call counts); then the same
    with a relative tolerance (depth-dependent selection can move a few nodes); else 1."""
    sig = signature(blocks)
    for k in range(1, kmax + 1):
        if len(sig) >= 2 * k and sig[-k:] == sig[-2 * k:-k]:
            return k, 'exact'
    for k in range(1, kmax + 1):
        if len(sig) >= 2 * k and all(abs(a - b) <= tol * max(a, b) for a, b in zip(sig[-k:], sig[-2 * k:-k])):
            return k, f'within {100 * tol:.0f}%'
    return 1, 'none found, using 1'


def load_classes(path):
    if not path:
        return [(c, re.compile(r)) for c, r in DEFAULT_CLASSES]
    txt = open(path).read()
    if txt.lstrip()[:1] in '[{':
        d = json.loads(txt)
        pairs = list(d.items()) if isinstance(d, dict) else [tuple(x) for x in d]
    else:
        pairs = []
        for line in txt.splitlines():
            line = line.strip()
            if line and not line.startswith('#'):
                c, r = line.split(None, 1)
                pairs.append((c, r))
    return [(c, re.compile(r)) for c, r in pairs]


def classify(name, classes):
    for c, rx in classes:
        if rx.search(name):
            return c
    return 'other'


def class_order(classes):
    return [c for c, _ in classes] + ['other']


# ---------- selection ----------

def add_select(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument('--block',
                   help='block index (negative from the end), "last" (default) or "largest"')
    g.add_argument('--ubatch', action='store_true', help='the last detected ubatch cycle of blocks')
    g.add_argument('--cycle', type=int, help='the last K blocks (override cycle detection)')
    g.add_argument('--all', action='store_true', help='every block')


def select(blocks, a, path):
    """-> (rows, description, logged_total_or_None, n_blocks_selected)."""
    if not blocks:
        sys.exit(f'{path}: no perf-logger rows found')
    n = len(blocks)
    if getattr(a, 'all', False):
        sel, how = blocks, f'all {n} blocks'
    elif getattr(a, 'ubatch', False) or getattr(a, 'cycle', None):
        k, why = (a.cycle, 'given') if a.cycle else cycle(blocks)
        k = min(k, n)
        sel, how = blocks[-k:], f'last {k} of {n} blocks (ubatch cycle {why})'
    else:
        b = str(getattr(a, 'block', None) or 'last')
        if b == 'last':
            i = n - 1
        elif b == 'largest':
            i = max(range(n), key=lambda j: blocks[j].total)
        else:
            i = int(b) % n
        sel, how = [blocks[i]], f'block {i} of {n} (log line {blocks[i].line})'
        if i == 0 and n > 1:
            how += ' [first graph: carries compiles and buffer growth]'
    rows = [r for b in sel for r in b.rows]
    logged = None if any(b.logged is None for b in sel) else sum(b.logged for b in sel)
    return rows, how, logged, len(sel)


def xcheck(rows, logged):
    s = sum(r[2] for r in rows)
    if logged is None:
        return f'row sum {s / 1e3:.3f} ms (no logged Total time)'
    d = 100 * (s - logged) / logged if logged else 0.0
    flag = '' if abs(d) <= 0.1 else '  MISMATCH: parsing bug until proven otherwise'
    return f'row sum {s / 1e3:.3f} ms vs logged Total time {logged / 1e3:.3f} ms ({d:+.3f}%){flag}'


def header(path, blocks, how, rows, logged):
    mode = 'concurrent' if is_concurrent(blocks) else 'serialising'
    print(f'== {path}: {how}; {mode} logger; {len(rows)} rows, {sum(r[1] for r in rows)} calls')
    print(f'   {xcheck(rows, logged)}')
    if mode == 'concurrent':
        print('   concurrent log: per-op attribution is smeared across each group; rank from a serialising log')


# ---------- llama-bench jsonl wall ----------

def jsonl_wall(path):
    """last llama-bench jsonl record with avg_ns -> dict(desc, per_ubatch_ms, ...) or None."""
    rec = None
    for line in open(path, errors='replace'):
        line = line.strip()
        if line.startswith('{') and '"avg_ns"' in line:
            try:
                rec = json.loads(line)
            except ValueError:
                m = re.search(r'"avg_ns":\s*(\d+)', line)
                if m:
                    rec = {'avg_ns': int(m.group(1))}
    if rec is None:
        return None
    total = rec['avg_ns'] / 1e6
    npr, ngen, nub = rec.get('n_prompt'), rec.get('n_gen'), rec.get('n_ubatch')
    note, units = '', None
    if npr is not None and ngen is not None:
        pp_ub = math.ceil(npr / nub) if npr and nub else 0
        units = pp_ub + ngen
        if npr and ngen:
            note = ' [mixed pp+tg test: per-ubatch wall is an average over unlike ubatches]'
    units = units or 1
    w = dict(total_ms=total, units=units, per_ms=total / units, rec=rec, note=note)
    ts = rec.get('avg_ts')
    if ts and ngen and not npr:
        w['ts_ms'] = 1000.0 / ts
    elif ts and npr and not ngen and nub:
        w['ts_ms'] = 1000.0 * min(npr, nub) / ts   # ms per ubatch from t/s
    return w


def wall_line(w):
    r = w['rec']
    desc = ', '.join(f'{k}={r[k]}' for k in ('n_prompt', 'n_gen', 'n_depth', 'n_ubatch') if k in r)
    s = (f'wall: last jsonl test ({desc or "fields missing"}) avg {w["total_ms"]:.2f} ms over '
         f'{w["units"]} ubatch(es) = {w["per_ms"]:.3f} ms/ubatch')
    if 'ts_ms' in w:
        s += f'; from avg_ts: {w["ts_ms"]:.3f} ms/ubatch'
    return s + w['note']


# ---------- subcommands ----------

def cmd_blocks(a):
    bl = parse(a.log)
    if not bl:
        sys.exit(f'{a.log}: no perf-logger rows found')
    print(f'== {a.log}: {len(bl)} blocks, {"concurrent" if is_concurrent(bl) else "serialising"} logger')
    print(f'{"idx":>4} {"line":>7} {"rows":>5} {"calls":>6} {"row sum ms":>11} {"logged ms":>10} {"diff%":>7}')
    for b in bl:
        lg = f'{b.logged / 1e3:10.3f}' if b.logged is not None else '         -'
        d = f'{100 * (b.total - b.logged) / b.logged:+7.3f}' if b.logged else '      -'
        print(f'{b.index:4d} {b.line:7d} {len(b.rows):5d} {b.calls:6d} {b.total / 1e3:11.3f} {lg} {d}')
    k, why = cycle(bl)
    sig = signature(bl)
    print(f'ubatch cycle: {k} block(s) ({why}); last 2k call counts: {sig[-2 * k:]}')
    if len(bl) > 1:
        print('use the last block or the last cycle (--ubatch), never block 0 or the largest')


def group_rows(rows, by, classes):
    agg = collections.OrderedDict()
    for label, n, t, avg, rate in rows:
        k = label if by == 'row' else kind(label) if by == 'kind' else classify(label, classes)
        e = agg.setdefault(k, [0, 0.0, collections.Counter()])
        e[0] += n
        e[1] += t
        e[2][re.sub(r'\(.*', '', label).strip()[:60]] += t
    return agg


def cmd_top(a):
    bl = parse(a.log)
    rows, how, logged = select(bl, a, a.log)[:3]
    if a.filter:
        rx = re.compile(a.filter)
        rows = [r for r in rows if rx.search(r[0])]
        logged = None
    header(a.log, bl, how, rows, logged)
    classes = load_classes(a.classes)
    agg = group_rows(rows, a.by, classes)
    tot = sum(e[1] for e in agg.values()) or 1.0
    if a.by == 'class':
        keys = [c for c in class_order(classes) if c in agg]
        for c in keys:
            n, t, ex = agg[c]
            top = '; '.join(f'{k} {v / 1e3:.2f}' for k, v in ex.most_common(3))
            print(f'  {c:12s} {t / 1e3:10.3f} ms {100 * t / tot:5.1f}% {n:7d} calls   top: {top}')
        gm = sum(agg[c][1] for c in agg if c in ('moe_gemm', 'dense_mm'))
        if gm:
            print(f'  matmul share (moe_gemm + dense_mm): {100 * gm / tot:.1f}%')
        return
    keys = sorted(agg, key=lambda k: -agg[k][1])[:a.n]
    for k in keys:
        n, t, _ = agg[k]
        print(f'  {k[:a.width]:<{a.width}} {n:6d} x {t / n:10.2f} us = {t / 1e3:9.3f} ms {100 * t / tot:5.1f}%')
    shown = sum(agg[k][1] for k in keys)
    print(f'  shown {shown / 1e3:.3f} of {tot / 1e3:.3f} ms ({len(keys)} of {len(agg)} entries)')


def per_label(bl):
    tot, cnt = collections.Counter(), collections.Counter()
    for b in bl:
        for label, n, t, avg, rate in b.rows:
            label = re.sub(r'\s+', ' ', label)
            tot[label] += t
            cnt[label] += n
    return tot, cnt


def cmd_sum(a):
    res = []
    for path in a.logs:
        bl = parse(path)
        if not bl:
            sys.exit(f'{path}: no perf-logger rows found')
        tot, cnt = per_label(bl)
        if a.by == 'kind':
            t2, c2 = collections.Counter(), collections.Counter()
            for k in tot:
                t2[kind(k)] += tot[k]
                c2[kind(k)] += cnt[k]
            tot, cnt = t2, c2
        logged = [b.logged for b in bl if b.logged is not None]
        s = sum(tot.values())
        lg = f'; logged Total time sum {sum(logged) / 1e3:.1f} ms ({len(logged)} blocks)' if logged else ''
        print(f'{path}: blocks {len(bl)}, total {s / 1e3:.1f} ms, {s / 1e3 / len(bl):.3f} ms/block{lg}')
        res.append((tot, cnt, len(bl)))
        if len(a.logs) == 1:
            for k, t in tot.most_common(a.n):
                print(f'  {k[:a.width]:<{a.width}} {cnt[k]:7d} calls {t / 1e3:10.3f} ms {100 * t / s:5.1f}%')
    if len(res) == 2:
        (ta, ca, ba), (tb, cb, bb) = res
        keys = sorted(set(ta) | set(tb), key=lambda k: -abs(tb[k] / bb - ta[k] / ba))
        print('label | A us/block | B us/block | B-A | A calls/block | B calls/block')
        for k in keys[:a.n]:
            x, y = ta[k] / ba, tb[k] / bb
            print(f'{k[:a.width]} | {x:.1f} | {y:.1f} | {y - x:+.1f} | {ca[k] / ba:.1f} | {cb[k] / bb:.1f}')


def gemm_rows(rows):
    """dense GEMM rows -> {label: [calls, us, flops per call, match, logger GFLOP, logger-rated us]},
    us in multi-op rows. The logger rate is summed as rate x time so it stays exact over blocks."""
    acc, skipped = collections.OrderedDict(), 0.0
    for label, n, t, avg, rate in rows:
        ms = members(label)
        gm = [GEMM.search(p) for p in ms]
        if not any(gm):
            continue
        if len(ms) > 1:   # chained / concurrent row: no single op to rate
            skipped += t
            continue
        g = gm[0]
        fl = 2 * int(g['m']) * int(g['n']) * int(g['k']) * int(g['batch'] or 1)
        e = acc.setdefault(label, [0, 0.0, fl, g, 0.0, 0.0])
        e[0] += n
        e[1] += t
        if rate is not None:
            e[4] += rate * t / 1e6
            e[5] += t
    return acc, skipped


def cmd_shapes(a):
    sides = []
    for path in a.logs:
        bl = parse(path)
        rows, how, logged = select(bl, a, path)[:3]
        header(path, bl, how, rows, logged)
        acc, skipped = gemm_rows(rows)
        if a.n_tokens is not None:
            acc = collections.OrderedDict((k, v) for k, v in acc.items() if int(v[3]['n']) == a.n_tokens)
        if a.filter:
            rx = re.compile(a.filter)
            acc = collections.OrderedDict((k, v) for k, v in acc.items() if rx.search(k))
        if skipped:
            print(f'   {skipped / 1e3:.3f} ms of GEMM time sits in multi-op rows (not rated; use a serialising log)')
        sides.append(acc)
    if len(sides) == 1:
        acc = sides[0]
        print(f'  {"label":<{a.width}} {"calls":>5} {"us/call":>10} {"TF/s":>7} {"logger TF/s":>11}')
        for k, (n, t, fl, g, lgf, lt) in sorted(acc.items(), key=lambda x: -x[1][1])[:a.n]:
            lr = f'{lgf / lt * 1e3:11.2f}' if lt else '          -'
            print(f'  {k[:a.width]:<{a.width}} {n:5d} {t / n:10.1f} {fl * n / t / 1e6:7.2f} {lr}')
        print(f'  total {sum(v[1] for v in acc.values()) / 1e3:.3f} ms in {len(acc)} shapes')
        return
    sa, sb = sides
    la, lb = a.labels
    keys = sorted(set(sa) | set(sb), key=lambda k: -abs(sa.get(k, [0, 0.0])[1] - sb.get(k, [0, 0.0])[1]))
    print(f'  dense GEMM per shape ({la} vs {lb}: calls x us/call = ms; sorted by |diff|)')
    for k in keys[:a.n]:
        ca, ta = sa.get(k, [0, 0.0])[:2]
        cb, tb = sb.get(k, [0, 0.0])[:2]
        pa = f'{ta / ca:8.1f}' if ca else '       -'
        pb = f'{tb / cb:8.1f}' if cb else '       -'
        print(f'    {k[:a.width]:<{a.width}} {la} {ca:4d} x {pa} = {ta / 1e3:7.3f} | {lb} {cb:4d} x {pb} = {tb / 1e3:7.3f} | diff {(ta - tb) / 1e3:+7.3f} ms')


def diff_tables(A, B, la, lb, classes, n, width, shapes):
    ca, cb = collections.Counter(), collections.Counter()
    for r in A:
        ca[classify(r[0], classes)] += r[2]
    for r in B:
        cb[classify(r[0], classes)] += r[2]
    ta, tb = sum(ca.values()), sum(cb.values())
    print(f'  op time {la} {ta / 1e3:.3f} ms, {lb} {tb / 1e3:.3f} ms, diff {(ta - tb) / 1e3:+.3f} ms'
          + (f' ({100 * (ta / tb - 1):+.1f}%)' if tb else ''))
    print('  by class:')
    for k in sorted(set(ca) | set(cb), key=lambda k: -abs(ca[k] - cb[k])):
        print(f'    {k:<18} {la} {ca[k] / 1e3:9.3f}  {lb} {cb[k] / 1e3:9.3f}  diff {(ca[k] - cb[k]) / 1e3:+8.3f} ms')
    oa, ob = collections.Counter(), collections.Counter()
    for r in A:
        oa[kind(r[0])] += r[2]
    for r in B:
        ob[kind(r[0])] += r[2]
    print(f'  by op kind (top {n} diffs; {la} / {lb} ms):')
    for x, k in sorted(((oa[k] - ob[k], k) for k in set(oa) | set(ob)), key=lambda v: -abs(v[0]))[:n]:
        print(f'    {k[:width]:<{width}} {x / 1e3:+8.3f} ms  ({oa[k] / 1e3:.3f} / {ob[k] / 1e3:.3f})')
    if shapes:
        sa, sk_a = gemm_rows(A)
        sb, sk_b = gemm_rows(B)
        if not sa and not sb:
            return
        keys = sorted(set(sa) | set(sb), key=lambda k: -abs(sa.get(k, [0, 0.0])[1] - sb.get(k, [0, 0.0])[1]))
        print(f'  dense GEMM per shape ({la} vs {lb}: calls x us/call = ms; sorted by |diff|):')
        sw = max(width, 52)
        for k in keys[:n]:
            ca_, ta_ = sa.get(k, [0, 0.0])[:2]
            cb_, tb_ = sb.get(k, [0, 0.0])[:2]
            pa = f'{ta_ / ca_:8.1f}' if ca_ else '       -'
            pb = f'{tb_ / cb_:8.1f}' if cb_ else '       -'
            print(f'    {k[:sw]:<{sw}} {la} {ca_:4d} x {pa} = {ta_ / 1e3:7.3f} | {lb} {cb_:4d} x {pb} = {tb_ / 1e3:7.3f} | diff {(ta_ - tb_) / 1e3:+7.3f} ms')
        if sk_a or sk_b:
            print(f'    (not per shape: GEMM time in multi-op rows {la} {sk_a / 1e3:.3f} ms, {lb} {sk_b / 1e3:.3f} ms)')


def cmd_diff(a):
    if a.block is None and not (a.cycle or a.all):
        a.ubatch = True   # default for diff: the detected ubatch of each log
    sel = []
    for path in (a.a, a.b):
        bl = parse(path)
        rows, how, logged = select(bl, a, path)[:3]
        header(path, bl, how, rows, logged)
        sel.append(rows)
    diff_tables(sel[0], sel[1], a.labels[0], a.labels[1], load_classes(a.classes), a.n, a.width, True)
    print('  (serialising A vs concurrent B of the same binary: the op time diff is what concurrency hides)')


def cmd_ubatch(a):
    if len(a.jsonl) not in (0, len(a.logs)):
        sys.exit('give one --jsonl per log')
    res = []
    for i, path in enumerate(a.logs):
        lab = a.labels[i]
        bl = parse(path)
        if not bl:
            sys.exit(f'{lab}: no perf-logger blocks in {path}')
        if a.cycle:
            k, why = min(a.cycle, len(bl)), 'given'
        else:
            k, why = cycle(bl)
        sel = bl[-k:]
        rows = [r for b in sel for r in b.rows]
        t = sum(r[2] for r in rows) / 1e3
        logged = None if any(b.logged is None for b in sel) else sum(b.logged for b in sel)
        w = jsonl_wall(a.jsonl[i] if a.jsonl else path)
        if a.tg:
            w = dict(per_ms=1000.0 / a.tg[i], note='', units=1, total_ms=1000.0 / a.tg[i],
                     rec={'from': f'--tg {a.tg[i]}'})
        print(f'{lab}: timed ubatch = last {k} of {len(bl)} blocks (cycle {why}), '
              f'{sum(r[1] for r in rows)} op calls, {t:.3f} ms op time')
        print(f'  {xcheck(rows, logged)}')
        if w:
            if 'from' in w['rec']:
                print(f'  wall: {w["per_ms"]:.3f} ms/ubatch ({w["rec"]["from"]})')
            else:
                print(f'  {wall_line(w)}')
            print(f'  op time {t:.3f} vs wall {w["per_ms"]:.3f} ms -> residual (host / unlogged / overlap) '
                  f'{w["per_ms"] - t:+.3f} ms')
        else:
            print('  wall: no llama-bench jsonl found (pass --jsonl FILE or --tg T)')
        res.append((lab, rows, t, w['per_ms'] if w else None))
    if len(res) == 2:
        (la, A, ta, wa), (lb, B, tb, wb) = res
        if wa and wb:
            print(f'wall {la} vs {lb}: {wa - wb:+.3f} ms ({100 * (wa / wb - 1):+.1f}%); op time {ta - tb:+.3f} ms; '
                  f'residual {(wa - ta) - (wb - tb):+.3f} ms')
        diff_tables(A, B, la, lb, load_classes(a.classes), a.n, a.width, a.shapes)


def cmd_xbackend(a):
    classes = load_classes(a.classes)
    sides = []
    if a.vk:
        bl = parse(a.vk)
        rows, how, logged = select(bl, a, a.vk)[:3]
        header(a.vk, bl, how, rows, logged)
        sides.append(('vulkan', [(r[0], r[1], r[2]) for r in rows]))
    if a.hip:
        rows = []
        with open(a.hip, newline='') as f:
            for r in csv.DictReader(f):
                rows.append((r['Name'], int(r['Calls']), float(r['TotalDurationNs']) / 1e3 / a.hip_div))
        print(f'== {a.hip}: rocprofv3 kernel stats, {len(rows)} kernels'
              + (f', divided by {a.hip_div}' if a.hip_div != 1 else '')
              + ' (whole profiled run: compare shares, or --hip-div by the graphs it covers)')
        sides.append(('hip', rows))
    if not sides:
        sys.exit('give --vk LOG and/or --hip kernel_stats.csv')
    order = class_order(classes)
    for name, rows in sides:
        tot, n, ex = collections.Counter(), collections.Counter(), collections.defaultdict(collections.Counter)
        for label, calls, us in rows:
            c = classify(label, classes)
            tot[c] += us
            n[c] += calls
            ex[c][label[:70]] += us
        T = sum(tot.values()) or 1.0
        cw = max(len(c) for c in order)
        print(f'-- {name}: {T / 1e3:.3f} ms kernel time')
        for c in order:
            if c in tot:
                top = '; '.join(f'{k} {100 * v / T:.1f}%' for k, v in ex[c].most_common(3))
                print(f'  {c:{cw}s} {100 * tot[c] / T:5.1f}% {tot[c] / 1e3:10.3f} ms {n[c]:7d} calls   {top}')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    def common(sp, n=25, width=60):
        sp.add_argument('-n', type=int, default=n, help=f'rows to show (default {n})')
        sp.add_argument('--width', type=int, default=width, help='label column width')
        sp.add_argument('--classes', help='class map file (JSON or "class regex" lines)')

    s = sub.add_parser('blocks', help='list blocks and the detected ubatch cycle')
    s.add_argument('log')
    s.set_defaults(fn=cmd_blocks)

    s = sub.add_parser('top', help='top rows of the selection (default: last block)')
    s.add_argument('log')
    add_select(s)
    common(s)
    s.add_argument('--by', choices=('row', 'kind', 'class'), default='row',
                   help='row label, dispatch kind (fusion label or op) or class')
    s.add_argument('--filter', help='only rows whose label matches this regex')
    s.set_defaults(fn=cmd_top)

    s = sub.add_parser('sum', help='totals per row label over all blocks; diff for two logs')
    s.add_argument('logs', nargs='+', metavar='LOG')
    common(s, width=90)
    s.add_argument('--by', choices=('row', 'kind'), default='row')
    s.set_defaults(fn=cmd_sum)

    s = sub.add_parser('diff', help='two logs: per class, per op kind, per dense GEMM shape')
    s.add_argument('a')
    s.add_argument('b')
    add_select(s)
    common(s, n=15, width=52)
    s.add_argument('--labels', nargs=2, default=('A', 'B'))
    s.set_defaults(fn=cmd_diff)

    s = sub.add_parser('shapes', help='dense GEMM per shape: us/call, TF/s; diff for two logs')
    s.add_argument('logs', nargs='+', metavar='LOG')
    add_select(s)
    common(s, width=56)
    s.add_argument('--n-tokens', type=int, help='only shapes with n == this')
    s.add_argument('--filter', help='only labels matching this regex')
    s.add_argument('--labels', nargs=2, default=('A', 'B'))
    s.set_defaults(fn=cmd_shapes)

    s = sub.add_parser('ubatch', help='timed ubatch: op time vs wall + residual; diff for two logs')
    s.add_argument('logs', nargs='+', metavar='LOG')
    s.add_argument('--cycle', type=int, help='blocks per ubatch (override detection)')
    s.add_argument('--jsonl', action='append', default=[],
                   help='llama-bench -o jsonl file per log (default: jsonl lines in the log itself)')
    s.add_argument('--tg', type=float, action='append', help='t/s per log instead of a jsonl (ms = 1000/t)')
    s.add_argument('--labels', nargs=2, default=('A', 'B'))
    s.add_argument('--shapes', action='store_true', help='add the dense GEMM per-shape diff')
    common(s, n=10, width=28)
    s.set_defaults(fn=cmd_ubatch)

    s = sub.add_parser('xbackend', help='per class: Vulkan block vs rocprofv3 kernel_stats.csv')
    s.add_argument('--vk', help='perf-logger log')
    s.add_argument('--hip', help='rocprofv3 *_kernel_stats.csv (Name, Calls, TotalDurationNs)')
    s.add_argument('--hip-div', type=float, default=1.0, help='divide HIP totals (e.g. by graphs profiled)')
    add_select(s)
    common(s)
    s.set_defaults(fn=cmd_xbackend)

    a = p.parse_args(argv)
    if getattr(a, 'logs', None) and len(a.logs) > 2:
        p.error('at most two logs')
    if a.cmd == 'ubatch' and a.tg and len(a.tg) != len(a.logs):
        p.error('give one --tg per log')
    a.fn(a)


if __name__ == '__main__':
    try:
        main()
    except BrokenPipeError:
        os._exit(0)

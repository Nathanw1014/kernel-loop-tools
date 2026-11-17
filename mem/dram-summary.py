#!/usr/bin/env python3
"""dram-summary.py <dram.tsv> [more.tsv ...] [--act N] [--line] [--timeline S] [--skip S] [--tau S]

Per file (= per cell): DRAM GB/s while busy, from a dram-sample.py record. Busy = SMU GfxActivity >= N (default 50); the
activity and the bandwidth fields carry the same firmware EWMA, so gating on activity keeps the ramp-in / ramp-out samples
consistent. --skip S drops the first S seconds of every file (model load). Prints n busy / n total, then mean / p50 / p95 / max of
rd, wr and rd+wr GB/s, plus the idle (act < 5) floor. --line = one compact line (job RESULTS / meta). --timeline S = per-S-second
means of act / rd / wr / tot over the whole file (t relative to the first sample), plus the deconvolved columns.
Calibration: rd / wr are recomputed from the raw u16 columns with the factor from the file header (cal=, written by dram-sample.py
from DRAM_CAL); DRAM_CAL in the environment here overrides it.
DECONVOLVED ("inst"): the firmware EWMA y_n = y_n-1 + k (x_n - y_n-1), k = 1 - exp(-dt / tau), is inverted per sample:
x_n = y_n-1 + (y_n - y_n-1) / k. tau defaults to the header's filter_us (else 1 s); --tau overrides. This recovers phases much shorter
than tau, at the cost of noise amplified by 1 / k. "inst" stats gate on the deconvolved activity (5-sample median) >= N."""
import math, os, statistics, sys

args = sys.argv[1:]
def opt(name, default, conv=float):
    if name in args:
        i = args.index(name); v = conv(args[i + 1]); del args[i:i + 2]; return v
    return default
TAU_ARG = opt('--tau', None); ACT = opt('--act', 50.0); TL = opt('--timeline', 0.0); SKIP = opt('--skip', 0.0)
LINE = '--line' in args
if LINE: args.remove('--line')
if not args or args[0] in ('-h', '--help'): print(__doc__); sys.exit(2)

def load(p):
    R = []; hdr = {}
    for l in open(p, errors='replace'):
        if l.startswith('# dram-sample'):
            for w in l.split():
                if '=' in w:
                    k, v = w.split('=', 1)
                    try: hdr[k] = float(v)
                    except ValueError: pass
            if 'GB/s/raw' in l and 'cal' in hdr:   # older sampler format: cal = GB/s per raw unit, not a factor
                hdr['cal'] /= 0.001
        if l.startswith('#'): continue
        f = l.rstrip('\n').split('\t')
        if len(f) < 14: continue
        try: R.append([float(x) for x in f])
        except ValueError: pass
    return R, hdr

def unwrap(R, cal, tau):
    # recompute rd / wr / tot from the raw u16 columns (MB/s, wrap at 65536) by continuity; see dram-sample.py
    gbs = 0.001 * cal
    for i, raw in ((3, 12), (4, 13)):
        w = 0; prev = None
        for r in R:
            if prev is not None:
                d = r[raw] - prev
                if d < -32768: w += 1
                elif d > 32768: w = max(0, w - 1)
            prev = r[raw]; r[i] = (w * 65536 + r[raw]) * gbs
    for r in R: r[5] = r[3] + r[4]
    # deconvolved columns appended: 14 act_x, 15 rd_x, 16 wr_x, 17 tot_x (first sample: x = y)
    for j, r in enumerate(R):
        if j == 0: r += [r[1], r[3], r[4], r[5]]; continue
        p = R[j - 1]; k = 1 - math.exp(-max(1e-4, r[0] - p[0]) / tau)
        r += [p[i] + (r[i] - p[i]) / k for i in (1, 3, 4, 5)]
    for j, r in enumerate(R):
        w = sorted(x[14] for x in R[max(0, j - 2):j + 3]); r.append(w[len(w) // 2])   # 18 = smoothed act_x
    return R

def pct(v, q):
    v = sorted(v); return v[min(len(v) - 1, int(q * (len(v) - 1) + 0.5))]

for p in args:
    R, hdr = load(p)
    cal = float(os.environ['DRAM_CAL']) if os.environ.get('DRAM_CAL') else hdr.get('cal', 1.0)
    tau = TAU_ARG if TAU_ARG is not None else (hdr['filter_us'] / 1e6 if hdr.get('filter_us') else 1.0)
    R = unwrap(R, cal, tau)
    if not R: print(f"{'dram:' if LINE else p + ':'} no samples"); continue
    t0 = R[0][0]; R = [r for r in R if r[0] - t0 >= SKIP]
    if not R: print(f"{'dram:' if LINE else p + ':'} no samples after --skip"); continue
    B = [r for r in R if r[1] >= ACT]; I = [r for r in R if r[1] < 5]; X = [r[17] for r in R if r[18] >= ACT]
    xs = f"inst n={len(X)} p50 {pct(X, .5):.1f} p95 {pct(X, .95):.1f}" if X else "inst n=0"
    idle = f"idle rd {statistics.mean(r[3] for r in I):.1f} wr {statistics.mean(r[4] for r in I):.1f}" if I else "idle n/a"
    if not B:
        print(f"{'dram' if LINE else p + ':'} {xs} | busy 0/{len(R)} (act>={ACT:.0f}) max tot {max(r[5] for r in R):.1f} GB/s, {idle}"); continue
    s = {k: [r[i] for r in B] for k, i in (('rd', 3), ('wr', 4), ('tot', 5))}
    span = B[-1][0] - B[0][0]
    if LINE:
        print(f"dram {xs} | busy n={len(B)} {span:.0f}s tot mean {statistics.mean(s['tot']):.1f} p50 {pct(s['tot'], .5):.1f} p95 {pct(s['tot'], .95):.1f} "
              f"max {max(s['tot']):.1f} GB/s (rd {statistics.mean(s['rd']):.1f} wr {statistics.mean(s['wr']):.1f}) uclk {statistics.mean(r[8] for r in B):.0f} fclk {statistics.mean(r[7] for r in B):.0f}")
    else:
        print(f"{p}: busy {len(B)}/{len(R)} samples (act>={ACT:.0f}, {span:.1f} s), {idle} GB/s; deconvolved tot {xs} GB/s (cal {cal:g}, tau {tau:g} s)")
        for k in ('rd', 'wr', 'tot'):
            v = s[k]; print(f"  {k:<4} mean {statistics.mean(v):7.1f}  p50 {pct(v, .5):7.1f}  p95 {pct(v, .95):7.1f}  max {max(v):7.1f} GB/s")
        print(f"  clocks (busy mean): gfxclk {statistics.mean(r[6] for r in B):.0f} fclk {statistics.mean(r[7] for r in B):.0f} "
              f"uclk {statistics.mean(r[8] for r in B):.0f} MHz, socket {statistics.mean(r[10] for r in B):.1f} W, gfx {statistics.mean(r[11] for r in B):.1f} W")
    if TL > 0:
        print(f"  {'t_s':>6} {'act':>4} {'rd':>7} {'wr':>7} {'tot':>7} | {'act_x':>5} {'rd_x':>7} {'wr_x':>7} {'tot_x':>7}")
        b0 = R[0][0]; cur = []; edge = b0 + TL
        for r in R + [None]:
            if r is None or r[0] >= edge:
                if cur: print(f"  {cur[0][0] - b0:6.1f} {statistics.mean(x[1] for x in cur):4.0f} {statistics.mean(x[3] for x in cur):7.1f} "
                              f"{statistics.mean(x[4] for x in cur):7.1f} {statistics.mean(x[5] for x in cur):7.1f} | {statistics.mean(x[14] for x in cur):5.0f} "
                              f"{statistics.mean(x[15] for x in cur):7.1f} {statistics.mean(x[16] for x in cur):7.1f} {statistics.mean(x[17] for x in cur):7.1f}")
                if r is None: break
                cur = []; edge = r[0] - (r[0] - b0) % TL + TL
            cur.append(r)

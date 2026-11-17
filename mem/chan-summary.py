#!/usr/bin/env python3
"""chan-summary.py <gcea.tsv>... [--line] [--busy F] [--from S] [--to S] [--timeline S] [--bytes B]
Per-instance GCEA traffic from gcea-sample (chan-run.sh). Each row = one window [t0,t1) of raw request counts per GCEA
instance (v0..vN). Windows alternate RD / WR groups. Only instances that are nonzero anywhere in the input count as active (some
parts expose more instances than count). Busy windows: total rate >= F x the run's p95 window rate (default 0.25) inside
[--from, --to]. Prints per counter: GB/s (requests x --bytes, default 32 B; verify on your part, DRAM-MONITOR.md), per-instance
share %, max/mean of the summed busy windows and the p50 / p95 / max of the PER-WINDOW max/mean (1.00 = even; camping >> 1; with
N active instances, fully camped on one = N). --timeline S: per-bin table of GB/s and max/mean. --line: one-line summary.
Coverage: HIP / KFD work of any process; Vulkan work may read zero depending on the driver (verify with camp-hip).
Caveat: GCEA sits before the last-level cache, so these are fabric requests (cache hits included), not DRAM CAS."""
import sys, statistics
args = sys.argv[1:]
def opt(name, default, conv=float):
    if name in args:
        i = args.index(name); v = conv(args[i + 1]); del args[i:i + 2]; return v
    return default
line = '--line' in args
if line: args.remove('--line')
busy_f = opt('--busy', 0.25); t_from = opt('--from', -1e18); t_to = opt('--to', 1e18)
tl = opt('--timeline', 0.0); B = opt('--bytes', 32.0)
if not args or args[0] in ('-h', '--help'): print(__doc__); sys.exit(2)
rows = {}  # counter -> list of (t0, t1, [v])
for p in args:
    for ln in open(p):
        if ln.startswith('#'):
            if 'unavailable' in ln and line: print('chan: ' + ln.split('chan:', 1)[1].strip()); sys.exit(0)
            continue
        f = ln.rstrip('\n').split('\t')
        if len(f) < 4: continue
        t0, t1 = float(f[0]), float(f[1])
        if t1 < t_from or t0 > t_to: continue
        rows.setdefault(f[2], []).append((t0, t1, [float(x) for x in f[3:]]))
if not rows:
    print('chan: no samples'); sys.exit(0)
NI = max((len(v) for rs in rows.values() for _, _, v in rs), default=0)
act = sorted({i for rs in rows.values() for _, _, v in rs for i, x in enumerate(v) if x > 0})
def mm(v):
    a = [v[i] if i < len(v) else 0.0 for i in act]; m = sum(a) / max(1, len(a))
    return (max(a) / m if m else 0.0), a
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(q * (len(xs) - 1) + 0.5))] if xs else 0.0
out = []
for name in sorted(rows):
    rs = rows[name]
    rate = [sum(v) * B / max(1e-9, t1 - t0) / 1e9 for t0, t1, v in rs]
    thr = busy_f * pct(rate, 0.95)
    bz = [(r, rt) for r, rt in zip(rs, rate) if rt >= thr and rt > 0]
    if not bz: out.append((name, None)); continue
    sumv = [0.0] * NI
    for (t0, t1, v), _ in bz:
        for i, x in enumerate(v): sumv[i] += x
    wmm = [mm(v)[0] for (t0, t1, v), _ in bz]
    tot_mm, a = mm(sumv)
    out.append((name, dict(n=len(bz), gbs=statistics.median([rt for _, rt in bz]), gbs_max=max(rt for _, rt in bz),
                           share=[100 * x / sum(a) for x in a], mm=tot_mm, wp50=pct(wmm, .5), wp95=pct(wmm, .95), wmax=max(wmm))))
short = lambda n: n.replace('GCEA_', '').replace('_SIZE_REQ', '')
if line:
    print('chan ' + ' | '.join(f"{short(n)} {d['gbs']:.1f} GB/s mm {d['mm']:.2f} win p95 {d['wp95']:.2f} n={d['n']}"
                                for n, d in out if d) + f" (inst {act[0]}-{act[-1]})" if act else 'chan: all zero (Vulkan work may not be counted on this driver; verify with camp-hip)')
    sys.exit(0)
print(f"active instances: {act}   (bytes/request {B:g}; busy = window rate >= {busy_f:g} x p95)")
for n, d in out:
    if not d: print(f"{n}: no busy windows"); continue
    print(f"{n}: busy windows {d['n']}, p50 {d['gbs']:.1f} GB/s (max {d['gbs_max']:.1f}), summed max/mean {d['mm']:.3f}, "
          f"per-window max/mean p50 {d['wp50']:.2f} p95 {d['wp95']:.2f} max {d['wmax']:.2f}")
    print('   share %: ' + ' '.join(f"{x:5.2f}" for x in d['share']))
if tl > 0:
    print(f"\ntimeline ({tl:g} s bins): GB/s and per-bin max/mean per counter")
    names = sorted(rows); bins = {}
    for n in names:
        for t0, t1, v in rows[n]:
            b = int(t0 // tl); d = bins.setdefault(b, {}).setdefault(n, [0.0, [0.0] * NI])
            d[0] += t1 - t0
            for i, x in enumerate(v): d[1][i] += x
    print('   t_s ' + ''.join(f"{short(n):>10} {'mm':>5}" for n in names))
    for b in sorted(bins):
        s = f"{b * tl:6.1f} "
        for n in names:
            if n in bins[b]:
                dt, v = bins[b][n]; s += f"{sum(v) * B / dt / 1e9:10.1f} {mm(v)[0]:5.2f}"
            else: s += f"{'':10} {'':5}"
        print(s)

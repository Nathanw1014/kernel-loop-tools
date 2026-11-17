#!/usr/bin/env python3
"""gcea-channels.py <counter_collection.csv>... [--top N]: per-memory-channel GPU traffic from rocprofv3 dispatch-counting runs.
Input: rocprofv3 CSV whose counters are named GCEA_RD_CHnn / GCEA_WR_CHnn, each one GCEA_{R,W}DRAM_SIZE_REQ selected on GCEA
instance nn by an extra-counter file (rocprofv3 -E <file.yaml>; one select(...,[DIMENSION_INSTANCE=[nn]]) entry per instance, see
DRAM-MONITOR.md). Prints, for reads and writes: total requests, per-channel share %, max/mean and min/mean (1.00 = even; camping
shows as max/mean >> 1) over the ACTIVE instances (nonzero anywhere in the input), then the N heaviest kernels with their whole-run
max/mean and the request-weighted p95 of the PER-DISPATCH max/mean (camping is per dispatch; the whole run can average out).
GB figures assume 32 B per request: verify that on your part (DRAM-MONITOR.md). HIP only (rocprofv3); no root needed."""
import collections, csv, sys
args = sys.argv[1:]; top = 8
if '--top' in args: i = args.index('--top'); top = int(args[i + 1]); del args[i:i + 2]
if not args or args[0] in ('-h', '--help'): print(__doc__); sys.exit(2)
tot = {'RD': collections.Counter(), 'WR': collections.Counter()}
ker = {'RD': collections.defaultdict(collections.Counter), 'WR': collections.defaultdict(collections.Counter)}
disp = {'RD': collections.defaultdict(collections.Counter), 'WR': collections.defaultdict(collections.Counter)}; dk = {}
for p in args:
    for r in csv.DictReader(open(p)):
        n = r['Counter_Name']
        if not n.startswith('GCEA_') or '_CH' not in n: continue
        kind, ch = n[5:7], int(n.split('_CH')[1]); v = float(r['Counter_Value'])
        tot[kind][ch] += v; ker[kind][r['Kernel_Name'][:70]][ch] += v
        key = (p, r['Dispatch_Id']); disp[kind][key][ch] += v; dk[key] = r['Kernel_Name'][:70]
NCH = 1 + max((c for k in tot for c in tot[k]), default=-1)
ACT = [i for i in range(NCH) if tot['RD'].get(i, 0) + tot['WR'].get(i, 0) > 0]
def line(c):
    v = [c.get(i, 0.0) for i in ACT]; m = sum(v) / max(1, len(ACT))
    return v, m, (max(v) / m if m else 0), (min(v) / m if m else 0)
for kind in ('RD', 'WR'):
    v, m, mx, mn = line(tot[kind])
    if not m: continue
    print(f"{kind}: total {sum(v):.4g} requests (~{sum(v) * 32 / 1e9:.1f} GB at 32 B), active instances {ACT}, max/mean {mx:.3f} min/mean {mn:.3f}")
    print('   share %: ' + ' '.join(f"{100 * x / sum(v):5.2f}" for x in v))
    ks = sorted(ker[kind].items(), key=lambda kv: -sum(kv[1].values()))[:top]
    for k, c in ks:
        kv, km, kmx, kmn = line(c)
        W = sorted((line(dc)[2], sum(dc.values())) for key, dc in disp[kind].items() if dk[key] == k and sum(dc.values()) > 0)
        acc, tw, p95 = 0, sum(w for _, w in W), 0
        for r_, w in W:
            acc += w
            if acc >= 0.95 * tw: p95 = r_; break
        print(f"   {100 * sum(kv) / sum(v):5.1f}% of {kind}  max/mean {kmx:5.2f} min/mean {kmn:5.2f}  per-dispatch max/mean p95 {p95:5.2f} (n={len(W)})  {k}")

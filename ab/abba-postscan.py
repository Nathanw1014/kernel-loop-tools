#!/usr/bin/env python3
"""abba-postscan.py <abba-dir> [--max-cv PCT]

Post-run scan of an abba.sh result dir.
  1. Slow-rep rule: every llama-bench record whose stddev_ts > PCT% of avg_ts (default 10) is reported with its cell,
     arm, depth and log paths. A noisy record means one repetition stalled; read its logs (and the kernel log around
     that time) before trusting the arm mean.
  2. Major page faults per cell, from the majflt=N field of the RESULTS.txt cell lines (samplers/flt-sample.sh): the
     median and the top cells, so a slow cell can be matched to a page-in burst (an mmap'd model under memory pressure).
"""
import sys, glob, json, re, os

args = sys.argv[1:]
if not args or args[0] in ('-h', '--help'):
    print(__doc__); sys.exit(0 if args else 2)
d = args[0]
max_cv = float(args[args.index('--max-cv') + 1]) if '--max-cv' in args else 10.0
slow = 0
for f in sorted(glob.glob(d + '/bench-*.jsonl')):
    if f.endswith('.throttled.jsonl'): continue
    cell = os.path.basename(f)[6:-6]
    for line in open(f, errors='replace'):
        try: r = json.loads(line)
        except Exception: continue
        a, s = r.get('avg_ts') or 0, r.get('stddev_ts') or 0
        if a and s > max_cv / 100 * a:
            slow += 1
            print(f"SLOW REP: cell {cell} depth {r.get('n_depth')} avg {a:.1f} +/- {s:.1f} t/s (samples {r.get('samples_ts')}) | logs {f} {f[:-6]}.err {f[:-6]}.therm")
print(f"slow-rep rule (stddev > {max_cv:g}% of mean): {slow} record(s)")
mf = []
try:
    for line in open(d + '/RESULTS.txt', errors='replace'):
        m = re.search(r'cell (\d+) (\S+) \(([^)]*)\).*?majflt=(\d+)', line)
        if m: mf.append((int(m.group(4)), m.group(1), m.group(2), m.group(3)))
except OSError: pass
if mf:
    mf.sort(reverse=True); med = sorted(x[0] for x in mf)[len(mf)//2]
    print(f"majflt per cell: median {med}; top: " + "; ".join(f"cell {c} {a} ({t}) {n}" for n, c, a, t in mf[:4]))

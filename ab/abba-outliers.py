#!/usr/bin/env python3
"""abba-outliers.py <abba-dir> [--ref ARM] [--ratio 0.85]

Repetition-level outlier check for an abba.sh result dir. The harness summary averages avg_ts per launch, so one
stalled repetition can move a whole arm. This reads samples_ts, flags every repetition below RATIO x the best
repetition of the same test in the same launch, prints the launch's major page faults (bench-<i>-<arm>.flt from
samplers/flt-sample.sh, "?" when absent), and recomputes per-arm means from the clean repetitions only (each launch =
mean of its clean reps). Warm-up (cell 0) and *.throttled.jsonl are skipped like the harness does; extra cells (any
index) are included. --ref defaults to the first arm in launch order (abba.sh's base arm).
"""
import collections, glob, json, statistics as st, sys

args = sys.argv[1:]
if not args:
    sys.exit(__doc__)
if args[0] in ('-h', '--help'):
    print(__doc__); sys.exit(0)
d = args[0]
ref = args[args.index('--ref') + 1] if '--ref' in args else None
ratio = float(args[args.index('--ratio') + 1]) if '--ratio' in args else 0.85   # a looser ratio lets real stalls through

raw = collections.defaultdict(list)    # (arm, test, depth) -> per-launch mean of all reps
clean = collections.defaultdict(list)  # (arm, test, depth) -> per-launch mean of clean reps
flags = []
files = sorted(glob.glob(d + '/bench-*.jsonl'), key=lambda x: int(x.rsplit('/', 1)[1][6:].split('-', 1)[0]))
for f in files:
    name = f.rsplit('/', 1)[1]
    if name.endswith('.throttled.jsonl'):
        continue
    i, arm = name[6:-6].split('-', 1)
    if i == '0':
        continue
    for l in open(f):
        if not l.startswith('{'):
            continue
        r = json.loads(l)
        key = (arm, 'pp' if r['n_prompt'] else 'tg', r['n_depth'])
        s = r.get('samples_ts') or [r['avg_ts']]
        best = max(s)
        ok = [x for x in s if x >= ratio * best]
        bad = [x for x in s if x < ratio * best]
        raw[key].append(st.mean(s))
        clean[key].append(st.mean(ok))
        if bad:
            mj = "?"
            try:
                mj = open(f[:-6] + ".flt").read().split()[0]   # flt-sample.sh: majflt minflt (<= 1 s before exit)
            except Exception:
                pass
            flags.append(f"cell {i} {arm} {key[1]}@d{key[2]}: samples {[round(x, 1) for x in s]} -> dropped {[round(x, 1) for x in bad]} (launch majflt {mj})")

if ref is None:
    ref = next(iter(raw))[0] if raw else '?'
print(f"abba-outliers {d}: ratio {ratio}, ref {ref}, {len(flags)} flagged repetition(s)")
for fl in flags:
    print("  FLAG " + fl)
tests = sorted({(k[1], k[2]) for k in raw})
arms = []
for k in raw:
    if k[0] not in arms:
        arms.append(k[0])
for t, dep in tests:
    r_ref = clean.get((ref, t, dep))
    for arm in arms:
        k = (arm, t, dep)
        if k not in raw:
            continue
        m_raw, m_clean = st.mean(raw[k]), st.mean(clean[k])
        dv = f" {100 * (m_clean / st.mean(r_ref) - 1):+.1f}% vs {ref}" if r_ref and arm != ref else ""
        mark = " (outlier-adjusted)" if abs(m_raw - m_clean) > 1e-9 else ""
        print(f"  {t}@d{dep} {arm}: n={len(raw[k])} raw {m_raw:.1f} clean {m_clean:.1f}{dv}{mark}")

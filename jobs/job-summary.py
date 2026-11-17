#!/usr/bin/env python3
"""job-summary.py <OUT>: SUMMARY.txt for a job-run record dir.

Bench cells named "<arm>.<rep>" are grouped by arm: per test the mean of the reps' avg_ts, the spread across reps
(max-min) and the ratio to the FIRST arm in spec order that is not a warm-up (arms named warm* are listed, never the reference). Everything is recomputed from cells/*/stdout.log (jsonl), so the
summary is reconstructable from the raw records."""
import json, os, re, sys

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    print(__doc__); sys.exit(0 if len(sys.argv) > 1 else 2)
out = sys.argv[1]
if not os.path.isdir(out): sys.exit(f"job-summary.py: no record dir {out}")
res = os.path.join(out, "RESULTS.txt")
order = []
for line in open(res, errors="replace") if os.path.exists(res) else []:
    if re.match(r"\S+ \S+: PREEMPTED", line):   # a preempted held block is re-run from the start; count only the re-run
        order = []
        continue
    m = re.match(r"\S+ cell (\S+) rc=(\d+)", line)
    if m:
        order.append((m.group(1), int(m.group(2)), "DIRTY" in line))

print(f"{out}\n")
print("cells:", " ".join(f"{n}{'' if rc == 0 else f'(rc={rc})'}{'(DIRTY)' if d else ''}" for n, rc, d in order))

arms, tests = {}, []
for name, rc, dirty in order:
    p = os.path.join(out, "cells", name, "stdout.log")
    if rc != 0 or not os.path.exists(p):
        continue
    arm = name.split(".")[0]
    for l in open(p, errors="replace"):
        if not l.startswith("{"):
            continue
        try:
            j = json.loads(l)
        except ValueError:
            continue
        t = (f"pp{j['n_prompt']}" if j.get("n_prompt") else f"tg{j.get('n_gen', 0)}") + f" d{j.get('n_depth', 0)}"
        if t not in tests:
            tests.append(t)
        arms.setdefault(arm, {}).setdefault(t, []).append((j["avg_ts"], dirty))

if arms:
    names = list(arms)
    base = next((a for a in names if not a.startswith('warm')), names[0])   # a discarded warm-up arm is never the reference
    print(f"\nbench t/s: mean over reps (spread = max-min), ratio vs '{base}'; * = includes a DIRTY rep")
    print(f"{'test':<16}" + "".join(f"  {a:>28}" for a in names))
    for t in tests:
        row = f"{t:<16}"
        b = arms[base].get(t)
        bm = sum(v for v, _ in b) / len(b) if b else None
        for a in names:
            v = arms[a].get(t)
            if not v:
                row += f"  {'-':>28}"
                continue
            m = sum(x for x, _ in v) / len(v)
            sp = max(x for x, _ in v) - min(x for x, _ in v)
            r = f" {m / bm:.3f}x" if bm and a != base and not a.startswith('warm') else ""
            cell = f"{m:.2f}±{sp:.2f}{'*' if any(d for _, d in v) else ''}{r} n{len(v)}"
            row += f"  {cell:>28}"
        print(row)

#!/usr/bin/env python3
"""therm-summary.py <therm file> [--absolute [--busy-only]] [--sclk-floor MHz | --sclk-floor-frac F]
                   [--power-floor W] [--tctl-max C] [--busy-pct P]

One line from a therm-sample.sh record: n, edge / tctl / power / sclk min/max/mean, and THROTTLE flags. Exit 2 when
any flag is raised, else 0.

Default rule (load-aware): among samples at >= --busy-pct (90) busy, the max clock must reach the floor (and the max
  power --power-floor, when set); a cell that is busy yet never boosts is the throttle signature. Clocks that track
  load rarely sit at the top level even at full power, so a rule on every sample flags every cell. No sample at
  --busy-pct = no clock verdict (the line says so).
--absolute: the older rule, sclk below the floor for > 10% of samples (--busy-only: only samples with gpu busy > 0);
  for GPUs that hold a fixed clock under load. --load-aware is accepted and is the default.
Always: Tctl above --tctl-max (default 90 C) is flagged.

Floor: --sclk-floor MHz, else --sclk-floor-frac (default 0.8, or 1.0 with --absolute) times the sclk_max_mhz
recorded in the file header. With neither available the clock rule is skipped and the line says so. Calibrate once:
run a known compute-bound cell on a cool machine and look at its busy-sample max clock and power.
"""
import argparse, os, re, statistics as st, sys

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('file')
ap.add_argument('--busy-only', action='store_true')
ap.add_argument('--load-aware', action='store_true')   # default now; kept for old callers
ap.add_argument('--absolute', action='store_true')
ap.add_argument('--sclk-floor', type=float, default=None)
ap.add_argument('--sclk-floor-frac', type=float, default=None)
ap.add_argument('--power-floor', type=float, default=0.0)
ap.add_argument('--tctl-max', type=float, default=90.0)
ap.add_argument('--busy-pct', type=float, default=90.0)
A = ap.parse_args()
f = A.file
if not os.path.isfile(f) or os.path.getsize(f) == 0:
    print('therm: no samples (cell produced no .therm file)'); sys.exit(0)
rows = []; smax = None
for l in open(f):
    if l.startswith('#'):
        m = re.search(r'sclk_max_mhz=(\d+)', l)
        if m: smax = float(m.group(1))
        continue
    if not l.strip(): continue
    p = l.split()
    try: rows.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4]), p[5], float(p[6]) if p[6] != '?' else -1))
    except Exception: pass
if not rows:
    print("therm: no samples"); sys.exit(0)
def mmm(v): return f"{min(v):.0f}/{max(v):.0f}/{st.mean(v):.0f}"
edge=[r[1] for r in rows]; tctl=[r[2] for r in rows]; pw=[r[3] for r in rows]; sclk=[r[4] for r in rows]
sel = [r for r in rows if r[6] > 0] if A.busy_only else rows
floor = A.sclk_floor
if floor is None and smax:
    floor = smax * (A.sclk_floor_frac if A.sclk_floor_frac is not None else (1.0 if A.absolute else 0.8))
flags = []; notes = []
if floor is None:
    notes.append('sclk rule skipped (no sclk_max_mhz in header; pass --sclk-floor)')
elif not A.absolute:
    hot = [r for r in rows if r[6] >= A.busy_pct]
    if not hot: notes.append(f'no sample at >= {A.busy_pct:g}% busy: no clock verdict')
    if hot and (max(r[4] for r in hot) < floor or (A.power_floor > 0 and max(r[3] for r in hot) < A.power_floor)):
        flags.append(f"THROTTLE busy>={A.busy_pct:g}% samples never boosted (max sclk {max(r[4] for r in hot):.0f} MHz < floor {floor:.0f}"
                     + (f" or max power {max(r[3] for r in hot):.0f} W < {A.power_floor:g}" if A.power_floor > 0 else "") + f", n={len(hot)})")
else:
    low = sum(1 for r in sel if r[4] < floor); frac = low / max(1, len(sel))
    if frac > 0.10: flags.append(f"THROTTLE sclk<{floor:.0f} {100*frac:.0f}% of {len(sel)} samples")
if max(tctl) > A.tctl_max: flags.append(f"THROTTLE tctl>{A.tctl_max:g} ({max(tctl):.1f} C)")
print(f"therm n={len(rows)} edge={mmm(edge)}C tctl={mmm(tctl)}C power={mmm(pw)}W sclk={mmm(sclk)}MHz"
      + (" " + "; ".join(flags) if flags else " ok") + ("" if not notes else " (" + "; ".join(notes) + ")"))
sys.exit(2 if flags else 0)

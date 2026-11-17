#!/usr/bin/env python3
# clk-summary.py <samples.tsv> [--busy-pct 50]: busy-gated (gpu_busy >= busy-pct) distributions from a clk-sample.py record: sclk mean,
# fclk / mclk / socclk level shares, vddgfx / vddnb mean, CPU all-core busy % mean, bench-process CPU % mean (100 = one core),
# power mean. Columns the device did not expose (na) are reported as na.
import collections, statistics, sys
if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
    sys.exit('usage: clk-summary.py <samples.tsv> [--busy-pct 50]')
gate = float(sys.argv[sys.argv.index('--busy-pct') + 1]) if '--busy-pct' in sys.argv else 50
R = [l.rstrip('\n').split('\t') for l in open(sys.argv[1]) if l.count('\t') == 10]
B = [r for r in R if r[3].isdigit() and int(r[3]) >= gate]
if not B: print('no busy samples'); sys.exit()
def share(i):
    c = collections.Counter(r[i] for r in B); return ' '.join(f"{k}:{100*v/len(B):.0f}%" for k, v in c.most_common(4))
def f(i):
    v = []
    for r in B:
        try: v.append(float(r[i]))
        except ValueError: pass
    return statistics.mean(v) if v else float('nan')
print(f"busy {len(B)}/{len(R)}: sclk {f(1):.0f} MHz, power {f(2):.1f} W | fclk {share(4)} | mclk {share(5)} | socclk {share(6)} | "
      f"vddgfx {f(7):.0f} mV vddnb {f(8):.0f} mV | cpu all-core {f(9):.0f}% | bench proc {f(10):.0f}% of a core")

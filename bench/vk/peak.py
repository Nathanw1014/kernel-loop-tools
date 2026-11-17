#!/usr/bin/env python3
"""peak.py: cooperative-matrix peak of this GPU, measured, not taken from a datasheet.

usage: peak.py <vkrun> <wmma_peak.spv | wmma_peak_i8.spv> <tag> [--reps N] [--iters I1,I2] [--groups G1,G2] [--subgroups 32,64]

Runs vkrun on a register-chain peak shader over a grid of configs (workgroups, wave size, accumulator chains NACC, co-issued
f32 FMAs NVALU) and prints T(FL)OP/s per config. Each config is timed at two ITER values; the rate comes from the slope
between them, so the fixed submit + wait cost cancels (a best-of wall time at one size understates the peak). The "long"
column is the single-size rate, for comparison.
One 16x16x16 MMA = 16*16*16*2 = 8192 flops (int ops for the i8 shader); 256 threads per workgroup = 8 wave32 or 4 wave64.

Guards against: quoting a kernel's "% of peak" against a peak that was never measured on this driver / clock, and against
reading the submit overhead as compute time.
"""
import argparse, re, subprocess, sys


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print(__doc__); return 0
    ap = argparse.ArgumentParser(usage=__doc__)
    ap.add_argument("vkrun"); ap.add_argument("spv"); ap.add_argument("tag")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--iters", default="4096,16384")
    ap.add_argument("--groups", default="320,640")
    ap.add_argument("--subgroups", default="32,64")
    a = ap.parse_args()
    i1, i2 = (int(x) for x in a.iters.split(","))
    gs = [int(x) for x in a.groups.split(",")]

    def run(g, sg, it, nv, na):
        out = subprocess.run([a.vkrun, a.spv, str(g), str(a.reps), str(sg), f"{it},{nv},{na}"], capture_output=True, text=True)
        m = re.search(r"best=([0-9.]+) ms", out.stdout)
        if not m:
            print("FAIL", out.stdout, out.stderr); sys.exit(1)
        return float(m.group(1))

    cfgs = []
    for sg in (int(x) for x in a.subgroups.split(",")):
        for g in gs:
            cfgs.append((g, sg, 0, 4))
        for na in (8, 2, 1):
            cfgs.append((gs[0], sg, 0, na))
        for nv in (8, 16, 32, 64, 128):
            cfgs.append((gs[0], sg, nv, 8))
    best = {}
    for g, sg, nv, na in cfgs:
        waves = 256 // sg
        t1 = run(g, sg, i1, nv, na); t2 = run(g, sg, i2, nv, na)
        ops = lambda it: g * waves * it * 8 * 8192
        if t2 <= t1:
            print(f"[peak {a.tag}] wave{sg} groups={g} NVALU={nv} NACC={na}: t({i2}) <= t({i1}), no slope (raise --iters)")
            continue
        slope = (ops(i2) - ops(i1)) / ((t2 - t1) * 1e-3) / 1e12
        long_ = ops(i2) / (t2 * 1e-3) / 1e12
        best[sg] = max(best.get(sg, 0), slope)
        print(f"[peak {a.tag}] wave{sg} groups={g} NVALU={nv} NACC={na} ms@{i1}={t1:.3f} ms@{i2}={t2:.3f} "
              f"slope={slope:.1f} T/s long={long_:.1f} T/s", flush=True)
    print(f"[peak {a.tag}] BEST slope " + " ".join(f"wave{k}={v:.1f}" for k, v in sorted(best.items())) + " T(FL)OPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

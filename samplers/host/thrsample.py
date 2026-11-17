#!/usr/bin/env python3
"""thrsample.py: per-thread CPU time of a process tree, sampled with the GPU's busy %, to see which host threads run while
the GPU works (and which ones the GPU waits for).

usage: thrsample.py <pid> <out.tsv> [period_ms=20]     sample until <pid> exits (run it in the background next to the job)
       thrsample.py --summary <out.tsv> [--busy 90]    CPU-seconds per thread name: total, and while gpu_busy >= busy

Samples every thread of <pid> and of all its descendants (an env / timeout wrapper's child is included), reading
/proc/<pid>/task/<tid>/stat (utime + stime) and gpu_busy_percent of the amdgpu card ($KLT_GPU_CARD, else the first card
that has it; 0 when there is none). TSV columns: t gpu_busy tid comm ticks.
Guards against: assuming decode is GPU-bound when a host thread (graph build, sampling, a busy-waiting submit thread)
eats a core per token; and against blaming "the CPU" without naming the thread.
Read-only: it sends no signals and never touches the sampled process. Overhead: one /proc walk per period.
"""
import glob, os, sys, time


def gpu_busy_path():
    card = os.environ.get("KLT_GPU_CARD")
    if card:
        return os.path.join(card, "gpu_busy_percent")
    for d in sorted(glob.glob("/sys/class/drm/card*/device")):
        if os.access(d + "/gpu_busy_percent", os.R_OK):
            return d + "/gpu_busy_percent"
    return None


def summary(path, busy_min):
    last = {}; busy_cpu = {}; all_cpu = {}; hz = os.sysconf('SC_CLK_TCK')
    for l in open(path):
        t, b, tid, comm, ticks = l.rstrip('\n').split('\t'); b = int(b); ticks = int(ticks)
        if tid in last:
            d = (ticks - last[tid]) / hz
            all_cpu[comm] = all_cpu.get(comm, 0) + d
            if b >= busy_min:
                busy_cpu[comm] = busy_cpu.get(comm, 0) + d
        last[tid] = ticks
    for c in sorted(all_cpu, key=lambda c: -busy_cpu.get(c, 0)):
        print(f"  {c:20s} cpu {all_cpu[c]:7.2f} s total, {busy_cpu.get(c, 0):7.2f} s while gpu_busy>={busy_min}")
    return 0


def sample(pid, out, per):
    gb = gpu_busy_path()
    with open(out, 'w') as f:
        while os.path.exists(f'/proc/{pid}'):
            t = time.time()
            try:
                b = open(gb).read().strip() if gb else '0'
            except OSError:
                b = '0'
            procs = [pid]; i = 0
            while i < len(procs):
                try:
                    procs += open(f'/proc/{procs[i]}/task/{procs[i]}/children').read().split()
                except OSError:
                    pass
                i += 1
            tids = []
            for p_ in procs:
                try:
                    tids += [(p_, t_) for t_ in os.listdir(f'/proc/{p_}/task')]
                except OSError:
                    pass
            if not tids:
                break
            for p_, tid in tids:
                try:
                    s = open(f'/proc/{p_}/task/{tid}/stat').read()
                except OSError:
                    continue
                comm = s[s.index('(') + 1:s.rindex(')')]; fl = s[s.rindex(')') + 2:].split()
                f.write(f"{t:.3f}\t{b}\t{tid}\t{comm}\t{int(fl[11]) + int(fl[12])}\n")
            time.sleep(per)
    return 0


def main():
    a = sys.argv[1:]
    if not a or a[0] in ('-h', '--help'):
        print(__doc__); return 0 if a else 2
    if a[0] == '--summary':
        busy = 90
        if '--busy' in a:
            i = a.index('--busy'); busy = int(a[i + 1]); del a[i:i + 2]
        if len(a) != 2:
            print("usage: thrsample.py --summary <out.tsv> [--busy 90]", file=sys.stderr); return 2
        return summary(a[1], busy)
    if len(a) < 2 or not a[0].isdigit():
        print("usage: thrsample.py <pid> <out.tsv> [period_ms=20]", file=sys.stderr); return 2
    return sample(a[0], a[1], (int(a[2]) if len(a) > 2 else 20) / 1000)


if __name__ == '__main__':
    sys.exit(main())

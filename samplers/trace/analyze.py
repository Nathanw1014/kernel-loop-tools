#!/usr/bin/env python3
"""analyze.py <trace.gz> [--window LO HI] [--gap-ms N] [--comm GLOB] [--timeline S] [--line] [--top K] [--report F] [--auto]

Summarise a trace-run trace. Times are seconds relative to the first event line (t0). Sections print only when the
trace has the events they need ("no sched events" etc.):
  gpu      compute/gfx job timeline (non-sdma rings): busy %, idle gaps > --gap-ms with the other events inside each gap
  sched    drm_sched done -> next job run on a backlogged compute ring, p50/p90/p99 + sum
  cs       amdgpu_cs_ioctl inter-arrival p50/p90/p99, GPU idle split into cpu-late vs queued-not-run, cs -> run
  fwait    first fence wait after a token's last cs: wait_start - last job done (> 0 = CPU reached the wait late)
  boundary per-token (cs bursts split at 2 ms gaps, COMM threads): last job done -> next token's first cs, token period
  dsleep   D-state sleeps of COMM threads (blocking I/O, e.g. file reads on the decode path): n, p50/p90/p99, sum
  block    block_rq_issue -> block_rq_complete latency per device
  syscall  enter -> exit latency per traced syscall of COMM threads (ioctl, pread64, preadv2, io_uring_enter)
  cpu      <trace>.cpu.tsv sampler (if present): mean freq of busy CPUs vs all
--line prints one summary line (for a job's results line) instead of the report; --report F writes the report to F as
well; --auto starts the window at the first compute-ring job (skips the model load; whole-run numbers still mix prefill
and decode: use --window over the phase you care about).
"""
import gzip, re, sys, bisect, collections, fnmatch, os, argparse
from array import array

ap = argparse.ArgumentParser()
ap.add_argument('trace'); ap.add_argument('--window', nargs=2, type=float, default=None)
ap.add_argument('--gap-ms', type=float, default=20.0); ap.add_argument('--comm', default='llama-*')
ap.add_argument('--timeline', type=float, default=0); ap.add_argument('--line', action='store_true')
ap.add_argument('--top', type=int, default=5)
ap.add_argument('--report', default=None, help='also write the full report to this file')
ap.add_argument('--auto', action='store_true', help='window = first compute-ring job run .. end (skips model load)')
A = ap.parse_args()
LO, HI = (A.window if A.window else (-1e18, 1e18))
if A.auto and not A.window:   # pre-scan: t0 and the first compute-ring job
    _h = re.compile(r'^\s*.+?-\d+\s+\[\d+\]\s+\S+\s+([\d.]+): (.*)$'); _t0 = None
    with (gzip.open if A.trace.endswith('.gz') else open)(A.trace, 'rt', errors='replace') as fh:
        for ln in fh:
            if ln[0] == '#': continue
            m = _h.match(ln)
            if not m: continue
            if _t0 is None: _t0 = float(m.group(1))
            if m.group(2).startswith('drm_sched_job_run') and 'ring=comp' in m.group(2): LO = float(m.group(1)) - _t0; break
cm = re.compile(fnmatch.translate(A.comm)).match

head = re.compile(r'^\s*(.+?)-(\d+)\s+\[(\d+)\]\s+\S+\s+([\d.]+): (.*)$')
rfence = re.compile(r'fence=([0-9:]+)'); rring = re.compile(r'ring=([^,]+)'); rjc = re.compile(r'job count:(\d+)')
rsw = re.compile(r'prev_comm=(.*?) prev_pid=(\d+) prev_prio=\S+ prev_state=(\S+) ==> next_comm=(.*?) next_pid=(\d+)')
rwk = re.compile(r'comm=(.*?) pid=(\d+)')
rblk_i = re.compile(r'^(\d+,\d+) (\S+) (\d+) \(.*?\) (\d+) \+ (\d+)')
rblk_c = re.compile(r'^(\d+,\d+) (\S+) \(.*?\) (\d+) \+ (\d+)')
rsys_en = re.compile(r'^(sys_\w+)\(')
rsys_ex = re.compile(r'^(sys_\w+) -> (\S+)')
entries = None
t0 = None
# per-section state
names = {}; ev_t = array('d'); ev_n = array('H'); ncount = collections.Counter()
run_t = {}; ring_of = {}; jobs = []           # (run, done, ring)
cs_all = []                                  # (t, fence, comm_match, tid)
fws = []; fwe = []                           # COMM dma_fence_wait_start / _end times
done_t = {}                                  # fence -> done time
sched_d = []; lastdone_ring = {}             # sched section
sw_out = {}; dsl = []; ssl = []; wk = {}; dwake = []; nsw = 0
blk_iss = {}; blk_lat = collections.defaultdict(list); blk_bytes = collections.Counter()
sys_st = {}; sys_lat = collections.defaultdict(list)
CHAT = {'drm_sched_job_run', 'drm_sched_job_done', 'dma_fence_wait_start', 'dma_fence_wait_end', 'amdgpu_cs_ioctl', 'sched_switch',
        'sched_wakeup', 'block_rq_issue', 'block_rq_complete', 'tracing_mark_write', 'workqueue_execute_start', 'workqueue_execute_end'}

def nid(n):
    i = names.get(n)
    if i is None: i = names[n] = len(names)
    return i

op = gzip.open if A.trace.endswith('.gz') else open
with op(A.trace, 'rt', errors='replace') as fh:
    for ln in fh:
        if ln[0] == '#':
            if entries is None and 'entries-in-buffer' in ln:
                m = re.search(r'(\d+)/(\d+)', ln); entries = (int(m.group(1)), int(m.group(2)))
            continue
        m = head.match(ln)
        if not m: continue
        comm, pid, cpu, t, rest = m.group(1), m.group(2), m.group(3), float(m.group(4)), m.group(5)
        if t0 is None: t0 = t
        r = t - t0
        if rest.startswith('sys_'):
            me = rsys_en.match(rest)
            if me and rest[len(me.group(1))] == '(':
                if LO <= r <= HI: sys_st[pid] = (t, me.group(1))
                continue
            mx = rsys_ex.match(rest)
            if mx:
                s = sys_st.pop(pid, None)
                if s and s[1] == mx.group(1) and LO <= r <= HI and cm(comm): sys_lat[s[1]].append(t - s[0])
                continue
            continue
        k = rest.find(': ')
        if k < 0: continue
        n = rest[:k]; a = rest[k + 2:]
        ncount[n] += 1
        ev_t.append(r); ev_n.append(nid(n))
        if n == 'drm_sched_job_run':
            fe = rfence.search(a).group(1); rg = rring.search(a).group(1); ring_of[fe] = rg; run_t[fe] = t
            mj = rjc.search(a)
            if LO <= r <= HI and rg.startswith('comp') and mj and int(mj.group(1)) > 0 and rg in lastdone_ring and t - lastdone_ring[rg] < 0.02:
                sched_d.append(t - lastdone_ring[rg])
        elif n == 'drm_sched_job_done':
            fe = rfence.search(a).group(1); done_t[fe] = t
            rg = ring_of.get(fe)
            if rg: lastdone_ring[rg] = t
            if fe in run_t: jobs.append((run_t.pop(fe), t, rg or '?', fe))
        elif n == 'amdgpu_cs_ioctl':
            if LO - 1 <= r <= HI + 1: cs_all.append((t, rfence.search(a).group(1), bool(cm(comm)), pid))
        elif n == 'dma_fence_wait_start' or n == 'dma_fence_wait_end':
            if LO - 1 <= r <= HI + 1 and cm(comm): (fws if n[-5:] == 'start' else fwe).append(t)
        elif n == 'sched_switch':
            nsw += 1
            ms = rsw.search(a)
            if not ms or not (LO <= r <= HI): continue
            if cm(ms.group(1)): sw_out[ms.group(2)] = (t, ms.group(3))
            np_ = ms.group(5)
            if np_ in sw_out:
                t1, st = sw_out.pop(np_)
                (dsl if st == 'D' else ssl).append(t - t1)
                if st == 'D' and np_ in wk: dwake.append(wk.pop(np_) - t1)
        elif n == 'sched_wakeup':
            mw = rwk.search(a)
            if mw and cm(mw.group(1)) and LO <= r <= HI: wk[mw.group(2)] = t
        elif n == 'block_rq_issue':
            mb = rblk_i.match(a)
            if mb: blk_iss[(mb.group(1), mb.group(4))] = (t, int(mb.group(3)), mb.group(2))
        elif n == 'block_rq_complete':
            mb = rblk_c.match(a)
            if mb:
                s = blk_iss.pop((mb.group(1), mb.group(3)), None)
                if s and LO <= r <= HI:
                    key = mb.group(1) + ' ' + ('R' if 'R' in s[2] else 'W' if 'W' in s[2] else s[2])
                    blk_lat[key].append(t - s[0]); blk_bytes[key] += s[1]

def q(x, p): return sorted(x)[min(len(x) - 1, int(len(x) * p))]
def pct(x, scale=1e3, fmt='%.3f'):
    if not x: return 'n=0'
    s = sorted(x); g = lambda p: s[min(len(s) - 1, int(len(s) * p))] * scale
    return ('n=%d p50 ' + fmt + ' p90 ' + fmt + ' p99 ' + fmt) % (len(s), g(.5), g(.9), g(.99))

out = []; line = []
if t0 is None:
    if A.report: open(A.report, 'w').write('trace: no events\n')
    print('trace no-events' if A.line else 'trace: no events'); sys.exit(0)
last = ev_t[-1] if len(ev_t) else 0.0
wlo, whi = max(LO, 0.0), min(HI, last)
out.append(f'trace {os.path.basename(A.trace)}: events {len(ev_t)} span {last:.2f}s window {wlo:.2f}-{whi:.2f}s comm {A.comm}')
if entries:
    loss = 1 - entries[0] / entries[1] if entries[1] else 0
    out.append(f'  ring buffer: entries-in-buffer/written {entries[0]}/{entries[1]} (lost {loss*100:.1f}%, oldest first on busy CPUs)')
    line.append(f'loss={loss*100:.0f}%')

# gpu: union of non-sdma job intervals inside the window
gfx = sorted((a, b, rg) for a, b, rg, fe in jobs if not rg.startswith('sdma') and LO <= a - t0 <= HI)
if gfx:
    busy = 0.0; gaps = []; s0, e0 = gfx[0][0], gfx[0][1]
    for a, b, rg in gfx[1:]:
        if a > e0:
            busy += e0 - s0
            if a - e0 > A.gap_ms / 1e3: gaps.append((a - e0, e0, a))
            s0, e0 = a, b
        else: e0 = max(e0, b)
    busy += e0 - s0; span = gfx[-1][1] - gfx[0][0]
    out.append(f'gpu: {len(gfx)} compute/gfx jobs over {span:.2f}s, busy {busy/span*100:.1f}% | idle gaps > {A.gap_ms:g} ms: {len(gaps)} sum {sum(g for g,_,_ in gaps):.3f}s')
    line.append(f'gpu_busy={busy/span*100:.0f}% gaps>{A.gap_ms:g}ms={len(gaps)}/{sum(g for g,_,_ in gaps):.2f}s')
    inv = {v: k for k, v in names.items()}
    for g, s, e in sorted(gaps, reverse=True)[:A.top]:
        i = bisect.bisect_left(ev_t, s - t0 - 0.05); j = bisect.bisect_right(ev_t, e - t0)
        c = collections.Counter(inv[ev_n[x]] for x in range(i, j) if inv[ev_n[x]] not in CHAT)
        out.append(f'  gap {g*1e3:8.1f} ms at {s-t0:8.2f}s: ' + (', '.join(f'{k} {v}' for k, v in c.most_common(6)) or 'no other events'))
else:
    out.append('gpu: no drm_sched job events in window')

if sched_d:
    out.append(f'sched: done -> next run (backlogged comp ring, < 20 ms) {pct(sched_d, 1e6, "%.0f")} us, sum {sum(sched_d):.3f}s')
    line.append(f'sched_p99={q(sched_d,.99)*1e6:.0f}us')

# cs: all cs-ioctls in [LO,HI]; boundary: COMM cs in [LO-1,HI+1], tokens whose first cs is in [LO,HI]
csw = [c for c in cs_all if LO <= c[0] - t0 <= HI]
if len(csw) > 1:
    gap = [b[0] - a[0] for a, b in zip(csw, csw[1:])]
    runs = {}; q2r = []
    for a, b, rg, fe in jobs: runs[fe] = (a, b, rg)
    jl = []
    for t, fe, _, _ in csw:
        j = runs.get(fe)
        if j and j[2].startswith('comp'): q2r.append(j[0] - t); jl.append((j[0], j[1], t))
    jl.sort(); end = None; late = dep = 0.0
    for a, b, c in jl:
        if end is not None and a > end:
            if c > end: late += c - end; dep += a - c
            else: dep += a - end
        end = b if end is None else max(end, b)
    out.append(f'cs: {len(csw)} cs-ioctls | inter-arrival {pct(gap, 1e6, "%.0f")} us | cs->run {pct(q2r, 1e6, "%.0f")} us | gpu idle cpu-late {late:.3f}s queued-not-run {dep:.3f}s')
    line.append(f'csgap p50/p90={q(gap,.5)*1e6:.0f}/{q(gap,.9)*1e6:.0f}us cpu_late={late:.2f}s')
    cs = [c for c in cs_all if c[2]]
    if len(cs) > 2:
        toks = []; cur = [cs[0]]
        for p_, c in zip(cs, cs[1:]):
            if c[0] - p_[0] > 0.002: toks.append(cur); cur = [c]
            else: cur.append(c)
        toks.append(cur); bd = []; per = []
        for a, b in zip(toks, toks[1:]):
            if not (LO <= a[0][0] - t0 <= HI): continue
            ld = max(done_t.get(x[1], 0) for x in a)
            if not ld: continue
            bd.append(b[0][0] - ld); per.append(b[0][0] - a[0][0])
        # first blocking fence wait after a token's last cs: wait_start - last done (> 0 = CPU late to wait)
        fw = []
        for a, b in zip(toks, toks[1:]):
            if not (LO <= a[0][0] - t0 <= HI): continue
            ld = max(done_t.get(x[1], 0) for x in a); nc = b[0][0]
            i = bisect.bisect_left(fws, a[-1][0]); j = bisect.bisect_left(fws, nc)
            if i >= j or not ld: continue
            st = fws[i]; e = bisect.bisect_left(fwe, st)
            if e >= len(fwe) or fwe[e] > nc: continue
            fw.append(st - ld)
        if fw:
            out.append(f'fwait: first fence wait after a token: wait_start - last done {pct(fw)} ms (> 0 = CPU reached the wait after the GPU finished)')
        if bd:
            out.append(f'boundary: {len(bd)} tokens | last done -> next first cs {pct(bd)} ms | period p50 {q(per,.5)*1e3:.2f} ms')
            line.append(f'boundary p50/p90={q(bd,.5)*1e3:.2f}/{q(bd,.9)*1e3:.2f}ms')
else:
    out.append('cs: no amdgpu_cs_ioctl events in window')

if nsw:
    if dsl:
        out.append(f'dsleep: D-state sleeps of {A.comm} {pct(dsl)} ms sum {sum(dsl):.2f}s | D -> wakeup {pct(dwake)} ms | S sleeps n={len(ssl)} p50 {q(ssl,.5)*1e3 if ssl else 0:.3f} ms')
        line.append(f'Dsleep p50/p99={q(dsl,.5)*1e3:.2f}/{q(dsl,.99)*1e3:.1f}ms sum={sum(dsl):.2f}s')
    else:
        out.append(f'dsleep: no D-state sleeps of {A.comm} threads')
else:
    out.append('dsleep: no sched events (set wake / io / all records them)')

if blk_lat:
    for k in sorted(blk_lat, key=lambda k: -len(blk_lat[k])):
        out.append(f'block {k}: {pct(blk_lat[k])} ms, {blk_bytes[k]/2**20:.0f} MiB')
    k = max(blk_lat, key=lambda k: len(blk_lat[k]))
    line.append(f'blk[{k}] p50/p99={q(blk_lat[k],.5)*1e3:.2f}/{q(blk_lat[k],.99)*1e3:.1f}ms n={len(blk_lat[k])}')
elif ncount.get('block_rq_issue'):
    out.append('block: issue events but no completions matched in window')
for s in sorted(sys_lat, key=lambda s: -len(sys_lat[s])):
    out.append(f'syscall {s}: {pct(sys_lat[s])} ms sum {sum(sys_lat[s]):.2f}s')
    if s != 'sys_ioctl': line.append(f'{s[4:]} p50/p99={q(sys_lat[s],.5)*1e3:.2f}/{q(sys_lat[s],.99)*1e3:.1f}ms n={len(sys_lat[s])}')

cpu_tsv = re.sub(r'\.gz$', '', A.trace) + '.cpu.tsv'
if os.path.exists(cpu_tsv):
    fb = []; fa = []
    tstart = None
    if A.window and entries is not None:
        pass
    for l in open(cpu_tsv):
        f = l.rstrip('\n').split('\t')
        if len(f) < 3: continue
        fr = [int(x) for x in f[1].split(',') if x]; bz = [float(x) for x in f[2].split(',') if x]
        fa += fr; fb += [x for x, b in zip(fr, bz) if b >= 0.5]
    if fa:
        out.append(f'cpu: sampler {os.path.basename(cpu_tsv)} mean freq busy(>=50%) CPUs {sum(fb)/max(1,len(fb))/1e6:.2f} GHz (n={len(fb)}), all {sum(fa)/len(fa)/1e6:.2f} GHz (whole run, not windowed)')

if A.timeline and gfx:
    B = A.timeline; busy = collections.Counter(); csn = collections.Counter(); end = -1
    for a, b, rg in gfx:
        a = max(a, end)
        if b <= a: continue
        x = a
        while x < b:
            k = int((x - t0) / B); nx = min(b, t0 + (k + 1) * B); busy[k] += nx - x; x = nx
        end = b
    for c in csw: csn[int((c[0] - t0) / B)] += 1
    for k in sorted(set(busy) | set(csn)):
        out.append(f'  t {k*B:7.1f}s busy {busy[k]/B*100:5.1f}% cs {csn[k]}')

if A.report:
    with open(A.report, 'w') as fo: fo.write('\n'.join(out) + '\n')
if A.line:
    print('trace ' + ' '.join(line))
else:
    print('\n'.join(out))

#!/usr/bin/env python3
"""isa-stats.py: per-pipeline ACO instruction counts for every ggml-vulkan pipeline a run compiles, with names. [AMD/RADV]

usage: RADV_DEBUG=asm,nocache GGML_VK_PIPELINE_STATS=<name-substring> <llama-bench / test-backend-ops ...> 2> stderr.log
       isa-stats.py <stderr.log> [name-regex]

GGML_VK_PIPELINE_STATS (upstream ggml-vulkan) prints "pipeline stats for <name>" plus VGPRs / LDS / waves per SIMD / code
size after each pipeline compile whose name contains the substring ("_" matches nearly all); RADV_DEBUG=asm prints the ISA
just before. Each ISA dump is attributed to the next stats line; the two must alternate 1:1, and the first output line
reports "N without stats" (dumps whose pipeline the stats filter skipped: expected with a filter; or a cache hit: keep
nocache) and "attribution errors" (a stats line with no dump of its own: two compiles interleaved; must be 0).
For each pipeline that has v_wmma: whole-shader counts, counts of the loop holding the most v_wmma (header .. last back-edge
block), VALU = v_* other than v_wmma, and a barrier check: the instruction directly in front of each s_barrier and any
s_waitcnt vmcnt within the 3 before it (static; a barrier at a block start shows its own block only).
For one pipeline outside ggml use bench/vk/vkasm + isa-histo.py.
"""
import re, sys, collections
if len(sys.argv) < 2 or sys.argv[1] in ('-h', '--help'):
    print(__doc__); sys.exit(0 if len(sys.argv) > 1 else 2)
log = sys.argv[1]; flt = re.compile(sys.argv[2]) if len(sys.argv) > 2 else None
segs = []; cur = None; stats = None; mism = 0; unnamed = 0
for l in open(log, errors='replace'):
    if l.startswith('Compute Shader'):
        if cur is not None and cur.get('name') is None: unnamed += 1
        cur = {'lines': [], 'name': None, 'st': {}}; segs.append(cur); stats = None; continue
    m = re.match(r'ggml_vulkan: pipeline stats for (\S+):', l)
    if m:
        if cur is None or cur['name'] is not None: mism += 1; stats = None; continue
        cur['name'] = m.group(1); stats = cur['st']; continue
    m = re.match(r'ggml_vulkan:   ([A-Za-z ]+): (\d+)', l)
    if m and stats is not None: stats[m.group(1)] = int(m.group(2)); continue
    if cur is not None and cur['name'] is None: cur['lines'].append(l)
if cur is not None and cur.get('name') is None: unnamed += 1
print(f"# {log}: {len(segs)} dumps, {sum(1 for s in segs if s['name'])} named, {unnamed} without stats "
      f"(filtered out by GGML_VK_PIPELINE_STATS, or a cache hit), attribution errors {mism}")
opre = re.compile(r'=\s+([a-z][a-z0-9_]+)\b')
opre2 = re.compile(r'\s+([sv]_[a-z0-9_]+|ds_[a-z0-9_]+|buffer_[a-z0-9_]+|global_[a-z0-9_]+)\b')
def op(l):
    m = opre.search(l) or opre2.match(l)
    return m.group(1) if m else None
def cnt(c):
    w = sum(v for k, v in c.items() if 'wmma' in k)
    valu = sum(v for k, v in c.items() if k.startswith('v_') and 'wmma' not in k)
    dsr = sum(v for k, v in c.items() if k.startswith('ds_read') or k.startswith('ds_load'))
    dsw = sum(v for k, v in c.items() if k.startswith('ds_write') or k.startswith('ds_store'))
    bl = sum(v for k, v in c.items() if k.startswith('buffer_load') or k.startswith('global_load'))
    return w, valu, dsr, dsw, bl
for s in segs:
    if not s['name'] or (flt and not flt.search(s['name'])): continue
    order = []; bbs = {}; kinds = {}; b = None; prevw = []; barr = []
    for l in s['lines']:
        t = l.strip()
        m = re.match(r'BB(\d+)$', t)
        if m: b = int(m.group(1)); order.append(b); bbs[b] = collections.Counter(); kinds[b] = ''; prevw = []; continue
        if b is None: continue
        if t.startswith('/* logical'): kinds[b] = t; continue
        o = op(l)
        if not o: continue
        if o == 's_barrier': barr.append((b, list(prevw)))
        # the 3 instructions in front of a barrier (s_delay_alu / s_clause are not instructions that wait)
        if not o.startswith('s_delay_alu') and o != 's_clause':
            prevw.append(t if o.startswith('s_waitcnt') else o); prevw = prevw[-3:]
        bbs[b][o] += 1
    tot = collections.Counter()
    for c in bbs.values(): tot.update(c)
    w, valu, dsr, dsw, bl = cnt(tot)
    if w == 0: continue
    best = None
    for h in order:
        if 'loop-header' not in kinds[h]: continue
        m = re.search(r'logical preds: ([^/]*)/', kinds[h]); preds = [int(x) for x in re.findall(r'BB(\d+)', m.group(1))] if m else []
        back = [p for p in preds if p >= h]   # >=: a single-block loop is its own back edge
        if not back: continue
        body = [x for x in order if h <= x <= max(back)]
        c = collections.Counter()
        for x in body: c.update(bbs[x])
        lw = cnt(c)[0]
        if best is None or lw > best[0]: best = (lw, h, body, c)
    st = s['st']
    print(f"\n{s['name']}: VGPRs {st.get('VGPRs')} LDS {st.get('LDS size')} sg/SIMD {st.get('Subgroups per SIMD')} code {st.get('Code size')}")
    print(f"  whole: wmma {w} ({','.join(k for k in tot if 'wmma' in k)}) valu {valu} ds_read {dsr} ds_write {dsw} buf_load {bl} waitcnt {tot['s_waitcnt']} barrier {tot['s_barrier']}  valu/wmma {valu/w:.2f}")
    if best:
        lw, h, body, c = best; _, lv, ldr, ldw, lbl = cnt(c)
        copies = c['v_mov_b16'] + c['v_mov_b32'] + c['v_swap_b32'] + c['v_swap_b16'] + 2 * c['v_dual_mov_b32']
        dual = sum(v for k, v in c.items() if k.startswith('v_dual_'))
        inb = [x for x in barr if x[0] in body]
        print(f"  loop BB{h}..BB{body[-1]}: wmma {lw} valu {lv} (copies {copies}, v_dual {dual}) ds_read {ldr} ds_write {ldw} buf_load {lbl} waitcnt {c['s_waitcnt']} barrier {c['s_barrier']}  valu/wmma {lv/lw if lw else 0:.2f}  lds_read/wmma {ldr/lw if lw else 0:.2f}")
        print("    top valu: " + ", ".join([f"{k}:{v}" for k, v in c.most_common(60) if k.startswith("v_") and "wmma" not in k][:14]))
    pre = collections.Counter(x[1][-1] if x[1] else '(block start)' for x in barr)
    vm1 = sum(1 for x in barr if x[1] and 'vmcnt' in x[1][-1])
    vm3 = sum(1 for x in barr if any('vmcnt' in y for y in x[1]))
    print(f"  barriers {len(barr)}: s_waitcnt vmcnt directly before {vm1}, within the 3 instructions before {vm3}; instruction directly before: " + "; ".join(f"{k} x{v}" for k, v in pre.items()))

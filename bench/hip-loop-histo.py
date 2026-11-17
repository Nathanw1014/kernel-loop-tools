#!/usr/bin/env python3
"""hip-loop-histo.py <kernel.s> <symbol>: loop histogram for an AMDGPU assembly listing (clang -S / hipcc -S, or
--save-temps). Finds the innermost-backedge loop with the most v_wmma instructions in the named kernel (ties: the longest
loop), prints its instruction counts by mnemonic and class, plus the kernel's resource metadata from the .amdhsa_kernel block
(LDS = group_segment_fixed_size, VGPR / SGPR, scratch, wave32, WGP mode). For a non-WMMA kernel it reports the longest loop."""
import re, sys, collections

if len(sys.argv) != 3 or sys.argv[1] in ('-h', '--help'): print(__doc__); sys.exit(2)
path, sym = sys.argv[1], sys.argv[2]
lines = open(path).read().split('\n')
start = next((i for i, l in enumerate(lines) if l.startswith(sym + ':')), None)
if start is None: sys.exit(f'hip-loop-histo: symbol {sym} not found in {path}')
end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith('.Lfunc_end')), len(lines))
body = lines[start:end]
labels = {}
for i, l in enumerate(body):
    m = re.match(r'^(\.LBB[\d_]+):', l)
    if m: labels[m.group(1)] = i
best = None
for i, l in enumerate(body):
    m = re.match(r'\s+s_(cbranch_\w+|branch)\s+(\.LBB[\d_]+)', l)
    if m and m.group(2) in labels and labels[m.group(2)] < i:
        s = labels[m.group(2)]
        c = collections.Counter()
        for x in body[s:i + 1]:
            mm = re.match(r'\s+([a-z][a-z0-9_]+)', x)
            if mm and not x.strip().startswith(';') and not x.strip().startswith('.'): c[mm.group(1)] += 1
        w = sum(v for k, v in c.items() if 'wmma' in k)
        if best is None or w > best[0] or (w == best[0] and i - s > best[3]): best = (w, s, i, i - s, c)
if best is None:
    print(f'no backward branch (loop) found in {sym}')
else:
    w, s, e, _, c = best
    t = sum(c.values())
    g = lambda p: sum(v for k, v in c.items() if k.startswith(p))
    print(f"LOOP lines {s}..{e}: total {t}  wmma {w}  valu {g('v_')}  ds_load {g('ds_load') + g('ds_read')}  ds_store {g('ds_store') + g('ds_write')}  "
          f"global_load {g('global_load') + g('buffer_load')}  s_waitcnt {g('s_waitcnt')}  barrier {c['s_barrier']}")
    print("   " + ", ".join(f"{k}:{v}" for k, v in c.most_common(40)))
# metadata from the .amdhsa_kernel block
k = next((i for i, l in enumerate(lines) if l.strip() == '.amdhsa_kernel ' + sym), None)
if k is not None:
    for l in lines[k:k + 80]:
        if re.search(r'group_segment_fixed_size|next_free_vgpr|next_free_sgpr|private_segment_fixed_size|wavefront_size32|workgroup_processor_mode', l):
            print('  ', l.strip())

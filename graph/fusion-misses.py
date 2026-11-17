#!/usr/bin/env python3
# needs: the same graph dump as fusion-census.py (post-<key>.txt and times-<key>.txt per graph; format in its header).
# fusion-misses.py <dumpdir> <key>: two adjacency misses and their node time in one graph (use a decode graph):
#   MUL_MAT -> RESHAPE -> ADD left unfused (no matmul + add epilogue fusion through the reshape), and
#   ADD -> RMS_NORM pairs that are adjacent vs split by other nodes (split extra = n_split x (mean split - mean adjacent)).
import importlib.util, pathlib, sys
if len(sys.argv) != 3 or sys.argv[1] in ('-h', '--help'):
    print('usage: fusion-misses.py <dumpdir> <key>', file=sys.stderr); sys.exit(2)
_spec = importlib.util.spec_from_file_location('fusion_census', pathlib.Path(__file__).resolve().parent / 'fusion-census.py')
_fc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_fc)
load, load_times = _fc.load, _fc.load_times
post, fused = load(f'{sys.argv[1]}/post-{sys.argv[2]}.txt'); t = load_times(f'{sys.argv[1]}/times-{sys.argv[2]}.txt')
byptr = {n['ptr']: n for n in post}
end, member = {}, set()
for h, f in fused.items():
    if f['label'] != '-' and f['enq']:
        for q in range(h, h + f['n']):
            end[q] = h + f['n'] - 1
            member.add(q)
via_add = [n for n in post if n['op'] == 'ADD' and n['i'] not in member and any(
    s in byptr and byptr[s]['op'] == 'RESHAPE' and byptr[s]['srcs'] and byptr.get(byptr[s]['srcs'][0], {}).get('op') == 'MUL_MAT'
    for s in n['srcs'])]
adj, split = [], []
for n in post:
    if n['op'] == 'RMS_NORM' and n['srcs'] and byptr.get(n['srcs'][0], {}).get('op') == 'ADD':
        a = byptr[n['srcs'][0]]
        (adj if n['i'] == end.get(a['i'], a['i']) + 1 else split).append(t.get(n['ptr'], 0))
tot = sum(t.values())
if not tot: sys.exit('fusion-misses: no times in the times file')
va = sum(t.get(n['ptr'], 0) for n in via_add)
ma = sum(adj) / len(adj) if adj else 0; ms = sum(split) / len(split) if split else 0
print(f'per-token node time {tot/1e3:.1f} us')
print(f'MUL_MAT -> RESHAPE -> ADD unfused: {len(via_add)} ADDs, {va/1e3:.1f} us ({100*va/tot:.2f}%)')
print(f'ADD -> RMS_NORM: {len(adj)} adjacent ({ma/1e3:.2f} us/norm), {len(split)} split ({ms/1e3:.2f} us/norm); split extra ~{len(split)*(ms-ma)/1e3:.1f} us ({100*len(split)*(ms-ma)/tot:.2f}%)')

#!/usr/bin/env python3
"""isa-histo.py: opcode histogram of an AMD ACO shader dump (RADV_DEBUG=asm, the print_program format). [AMD/RADV]

usage: isa-histo.py <dump.txt> [--top N]          whole dump: totals per class + the N most frequent opcodes (default 14)
       isa-histo.py <dump.txt> --loop [--top N]   only the loop whose body holds the most v_wmma (the GEMM main loop)

Classes: total, wmma, VALU (v_*), LDS reads / writes (ds_read*/ds_load*, ds_write*/ds_store*), buffer/global loads,
register copies (v_mov_b16/b32, v_swap_*, 2x v_dual_mov_b32), address arithmetic, conversions, s_waitcnt.
The loop body runs from the loop-header block to its back-edge block (the largest logical predecessor of the header), so a
body split by branches is still counted whole.
Get a dump of one pipeline with bench/vk/vkasm (RADV_DEBUG=asm,nocache); for every pipeline ggml-vulkan builds, with names
attached, use isa-stats.py.
Guards against: reading a kernel's loop from source when the compiler emitted something else (spilled copies, extra
waitcnts, the wrong instruction mix). Static counts only: they say what is in the loop, not what it costs.
"""
import collections, re, sys

OPRE = re.compile(r'=\s+([a-z][a-z0-9_]+)\b')
OPRE2 = re.compile(r'\s+([sv]_[a-z0-9_]+|ds_[a-z0-9_]+|buffer_[a-z0-9_]+|global_[a-z0-9_]+)\b')
ADDR = ('v_add3_u32', 'v_add_lshl_u32', 'v_mul_lo_u32', 'v_lshl_add_u32', 'v_add_u32', 'v_lshlrev_b32', 'v_lshrrev_b32',
        'v_mad_u32_u24', 'v_mad_u64_u32')


def op(line):
    m = OPRE.search(line) or OPRE2.match(line)
    return m.group(1) if m else None


def classes(c):
    pre = lambda *p: sum(v for k, v in c.items() if k.startswith(p))
    return {
        'total': sum(c.values()),
        'wmma': sum(v for k, v in c.items() if 'wmma' in k),
        'valu': sum(v for k, v in c.items() if k.startswith('v_') and 'wmma' not in k),
        'ds_read': pre('ds_read', 'ds_load'), 'ds_write': pre('ds_write', 'ds_store'),
        'vmem_load': pre('buffer_load', 'global_load'),
        'copies': c['v_mov_b16'] + c['v_mov_b32'] + c['v_swap_b32'] + c['v_swap_b16'] + 2 * c['v_dual_mov_b32'],
        'addr': sum(c.get(k, 0) for k in ADDR),
        'cvt': sum(v for k, v in c.items() if 'cvt' in k),
        'waitcnt': c['s_waitcnt'],
    }


def main():
    a = sys.argv[1:]
    if not a or a[0] in ('-h', '--help'):
        print(__doc__); return 0 if a else 2
    top = 14
    if '--top' in a:
        i = a.index('--top'); top = int(a[i + 1]); del a[i:i + 2]
    loop = '--loop' in a
    a = [x for x in a if x != '--loop']
    order, bbs, kinds, cur, whole = [], {}, {}, None, collections.Counter()
    for l in open(a[0], errors='ignore'):
        m = re.match(r'BB(\d+)$', l.strip())
        if m:
            cur = int(m.group(1)); order.append(cur); bbs[cur] = collections.Counter(); kinds[cur] = ''; continue
        if cur is not None and l.startswith('/* logical'):
            kinds[cur] = l.strip(); continue
        o = op(l)
        if o:
            whole[o] += 1
            if cur is not None:
                bbs[cur][o] += 1
    if not whole:
        print(f"{a[0]}: no ACO instructions found (was it RADV_DEBUG=asm output, and was the pipeline cache off?)")
        return 1
    if not loop:
        cl = classes(whole)
        print('  '.join(f"{k} {v}" for k, v in cl.items()))
        for k, v in whole.most_common(top):
            print(f"  {v:5d} {k}")
        return 0
    best = None
    for h in [b for b in order if 'loop-header' in kinds[b]]:
        m = re.search(r'logical preds: ([^/]*)/', kinds[h])
        preds = [int(x) for x in re.findall(r'BB(\d+)', m.group(1))] if m else []
        back = [p for p in preds if p > h]
        if not back:
            continue
        body = [b for b in order if h <= b <= max(back)]
        c = collections.Counter()
        for b in body:
            c.update(bbs[b])
        w = classes(c)['wmma']
        if best is None or w > best[0]:
            best = (w, h, body, c)
    if best is None:
        print("no loop with a back edge found"); return 1
    w, h, body, c = best
    cl = classes(c)
    print(f"LOOP BB{h}..BB{body[-1]} ({len(body)} blocks): " + '  '.join(f"{k} {v}" for k, v in cl.items()))
    print("   " + ", ".join(f"{k}:{v}" for k, v in c.most_common(top)))
    return 0


if __name__ == '__main__':
    sys.exit(main())

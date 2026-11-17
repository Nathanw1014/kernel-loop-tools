#!/usr/bin/env python3
# needs: a ggml-vulkan patch that, per distinct graph, writes the node list before (pre-<key>.txt) and after (post-<key>.txt) graph_optimize with each fusion decision, plus per-node perf-logger times (times-<key>.txt), in the format below.
# fusion census: per producer -> consumer edge type, how many pairs were fused, adjacent but not fused, or split by other nodes.
# usage: fusion-census.py <dumpdir> [--key KEY] [--all]
#   KEY = the file-name key of one graph (default: every post-*.txt in the dir); --all = every edge type, not only fired or
#   listed ones (INTEREST).
# Dump format, one line per record, space separated (ptr = tensor address as printed by %p, '(nil)' for none):
#   N <idx> <ptr> <OP> <name> <type> <ne0> <ne1> <ne2> <ne3> <flags> V <view_src ptr> [S<j> <src j ptr>]...
#       one per graph node in execution order; name may be empty or contain spaces; flags = tensor->flags (2 = OUTPUT)
#   F <idx> <fusion label or -> <n nodes consumed> <enqueued 0|1> <overlap-disabled label or ->
#       post files only: the fusion decision at node idx (label '-' = dispatched unfused; only enqueued fusions count); the last
#       field names a fusion that matched but was skipped because its output overlaps an input buffer
#   T <ptr> - <ns>      times file: perf-logger time of the dispatch whose output is ptr (summed over repeats)
# pre-* comes from the first build of the graph (different pointers): nodes are matched by (op, name, shape, occurrence).
import sys, os, glob, collections, argparse, re
NRE = re.compile(r'^N (\d+) (\S+) (\S+) (.*?) (\S+) (-?\d+) (-?\d+) (-?\d+) (-?\d+) (\d+) V (\S+)(.*)$')

INTEREST = {('MUL_MAT','SWIGLU'),('MUL_MAT','ADD'),('MUL_MAT_ID','SWIGLU'),('MUL_MAT_ID','MUL'),('RMS_NORM','MUL'),('MUL','ROPE'),('SILU','MUL'),('SIGMOID','MUL'),('ADD','ADD'),('ADD','RMS_NORM'),('MUL_MAT','CPY_f16'),('MUL_MAT_ID','CPY_f16'),('CPY_f16','SWIGLU'),('MUL','ADD'),('RMS_NORM','SCALE'),('SSM_CONV','SILU'),('MUL','MUL')}
EMPTY = {'NONE', 'RESHAPE', 'VIEW', 'PERMUTE', 'TRANSPOSE'}

def load(path):
    nodes, fused = [], {}
    for line in open(path):
        p = line.split()
        if not p: continue
        if p[0] == 'N':
            m = NRE.match(line)
            i = int(m.group(1)); ptr = m.group(2); op = m.group(3); name = m.group(4); typ = m.group(5)
            ne = tuple(int(m.group(k)) for k in range(6, 10)); flags = int(m.group(10)); view = m.group(11)
            srcs = m.group(12).split()[1::2]
            nodes.append(dict(i=i, ptr=ptr, op=op, name=name, type=typ, ne=ne, flags=flags, srcs=srcs, view=view))
        elif p[0] == 'F':
            fused[int(p[1])] = dict(label=p[2], n=int(p[3]), enq=int(p[4]), off=p[5])
    return nodes, fused

def load_times(path):
    t = collections.defaultdict(list)
    if not os.path.exists(path):
        return {}
    for line in open(path):
        p = line.split()
        if len(p) >= 4 and p[0] == 'T' and p[2] == '-':
            t[p[1]].append(int(p[3]))
    return {k: sum(v) for k, v in t.items()}

def analyze(d, key, interest_all=False, out=sys.stdout):
    post, fused = load(f'{d}/post-{key}.txt')
    pre_path = f'{d}/pre-{key}.txt'
    pre = load(pre_path)[0] if os.path.exists(pre_path) else None
    times = load_times(f'{d}/times-{key}.txt')
    pos = {n['ptr']: n['i'] for n in post}
    # pre dumps come from the first build of the graph (other pointers): match nodes by (op, name, shape, occurrence)
    def ident(lst):
        occ = collections.Counter(); out = {}
        for n in lst:
            k = (n['op'], n['name'], n['ne']); out[n['ptr']] = k + (occ[k],); occ[k] += 1
        return out
    prepos = {}
    if pre:
        pre_id = ident(pre); post_id = ident(post)
        id2pre = {pre_id[n['ptr']]: n['i'] for n in pre}
        prepos = {ptr: id2pre[k] for ptr, k in post_id.items() if k in id2pre}
    byptr = {n['ptr']: n for n in post}
    uses = collections.Counter()
    consumers = collections.defaultdict(list)
    for n in post:
        for s in n['srcs']:
            uses[s] += 1
            consumers[s].append(n)
    # fused-dispatch membership: node index -> (head index, label)
    member = {}
    for h, f in fused.items():
        if f['label'] != '-' and f['enq']:
            for k in range(h, h + f['n']):
                member[k] = (h, f['label'])
    off = collections.Counter(f['off'] for f in fused.values() if f['off'] != '-')
    labels = collections.Counter(f['label'] for f in fused.values() if f['label'] != '-' and f['enq'])

    def opk(n):
        op = n['op']
        if op == 'CPY':
            op = 'CPY_' + n['type']
        return op

    stats = collections.defaultdict(lambda: dict(total=0, fused=0, adj=0, split=0, preadj=0, t_cons_unfused=0, t_prod_unfused=0, by=collections.Counter(), gap=collections.Counter(), cons_seen=set()))
    for c in post:
        if c['op'] in EMPTY:
            continue
        seen = set()
        for s in c['srcs']:
            if s in seen or s not in byptr:
                continue
            seen.add(s)
            p = byptr[s]
            via = ''
            # step back through single-use reshape / view nodes (e.g. a product reshaped before its residual ADD)
            while p['op'] in ('RESHAPE', 'VIEW') and uses[p['ptr']] == 1 and p['srcs'] and p['srcs'][0] in byptr:
                via = '~' + p['op'][0]
                s = p['srcs'][0]
                p = byptr[s]
            if p['op'] in EMPTY:
                continue
            if uses[s] != 1 or (p["flags"] & 2):   # GGML_TENSOR_FLAG_OUTPUT
                continue
            k = (opk(p), via + opk(c))
            st = stats[k]
            st['total'] += 1
            pi, ci = p['i'], c['i']
            mp, mc = member.get(pi), member.get(ci)
            if mp and mc and mp[0] == mc[0]:
                st['fused'] += 1
                st['by'][mp[1]] += 1
            elif ci - pi == 1 or (via and all(post[j]['op'] in EMPTY for j in range(pi + 1, ci))):
                st['adj'] += 1
            else:
                st['split'] += 1
                between = [post[j]['op'] for j in range(pi + 1, ci) if post[j]['op'] not in EMPTY]
                st['gap'][len(between) if between else 0] += 1
            if prepos and s in prepos and prepos.get(c['ptr'], -9) - prepos[s] == 1:
                st['preadj'] += 1
            if not (mp and mc and mp[0] == mc[0]) and mc is None and c['ptr'] not in st['cons_seen']:
                st['cons_seen'].add(c['ptr'])
                st['t_cons_unfused'] += times.get(c['ptr'], 0)
    return stats, labels, off, times, post, fused, member

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='fusion census over a graph dump dir (see the header comment for the format)')
    ap.add_argument('dir'); ap.add_argument('--key'); ap.add_argument('--all', action='store_true')
    a = ap.parse_args()
    keys = [a.key] if a.key else sorted(os.path.basename(f)[5:-4] for f in glob.glob(f'{a.dir}/post-*.txt'))
    for key in keys:
        stats, labels, off, times, post, fused, member = analyze(a.dir, key)
        tot_t = sum(times.values())
        print(f'== {key}: {len(post)} nodes, perf total {tot_t/1e6:.1f} ms')
        print('fired labels:', ' '.join(f'{k}={v}' for k, v in sorted(labels.items())))
        if off:
            print('overlap-disabled:', dict(off))
        print(f"{'producer -> consumer':42s} {'pairs':>6s} {'fused':>6s} {'adj-not':>7s} {'split':>6s} {'pre-adj':>7s} {'unfused cons ms':>15s}  fused-by / gap sizes")
        for k, st in sorted(stats.items(), key=lambda x: -x[1]['total']):
            if not a.all and st['fused'] == 0 and k not in INTEREST:
                continue
            print(f"{k[0]+' -> '+k[1]:42s} {st['total']:6d} {st['fused']:6d} {st['adj']:7d} {st['split']:6d} {st['preadj']:7d} {st['t_cons_unfused']/1e6:15.2f}  {dict(st['by'])} {dict(sorted(st['gap'].items()))}")

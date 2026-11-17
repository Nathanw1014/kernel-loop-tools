#!/usr/bin/env python3
"""camp.py: static stride census of one ubatch graph (graph/graphdump TSV): which activation / cache / state tensors does each
GEMM, flash-attention or scan node read, at which row stride, and does that stride camp on a few DRAM channels?

usage: camp.py <graph.tsv> [--all] [--granule BYTES] [--channels N]

For every non-quantised (f32 / f16 / bf16) src of a MUL_MAT, MUL_MAT_ID, FLASH_ATTN_EXT, GATED_DELTA_NET, SSM_CONV, SSM_SCAN or
RWKV_WKV7 node (every node with --all), grouped by (consumer op, consumer, dst shape, src slot, src, type, ne, nb1, nb2):
the row stride nb1 (and nb2) in bytes and the camping predictor gcd(nb / granule, channels). If addresses interleave across
<channels> memory channels every <granule> bytes, a stride that is a multiple of granule*channels sends every row to the same
channel: gcd = channels is the worst case, 1 the best. A stride that is not a multiple of the granule is fractional in
granules and de-aliases: printed "frac". Set --granule / --channels to your memory controller's interleave (defaults 256 B,
16 channels; check your platform's documentation or measure: mem/camp-hip and mem/chan-*.sh are the runtime check).
Layer numbers are folded so N identical layers make one row with count N. Output is a markdown table.
Guards against: a padded or power-of-two stride that serialises a kernel's loads on one channel, invisible in op timing
until you look at addresses.
"""
import sys, re, math, collections

CONSUMERS = ('MUL_MAT', 'MUL_MAT_ID', 'FLASH_ATTN_EXT', 'GATED_DELTA_NET', 'SSM_CONV', 'SSM_SCAN', 'RWKV_WKV7')
QUANT_FREE = ('f32', 'f16', 'bf16')


def parse_desc(s):
    name, op, typ, ne, nb, vsrc, voffs, use = s.split('|')
    return dict(name=name, op=op, type=typ, ne=[int(x) for x in ne.split(',')], nb=[int(x) for x in nb.split(',')],
                vsrc=vsrc, voffs=int(voffs), use=use)


def load(fn):
    nodes = []
    for line in open(fn):
        if line.startswith('#'):
            continue
        f = line.rstrip('\n').split('\t')
        node = parse_desc(f[1]); node['idx'] = int(f[0]); node['src'] = {}
        i = 2
        while i < len(f):
            node['src'][int(f[i][1:])] = parse_desc(f[i + 1]); i += 2
        nodes.append(node)
    return nodes


GRANULE, CHANNELS = 256, 16


def camp(nb):
    if nb == 0:
        return '-'
    if nb % GRANULE:
        return 'frac'
    return str(math.gcd(nb // GRANULE, CHANNELS))


def base(name):
    # strip layer numbers so identical layers fold into one row
    return re.sub(r'-\d+\b', '-L', re.sub(r'\b\d+\b', 'L', name))


def main():
    global GRANULE, CHANNELS
    a = sys.argv[1:]
    if not a or a[0] in ('-h', '--help'):
        print(__doc__); return 0 if a else 2
    for flag in ('--granule', '--channels'):
        if flag in a:
            i = a.index(flag); v = int(a[i + 1]); del a[i:i + 2]
            if flag == '--granule': GRANULE = v
            else: CHANNELS = v
    allops = '--all' in a
    a = [x for x in a if x != '--all']
    nodes = load(a[0])
    G = collections.OrderedDict()
    ops = collections.Counter(n['op'] for n in nodes)
    for n in nodes:
        if not allops and not (n['op'] in CONSUMERS or 'QSA' in n['op'] or 'ATTN' in n['op']):
            continue
        for j, s in n['src'].items():
            if s['type'] not in QUANT_FREE:
                continue
            if s['use'] == 'W' and not allops:
                kind = 'weight(f)'
            else:
                kind = {'C': 'compute', 'A': 'cache/state/input', 'W': 'weight(f)'}.get(s['use'], s['use'])
            w0 = n['src'].get(0)
            cons = base(w0['name']) if (w0 and w0['use'] == 'W') else base(n['name'])
            cons = re.sub(r'^node_\d+$', 'node', cons)
            sname = re.sub(r'^node_\d+', 'node', base(s['name']))
            k = (n['op'], cons, f"{n['ne'][0]}x{n['ne'][1]}x{n['ne'][2]}", j, sname, s['type'],
                 s['ne'][0], s['ne'][1], s['ne'][2], s['nb'][1], s['nb'][2], kind, base(s['vsrc']))
            G.setdefault(k, 0); G[k] += 1
    print(f"# {a[0]}: {len(nodes)} nodes, granule {GRANULE} B, {CHANNELS} channels; ops: " + ', '.join(f'{o} {c}' for o, c in ops.most_common()))
    print("| consumer op | consumer (weight or node) | dst ne | src | src name | type | src ne0 x ne1 x ne2 | nb1 B | gcd(nb1/granule,channels) | nb2 B | gcd(nb2) | footprint MB | buffer | view of | count |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for k, c in G.items():
        op, cn, cne, j, sn, t, a, b, d, nb1, nb2, kind, vs = k
        fp = max(nb1 * b, nb2 * d) / 1e6
        print(f"| {op} | {cn} | {cne} | src{j} | {sn} | {t} | {a} x {b} x {d} | {nb1} | {camp(nb1)} | {nb2} | {camp(nb2)} | {fp:.1f} | {kind} | {vs} | {c} |")


if __name__ == '__main__':
    sys.exit(main())

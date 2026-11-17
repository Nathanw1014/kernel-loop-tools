#!/usr/bin/env python3
"""sync-census.py <sync-logger log> [--min-nodes N]: barrier census from the ggml-vulkan sync logger.
Input: stderr of any llama.cpp tool run with GGML_VK_SYNC_LOGGER=1 (upstream): a line "sync" before every inserted pipeline
barrier and one "<node idx> <OP> <name> ..." line per dispatched node. A graph evaluation starts where the node index resets to 0.
Prints the number of evaluations, nodes / syncs for the last few large ones (>= N nodes, default 1000, so small helper graphs are
skipped), and for the last large one a histogram of the op dispatched right after each sync (which ops start a new barrier)."""
import sys, re, collections

args = sys.argv[1:]
min_nodes = 1000
if '--min-nodes' in args:
    i = args.index('--min-nodes'); min_nodes = int(args[i + 1]); del args[i:i + 2]
if len(args) != 1 or args[0] in ('-h', '--help'): print(__doc__); sys.exit(2)
evals = []; cur = None; pend = False
for l in open(args[0], errors='replace'):
    l = l.rstrip('\n')
    if l == 'sync': pend = True; continue
    m = re.match(r'^(\d+) ([A-Z_0-9]+)( |$)', l)
    if not m: continue
    i = int(m.group(1)); op = m.group(2)
    if i == 0 or cur is None: cur = {'nodes': 0, 'syncs': 0, 'after': collections.Counter()}; evals.append(cur)
    cur['nodes'] += 1
    if pend: cur['syncs'] += 1; cur['after'][op] += 1; pend = False
big = [e for e in evals if e['nodes'] >= min_nodes]
print(f"evals={len(evals)} big(>={min_nodes} nodes)={len(big)} per-big-eval nodes/syncs: " + ' '.join(f"{e['nodes']}/{e['syncs']}" for e in big[-4:]))
if not big and evals:
    print(f"no eval reaches {min_nodes} nodes (largest {max(e['nodes'] for e in evals)}): pass --min-nodes N below that "
          "(small models and decode graphs are smaller than a full prefill graph)")
if not evals: print('no sync-logger blocks found: was GGML_VK_SYNC_LOGGER=1 set, and is stderr in this file?')
if big: print("sync -> next op (last big eval): " + ' '.join(f"{k}={v}" for k, v in big[-1]['after'].most_common(14)))

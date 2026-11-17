#!/usr/bin/env python3
# timing_report.py <decoded capture dir> <kernel regex> [label]   (timing-mode SQTT captures; CPU only)
# For the code object(s) whose name in <dir>.extract.log matches <kernel regex> (a debug label when the capture
# carried GGML_VK_DEBUG_MARKERS, else "pipeline <hash>"):
#   - stall classes: latency / stall share per instruction class (s_waitcnt vmcnt vs lgkmcnt, VALU, WMMA, s_barrier, loads, ...)
#   - buffer / global load latency: per-instruction mean latency (cycles per hit) percentiles over the load instructions
#   - trust: how much decoded. Capture-level: blanked exec tokens, unsizable headers (decoder may lose sync after each), nibbles,
#     DECODER-WARNING lines (Data Lost). Kernel-level: instruction coverage (executed / all instructions of the code object) and
#     hits of the entry instruction (one per decoded wave start) vs the waves the occupancy table puts on the traced SE.
# code.json rows: ISA, _, line, source, codeobj, vaddr, hits, latency, stall, idle (cycles summed over the traced waves).
import json, re, sys, os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import wavetrace

if len(sys.argv) < 3:
    sys.exit('usage: timing_report.py <decoded capture dir> <kernel regex> [label]')
d = Path(sys.argv[1]); pat = re.compile(sys.argv[2]); label = sys.argv[3] if len(sys.argv) > 3 else d.name
ext = Path(str(d) + '.extract.log')
cos = {}
if ext.exists():
    for l in ext.read_text(errors='replace').splitlines():
        m = re.match(r'\s*co (\d+): base=\S+ text=(\d+) (.*)$', l)
        if m and pat.search(m.group(3)):
            cos[int(m.group(1))] = m.group(3)
info = json.loads((d / 'capture-info.json').read_text()) if (d / 'capture-info.json').exists() else {}
san = info.get('sanitize', {})
warns = [l for l in (d / 'decode.log').read_text(errors='replace').splitlines() if 'DECODER-WARNING' in l] if (d / 'decode.log').exists() else []
print(f'== {label}: {d.name}')
print(f'  capture: blanked exec tokens {san.get("blanked_exec_tokens", "?")}, unsizable headers {san.get("unknown_headers", "?")} '
      f'of {san.get("nibbles", "?")} nibbles, decoder warnings {len(warns)}' + (f' (e.g. {warns[0].split(": ", 2)[-1][:80]})' if warns else ''))
cj = d / 'decoded' / 'code.json'
if not cj.exists():
    print('  no decoded/code.json'); sys.exit(1)
code = json.loads(cj.read_text())['code']
hit_cos = sorted({r[4] for r in code if r[6] > 0})
if not cos:
    print(f'  no code object matches /{pat.pattern}/; code objects with hits: {hit_cos}'); sys.exit(1)
for co, name in cos.items():
    rows = [r for r in code if r[4] == co]
    ex = [r for r in rows if r[6] > 0]
    print(f'  code object {co} ({name[:60]}): {len(ex)} of {len(rows)} instructions executed ({len(ex) / max(len(rows), 1):.0%} coverage)')
    if not ex:
        continue
    entry = min(rows, key=lambda r: r[5])
    print(f'    entry instruction hits (decoded wave starts on the traced CU): {entry[6]}')
    lat, st = sum(r[7] for r in ex), sum(r[8] for r in ex)
    agg = {}
    for r in ex:
        a = agg.setdefault(wavetrace.isa_class(r[0]), [0, 0, 0, 0])
        a[0] += 1; a[1] += r[6]; a[2] += r[7]; a[3] += r[8]
    print(f'    latency {lat:.4g} cyc, stall {st:.4g} cyc; {"class":22} {"#ins":>5} {"hits":>9} {"lat%":>6} {"stall%":>7} {"lat/hit":>8}')
    for k, a in sorted(agg.items(), key=lambda kv: -kv[1][2]):
        print(f'      {k:22} {a[0]:5} {a[1]:9} {a[2] / max(lat, 1):6.1%} {a[3] / max(st, 1):7.1%} {a[2] / a[1]:8.1f}')
    ld = sorted(r[7] / r[6] for r in ex if wavetrace.isa_class(r[0]).startswith('buffer_load'))
    if ld:
        q = lambda p: ld[min(len(ld) - 1, int(p * len(ld)))]
        print(f'    buffer/global load latency per instruction (cyc/hit, {len(ld)} load instructions): '
              f'p10 {q(0.1):.0f} p50 {q(0.5):.0f} p90 {q(0.9):.0f} max {ld[-1]:.0f}')
    wv = sorted(r[7] / r[6] for r in ex if wavetrace.isa_class(r[0]).startswith('s_waitcnt vm'))
    if wv:
        print(f'    s_waitcnt vmcnt per instruction (cyc/hit, {len(wv)}): p50 {wv[len(wv) // 2]:.0f} max {wv[-1]:.0f}')

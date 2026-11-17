#!/usr/bin/env python3
# consist-gate.py: multi-slot consistency gate for llama-server.
#
# One launch of llama-server (started and stopped here, same process group: run it as a job-run `run` cell), then:
#   alone.1 / alone.2 : each prompt served by itself, one after another, twice      -> repeat check (alone.1 == alone.2)
#   conc.1  / conc.2  : all prompts sent at once (one thread each), twice           -> concurrency check (conc.k == alone.1)
#                                                                                       and concurrent repeat (conc.1 == conc.2)
# Greedy (temperature 0, top_k 1), ignore_eos, cache_prompt false, no n_probs (n_probs makes streams launch-dependent).
# Token ids come from return_tokens. Prompts have different lengths on purpose (ubatch / tile boundaries differ per slot).
# Concurrent phases must have every request in flight at once on distinct slots (census line), else rc 3 NOT ENGAGED.
# For each prompt with a conc-vs-alone mismatch: the alone top-5 (id, logprob) at the first differing index (margin line):
# a near tie there reads as batch-shape numerics, a wide gap as a wrong-result candidate (not decided here).
# Any mismatch = rc 1 and a MISMATCH line with the first differing index. A mismatch is a finding to report, not something
# this script decides is a bug or not.
#
# usage: consist-gate.py --server BIN --model GGUF --out DIR --text FILE [--port 8191] [--n-predict 64] [--np 4]
#                        [--ready-timeout 900] [--req-timeout 900] -- <extra llama-server args>
# --text: a plain-text corpus of at least ~250 kB (prompts are cut at fixed offsets from 100000 to 245000 chars), e.g. a
#         wikitext-2 raw test file. Refuses to start while $KLT_STOP exists.
# Extra args after -- go to llama-server unchanged (e.g. -ngl 99 -fa on -c 8192 -kvu).
# The script adds only -m, --host 127.0.0.1, --port and -np.
# --control: negative control for the comparator: prompt 0 gets one extra word in the concurrent phases only, so
#            conc-vs-alone for prompt 0 MUST report a mismatch (rc 1); a PASS with --control means the gate is blind.
# --probs K: every phase request carries n_probs K (pre-sampling, full-vocab softmax of the raw logits:
#            post_sampling_probs false), so each generated token has its own logprob + top-K. For every compared pair the
#            script prints per-step |delta logprob| of the reference token (alone.1 / slot 0) while the two streams agree, and at
#            the first differing index: the delta of the reference token there (from the other run's top-K; missing = inf),
#            and the reference run's gap between its token and the other run's token (near tie = small).
# --tol NATS (needs --probs): tolerance mode. repeat-alone must stay identical (exact); a conc check passes when every
#            per-step delta of the reference token before the first differing index is <= NATS and, at that index, the
#            shift of the two tokens' difference (ref gap + other gap; inf when either token is missing from the other run's
#            top-K, so use K >= 20) is <= NATS (batch-shape numerics at a near tie), else FAIL (wrong-result candidate).
#            With --control, rc 0 only when both prompt 0 conc-vs-alone pairs are TOL-FAIL (the tolerance is not blind). rc 0 = all within tolerance, 1 = beyond tolerance or repeat-alone differs.
# --same IDX: all slots get prompt IDX (control: no slot can read different content from another; adds slot-vs-slot checks
#            of the concurrent phases against slot 0).
import argparse, json, math, os, signal, socket, subprocess, sys, threading, time, urllib.request, urllib.error

ap = argparse.ArgumentParser()
ap.add_argument('--server', required=True)
ap.add_argument('--model', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--text', required=True, help='plain-text corpus, >= ~250 kB')
ap.add_argument('--port', type=int, default=8191)
ap.add_argument('--n-predict', type=int, default=64)
ap.add_argument('--np', type=int, default=4)
ap.add_argument('--ready-timeout', type=int, default=900)
ap.add_argument('--req-timeout', type=int, default=900)
ap.add_argument('--control', action='store_true')
ap.add_argument('--probs', type=int, default=0)
ap.add_argument('--tol', type=float, default=0.0)
ap.add_argument('--same', type=int, default=-1)
ap.add_argument('extra', nargs=argparse.REMAINDER)
a = ap.parse_args()
if a.tol > 0 and a.probs <= 0:
    print('--tol needs --probs K', flush=True)
    sys.exit(2)
extra = a.extra[1:] if a.extra[:1] == ['--'] else a.extra
os.makedirs(a.out, exist_ok=True)

def log(msg):
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}"
    print(line, flush=True)

_home = os.environ.get('KLT_HOME') or os.path.join(os.path.expanduser('~'), '.kernel-loop')
_stop = os.environ.get('KLT_STOP') or os.path.join(_home, 'STOP-GPU')   # same defaults as lib/klt-env.sh
if os.path.exists(_stop):
    log(f"FAIL {_stop} exists (device error recorded); refusing to start")
    sys.exit(70)

# port must be free before the load (a live server on it would answer instead of ours)
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)   # TIME_WAIT from a previous run is not a live server
try:
    s.bind(('127.0.0.1', a.port))
except OSError:
    log(f"FAIL port {a.port} busy")
    sys.exit(2)
finally:
    s.close()

# prompts: different lengths (chars), fixed offsets into the text
txt = open(a.text, errors='ignore').read()
spans = [(100000, 300), (120000, 1100), (140000, 2600), (160000, 5200), (180000, 700), (200000, 1800), (220000, 3800), (240000, 450)]
if len(txt) < max(o + n for o, n in spans[:max(a.np, a.same + 1)]):
    log(f"FAIL --text {a.text} is too short ({len(txt)} chars) for the fixed prompt offsets")
    sys.exit(2)
prompts = [txt[o:o + n] for o, n in spans[:a.np]]
if a.same >= 0:
    o, n = spans[a.same]
    prompts = [txt[o:o + n]] * a.np

srv_log = open(os.path.join(a.out, 'server.log'), 'w')
cmd = [a.server, '-m', a.model, '--host', '127.0.0.1', '--port', str(a.port), '-np', str(a.np)] + extra
log('server: ' + ' '.join(cmd))
srv = subprocess.Popen(cmd, stdout=srv_log, stderr=subprocess.STDOUT)   # same process group as this cell
base = f"http://127.0.0.1:{a.port}"

def stop_server():
    if srv.poll() is None:
        srv.send_signal(signal.SIGTERM)
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.kill()
            srv.wait(timeout=30)
    gone = srv.poll() is not None
    log(f"server pid {srv.pid} stopped rc={srv.returncode} gone={gone}")

def fail(msg, rc=2):
    log('FAIL ' + msg)
    stop_server()
    sys.exit(rc)

t0 = time.time()
while True:
    if srv.poll() is not None:
        fail(f"server exited rc={srv.returncode} before ready (see server.log)")
    try:
        with urllib.request.urlopen(base + '/health', timeout=5) as r:
            if r.status == 200:
                break
    except (urllib.error.URLError, ConnectionError, socket.timeout, OSError):
        pass
    if time.time() - t0 > a.ready_timeout:
        fail('server not ready in time')
    time.sleep(2)
log(f"server ready in {time.time() - t0:.0f} s")

def req(prompt):
    t_start = time.time()
    body = {"prompt": prompt, "n_predict": a.n_predict, "cache_prompt": False, "temperature": 0.0, "top_k": 1,
            "ignore_eos": True, "return_tokens": True, "seed": 1}
    if a.probs > 0:
        body["n_probs"] = a.probs
        body["post_sampling_probs"] = False
    r = urllib.request.urlopen(urllib.request.Request(base + '/completion', data=json.dumps(body).encode(),
                                                      headers={"Content-Type": "application/json"}), timeout=a.req_timeout)
    j = json.load(r)
    t_end = time.time()
    toks = j.get('tokens') or []
    steps = []
    for cp in (j.get('completion_probabilities') or []):
        steps.append({"id": cp.get('id'), "lp": cp.get('logprob'),
                      "top": [[t.get('id'), t.get('logprob')] for t in (cp.get('top_logprobs') or [])]})
    return {"tokens": toks, "n_prompt": j.get('timings', {}).get('prompt_n'), "content": j.get('content', '')[:160],
            "id_slot": j.get('id_slot'), "t_start": t_start, "t_end": t_end, "steps": steps}

phases = {}
try:
    for ph in ('alone.1', 'alone.2'):
        phases[ph] = [req(p) for p in prompts]
        log(f"{ph}: n_prompt {[x['n_prompt'] for x in phases[ph]]} n_gen {[len(x['tokens']) for x in phases[ph]]}")
    for ph in ('conc.1', 'conc.2'):
        res = [None] * len(prompts)
        errs = []
        def work(i):
            try:
                res[i] = req(prompts[i] + (' However' if (a.control and i == 0) else ''))
            except Exception as e:   # noqa: BLE001 - reported below
                errs.append(f"slot {i}: {e}")
        th = [threading.Thread(target=work, args=(i,)) for i in range(len(prompts))]
        for t in th:
            t.start()
        for t in th:
            t.join()
        if errs:
            fail(f"{ph}: " + '; '.join(errs))
        phases[ph] = res
        log(f"{ph}: n_prompt {[x['n_prompt'] for x in res]} n_gen {[len(x['tokens']) for x in res]}")
        # engagement census: all requests in flight at once, on distinct slots
        slots = sorted({x['id_slot'] for x in res})
        overlap = max(x['t_start'] for x in res) < min(x['t_end'] for x in res)
        log(f"{ph}: census slots {slots} all-in-flight-together {overlap} "
            f"spans_s {[round(x['t_end'] - x['t_start'], 2) for x in res]}")
        if len(slots) != len(prompts) or not overlap:
            fail(f"{ph}: NOT ENGAGED (slots {slots}, overlap {overlap}): the concurrent phase did not run concurrently", 3)
except Exception as e:   # noqa: BLE001
    fail(f"request error: {e}")

def first_diff(x, y):
    for k in range(min(len(x), len(y))):
        if x[k] != y[k]:
            return k
    return None if len(x) == len(y) else min(len(x), len(y))

def top_at(i, prefix):
    # top-5 of the next token after prompt i + prefix (alone, n_probs: diagnostic only, never compared as a stream)
    tk = json.load(urllib.request.urlopen(urllib.request.Request(base + '/tokenize', data=json.dumps(
        {"content": prompts[i], "add_special": True}).encode(), headers={"Content-Type": "application/json"}), timeout=60))['tokens']
    body = {"prompt": tk + list(prefix), "n_predict": 1, "cache_prompt": False, "temperature": 0.0, "top_k": 40,
            "n_probs": 5, "ignore_eos": True, "return_tokens": True}
    j = json.load(urllib.request.urlopen(urllib.request.Request(base + '/completion', data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"}), timeout=a.req_timeout))
    cp = (j.get('completion_probabilities') or [{}])[0]
    tops = cp.get('top_logprobs') or cp.get('top_probs') or cp.get('probs') or []
    return [(t.get('id'), round(t.get('logprob', t.get('prob', float('nan'))), 4)) for t in tops]

checks = []   # (name, prompt index, first diff or None)
ref = phases['alone.1']
for i in range(len(prompts)):
    for ph, name in (('alone.2', 'repeat-alone'), ('conc.1', 'conc-vs-alone'), ('conc.2', 'conc-vs-alone'),):
        checks.append((f"{name}[{ph}]", i, first_diff(ref[i]['tokens'], phases[ph][i]['tokens'])))
    checks.append(("repeat-conc[conc.2 vs conc.1]", i, first_diff(phases['conc.1'][i]['tokens'], phases['conc.2'][i]['tokens'])))
    if a.same >= 0 and i > 0:
        for ph in ('conc.1', 'conc.2'):
            checks.append((f"slot-vs-slot[{ph} vs slot-0 request]", i, first_diff(phases[ph][0]['tokens'], phases[ph][i]['tokens'])))

def pair_of(name, i):
    # (reference result, other result) for a check name
    if name.startswith('repeat-alone'):
        return ref[i], phases['alone.2'][i]
    if name.startswith('conc-vs-alone'):
        return ref[i], phases[name.split('[')[1].rstrip(']')][i]
    if name.startswith('repeat-conc'):
        return phases['conc.1'][i], phases['conc.2'][i]
    ph = name.split('[')[1].split(' ')[0]
    return phases[ph][0], phases[ph][i]

def deltas(r, o):
    # per-step |delta logprob| of the reference token; at the first differing index the reference token's delta (from the
    # other run's top-K, inf when absent) and the reference run's gap between its token and the other run's token
    rs, os_ = r['steps'], o['steps']
    out, flip = [], None
    for k in range(min(len(rs), len(os_))):
        if rs[k]['id'] == os_[k]['id']:
            out.append(abs(rs[k]['lp'] - os_[k]['lp']))
            continue
        a_tok, c_tok = rs[k]['id'], os_[k]['id']
        o_lp = {t: lp for t, lp in os_[k]['top']}
        r_lp = {t: lp for t, lp in rs[k]['top']}
        d_at = abs(o_lp[a_tok] - rs[k]['lp']) if a_tok in o_lp else math.inf
        gap_r = (rs[k]['lp'] - r_lp[c_tok]) if c_tok in r_lp else None
        gap_o = (os_[k]['lp'] - o_lp[a_tok]) if a_tok in o_lp else None
        shift = (gap_r + gap_o) if (gap_r is not None and gap_o is not None) else math.inf
        out.append(shift)
        flip = {"index": k, "ref_token": a_tok, "other_token": c_tok, "delta_ref_token": d_at,
                "ref_gap": gap_r, "other_gap": gap_o, "shift": shift}
        break
    return out, flip

def fmt(x):
    return 'inf' if x is None or x == math.inf else f"{x:.4f}"

delta_rows = []
if a.probs > 0:
    for name, i, d in checks:
        r, o = pair_of(name, i)
        ds, flip = deltas(r, o)
        if not ds:
            log(f"delta {name} prompt {i}: no probs returned")
            delta_rows.append({"check": name, "prompt": i, "max": math.inf, "n": 0, "flip": None})
            continue
        srt = sorted(ds)
        pre = ds[:-1] if flip else ds
        row = {"check": name, "prompt": i, "n": len(ds), "max": max(ds), "p50": srt[len(srt) // 2],
               "max_pre": max(pre) if pre else 0.0, "flip": flip, "per_step": ds}
        delta_rows.append(row)
        fl = (f" | first diff {flip['index']}: shift {fmt(flip['shift'])} (ref gap to other token {fmt(flip['ref_gap'])} + "
              f"other gap {fmt(flip['other_gap'])}), ref token delta {fmt(flip['delta_ref_token'])}") if flip else ''
        log(f"delta {name} prompt {i}: steps {len(ds)} max {fmt(row['max'])} p50 {fmt(row['p50'])} "
            f"max before diff {fmt(row['max_pre'])}{fl}")
    for kind in ('repeat-alone', 'conc-vs-alone', 'repeat-conc', 'slot-vs-slot'):
        rows = [x for x in delta_rows if x['check'].startswith(kind)]
        if rows:
            allv = sorted(v for x in rows for v in (x.get('per_step', [])[:-1] if x['flip'] else x.get('per_step', [])))
            shifts = sorted(round(x['flip']['shift'], 4) for x in rows if x['flip'])
            log(f"delta summary {kind}: pairs {len(rows)} steps {len(allv)} max {fmt(max(allv) if allv else None)} "
                f"p50 {fmt(allv[len(allv) // 2] if allv else None)} p90 {fmt(allv[int(len(allv) * 0.9)] if allv else None)} "
                f"(same-token steps) | flips {len(shifts)} shifts {shifts}")

margins = {}
for i in range(len(prompts)):
    ds = [d for n, j, d in checks if j == i and d is not None and n.startswith('conc-vs-alone')]
    if ds and not a.control and a.probs <= 0:
        d = min(ds)
        try:
            alt = next(phases[ph][i]['tokens'][d] for ph in ('conc.1', 'conc.2')
                       if first_diff(ref[i]['tokens'], phases[ph][i]['tokens']) == d)
            margins[i] = {"index": d, "alone_token": ref[i]['tokens'][d], "conc_token": alt,
                          "alone_top5": top_at(i, ref[i]['tokens'][:d])}
            log(f"margin prompt {i} at index {d}: alone token {ref[i]['tokens'][d]} conc token {alt} | alone top5 (id, logprob) {margins[i]['alone_top5']}")
        except Exception as e:   # noqa: BLE001 - diagnostic only
            log(f"margin prompt {i}: diagnostic failed: {e}")
stop_server()

empty = [i for i in range(len(prompts)) if len(ref[i]['tokens']) < a.n_predict]
bad = [c for c in checks if c[2] is not None]
for name, i, d in bad:
    log(f"MISMATCH {name} prompt {i} (n_prompt {ref[i]['n_prompt']}): first differing token index {d} of {a.n_predict}")
json.dump({"prompts_chars": [len(p) for p in prompts], "phases": phases, "margins": margins,
           "checks": [{"check": n, "prompt": i, "first_diff": d} for n, i, d in checks],
           "deltas": [{k: (None if v == math.inf else v) for k, v in x.items()} for x in delta_rows]},
          open(os.path.join(a.out, 'consist.json'), 'w'), indent=1)
if empty:
    log(f"FAIL short streams (return_tokens empty?) for prompts {empty}")
    sys.exit(1)
if a.tol > 0:
    rep_bad = [c for c in bad if c[0].startswith('repeat-alone')]
    over = [x for x in delta_rows if not x['check'].startswith('repeat-alone') and x['max'] > a.tol]
    for x in over:
        log(f"TOL-FAIL {x['check']} prompt {x['prompt']}: max step metric {fmt(x['max'])} > {a.tol}")
    if a.control:
        want = [x for x in over if x['prompt'] == 0 and x['check'].startswith('conc-vs-alone')]
        log(f"control (tol): prompt 0 conc-vs-alone TOL-FAIL {len(want)}/2 (want 2); repeat-alone mismatches {len(rep_bad)}")
        sys.exit(0 if len(want) == 2 and not rep_bad else 1)
    if rep_bad:
        log(f"consist TOL FAIL: repeat-alone differs {len(rep_bad)} (must be exact)")
        sys.exit(1)
    if over:
        log(f"consist TOL FAIL: {len(over)} pairs beyond {a.tol} nats ({len(bad)} token mismatches)")
        sys.exit(1)
    log(f"consist TOL PASS: repeat-alone exact; every compared step within {a.tol} nats "
        f"({len(bad)} token mismatches, each within the tolerance at its flip; {len(prompts)} prompts x {a.n_predict} tokens, np {a.np})")
    sys.exit(0)
if a.control:
    want = [c for c in bad if c[1] == 0 and c[0].startswith('conc-vs-alone')]
    other = [c for c in bad if not (c[1] == 0 and c[0].startswith('conc-vs-alone'))]
    log(f"control: prompt 0 conc-vs-alone mismatches {len(want)}/2 (want 2), other mismatches {len(other)}")
    sys.exit(0 if len(want) == 2 and not other else 1)
if bad:
    log(f"consist FAIL: {len(bad)}/{len(checks)} checks differ ({len(prompts)} prompts x {a.n_predict} tokens)")
    sys.exit(1)
log(f"consist PASS: {len(checks)}/{len(checks)} checks identical ({len(prompts)} prompts x {a.n_predict} tokens, np {a.np})")

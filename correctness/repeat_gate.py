#!/usr/bin/env python3
# repeat_gate.py: send the SAME request N times into ONE llama-server process and compare the
# responses to EACH OTHER.
#
# Why: a bug that makes every request after the first produce a different greedy continuation is
# invisible to any gate that compares arm A against arm B at the same request index (both arms
# drift identically by index, so they always agree), and to perplexity / KLD (they request logits
# for every row, which makes the graph's output-row gather a no-op). The check that catches it
# needs no second arm and no reference: repeat one request and diff the answers against each other.
#
# usage:
#   repeat_gate.py --bin BUILD/bin/llama-server --model M.gguf [-- SERVER ARGS...]
#   repeat_gate.py --bin ... --model ... --reps 8 --predict 48 --prompt-file p.txt
#
#   repeat_gate.py ... --lock            # take the GPU lock ($KLT_LOCK) yourself; omit when a runner already holds it
#   repeat_gate.py --bin ... --model ... --multiturn --predict 800 [--prompt-file doc.txt] -- -c 16384
#   repeat_gate.py --selftest            # classifier controls for --multiturn, no server
#
# exit status: 0 = PASS, 1 = FAIL. A difference in request 0 only is a FAIL by default (request-order
# dependence); --allow-warmup downgrades it to WARN (exit 0). Refuses to start while $KLT_STOP exists.

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

def klt_home() -> str:
    return os.environ.get("KLT_HOME") or os.path.join(os.path.expanduser("~"), ".kernel-loop")


def klt_lock() -> str:   # same defaults as lib/klt-env.sh
    return os.environ.get("KLT_LOCK") or os.path.join(klt_home(), "gpu.lock")


def klt_stop() -> str:
    return os.environ.get("KLT_STOP") or os.path.join(klt_home(), "STOP-GPU")


# Several prompts of DIFFERENT token lengths, deliberately.
#
# This class of bug is extremely sensitive to where the ubatch boundaries land: a single
# trailing newline (one token) can flip the verdict either way. One prompt is therefore a coin
# toss, not a gate. These lengths straddle the -ub 512 split so the final ubatch has a
# different row count in each.
_PARA = ("The lightning indexer selects a sparse subset of key blocks for each query token. "
         "Correctness of that selection is what makes the attention output reproducible. ")
_TAIL = "Explain, step by step, why a tie in the selection scores can break greedy reproducibility."

DEFAULT_PROMPTS = {
    "long-nl":   _PARA * 40 + "\n" + _TAIL + "\n",
    "long-nonl": _PARA * 40 + "\n" + _TAIL,
    "mid":       _PARA * 18 + "\n" + _TAIL + "\n",
    "short":     _PARA * 4  + "\n" + _TAIL + "\n",
}


def wait_healthy(port: int, proc: subprocess.Popen, timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if b'"ok"' in r.read():
                    return True
        except Exception:
            pass
        time.sleep(2)
    return False


CACHE_PROMPT = False  # set by --cache-prompt: let the server reuse the cached prefill between identical requests
SUBTOKEN = 0          # set by --subtoken N: request n_probs=N and compare the per-position logprob streams

def generate(port: int, prompt: str, predict: int, seed: int) -> str:
    body = json.dumps({
        "prompt": prompt,
        "n_predict": predict,
        "temperature": 0,
        "top_k": 1,
        "cache_prompt": CACHE_PROMPT,   # default False: force a full re-prefill every time
        "ignore_eos": True,      # fixed length, so EOS variation cannot mask a difference
        "seed": seed,
        **({"n_probs": SUBTOKEN} if SUBTOKEN else {}),
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", body,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1800) as r:
            js = json.loads(r.read())
    except urllib.error.HTTPError as e:
        # a server-side parse failure is itself a difference worth reporting
        LAST_STREAM.append([])
        return f"<<HTTP-{e.code}>>"
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        # the server died mid-request (e.g. a GPU command-submission OOM);
        # report it as a distinct sentinel so the gate FAILs with a reason instead of crashing
        print(f"  request failed: {type(e).__name__}: {e}", flush=True)
        LAST_STREAM.append([])
        return f"<<DISCONNECT:{type(e).__name__}>>"
    LAST_STREAM.append(probs_stream(js) if SUBTOKEN else [])
    return js["content"]


LAST_STREAM: list = []   # per-request logprob streams, appended by generate() in call order


def probs_stream(js) -> list:
    """[( (id, logprob), (id, logprob), ... ), ...] per generated position, from completion_probabilities.
    Server layout (tools/server/server-task.cpp): each position is {id, token, bytes, logprob,
    top_logprobs:[{id, token, bytes, logprob}, ...]} with post_sampling_probs=false (the default), or
    {.., prob, top_probs:[...]} with it on. Candidate 0 is the position's own token."""
    import math
    def lp_of(c):
        if "logprob" in c:
            return float(c["logprob"])
        pr = float(c.get("prob", 0.0))
        return math.log(pr) if pr > 0 else float("-inf")
    def id_of(c):
        return c.get("id", c.get("token", c.get("tok_str")))
    out = []
    for pos in js.get("completion_probabilities", []) or []:
        row = [(id_of(pos), lp_of(pos))]
        for c in pos.get("top_logprobs") or pos.get("top_probs") or pos.get("probs") or []:
            row.append((id_of(c), lp_of(c)))
        out.append(tuple(row))
    return out


def check_subtoken(name: str, streams: list) -> str:
    """Compare every rep's logprob stream against rep 0. QUIET = all identical (values compared
    exactly, as the server reported them); DETECT otherwise. NB: under graph reuse the server's
    reported logprob for an early position can be aliased by later decodes, so the POSITION is
    indicative only; the verdict is exact."""
    if not streams or not streams[0]:
        print(f"[{name}] SUBTOKEN: no completion_probabilities in the response (server too old?)")
        return "ERROR"
    n_c0 = len(streams[0][0]) if streams[0] else 0
    if n_c0 < 2:
        print(f"[{name}] SUBTOKEN: ERROR - {n_c0} candidate(s) per position; the server returned no top_logprobs list, refusing a vacuous QUIET")
        return "ERROR"
    base = streams[0]
    n_pos = len(base); n_c = len(base[0]) if base else 0
    diffs = []
    for i, st in enumerate(streams[1:], 1):
        first = None; maxd = 0.0; ndiff = 0
        for p in range(min(len(base), len(st))):
            if base[p] != st[p]:
                ndiff += 1
                if first is None:
                    first = p
                for (ta, la), (tb, lb) in zip(base[p], st[p]):
                    if ta == tb and la == la and lb == lb:
                        maxd = max(maxd, abs(la - lb))
                    else:
                        maxd = float("inf")   # candidate identity changed: not a small delta
        if len(base) != len(st):
            ndiff += abs(len(base) - len(st)); first = first if first is not None else min(len(base), len(st))
        if first is not None:
            diffs.append((i, first, ndiff, maxd))
    if not diffs:
        print(f"[{name}] SUBTOKEN: QUIET - {len(streams)} reps, {n_pos} positions x {n_c} candidates, logprob streams identical")
        return "QUIET"
    worst = max(d[3] for d in diffs)
    earliest = min(d[1] for d in diffs)
    print(f"[{name}] SUBTOKEN: DETECT - {len(diffs)}/{len(streams)-1} reps differ from rep 0; "
          f"earliest position {earliest}; max |dlogprob| {'inf (candidate set changed)' if worst == float('inf') else f'{worst:.3e}'}")
    for i, first, ndiff, maxd in diffs[:8]:
        print(f"  rep {i}: first differing position {first}, {ndiff} positions differ, max |dlogprob| "
              f"{'inf' if maxd == float('inf') else f'{maxd:.3e}'}")
    return "DETECT"


def sentinel_count(outs: list) -> int:
    return sum(1 for o in outs if o.startswith("<<"))


def first_diff(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def check(name: str, outs: list[str]) -> str:
    n_sent = sentinel_count(outs)
    if n_sent:
        print(f"[{name}] {n_sent}/{len(outs)} requests failed ({sorted({o for o in outs if o.startswith('<<')})}); the server died or refused mid-run, so this is not a repeatability verdict")
        return "FAIL"
    uniq = len(set(outs))
    print(f"\n[{name}] {uniq} unique / {len(outs)} requests")
    for i, o in enumerate(outs):
        print(f"  rep {i}  sha {hashlib.sha256(o.encode()).hexdigest()[:16]}  len {len(o)}")
    if uniq == 1:
        return "PASS"
    # classify: first request differing from an otherwise stable tail is the known
    # warm-up signature; anything else is drift and is a hard failure
    if len(set(outs[1:])) == 1:
        j = first_diff(outs[0], outs[1])
        print(f"  rep 0 differs from a stable tail, first at char {j}")
        print(f"    rep0: {outs[0][j:j+60]!r}")
        print(f"    rest: {outs[1][j:j+60]!r}")
        return "WARN"
    for i, o in enumerate(outs[1:], 1):
        if o == outs[0]:
            continue
        j = first_diff(outs[0], o)
        print(f"  rep {i} first differs from rep 0 at char {j}: {outs[0][j:j+40]!r} vs {o[j:j+40]!r}")
    return "FAIL"


# ---------------------------------------------------------------------------------------------
# Multi-turn mode (--multiturn): the prompt cache and context checkpoints.
#
# The repeat gate above sends each request with cache_prompt:false, so it never exercises the
# path real chat traffic takes: turn N+1 reuses turn N's KV (and, on hybrid/recurrent models,
# restores a context checkpoint) and prefills only the new suffix. An edit-and-regenerate turn
# is the only traffic that makes the server roll back to a checkpoint mid-history; plain appends
# read the same whether --ctx-checkpoints is 1 or 32.
#
# Method: one conversation is run through /v1/chat/completions twice in one server process.
#   phase 1 (cached):    turns 1..T appended in order with cache_prompt:true, then one edit turn
#                        ([system, U1, A1, U2 edited]) so the divergence lands mid-history
#   phase 2 (reference): the identical message lists again with cache_prompt:false, i.e. a full
#                        re-prefill of each prompt
# Each cached turn is compared with its reference. The two are NOT expected to be bit-identical:
# a suffix prefill splits into different ubatches than a full one, and that moves logits at
# numerical scale. So a divergence is classified with the logprobs at the first differing
# position: if each side's chosen token is within --tie-eps nats of the other in BOTH streams it is
# a near-tie flip (WARN); if the other side's token is far below, or absent from the top list, the
# cached path computed something different (FAIL). Matching tokens whose logprob moved more than
# --max-drift nats before the divergence also FAIL.
#
# Engagement is checked, not assumed: every append turn after the first must report a cache hit
# covering at least the previous turn's prompt, otherwise the path under test never ran.

MT_QUESTIONS = [
    "Summarise the document above in three sentences.",
    "Which people, places, dates or numbers does it mention? List them with one line each.",
    "Continue the document with one more paragraph in the same style.",
    "What questions does the document leave unanswered? Give three.",
    "Rewrite the first paragraph for a ten-year-old reader.",
    "Which statement in the document would be the hardest to verify, and why?",
]
MT_EDIT = "Which dates or numbers does it mention, and what does each one refer to?"
MT_DEFAULT_DOC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts", "multiturn-doc.txt")


def mt_chat(port: int, messages: list, predict: int, seed: int, cache_prompt: bool, nprobs: int,
            no_think: bool = False) -> dict:
    body = json.dumps({
        "messages": messages,
        "max_tokens": predict,
        "temperature": 0,
        "top_k": 1,
        "seed": seed,
        "cache_prompt": cache_prompt,
        "logprobs": True,
        "top_logprobs": nprobs,
        **({"chat_template_kwargs": {"enable_thinking": False}} if no_think else {}),
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            js = json.loads(r.read())
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP-{e.code}: {e.read()[:200]!r}"}
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        return {"error": f"DISCONNECT:{type(e).__name__}: {e}"}
    ch = js["choices"][0]
    msg = ch.get("message", {})
    stream = []
    for pos in ((ch.get("logprobs") or {}).get("content") or []):
        cands = {c.get("id", c.get("token")): float(c["logprob"]) for c in pos.get("top_logprobs") or []}
        tok = pos.get("id", pos.get("token"))
        cands.setdefault(tok, float(pos["logprob"]))
        stream.append((tok, float(pos["logprob"]), cands))
    t = js.get("timings") or {}
    u = js.get("usage") or {}
    return {
        "content": msg.get("content") or "",
        "reasoning": msg.get("reasoning_content") or "",
        "stream": stream,
        "prompt_n": t.get("prompt_n"),
        "cache_n": t.get("cache_n"),
        "prompt_tokens": u.get("prompt_tokens"),
        "completion_tokens": u.get("completion_tokens"),
        "finish_reason": ch.get("finish_reason"),
    }


def mt_loop_period(text: str, span: int = 300, max_period: int = 200) -> int:
    """Period of a repetition covering the last `span` chars of text, or 0. Catches a triple quote
    repeated 400 times and sentence loops alike; normal prose never repeats exactly over 300 chars."""
    if len(text) < span:
        return 0
    tail = text[-span:]
    for p in range(1, max_period + 1):
        if all(tail[i] == tail[i - p] for i in range(p, span)):
            return p
    return 0


def mt_compare(cached: list, ref: list, tie_eps: float, max_drift: float) -> tuple:
    """(verdict, detail) for two logprob streams of the same request. Streams are lists of
    (token_id, logprob, {token_id: logprob}) per generated position."""
    drift = 0.0
    n = min(len(cached), len(ref))
    for p in range(n):
        tc, lc, cc = cached[p]
        tr, lr, cr = ref[p]
        if tc == tr:
            drift = max(drift, abs(lc - lr))
            if drift > max_drift:
                return "FAIL", f"token {p} matches but its logprob moved {drift:.3f} nats (> --max-drift {max_drift})"
            continue
        # first differing token: how far below the winner was the loser, in each stream?
        gap_ref = lr - cr[tc] if tc in cr else float("inf")     # reference's view of the cached pick
        gap_cached = lc - cc[tr] if tr in cc else float("inf")  # cached path's view of the reference pick
        gap = max(gap_ref, gap_cached)
        where = f"token {p}/{n}, gaps ref {gap_ref:.4f} / cached {gap_cached:.4f} nats, prefix drift {drift:.2e}"
        if gap <= tie_eps:
            return "WARN", f"near-tie flip at {where}"
        return "FAIL", f"divergence at {where} (> --tie-eps {tie_eps})"
    if len(cached) != len(ref):
        return "WARN", f"identical for {n} tokens, then one side stopped (lengths {len(cached)} vs {len(ref)}); prefix drift {drift:.2e}"
    return "PASS", f"{n} tokens identical, max logprob drift {drift:.2e}"


def mt_selftest() -> int:
    """Positive and negative controls for the classifier, no server needed. Each case states what a
    CORRECT classifier must return, so a broken one cannot pass by returning PASS everywhere."""
    same = [(1, -0.1, {1: -0.1, 2: -3.0}), (5, -0.2, {5: -0.2, 6: -2.5})]
    tie_a = [(1, -0.1, {1: -0.1, 2: -3.0}), (5, -0.690, {5: -0.690, 6: -0.700})]
    tie_b = [(1, -0.1, {1: -0.1, 2: -3.0}), (6, -0.692, {6: -0.692, 5: -0.699})]
    far_b = [(1, -0.1, {1: -0.1, 2: -3.0}), (9, -0.05, {9: -0.05, 8: -4.0})]           # ref never ranked 9
    drift_b = [(1, -0.1, {1: -0.1, 2: -3.0}), (5, -1.9, {5: -1.9, 6: -2.0})]
    cases = [
        ("identical streams", same, same, "PASS"),
        ("near-tie flip", tie_a, tie_b, "WARN"),
        ("confident different token", far_b, same, "FAIL"),
        ("same token, logprob moved 1.7 nats", drift_b, same, "FAIL"),
        ("early stop, identical prefix", same[:1], same, "WARN"),
    ]
    bad = 0
    for name, a, b, want in cases:
        got, detail = mt_compare(a, b, 0.05, 1.0)
        ok = got == want
        bad += not ok
        print(f"  {'ok  ' if ok else 'BAD '} {name}: want {want}, got {got} ({detail})")
    loops = [('"' * 400, True), ("the cat sat on the mat. " * 20, True),
             (open(MT_DEFAULT_DOC).read()[:2000] if os.path.exists(MT_DEFAULT_DOC) else "", False)]
    for text, want in loops:
        if not text:
            continue
        got = mt_loop_period(text) > 0
        bad += got != want
        print(f"  {'ok  ' if got == want else 'BAD '} loop detector on {text[:24]!r}...: want {want}, got {got}")
    print(f"SELFTEST: {'PASS' if not bad else f'FAIL ({bad} wrong)'}")
    return 1 if bad else 0


def run_multiturn(args, log_path: str | None) -> int:
    doc_path = args.prompt_file[0] if args.prompt_file else MT_DEFAULT_DOC
    doc = open(doc_path).read().strip()
    turns = max(2, min(args.turns, len(MT_QUESTIONS)))
    nprobs = SUBTOKEN or 5
    system = ("You answer questions about the document below. Answer from the document only.\n\n"
              "DOCUMENT:\n" + doc)

    # phase 1: the cached conversation, history built from the model's own answers
    requests = []   # (label, messages, is_append_after_first)
    cached = []
    history = [{"role": "system", "content": system}]
    empty_turns = 0
    for t in range(turns):
        history.append({"role": "user", "content": MT_QUESTIONS[t]})
        msgs = list(history)
        r = mt_chat(args.port, msgs, args.predict, args.seed, True, nprobs, args.no_think)
        requests.append((f"turn{t + 1}", msgs, t > 0))
        cached.append(r)
        if "error" in r:
            break
        history.append({"role": "assistant", "content": r["content"]})
        if not r["content"].strip():
            empty_turns += 1
        print(f"  cached turn{t + 1}: prompt {r['prompt_tokens']} tok, processed {r['prompt_n']}, "
              f"cache {r['cache_n']}, generated {r['completion_tokens']} ({r['finish_reason']})", flush=True)
    if all("error" not in r for r in cached) and len(history) >= 5:
        msgs = history[:4] + [{"role": "user", "content": MT_EDIT}]    # [sys, U1, A1, U2 edited]
        r = mt_chat(args.port, msgs, args.predict, args.seed, True, nprobs, args.no_think)
        requests.append(("edit-regen", msgs, False))
        cached.append(r)
        if "error" not in r:
            print(f"  cached edit-regen: prompt {r['prompt_tokens']} tok, processed {r['prompt_n']}, "
                  f"cache {r['cache_n']}, generated {r['completion_tokens']} ({r['finish_reason']})", flush=True)

    # phase 2: the same message lists, full re-prefill each
    refs = [mt_chat(args.port, msgs, args.predict, args.seed, False, nprobs, args.no_think) for _, msgs, _ in requests]

    verdicts = []
    print()
    prev_prompt = None
    for (label, msgs, is_append), c, ref in zip(requests, cached, refs):
        if "error" in c or "error" in ref:
            print(f"[{label}] FAIL - request error: cached {c.get('error')} / reference {ref.get('error')}")
            verdicts.append("FAIL")
            continue
        if not c["stream"] or len(next(iter(c["stream"]))[2]) < 2:
            print(f"[{label}] ERROR - no top_logprobs in the response; refusing a vacuous verdict")
            verdicts.append("FAIL")
            continue
        v, detail = mt_compare(c["stream"], ref["stream"], args.tie_eps, args.max_drift)
        notes = []
        # an append must reuse at least the previous request's prompt; a hybrid/recurrent model may
        # re-process the previous answer (it resumes from a checkpoint before it), so the answer is
        # not required to be cached. 64 tokens of slack for chat-template re-rendering at the seam.
        if is_append and prev_prompt is not None:
            if not c["cache_n"] or c["cache_n"] < prev_prompt - 64:
                v = "FAIL"
                notes.append(f"cache NOT used on an append turn (cached {c['cache_n']} of {c['prompt_tokens']}, "
                             f"previous prompt was {prev_prompt}): the path under test did not run")
        if label.startswith("turn"):
            prev_prompt = c["prompt_tokens"]
        if label == "edit-regen":
            notes.append(f"edit reprocessed {c['prompt_n']} of {c['prompt_tokens']} tokens "
                         f"({'full reset' if c['prompt_n'] == c['prompt_tokens'] else 'partial reuse'})")
        lc = mt_loop_period(c["reasoning"] + c["content"])
        lr = mt_loop_period(ref["reasoning"] + ref["content"])
        if lc and not lr:
            v = "FAIL"
            notes.append(f"cached output ends in a period-{lc} loop the reference does not have")
        elif lc:
            notes.append(f"both sides loop (period {lc}/{lr}): the model, not the cache path")
        print(f"[{label}] {v} - {detail}")
        for n in notes:
            print(f"  {n}")
        verdicts.append(v)
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            for side, res in (("cached", c), ("ref", ref)):
                with open(os.path.join(args.save_dir, f"mt-{label}.{side}.json"), "w") as f:
                    json.dump({"messages": msgs, **res}, f)

    if log_path:
        try:
            log = open(log_path, errors="replace").read()
            n_cr = log.count("created context checkpoint")
            n_rs = log.count("restored context checkpoint")
            lv4 = any(a in ("-lv", "--log-verbosity", "--verbosity") and i + 1 < len(args.server_args)
                      and args.server_args[i + 1] >= "4" for i, a in enumerate(args.server_args))
            if n_cr or n_rs:
                hint = ""
            elif lv4:
                hint = " (at -lv 4: expected for a pure-attention model, a defect for a hybrid/recurrent one)"
            else:
                hint = " (these lines only print at -lv 4; 0 here without it proves nothing)"
            print(f"\nserver log {log_path}: {n_cr} checkpoints created, {n_rs} restored{hint}")
        except OSError:
            pass

    if empty_turns:
        print(f"\nNOTE: {empty_turns}/{turns} answers had empty content (reasoning used all of --predict), so the "
              "history barely grew; raise --predict or pass --no-think for a conversation that builds up")
    print()
    if "FAIL" in verdicts:
        print("MULTITURN GATE: FAIL - the cached/checkpointed path disagrees with a full re-prefill, or did not engage")
        return 1
    n_warn = verdicts.count("WARN")
    if n_warn:
        print(f"MULTITURN GATE: PASS with {n_warn} near-tie flip(s) - ubatch-shape numerics, within --tie-eps")
        return 0
    print("MULTITURN GATE: PASS - every cached turn matches its full re-prefill")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="path to llama-server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--predict", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--cache-prompt", action="store_true", help="send cache_prompt:true (server default) instead of forcing a re-prefill")
    ap.add_argument("--allow-warmup", action="store_true", help="a rep-0-only difference is WARN (exit 0). Default is strict: any difference FAILS, because a rep-0-only difference is request-order dependence (state leaking from an earlier request), not warm-up")
    ap.add_argument("--subtoken", type=int, default=0, metavar="N", help="also request n_probs=N and gate on the per-position logprob streams (sub-token detector); DETECT fails the gate")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--save-dir", default=os.environ.get("REPEAT_GATE_SAVE_DIR"),
                    help="dump every prompt and every raw response into this directory")
    ap.add_argument("--prompt-file", action="append", default=[],
                    help="repeatable; defaults to four built-in prompts of different lengths")
    ap.add_argument("--load-timeout", type=int, default=2700)
    ap.add_argument("--lock", nargs="?", const=klt_lock(), default=None, metavar="PATH",
                    help="take this flock (bare --lock: $KLT_LOCK) to serialise against other GPU jobs; "
                         "omit when the caller already holds the GPU lock, or this deadlocks")
    ap.add_argument("--multiturn", action="store_true",
                    help="multi-turn + edit-and-regenerate conversation, cached vs full re-prefill "
                         "(see the comment block above run_multiturn); --prompt-file is the document, "
                         "--predict the tokens per turn")
    ap.add_argument("--turns", type=int, default=4, help="--multiturn: appended turns before the edit (2-6)")
    ap.add_argument("--no-think", action="store_true",
                    help="--multiturn: send chat_template_kwargs enable_thinking=false (reasoning models)")
    ap.add_argument("--tie-eps", type=float, default=0.15,
                    help="--multiturn: a divergence is a near-tie flip if both logprob gaps are within this many nats")
    ap.add_argument("--max-drift", type=float, default=1.0,
                    help="--multiturn: FAIL if a matching token's logprob moves more than this many nats")
    ap.add_argument("--selftest", action="store_true", help="run the --multiturn classifier controls and exit")
    ap.add_argument("server_args", nargs="*", help="extra llama-server args after --")
    if "--selftest" in sys.argv:
        return mt_selftest()
    args = ap.parse_args()
    global CACHE_PROMPT
    CACHE_PROMPT = bool(args.cache_prompt)
    global SUBTOKEN
    SUBTOKEN = max(0, int(args.subtoken))

    if args.prompt_file:
        prompts = {p: open(p).read() for p in args.prompt_file}
    else:
        prompts = DEFAULT_PROMPTS

    if os.path.exists(klt_stop()):
        print(f"REPEAT GATE: refusing to start, {klt_stop()} exists (device error recorded)", file=sys.stderr)
        return 70

    lock_fd = None
    if args.lock:
        import fcntl
        os.makedirs(os.path.dirname(os.path.abspath(args.lock)), exist_ok=True)
        lock_fd = open(args.lock, "w")
        print(f"waiting for GPU lock {args.lock} ...", flush=True)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        with open(args.lock + ".owner", "w") as f:
            f.write(f"pid={os.getpid()} label=repeat_gate started={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")

    cmd = [args.bin, "-m", args.model, "--port", str(args.port),
           "--host", "127.0.0.1", "-np", "1"] + args.server_args
    print("+ " + " ".join(cmd), flush=True)
    log_path = log_f = None
    if args.multiturn:
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            log_path = os.path.join(args.save_dir, "server.log")
        else:
            import tempfile
            log_path = tempfile.mkstemp(prefix="repeat-gate-server-", suffix=".log")[1]
        log_f = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_f or subprocess.DEVNULL, stderr=log_f or subprocess.DEVNULL)
    try:
        if not wait_healthy(args.port, proc, args.load_timeout):
            print("FAIL: server did not become healthy", file=sys.stderr)
            return 1
        if args.multiturn:
            return run_multiturn(args, log_path)
        verdicts = []
        sub_verdicts = []
        for name, prompt in prompts.items():
            LAST_STREAM.clear()
            outs = [generate(args.port, prompt, args.predict, args.seed)
                    for _ in range(args.reps)]
            streams = list(LAST_STREAM)
            if args.save_dir:
                os.makedirs(args.save_dir, exist_ok=True)
                with open(os.path.join(args.save_dir, f"{os.path.basename(name)}.prompt.txt"), "w") as f:
                    f.write(prompt)
                for i, o in enumerate(outs):
                    with open(os.path.join(args.save_dir, f"{os.path.basename(name)}.rep{i}.txt"), "w") as f:
                        f.write(o)
                if SUBTOKEN:
                    with open(os.path.join(args.save_dir, f"{os.path.basename(name)}.probs.json"), "w") as f:
                        json.dump(streams, f)
            verdicts.append(check(name, outs))
            if SUBTOKEN:
                sub_verdicts.append(check_subtoken(name, streams))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        if log_f:
            log_f.close()
        if lock_fd:
            try:
                os.unlink(args.lock + ".owner")
            except OSError:
                pass
            lock_fd.close()

    print()
    sub_rc = 0
    if SUBTOKEN:
        if "DETECT" in sub_verdicts or "ERROR" in sub_verdicts:
            print("SUBTOKEN GATE: DETECT - logprob streams differ between identical requests (or could not be read)")
            sub_rc = 1
        else:
            print("SUBTOKEN GATE: QUIET - logprob streams identical across all requests")
    if "FAIL" in verdicts:
        print("REPEAT GATE: FAIL - identical requests produced different output")
        return 1
    if "WARN" in verdicts:
        if args.allow_warmup:
            print("REPEAT GATE: WARN - only the first request differs (allowed by --allow-warmup)")
            return sub_rc
        print("REPEAT GATE: FAIL - only the first request differs; that is request-order dependence, not warm-up. Use --allow-warmup to report it as WARN")
        return 1
    print("REPEAT GATE: PASS - all requests byte-identical")
    return sub_rc


if __name__ == "__main__":
    sys.exit(main())

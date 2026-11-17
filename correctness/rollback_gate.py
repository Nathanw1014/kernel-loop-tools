#!/usr/bin/env python3
# rollback_gate.py: bit-exact cache rollback / checkpoint-restore invariance gate for llama-server.
#
#   rollback_gate.py --bin <llama-server> --model <gguf> --prompt-file <text> [--port N] [--out DIR]
#                    [--min-avail-gb G] [--gtt-max-gib G] [--settle-s S] -- <server args>
#
# Each gate arm runs on a FRESH server launch (one slot), so the cache layout is identical. Every arm runs the same request sequence with cache_prompt:true:
#   Q (unrelated, resets the slot)  ->  P  ->  P+X (+G generated tokens)  ->  P+Y (+N generated)
# The arms differ only in the CONTENT of the discarded branch X (same length, same G), so the server
# takes the identical path in every arm: the same lcp, the same seq_rm or checkpoint restore, the same
# chunk boundaries for everything it reprocesses. The P+Y result (token ids and top-k logprobs at every
# generated position) must therefore be BIT-IDENTICAL across arms. A difference means the discarded
# branch leaked into the kept state (stale KV cells visible to attention, a checkpoint restored from the
# wrong point, recurrent state not rolled back).
#
#   null      arm A run twice                 must match (determinism; else the gate is meaningless)
#   short     X1 vs X2, |X|+G inside the SWA window   -> the in-place seq_rm path
#   long      X1 vs X2, |X|+G past the SWA window     -> checkpoint restore on SWA / hybrid models
#   control   P+Y vs P'+Y, P' = P with one token changed mid-prompt  must DIFFER (the compare resolves a
#             one-token change in the kept prefix; a leak of a whole branch is far larger)
# n_probs output can be launch-dependent on some models; the null control (arm A on two launches) checks that here.
# Before every launch: wait for other processes' GTT use <= --gtt-max-gib and MemAvailable >= --min-avail-gb, then --settle-s.
# Refuses to start while $KLT_STOP exists. arms.json in --out holds every arm's stream (compare two runs: arms-compare.py).
import argparse, glob, json, os, subprocess, sys, time, urllib.request

def klt_env(name, default):   # same defaults as lib/klt-env.sh
    home = os.environ.get("KLT_HOME") or os.path.join(os.path.expanduser("~"), ".kernel-loop")
    return os.environ.get(name) or os.path.join(home, default)

def gpu_card():   # KLT_GPU_CARD, else the first amdgpu card with gpu_busy_percent (lib/klt-env.sh klt_gpu_card)
    if os.environ.get("KLT_GPU_CARD"):
        return os.environ["KLT_GPU_CARD"]
    for d in sorted(glob.glob("/sys/class/drm/card*/device")):
        if os.access(d + "/gpu_busy_percent", os.R_OK):
            return d
    return None

def post(port, path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())

def tok(port, text):
    return post(port, "/tokenize", {"content": text, "add_special": False})["tokens"]

def complete(port, tokens, n):
    js = post(port, "/completion", {"prompt": tokens, "n_predict": n, "temperature": 0, "top_k": 1, "seed": 1,
                                     "cache_prompt": True, "n_probs": 8, "ignore_eos": True, "return_tokens": True})
    probs = [(p.get("id"), round(p.get("logprob", 0.0), 9),
              tuple((c.get("id"), c.get("logprob")) for c in p.get("top_logprobs", [])))
             for p in js.get("completion_probabilities", [])]
    t = js.get("timings", {})
    return {"tokens": js.get("tokens"), "probs": probs, "cache_n": t.get("cache_n"), "prompt_n": t.get("prompt_n")}

def run_arm(port, bos, Q, P, X, Y, G, N):
    complete(port, bos + Q, 1)
    complete(port, bos + P, 1)
    complete(port, bos + P + X, G)
    return complete(port, bos + P + Y, N)

def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a["probs"], b["probs"])):
        if x != y:
            return i, x, y
    return (None if len(a["probs"]) == len(b["probs"]) else min(len(a["probs"]), len(b["probs"]))), None, None

class Server:
    def __init__(self, a, tag):
        self.a = a
        # every model load waits for the previous process's memory to drain, then a settle period:
        # back-to-back launches can otherwise load into memory that is not free yet
        card = gpu_card()
        while True:
            gtt = (int(open(card + "/mem_info_gtt_used").read()) >> 30) if card else 0
            avail = next(int(l.split()[1]) for l in open("/proc/meminfo") if l.startswith("MemAvailable")) >> 20
            if gtt <= a.gtt_max_gib and avail >= a.min_avail_gb:
                break
            time.sleep(5)
        time.sleep(a.settle_s)
        self.log = open(os.path.join(a.out, f"server-{tag}.log"), "w")
        self.p = subprocess.Popen([a.bin, "-m", a.model, "--port", str(a.port), "-np", "1", "-lv", "4"] + a.server_args,
                                  stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while True:
            if self.p.poll() is not None:
                raise RuntimeError("server exited during load")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/health", timeout=5).read(); return
            except Exception:
                if time.time() - t0 > a.load_timeout:
                    self.stop(); raise RuntimeError("load timeout")
                time.sleep(2)
    def stop(self):
        self.p.terminate()
        try: self.p.wait(60)
        except subprocess.TimeoutExpired: self.p.kill(); self.p.wait()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True); ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8750); ap.add_argument("--out", default=".")
    ap.add_argument("--prompt-file", required=True); ap.add_argument("--long-x", type=int, default=2500)
    ap.add_argument("--load-timeout", type=int, default=1800)
    ap.add_argument("--min-avail-gb", type=int, default=0,
                    help="MemAvailable floor before each launch (default 0 = off; set to the model's resident size plus a few GB)")
    ap.add_argument("--gtt-max-gib", type=int, default=2, help="wait until GPU GTT in use is at most this before each launch")
    ap.add_argument("--settle-s", type=int, default=30, help="sleep after the memory gate before each launch")
    ap.add_argument("server_args", nargs="*")
    a = ap.parse_args()
    if os.path.exists(klt_env("KLT_STOP", "STOP-GPU")):
        print(f"ROLLBACK GATE: refusing to start, {klt_env('KLT_STOP', 'STOP-GPU')} exists (device error recorded)"); return 70
    os.makedirs(a.out, exist_ok=True)
    words = open(a.prompt_file).read().split()
    srv = Server(a, "tok")
    try:
        bos = post(a.port, "/tokenize", {"content": "", "add_special": True})["tokens"]
        P = tok(a.port, " ".join(words[:700]))[:1000]
        Q = tok(a.port, "Unrelated reset prompt about gardening tools and soil.")
        Y = tok(a.port, "\n\nQuestion: summarise the passage above in two sentences.\nAnswer:")
        pool = tok(a.port, " ".join(words[700:9000]))
        G, N = 16, 32
        X = lambda n, off: pool[off:off + n]
        # diagnostic, ONE launch: the same arm three times (fresh cache, then reused cells)
        d1 = run_arm(a.port, bos, Q, P, X(48, 0), Y, G, N)
        d2 = run_arm(a.port, bos, Q, P, X(48, 0), Y, G, N)
        d3 = run_arm(a.port, bos, Q, P, X(48, 0), Y, G, N)
    finally:
        srv.stop()
    same = lambda x, y: first_diff(x, y)[0] is None
    print(f"[diag    one launch] arm x3: 1==2 {same(d1, d2)}, 2==3 {same(d2, d3)}, 1==3 {same(d1, d3)} "
          "(fresh cache vs reused cells; all True = in-launch state is clean)")
    P2 = list(P); P2[len(P2) // 2] = pool[5000]
    arms = {"A1": (P, X(48, 0)), "A2": (P, X(48, 0)), "B": (P, X(48, 3000)),
            "L1": (P, X(a.long_x, 0)), "L2": (P, X(a.long_x, 3000)), "C": (P2, X(48, 0))}
    res = {}
    for name, (p, x) in arms.items():      # gate: every arm on a FRESH launch, identical cache layout
        srv = Server(a, name)
        try:
            res[name] = run_arm(a.port, bos, Q, p, x, Y, G, N)
        finally:
            srv.stop()
    verdicts = []
    def cmp(name, x, y, must_match=True):
        i, u, v = first_diff(x, y)
        ok = (i is None) if must_match else (i is not None)
        det = "bit-identical" if i is None else f"first difference at generated token {i}: {u[:2] if u else u} vs {v[:2] if v else v}"
        print(f"[{name}] {'PASS' if ok else 'FAIL'} - {det} (cache_n {x['cache_n']}/{y['cache_n']}, prompt_n {x['prompt_n']}/{y['prompt_n']})")
        verdicts.append(ok)
    cmp("null    A vs A", res["A1"], res["A2"])
    cmp("short   X1 vs X2", res["A1"], res["B"])
    cmp("long    X1 vs X2", res["L1"], res["L2"])
    cmp("control P vs P'", res["A1"], res["C"], must_match=False)
    json.dump({"diag": [d1, d2, d3], **res}, open(os.path.join(a.out, "arms.json"), "w"))
    if not verdicts[0]:
        print("ROLLBACK GATE: INVALID - null control differs across launches, results meaningless"); return 1
    if not verdicts[3]:
        print("ROLLBACK GATE: INVALID - positive control did not differ"); return 1
    print("ROLLBACK GATE: " + ("PASS - discarded branches never reach the kept state" if all(verdicts) else "FAIL - a discarded branch changed the result"))
    return 0 if all(verdicts) else 1

if __name__ == "__main__":
    sys.exit(main())

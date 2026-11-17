#!/usr/bin/env python3
"""suite-plan.py: shard planner, merger, op profiler and line timestamper for the job-run `suite` cell (jobs/job-suite).

  suite-plan.py plan  <profile.json> <N> <plan.tsv>
      Split one full test-backend-ops test run into <= N shard processes balanced by the per-op seconds in the profile.
      plan.tsv lines: idx <TAB> -o filter <TAB> -p filter or '-' <TAB> est seconds <TAB> label.
      Coverage is exact by construction, also for ops the profile has never seen:
        * a heavy op gets its own shard(s) (-o NAME, exact); a very heavy op is split by the value of its first param
          (-p '^(?:g1|g2)(?:,|$)' parts plus one complement part -p '^(?!(?:g1|g2)(?:,|$))', so unseen values land there);
        * every other op name is covered by the alternatives of the trie complement of the dedicated names (^P$, ^P[^X], ^P.),
          disjoint by construction and packed into the remaining shards. test-backend-ops treats a -o entry with '(' as a
          full case string and splits entries on ',', so these regexes use neither (top-level '|' only).
  suite-plan.py uniform <ops.txt> <profile.json>
      Write a stand-in profile with weight 1 per op name (one name per line, e.g. from `test-backend-ops --list-ops`), for the
      first run on a machine with no recorded profile: shards by op count, coverage still exact.
  suite-plan.py merge <shards dir> <N> <wall s> [profile-in] [profile-out] [binsha] [args] [host]
      Merge shards/<i>.log (+ .ts, .rc) into one test-backend-ops shaped console output on stdout (same "N/M tests passed",
      "Failing tests:", "Backend X: OK|FAIL", "k/n backends passed" lines), write shards/merged.json, exit 0 only if every
      backend passed, no shard crashed and the partition check holds. Writes profile-out (per-op seconds) for a full run.
  suite-plan.py ts <log> <ts>
      Copy stdin to <log> unchanged and write one epoch time per line to <ts> (same line order); the per-line times give the
      per-op / per-case seconds of the profile. Writes <log>.done at EOF so job-suite knows the log is complete.
"""
import json, math, os, re, sys, time

ANSI = re.compile(r"\x1b\[[0-9;]*m")
CASE = re.compile(r"^(?:  |\[[A-Za-z0-9_]+\] .*? {3})([A-Za-z0-9_]+)\((.*)\): (.*)$")  # a failed compare prints "[OP] ERR = a > b   NAME(..): FAIL"
PASSED = re.compile(r"^  (\d+)/(\d+) tests passed")
BINIT = re.compile(r"^Backend (\d+)/(\d+): (.+)$")
BSTAT = re.compile(r"^  Backend (.+): (OK|FAIL)$")
OVER = re.compile(r"^(\d+)/(\d+) backends passed$")
GREEN, RED, RST = "\x1b[1;32m", "\x1b[1;31m", "\x1b[0m"


def split_vars(v):  # top-level comma split of a test case's params
    out, d, st = [], 0, 0
    for i, c in enumerate(v):
        if c in "[(":
            d += 1
        elif c in "])":
            d -= 1
        elif c == "," and d == 0:
            out.append(v[st:i])
            st = i + 1
    out.append(v[st:])
    return out


def first_var(v):
    return split_vars(v)[0]


def groups(o, depth):  # {prefix of the first <depth> params: [t, n]} from the per-case times (old profiles: first-param groups)
    if "c" not in o:
        return {k: list(x) for k, x in o.get("g", {}).items()}
    g = {}
    for v, (t, n) in o["c"].items():
        k = ",".join(split_vars(v)[:depth])
        x = g.setdefault(k, [0.0, 0])
        x[0] += t
        x[1] += n
    return g


def pick_groups(o, target):  # coarsest param-prefix depth whose heaviest group fits a shard (else per case)
    last = None
    for depth in range(1, 64):
        g = groups(o, depth)
        if g == last:
            break
        last = g
        if not g or max(x[0] for x in g.values()) <= target * 0.9:
            break
    return last


def resc(s):  # ECMAScript regex literal
    return re.sub(r"([\\^$.|?*+()\[\]{}/])", r"\\\1", s)


def cesc(c):  # inside a [...] class
    return "\\" + c if c in "\\]^-[" else c


# ---- trie complement: alternatives are (kind, prefix, excluded chars); kinds end '^P$', div '^P[^X]' (X empty -> '^P.'), lit '^P'
def alt_re(a):
    k, p, x = a
    if k == "end":
        return "^" + resc(p) + "$"
    if k == "lit":
        return "^" + resc(p)
    return "^" + resc(p) + ("[^" + "".join(cesc(c) for c in sorted(x)) + "]" if x else ".")


def alt_match(a, name):
    k, p, x = a
    if k == "end":
        return name == p
    if k == "lit":
        return name.startswith(p)
    return name.startswith(p) and len(name) > len(p) and name[len(p)] not in x


def complement(S):
    alts = []

    def rec(p):
        kids = sorted({s[len(p)] for s in S if s.startswith(p) and len(s) > len(p)})
        if p not in S and p:
            alts.append(("end", p, frozenset()))
        alts.append(("div", p, frozenset(kids)))
        for c in kids:
            rec(p + c)

    rec("")
    return alts


def expand(a, names):
    k, p, x = a
    if k == "end":
        return None
    if k == "lit":
        return [("end", p, frozenset())] + (expand(("div", p, frozenset()), names) or [("div", p, frozenset())])
    nxt = sorted({n[len(p)] for n in names if alt_match(a, n)})
    if not nxt:
        return None
    return [("lit", p + c, frozenset()) for c in nxt] + [("div", p, frozenset(x | set(nxt)))]


def lpt(items, k):  # items: (weight, payload) -> k bins [(w, [payloads])]
    bins = [[0.0, []] for _ in range(k)]
    for w, it in sorted(items, key=lambda t: -t[0]):
        b = min(bins, key=lambda b: b[0])
        b[0] += w
        b[1].append(it)
    return bins


def plan(prof_path, n, out):
    prof = json.load(open(prof_path))
    ops = prof["ops"]
    T = sum(o["t"] for o in ops.values()) or 1.0
    n = max(1, n)
    rows = []
    if n == 1:
        rows.append(("", "-", T, "all"))
    else:
        target = T / n
        heavy = sorted([op for op, o in ops.items() if o["t"] > target * 0.8], key=lambda op: -ops[op]["t"])
        ks, gsel = {}, {}
        for op in heavy:
            gsel[op] = pick_groups(ops[op], target)
            ng = len(gsel[op])
            ks[op] = max(1, min(ng, round(ops[op]["t"] / target))) if ops[op]["t"] > target * 1.25 and ng >= 2 else 1
        while heavy and sum(ks.values()) > n - 1:  # keep >= 1 shard for the rest: shrink splits, then drop the lightest heavy op
            sp = [op for op in heavy if ks[op] > 1]
            if sp:
                ks[min(sp, key=lambda op: ops[op]["t"] / ks[op])] -= 1
            else:
                del ks[heavy.pop()]
        S = set(ks)
        for op in sorted(S, key=lambda op: -ops[op]["t"]):
            on = op if re.fullmatch(r"[A-Z0-9_]+", op) else "^" + resc(op) + "$"
            if ks[op] == 1:
                rows.append((on, "-", ops[op]["t"], op))
                continue
            g = gsel[op]
            parts = lpt([(v[0], gk) for gk, v in g.items()], ks[op])
            parts.sort(key=lambda b: -len(b[1]))  # the part with the most groups becomes the complement
            rest = [gk for b in parts[1:] for gk in b[1]]
            alt = "(?:" + "|".join(resc(gk) for gk in sorted(rest)) + ")(?:,|$)"
            rows.append((on, "^(?!" + alt + ")", parts[0][0], "%s rest(%d groups + unseen)" % (op, len(parts[0][1]))))
            for b in parts[1:]:
                rows.append((on, "^(?:" + "|".join(resc(gk) for gk in sorted(b[1])) + ")(?:,|$)", b[0],
                             "%s %s" % (op, ",".join(sorted(b[1]))[:60])))
        k = n - len(rows)
        names = [op for op in ops if op not in S]
        w = lambda a: sum(ops[nm]["t"] for nm in names if alt_match(a, nm))
        alts = complement(S)
        tr = (T - sum(ops[op]["t"] for op in S)) / max(1, k)
        for _ in range(4000):  # expand the heaviest multi-op alternative until each one fits a shard
            cand = [(w(a), a) for a in alts if sum(1 for nm in names if alt_match(a, nm)) > 1]
            cand = [c for c in cand if c[0] > tr * 0.35]
            if not cand:
                break
            a = max(cand, key=lambda c: c[0])[1]
            e = expand(a, names)
            if not e:
                break
            alts.remove(a)
            alts.extend(e)
        for bw, bl in lpt([(w(a), a) for a in alts], k):
            if not bl:
                continue
            o = "|".join(alt_re(a) for a in sorted(bl, key=alt_re))
            assert "(" not in o and "," not in o, o
            mem = sorted(nm for nm in names if any(alt_match(a, nm) for a in bl))
            rows.append((o, "-", bw, "%d ops: %s" % (len(mem), ",".join(mem)[:80])))
    with open(out, "w") as f:
        for i, (o, p, est, lab) in enumerate(rows):
            f.write("%d\t%s\t%s\t%.0f\t%s\n" % (i, o or "-", p, est, lab))
    print("plan: %d shards from profile %s (%s, %d cases, %.0f s of case time); est max shard %.0f s" % (
        len(rows), os.path.basename(prof_path), prof.get("date", "?"), prof.get("cases", 0), T, max(r[2] for r in rows)))


def uniform(ops_path, out):
    names = sorted({l.strip() for l in open(ops_path) if l.strip()})
    if not names:
        print("uniform: no op names in %s" % ops_path, file=sys.stderr)
        sys.exit(1)
    ops = {nm: {"t": 1.0, "n": 1, "c": {}} for nm in names}
    json.dump({"uniform": True, "date": "uniform", "cases": 0, "ops": ops}, open(out, "w"))
    print("uniform: %d op names, weight 1 each" % len(names))


def stamp(log, tsf):
    with open(log, "wb") as lo, open(tsf, "w") as ts:
        for line in sys.stdin.buffer:
            lo.write(line)
            lo.flush()
            ts.write("%.3f\n" % time.time())
            ts.flush()
    open(log + ".done", "w").close()


# ---- merge
def parse(log, tsf):
    raw = open(log, errors="replace").read().splitlines() if os.path.exists(log) else []
    ts = [float(x) for x in open(tsf).read().split()] if os.path.exists(tsf) else []
    sh = {"head": [], "sections": [], "over": None, "cases": [], "lines": raw}
    cur = None
    prev = ts[0] if ts else 0.0
    infail = False
    for i, l in enumerate(raw):
        c = ANSI.sub("", l)
        t = ts[i] if i < len(ts) else prev
        dt, prev = t - prev, t
        m = BINIT.match(c)
        if m:
            cur = {"idx": int(m.group(1)), "n": int(m.group(2)), "name": m.group(3), "head": [l], "body": [], "skip": False,
                   "passed": None, "fail": [], "status": None, "inhead": True}
            sh["sections"].append(cur)
            infail = False
            continue
        if OVER.match(c):
            sh["over"] = c
            cur = None
            continue
        if cur is None:
            if not sh["sections"] and sh["over"] is None:
                sh["head"].append(l)
            continue
        if cur["inhead"]:
            if c.startswith("  Device ") or c == "":
                cur["head"].append(l)
                continue
            if len(cur["head"]) == 1 and c.startswith("  ") and not CASE.match(c):  # "  Skipping ..." right after the init line
                cur["head"].append(l)
                cur["skip"] = True
                continue
            cur["inhead"] = False
        m = PASSED.match(c)
        if m:
            cur["passed"] = (int(m.group(1)), int(m.group(2)))
            continue
        if c == "Failing tests:":
            infail = True
            continue
        m = BSTAT.match(c)
        if m:
            cur["status"] = m.group(2)
            infail = False
            continue
        if infail:
            if c.strip():
                cur["fail"].append(c.strip())
            continue
        if c == "" and cur["passed"] is not None:
            continue
        cur["body"].append(l)
        m = CASE.match(c)
        if m:
            st = m.group(3)
            k = "OK" if st.endswith("OK") else "FAIL" if "FAIL" in st else "NS" if st.startswith("not supported") else "OTHER"
            sh["cases"].append((m.group(1), m.group(2), k, dt))
    return sh


def merge(d, n, wall, prof_in, prof_out, binsha, args, host):
    shards = []
    for i in range(n):
        rc = open(os.path.join(d, "%d.rc" % i)).read().strip() if os.path.exists(os.path.join(d, "%d.rc" % i)) else "?"
        s = parse(os.path.join(d, "%d.log" % i), os.path.join(d, "%d.ts" % i))
        s["rc"] = rc
        s["i"] = i
        shards.append(s)
    base = next((s for s in shards if s["over"]), None) or next((s for s in shards if s["sections"]), shards[0])
    out = list(base["head"])
    ok_all, crashes, nbok = True, [], 0
    tot_p = tot_t = 0
    fails = []
    for sec in base["sections"]:
        if sec["skip"]:
            out += sec["head"]
            nbok += 1
            continue
        out += sec["head"]
        P = T = 0
        sok = True
        for s in shards:
            ss = next((x for x in s["sections"] if x["idx"] == sec["idx"]), None)
            if ss is None or ss["passed"] is None or ss["status"] is None:
                last = ANSI.sub("", s["lines"][-1]) if s["lines"] else "(no output)"
                out += (ss["body"] if ss else [])
                out.append("  suite shard %d/%d CRASHED rc=%s after %d case lines (last: %s): FAIL" % (
                    s["i"], n, s["rc"], len(s["cases"]), last[:120]))
                crashes.append(s["i"])
                sok = False
                continue
            out += ss["body"]
            P += ss["passed"][0]
            T += ss["passed"][1]
            for f in ss["fail"]:
                if f not in fails:
                    fails.append(f)
            if ss["status"] != "OK":
                sok = False
        out.append("  %d/%d tests passed" % (P, T))
        tot_p += P
        tot_t += T
        if fails:
            out += ["", "Failing tests:"] + ["  " + f for f in fails]
        out.append("  Backend %s: %s" % (sec["name"], (GREEN + "OK" + RST) if sok else (RED + "FAIL" + RST)))
        ok_all &= sok
        nbok += sok
    nb = base["sections"][0]["n"] if base["sections"] else 0
    if not base["sections"]:
        ok_all = False
        out.append("  suite: no shard printed a backend section: FAIL")
    # partition check: no op in two shards unless they are -p parts of a split op; the case lines of split parts are disjoint;
    # the case-line count equals the profile's when the profile came from this very binary with the same args
    plan = {}
    pf = os.path.join(d, "plan.tsv")
    if os.path.exists(pf):
        for l in open(pf):
            i, o, p, est, lab = l.rstrip("\n").split("\t")
            plan[int(i)] = (o, p)
    key_sh, op_sh = {}, {}
    for s in shards:
        for op, v, k, dt in s["cases"]:
            key_sh.setdefault((op, v), set()).add(s["i"])
            op_sh.setdefault(op, set()).add(s["i"])
    dup = ["%s(%s) in shards %s" % (op, v[:60], sorted(ws)) for (op, v), ws in key_sh.items() if len(ws) > 1]
    dup += ["%s in shards %s" % (op, sorted(ws)) for op, ws in op_sh.items()
            if len(ws) > 1 and any(plan.get(i, ("", "-"))[1] == "-" for i in ws)]
    ncases = sum(len(s["cases"]) for s in shards)
    pin = json.load(open(prof_in)) if prof_in and os.path.exists(prof_in) else None
    part = "ok"
    if dup:
        part = "OVERLAP " + "; ".join(dup[:5])
    elif pin and pin.get("binsha") == binsha and pin.get("args") == args and n > 1:
        if pin.get("cases") != ncases:
            part = "COUNT %d case lines vs %d in the unsharded profile of this binary" % (ncases, pin.get("cases"))
        else:
            part = "ok (%d case lines = profile of this binary)" % ncases
    else:
        part = "ok (%d case lines; no same-binary profile to compare)" % ncases
    if not part.startswith("ok"):
        ok_all = False
    out.append("%d/%d backends passed" % (nbok, nb))
    out.append((GREEN + "OK" + RST) if ok_all and not crashes else (RED + "FAIL" + RST))
    if not part.startswith("ok"):
        out.append("suite: PARTITION FAIL: " + part)
    rc = 0 if ok_all and not crashes and nbok == nb else 1
    cpu_case = sum(dt for s in shards for *_x, dt in s["cases"])
    out.append("suite: shards=%d wall=%.0fs cases=%d passed=%d/%d fails=%d crashes=%d partition=%s case-time=%.0fs" % (
        n, wall, ncases, tot_p, tot_t, len(fails), len(crashes), part.split(" ")[0], cpu_case))
    print("\n".join(out))
    json.dump({"rc": rc, "passed": tot_p, "total": tot_t, "fails": fails, "crashes": crashes, "partition": part,
               "cases": ncases, "wall": wall, "shards": n}, open(os.path.join(d, "merged.json"), "w"))
    if prof_out and not crashes:
        ops = {}
        for s in shards:
            for op, v, k, dt in s["cases"]:
                o = ops.setdefault(op, {"t": 0.0, "n": 0, "c": {}})
                o["t"] += dt
                o["n"] += 1
                c = o["c"].setdefault(v, [0.0, 0])
                c[0] += dt
                c[1] += 1
        for o in ops.values():
            o["t"] = round(o["t"], 3)
            o["c"] = {k: [round(v[0], 3), v[1]] for k, v in o["c"].items()}
        tmp = prof_out + ".tmp%d" % os.getpid()
        json.dump({"host": host, "date": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()), "binsha": binsha, "args": args,
                   "shards": n, "wall": round(wall), "cases": ncases, "source": os.path.dirname(os.path.abspath(d)), "ops": ops},
                  open(tmp, "w"))
        os.replace(tmp, prof_out)
    return rc


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "plan" and len(a) == 4:
        plan(a[1], int(a[2]), a[3])
    elif a and a[0] == "uniform" and len(a) == 3:
        uniform(a[1], a[2])
    elif a and a[0] == "ts" and len(a) == 3:
        stamp(a[1], a[2])
    elif a and a[0] == "merge" and len(a) >= 4:
        a += [""] * (9 - len(a))
        sys.exit(merge(a[1], int(a[2]), float(a[3]), a[4], a[5], a[6], a[7], a[8]))
    else:
        print(__doc__)
        sys.exit(2)

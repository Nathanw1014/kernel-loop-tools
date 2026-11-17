#!/usr/bin/env python3
"""Wave-level timeline from a decoded RADV/ATT thread trace.

  wavetrace.py check   <dir>              validate the decode (clock sync, window)
  wavetrace.py summary <dir>              per-dispatch wave table
  wavetrace.py trace   <dir> out.json     Chrome/Perfetto trace: one row per wave slot
  wavetrace.py isa     <dir> [codeobj]    per-instruction-class latency/stall (timing-mode captures)

<dir> is an rgpwaves.py extract dir that decode.sh has run on (holds decoded/,
markers.json, capture-info.json). Times are on the decoder's realtime axis (100 MHz
GPU timestamp, 10 ns ticks), fitted per shader engine from realtime.json pairs.
"""
import json
import sys
from pathlib import Path

REALTIME_NS = 10.0
FIELDS = ["time", "cu", "simd", "wave_id", "start", "kernel_id", "me_id", "pipe_id",
          "is_ext", "workgroup_id", "cluster_id"]


def load(d):
    d = Path(d)
    dec = d / "decoded"
    occ = json.loads((dec / "occupancy.json").read_text())
    rt = json.loads((dec / "realtime.json").read_text())
    markers = json.loads((d / "markers.json").read_text()) if (d / "markers.json").exists() else []
    info = json.loads((d / "capture-info.json").read_text()) if (d / "capture-info.json").exists() else {}
    man = d / "radv-manifest.json"
    by_pso = {o["api_pso"]: o["name"] for o in json.loads(man.read_text())["codeObjects"]
              if o.get("api_pso") is not None} if man.exists() else {}
    for m in markers:
        m["_by_pso"] = by_pso
    if occ.get("occupancy_fields") != FIELDS:
        raise ValueError(f"unexpected occupancy schema {occ.get('occupancy_fields')}")
    return occ, rt, markers, info


def pairs(rt, se):
    return [(float(g), float(r)) for g, r in rt.get(f"SE{se}", [])]


def anomalies(pts):
    """Port of rocprof studio realtime_anomalies: shader clock vs realtime jumps."""
    w = list(zip(pts, pts[1:]))
    ratios = sorted((b[0] - a[0]) / (b[1] - a[1]) for a, b in w if b[1] > a[1])
    ratios = [r for r in ratios if 0 < r < 1e4]
    cadence = sorted(b[1] - a[1] for a, b in w if b[1] > a[1])
    if not ratios or not cadence:
        return None
    med, step = ratios[len(ratios) // 2], cadence[len(cadence) // 2]
    bad = 0
    for a, b in w:
        dg, dr = b[0] - a[0], b[1] - a[1]
        if dr <= 0 or dg < 0 or dg > 4 * dr * med + 1000 or (dr > 8 * step and dg * 64 < dr * med):
            bad += 1
    return bad


def fit(pts):
    """Least-squares gfx->realtime line; returns f(gfx) -> ns relative to the first pair."""
    n = len(pts)
    mg = sum(p[0] for p in pts) / n
    mr = sum(p[1] for p in pts) / n
    var = sum((p[0] - mg) ** 2 for p in pts)
    k = sum((p[0] - mg) * (p[1] - mr) for p in pts) / var
    r0 = pts[0][1]
    return lambda g: ((mr + k * (g - mg)) - r0) * REALTIME_NS, r0, k


def waves(occ, se):
    """Pair start/end rows by (cu, simd, wave_id) in time order."""
    open_, out, orphan = {}, [], 0
    for row in sorted(occ[se], key=lambda r: (r[0], -r[4])):
        t, cu, simd, slot, start, kid = row[:6]
        key = (cu, simd, slot)
        if start:
            if key in open_:
                orphan += 1
            open_[key] = row
        else:
            s = open_.pop(key, None)
            if s is None:
                orphan += 1
                continue
            out.append({"se": int(se), "cu": cu, "simd": simd, "slot": slot, "begin": s[0], "end": t,
                        "kernel_id": s[5], "workgroup": s[9]})
    return out, orphan + len(open_)




def check(d):
    occ, rt, markers, info = load(d)
    ok = True
    for se in [k for k in occ if k.isdigit()]:
        pts = pairs(rt, se)
        bad = anomalies(pts) if len(pts) >= 2 else None
        gfx = sorted(p[0] for p in pts)
        steps = sorted(b - a for a, b in zip(gfx, gfx[1:]))
        slack = 3 * steps[len(steps) // 2] if steps else 0
        lo, hi = (gfx[0] - slack, gfx[-1] + slack) if gfx else (0, 0)
        outside = sum(1 for r in occ[se] if not lo <= r[0] <= hi)
        ws, orphan = waves(occ, se)
        n_disp = len(set(w["kernel_id"] for w in ws))
        status = "OK" if bad == 0 and outside == 0 and orphan == 0 else "FAIL"
        ok &= status == "OK"
        print(f"SE{se}: {status}  realtime pairs={len(pts)} clock anomalies={bad} "
              f"occupancy rows outside window={outside}/{len(occ[se])} waves={len(ws)} "
              f"unpaired={orphan} dispatches={n_disp}")
    if markers:
        groups = split_dispatches(all_waves(occ, rt), markers, occ["dispatches"], markers[0]["_by_pso"] if markers else None)
        labelled = sum(1 for _, m in groups if m is not None)
        idle = len(markers) - labelled  # markers whose dispatch launched no traced waves (e.g. empty grids)
        print(f"markers={len(markers)} dispatches={len(groups)} matched={labelled} unmatched markers={idle} "
              f"{'OK' if labelled == len(groups) else 'MISMATCH'}")
        meta = [m for m in markers if (m.get("command") or "").startswith("vkCmd") and not (m.get("command") or "").startswith("vkCmdDispatch")]
        if meta and labelled < len(groups):
            print(f"note: {len(meta)} marker(s) are driver transfer commands ({', '.join(sorted({m['command'] for m in meta}))}): RADV runs "
                  "them with its own internal shader, which carries no ggml label. A model load issues one such submit per "
                  "tensor upload; capture the compute submits (the largest captures, e.g. WT_KEEP_LAST at the end of a run).")
        ok &= labelled == len(groups)
    san = info.get("sanitize", {})
    if san.get("blanked_exec_tokens"):
        print(f"note: {san['blanked_exec_tokens']} RADV exec tokens blanked "
              "(default instruction-timing mask; per-instruction data incomplete)")
    return ok


def all_waves(occ, rt):
    out = []
    for se in [k for k in occ if k.isdigit()]:
        f, _, _ = fit(pairs(rt, se))
        ws, _ = waves(occ, se)
        for w in ws:
            w["t0"], w["t1"] = f(w["begin"]), f(w["end"])
        out += ws
    if not out:
        sys.exit("wavetrace: no waves in this capture (empty submit, or a capture overwritten by a later one: check the driver's\n"
                 "per-submit file naming and the capture size; `check` prints the trust counters)")
    base = min(w["t0"] for w in out)
    for w in out:
        w["t0"] -= base
        w["t1"] -= base
    return out


def union(iv):
    total, end = 0.0, None
    for a, b in sorted(iv):
        if end is None or a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def peak(iv):
    ev = sorted([(a, 1) for a, _ in iv] + [(b, -1) for _, b in iv])
    cur = best = 0
    for _, s in ev:
        cur += s
        best = max(best, cur)
    return best


def marker_key(m, by_pso=None):
    if by_pso and m.get("pipeline") in by_pso:
        return by_pso[m["pipeline"]]
    if m["labels"] and not m["labels"][-1].startswith("submit "):
        return stem(m["labels"][-1])
    if m.get("pipeline") is not None:
        return f"pipeline {m['pipeline']:016x}"
    return m.get("command") or ""


def islands(waves):
    """Split waves (sorted by t0) into busy islands: spans where at least one is alive."""
    out, end = [], None
    for w in waves:
        if end is None or w["t0"] > end:
            out.append([])
            end = w["t1"]
        out[-1].append(w)
        end = max(end, w["t1"])
    return out


def sized(ms, total):
    """Waves per dispatch from marker grids, if every marker has one and they divide the total."""
    if not all(m.get("dims") for m in ms):
        return None
    groups = [m["dims"][0] * m["dims"][1] * m["dims"][2] for m in ms]
    if not sum(groups) or total % sum(groups):
        return None
    per = total // sum(groups)
    return [g * per for g in groups]


def split_dispatches(ws, markers, names, by_pso=None):
    """The decoder numbers kernels by pipeline, not by dispatch: every dispatch of one pipeline in
    the submit (back-to-back, or one per layer with other ops between) shares a kernel_id. Each
    kernel_id gets the markers that ran its pipeline (matched by debug-label stem or API PSO hash),
    and its waves are split into that many dispatches: by busy islands (gaps where none of its
    waves is alive), merging the smallest gaps when there are more islands than markers, else into
    equal chunks by start time when back-to-back dispatches overlap.
    Returns [(waves, marker or None)] in launch order."""
    by_kid = {}
    for w in ws:
        by_kid.setdefault(w["kernel_id"], []).append(w)
    out = []
    for kid, mine in by_kid.items():
        mine.sort(key=lambda w: w["t0"])
        name = names.get(str(kid), "")
        parts = set(name.split(" / ")) | {name}
        ms = [m for m in markers if marker_key(m, by_pso) in parts]
        k = len(ms)
        isl = islands(mine)
        while len(isl) > max(k, 1):
            gaps = [isl[i + 1][0]["t0"] - max(w["t1"] for w in isl[i]) for i in range(len(isl) - 1)]
            i = gaps.index(min(gaps))
            isl[i:i + 2] = [isl[i] + isl[i + 1]]
        if k <= 1:
            out.append((mine, ms[0] if ms else None))
        elif len(isl) == k:
            out += list(zip(isl, ms))
        elif sized(ms, len(mine)):
            # Overlapping dispatches of different grids: size each from its marker's workgroup
            # count and take them in launch order.
            at = 0
            for m, n in zip(ms, sized(ms, len(mine))):
                out.append((mine[at:at + n], m))
                at += n
        elif len(mine) % k == 0:
            n = len(mine) // k
            out += [(mine[i * n:(i + 1) * n], m) for i, m in enumerate(ms)]
        else:
            print(f"warning: kernel {kid} ({name}) has {len(mine)} waves in {len(isl)} islands for {k} "
                  "dispatches; not split", file=sys.stderr)
            out.append((mine, ms[0]))
    out.sort(key=lambda g: g[0][0]["t0"])
    return out


def stem(label):
    if label.endswith(")") and " (" in label:
        head, dims = label[:-1].rsplit(" (", 1)
        if dims and all(c.isdigit() or c in ", " for c in dims):
            return head.rstrip()
    return label


def per_dispatch(d):
    occ, rt, markers, _ = load(d)
    ws = all_waves(occ, rt)
    rows = []
    for i, (mine, m) in enumerate(split_dispatches(ws, markers, occ["dispatches"], markers[0]["_by_pso"] if markers else None), 1):
        kid = mine[0]["kernel_id"]
        for w in mine:
            w["dispatch"] = i
        iv = [(w["t0"], w["t1"]) for w in mine]
        life = [b - a for a, b in iv]
        span = max(b for _, b in iv) - min(a for a, _ in iv)
        m = m or {}
        op = next((l for l in m.get("labels", []) if not l.startswith("submit ")), None)
        rows.append({
            "dispatch": i, "kernel_id": kid, "label": op or occ["dispatches"].get(str(kid), "?"),
            "kernel": occ["dispatches"].get(str(kid), "?"), "groups": m.get("dims"),
            "waves": len(mine), "start_us": min(a for a, _ in iv) / 1e3, "span_us": span / 1e3,
            "active_us": union(iv) / 1e3, "wave_mean_us": sum(life) / len(life) / 1e3,
            "wave_max_us": max(life) / 1e3, "peak_waves": peak(iv),
            "mean_waves": sum(life) / span if span else 0.0,
            "cus": len({(w["se"], w["cu"]) for w in mine}),
        })
    return rows, ws


def summary(d):
    rows, _ = per_dispatch(d)
    print(f"{'id':>3} {'start':>9} {'active':>9} {'waves':>6} {'CUs':>4} {'peak':>5} {'mean':>6} "
          f"{'wave_avg':>9} {'wave_max':>9}  label / kernel")
    for r in rows:
        print(f"{r['dispatch']:>3} {r['start_us']:>8.1f}u {r['active_us']:>8.1f}u {r['waves']:>6} "
              f"{r['cus']:>4} {r['peak_waves']:>5} {r['mean_waves']:>6.1f} {r['wave_mean_us']:>8.2f}u "
              f"{r['wave_max_us']:>8.2f}u  {r['label']}  [{r['kernel']} {r['groups']}]")
    total = union([(r["start_us"], r["start_us"] + r["span_us"]) for r in rows])
    print(f"GPU-active (union of dispatch spans): {total:.1f} us over {len(rows)} dispatches")


def trace(d, out):
    rows, ws = per_dispatch(d)
    by_id = {r["dispatch"]: r for r in rows}
    ev = []
    ses = sorted({w["se"] for w in ws})
    ev.append({"ph": "M", "name": "process_name", "pid": 0, "args": {"name": "dispatches"}})
    for se in ses:
        ev.append({"ph": "M", "name": "process_name", "pid": 1 + se, "args": {"name": f"SE{se} waves"}})
    for r in rows:
        ev.append({"ph": "X", "pid": 0, "tid": 0, "name": r["label"], "ts": r["start_us"], "dur": r["span_us"],
                   "args": {k: r[k] for k in ("kernel", "groups", "waves", "peak_waves", "active_us", "wave_mean_us")}})
    tids = {}
    for w in ws:
        key = (w["se"], w["cu"], w["simd"], w["slot"])
        if key not in tids:
            tids[key] = len(tids) + 1
            ev.append({"ph": "M", "name": "thread_name", "pid": 1 + w["se"], "tid": tids[key],
                       "args": {"name": f"CU{w['cu']:02d} SIMD{w['simd']} slot{w['slot']:02d}"}})
            ev.append({"ph": "M", "name": "thread_sort_index", "pid": 1 + w["se"], "tid": tids[key],
                       "args": {"sort_index": w["cu"] * 10000 + w["simd"] * 100 + w["slot"]}})
        ev.append({"ph": "X", "pid": 1 + w["se"], "tid": tids[key], "name": by_id[w["dispatch"]]["label"],
                   "ts": w["t0"] / 1e3, "dur": (w["t1"] - w["t0"]) / 1e3,
                   "args": {"kernel": by_id[w["dispatch"]]["kernel"], "workgroup": w["workgroup"]}})
    Path(out).write_text(json.dumps({"traceEvents": ev, "displayTimeUnit": "ns"}))
    print(f"wrote {out}: {len(ws)} waves on {len(tids)} slots, {len(rows)} dispatches")


def isa_class(isa):
    op = isa.split()[0]
    if op == "s_waitcnt":
        return "s_waitcnt " + "+".join(k for k in ("vm", "lgkm", "vs") if k + "cnt" in isa)
    for prefix, name in (("v_wmma", "v_wmma"), ("ds_load_u16", "ds_load 16-bit"), ("ds_load", "ds_load wide"),
                         ("ds_store", "ds_store"), ("buffer_load_d16", "buffer_load 16-bit"),
                         ("global_load_d16", "buffer_load 16-bit"), ("buffer_load", "buffer_load wide"),
                         ("global_load", "buffer_load wide"), ("buffer_store", "store"), ("global_store", "store"),
                         ("s_barrier", "s_barrier"), ("s_waitcnt_", "s_waitcnt_depctr"),
                         ("v_exp", "v_ trans"), ("v_log", "v_ trans"), ("v_rcp", "v_ trans"), ("v_", "v_ ALU"),
                         ("s_", "s_ scalar")):
        if op.startswith(prefix):
            return name
    return op


def isa(d, codeobj=None):
    """code.json rows: ISA, _, line, source, codeobj, vaddr, hits, latency, stall, idle (cycles,
    summed over the traced CU's waves). Latency overlaps across waves sharing a SIMD; stall is
    the cycles an instruction waited to issue."""
    code = json.loads((Path(d) / "decoded" / "code.json").read_text())["code"]
    objs = sorted({r[4] for r in code if r[6] > 0})
    for co in ([int(codeobj)] if codeobj else objs):
        rows = [r for r in code if r[4] == co and r[6] > 0]
        lat, st = sum(r[7] for r in rows), sum(r[8] for r in rows)
        if not lat:
            continue
        agg = {}
        for r in rows:
            a = agg.setdefault(isa_class(r[0]), [0, 0, 0, 0])
            a[0] += 1; a[1] += r[6]; a[2] += r[7]; a[3] += r[8]
        print(f"code object {co}: {len(rows)} executed instructions, latency {lat:.3g} cyc, stall {st:.3g} cyc")
        print(f"  {'class':22} {'#ins':>5} {'hits':>10} {'lat%':>6} {'stall%':>7} {'lat/hit':>8}")
        for k, a in sorted(agg.items(), key=lambda kv: -kv[1][2]):
            print(f"  {k:22} {a[0]:5} {a[1]:10} {a[2] / lat:6.1%} {a[3] / max(st, 1):7.1%} {a[2] / a[1]:8.1f}")


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] in ("-h", "--help"):
        print(__doc__); sys.exit(0)
    if len(a) >= 2 and a[0] == "check":
        sys.exit(0 if check(a[1]) else 1)
    elif len(a) >= 2 and a[0] == "summary":
        summary(a[1])
    elif len(a) >= 2 and a[0] == "isa":
        isa(a[1], a[2] if len(a) > 2 else None)
    elif len(a) >= 3 and a[0] == "trace":
        trace(a[1], a[2])
    else:
        sys.exit(__doc__)

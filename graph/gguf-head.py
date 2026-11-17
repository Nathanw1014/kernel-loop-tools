#!/usr/bin/env python3
"""gguf-head.py: header-only GGUF parse, local file or http(s) URL (HTTP Range requests; tensor data is never downloaded).

usage: gguf-head.py <path-or-url> [<path-or-url> ...] [--json out.json]
Several arguments = the shards of one split model. Prints architecture, name, file type, tensor count, tensor and file bytes,
and bytes per tensor type; --json also writes the hparams, every tensor-name pattern (blk.{bid}.*) with count, types, shape
and bytes, and the type mix. Pure Python, no gguf package needed.
Use it to size a model (and its weight bytes read per token: a roofline input) before downloading it, to see which tensors
a quant actually stores in which type, and to check that a bench's model is what its name claims.
Type ids follow ggml.h; an id this table does not know prints as its number.
"""
import json, re, struct, sys, os, urllib.request

TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K",
         12: "Q4_K", 13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
         20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32", 27: "I64", 28: "F64",
         29: "IQ1_M", 30: "BF16", 34: "TQ1_0", 35: "TQ2_0", 39: "MXFP4",
         40: "NVFP4", 41: "Q1_0", 42: "Q2_0"}


class Src:
    def __init__(self, loc):
        self.loc, self.buf, self.size = loc, b"", None
        self.url = loc.startswith("http://") or loc.startswith("https://")
        if not self.url:
            self.size = os.path.getsize(loc)
        self.fetched = 0

    def need(self, end):
        while len(self.buf) < end:
            want = max(end, 2 * len(self.buf), 1 << 20)
            if self.url:
                req = urllib.request.Request(self.loc, headers={"Range": f"bytes=0-{want - 1}", "User-Agent": "gguf-head"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    cr = r.headers.get("Content-Range")
                    if cr and "/" in cr:
                        self.size = int(cr.split("/")[1])
                    data = r.read(want)
                if r.status == 200 and self.size is None:
                    self.size = len(data)
                if len(data) <= len(self.buf):
                    raise EOFError("no more data")
                self.buf = data
            else:
                with open(self.loc, "rb") as f:
                    self.buf = f.read(want)
                if len(self.buf) < end and len(self.buf) == self.size:
                    raise EOFError("short file")
        self.fetched = max(self.fetched, len(self.buf))


class Rd:
    def __init__(self, src):
        self.s, self.p = src, 0

    def take(self, n):
        self.s.need(self.p + n)
        b = self.s.buf[self.p:self.p + n]
        self.p += n
        return b

    def u(self, fmt):
        n = struct.calcsize(fmt)
        return struct.unpack("<" + fmt, self.take(n))[0]

    def str(self):
        return self.take(self.u("Q")).decode("utf-8", "replace")

    SC = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def val(self, t):
        if t in self.SC:
            return self.u(self.SC[t])
        if t == 8:
            return self.str()
        if t == 9:
            et, n = self.u("I"), self.u("Q")
            if et in self.SC:
                fmt = self.SC[et]
                raw = self.take(n * struct.calcsize(fmt))
                return list(struct.unpack("<" + fmt * n, raw)) if n <= 4096 else {"array_len": n, "type": et}
            items = [self.val(et) for _ in range(n)]
            return items if n <= 4096 else {"array_len": n, "type": et}
        raise ValueError(f"bad kv type {t}")


def parse(loc):
    s = Src(loc)
    r = Rd(s)
    if r.take(4) != b"GGUF":
        raise ValueError(f"{loc}: not a GGUF")
    ver, nt, nkv = r.u("I"), r.u("Q"), r.u("Q")
    kv = {}
    for _ in range(nkv):
        k = r.str()
        kv[k] = r.val(r.u("I"))
    ti = []
    for _ in range(nt):
        name = r.str()
        nd = r.u("I")
        dims = [r.u("Q") for _ in range(nd)]
        ti.append([name, dims, r.u("I"), r.u("Q")])
    align = kv.get("general.alignment", 32)
    data0 = (r.p + align - 1) // align * align
    size = s.size
    if size is None and s.url:  # HEAD-less servers: one tiny ranged request for the length
        s.need(r.p + 1)
        size = s.size
    order = sorted(range(len(ti)), key=lambda i: ti[i][3])
    for j, i in enumerate(order):
        end = ti[order[j + 1]][3] if j + 1 < len(order) else (size - data0 if size else ti[i][3])
        ti[i].append(end - ti[i][3])
    return {"loc": loc, "version": ver, "kv": kv, "tensors": ti, "file_bytes": size, "header_bytes": data0, "fetched": s.fetched}


def pattern(name):
    return re.sub(r"^blk\.\d+\.", "blk.{bid}.", re.sub(r"\.(weight|bias|scale)$", "", name))


def summary(parts):
    kv = {}
    for p in parts:
        for k, v in p["kv"].items():
            kv.setdefault(k, v)
    tens = [t for p in parts for t in p["tensors"]]
    arch = kv.get("general.architecture", "?")
    pats, types = {}, {}
    for name, dims, typ, off, nb in tens:
        pt = pattern(name)
        pats.setdefault(pt, {"n": 0, "types": set(), "shape": dims, "bytes": 0})
        pats[pt]["n"] += 1
        pats[pt]["bytes"] += nb
        pats[pt]["types"].add(TYPES.get(typ, str(typ)))
        types[TYPES.get(typ, str(typ))] = types.get(TYPES.get(typ, str(typ)), 0) + nb
    hp = {k[len(arch) + 1:]: v for k, v in kv.items() if k.startswith(arch + ".")}
    return {"arch": arch, "name": kv.get("general.name"), "file_type": kv.get("general.file_type"),
            "split_count": kv.get("split.count"), "n_tensors": len(tens), "tensor_bytes": sum(t[4] for t in tens),
            "file_bytes": sum(p["file_bytes"] or 0 for p in parts), "header_fetched": sum(p["fetched"] for p in parts),
            "hparams": {k: v for k, v in hp.items() if not (isinstance(v, list) and len(v) > 64)},
            "hparams_long": {k: (f"list[{len(v)}]" if isinstance(v, list) else v) for k, v in hp.items() if isinstance(v, list) and len(v) > 64},
            "patterns": {k: {"n": v["n"], "types": sorted(v["types"]), "shape": v["shape"], "bytes": v["bytes"]} for k, v in sorted(pats.items())},
            "bytes_by_type": dict(sorted(types.items(), key=lambda x: -x[1]))}


if __name__ == "__main__":
    args = [a for a in sys.argv[1:]]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__); sys.exit(0 if args else 2)
    out = None
    if "--json" in args:
        i = args.index("--json"); out = args[i + 1]; del args[i:i + 2]
    parts = [parse(a) for a in args]
    sm = summary(parts)
    if out:
        json.dump(sm, open(out, "w"), indent=1, default=str)
    G = 1 << 30
    print(f"arch={sm['arch']} name={sm['name']} file_type={sm['file_type']} shards={len(parts)} tensors={sm['n_tensors']} "
          f"tensor_bytes={sm['tensor_bytes'] / G:.2f} GiB file_bytes={sm['file_bytes'] / G:.2f} GiB (header fetched {sm['header_fetched'] / 1e6:.1f} MB)")
    print("types: " + " ".join(f"{k}={v / G:.2f}G" for k, v in sm["bytes_by_type"].items()))

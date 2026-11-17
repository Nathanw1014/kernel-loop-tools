#!/usr/bin/env python3
"""RADV .rgp thread-trace extractor for rocprof-trace-decoder.

Splits a RADV per-submit .rgp capture into the inputs rocprof-trace-decoder takes
(one SQTT stream per shader engine, each shader ELF with its load address), plus the
RGP dispatch markers RADV writes into the stream (debug labels, pipeline binds, one
event per hardware dispatch).

Ported from the RGP reader of the MIT-licensed rocprof studio GUI (see third_party/LICENSE
and third_party/PROVENANCE).

  rgpwaves.py extract capture.rgp outdir > outdir.extract.log

The printed code-object list ("co N: base=... text=... name") is what timing_report.py reads
from <outdir>.extract.log.
"""
import os
import json
import struct
import sys
from pathlib import Path

MAGIC = 0x50303042  # "B00P"
HEADER_BYTES = 56
CHUNK_HEADER_BYTES = 16

CHUNK_ASIC_INFO = 0
CHUNK_SQTT_DESC = 1
CHUNK_SQTT_DATA = 2
CHUNK_CODE_OBJECT_DATABASE = 9
CHUNK_CODE_OBJECT_LOADER_EVENTS = 10
CHUNK_PSO_CORRELATION = 11

USERDATA_2 = 0xC342
USERDATA_3 = 0xC343
RGP_DISPATCH = (6, 7)  # vkCmdDispatch, vkCmdDispatchIndirect
EXEC_TOKEN_BITS = 24

INTERNAL_COMMANDS = {
    8: "vkCmdCopyBuffer", 9: "vkCmdCopyImage", 10: "vkCmdBlitImage",
    11: "vkCmdCopyBufferToImage", 12: "vkCmdCopyImageToBuffer", 13: "vkCmdUpdateBuffer",
    14: "vkCmdFillBuffer", 15: "vkCmdClearColorImage", 16: "vkCmdClearDepthStencilImage",
    18: "vkCmdResolveImage", 21: "vkCmdResetQueryPool", 22: "vkCmdCopyQueryPoolResults",
}


def u32(d, at):
    return struct.unpack_from("<I", d, at)[0]


def u64(d, at):
    return struct.unpack_from("<Q", d, at)[0]


def name_at(d, at, n):
    raw = bytes(d[at:at + n])
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace").strip()


def chunks(d):
    if len(d) < HEADER_BYTES or u32(d, 0) != MAGIC:
        raise ValueError("not an RGP capture")
    at = u32(d, 16)
    out = []
    while at + CHUNK_HEADER_BYTES <= len(d):
        kind = u32(d, at) & 0xFF
        size = u32(d, at + 8)
        if size < CHUNK_HEADER_BYTES or at + size > len(d):
            break
        out.append((kind, at, size))
        at += size
    if not out:
        raise ValueError("RGP capture contains no readable chunks")
    return out


def asic(d, off):
    return {
        "name": name_at(d, off + 152, 32),
        "device_id": u32(d, off + 40),
        "shader_clock_hz": u64(d, off + 24),
        "shader_engines": u32(d, off + 56),
        "cus_per_engine": u32(d, off + 60),
    }


def gfx11_token_bits():
    """gfx11 SQTT token lengths in bits by low 8 header bits; 0 = unknown header.
    The 1110 / x111 families are RADV-enabled ALU/VMEM exec tokens the decoder lacks."""
    bits = [0] * 256
    exec_ = [False] * 256

    def add(pattern, length, token, is_exec):
        for i in range(pattern, 256, 1 << length):
            bits[i] = token
            exec_[i] = is_exec

    for pattern, length, token in [
        (0b010, 3, 20), (0b011, 3, 12), (0b1101, 4, 12), (0b00100, 5, 24), (0b10100, 5, 24),
        (0b0100001, 7, 64), (0b01100, 5, 32), (0b11100, 5, 48), (0b00101, 5, 20), (0b10101, 5, 20),
        (0b00110, 5, 52), (0b10110, 5, 28), (0b0110001, 7, 64), (0b1000, 4, 8), (0b0000, 4, 4),
        (0b1010001, 7, 24), (0b01100001, 8, 24), (0b11100001, 8, 32), (0b1001, 4, 64),
        (0b1110001, 7, 64), (0b0000001, 7, 64), (0b0010001, 7, 64),
        (0b1010001, 7, 24), (0b0110001, 7, 48), (0b0000001, 7, 48),  # gfx11 overrides
    ]:
        add(pattern, length, token, False)
    add(0b1110, 4, EXEC_TOKEN_BITS, True)
    add(0b111, 3, EXEC_TOKEN_BITS, True)
    return bits, exec_


BITS, EXEC = gfx11_token_bits()


def nibbles(buf):
    out = bytearray(len(buf) * 2)
    out[0::2] = bytes(b & 0xF for b in buf)
    out[1::2] = bytes(b >> 4 for b in buf)
    return out


def sanitize_sqtt(buf):
    """Blank RADV exec tokens to zero nibbles (NOPs) so the decoder stays in sync."""
    nib = nibbles(buf)
    total = len(nib)
    blanked = unknown = 0
    i = 0
    while i + 2 <= total:
        header = nib[i] | (nib[i + 1] << 4)
        n = BITS[header] // 4
        if n == 0:
            unknown += 1
            i += 1
            continue
        if EXEC[header] and i + n <= total:
            for j in range(i, i + n):
                buf[j >> 1] &= 0x0F if j & 1 else 0xF0
            blanked += 1
        i += n
    return blanked, unknown


def dispatch_markers(body):
    """RGP instrumentation markers RADV writes as SQ_THREAD_TRACE_USERDATA_2/3 register
    tokens. Returns (dispatch markers in stream order, complete)."""
    nib = nibbles(body)
    total = len(nib)
    streams = {}
    i = 0
    while i + 2 <= total:
        header = nib[i] | (nib[i + 1] << 4)
        n = max(BITS[header] // 4, 1)
        if header & 0xF == 0b1001 and i + 16 <= total:
            v = 0
            for j in range(15, -1, -1):
                v = (v << 4) | nib[i + j]
            if (v >> 16) & 0xFFFF in (USERDATA_2, USERDATA_3):
                streams.setdefault((((v >> 9) & 3), ((v >> 7) & 3)), []).append(v >> 32)
        i += n
    out = []
    for (me, pipe), dw in sorted(streams.items()):
        labels, bound = [], None
        at = 0
        while at < len(dw):
            d0 = dw[at]
            kind = d0 & 0xF
            if kind == 0:
                size = 6 if d0 >> 31 == 1 else 3
            elif kind in (2, 0xC):
                size = 3
            elif kind == 1:
                size = 4
            elif kind in (3, 4, 9):
                size = 2
            elif kind == 5:
                if (d0 >> 12) & 0xFF == 1:
                    size = 1
                elif at + 1 < len(dw):
                    size = 2 + dw[at + 1] // 4
                else:
                    return out, False
            elif kind == 6:
                size = 1
            else:
                return out, False
            m = dw[at:at + size]
            if len(m) < size:
                return out, False
            if kind == 0:
                api = (d0 >> 7) & 0xFFFFFF
                out.append({
                    "me": me, "pipe": pipe, "api": api,
                    "pipeline": bound if api in RGP_DISPATCH else None,
                    "dims": [m[3], m[4], m[5]] if size == 6 else None,
                    "labels": list(labels),
                    "command": INTERNAL_COMMANDS.get(api),
                })
            elif kind == 1:
                labels, bound = [], None
            elif kind == 5:
                sub = (d0 >> 12) & 0xFF
                if sub == 1 and labels:
                    labels.pop()
                elif sub == 2:
                    raw = b"".join(struct.pack("<I", w) for w in m[2:])
                    labels.append(name_at(raw, 0, len(raw)))
            elif kind == 0xC and (d0 >> 7) & 1 == 1:  # compute bind point
                bound = m[1] | (m[2] << 32)
            at += size
    return out, True


def stem(label):
    """Drop ggml's trailing ' (x,y,z)' workgroup counts from a debug label."""
    if label.endswith(")") and " (" in label:
        head, dims = label[:-1].rsplit(" (", 1)
        if dims and all(c.isdigit() or c in ", " for c in dims):
            return head.rstrip()
    return label


def pipeline_names(markers):
    seen = {}
    for m in markers:
        if m["pipeline"] is None or not m["labels"]:
            continue
        s = stem(m["labels"][-1])
        counts = seen.setdefault(m["pipeline"], {})
        counts[s] = counts.get(s, 0) + 1
    names = {}
    for pipeline, counts in seen.items():
        ranked = sorted((s for s in counts if s), key=lambda s: (-counts[s], s))
        if ranked:
            names[pipeline] = " / ".join(ranked[:3]) + (" / …" if len(ranked) > 3 else "")
    return names


def code_objects(d, off):
    count = u32(d, off + 28)
    at = off + 32
    out = []
    for _ in range(count):
        size = u32(d, at)
        out.append(bytes(d[at + 4:at + 4 + size]))
        at = (at + 4 + size + 3) & ~3
    return out


def loader_events(d, off):
    record, count = u32(d, off + 24), u32(d, off + 28)
    return [(u64(d, off + 32 + i * record + 8),
             (u64(d, off + 32 + i * record + 16), u64(d, off + 32 + i * record + 24)))
            for i in range(count)]


def pso_correlation(d, off):
    record, count = u32(d, off + 24), u32(d, off + 28)
    out = []
    for i in range(count):
        at = off + 32 + i * record
        out.append((u64(d, at), (u64(d, at + 8), u64(d, at + 16)), name_at(d, at + 24, 64)))
    return out


def elf_text_size(elf):
    shoff = u64(elf, 0x28)
    entsize, count, names = struct.unpack_from("<HHH", elf, 0x3A)
    strtab = u64(elf, shoff + names * entsize + 24)
    for i in range(count):
        h = shoff + i * entsize
        name = u32(elf, h)
        if elf[strtab + name:strtab + name + 6] == b".text\0":
            return u64(elf, h + 32)
    return len(elf)


def elf_pipeline_hash(elf):
    key = b".internal_pipeline_hash"
    at = elf.find(key)
    if at < 0:
        return None
    v = elf[at + len(key):at + len(key) + 19]
    if len(v) == 19 and v[0] == 0x92 and v[1] == 0xCF and v[10] == 0xCF:
        return (struct.unpack(">Q", v[2:10])[0], struct.unpack(">Q", v[11:19])[0])
    return None


def extract(rgp, outdir):
    d = bytearray(Path(rgp).read_bytes())
    table = chunks(d)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    info = {"rgp": str(Path(rgp).resolve()), "version": [u32(d, 4), u32(d, 8)]}
    for kind, off, _ in table:
        if kind == CHUNK_ASIC_INFO:
            info["asic"] = asic(d, off)

    sections, se = [], None
    for kind, off, _ in table:
        if kind == CHUNK_SQTT_DESC:
            se = u32(d, off + 16)
        elif kind == CHUNK_SQTT_DATA:
            o, n = u32(d, off + 16), u32(d, off + 20)
            sections.append((se, bytearray(d[o:o + n])))
    if not sections:
        raise ValueError("capture has no thread trace (counters-only?)")

    # Markers are read from the raw stream; sanitising only blanks exec tokens.
    marker_sets = [(se, dispatch_markers(body)) for se, body in sections]
    blanked = unknown = 0
    for _, body in sections:
        b, u = sanitize_sqtt(body)
        blanked += b
        unknown += u
    info["sanitize"] = {"blanked_exec_tokens": blanked, "unknown_headers": unknown}
    info["sanitize"]["nibbles"] = sum(len(body) * 2 for _, body in sections)
    # RGPW_ALLOW_UNKNOWN=1 (timing-mode capture on stock Mesa, partial decode accepted): keep going and
    # record the count; the decoder may lose sync after each unsizable header, so per-instruction data is partial
    if unknown and os.environ.get("RGPW_ALLOW_UNKNOWN", "0") == "0":
        raise ValueError(f"{unknown} unsizable token headers: default instruction-timing mask; "
                         "capture with RADV_THREAD_TRACE_INSTRUCTION_TIMING=false or the compat patch")

    loads, psos, elves = [], [], []
    for kind, off, _ in table:
        if kind == CHUNK_CODE_OBJECT_DATABASE:
            elves = code_objects(d, off)
        elif kind == CHUNK_CODE_OBJECT_LOADER_EVENTS:
            loads = loader_events(d, off)
        elif kind == CHUNK_PSO_CORRELATION:
            psos = pso_correlation(d, off)

    markers = marker_sets[0][1][0] if marker_sets else []
    names = pipeline_names(markers)
    traces, objects = [], []
    for se, body in sections:
        p = outdir / f"radv_shader_engine_{se}_0.att"
        p.write_bytes(body)
        traces.append({"shaderEngine": se, "path": p.name})
    for i, elf in enumerate(elves, 1):
        h = elf_pipeline_hash(elf)
        base = next((b for b, lh in loads if lh == h), None)
        if base is None and len(loads) == 1 and i == 1:
            base = loads[0][0]
        if base is None:
            raise ValueError(f"code object {i} has no loader event")
        api = next(((a, n) for a, ih, n in psos if ih == h), None)
        if api and api[1]:
            label = api[1]
        elif api:
            label = names.get(api[0], f"pipeline {api[0]:016x}")
        else:
            label = f"code object {i}"
        if any(o["name"] == label for o in objects):  # two pipelines share a debug label
            label = f"{label} #{i}"
        p = outdir / f"radv_code_object_id_{i}.out"
        p.write_bytes(elf)
        objects.append({"id": i, "path": p.name, "base": base, "size": elf_text_size(elf),
                        "name": label, "api_pso": api[0] if api else None})
    (outdir / "radv-manifest.json").write_text(json.dumps({"traces": traces, "codeObjects": objects}, indent=1))
    info["markers"] = {str(se): {"count": len(m), "complete": c} for se, (m, c) in marker_sets}
    (outdir / "markers.json").write_text(json.dumps(markers))
    (outdir / "capture-info.json").write_text(json.dumps(info, indent=1))
    return info, objects


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] != "extract":
        sys.exit(__doc__)
    info, objects = extract(sys.argv[2], sys.argv[3])
    print(json.dumps(info, indent=1))
    for o in objects:
        print(f"  co {o['id']}: base=0x{o['base']:x} text={o['size']} {o['name']}")

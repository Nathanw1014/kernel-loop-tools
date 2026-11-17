#!/usr/bin/env python3
"""pagecache.py: put model files into a known page-cache state before a run, without root.

usage: pagecache.py res  <file>... [--tag T]    residency: resident GiB / size per file (mmap + mincore; faults nothing in)
       pagecache.py drop <file>... [--tag T]    posix_fadvise(DONTNEED) on each file, then res
       pagecache.py warm <file>... [--tag T]    read each file once (16 MiB chunks) to pull it into the cache, then res

Prints one line: "pagecache <tag> resident X/Y GiB (Z%) <per-file %>" (split GGUF shards are labelled by shard number).
drop releases only this file's clean pages that no process has mapped: a running server that mmap'd the model keeps them,
so res after drop says what really left. Unlike `echo 3 > /proc/sys/vm/drop_caches` it needs no root and leaves every
other file's cache alone.
Guards against: an A/B whose first arm loads from disk and whose second arm loads from cache (load time, first-token
latency and even prefill on mmap'd weights differ), and against claiming "cold" without checking. Record the res line
before and after each arm. samplers/fincore.py is the read-only residency check; this adds drop / warm.
"""
import ctypes, ctypes.util, os, re, sys, time

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
PROT_READ, MAP_SHARED = 1, 1
MAP_FAILED = ctypes.c_void_p(-1).value
PG = os.sysconf("SC_PAGE_SIZE")
CHUNK = 1 << 30   # map 1 GiB at a time


def resident(path):
    size = os.path.getsize(path)
    res = 0
    fd = os.open(path, os.O_RDONLY)
    try:
        off = 0
        while off < size:
            ln = min(CHUNK, size - off)
            addr = libc.mmap(None, ln, PROT_READ, MAP_SHARED, fd, off)
            if addr in (None, MAP_FAILED):
                raise OSError(ctypes.get_errno(), "mmap failed")
            vec = ctypes.create_string_buffer((ln + PG - 1) // PG)
            rc = libc.mincore(addr, ln, vec)
            libc.munmap(addr, ln)
            if rc != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            res += sum(b & 1 for b in vec.raw)
            off += ln
    finally:
        os.close(fd)
    return min(res * PG, size), size


def report(tag, files):
    tot_r = tot_s = 0
    parts = []
    for f in files:
        r, s = resident(f)
        tot_r += r; tot_s += s
        m = re.search(r"-(\d{5})-of-\d{5}", f)
        parts.append("%s=%.1f%%" % (m.group(1) if m else os.path.basename(f), 100.0 * r / s if s else 100.0))
    print("pagecache %s resident %.2f/%.2f GiB (%.1f%%) %s" % (
        tag, tot_r / 2**30, tot_s / 2**30, 100.0 * tot_r / tot_s if tot_s else 0, " ".join(parts)), flush=True)


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__); return 0 if a else 2
    tag = "-"
    if "--tag" in a:
        i = a.index("--tag"); tag = a[i + 1]; del a[i:i + 2]
    cmd, files = a[0], a[1:]
    if cmd not in ("res", "drop", "warm") or not files:
        print("usage: pagecache.py res|drop|warm <file>... [--tag T]", file=sys.stderr); return 2
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        print(f"pagecache: not a file: {' '.join(missing)}", file=sys.stderr); return 2
    t0 = time.time()
    if cmd == "drop":
        for f in files:
            fd = os.open(f, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
    elif cmd == "warm":
        mv = memoryview(bytearray(16 << 20))
        for f in files:
            with open(f, "rb", buffering=0) as fh:
                while fh.readinto(mv):
                    pass
    if cmd != "res":
        print("pagecache %s %s took %.1f s" % (tag, cmd, time.time() - t0), flush=True)
    report(tag, files)
    return 0


if __name__ == "__main__":
    sys.exit(main())

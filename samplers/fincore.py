#!/usr/bin/env python3
# fincore.py <file>... : page-cache residency per file (resident MiB / size MiB), like util-linux fincore.
# libc mmap(PROT_READ) + mincore(2): read-only, touches no file pages (mincore only reports residency).
import ctypes, ctypes.util, os, sys

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
PAGE = os.sysconf("SC_PAGE_SIZE")
PROT_READ, MAP_SHARED = 1, 1
MAP_FAILED = ctypes.c_void_p(-1).value
CHUNK = 1 << 30   # map 1 GiB at a time

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    sys.exit("usage: fincore.py <file>...  (page-cache residency per file)")

for path in sys.argv[1:]:
    real = os.path.realpath(path)
    size = os.path.getsize(real)
    resident = 0
    fd = os.open(real, os.O_RDONLY)
    try:
        off = 0
        while off < size:
            ln = min(CHUNK, size - off)
            addr = libc.mmap(None, ln, PROT_READ, MAP_SHARED, fd, off)
            if addr in (None, MAP_FAILED):
                raise OSError(ctypes.get_errno(), "mmap failed")
            npages = (ln + PAGE - 1) // PAGE
            vec = ctypes.create_string_buffer(npages)
            rc = libc.mincore(addr, ln, vec)
            libc.munmap(addr, ln)
            if rc != 0:
                raise OSError(ctypes.get_errno(), "mincore failed")
            resident += sum(b & 1 for b in vec.raw)
            off += ln
    finally:
        os.close(fd)
    print(f"{os.path.basename(real)}: {resident * PAGE / 2**20:.0f} MiB resident of {size / 2**20:.0f} MiB ({100 * resident * PAGE / max(size, 1):.1f}%)")

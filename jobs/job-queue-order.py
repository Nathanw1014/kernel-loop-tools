#!/usr/bin/env python3
"""job-queue-order.py <id> <front|before|after> [<target id>] < pending ids (one per line, any order)

Prints "old new" rename lines that move <id> to the front of the pending queue, or right before / after <target>, as seen by
the daemon's `ls pending/*.job | sort | head -1`. Bumped ids are "0" + 19 digits + "-" + original id (the fixed width keeps
C and locale collation in agreement, and "0..." sorts before every "<YYYYmmddTHHMMSS>Z-label" id). Only the moved job is renamed when
there is a numeric gap; otherwise every job ahead of it (and it) is renumbered in its current order. Exit 2 on bad input."""
import re, sys

W = 19; TOP = 10 ** W
BUMP = re.compile(r"^0(\d{%d})-(.+)$" % W)

def num(n):
    m = BUMP.match(n)
    return int(m.group(1)) if m else None

def base(n):
    m = BUMP.match(n)
    return m.group(2) if m else n

def key(k, n):
    return "0%0*d-%s" % (W, k, base(n))

def main():
    mv, mode = sys.argv[1], sys.argv[2]
    tgt = sys.argv[3] if len(sys.argv) > 3 else ""
    names = sorted(l.strip() for l in sys.stdin if l.strip())
    if mv not in names:
        print("no pending job %s" % mv, file=sys.stderr); return 2
    if mode != "front" and tgt not in names:
        print("no pending job %s" % tgt, file=sys.stderr); return 2
    if tgt == mv:
        print("target is the job itself", file=sys.stderr); return 2
    L = [n for n in names if n != mv]
    idx = 0 if mode == "front" else L.index(tgt) + (1 if mode == "after" else 0)
    L.insert(idx, mv)
    nxt = L[idx + 1] if idx + 1 < len(L) else None
    upper = num(nxt) if nxt is not None and num(nxt) is not None else TOP
    ren = {}
    prev = L[idx - 1] if idx > 0 else None
    lower = 0 if prev is None else num(prev)
    if lower is not None and upper - lower >= 2:
        ren[mv] = key((lower + upper) // 2, mv)          # only the moved job changes name
    else:                                               # renumber everything up to and including the moved job
        P = L[: idx + 1]
        for i, n in enumerate(P):
            ren[n] = key((i + 1) * upper // (len(P) + 1), n)
    for n in L:
        if n in ren and ren[n] != n:
            print(n, ren[n])
    return 0

sys.exit(main())

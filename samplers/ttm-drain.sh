#!/usr/bin/env bash
# ttm-drain.sh <need_gb>: release GPU pages parked in the kernel TTM page pool when MemAvailable is short. [AMD/TTM, ROOT]
# Needs passwordless sudo (sudo -n) and debugfs mounted (/sys/kernel/debug/ttm/page_pool); without them it changes nothing
# and prints "ttm=unread(...)".
# Why: freed write-combined GTT pages go back to the TTM page pool, which MemAvailable does not count, so a memory gate can
# wait forever while tens of GiB sit pooled. If MemAvailable < need_gb AND the pool holds > TTM_DRAIN_MIN_POOL_GIB (4),
# run the shrinkers (drop_caches=2: slab only, the page cache and any mmap'd model stay), re-check, append a line to
# $KLT_HOME/ttm-drain.log. Prints one token for the caller's settle line; always exits 0.
. "$(dirname "$(readlink -f "$0")")/../lib/klt-env.sh"
NEED=${1:-0}; MINP=${TTM_DRAIN_MIN_POOL_GIB:-4}; LOG=$KLT_HOME/ttm-drain.log
ma(){ awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo; }
pool(){ sudo -n cat /sys/kernel/debug/ttm/page_pool 2>/dev/null | awk '/^total/{print int($3/262144)}'; }   # 4 KiB pages -> GiB
a=$(ma); [ "$a" -ge "$NEED" ] && { echo "ttm=ok"; exit 0; }
p=$(pool); [ -z "$p" ] && { echo "ttm=unread(MemAvailable ${a}GB)"; exit 0; }
[ "$p" -le "$MINP" ] && { echo "ttm=pool${p}GiB(MemAvailable ${a}GB)"; exit 0; }
sudo -n sh -c 'echo 2 > /proc/sys/vm/drop_caches'; sleep 2; b=$(ma); p2=$(pool)
echo "$(date -u +%FT%TZ) ttm-drain: MemAvailable ${a}GB < ${NEED} with pool ${p}GiB -> drop_caches=2 -> MemAvailable ${b}GB pool ${p2:-?}GiB (caller pid $PPID: $(tr '\0' ' ' < /proc/$PPID/cmdline 2>/dev/null | cut -c1-120))" >> "$LOG"
echo "ttm=DRAINED(pool ${p}->${p2:-?}GiB,MemAvailable ${a}->${b}GB)"

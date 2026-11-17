#!/bin/bash
# chan-stop.sh <pid> <out.tsv>: SIGTERM a chan-start.sh sampler (empty pid = nothing to stop), wait for it, print the chan-summary
# --line on stdout ("chan: unavailable (...)" when it never started). CHAN_TAIL_S (default 0.3 s) lets the last window close.
case ${2:-} in ''|-h|--help) echo "usage: chan-stop.sh <pid|''> <out.tsv>" >&2; exit 2;; esac
T=$(dirname "$(readlink -f "$0")"); pid=${1:-}; out=$2
if [ -n "$pid" ]; then
  sleep "${CHAN_TAIL_S:-0.3}"; kill -TERM "$pid" 2>/dev/null
  for i in $(seq 50); do kill -0 "$pid" 2>/dev/null || break; sleep 0.1; done
  kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null
fi
python3 "$T/chan-summary.py" "$out" --line 2>&1 | tail -1

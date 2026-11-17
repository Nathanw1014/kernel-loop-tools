#!/usr/bin/env bash
# flt-io-sample.sh <parent pid> <out file>: samplers/flt-sample.sh plus I/O counters. Every 1 s, write
# "majflt minflt read_bytes rchar" of the parent's first child (e.g. the llama-bench / llama-server that `timeout` runs) to
# <out file>; the file always holds the latest line, at most 1 s before the child exited.
#   read_bytes  bytes fetched from the block layer (page-cache misses, readahead included)
#   rchar       bytes the process asked read() / pread() for (cache hits included)
# majflt with read_bytes ~ 0 = faults served from the page cache; read_bytes >> model size = the model is being re-read.
# Guards against: a cold-cache run passing as a warm one (or the reverse) in an A/B, and against blaming the GPU for time
# spent waiting on storage. Read-only, no signals: the child's kill path (timeout / watchdog) is untouched.
case ${1:-} in -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0;; esac
[ $# -eq 2 ] || { echo "usage: flt-io-sample.sh <parent pid> <out file>" >&2; exit 2; }
case $1 in *[!0-9]*|"") echo "flt-io-sample.sh: parent pid must be a number" >&2; exit 2;; esac
p=$1; out=$2; : > "$out"; c=""
for i in $(seq 1 20); do c=$(pgrep -P "$p" | head -1); [ -n "$c" ] && break; sleep 0.2; done
[ -z "$c" ] && exit 0
while kill -0 "$c" 2>/dev/null; do
  s=$(cat /proc/$c/stat 2>/dev/null) || break
  set -- ${s##*) }   # fields after "(comm) ": minflt = field 10 -> $8, majflt = field 12 -> $10
  rb=$(awk '/^read_bytes:/{print $2}' /proc/$c/io 2>/dev/null); rc=$(awk '/^rchar:/{print $2}' /proc/$c/io 2>/dev/null)
  echo "${10} ${8} ${rb:-?} ${rc:-?}" > "$out.tmp" && mv "$out.tmp" "$out"
  sleep 1
done

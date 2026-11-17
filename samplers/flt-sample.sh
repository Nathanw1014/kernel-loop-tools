#!/usr/bin/env bash
# flt-sample.sh <parent pid> <out file>: every 1 s, write "majflt minflt" of the parent's first child (e.g. the llama-bench that
# `timeout` runs; `env` execs into it) to <out file>; the last line is the count at most 1 s before exit. Read-only, no signals:
# the bench's kill path (timeout / watchdog) is untouched. (Wrapping the bench in GNU time instead would sit between timeout
# and the bench and orphan it on a kill.)
p=${1:?parent pid}; out=${2:?out file}; : > "$out"; c=""
for i in $(seq 1 20); do c=$(pgrep -P "$p" | head -1); [ -n "$c" ] && break; sleep 0.2; done
[ -z "$c" ] && exit 0
while kill -0 "$c" 2>/dev/null; do
  s=$(cat /proc/$c/stat 2>/dev/null) || break
  set -- ${s##*) }   # fields after "(comm) ": $1 = state (field 3), minflt = field 10 -> $8, majflt = field 12 -> $10
  echo "${10} ${8}" > "$out.tmp" && mv "$out.tmp" "$out"
  sleep 1
done

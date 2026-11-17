#!/usr/bin/env bash
# memtrace.sh <out-prefix> <cmd> [args...]: run cmd and record where its memory goes, once a second, until it exits.
#   <prefix>.tsv   time | MemAvailable Cached AnonPages Shmem (kB) | the process's VmRSS RssAnon RssFile RssShmem (kB)
#                  | GPU GTT in use (MiB, the largest of all cards); last line "rc=<exit code>"
#   <prefix>.maps  every 3 s: the process's mappings over ~200 MB by Rss, largest first (anon mappings show buffer sizes;
#                  file mappings show what of the model is mmap'd and resident)
# Guards against: guessing why a load is slow or a job is OOM-killed (page cache vs anon copy vs GTT) from MemAvailable
# alone, and against reading the steady state when the peak is what kills.
# Exit: the command's exit code. Samples only cmd's own pid (wrap a launcher with exec if it forks the real process).
set -u
case ${1:-} in -h|--help|"") awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; [ -n "${1:-}" ] && exit 0 || exit 2;; esac
[ $# -ge 2 ] || { echo "usage: memtrace.sh <out-prefix> <cmd> [args...]" >&2; exit 2; }
P=$1; shift
"$@" &
PID=$!
( n=0
  while kill -0 $PID 2>/dev/null; do
    t=$(date +%T)
    ma=$(awk '/MemAvailable/{a=$2}/^Cached/{c=$2}/AnonPages/{n=$2}/^Shmem:/{s=$2}END{print a,c,n,s}' /proc/meminfo)
    st=$(awk '/VmRSS/{r=$2}/RssAnon/{a=$2}/RssFile/{f=$2}/RssShmem/{s=$2}END{print r,a,f,s}' /proc/$PID/status 2>/dev/null)
    g=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | sort -n | tail -1)
    echo "$t $ma | $st | gtt $(( ${g:-0} / 1048576 ))" >> "$P.tsv"
    n=$((n+1)); if [ $((n % 3)) = 1 ]; then
      echo "== $t" >> "$P.maps"
      awk '/^[0-9a-f]+-[0-9a-f]+ /{name=$6; range=$1} /^Rss:/{if($2>200000) printf "%d kB %s %s\n",$2,range,name}' /proc/$PID/smaps 2>/dev/null | sort -rn | head -8 >> "$P.maps"
    fi
    sleep 1
  done ) &
SAMP=$!
wait $PID; rc=$?
wait $SAMP 2>/dev/null
echo "rc=$rc" >> "$P.tsv"
exit $rc

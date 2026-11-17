#!/usr/bin/env bash
# gpu-state.sh: GPU exclusivity probe for bench runners. Source it (run-bench.sh does); it defines
#   gpu_exclusive   "yes", or "no (<n> proc, <MiB> resident)" for other processes holding amdgpu memory
#   gpu_busy_pre    max gpu_busy_percent over 0.5 s before the run
#   mem_settle      wait until MemAvailable >= BENCH_MEM_MIN_GB
#
# Ask the driver, not the process list: the only contention that perturbs a measurement is another
# process holding amdgpu memory while we run, and amdgpu fdinfo reports that directly. A process-name
# grep matches unrelated processes and cannot say what its count means.

if ! declare -F klt_gpu_card >/dev/null; then
  . "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../lib/klt-env.sh"
fi
DRM_BUSY=""
_card=$(klt_gpu_card 2>/dev/null) && DRM_BUSY=$_card/gpu_busy_percent

# Counts and bytes only. Process names are deliberately NOT part of the record: the fact a reader
# needs is "was the GPU ours", not which other programs happened to be up.
gpu_exclusive() {
  local -A peak=(); local f pid kib n=0 tot=0
  while IFS= read -r f; do
    pid=${f#/proc/}; pid=${pid%%/*}
    [ "$pid" = "$$" ] && continue
    # One process opens the render node several times and every fd repeats the same client
    # totals, so summing across fds double counts. Take the largest single fd per pid.
    kib=$(awk '/^drm-resident-(gtt|vram):/{s+=$2} END{print s+0}' "$f" 2>/dev/null)
    [ "${kib:-0}" -gt "${peak[$pid]:-0}" ] && peak[$pid]=$kib
  done < <(grep -ls '^drm-driver:[[:space:]]*amdgpu' /proc/[0-9]*/fdinfo/* 2>/dev/null)
  for pid in "${!peak[@]}"; do
    [ "${peak[$pid]}" -gt 0 ] || continue
    n=$((n+1)); tot=$((tot+peak[$pid]))
  done
  if [ "$n" -eq 0 ]; then echo yes; else echo "no ($n proc, $((tot/1024)) MiB resident)"; fi
}

gpu_busy_pre() {
  local i m=0 v
  [ -n "$DRM_BUSY" ] && [ -r "$DRM_BUSY" ] || { echo unknown; return; }
  for i in 1 2 3 4 5; do
    v=$(cat "$DRM_BUSY" 2>/dev/null); v=${v:-0}
    [ "$v" -gt "$m" ] && m=$v
    sleep 0.1
  done
  echo "$m (max of 5 samples over 0.5 s)"
}

# The GPU lock serialises GPU use, not memory. A process that pinned a large mmap'd model can release
# it many seconds after it exits, and a job that takes the lock the instant the previous holder exits
# can load into memory that is not free yet (OOM). Call after taking the lock: wait until
# MemAvailable >= BENCH_MEM_MIN_GB for up to BENCH_MEM_WAIT s, then proceed with a warning.
# BENCH_MEM_MIN_GB defaults to 0 (off): set it to the model's resident size plus a few GB.
mem_settle() {
  local need=${BENCH_MEM_MIN_GB:-0} wait=${BENCH_MEM_WAIT:-180} i avail
  [ "$need" -gt 0 ] || return 0
  for ((i=0; i<wait; i+=5)); do
    avail=$(awk '/MemAvailable/{print int($2/1024/1024)}' /proc/meminfo)
    [ "$avail" -ge "$need" ] && { [ $i -gt 0 ] && echo "[mem_settle] waited ${i}s for ${need} GB (now ${avail} GB)"; return 0; }
    sleep 5
  done
  echo "[mem_settle] WARNING: only ${avail} GB available after ${wait}s (need ${need}); proceeding" >&2
  return 1
}

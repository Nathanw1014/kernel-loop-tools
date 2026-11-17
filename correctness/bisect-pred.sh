#!/usr/bin/env bash
# bisect-pred.sh: predicate for `git bisect run`, for failures no env knob can isolate.
#
#   git -C <worktree> bisect start <bad> <good>
#   git -C <worktree> bisect run <repo>/correctness/bisect-pred.sh <build-dir> ops "<op filter>" [<-p param filter>]
#   git -C <worktree> bisect run <repo>/correctness/bisect-pred.sh <build-dir> ppl <max-ppl> <model> <corpus> [llama-perplexity flags...]
#
# ops: good when test-backend-ops on the filter runs > 0 cases and reports zero failures (0 cases = skip, never "good").
# ppl: good when llama-perplexity -c 4096 --chunks 4 on <corpus> is <= <max-ppl> (base PPL + tolerance) and not NaN.
#      Extra flags go after the defaults (-c 4096 -b 2048 -ub 2048 --chunks 4 -ngl 999 -fa on), so they can override them.
# Exit codes: 0 good, 1 bad, 125 skip (build failed / nothing ran), 128 abort (device error: never bisect through a GPU
# fault; also writes $KLT_STOP).
# Env: BACKEND (test-backend-ops -b, default Vulkan0), BUILD_JOBS (default nproc), MEM_GB (MemAvailable to wait for before
# the test, default 0 = off: set it to the model's resident size plus a few GB), GTT_MAX_GIB (GPU GTT in use by others to
# wait below, default 2).
# Builds incrementally in <build-dir>; takes the GPU lock ($KLT_LOCK) only for the test itself.
set -uo pipefail
case ${1:-} in -h|--help|'') sed -n '2,16p' "$0"; exit 2;; esac
BD=$1; MODE=${2:-}; shift; [ $# -gt 0 ] && shift
. "$(dirname "$(readlink -f "$0")")/../lib/klt-env.sh"
# KLT_VK_ICD pins the Vulkan ICD (KLT_VK_LIBPATH: extra library dirs for it, e.g. a matching libdrm), as in abba.sh / job-run.
[ -n "${KLT_VK_ICD:-}" ] && { [ -f "$KLT_VK_ICD" ] || { echo "KLT_VK_ICD missing: $KLT_VK_ICD" >&2; exit 128; }; export VK_ICD_FILENAMES=$KLT_VK_ICD VK_DRIVER_FILES=$KLT_VK_ICD; }
[ -n "${KLT_VK_LIBPATH:-}" ] && export LD_LIBRARY_PATH=$KLT_VK_LIBPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
[ -e "$KLT_STOP" ] && { echo "bisect-pred: $KLT_STOP exists, aborting the bisect" >&2; exit 128; }
case $MODE in
  ops) [ $# -ge 1 ] || { echo "ops needs an op filter" >&2; exit 128; };;
  ppl) [ $# -ge 3 ] || { echo "ppl needs <max-ppl> <model> <corpus>" >&2; exit 128; }
       MAXP=$1; MODEL=$2; CORPUS=$3; shift 3; F=("$@")
       [ -r "$MODEL" ] && [ -r "$CORPUS" ] || { echo "model or corpus not readable" >&2; exit 128; };;
  *) echo "mode must be ops or ppl" >&2; exit 128;;
esac
GPU=$(klt_gpu_card) || { echo "no amdgpu card (set KLT_GPU_CARD)" >&2; exit 128; }
GTT=$GPU/mem_info_gtt_used
SHA=$(git rev-parse --short HEAD); LOG=$BD/bisect-$SHA.log
nice -n 10 cmake --build "$BD" -j "${BUILD_JOBS:-$(nproc)}" --target test-backend-ops llama-perplexity > "$BD/bisect-build-$SHA.log" 2>&1 || exit 125
B=$BD/bin
exec 9>"$KLT_LOCK"; flock 9; echo "pid=$$ label=bisect-$SHA started=$(date -Is)" > "$KLT_LOCK.owner"
while [ $(( $(cat "$GTT") / 1048576 )) -gt $(( ${GTT_MAX_GIB:-2} * 1024 )) ] || [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -lt "${MEM_GB:-0}" ]; do sleep 10; done
case $MODE in
  ops) LD_LIBRARY_PATH=$B${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} timeout 3600 "$B/test-backend-ops" test -b "${BACKEND:-Vulkan0}" -o "$1" ${2:+-p "$2"} 9>&- > "$LOG" 2>&1 ;;
  ppl) LD_LIBRARY_PATH=$B${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} "$B/llama-perplexity" -m "$MODEL" -f "$CORPUS" -c 4096 -b 2048 -ub 2048 --chunks 4 -ngl 999 -fa on "${F[@]}" 9>&- > "$LOG" 2>&1 ;;
esac
rm -f "$KLT_LOCK.owner"; exec 9>&-
if grep -aqE 'DeviceLost|Not enough memory for command|BO_VA' "$LOG"; then
  echo "$(date -u +%FT%TZ) bisect-pred $SHA: device error in $LOG" > "$KLT_STOP"; exit 128
fi
case $MODE in
  ops) line=$(sed -E 's/\x1b\[[0-9;]*m//g' "$LOG" | grep -E 'tests passed' | tail -1); echo "$SHA $line" >> "$BD/bisect.txt"
       n=$(echo "$line" | grep -oE '[0-9]+/[0-9]+' | cut -d/ -f2); [ "${n:-0}" -gt 0 ] || exit 125
       grep -aq 'FAIL' "$LOG" && exit 1 || exit 0 ;;
  ppl) p=$(grep -aoE 'Final estimate: PPL = [0-9.]+' "$LOG" | grep -oE '[0-9.]+$'); echo "$SHA ppl=${p:-none}" >> "$BD/bisect.txt"
       [ -n "$p" ] || exit 1
       python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)" "$p" "$MAXP" ;;
esac

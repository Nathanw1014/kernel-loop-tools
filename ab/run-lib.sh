#!/usr/bin/env bash
# run-lib.sh: shared body of the record-keeping runners run-quality.sh, run-tbo.sh and run-batched.sh (source it; not a
# command). Same discipline as run-bench.sh: refuse while $KLT_STOP exists, take $KLT_LOCK (owner line in $KLT_LOCK.owner),
# settle memory, pin the Vulkan ICD from $KLT_VK_ICD, write <id>.meta with build / env / argv / machine state, then the
# runner writes the verbatim output next to it and appends one row to an index under $KLT_RESULTS.
#
# Functions (call in this order):
#   rl_start <tag> <label> <binary>   STOP check, lock, mem_settle, run id; sets ID META RAW and R=$KLT_RESULTS
#   rl_meta <backend> <argv...>        writes $META
#   rl_finish <rc>                     appends exit_code to $META
# Env: BENCH_NO_LOCK=1, BENCH_LOCK_WAIT=s (7200), BENCH_MEM_MIN_GB, BENCH_SOURCE_TREE (as run-bench.sh).
# Exit codes a runner inherits: 70 STOP file present, 75 lock wait timed out, 78 KLT_VK_ICD missing, 2 usage.

RL_HERE=$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")
. "$RL_HERE/../lib/klt-env.sh"

rl_start() {
  RL_TAG=$1; LABEL=$2; BIN=$3
  [ -x "$BIN" ] || { echo "[$RL_TAG] not executable: $BIN" >&2; exit 2; }
  [ -e "$KLT_STOP" ] && { echo "[$RL_TAG] $KLT_STOP exists (device error recorded); refusing to run: $(head -1 "$KLT_STOP")" >&2; exit 70; }
  R=$KLT_RESULTS; mkdir -p "$R/raw"
  LOCK_WAITED=0
  if [ "${BENCH_NO_LOCK:-0}" != 1 ]; then
    local wait=${BENCH_LOCK_WAIT:-7200} t0
    exec 9>"$KLT_LOCK" || { echo "[$RL_TAG] cannot open lock $KLT_LOCK" >&2; exit 1; }
    if ! flock -n 9; then
      echo "[$RL_TAG] GPU busy, lock held by: $(cat "$KLT_LOCK.owner" 2>/dev/null || echo unknown); waiting up to ${wait}s"
      t0=$SECONDS
      flock -w "$wait" 9 || { echo "[$RL_TAG] TIMED OUT after ${wait}s waiting for the GPU lock; refusing to run" >&2; exit 75; }
      LOCK_WAITED=$((SECONDS - t0))
    fi
    printf 'pid=%s label=%s started=%s\n' "$$" "$LABEL" "$(date -Is)" > "$KLT_LOCK.owner"
    trap 'rm -f "$KLT_LOCK.owner"' EXIT
  fi
  . "$RL_HERE/gpu-state.sh"
  mem_settle
  TS=$(date -u +%Y%m%dT%H%M%SZ); ID="${TS}_${LABEL}"
  META="$R/raw/$ID.meta"; RAW="$R/raw/$ID.raw"
}

rl_icd() {   # rl_icd <backend>: pin the ICD for vulkan runs; no silent fallback
  if [ "$1" = vulkan ] && [ -n "$KLT_VK_ICD" ]; then
    [ -f "$KLT_VK_ICD" ] || { echo "[$RL_TAG] KLT_VK_ICD=$KLT_VK_ICD missing; refusing to run (unset it for the system driver)" >&2; exit 78; }
    export VK_ICD_FILENAMES=$KLT_VK_ICD VK_DRIVER_FILES=$KLT_VK_ICD
    [ -n "${KLT_VK_LIBPATH:-}" ] && export LD_LIBRARY_PATH=$KLT_VK_LIBPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
  fi
}

rl_meta() {
  local backend=$1; shift
  local tree=${BENCH_SOURCE_TREE:-$(dirname "$BIN")}
  SRC_COMMIT=$(git -C "$tree" rev-parse --short HEAD 2>/dev/null || echo unknown)
  {
    echo "run_id: $ID"
    echo "timestamp_utc: $TS"
    echo "label: $LABEL"
    echo "backend: $backend"
    echo "binary: $BIN"
    echo "source_commit: $SRC_COMMIT"
    echo "source_branch: $(git -C "$tree" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "source_dirty_files: $(git -C "$tree" status --porcelain 2>/dev/null | wc -l)"
    echo "argv: $BIN $*"
    echo "--- ggml/vk env ---"
    env | grep -E '^(GGML_|VK_|LLAMA_|RADV_|AMD_|LD_LIBRARY_PATH=)' | sort
    echo "vk_driver: $(vulkaninfo --summary 2>/dev/null | grep driverInfo | sed 's/.*= *//' | paste -sd';')"
    echo "--- machine ---"
    echo "kernel: $(uname -r)"
    echo "cmdline_iommu: $(tr ' ' '\n' < /proc/cmdline | grep -i iommu | tr '\n' ' ')"
    echo "mem_free_gib: $(free -g | awk '/^Mem:/{print $4" free / "$7" avail / "$2" total"}')"
    echo "gpu_exclusive: $(gpu_exclusive)"
    echo "gpu_busy_pct_pre: $(gpu_busy_pre)"
    echo "gpu_lock_wait_s: ${LOCK_WAITED:-0}"
  } > "$META"
}

rl_finish() { echo "exit_code: $1" >> "$META"; }

rl_index() {   # rl_index <index file> <header> <row>: create with header once, then append the row
  [ -f "$1" ] || printf '%s\n' "$2" > "$1"
  printf '%s\n' "$3" >> "$1"
}

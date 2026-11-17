#!/usr/bin/env bash
# run-bench.sh: record-keeping benchmark runner.
#
# usage: run-bench.sh <label> <backend:vulkan|rocm> <llama-bench binary> -- <llama-bench args...>
#
# Every invocation writes THREE artifacts under $KLT_RESULTS/raw/, all keyed by the same run id:
#   <id>.meta      timestamp, label, binary, build hash, env, exact argv, machine state
#   <id>.raw       verbatim stderr of the benchmark (device / init lines included), unfiltered
#   <id>.json      llama-bench -o json results (for graphing)
# and appends one row to $KLT_RESULTS/INDEX.tsv.
#
# Nothing here summarises or interprets. Claims should be reconstructable from .json/.raw alone.
#
# Env:
#   KLT_VK_ICD          Vulkan ICD json for vulkan runs (empty = system driver); a missing file refuses the run
#   BENCH_NO_LOCK=1     skip the GPU lock ($KLT_LOCK); BENCH_LOCK_WAIT=s max wait for it (default 7200)
#   BENCH_MEM_MIN_GB    wait for this much MemAvailable after the lock (default 0 = off; see gpu-state.sh)
#   BENCH_SOURCE_TREE   git tree the binary was built from (default: the binary's directory)
#   DRAM_SAMPLE=1       also record raw/<id>.dram.tsv with mem/dram-sample.py (DRAM_PERIOD_MS, default 50)
#                       and a "dram:" busy-GB/s line in <id>.meta (mem/DRAM-MONITOR.md)
#   CHAN_SAMPLE=1       per-channel DRAM requests via mem/chan-start.sh (CHAN_PERIOD_MS, default 100)
#                       -> raw/<id>.chan.tsv and a "chan:" line in <id>.meta
# Refuses to start while $KLT_STOP exists (written by a tool that saw a device error).
set -uo pipefail
usage() { sed -n '2,23p' "$0"; exit 2; }
[ $# -ge 3 ] || usage
case ${1:-} in -h|--help) usage;; esac
LABEL=$1; BACKEND=$2; BIN=$3; shift 3
[ "${1:-}" = "--" ] && shift
ARGS=("$@")
[ -x "$BIN" ] || { echo "[run-bench] not executable: $BIN" >&2; exit 2; }

HERE=$(dirname "$(readlink -f "$0")")
. "$HERE/../lib/klt-env.sh"
[ -e "$KLT_STOP" ] && { echo "[run-bench] $KLT_STOP exists (device error recorded); refusing to run: $(head -1 "$KLT_STOP")" >&2; exit 70; }
R=$KLT_RESULTS
mkdir -p "$R/raw"
INDEX="$R/INDEX.tsv"
[ -f "$INDEX" ] || printf 'run_id\ttimestamp\tlabel\tbackend\tbinary\tbuild\targs\n' > "$INDEX"

# --- GPU mutex --------------------------------------------------------------
# Two jobs benching the same GPU concurrently does not perturb results, it destroys them: a
# co-resident process can halve a cell and even make a depth curve get faster with depth.
# Serialise instead of hoping.
LOCKWAIT=${BENCH_LOCK_WAIT:-7200}
LOCK_WAITED=0
if [ "${BENCH_NO_LOCK:-0}" != 1 ]; then
  exec 9>"$KLT_LOCK" || { echo "[run-bench] cannot open lock $KLT_LOCK" >&2; exit 1; }
  if ! flock -n 9; then
    echo "[run-bench] GPU busy, lock held by: $(cat "$KLT_LOCK.owner" 2>/dev/null || echo unknown)"
    echo "[run-bench] waiting up to ${LOCKWAIT}s ..."
    _t0=$SECONDS
    if ! flock -w "$LOCKWAIT" 9; then
      echo "[run-bench] TIMED OUT after ${LOCKWAIT}s waiting for the GPU lock; refusing to run" >&2
      exit 75
    fi
    LOCK_WAITED=$((SECONDS-_t0))
    echo "[run-bench] lock acquired after ${LOCK_WAITED}s"
  fi
  printf 'pid=%s label=%s started=%s\n' "$$" "$LABEL" "$(date -Is)" > "$KLT_LOCK.owner"
  trap 'rm -f "$KLT_LOCK.owner"' EXIT
fi
# ----------------------------------------------------------------------------

. "$HERE/gpu-state.sh"   # gpu_exclusive / gpu_busy_pre / mem_settle
mem_settle   # wait for the previous lock holder's pinned memory to be released

TS=$(date -u +%Y%m%dT%H%M%SZ)
ID="${TS}_${LABEL}"
META="$R/raw/$ID.meta"; RAW="$R/raw/$ID.raw"; JSON="$R/raw/$ID.json"

# Vulkan driver: KLT_VK_ICD pins one ICD for the run (base and candidate must share a driver);
# empty = whatever the caller's environment selects. No silent fallback: a missing ICD refuses the run.
if [ "$BACKEND" = vulkan ] && [ -n "$KLT_VK_ICD" ]; then
  [ -f "$KLT_VK_ICD" ] || { echo "[run-bench] KLT_VK_ICD=$KLT_VK_ICD missing; refusing to run (unset it for the system driver)" >&2; exit 78; }
  export VK_ICD_FILENAMES=$KLT_VK_ICD VK_DRIVER_FILES=$KLT_VK_ICD
  [ -n "${KLT_VK_LIBPATH:-}" ] && export LD_LIBRARY_PATH=$KLT_VK_LIBPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi

# A binary built without its git dir (e.g. inside a container) reports build_commit "unknown".
# Resolve the source commit on the host instead; BENCH_SOURCE_TREE overrides for binaries that
# do not live in the tree they were built from.
TREE=${BENCH_SOURCE_TREE:-$(dirname "$BIN")}
SRC_COMMIT=$(git -C "$TREE" rev-parse --short HEAD 2>/dev/null || echo unknown)
SRC_BRANCH=$(git -C "$TREE" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)
# Some llama-bench builds have no --version (they print help instead). Probe it, but only trust
# output that looks like a version.
VERSION_RAW=$("$BIN" --version 2>&1 | head -3)
# NB: the failure text itself contains "--version", so test for failure markers, not for
# the word "version". Anything that looks like usage/help/an error is not a version string.
if ! printf '%s' "$VERSION_RAW" | grep -qiE 'error|invalid parameter|usage:|Found [0-9]+ Vulkan'; then
  BUILD=$(printf '%s' "$VERSION_RAW" | tr '\n' ' ' | sed 's/  */ /g')
else
  BUILD=""   # filled from the run's own JSON below (build_commit/build_number)
fi

{
  echo "run_id: $ID"
  echo "timestamp_utc: $TS"
  echo "label: $LABEL"
  echo "backend: $BACKEND"
  echo "binary: $BIN"
  echo "build: ${BUILD:-unknown}"
  echo "source_commit: $SRC_COMMIT"
  echo "source_branch: $SRC_BRANCH"
  echo "argv: llama-bench ${ARGS[*]}"
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

# One process: json on stdout for graphing, the full stderr (device / init lines) as the raw record.
M=$KLT_ROOT/mem
CSP=""; [ "${CHAN_SAMPLE:-0}" = 1 ] && CSP=$(CHAN_WATCH_PID=$$ "$M/chan-start.sh" "$R/raw/$ID.chan.tsv" "${CHAN_PERIOD_MS:-100}" 9>&- 2>> "$META")
if [ "${DRAM_SAMPLE:-0}" = 1 ]; then
  "$BIN" "${ARGS[@]}" -o json > "$JSON" 2> "$RAW" & BP=$!
  python3 "$M/dram-sample.py" "$R/raw/$ID.dram.tsv" $BP "${DRAM_PERIOD_MS:-50}" 9>&- 2>> "$META" & DSP=$!
  wait $BP; RC=$?; wait $DSP
  echo "dram: $(python3 "$M/dram-summary.py" "$R/raw/$ID.dram.tsv" --line 2>&1 | tail -1) ($ID.dram.tsv)" >> "$META"
else
  "$BIN" "${ARGS[@]}" -o json > "$JSON" 2> "$RAW"
  RC=$?
fi
[ "${CHAN_SAMPLE:-0}" = 1 ] && echo "chan: $("$M/chan-stop.sh" "$CSP" "$R/raw/$ID.chan.tsv") ($ID.chan.tsv)" >> "$META"
echo "exit_code: $RC" >> "$META"
# Authoritative build identity: llama-bench emits build_commit/build_number into its own JSON.
if [ -s "$JSON" ]; then
  BJ=$(python3 -c "import json,sys;d=json.load(open(sys.argv[1]));r=d[0] if isinstance(d,list) and d else {};print(('%s (%s)'%(r.get('build_commit','?'),r.get('build_number','?'))))" "$JSON" 2>/dev/null)
  [ -n "$BJ" ] && echo "build_from_json: $BJ" >> "$META"
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ID" "$TS" "$LABEL" "$BACKEND" "$BIN" "${BUILD:-unknown}" "${ARGS[*]}" >> "$INDEX"
echo "[run-bench] $ID rc=$RC -> $JSON"
exit $RC

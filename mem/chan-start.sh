#!/bin/bash
# chan-start.sh <out.tsv> [period_ms=100]: start the gcea-sample sampler (per-instance GCEA requests of HIP / KFD work in any process;
# Vulkan work may read zero depending on the driver; no root) in the background, wait CHAN_LEAD_S (default 1 s, covers HSA init),
# print its PID on stdout. Prints nothing (and writes the reason to <out.tsv>) when unavailable (no sampler binary, no ROCm runtime,
# no /dev/kfd access). Stop with chan-stop.sh <pid> <out.tsv>.
# Env:
#   GCEA_SAMPLE     sampler binary (default <this dir>/bin/gcea-sample, as built by build-gcea-sample.sh)
#   CHAN_ROCM       ROCm root holding lib/librocprofiler-sdk.so.1 (default $ROCM_PATH, then /opt/rocm)
#   CHAN_GROUPS     counter groups (default GCEA_RDRAM_SIZE_REQ then GCEA_WDRAM_SIZE_REQ, round-robin)
#   CHAN_WATCH_PID  the sampler exits when this pid exits (default the calling shell), so a hard-killed cell cannot leak a KFD process
case ${1:-} in ''|-h|--help) echo "usage: chan-start.sh <out.tsv> [period_ms=100]" >&2; exit 2;; esac
T=$(dirname "$(readlink -f "$0")"); out=$1; per=${2:-100}
for R in "${CHAN_ROCM:-}" "${ROCM_PATH:-}" /opt/rocm; do
  [ -n "$R" ] && [ -e "$R/lib/librocprofiler-sdk.so.1" ] && break; R=
done
S=${GCEA_SAMPLE:-$T/bin/gcea-sample}; why=
[ -x "$S" ] || why="no $S"; [ -z "$why" ] && [ -z "$R" ] && why="no ROCm runtime (set CHAN_ROCM)"
[ -z "$why" ] && { [ -r /dev/kfd ] && [ -w /dev/kfd ] || why="no /dev/kfd access"; }
[ -n "$why" ] && { echo "# chan: unavailable ($why)" > "$out"; exit 0; }
LD_LIBRARY_PATH="$R/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$S" "$out" "${CHAN_WATCH_PID:-$PPID}" "$per" ${CHAN_GROUPS:-} > /dev/null 2> "$out.err" 9>&- < /dev/null & D=$!
sleep "${CHAN_LEAD_S:-1}"
kill -0 $D 2>/dev/null && echo $D || { echo "# chan: unavailable (sampler exited: $(tail -1 "$out.err" 2>/dev/null))" >> "$out"; }

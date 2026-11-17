#!/usr/bin/env bash
# run-tbo.sh: record-keeping runner for test-backend-ops; sibling of run-bench.sh.
#
# usage: run-tbo.sh <label> <backend:vulkan|rocm|cpu> <test-backend-ops binary> -- <test-backend-ops args...>
#   e.g. run-tbo.sh fa-tip vulkan ./tip/bin/test-backend-ops -- -b Vulkan0 -o FLASH_ATTN_EXT
#        run-tbo.sh fa-perf vulkan ./tip/bin/test-backend-ops -- perf -b Vulkan0 -o FLASH_ATTN_EXT
#
# Why not run-bench.sh: it appends "-o json", and in test-backend-ops -o is the OP FILTER, so that run selects no tests at all
# and still exits 0. This runner passes the args through untouched.
# Writes <id>.meta and <id>.raw (verbatim stdout + stderr) under $KLT_RESULTS/raw/ and one row to $KLT_RESULTS/INDEX.tsv
# (run-bench.sh's columns). The closing line echoes the "N/M tests passed" summary; a test-mode run that printed no such
# line, or 0/0, is flagged NO-TESTS (also "no_tests: yes" in .meta) and exits 3 if the binary itself exited 0: a filter that
# matches nothing is not a pass.
# Same lock, STOP refusal, memory settle and ICD pinning as run-bench.sh.
set -uo pipefail
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
case ${1:-} in -h|--help) usage; exit 0;; esac
[ $# -ge 3 ] || { usage >&2; exit 2; }
LABEL=$1; BACKEND=$2; BINARG=$3; shift 3
[ "${1:-}" = "--" ] && shift
ARGS=("$@")
. "$(dirname "$(readlink -f "$0")")/run-lib.sh"
rl_start run-tbo "$LABEL" "$BINARG"
rl_icd "$BACKEND"
rl_meta "$BACKEND" "${ARGS[@]}"
"$BIN" "${ARGS[@]}" > "$RAW" 2>&1
RC=$?
rl_finish $RC
SUM=$(grep -oE '[0-9]+/[0-9]+ tests passed' "$RAW" | tail -1)
MODE=test; case " ${ARGS[*]} " in *" perf "*) MODE=perf;; *" grad "*) MODE=grad;; *" support "*) MODE=support;; esac
TOTAL=$(echo "$SUM" | sed -n 's|^[0-9]*/\([0-9]*\) .*|\1|p')
if [ "$MODE" = test ] && [ "${TOTAL:-0}" = 0 ]; then
  SUM="NO-TESTS (${SUM:-no 'tests passed' line})"
  echo "no_tests: yes" >> "$META"
  [ "$RC" = 0 ] && RC=3
fi
rl_index "$R/INDEX.tsv" $'run_id\ttimestamp\tlabel\tbackend\tbinary\tbuild\targs' \
  "$ID"$'\t'"$TS"$'\t'"$LABEL"$'\t'"$BACKEND"$'\t'"$BIN"$'\t'"$SRC_COMMIT"$'\t'"${ARGS[*]}"
echo "[run-tbo] $ID rc=$RC ${SUM:-} -> $RAW"
exit $RC

#!/usr/bin/env bash
# run-batched.sh: record-keeping runner for llama-batched-bench (concurrency / aggregate-throughput sweeps); sibling of
# run-bench.sh.
#
# usage: run-batched.sh <label> <backend:vulkan|rocm|cpu> <llama-batched-bench binary> -- <llama-batched-bench args...>
#   e.g. run-batched.sh np-sweep vulkan ./tip/bin/llama-batched-bench -- -m M.gguf -c 16384 -npp 512 -ntg 128 -npl 1,2,4,8
#
# llama-batched-bench has no "-o json"; it has --output-format jsonl, which this runner appends. Artifacts under
# $KLT_RESULTS/raw/: <id>.meta, <id>.raw (stderr, verbatim), <id>.jsonl (stdout); one row to $KLT_RESULTS/INDEX.tsv.
# Same lock, STOP refusal, memory settle and ICD pinning as run-bench.sh.
set -uo pipefail
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
case ${1:-} in -h|--help) usage; exit 0;; esac
[ $# -ge 3 ] || { usage >&2; exit 2; }
LABEL=$1; BACKEND=$2; BINARG=$3; shift 3
[ "${1:-}" = "--" ] && shift
ARGS=("$@")
. "$(dirname "$(readlink -f "$0")")/run-lib.sh"
rl_start run-batched "$LABEL" "$BINARG"
OUTJ="$R/raw/$ID.jsonl"
rl_icd "$BACKEND"
rl_meta "$BACKEND" "${ARGS[@]}" --output-format jsonl
"$BIN" "${ARGS[@]}" --output-format jsonl > "$OUTJ" 2> "$RAW"
RC=$?
rl_finish $RC
rl_index "$R/INDEX.tsv" $'run_id\ttimestamp\tlabel\tbackend\tbinary\tbuild\targs' \
  "$ID"$'\t'"$TS"$'\t'"$LABEL"$'\t'"$BACKEND"$'\t'"$BIN"$'\t'"$SRC_COMMIT"$'\t'"${ARGS[*]}"
echo "[run-batched] $ID rc=$RC -> $OUTJ"
exit $RC

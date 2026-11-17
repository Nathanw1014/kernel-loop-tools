#!/usr/bin/env bash
# run-quality.sh: record-keeping runner for llama-perplexity (perplexity and KL divergence); sibling of run-bench.sh.
#
# usage: run-quality.sh <label> <backend:vulkan|rocm|cpu> <llama-perplexity binary> -- <llama-perplexity args...>
#   e.g. run-quality.sh tip-kld vulkan ./tip/bin/llama-perplexity -- -m M.gguf -f wiki.txt -c 4096 \
#            --kl-divergence-base base.kld --kl-divergence
#
# llama-perplexity has no JSON output, so each run writes two artifacts under $KLT_RESULTS/raw/, keyed by one run id:
#   <id>.meta   timestamp, label, binary, source commit, env, exact argv, machine state (ab/run-lib.sh)
#   <id>.raw    verbatim stdout + stderr, unfiltered
# and appends one row to $KLT_RESULTS/QUALITY.tsv with PPL, mean KLD and same-top-p parsed from .raw. Those columns are a
# convenience; .raw is the record. The KLD block's labels are padded ("Mean    KLD:"); the parse handles that, and a run
# without the block gets NA, never a number from another line.
# Same lock, STOP refusal, memory settle and ICD pinning as run-bench.sh (env: KLT_VK_ICD, BENCH_NO_LOCK, BENCH_LOCK_WAIT,
# BENCH_MEM_MIN_GB, BENCH_SOURCE_TREE). Compare two runs with correctness/kldtool.py log A.raw B.raw.
set -uo pipefail
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
case ${1:-} in -h|--help) usage; exit 0;; esac
[ $# -ge 3 ] || { usage >&2; exit 2; }
LABEL=$1; BACKEND=$2; BINARG=$3; shift 3
[ "${1:-}" = "--" ] && shift
ARGS=("$@")
. "$(dirname "$(readlink -f "$0")")/run-lib.sh"
rl_start run-quality "$LABEL" "$BINARG"
rl_icd "$BACKEND"
rl_meta "$BACKEND" "${ARGS[@]}"
"$BIN" "${ARGS[@]}" > "$RAW" 2>&1
RC=$?
rl_finish $RC
PPL=$(grep -oE 'Final estimate: PPL = [0-9.]+' "$RAW" | tail -1 | grep -oE '[0-9.]+$')
[ -n "$PPL" ] || PPL=$(sed 's/[[:space:]]\+/ /g' "$RAW" | grep -oE 'Mean PPL\(Q\) ?: [0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
KLD=$(sed 's/[[:space:]]\+/ /g' "$RAW" | grep -oE 'Mean KLD: -?[0-9.]+' | tail -1 | grep -oE -- '-?[0-9.]+$')
TOP1=$(sed 's/[[:space:]]\+/ /g' "$RAW" | grep -oE 'Same top p: [0-9.]+' | tail -1 | grep -oE '[0-9.]+$')
rl_index "$R/QUALITY.tsv" $'run_id\ttimestamp\tlabel\tbackend\tbinary\tcommit\tppl\tkld_mean\ttop1_agree\targs' \
  "$ID"$'\t'"$TS"$'\t'"$LABEL"$'\t'"$BACKEND"$'\t'"$BIN"$'\t'"$SRC_COMMIT"$'\t'"${PPL:-NA}"$'\t'"${KLD:-NA}"$'\t'"${TOP1:-NA}"$'\t'"${ARGS[*]}"
echo "[run-quality] $ID rc=$RC ppl=${PPL:-NA} kld=${KLD:-NA} top1=${TOP1:-NA} -> $RAW"
exit $RC

#!/bin/bash
# dram-run.sh <out.tsv> [period_ms=50] -- <cmd> [args]: run <cmd> with dram-sample.py on it, then print the busy DRAM GB/s summary
# line (dram-summary.py --line) to stderr; exits with <cmd>'s rc. For hand-written cell scripts: prefix the GPU command, e.g.
#   ${DRAM_SAMPLE:+$KLT_ROOT/mem/dram-run.sh $cell/dram.tsv --} llama-bench ...
# It takes no lock: run it inside the caller's lock. Env: KLT_GPU_CARD, DRAM_CAL (see dram-sample.py). Method: DRAM-MONITOR.md.
case ${1:-} in ''|-h|--help) echo "usage: dram-run.sh <out.tsv> [period_ms=50] -- <cmd> [args]" >&2; exit 2;; esac
T=$(dirname "$(readlink -f "$0")"); out=$1; shift; per=50
[ "${1:-}" != -- ] && { per=$1; shift; }; [ "${1:-}" = -- ] && shift
"$@" & P=$!
python3 "$T/dram-sample.py" "$out" $P "$per" 9>&- & D=$!
wait $P; rc=$?; wait $D
echo "$(python3 "$T/dram-summary.py" "$out" --line 2>&1 | tail -1) ($out)" >&2
exit $rc

#!/bin/bash
# chan-run.sh <out.tsv> [period_ms=100] -- <cmd> [args]: run <cmd> with the gcea-sample per-channel sampler beside it (device-wide
# per-instance GCEA DRAM read / write requests via rocprofiler-sdk device counting, no root), then print the chan-summary.py --line
# to stderr; exits with <cmd>'s rc. Coverage: HIP / KFD work of any process; Vulkan work may read zero depending on the driver
# (verify with camp-hip before trusting a zero). Same calling style as dram-run.sh; takes no lock (run it inside the caller's lock).
# Without ROCm / KFD access the command still runs and "chan: unavailable (...)" is printed. Env: see chan-start.sh. Method: DRAM-MONITOR.md.
case ${1:-} in ''|-h|--help) echo "usage: chan-run.sh <out.tsv> [period_ms=100] -- <cmd> [args]" >&2; exit 2;; esac
T=$(dirname "$(readlink -f "$0")"); out=$1; shift; per=100
[ "${1:-}" != -- ] && { per=$1; shift; }; [ "${1:-}" = -- ] && shift
D=$("$T/chan-start.sh" "$out" "$per")
"$@"; rc=$?
echo "$("$T/chan-stop.sh" "$D" "$out") ($out)" >&2
exit $rc

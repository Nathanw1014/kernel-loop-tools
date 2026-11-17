#!/bin/bash
# needs: per-family fusion gates in ggml-vulkan (e.g. GGML_VK_DISABLE_FUSIONS=<comma-separated fusion names>; upstream only has the all-or-nothing GGML_VK_DISABLE_FUSION) and a llama-eval-callback patch that reads EVCB_ASK_NONE=1 (ask for no node, whole graph) and EVCB_LOGITS_DUMP=<file> (write f32 logits of every prompt position, n_tokens x n_vocab).
# fusion-family-bisect.sh <base-bin> <tip-bin> <families-file> <model.gguf> <prompt.txt> <unfused-ref.bin> <n_vocab> [out-dir]
#
# Per-family fusion bisect: disable ONE fused family at a time on BOTH builds, run the whole prompt as one graph with all
# logits, and compare base vs tip argmax agreement and each against an unfused reference. The family whose removal restores
# base == tip parity carries the fused difference between the two builds.
#   base-bin / tip-bin   llama-eval-callback binaries of the two builds (both with the patches above)
#   families-file        one family per line: <label> VAR=VALUE [VAR=VALUE ...]; '#' comments. Include a control line whose
#                        env changes nothing (e.g. "none KLT_NOOP=1"), so you see the unchanged disagreement first. Example:
#                          none          KLT_NOOP=1
#                          rms_norm_mul  GGML_VK_DISABLE_FUSIONS=RMS_NORM_MUL
#                          multi_add     GGML_VK_DISABLE_FUSIONS=MULTI_ADD
#                          glu           GGML_VK_DISABLE_FUSIONS=GLU
#   unfused-ref.bin      logits dump of the same prompt with every fusion off (upstream GGML_VK_DISABLE_FUSION=1); check it
#                        is bit-equal on both builds before trusting it
#   out-dir              default $KLT_RESULTS/fusion-family-bisect-<UTC stamp>
# env: FFB_ARGS llama args after -m / -f (default "-ngl 99 -fa on -c 512 -b 512 -ub 512 -n 1"; -b / -ub must cover the prompt),
#      FFB_NO_LOCK=1 when the caller already holds $KLT_LOCK. Driver env (VK_ICD_FILENAMES etc.) passes through.
set -u
if [ $# -lt 7 ] || [ "${1:-}" = -h ] || [ "${1:-}" = --help ]; then awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2; fi
. "$(dirname "$(readlink -f "$0")")/../lib/klt-env.sh"
BASE=$1 TIP=$2 FAM=$3 M=$4 P=$5 REF=$6 V=$7
D=${8:-$KLT_RESULTS/fusion-family-bisect-$(date -u +%Y%m%dT%H%M%SZ)}
for f in "$BASE" "$TIP"; do [ -x "$f" ] || { echo "not executable: $f" >&2; exit 2; }; done
for f in "$FAM" "$M" "$P" "$REF"; do [ -r "$f" ] || { echo "not readable: $f" >&2; exit 2; }; done
[ -e "$KLT_STOP" ] && { echo "refusing: $KLT_STOP exists" >&2; exit 1; }
mkdir -p "$D/fam" || exit 1
read -r -a ARGS <<< "${FFB_ARGS:--ngl 99 -fa on -c 512 -b 512 -ub 512 -n 1}"

run_pair () { # $1 label, $2.. env assignments
    local label=$1; shift
    [ -n "${FFB_NO_LOCK:-}" ] || echo "pid=$$ label=fusion-family-$label started=$(date -Is)" > "$KLT_LOCK.owner"
    env "$@" EVCB_ASK_NONE=1 EVCB_LOGITS_DUMP="$D/fam/base-$label.bin" \
        "$BASE" -m "$M" -f "$P" "${ARGS[@]}" > "$D/fam-$label-base.log" 2>&1 < /dev/null
    env "$@" EVCB_ASK_NONE=1 EVCB_LOGITS_DUMP="$D/fam/tip-$label.bin" \
        "$TIP" -m "$M" -f "$P" "${ARGS[@]}" > "$D/fam-$label-tip.log" 2>&1 < /dev/null
    python3 - "$label" "$D/fam" "$REF" "$V" << 'EOF'
import numpy as np, sys
lab, F_DIR, REF, V = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
try:
    F = np.fromfile(f'{F_DIR}/base-{lab}.bin', dtype=np.float32).reshape(-1, V)
    U = np.fromfile(f'{F_DIR}/tip-{lab}.bin', dtype=np.float32).reshape(-1, V)
except Exception as e:
    print(f'FAMILY-OFF {lab}: dump missing ({e})'); sys.exit(0)
R = np.fromfile(REF, dtype=np.float32).reshape(-1, V)
print(f'FAMILY-OFF {lab:14s} base vs tip: {(F.argmax(1)==U.argmax(1)).mean()*100:7.3f}% max|d| {np.abs(F-U).max():7.4f} | '
      f'base vs unfused {(F.argmax(1)==R.argmax(1)).mean()*100:7.3f}% | tip vs unfused {(U.argmax(1)==R.argmax(1)).mean()*100:7.3f}%', flush=True)
EOF
}

run_all () {
    local label rest
    while read -r label rest; do
        case $label in ''|'#'*) continue;; esac
        read -r -a envs <<< "$rest"
        run_pair "$label" "${envs[@]}"
    done < "$FAM"
}

if [ -n "${FFB_NO_LOCK:-}" ]; then
    run_all
else
    ( exec 9>"$KLT_LOCK"; flock 9; trap 'rm -f "$KLT_LOCK.owner"' EXIT; run_all )
fi
echo "FAMILY-BISECT-DONE ($D)"

#!/bin/bash
# decode.sh <extracted-dir>: run rocprof-trace-decoder on rgpwaves.py output, inside docker.
# Writes <extracted-dir>/decoded/ (occupancy.json, realtime.json, code.json, ...) and decode.log.
# Env:
#   DECODE_IMAGE  docker image that ships the decoder library
#                 (default rocm/dev-ubuntu-24.04:10.1.0-full; any ROCm image with
#                 librocprof-trace-decoder.so works: re-check a known capture when you change it)
#   DECODE_LIB    decoder library path inside the image (default /opt/rocm/lib/librocprof-trace-decoder.so)
#   DECODE_MEM    container memory cap (default 24g); decode RAM scales with capture size
# The vendored decode.py and Python decoder package are in third_party/ (MIT, see LICENSE, PROVENANCE).
set -euo pipefail
[ $# -eq 1 ] || { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2; }
D=$(realpath "$1"); H=$(dirname "$(realpath "$0")")
rm -rf "$D/decoded"
docker run --rm --memory "${DECODE_MEM:-24g}" --memory-swap "${DECODE_MEM:-24g}" -u "$(id -u):$(id -g)" -v "$D:$D" -v "$H/third_party:/tp:ro" -w "$D" \
  "${DECODE_IMAGE:-rocm/dev-ubuntu-24.04:10.1.0-full}" \
  python3 /tp/decode.py "$D" "$D/decoded" --python-package /tp/decoder-python.zip \
    --lib "${DECODE_LIB:-/opt/rocm/lib/librocprof-trace-decoder.so}" 2>&1 | tee "$D/decode.log"

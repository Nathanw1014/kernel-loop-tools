#!/bin/bash
# percommit.sh <worktree> <builddir> <commit-list-file> <out-tsv>
# Checks out each commit in turn (detached) and builds test-backend-ops incrementally; one line per commit:
#   sha rc seconds first-error
# Configure flags: CMAKE_FLAGS (default: Release, GGML_VULKAN=ON, GGML_NATIVE=OFF, LLAMA_BUILD_TESTS=ON) plus
# CMAKE_EXTRA (e.g. -DVulkan_GLSLC_EXECUTABLE=/path/to/glslc -DSPIRV-Headers_DIR=/path/to/cmake/SPIRV-Headers).
# Base and candidate must use the same glslc. BUILD_JOBS (default nproc).
# The build does not take the GPU lock: on a machine shared with benches run it under the lock (flock "$KLT_LOCK" ...).
case ${1:-} in -h|--help|'') sed -n '2,8p' "$0"; exit 2;; esac
[ $# -eq 4 ] || { sed -n '2,8p' "$0"; exit 2; }
wt=$1; bd=$2; list=$3; out=$4; : > "$out"
FL=${CMAKE_FLAGS:--DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DLLAMA_BUILD_TESTS=ON}
git -C "$wt" checkout -q --detach "$(head -n1 "$list")"
cmake -S "$wt" -B "$bd" $FL ${CMAKE_EXTRA:-} > /dev/null 2>&1
while read c; do
  git -C "$wt" checkout -q --detach "$c" || { echo -e "$c\tCHECKOUT-FAIL" >> "$out"; continue; }
  t0=$(date +%s)
  nice -n 10 cmake --build "$bd" -j "${BUILD_JOBS:-$(nproc)}" --target test-backend-ops > "$bd/last.log" 2>&1; rc=$?
  err=$(grep -m1 -E "error:|Error " "$bd/last.log" | cut -c1-200)
  echo -e "$c\t$rc\t$(( $(date +%s)-t0 ))\t$err" >> "$out"
done < "$list"
echo "percommit done: $(grep -c -P '\t0\t' "$out") ok of $(wc -l < "$out")"

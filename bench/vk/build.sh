#!/usr/bin/env bash
# build.sh [out_dir]: build the plain-Vulkan kit (vkrun, vkgemm, vkasm, isa-pipe, vkq) and its shaders (wmma_peak*.spv,
# spin.spv). Needs a C++17 compiler, the Vulkan headers + loader (libvulkan-dev) and glslc with GL_KHR_cooperative_matrix
# (set GLSLC=; default: glslc on PATH). Output defaults to bench/vk/bin. CPU only: nothing here touches the GPU.
set -euo pipefail
case ${1:-} in -h|--help) sed -n '2,4p' "$0" | sed 's/^# \{0,1\}//'; exit 0;; esac
HERE=$(dirname "$(readlink -f "$0")")
OUT=${1:-$HERE/bin}; GLSLC=${GLSLC:-glslc}; CXX=${CXX:-g++}
mkdir -p "$OUT"
for t in vkrun vkgemm vkasm isa-pipe; do
  "$CXX" -O2 -std=c++17 -Wall -o "$OUT/$t" "$HERE/$t.cpp" -lvulkan
done
"$CXX" -O2 -std=c++17 -Wall -pthread -o "$OUT/vkq" "$HERE/vkq.cpp" -lvulkan
for s in wmma_peak wmma_peak_i8 spin; do
  "$GLSLC" -O --target-env=vulkan1.3 "$HERE/$s.comp" -o "$OUT/$s.spv"
done
echo "built: $(cd "$OUT" && ls | tr '\n' ' ')"

#!/usr/bin/env bash
# shader-diff.sh <build-A> <build-B>: did a change alter the generated Vulkan shaders? Compares two llama.cpp build dirs:
#   1. the SPIR-V the shader generator wrote (every *.spv under ggml/src/ggml-vulkan/, matched by path), and
#   2. the generated embeds (ggml/src/ggml-vulkan/*.comp.cpp and ggml-vulkan-shaders.hpp) that are compiled into the library.
# Prints "changed <path>", "new <path>" (only in B), "gone <path>" (only in A), then a count line.
# Exit: 0 identical, 1 differences, 2 usage / no generated shaders found in a dir (wrong path, or not a Vulkan build).
# Use: a host-side-only change must show 0 changed shaders (if not, the A/B is not measuring what you think); a shader
# change must show exactly the files you meant to touch. Both builds must use the same glslc, or everything differs.
# Read-only; no GPU.
set -u
case ${1:-} in -h|--help) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; exit 0;; esac
[ $# -eq 2 ] || { echo "usage: shader-diff.sh <build-A> <build-B>" >&2; exit 2; }
A=$(readlink -f "$1")/ggml/src/ggml-vulkan; B=$(readlink -f "$2")/ggml/src/ggml-vulkan
list() { (cd "$1" && find . \( -name '*.spv' -o -name '*.comp.cpp' -o -name 'ggml-vulkan-shaders.hpp' \) -type f | sort); }
LA=$(list "$A" 2>/dev/null); LB=$(list "$B" 2>/dev/null)
[ -n "$LA" ] || { echo "no generated shaders under $A" >&2; exit 2; }
[ -n "$LB" ] || { echo "no generated shaders under $B" >&2; exit 2; }
n=0; d=0
while IFS= read -r f; do
  if [ -e "$B/$f" ]; then
    n=$((n+1)); cmp -s "$A/$f" "$B/$f" || { d=$((d+1)); echo "changed ${f#./}"; }
  fi
done <<< "$LA"
new=$(comm -13 <(echo "$LA") <(echo "$LB")); gone=$(comm -23 <(echo "$LA") <(echo "$LB"))
[ -n "$new" ] && echo "$new" | sed 's|^\./|new |'
[ -n "$gone" ] && echo "$gone" | sed 's|^\./|gone |'
nn=$(printf '%s' "$new" | grep -c . || true); ng=$(printf '%s' "$gone" | grep -c . || true)
echo "compared $n generated files: $d changed, $nn only in B, $ng only in A"
[ "$d" -eq 0 ] && [ "$nn" -eq 0 ] && [ "$ng" -eq 0 ]

#!/bin/bash
# build-dram-mm.sh <ggml source tree> <build dir with bin/libggml*.so> [out=<build dir>/bin/dram-mm]: compile dram-mm.cpp against
# an existing llama.cpp / ggml build (rpath to its bin/). Host C++ only, seconds; a compile perturbs a timed GPU job on the same
# machine, so run it under your GPU lock if one may be timing.
set -eu
[ $# -ge 2 ] || { echo "usage: build-dram-mm.sh <ggml source tree> <build dir> [out]" >&2; exit 2; }
T=$1; B=$2; O=${3:-$B/bin/dram-mm}
g++ -O2 -std=c++17 -I"$T/ggml/include" "$(dirname "$(readlink -f "$0")")/dram-mm.cpp" -o "$O" -L"$B/bin" -lggml -lggml-base -Wl,-rpath,"$B/bin"
echo "built $O ($(git -C "$T" describe --always --dirty 2>/dev/null))"

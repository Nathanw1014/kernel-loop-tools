#!/bin/bash
# build-gcea-sample.sh [out_dir=<this dir>/bin] [rocm_root=$ROCM_PATH or /opt/rocm]: build gcea-sample.cpp against ROCm
# (rocprofiler-sdk + hsa-runtime). Host C++ only, no GPU code; chan-start.sh looks for <this dir>/bin/gcea-sample by default.
# A compile perturbs a timed GPU job on the same machine: run it under your GPU lock if one may be timing.
set -e
case ${1:-} in -h|--help) echo "usage: build-gcea-sample.sh [out_dir] [rocm_root]" >&2; exit 2;; esac
S=$(dirname "$(readlink -f "$0")")
O=${1:-$S/bin}
R=${2:-${ROCM_PATH:-/opt/rocm}}
[ -e "$R/include/rocprofiler-sdk/rocprofiler.h" ] || { echo "no rocprofiler-sdk headers under $R (pass rocm_root or set ROCM_PATH)" >&2; exit 1; }
mkdir -p "$O"
g++ -O2 -std=c++17 -Wall -D__HIP_PLATFORM_AMD__ -I"$R/include" "$S/gcea-sample.cpp" -o "$O/gcea-sample" \
    -L"$R/lib" -lrocprofiler-sdk -lhsa-runtime64 -Wl,-rpath,"$R/lib" -Wl,--export-dynamic -lpthread
echo "built $O/gcea-sample"

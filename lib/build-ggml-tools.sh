#!/usr/bin/env bash
# build-ggml-tools.sh <llama.cpp source> <build dir> [out_dir]: build the tools that link ggml / llama against an existing
# llama.cpp build (shared libs, the default BUILD_SHARED_LIBS=ON): graph/graphdump, graph/topkdump,
# correctness/fa_masked_leak, correctness/topk_repeat. Output defaults to <build dir>/bin, next to the libraries, so the
# binaries find them through their rpath ($ORIGIN). Build against the same tree you test: a probe linked to another
# build's libggml tests that build, not yours. CPU only; takes no GPU lock (a build is not a GPU job).
set -euo pipefail
case ${1:-} in -h|--help) sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'; exit 0;; esac
[ $# -ge 2 ] || { sed -n '2p' "$0" | sed 's/^# //' >&2; exit 2; }
SRC=$(readlink -f "$1"); BLD=$(readlink -f "$2"); OUT=${3:-$BLD/bin}
ROOT=$(dirname "$(dirname "$(readlink -f "$0")")")
CXX=${CXX:-g++}
[ -f "$SRC/include/llama.h" ] && [ -f "$SRC/ggml/include/ggml.h" ] || { echo "not a llama.cpp source tree: $SRC" >&2; exit 2; }
LIB=$BLD/bin; [ -e "$LIB/libllama.so" ] || LIB=$(dirname "$(find "$BLD" -name libllama.so -print -quit)")
[ -e "$LIB/libllama.so" ] || { echo "no libllama.so under $BLD (shared build needed)" >&2; exit 2; }
COMMON=$(find "$LIB" "$BLD" -maxdepth 2 \( -name 'libllama-common.so' -o -name 'libcommon.a' -o -name 'libllama-common.a' \) -print -quit)
mkdir -p "$OUT"
INC=(-I"$SRC/include" -I"$SRC/ggml/include" -I"$SRC/common" -I"$SRC/vendor")
LNK=(-L"$LIB" -Wl,-rpath,"\$ORIGIN" -Wl,-rpath,"$LIB" -lllama -lggml -lggml-base)
set -x
"$CXX" -O2 -std=c++17 "${INC[@]}" -o "$OUT/graphdump" "$ROOT/graph/graphdump.cpp" "${LNK[@]}"
"$CXX" -O2 -std=c++17 "${INC[@]}" -o "$OUT/fa_masked_leak" "$ROOT/correctness/fa_masked_leak.cpp" "${LNK[@]}"
"$CXX" -O2 -std=c++17 "${INC[@]}" -o "$OUT/topk_repeat" "$ROOT/correctness/topk_repeat.cpp" "${LNK[@]}"
{ set +x; } 2>/dev/null
if [ -n "$COMMON" ]; then
  set -x
  "$CXX" -O2 -std=c++17 "${INC[@]}" -o "$OUT/topkdump" "$ROOT/graph/topkdump.cpp" "$COMMON" "${LNK[@]}" -pthread
  { set +x; } 2>/dev/null
else
  echo "no common library (libllama-common / libcommon) in $BLD: topkdump not built" >&2
fi
echo "built into $OUT: graphdump fa_masked_leak topk_repeat${COMMON:+ topkdump}"

#!/usr/bin/env bash
# batch-validate.sh: validate a whole batch of env-gated features in one GPU pass, base vs tip.
#
#   batch-validate.sh <out-dir> <base-bin-dir> <tip-bin-dir> <knobs-file> <model> <corpus> [model flags...]
#
# <knobs-file>: one opt-out per line, "NAME=VALUE [NAME=VALUE ...]  # comment" (each line is one feature off);
#   see correctness/knobs.example.
# <model>, <corpus>: the model for the perplexity / bench stages and a raw text file for llama-perplexity.
# [model flags...]: extra flags for llama-perplexity and llama-bench (offload / load-mode flags for this model); none by default.
#
# Stages (one lock per cell, refuses while $KLT_STOP exists, stops on DeviceLost / OOM / BO_VA and writes $KLT_STOP):
#   1  test-backend-ops full suite on base and tip -> failing-SET diff (new / fixed / pre-existing)
#   2  knob isolation: only if stage 1 found new failures, rerun just those op families with each knob alone
#   3  llama-perplexity c4096 x4: base, tip, and tip with each knob alone (rounding-level check)
#   4  ABBA llama-bench pp2048 ub2048 d0,16384 base vs tip
#   5  perf-logger run on the tip (GGML_VK_PERF_LOGGER): op counts / time per 2048-token ubatch (engagement proof)
# Env: BACKEND (test-backend-ops -b, default Vulkan0), MEM_GB (MemAvailable before each model cell, default 0 = off:
# set it to the model's resident size plus a few GB), SUITE_MEM_GB (same for op-test cells, default 0), GTT_MAX_GIB
# (default 2), SETTLE_S (sleep after the memory gate, default 30), LOAD_MAX (loadavg gate, default 4).
# SUITE_ARGS: extra test-backend-ops args for stage 1, e.g. "-o MUL_MAT,SOFT_MAX" to limit it to the ops the batch touches
# (default: the full suite).
# Results: <out-dir>/BATCH.txt; raw logs next to it.
set -uo pipefail
case ${1:-} in -h|--help|'') sed -n '2,20p' "$0"; exit 2;; esac
[ $# -ge 6 ] || { echo "need <out-dir> <base-bin> <tip-bin> <knobs-file> <model> <corpus> [model flags...]" >&2; exit 2; }
OUT_DIR=$1; BASE=$2; TIP=$3; KNOBS=$4; MODEL=$5; W=$6; shift 6
MFLAGS=("$@")
[ -r "$KNOBS" ] && [ -r "$MODEL" ] && [ -r "$W" ] || { echo "knobs file, model or corpus not readable" >&2; exit 2; }
. "$(dirname "$(readlink -f "$0")")/../lib/klt-env.sh"
[ -e "$KLT_STOP" ] && { echo "$KLT_STOP exists (device error recorded); refusing to run" >&2; exit 70; }
# KLT_VK_ICD pins the Vulkan ICD (KLT_VK_LIBPATH: extra library dirs for it, e.g. a matching libdrm), as in abba.sh / job-run.
[ -n "${KLT_VK_ICD:-}" ] && { [ -f "$KLT_VK_ICD" ] || { echo "KLT_VK_ICD missing: $KLT_VK_ICD" >&2; exit 78; }; export VK_ICD_FILENAMES=$KLT_VK_ICD VK_DRIVER_FILES=$KLT_VK_ICD; }
[ -n "${KLT_VK_LIBPATH:-}" ] && export LD_LIBRARY_PATH=$KLT_VK_LIBPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
GPU=$(klt_gpu_card) || { echo "no amdgpu card (set KLT_GPU_CARD)" >&2; exit 2; }
GTT=$GPU/mem_info_gtt_used; BK=${BACKEND:-Vulkan0}; MEM=${MEM_GB:-0}; SMEM=${SUITE_MEM_GB:-0}
mkdir -p $OUT_DIR; OUT=$OUT_DIR/BATCH.txt
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a $OUT; }
quiet(){ ! pgrep -x 'llama-quantize|llama-imatrix' >/dev/null && awk -v m="${LOAD_MAX:-4}" '{exit !($1 < m)}' /proc/loadavg; }
ready(){ until quiet && [ $(( $(cat $GTT) / 1048576 )) -le $(( ${GTT_MAX_GIB:-2} * 1024 )) ] && [ $(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo) -ge ${1:-0} ]; do sleep 10; done; sleep ${SETTLE_S:-30}; }
stopall(){ say "STOP $*"; echo "$(date -u +%FT%TZ) batch-validate $OUT_DIR: $*" > "$KLT_STOP"; exit 70; }
fatal(){ grep -aE 'DeviceLost|Not enough memory for command|BO_VA' "$1" | grep -qv 'pinned memory' && stopall "device error in $1"
         journalctl -k --since "-10 min" --no-pager 2>/dev/null | grep -q 'BO_VA' && stopall "kernel BO_VA after $1"; return 0; }
cell(){ local lab=$1 mem=$2; shift 2; [ -e "$KLT_STOP" ] && { say "STOP file present, not starting $lab"; exit 70; }
  ( exec 9>"$KLT_LOCK"; flock 9; echo "pid=$$ label=batch-$lab started=$(date -Is)" > "$KLT_LOCK.owner"; trap 'rm -f "$KLT_LOCK.owner"' EXIT; ready $mem; "$@" ) || exit $?; }
failset(){ sed -E 's/\x1b\[[0-9;]*m//g' "$1" | python3 -c "
import sys,re
fails=set()
for l in sys.stdin:
    m=re.search(r'([A-Z_0-9]+\(.*\)):\s*FAIL\b',l)   # the case name and FAIL are on one line (prefixes like '[CONCAT] ERR = ...' allowed)
    if m: fails.add(m.group(1))
print('\n'.join(sorted(fails)))"; }
knob_lines(){ grep -vE '^\s*(#|$)' $KNOBS | sed 's/#.*//'; }

say "=== batch: base $BASE  tip $TIP  knobs $(knob_lines | wc -l)"
# ---- 1 op suites
suite(){ env "$@" LD_LIBRARY_PATH=$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} timeout 7200 $BIN/test-backend-ops test -b $BK ${SUITE_ARGS:-} 9>&- > $OUT_DIR/suite-$TAG.log 2>&1; fatal $OUT_DIR/suite-$TAG.log
  say "suite $TAG: $(sed -E 's/\x1b\[[0-9;]*m//g' $OUT_DIR/suite-$TAG.log | grep -E 'tests passed' | tail -1)"; failset $OUT_DIR/suite-$TAG.log > $OUT_DIR/fails-$TAG.txt; }
BIN=$BASE TAG=base cell suite-base $SMEM suite X=1
BIN=$TIP  TAG=tip  cell suite-tip  $SMEM suite X=1
comm -13 $OUT_DIR/fails-base.txt $OUT_DIR/fails-tip.txt > $OUT_DIR/fails-new.txt
comm -23 $OUT_DIR/fails-base.txt $OUT_DIR/fails-tip.txt > $OUT_DIR/fails-fixed.txt
comm -12 $OUT_DIR/fails-base.txt $OUT_DIR/fails-tip.txt > $OUT_DIR/fails-pre.txt
say "suite diff: NEW $(wc -l < $OUT_DIR/fails-new.txt) | fixed $(wc -l < $OUT_DIR/fails-fixed.txt) | pre-existing $(wc -l < $OUT_DIR/fails-pre.txt)"
sed 's/^/  NEW: /' $OUT_DIR/fails-new.txt | head -20 | tee -a $OUT

# ---- 2 knob isolation of new failures
if [ -s $OUT_DIR/fails-new.txt ]; then
  ops=$(sed -E 's/\(.*//' $OUT_DIR/fails-new.txt | sort -u | paste -sd'|')
  iso(){ local i=0; while read -r k; do i=$((i+1))
      : > $OUT_DIR/iso-$i.log
      for op in ${ops//|/ }; do env $k LD_LIBRARY_PATH=$TIP${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} timeout 3600 $TIP/test-backend-ops test -b $BK -o "$op" 9>&- >> $OUT_DIR/iso-$i.log 2>&1; done
      fatal $OUT_DIR/iso-$i.log
      failset $OUT_DIR/iso-$i.log > $OUT_DIR/iso-$i.fails
      local left=$(comm -12 $OUT_DIR/fails-new.txt $OUT_DIR/iso-$i.fails | wc -l)
      say "  isolate [$k]: new failures left $left / $(wc -l < $OUT_DIR/fails-new.txt) $( [ $left -eq 0 ] && echo '<- CULPRIT (knob clears them)')"
    done < <(knob_lines); }
  say "=== 2 knob isolation on ops: $ops"
  cell iso $SMEM iso
fi

# ---- 3 perplexity: base, tip, tip with each knob alone
ppl(){ env "$@" LD_LIBRARY_PATH=$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} $BIN/llama-perplexity -m $MODEL -f $W -c 4096 -b 2048 -ub 2048 --chunks 4 -ngl 999 -fa on "${MFLAGS[@]}" 9>&- > $OUT_DIR/ppl-$TAG.log 2>&1; fatal $OUT_DIR/ppl-$TAG.log
  say "ppl $TAG [$*]: $(grep -aoE 'Final estimate: PPL = [0-9.]+ \+/- [0-9.]+' $OUT_DIR/ppl-$TAG.log) | $(grep -aoE '\[[0-9]+\][0-9.]+' $OUT_DIR/ppl-$TAG.log | tr '\n' ' ')"; }
say "=== 3 perplexity"
BIN=$BASE TAG=base cell ppl-base $MEM ppl X=1
BIN=$TIP  TAG=tip  cell ppl-tip  $MEM ppl X=1
i=0; while read -r k; do i=$((i+1)); BIN=$TIP TAG=knob$i cell ppl-k$i $MEM ppl $k; done < <(knob_lines)

# ---- 4 ABBA perf
bench(){ LD_LIBRARY_PATH=$BIN${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} $BIN/llama-bench -m $MODEL -ngl 99 -fa 1 "${MFLAGS[@]}" -p 2048 -n 0 -b 2048 -ub 2048 -d 0,16384 -r 2 -o jsonl 9>&- > $OUT_DIR/bench-$TAG.jsonl 2> $OUT_DIR/bench-$TAG.err; fatal $OUT_DIR/bench-$TAG.err
  say "bench $TAG: $(python3 -c "
import json
rows=[json.loads(l) for l in open('$OUT_DIR/bench-$TAG.jsonl') if l.startswith('{')]
print(' '.join(f\"d{r['n_depth']}:{r['avg_ts']:.1f}+-{r['stddev_ts']:.1f}\" for r in rows))")"; }
say "=== 4 ABBA perf"
n=0; for arm in base tip tip base; do n=$((n+1)); BIN=$BASE; [ $arm = tip ] && BIN=$TIP; TAG=$arm-$n cell bench-$n $MEM bench; done

# ---- 5 engagement: perf logger op counts on the tip (last ubatch)
eng(){ GGML_VK_PERF_LOGGER=1 LD_LIBRARY_PATH=$TIP${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} $TIP/llama-bench -m $MODEL -ngl 99 -fa 1 "${MFLAGS[@]}" -p 2048 -n 0 -b 2048 -ub 2048 -d 0 -r 1 9>&- > $OUT_DIR/engage.log 2>&1; fatal $OUT_DIR/engage.log
  python3 - $OUT_DIR/engage.log <<'PY' | tee -a $OUT
import re,sys
blocks=[];cur=None
for l in open(sys.argv[1],errors='ignore'):
    if l.startswith('Vulkan Timings'): cur=[]
    elif l.startswith('Total time') and cur is not None: blocks.append(cur); cur=None
    elif cur is not None:
        m=re.match(r'(.*?): (\d+) x ([\d.]+) us = ([\d.e+]+) us',l)
        if m: cur.append((m.group(1).split()[0],int(m.group(2)),float(m.group(4))))
agg={}
for name,k,us in (blocks[-1] if blocks else []):
    a=agg.setdefault(name,[0,0.0]); a[0]+=k; a[1]+=us
for name,(k,us) in sorted(agg.items(),key=lambda x:-x[1][1])[:30]:
    print(f"  engage {name:28s} {k:5d}x {us/1e3:8.1f} ms")
PY
}
say "=== 5 engagement (tip, perf logger, top ops by time)"
cell engage $MEM eng
say "BATCH-DONE"

#!/usr/bin/env bash
# abba.sh: counterbalanced ABBA llama-bench harness, one process per cell.
#
#   abba.sh [--dry-run] <out-dir> <preset-file> <label=bin-dir[:ENV=V,ENV=V]> <label=bin-dir[:...]> [more arms]
#
# <preset-file>: a bash file defining MODEL, ARGS (llama-bench flags, array) and MEM_GB; see ab/presets.example.
# <bin-dir>: directory holding llama-bench. ENV=V pairs after ':' are set for that arm only.
#
#   ROUNDS=N (default 2): each round runs the arms forward then reversed (A B .. B A), one llama-bench process per cell,
#   after one discarded warm-up launch of the first arm. 2 rounds = 4 independent launches per arm.
#   ROUNDS=1 = two launches per arm (forward, then reversed).
#   PP / DEPTHS / NGEN / REPS: -p / -d / -n / -r per launch (defaults 2048 / 0,8192,32768 / 32 / 2; env wins over the preset).
#
# Per cell: refuse while $KLT_STOP exists; flock $KLT_LOCK + owner line; at least ABBA_GAP_S (120) since the previous cell
# exited; samplers/cooldown.sh; then GTT used <= ABBA_GTT_MAX_GIB (2), MemAvailable >= MEM_GB, loadavg < ABBA_LOAD_MAX (3)
# and none of ABBA_BUSY_PROCS running must hold for ABBA_STABLE_S (60) straight. The readings go into the cell line.
# A device error (DeviceLost / out of device memory / amdgpu BO_VA, ring timeout or GPU reset in the kernel log) writes
# $KLT_STOP and stops everything; the watchdog kills only our own pid. Cell timeout ABBA_CELL_TIMEOUT (5400 s).
# KLT_VK_ICD (optional) pins the Vulkan ICD for every arm; its driverInfo is logged once.
# Records: <out>/bench-<i>-<arm>.jsonl + .err verbatim, .therm (samplers/therm-sample.sh), .flt (samplers/flt-sample.sh,
# major/minor faults of the launch), <out>/RESULTS.txt, <out>/ARMS.txt (bin, source tree, commit, env per arm).
# Follow-ups: ab/abba-outliers.py <out> (rep-level outliers), ab/abba-postscan.py <out> (noisy records, page-fault cells).
set -uo pipefail
usage() { sed -n '2,22p' "$0"; exit 2; }
DRY=0; [ "${1:-}" = --dry-run ] && { DRY=1; shift; }
case ${1:-} in -h|--help|'') usage;; esac
OUT=${1:?out-dir}; PRESET=${2:?preset-file}; shift 2; [ $# -ge 2 ] || { echo "need >= 2 arms label=bin[:ENV=V,...]"; exit 2; }
[ -r "$PRESET" ] || { echo "preset file not readable: $PRESET (see ab/presets.example)"; exit 2; }
HERE=$(dirname "$(readlink -f "$0")"); TH=$HERE/../samplers
. "$HERE/../lib/klt-env.sh"
_PP=${PP:-}; _DEPTHS=${DEPTHS:-}; _NGEN=${NGEN:-}; _REPS=${REPS:-}
MODEL=""; ARGS=(); MEM_GB=""
. "$PRESET"
[ -n "$MODEL" ] || { echo "preset $PRESET does not set MODEL"; exit 2; }
[ -n "$MEM_GB" ] || { echo "preset $PRESET does not set MEM_GB"; exit 2; }
[ "$DRY" = 1 ] || [ -r "$MODEL" ] || { echo "model not readable: $MODEL"; exit 2; }
PP=${_PP:-${PP:-2048}}; DEPTHS=${_DEPTHS:-${DEPTHS:-0,8192,32768}}; NGEN=${_NGEN:-${NGEN:-32}}; REPS=${_REPS:-${REPS:-2}}
ROUNDS=${ROUNDS:-2}; PNAME=$(basename "$PRESET"); PNAME=${PNAME%.*}
GAP_S=${ABBA_GAP_S:-120}; STABLE_S=${ABBA_STABLE_S:-60}; GTT_MAX=${ABBA_GTT_MAX_GIB:-2}; LOAD_MAX=${ABBA_LOAD_MAX:-3}
BUSY_PROCS=${ABBA_BUSY_PROCS:-llama-quantize|llama-imatrix}; CELL_TO=${ABBA_CELL_TIMEOUT:-5400}
CARD=$(klt_gpu_card) || { [ $DRY = 1 ] || { echo "no amdgpu card found (set KLT_GPU_CARD=/sys/class/drm/cardN/device)"; exit 2; }; }
GTT=$CARD/mem_info_gtt_used
ICDENV=(); [ -n "$KLT_VK_ICD" ] && ICDENV=(VK_ICD_FILENAMES=$KLT_VK_ICD VK_DRIVER_FILES=$KLT_VK_ICD)
[ -n "${KLT_VK_LIBPATH:-}" ] && export LD_LIBRARY_PATH=$KLT_VK_LIBPATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}   # e.g. the ICD's libdrm
mkdir -p "$OUT"; RES=$OUT/RESULTS.txt
say(){ echo "$(date -u +%FT%TZ) $*" | tee -a "$RES"; }
declare -A BIN ENVS; ORDER=()
for a in "$@"; do lab=${a%%=*}; rest=${a#*=}; b=${rest%%:*}; e=""; [ "$rest" != "$b" ] && e=${rest#*:}
  [ -x "$b/llama-bench" ] || { echo "no llama-bench in $b"; exit 2; }
  BIN[$lab]=$b; ENVS[$lab]=${e//,/ }; ORDER+=("$lab"); done
{ echo "preset=$PRESET model=$MODEL flags=${ARGS[*]} -p $PP -n $NGEN -d $DEPTHS -r $REPS rounds=$ROUNDS mem_gb=$MEM_GB icd=${KLT_VK_ICD:-system} platform_profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null) boot=$(uptime -s)"
  for lab in "${ORDER[@]}"; do b=${BIN[$lab]}; src=$(grep -m1 CMAKE_HOME_DIRECTORY "${b%/bin}/CMakeCache.txt" 2>/dev/null | cut -d= -f2)
    c=$(git -C "${src:-/}" rev-parse --short HEAD 2>/dev/null || cat "$b/COMMIT" "$b/../COMMIT" 2>/dev/null | head -1 || echo ?)
    echo "arm $lab bin=$b src=${src:-?} commit=${c:-?} dirty=$(git -C "${src:-/}" status --short 2>/dev/null | wc -l) env=[${ENVS[$lab]}]"; done
} | tee "$OUT/ARMS.txt"
[ -z "$KLT_VK_ICD" ] || [ -f "$KLT_VK_ICD" ] || { echo "KLT_VK_ICD missing: $KLT_VK_ICD"; exit 78; }
compgen -G '/dev/dri/renderD*' >/dev/null || { echo "ABORT: no /dev/dri render node (sandboxed shell?)"; exit 1; }

quiet(){ ! pgrep -x "$BUSY_PROCS" >/dev/null && awk -v m="$LOAD_MAX" '{exit !($1 < m)}' /proc/loadavg; }
LAST_EXIT=$OUT/.last-exit
settle(){ local last; last=$(cat "$LAST_EXIT" 2>/dev/null || echo 0); local now; now=$(date +%s)
  [ $((now-last)) -lt "$GAP_S" ] && sleep $((GAP_S-now+last))
  local cool; cool=$("$TH/cooldown.sh")   # never start a cell on a hot package
  local ok=0 g a; while :; do g=$(( $(cat "$GTT") / 1073741824 )); a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if quiet && [ $g -le "$GTT_MAX" ] && [ $a -ge "$MEM_GB" ]; then ok=$((ok+5)); [ $ok -ge "$STABLE_S" ] && break; else ok=0; fi; sleep 5; done
  SETTLE="settle: $cool gtt_used=${g}GiB MemAvailable=${a}GB stable ${STABLE_S}s, $(( $(date +%s) - last ))s since previous exit, platform_profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null)"; }
KERR='BO_VA|amdgpu.*(ring .* timeout|GPU reset)'
stopall(){ say "STOP $*"; echo "$(date -u +%FT%TZ) abba.sh $OUT: $*" > "$KLT_STOP"; rm -f "$KLT_LOCK.owner"; exit 70; }
fatal(){ grep -aE 'DeviceLost|Not enough memory for command|BO_VA|ErrorOutOfDeviceMemory' "$1" | grep -qv 'pinned memory' && stopall "device error in $1"
         journalctl -k --since "-10 min" --no-pager 2>/dev/null | grep -qE "$KERR" && stopall "kernel amdgpu error after $1"; return 0; }
# stall watchdog: every 60 s, amdgpu kernel error or the bench in D-state with gpu_busy 0 for 5 min -> kill OUR pid only
watchdog(){ local pid=$1 idle=0; while kill -0 $pid 2>/dev/null; do sleep 60
    if journalctl -k --since "-2 min" --no-pager 2>/dev/null | grep -qE "$KERR"; then say "WATCHDOG amdgpu error, killing pid $pid"; kill $pid; return; fi
    local busy; busy=$(cat "$CARD/gpu_busy_percent" 2>/dev/null || echo 1); local st; st=$(awk '{print $3}' /proc/$pid/stat 2>/dev/null)
    if [ "${busy:-1}" -eq 0 ] && [ "$st" = D ]; then idle=$((idle+1)); else idle=0; fi
    [ $idle -ge 5 ] && { say "WATCHDOG pid $pid D-state with gpu idle 5 min, killing"; kill $pid; return; }
  done; }

cell(){ local i=$1 arm=$2 tag=$3; local b=${BIN[$arm]}; local o=$OUT/bench-$i-$arm
  local cmd=(env "${ICDENV[@]}" "LD_LIBRARY_PATH=$b${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ${ENVS[$arm]} "$b/llama-bench" -m "$MODEL" "${ARGS[@]}" -p "$PP" -n "$NGEN" -d "$DEPTHS" -r "$REPS" -o jsonl)
  if [ $DRY = 1 ]; then echo "[dry] cell $i $arm ($tag): ${cmd[*]}"; return; fi
  [ -e "$KLT_STOP" ] && { say "STOP file $KLT_STOP present, not starting cell $i: $(head -1 "$KLT_STOP")"; exit 70; }
  ( exec 9>"$KLT_LOCK"; flock 9; echo "pid=$$ label=abba-$PNAME-$arm started=$(date -Is)" > "$KLT_LOCK.owner"; trap 'rm -f "$KLT_LOCK.owner"' EXIT
    settle
    [ "$i" = 0 ] && say "driverInfo: $(env "${ICDENV[@]}" vulkaninfo --summary 2>/dev/null | grep driverInfo | sed 's/^ *//' | paste -sd';')"
    "$TH/therm-sample.sh" "$o.therm" 2 9>&- & local tp=$!
    timeout "$CELL_TO" "${cmd[@]}" 9>&- > "$o.jsonl" 2> "$o.err" & local bp=$!
    "$TH/flt-sample.sh" $bp "$o.flt" 9>&- & local fp=$!
    watchdog $bp 9>&- & local wp=$!
    wait $bp; local rc=$?; kill $wp 2>/dev/null; wait $wp 2>/dev/null; date +%s > "$LAST_EXIT"
    kill $tp 2>/dev/null; wait $tp 2>/dev/null; wait $fp 2>/dev/null
    local therm; therm=$(python3 "$TH/therm-summary.py" "$o.therm" --busy-only --load-aware); local trc=$?
    [ $trc = 2 ] && echo "$i $arm" >> "$OUT/THROTTLED.txt"
    fatal "$o.err"
    # the device line is a debug-level log upstream: empty at default verbosity, so it is a label, not proof of the device
    say "cell $i $arm ($tag) rc=$rc [$SETTLE] [$therm] majflt=$(cut -d' ' -f1 "$o.flt" 2>/dev/null) $(grep -m1 -oE 'ggml_vulkan: 0 = [^|]*' "$o.err") $(python3 - "$o.jsonl" <<'PY'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1]) if l.startswith('{')]
print(' '.join(f"{'pp' if r['n_prompt'] else 'tg'}@d{r['n_depth']}:{r['avg_ts']:.1f}" for r in rows))
PY
)"
  ) || exit $?
}
therm_now(){ local h; h=$(ls -d "$CARD"/hwmon/hwmon* 2>/dev/null | head -1)
  local k; for k in /sys/class/hwmon/hwmon*; do [ "$(cat "$k/name" 2>/dev/null)" = k10temp ] && break; k=""; done
  echo "edge=$(( $(cat "$h/temp1_input" 2>/dev/null || echo 0) / 1000 ))C tctl=$([ -n "$k" ] && awk '{printf "%.1f", $1/1000}' "$k/temp1_input")C sclk=$(grep -m1 '\*' "$CARD/pp_dpm_sclk" 2>/dev/null | awk '{print $2}')"; }
say "=== abba $PNAME arms: ${ORDER[*]} rounds=$ROUNDS (see ARMS.txt) platform_profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null) idle $(therm_now)"
cell 0 "${ORDER[0]}" warmup-discard
i=0; for r in $(seq 1 "$ROUNDS"); do
  for arm in "${ORDER[@]}"; do i=$((i+1)); cell $i "$arm" "round $r fwd"; done
  for ((k=${#ORDER[@]}-1; k>=0; k--)); do i=$((i+1)); cell $i "${ORDER[$k]}" "round $r rev"; done
done
[ $DRY = 1 ] && exit 0
# throttle policy: a cell flagged by therm-summary.py (exit 2) is marked. THERM_RERUN=1 also re-runs every arm involved once
# (forward then reverse over those arms) and the summary uses the re-runs. Default 0: validate the throttle thresholds on
# your machine first (memory-bound phases run below max clock without throttling, which a bad floor flags as throttle).
if [ "${THERM_RERUN:-0}" = 1 ] && [ -s "$OUT/THROTTLED.txt" ]; then
  RERUN=($(awk '{print $2}' "$OUT/THROTTLED.txt" | sort -u)); say "THROTTLE: cells $(tr '\n' ';' < "$OUT/THROTTLED.txt") -> re-running arms: ${RERUN[*]}"
  mv "$OUT/THROTTLED.txt" "$OUT/THROTTLED-pass1.txt"
  for arm in "${RERUN[@]}"; do i=$((i+1)); for c in "$OUT"/bench-*-"$arm".jsonl; do [ -e "$c" ] && mv "$c" "${c%.jsonl}.throttled.jsonl"; done; cell $i "$arm" "rerun fwd"; done
  for ((k=${#RERUN[@]}-1; k>=0; k--)); do i=$((i+1)); cell $i "${RERUN[$k]}" "rerun rev"; done
  [ -s "$OUT/THROTTLED.txt" ] && say "THROTTLE persists after the re-run: $(tr '\n' ';' < "$OUT/THROTTLED.txt") (results marked, not re-run again)"
fi
python3 - "$OUT" "${ORDER[@]}" <<'PY' | tee -a "$RES"
import json, glob, sys, collections, statistics as st
d, arms = sys.argv[1], sys.argv[2:]; acc = collections.defaultdict(list)
for f in sorted(glob.glob(d + '/bench-*.jsonl'), key=lambda x: int(x.rsplit('/', 1)[1][6:].split('-', 1)[0])):
    i, arm = f.rsplit('/', 1)[1][6:-6].split('-', 1)
    if i == '0': continue   # warm-up discarded
    if f.endswith('.throttled.jsonl'): continue
    for l in open(f):
        if l.startswith('{'):
            r = json.loads(l); acc[(arm, 'pp' if r['n_prompt'] else 'tg', r['n_depth'])].append(r['avg_ts'])
import os
thr = set()
for fn in ('THROTTLED-pass1.txt', 'THROTTLED.txt'):
    if os.path.exists(d + '/' + fn):
        for l in open(d + '/' + fn):
            if l.split(): thr.add(l.split()[1])
print('summary (mean of independent launches, spread = max/min-1, drift = last launch vs first in launch order; base = first arm' + (f"; THROTTLE-marked arms: {' '.join(sorted(thr))}" if thr else '; no throttle flags') + ')')
for kind in ('pp', 'tg'):
    for dep in sorted({k[2] for k in acc}):
        base = acc.get((arms[0], kind, dep), [])
        row = [f"{kind}@d{dep}:"]
        for a in arms:
            v = acc.get((a, kind, dep), [])
            if not v: continue
            m = st.mean(v); sp = (max(v) / min(v) - 1) * 100
            drift = f" drift {100 * (v[-1] / v[0] - 1):+.1f}%" if len(v) >= 2 else ''
            dl = f" ({100 * (m / st.mean(base) - 1):+.1f}%)" if base and a != arms[0] else ''
            row.append(f"{a} {m:.1f} n={len(v)} spread {sp:.1f}%{drift}{dl}" + (' [THROTTLED]' if a in thr else ''))
        print('  '.join(row))
PY
say DONE

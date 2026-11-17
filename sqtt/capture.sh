#!/bin/bash
# capture.sh <label> <outdir> -- <cmd...>
# Runs <cmd> under RADV per-submit thread tracing (SQTT, RGP captures) with the GPU flock held.
# Mesa writes captures to a fixed /tmp path at 1 s resolution; /tmp is replaced by <outdir>/tmp
# inside a private mount namespace (sudo -n unshare) so captures never land on the root fs and
# never collide with other users of /tmp. The command itself runs as the calling user.
#
# Modes (WT_MODE):
#   occupancy (default)  wave start/end only; works on a stock recent RADV. WT_ICD / WT_LDP optional.
#   timing               per-instruction timing in the token layout rocprof-trace-decoder reads
#                        (RADV_THREAD_TRACE_ROCPROF_COMPAT): needs a patched Mesa, so WT_ICD (its
#                        ICD json) and WT_LDP (its lib dir, plus a matching libdrm if built
#                        separately) are required.
#   off                  untraced baseline under the same wrapper (same env, same lock).
#
# Env:
#   WT_ICD=path.json   Vulkan ICD json of the Mesa to trace with (required for timing; default $KLT_VK_ICD, else the
#                      caller's VK_ICD_FILENAMES: the environment is reset for the command, so nothing else carries over)
#   WT_LDP=dir[:dir]   LD_LIBRARY_PATH for that Mesa (required for timing; default $KLT_VK_LIBPATH + the caller's LD_LIBRARY_PATH)
#   WT_BUF=bytes       thread-trace buffer per shader engine (default 256 MiB)
#   WT_SE_MASK=0x..    limit per-instruction records to these shader engines (timing mode)
#   WT_KEEP=lo:hi      keep only submits lo..hi (others deleted as written; bounds disk use)
#   WT_KEEP_LAST=N     keep only the N highest-numbered submits (rolling)
#   WT_TRIGGER_AFTER=s WT_TRIGGER_EVERY=s
#                      no per-submit tracing: touch MESA_VK_TRACE_TRIGGER from s seconds after
#                      launch, every s seconds; each touch captures the next submit. Needs a Mesa
#                      patch: stock Mesa checks the trigger file only at present, and a compute-only
#                      process never presents. Pair with
#                      WT_KEEP_NEWEST=N or WT_KEEP_LARGEST=N (WT_KEEP_MAX_MB drops captures above
#                      that size first: decode RAM scales with capture size).
#   WT_TRIGGER_SKIP=N  capture the N-th submit after each touch (needs a Mesa patch that reads
#                      MESA_VK_TRACE_TRIGGER_SKIP). ggml blocks on a fence after recording a graph,
#                      so N=0 always lands on the next graph's opening upload/copy.
#   WT_MEMAVAIL_GB=G   refuse to start below G GiB MemAvailable (unset = no check). Choose it as the
#                      model's resident size plus capture buffers, so a capture never pushes the
#                      run into swap or OOM.
#   WT_LOCK_WAIT=s     max wait for the GPU lock (default 3600)
#   WT_TIMEOUT=s       wrap <cmd> in timeout(1). A traced run is much slower than an untraced one
#                      and a hung capture holds the lock: always set one for unattended runs.
# Lock and stop file come from lib/klt-env.sh ($KLT_LOCK, owner line in $KLT_LOCK.owner). Refuses
# to start while $KLT_STOP exists; writes it when the kernel log or the run log shows a device
# error during the run (same pattern as jobs/job-run), since later GPU work would be invalid.
set -euo pipefail
. "$(dirname "$(readlink -f "$0")")/../lib/klt-env.sh"
[ $# -ge 3 ] || { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 2; }
LABEL=$1; OUT=$(realpath -m "$2"); shift 2; [ "$1" = "--" ] && shift
MODE=${WT_MODE:-occupancy}
BUF=${WT_BUF:-268435456}
COMPAT=false; TIMING=false; TRACE=rgp
case $MODE in
  occupancy) ;;
  timing)
    : "${WT_ICD:?timing mode needs WT_ICD (ICD json of a Mesa with the rocprof-compat layout)}"
    : "${WT_LDP:?timing mode needs WT_LDP (library dir of that Mesa)}"
    TIMING=true; COMPAT=true ;;
  off) TRACE="" ;;
  *) echo "WT_MODE must be occupancy, timing or off" >&2; exit 2 ;;
esac
# The command runs under setpriv --reset-env, so the caller's driver choice must be passed explicitly: default to the
# pinned ICD ($KLT_VK_ICD, else the caller's VK_ICD_FILENAMES) and the caller's library path, never silently the system Mesa.
ICD=${WT_ICD:-${KLT_VK_ICD:-${VK_ICD_FILENAMES:-}}}
LDP=${WT_LDP:-${KLT_VK_LIBPATH:+$KLT_VK_LIBPATH:}${LD_LIBRARY_PATH:-}}; LDP=${LDP%:}
[ -n "$ICD" ] && [ ! -r "$ICD" ] && { echo "WT_ICD not readable: $ICD" >&2; exit 2; }
[ -e "$KLT_STOP" ] && { echo "$KLT_STOP exists (device error recorded); refusing: $(head -1 "$KLT_STOP")" >&2; exit 70; }
if [ -n "${WT_MEMAVAIL_GB:-}" ]; then
  avail=$(awk '/^MemAvailable:/ {print int($2 / 1048576)}' /proc/meminfo)
  [ "$avail" -ge "$WT_MEMAVAIL_GB" ] || { echo "MemAvailable ${avail} GiB < WT_MEMAVAIL_GB=$WT_MEMAVAIL_GB; refusing" >&2; exit 70; }
fi
mkdir -p "$OUT/tmp"

echo "owner before queue: $(cat "$KLT_LOCK.owner" 2>/dev/null || echo none)"
exec 9>"$KLT_LOCK"
flock -w "${WT_LOCK_WAIT:-3600}" 9 || { echo "GPU lock not acquired within ${WT_LOCK_WAIT:-3600} s" >&2; exit 75; }
echo "pid=$$ label=sqtt-$LABEL started=$(date -Is)" > "$KLT_LOCK.owner"
trap 'rm -f "$KLT_LOCK.owner"' EXIT
CARD=$(klt_gpu_card 2>/dev/null || true)
echo "lock acquired $(date -Is); gpu_busy=$(cat "$CARD/gpu_busy_percent" 2>/dev/null || echo ?)"
UID_=$(id -u); GID_=$(id -g)
KSTART=$(date '+%F %T')
set +e
PERSUB=1; TRIG=""
if [ -n "${WT_TRIGGER_EVERY:-}" ]; then
  PERSUB=0; TRIG="$OUT/trigger"; rm -f "$TRIG"
  ( sleep "${WT_TRIGGER_AFTER:-30}"; while true; do touch "$TRIG"; sleep "$WT_TRIGGER_EVERY"; done ) & TOUCHER=$!
fi
if [ -n "${WT_KEEP_LARGEST:-}" ]; then
  # only files at least 3 s old are judged, so a capture still being written is never deleted
  ( while true; do
      [ -n "${WT_KEEP_MAX_MB:-}" ] && find "$OUT/tmp" -name '*.rgp' -mmin +0.05 -size +"${WT_KEEP_MAX_MB}"M -delete 2>/dev/null
      find "$OUT/tmp" -name '*.rgp' -mmin +0.05 -printf '%s %p\n' 2>/dev/null | sort -rn \
        | tail -n +"$((WT_KEEP_LARGEST + 1))" | cut -d' ' -f2- | xargs -r rm -f
      sleep 1
    done ) & KEEPER=$!
elif [ -n "${WT_KEEP_NEWEST:-}" ]; then
  ( while true; do
      ls -t "$OUT"/tmp/*.rgp 2>/dev/null | tail -n +"$((WT_KEEP_NEWEST + 1))" | xargs -r rm -f
      sleep 1
    done ) & KEEPER=$!
elif [ -n "${WT_KEEP_LAST:-}" ]; then
  ( while true; do
      ls "$OUT"/tmp/*_submit*.rgp 2>/dev/null | sort -V | head -n -"$WT_KEEP_LAST" | xargs -r rm -f
      sleep 0.2
    done ) & KEEPER=$!
elif [ -n "${WT_KEEP:-}" ]; then
  ( lo=${WT_KEEP%:*}; hi=${WT_KEEP#*:}
    while true; do
      for f in "$OUT"/tmp/*_submit*.rgp; do
        [ -e "$f" ] || continue; n=${f##*_submit}; n=${n%.rgp}
        { [ "$n" -lt "$lo" ] || [ "$n" -gt "$hi" ]; } && rm -f "$f"
      done; sleep 0.2
    done ) & KEEPER=$!
fi
[ -n "${WT_TIMEOUT:-}" ] && set -- timeout --kill-after=30 "$WT_TIMEOUT" "$@"
# GGML_VK_DEBUG_MARKERS=1 makes ggml label each dispatch, which rgpwaves.py turns into kernel names
sudo -n unshare -m --propagation private bash -c '
  mount --bind "$0" /tmp
  exec setpriv --reuid="$1" --regid="$2" --init-groups --reset-env -- env \
    HOME="$3" PATH=/usr/local/bin:/usr/bin:/bin \
    ${4:+VK_DRIVER_FILES="$4" VK_ICD_FILENAMES="$4"} \
    ${5:+LD_LIBRARY_PATH="$5"} \
    ${8:+MESA_VK_TRACE="$8"} MESA_VK_TRACE_PER_SUBMIT="${11}" ${12:+MESA_VK_TRACE_TRIGGER="${12}"} MESA_VK_TRACE_TRIGGER_SKIP="${13}" \
    RADV_THREAD_TRACE_INSTRUCTION_TIMING="$6" RADV_THREAD_TRACE_BUFFER_SIZE="$7" \
    RADV_THREAD_TRACE_CACHE_COUNTERS=false RADV_THREAD_TRACE_ROCPROF_COMPAT="$9" \
    RADV_THREAD_TRACE_INSTRUCTION_TIMING_SE_MASK="${10}" \
    GGML_VK_DEBUG_MARKERS=1 \
    "${@:14}"
' "$OUT/tmp" "$UID_" "$GID_" "$HOME" "$ICD" "$LDP" \
  "$TIMING" "$BUF" "$TRACE" "$COMPAT" "${WT_SE_MASK:-0xffffffff}" "$PERSUB" "$TRIG" "${WT_TRIGGER_SKIP:-0}" "$@" 2>&1 | tee "$OUT/run.log"
rc=${PIPESTATUS[0]}
[ -n "${TOUCHER:-}" ] && kill "$TOUCHER"
[ -n "${KEEPER:-}" ] && { sleep 2; kill "$KEEPER"; }
echo "exit=$rc; captures: $(ls "$OUT/tmp"/*.rgp 2>/dev/null | wc -l) ($(du -sh "$OUT/tmp" | cut -f1))"
DEVERR='DeviceLost|ErrorDeviceLost|Not enough memory for command submission|BO_VA|ring .* timeout|GPU reset|device wedged'
dev=$( { journalctl -k --since "$KSTART" --no-pager 2>/dev/null; cat "$OUT/run.log"; } | grep -aE "$DEVERR" | head -1 | cut -c1-160)
if [ -n "$dev" ]; then
  echo "$(date -u +%FT%TZ) sqtt/capture.sh $OUT: device error: $dev" > "$KLT_STOP"
  echo "device error, wrote $KLT_STOP: $dev" >&2; exit 70
fi
exit $rc

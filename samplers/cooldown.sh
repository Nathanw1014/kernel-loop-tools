#!/usr/bin/env bash
# cooldown.sh [tctl_max] [edge_max] [timeout_s] [AMD]: block until the CPU Tctl (k10temp) < tctl_max and the GPU edge
# temperature < edge_max, so a bench cell never starts on a hot package (a build just before a cell heats it, and a cell
# started hot reads low). Prints one summary token for the cell header:
#   cooldown=<waited>s(tctl A->B,edge C->D)
# Exits 0 either way (a timeout is reported, not fatal). A missing sensor reads 0 (passes).
# Defaults (args > env > built-in): COOL_TCTL=60, COOL_EDGE=50, COOL_S=600. Pick limits a few degrees above what the
# machine reads after ~10 idle minutes at your ambient; too low and every cell waits the full timeout.
# COOL_TTM_GB=<GB> (opt-in, default unset = skip): also run ttm-drain.sh <GB> and append its token (needs sudo -n).
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "$0"; }
case ${1:-} in -h|--help) usage; exit 0;; esac
for a in "$@"; do case $a in *[!0-9]*|"") echo "cooldown.sh: numeric args only (got '$a')" >&2; exit 2;; esac; done
T=${1:-${COOL_TCTL:-60}}; E=${2:-${COOL_EDGE:-50}}; TO=${3:-${COOL_S:-600}}
S=$(dirname "$(readlink -f "$0")"); . "$S/../lib/klt-env.sh"
hw(){ for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name 2>/dev/null)" = "$1" ] && { echo $h; return; }; done; }
C=$(klt_gpu_card 2>/dev/null); A=$(ls -d $C/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$A" ] || A=$(hw amdgpu)
K=$(hw k10temp); t0=$(date +%s)
rd(){ echo $(( $(cat $K/temp1_input 2>/dev/null || echo 0) / 1000 )) $(( $(cat $A/temp1_input 2>/dev/null || echo 0) / 1000 )); }
ttm(){ [ -n "${COOL_TTM_GB:-}" ] && echo " $("$S/ttm-drain.sh" "$COOL_TTM_GB")"; }
set -- $(rd); st=$1; se=$2
while :; do set -- $(rd); [ $1 -lt $T ] && [ $2 -lt $E ] && break; [ $(( $(date +%s) - t0 )) -ge $TO ] && { echo "cooldown=$(( $(date +%s) - t0 ))s(TIMEOUT tctl $st->$1,edge $se->$2)$(ttm)"; exit 0; }; sleep 5; done
echo "cooldown=$(( $(date +%s) - t0 ))s(tctl $st->$1,edge $se->$2)$(ttm)"

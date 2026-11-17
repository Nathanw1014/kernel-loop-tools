#!/usr/bin/env bash
# therm-sample.sh <outfile> [interval_s=10]: append one line per interval until killed [AMD]:
#   epoch edge_C tctl_C power_W sclk_MHz(actual, freq1_input) dpm_level(asterisked pp_dpm_sclk line) gpu_busy_%
# The GPU hwmon is the one under the card dir (klt_gpu_card; KLT_GPU_CARD overrides); the CPU sensor is resolved by
# name (k10temp; hwmon numbers move across reboots). tctl reads 0 when there is no k10temp. The header records the
# highest pp_dpm_sclk level (sclk_max_mhz) so therm-summary.py can derive its throttle floor. Reads sysfs only.
. "$(dirname "$(readlink -f "$0")")/../lib/klt-env.sh"
OUT=${1:?outfile}; IV=${2:-10}
hw(){ for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name 2>/dev/null)" = "$1" ] && { echo $h; return; }; done; }
D=$(klt_gpu_card) || { echo "therm-sample: no amdgpu card (set KLT_GPU_CARD)" >&2; exit 1; }
A=$(ls -d $D/hwmon/hwmon* 2>/dev/null | head -1); [ -n "$A" ] || A=$(hw amdgpu); K=$(hw k10temp)
SMAX=$(grep -oE '[0-9]+ *Mhz' $D/pp_dpm_sclk 2>/dev/null | tr -dc '0-9\n' | sort -n | tail -1)
PW=power1_average; [ -r $A/$PW ] || PW=power1_input
echo "# $(date -u +%FT%TZ) card=$D amdgpu=$A k10temp=${K:-none} profile=$(cat /sys/firmware/acpi/platform_profile 2>/dev/null) interval=${IV}s sclk_max_mhz=${SMAX:-?}" >> $OUT
echo "# epoch edge_C tctl_C power_W sclk_MHz dpm_level busy_pct" >> $OUT
while :; do
  e=$(( $(cat $A/temp1_input 2>/dev/null || echo 0) / 1000 )); t=$(awk '{printf "%.1f", $1/1000}' $K/temp1_input 2>/dev/null); p=$(awk '{printf "%.1f", $1/1e6}' $A/$PW 2>/dev/null)
  f=$(( $(cat $A/freq1_input 2>/dev/null || echo 0) / 1000000 )); l=$(grep -m1 '\*' $D/pp_dpm_sclk 2>/dev/null | awk '{print $2}' | tr -d 'Mhz'); b=$(cat $D/gpu_busy_percent 2>/dev/null)
  echo "$(date +%s) $e ${t:-0} ${p:-0} $f ${l:-?} ${b:-?}" >> $OUT; sleep $IV; done

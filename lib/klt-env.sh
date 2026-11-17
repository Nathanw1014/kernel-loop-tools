# klt-env.sh: shared paths and defaults for kernel-loop-tools. Source it from any bash tool:
#   . "$(dirname "$(readlink -f "$0")")/../lib/klt-env.sh"
# Every value can be overridden from the environment; nothing here is machine-specific.

KLT_ROOT=${KLT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}   # repo checkout
KLT_HOME=${KLT_HOME:-$HOME/.kernel-loop}           # machine state: lock, owner, STOP file, queue, progress
KLT_RESULTS=${KLT_RESULTS:-$HOME/kernel-loop-results}  # record root; job OUT dirs must live under it
KLT_LOCK=${KLT_LOCK:-$KLT_HOME/gpu.lock}           # the one GPU flock every tool takes (owner line in $KLT_LOCK.owner)
KLT_STOP=${KLT_STOP:-$KLT_HOME/STOP-GPU}           # written on a device error; every tool refuses to start while it exists
KLT_QUEUE=${KLT_QUEUE:-$KLT_HOME/queue}            # job-queue state
KLT_VK_ICD=${KLT_VK_ICD:-}                         # optional Vulkan ICD json for MESA=release cells (empty = system driver)
KLT_GPU_CARD=${KLT_GPU_CARD:-}                     # /sys/class/drm/cardN/device; empty = first amdgpu card found

klt_gpu_card() {   # print the amdgpu device dir (honours KLT_GPU_CARD)
  if [ -n "$KLT_GPU_CARD" ]; then echo "$KLT_GPU_CARD"; return; fi
  local d; for d in /sys/class/drm/card*/device; do
    [ -r "$d/gpu_busy_percent" ] && { echo "$d"; return; }
  done; return 1
}

mkdir -p "$KLT_HOME" 2>/dev/null || true

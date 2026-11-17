#!/usr/bin/env python3
# clk-sample.py <out.tsv> <pid> [period_ms=50]: sysfs / procfs sampler until <pid> exits [AMD]. Columns: t sclk_MHz power_W busy_pct
# fclk_MHz mclk_MHz socclk_MHz vddgfx_mV vddnb_mV cpu_all_busy_pct proc_cpu_pct (proc = the bench process, % of one core).
# fclk / mclk / socclk = the pp_dpm_* line marked '*' (na when none is marked: the SMU can run between listed levels).
# Any sensor the device does not expose reads na (vddnb exists on APUs only). Card: KLT_GPU_CARD, else the first
# /sys/class/drm/card*/device with gpu_busy_percent (same rule as lib/klt-env.sh).
import glob, os, sys, time

def gpu_card():
    if os.environ.get('KLT_GPU_CARD'):
        return os.environ['KLT_GPU_CARD']
    for d in sorted(glob.glob('/sys/class/drm/card*/device')):
        if os.access(d + '/gpu_busy_percent', os.R_OK):
            return d
    sys.exit('clk-sample: no amdgpu card found (set KLT_GPU_CARD)')

if len(sys.argv) < 3 or sys.argv[1] in ('-h', '--help'):
    sys.exit(open(__file__).read().split('\nimport', 1)[0].replace('#!/usr/bin/env python3\n', ''))
out = open(sys.argv[1], 'w', buffering=1); pid = int(sys.argv[2]); per = (int(sys.argv[3]) if len(sys.argv) > 3 else 50) / 1000.0
d = gpu_card(); hw = sorted(glob.glob(d + '/hwmon/hwmon*'))[0]; hz = os.sysconf('SC_CLK_TCK')
pw_file = hw + '/power1_input' if os.path.exists(hw + '/power1_input') else hw + '/power1_average'
def rd(p):
    try:
        with open(p) as f: return f.read().strip()
    except Exception: return 'na'
def num(p, div, fmt):
    v = rd(p)
    try: return fmt % (int(v) / div)
    except ValueError: return 'na'
def dpm(name):
    try:
        for l in open(d + '/' + name):
            if '*' in l: return l.split(':')[1].strip().split('M')[0]
    except Exception: pass
    return 'na'
def cpu_all():
    v = [int(x) for x in open('/proc/stat').readline().split()[1:]]
    return sum(v), v[3] + v[4]
def proc_t():
    try:
        f = open(f'/proc/{pid}/stat').read().rsplit(')', 1)[1].split(); return int(f[11]) + int(f[12])
    except Exception: return None
pt, pi = cpu_all(); pp = proc_t(); pw = time.monotonic()
while os.path.exists(f'/proc/{pid}'):
    time.sleep(per)
    t = time.monotonic(); tot, idle = cpu_all(); pc = proc_t()
    allb = 100.0 * (1 - (idle - pi) / max(1, tot - pt)); prc = 100.0 * ((pc - pp) / hz) / max(1e-6, t - pw) if (pc is not None and pp is not None) else -1
    pt, pi, pp, pw = tot, idle, pc, t
    out.write(f"{t:.3f}\t{num(hw+'/freq1_input', 1e6, '%.0f')}\t{num(pw_file, 1e6, '%.1f')}\t{rd(d+'/gpu_busy_percent')}\t"
              f"{dpm('pp_dpm_fclk')}\t{dpm('pp_dpm_mclk')}\t{dpm('pp_dpm_socclk')}\t{rd(hw+'/in0_input')}\t{rd(hw+'/in1_input')}\t{allb:.0f}\t{prc:.0f}\n")

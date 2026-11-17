#!/usr/bin/env python3
# dram-sample.py <out.tsv> <pid> [period_ms=50]: DRAM traffic sampler from the amdgpu gpu_metrics sysfs table (no root,
# read-only), until <pid> exits (pid 0 = until SIGTERM / SIGINT). Decodes gpu_metrics v3.0 (APUs on SMU 14.0.x); layout,
# how to verify it, and calibration: DRAM-MONITOR.md. Columns (first lines are '#' headers):
#   t (CLOCK_MONOTONIC s)  act (SMU GfxActivity %, time filtered)  busy (gpu_busy_percent)  rd wr tot (DRAM GB/s, decimal, calibrated)
#   gfxclk fclk uclk socclk (SMU time-filtered MHz)  sock_W gfx_W  rd_raw wr_raw (the raw u16 fields)
# Card: $KLT_GPU_CARD (/sys/class/drm/cardN/device) or the first amdgpu card found (same rule as lib/klt-env.sh klt_gpu_card).
# Calibration: DRAM_CAL = dimensionless factor on the raw MB/s fields (default 1.0); it is written to the header and
# dram-summary.py applies it from there.
# EWMA: the firmware filters every field (time_filter_alphavalue, logged as filter_us), so this measures sustained traffic over
# phases of a few tau, not per-kernel bursts; dram-summary.py can deconvolve.
# WRAP: the u16 fields are MB/s and wrap modulo 65536 (65.5 GB/s). rd / wr are unwrapped by continuity, which only works while one
# sample's change stays under half the range (32.8 GB/s), so: period <= 100 ms, and start the sampler while traffic is under
# 65.5 GB/s (before the job's GPU phase). dram-summary.py re-unwraps from rd_raw / wr_raw.
import glob, os, signal, struct, sys, time

if len(sys.argv) < 3 or sys.argv[1] in ('-h', '--help'):
    print('usage: dram-sample.py <out.tsv> <pid|0> [period_ms=50]   (env: KLT_GPU_CARD, DRAM_CAL; see the header comment)', file=sys.stderr)
    sys.exit(2)
CAL = float(os.environ.get('DRAM_CAL', '1.0'))     # correction factor; raw u16 unit = MB/s (driver_if comment)
GBS = 0.001 * CAL                                  # GB/s per raw unit
out = open(sys.argv[1], 'w', buffering=1); pid = int(sys.argv[2]); per = (int(sys.argv[3]) if len(sys.argv) > 3 else 50) / 1000.0

def find_card():
    d = os.environ.get('KLT_GPU_CARD', '')
    if d: return d
    return next((d for d in sorted(glob.glob('/sys/class/drm/card*/device'))
                 if os.access(d + '/gpu_busy_percent', os.R_OK)), None)
dev = find_card()
if dev is None or not os.path.exists(dev + '/gpu_metrics'):
    sys.exit('dram-sample: no amdgpu card with gpu_metrics (set KLT_GPU_CARD=/sys/class/drm/cardN/device)')
GM = dev + '/gpu_metrics'
def rd(p):
    with open(p, 'rb') as f: return f.read()
b = rd(GM); size, fmt, crev = struct.unpack_from('<HBB', b, 0)
if (fmt, crev) != (3, 0) or size < 264 or len(b) < 264:
    sys.exit(f'dram-sample: {GM} is gpu_metrics v{fmt}.{crev} size {size}; only v3.0 (APU, 264 B) is decoded')
alpha_us, = struct.unpack_from('<I', b, 256)
stop = False
def on_sig(*_):
    global stop; stop = True
signal.signal(signal.SIGTERM, on_sig); signal.signal(signal.SIGINT, on_sig)
out.write(f"# dram-sample {dev} gpu_metrics v3.0 cal={CAL} period_ms={per*1000:.0f} filter_us={alpha_us}\n"
          "# t\tact\tbusy\trd\twr\ttot\tgfxclk\tfclk\tuclk\tsocclk\tsock_W\tgfx_W\trd_raw\twr_raw\n")
if per > 0.1: print('dram-sample: period > 100 ms cannot unwrap the 16-bit counters reliably', file=sys.stderr)
nxt = time.monotonic(); last = {}; wraps = {}
def unwrap(k, raw):
    if k in last:
        d = raw - last[k]
        if d < -32768: wraps[k] = wraps.get(k, 0) + 1
        elif d > 32768: wraps[k] = max(0, wraps.get(k, 0) - 1)
    last[k] = raw; return wraps.get(k, 0) * 65536 + raw
while not stop and (pid == 0 or os.path.exists(f'/proc/{pid}')):
    try:
        b = rd(GM); t = time.monotonic()
        act, = struct.unpack_from('<H', b, 42); r, w = struct.unpack_from('<2H', b, 94)
        sock, = struct.unpack_from('<I', b, 112); gfxp, = struct.unpack_from('<I', b, 124)
        gclk, soc, _vpe, _ipu, fclk, _vclk, uclk = struct.unpack_from('<7H', b, 174)
        busy = open(dev + '/gpu_busy_percent').read().strip()
        ur, uw = unwrap('r', r), unwrap('w', w)
        out.write(f"{t:.3f}\t{act}\t{busy}\t{ur*GBS:.2f}\t{uw*GBS:.2f}\t{(ur+uw)*GBS:.2f}\t{gclk}\t{fclk}\t{uclk}\t{soc}\t"
                  f"{sock/1000:.1f}\t{gfxp/1000:.1f}\t{r}\t{w}\n")
    except Exception as e:
        out.write(f"# read error {e}\n")
    nxt += per; d = nxt - time.monotonic()
    if d > 0: time.sleep(d)
    else: nxt = time.monotonic()

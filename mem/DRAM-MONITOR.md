# DRAM and memory-channel monitoring (AMD, no root)

Two instruments, both read-only and unprivileged:

- **Totals**: `dram-sample.py` reads the amdgpu `gpu_metrics` sysfs table (DRAM read / write MB/s, clocks, power) beside a
  command; `dram-summary.py` unwraps, deconvolves and summarises it. `dram-run.sh <out.tsv> [ms] -- cmd` wraps both.
- **Per channel**: `gcea-sample` reads per-instance GCEA (memory interface) request counters through the rocprofiler-sdk device
  counting service; `chan-run.sh` / `chan-start.sh` / `chan-stop.sh` run it beside a command and `chan-summary.py` reports balance.
  `camp-hip` is its positive control. `gcea-channels.py` does the same per kernel from a `rocprofv3` run.

Neither takes a lock: run them inside the caller's GPU lock.

## gpu_metrics v3.0 layout and how to verify it

`dram-sample.py` decodes `struct gpu_metrics_v3_0` (format 3, content 0, 264 bytes), the table APUs on SMU 14.0.x expose. Source
in the Linux tree:

- `drivers/gpu/drm/amd/include/kgd_pp_interface.h`: `struct gpu_metrics_v3_0` (naturally aligned, not packed).
- `drivers/gpu/drm/amd/pm/swsmu/smu14/smu_v14_0_0_ppt.c`: `smu_v14_0_0_get_gpu_metrics()` fills it from the PMFW `SmuMetrics_t`
  (`drivers/gpu/drm/amd/pm/swsmu/inc/pmfw_if/smu14_driver_if_v14_0_0.h`, where the DRAM fields are commented as MB/sec).
- It calls `smu_cmn_get_metrics_table(smu, &metrics, true)` (bypass the cache), so every sysfs read pulls a fresh table from the SMU.

Offsets used (`offsetof()` of the struct compiled with gcc):

| off | type | field | unit / note |
|---|---|---|---|
| 0 | u16, u8, u8 | structure_size, format_revision, content_revision | must read 264, 3, 0 |
| 42 | u16 | average_gfx_activity | %, firmware-filtered |
| 94 | u16 | average_dram_reads | MB/s, firmware-filtered, **wraps at 65536** |
| 96 | u16 | average_dram_writes | MB/s, firmware-filtered, **wraps at 65536** |
| 112 | u32 | average_socket_power | mW |
| 124 | u32 | average_gfx_power | mW |
| 174..186 | 7 x u16 | average_gfxclk, socclk, vpeclk, ipuclk, fclk, vclk, uclk frequency | MHz |
| 256 | u32 | time_filter_alphavalue | firmware filter time constant, us |

Verify on your machine before trusting a number:

1. Header: `xxd -l 4 /sys/class/drm/cardN/device/gpu_metrics` must read `08 01 03 00` (264, v3, content 0). The sampler refuses
   anything else; another format needs its own offsets.
2. Offsets: compile a one-line `printf("%zu", offsetof(struct gpu_metrics_v3_0, average_dram_reads))` against the header from
   **your** kernel version. Fields have been added to these structs over time.
3. Plausibility: at idle the activity field should track `gpu_busy_percent`, the uclk field should be a constant memory clock,
   and the DRAM fields should sit near a small floor that rises when you start a known stream (calibration below).

## The u16 wrap and the continuity unwrap

The DRAM fields are u16 MB/s, so they wrap at 65 536 MB/s (65.5 GB/s), below what most GPUs stream. Both tools unwrap by
continuity: a sample-to-sample drop of more than half the range (32 768) counts as one wrap up, a rise of more than half the range
as one wrap down (clamped at zero). That is unambiguous only while one sample's true change stays under 32.8 GB/s. Because the
firmware filters the field (next section), the largest change per sample is `peak x (1 - exp(-dt / tau))`, so the condition is

    dt < -tau x ln(1 - 32.768 / peak_GBps)

For example, a 256 GB/s part with tau = 1 s needs dt below about 137 ms; the sampler warns above 100 ms and defaults to 50 ms.
The unwrap also needs a correct starting wrap count of zero: **start the sampler while traffic is under 65.5 GB/s** (before the
job's GPU phase). `dram-summary.py` re-unwraps from the raw columns (`rd_raw`, `wr_raw`), so it is the authoritative reading.

## Firmware EWMA and deconvolution

The firmware smooths every field with an exponential moving average whose time constant is `time_filter_alphavalue` (logged in the
tsv header as `filter_us`). A step reaches 63% after one tau, so the raw fields measure sustained traffic over phases of a few tau.
To verify tau: start a cold, steady stream and read the step response; it should reach 63% / 86% / 95% at 1 / 2 / 3 tau.

`dram-summary.py` inverts the filter per sample:

    y_n = y_(n-1) + k (x_n - y_(n-1)),  k = 1 - exp(-dt / tau)    =>    x_n = y_(n-1) + (y_n - y_(n-1)) / k

The deconvolved ("inst") series recovers phases much shorter than tau, but noise is amplified by 1 / k, so it is gated on a
5-sample median of the deconvolved activity. Use the inst p50 for a steady phase, the `--timeline S` columns for structure, and
prefer phases of at least 3 tau when you read the filtered value directly. Phases shorter than a few samples are averaged with
their neighbours even after deconvolution.

## What is and is not counted

- Counted: all traffic at the memory controller, **after** the last-level cache (on APUs the memory-attached cache / MALL). That
  includes the CPU, the display engine and any other client, not only the GPU job.
- Not counted: cache hits. A working set that fits the last-level cache reads as idle here even at a high kernel rate.
- So: subtract the idle floor of the same cell (the `idle` field, samples with activity under 5%) for GPU-only bytes, and keep
  CPU-heavy phases (model load from page cache) out with `--skip`.
- The per-channel GCEA counters sit **before** the last-level cache: they count fabric requests, cache hits included. The two
  instruments agree only for streams that miss the cache.

## Calibration against dram-mm

`bench/dram-mm` repeats one ggml MUL_MAT back to back for N seconds, in the same graph form as `test-backend-ops perf`, and prints
us/call and the minimum bytes per call (A + B read once, C written once). That gives streams of known bytes:

1. **Read stream**: a mat-vec (`n = 1`) whose A is many times larger than the last-level cache, so it is read cold every call and
   A dominates the bytes. Run it for at least 10 s under `dram-run.sh` and compare `dram-summary.py` inst p50 (minus the idle floor)
   with dram-mm's `min-bytes GB/s`.
2. **Write stream**: a GEMM with small A and B and a large f32 C, so writes dominate. Compare the `wr` column with C bytes / call.
3. **Negative control**: a mat-vec whose A fits the last-level cache must read close to the idle floor.
4. Factor = known / measured. A factor near 1.049 or 0.954 means a MB vs MiB unit mix-up somewhere; a factor that differs between
   the read and the write stream means one of the two streams is not what you think. Set `DRAM_CAL=<factor>` (default 1.0); the
   sampler writes it to the tsv header and `dram-summary.py` applies it from there (`DRAM_CAL` at summary time overrides).
5. Measure the sampler's overhead once: the same steady cell with and without it, at the period you will use.

## Per channel: gcea-sample, with camp-hip as the positive control

`gpu_metrics` has totals only. `gcea-sample` opens KFD in its own process and samples `GCEA_RDRAM_SIZE_REQ` /
`GCEA_WDRAM_SIZE_REQ` per GCEA instance (`DIMENSION_INSTANCE`) through the rocprofiler-sdk device counting service, in windows of
`period_ms`; read and write groups alternate (one event per instance per pass). `chan-summary.py` reports per-instance share and
the max/mean ratio per window (1.00 = even; N active instances fully camped on one = N).

**Coverage**: HIP / KFD work of any process. Vulkan work may read zero depending on the driver: a driver that dispatches with compute
perf counting disabled is invisible to this sampler. Never read a zero, or an even spread, as a verdict on a Vulkan kernel until a
positive control has fired in the same setup. To check what counters a Vulkan driver exposes itself, run `vk-pc-list`
(`VK_KHR_performance_query`; acquiring the profiling lock usually pins clocks, so it is a diagnostic, not a bench mode).

**Positive control** (`camp-hip <stride_B> <chunk_B> [secs] [buf_MiB]`, reads `chunk` bytes at the start of every `stride`-byte row):

1. Sweep the stride with a small chunk (64-256 B). Strides that are a multiple of the channel interleave period put every request
   on one instance (max/mean = number of active instances) and run far slower than their neighbours. The smallest such stride is
   the interleave period; the chunk size at which the spread starts is the interleave granule.
2. Pad a camped stride by one granule: it must spread evenly (max/mean near 1.00). If neither arm camps, the sampler is not seeing
   the traffic and its zeros mean nothing.
3. **Bytes per request**: on a contiguous, DRAM-bound arm (`stride == chunk`, buffer far larger than the cache), requests x B must
   equal camp-hip's touched GB/s and the `gpu_metrics` total; that fixes `--bytes` for `chan-summary.py` (default 32).
4. **Window reset**: a steady stream must give the same count per window (no growth), and an idle machine all zeros.
5. **Active instances**: some parts expose more instances than count; the tools treat only instances nonzero anywhere in the input
   as active and compute balance over those.

**Per kernel** (HIP): write a rocprofv3 extra-counter file with one entry per instance and channel kind, e.g.

```yaml
rocprofiler-sdk:
  counters-schema-version: 1
  counters:
  - name: GCEA_RD_CH00
    description: GCEA_RDRAM_SIZE_REQ on GCEA instance 0
    properties: []
    definitions:
    - architectures: [<your gfx target>]
      expression: select(GCEA_RDRAM_SIZE_REQ,[DIMENSION_INSTANCE=[0]])
  # ... GCEA_RD_CH01 .. CHnn, then GCEA_WR_CH00 .. CHnn on GCEA_WDRAM_SIZE_REQ
```

then `rocprofv3 -E <file.yaml> --pmc "GCEA_RD_CH00 ... CHnn" --pmc "GCEA_WR_CH00 ... CHnn" -d <dir> -o x --output-format csv -- <app>`
(two passes, application replay) and `gcea-channels.py <csv>...` for per-channel share and per-dispatch max/mean per kernel.
Camping is per dispatch: a whole run can average out while single kernels camp.

## Tools and settings

| tool | args / env |
|---|---|
| `dram-sample.py <out.tsv> <pid\|0> [period_ms=50]` | `KLT_GPU_CARD` (default: first amdgpu card), `DRAM_CAL` (default 1.0) |
| `dram-summary.py <tsv>... [--line] [--timeline S] [--act N] [--skip S] [--tau S]` | `DRAM_CAL` overrides the header factor; tau defaults to the header's `filter_us` |
| `dram-run.sh <out.tsv> [period_ms] -- cmd` | wrapper; summary line to stderr, exits with cmd's rc |
| `build-gcea-sample.sh [out_dir] [rocm_root]` | `ROCM_PATH` (default `/opt/rocm`); output defaults to `mem/bin/gcea-sample` |
| `chan-start.sh <tsv> [period_ms=100]` / `chan-stop.sh <pid> <tsv>` / `chan-run.sh <tsv> [period_ms] -- cmd` | `GCEA_SAMPLE`, `CHAN_ROCM` (else `ROCM_PATH`, `/opt/rocm`), `CHAN_GROUPS`, `CHAN_WATCH_PID`, `CHAN_LEAD_S`, `CHAN_TAIL_S`; needs read / write access to `/dev/kfd` |
| `chan-summary.py <tsv>... [--line] [--timeline S] [--from S] [--to S] [--busy F] [--bytes B]` | |
| `camp-hip <stride_B> <chunk_B> [secs=6] [buf_MiB=1024]` | build: `hipcc -O3 camp-hip.cpp -o camp-hip` |
| `gcea-channels.py <csv>... [--top N]` | rocprofv3 CSV with `GCEA_{RD,WR}_CHnn` counters |
| `vk-pc-list` | build: `cc vk-pc-list.c -o vk-pc-list -lvulkan` |

Without a ROCm runtime or `/dev/kfd` access, `chan-run.sh` still runs the command and the summary says `chan: unavailable (...)`.

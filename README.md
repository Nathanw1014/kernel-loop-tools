# kernel-loop-tools

Tools for running GPU kernel optimisation loops on llama.cpp / ggml: correctness gates that
prove they could fail, engagement checks, op-level and in-model timing, counterbalanced A/B
harnesses, hardware samplers, and a small job queue so agents (or people) stop polling.

Written against Vulkan (RADV) on AMD RDNA 3.5 under Linux. Most of it is backend-neutral;
tools that read amdgpu sysfs, RADV or ROCm are marked **[AMD]**.

**Read [GUIDE.md](GUIDE.md) first.** It is the method: one section per loop stage, the trap
each tool guards against, and a per-iteration checklist. This README is only the map.

## Layout

| dir | what | guide section |
|---|---|---|
| `lib/` | `klt-env.sh`: shared paths and defaults, sourced by every bash tool | all |
| `jobs/` | `job-run` (declarative job specs), `job-queue` (FIFO + fast / share lanes), `job-wait` (one blocking wait), `gpu-status`, `job-hazard` (fresh-input and repeat loops), `job-suite` (sharded test suite), `job-summary.py`, `stage-check` (idea linter) + `stage-check-dead.tsv`, `qos-run` / `qoshold.c` (optional CPU latency hold), `kernel-loop-queue.service` | 1, 10 |
| `ab/` | `run-bench.sh` (run id + `.meta` / `.raw` / `.json`), `abba.sh` (counterbalanced ABBA, preset files), `abba-outliers.py`, `abba-postscan.py`, `gpu-state.sh`, `presets.example` | 7, 9 |
| `correctness/` | `repeat_gate.py`, `rollback_gate.py`, `consist-gate.py`, `arms-compare.py`, `bisect-pred.sh`, `batch-validate.sh`, `percommit.sh`, `knobs.example` | 2, 3, 4 |
| `samplers/` | `therm-sample.sh` / `therm-summary.py`, `clk-sample.py` / `clk-summary.py`, `flt-sample.sh`, `fincore.py`, `cooldown.sh`, `ttm-drain.sh`, `trace/` (tracefs wrapper + analyser) | 2, 7, 8 |
| `mem/` | DRAM bandwidth (`dram-sample.py`, `dram-summary.py`, `dram-run.sh`), per-channel counters (`gcea-sample.cpp`, `chan-*.sh`, `chan-summary.py`, `gcea-channels.py`), `camp-hip.cpp` (channel-camping positive control), `vk-pc-list.c`, `DRAM-MONITOR.md` | 8 |
| `bench/` | `mmid_det.cpp` (whole-graph op bench), `dram-mm.cpp` (steady-state matmul), `kv-membench/` (stride / occupancy / latency, plain Vulkan), `lds.py`, `hip-loop-histo.py` | 5 |
| `graph/` | `fusion-census.py`, `fusion-misses.py`, `sync-census.py`, `split-bisect.py`, `fusion-family-bisect.sh`, `model-census.py` | 3, 4, 5 |
| `perflog/` | `perflog.py` (perf-logger parser: `blocks`, `top`, `sum`, `diff`, `shapes`, `ubatch`, `xbackend`), `vktrace.py`, `tracecmp.py`, `tests/` | 6 |
| `sqtt/` | wave / instruction-level traces from RGP captures; see `sqtt/README.md` | 8 |

## Setup

Nothing to install for the Python and bash tools (Python 3 stdlib, bash, flock, coreutils).
C/C++ tools build from the line in each file's header or its `build-*.sh`.

All state goes through `lib/klt-env.sh`; override any of these in your environment:

| var | default | what |
|---|---|---|
| `KLT_HOME` | `~/.kernel-loop` | lock, owner file, STOP file, queue, progress |
| `KLT_RESULTS` | `~/kernel-loop-results` | record root; job `OUT` dirs must live under it |
| `KLT_LOCK` | `$KLT_HOME/gpu.lock` | the one GPU flock every tool takes |
| `KLT_STOP` | `$KLT_HOME/STOP-GPU` | written on a device error; every tool refuses to start while it exists |
| `KLT_VK_ICD` | unset (system driver) | optional Vulkan ICD json for `MESA=release` cells |
| `KLT_GPU_CARD` | first amdgpu card | `/sys/class/drm/cardN/device` |

Per-tool knobs are documented in each tool's header (`--help` or the top comment).

**If you already have a lock convention**, point `KLT_LOCK` at it, so these tools and your
existing scripts serialise on the same file.

### Opt-in features that need privileges

- `TRACE=1` / `KLT_TRACE=1` (kernel trace around a cell) and `KLT_TTM_DRAIN=1` (release the
  TTM page pool) use `sudo -n`. Both are off by default.
- `QOS_US=<us>` needs write access to `/dev/cpu_dma_latency` and a built `jobs/qoshold`.
- Per-channel counters (`mem/gcea-sample`) need ROCm and KFD access; no root.

### Tools whose input is not upstream

Some parsers read output that upstream llama.cpp does not produce yet. Each says so on a
`# needs:` line in its header, with what the patch must emit: `graph/fusion-census.py`,
`graph/fusion-misses.py` (graph dump), `graph/split-bisect.py`,
`graph/fusion-family-bisect.sh` (eval-callback and per-family fusion gates),
`perflog/vktrace.py`, `perflog/tracecmp.py` (per-dispatch Chrome-trace exporter), and the
per-instruction timing mode of `sqtt/`.

## Quick start

```bash
# 1. one A/B, recorded properly
cp ab/presets.example my-model.sh   # edit MODEL, ARGS, MEM_GB
ab/abba.sh out/fa-test my-model.sh base=./base/bin tip=./tip/bin
ab/abba-outliers.py out/fa-test

# 2. where does the time go, per op, in the model
GGML_VK_PERF_LOGGER=1 ./tip/bin/llama-bench -m model.gguf -p 2048 -r 1 2> plog.err
perflog/perflog.py ubatch plog.err --shapes

# 3. queue it instead of watching it
jobs/job-run check my.job && jobs/job-queue submit my.job
jobs/job-wait --marker my-label-DONE --file "$KLT_RESULTS/my/RESULTS.txt"
```

## Safety

- One GPU, one lock. Builds, benches, test suites and captures all take `$KLT_LOCK`.
- A device error writes `$KLT_STOP`; everything idles until a human reads and removes it.
- Never restart the queue service while a job runs (systemd kills its cgroup, job included):
  `job-queue hold`, `job-wait --lock-free`, restart, `job-queue release`.

## Licence

MIT, see [LICENSE](LICENSE). Third-party code keeps its own licence (below).

## Third-party code

`sqtt/third_party/` vendors an RGP / SQTT trace decoder under the MIT licence; see its
`LICENSE` and `PROVENANCE`. The bundled `decoder-python.zip` carries its own licence files.

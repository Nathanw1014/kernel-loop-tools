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
| `lib/` | `klt-env.sh`: shared paths and defaults, sourced by every bash tool; `build-ggml-tools.sh` (builds the tools that link a llama.cpp build) | all |
| `jobs/` | `job-run` (declarative job specs), `job-queue` (FIFO + fast / share lanes), `job-wait` (one blocking wait), `gpu-status`, `job-hazard` (fresh-input and repeat loops), `job-suite` (sharded test suite), `job-summary.py`, `stage-check` (idea linter) + `stage-check-dead.tsv`, `qos-run` / `qoshold.c` (optional CPU latency hold), `kernel-loop-queue.service` | 1, 10 |
| `ab/` | `run-bench.sh` (run id + `.meta` / `.raw` / `.json`), its siblings `run-quality.sh` (perplexity / KLD), `run-tbo.sh` (test-backend-ops), `run-batched.sh` (llama-batched-bench) on `run-lib.sh`, `abba.sh` (counterbalanced ABBA, preset files), `abba-outliers.py`, `abba-postscan.py`, `gpu-state.sh`, `presets.example` | 7, 9 |
| `correctness/` | `repeat_gate.py` (`--multiturn`, `--selftest`), `rollback_gate.py` (`--p-tokens`, `--arms`), `consist-gate.py`, `arms-compare.py`, `bisect-pred.sh`, `batch-validate.sh`, `percommit.sh`, `knobs.example`, `kldtool.py` (KLD files and logs), `fa_masked_leak.cpp` / `topk_repeat.cpp` (backend probes with positive controls), `shader-diff.sh`, `prompts/` | 2, 3, 4 |
| `samplers/` | `therm-sample.sh` / `therm-summary.py`, `clk-sample.py` / `clk-summary.py`, `flt-sample.sh`, `fincore.py`, `pagecache.py` (res / drop / warm), `cooldown.sh`, `ttm-drain.sh`, `trace/` (tracefs wrapper + analyser), `host/` (`thrsample.py`, `memtrace.sh`, `flt-io-sample.sh`, `cpustress.c` and `balloon.c` as contention controls) | 2, 7, 8 |
| `mem/` | DRAM bandwidth (`dram-sample.py`, `dram-summary.py`, `dram-run.sh`), per-channel counters (`gcea-sample.cpp`, `chan-*.sh`, `chan-summary.py`, `gcea-channels.py`), `camp-hip.cpp` (channel-camping positive control), `vk-pc-list.c`, `DRAM-MONITOR.md` | 8 |
| `bench/` | `mmid_det.cpp` (whole-graph op bench), `dram-mm.cpp` (steady-state matmul), `kv-membench/` (stride / occupancy / latency, plain Vulkan), `lds.py`, `hip-loop-histo.py`, `vk/` (plain-Vulkan kit: `vkrun`, `peak.py` + `wmma_peak*.comp`, `vkgemm`, `vkasm`, `isa-pipe`, `isa-histo.py`, `isa-stats.py`, `vkq` + `spin.comp`, `build.sh`) | 5 |
| `graph/` | `fusion-census.py`, `fusion-misses.py`, `sync-census.py`, `split-bisect.py`, `fusion-family-bisect.sh`, `model-census.py`, `graphdump.cpp` + `camp.py` (static stride / camping census), `evcb-diff.py` (first divergent node), `topkdump.cpp` + `moe-route.py` (MoE routing census), `gguf-head.py` | 3, 4, 5 |
| `perflog/` | `perflog.py` (perf-logger parser: `blocks`, `top`, `sum`, `diff`, `shapes`, `ubatch`, `xbackend`), `vktrace.py`, `tracecmp.py`, `tests/` | 6 |
| `sqtt/` | wave / instruction-level traces from RGP captures; see `sqtt/README.md` | 8 |

## Setup

Nothing to install for the Python and bash tools (Python 3 stdlib, bash, flock, coreutils).
C/C++ tools build from the line in each file's header or its `build-*.sh`: `bench/vk/build.sh` for the plain-Vulkan kit
(Vulkan headers + glslc), `lib/build-ggml-tools.sh <llama.cpp source> <build dir>` for the tools that link libllama /
libggml (`graphdump`, `topkdump`, `fa_masked_leak`, `topk_repeat`). Link those against the build you are testing.
Python tools need only the stdlib, except `kldtool.py peek|diff` (numpy).

All state goes through `lib/klt-env.sh`; override any of these in your environment:

| var | default | what |
|---|---|---|
| `KLT_HOME` | `~/.kernel-loop` | lock, owner file, STOP file, queue, progress |
| `KLT_RESULTS` | `~/kernel-loop-results` | record root; job `OUT` dirs must live under it |
| `KLT_LOCK` | `$KLT_HOME/gpu.lock` | the one GPU flock every tool takes |
| `KLT_STOP` | `$KLT_HOME/STOP-GPU` | written on a device error; every tool refuses to start while it exists |
| `KLT_VK_ICD` | unset (system driver) | optional Vulkan ICD json pinned for every run (`MESA=release` cells, `abba.sh`, `run-bench.sh`, `batch-validate.sh`, `bisect-pred.sh`) |
| `KLT_VK_LIBPATH` | unset | extra library dirs that ICD needs (for example a matching libdrm), prepended to `LD_LIBRARY_PATH` |
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

## Tested

The tools were run on one AMD RDNA 3.5 iGPU under Linux (RADV from a stock Mesa release; ROCm for the HIP-only channel
counters) against upstream llama.cpp master, with small models. `--help` on every tool prints usage and does nothing
else. This says what has been exercised, not how fast anything is.

- **Ran, with a control that had to fail and did:** the job verbs (`repeat` DIFFERS on a non-deterministic command,
  `hazard` / `tbo` report NO TEST RAN on an empty filter, timeout, GATE, device-error STOP and refusal), the queue lanes
  (fast yield, share, hold / release / cancel, yield-file preempt and requeue), `abba.sh` self-vs-self, KLD self vs a
  quantised-KV control, `kldtool.py diff`, `bisect-pred.sh` good / skip / bad, the `consist-gate.py` and
  `repeat_gate.py --selftest` controls, `fa_masked_leak`, `topk_repeat`, `evcb-diff.py`, `moe-route.py cmp`, `vkq`
  (delay control reads SERIAL), `camp-hip` under the channel sampler (camped vs padded stride), `shader-diff.sh`.
- **Ran, output checked against a reference:** `perflog.py` (row sums equal the logger's own totals), the DRAM, channel
  and trace summarisers (same output as the tools they were derived from), `lds.py` (upstream's formula),
  `job-summary.py`, the sharded `suite` (partition check, then a master-cache hit with no GPU), `rollback_gate.py` +
  `arms-compare.py` (two runs give identical streams), SQTT occupancy capture through decode (every debug marker matched
  to its dispatch), `vktrace.py` on a build with the trace patch, `isa-histo.py` / `isa-stats.py` (zero attribution
  errors).
- **Ran, output read but no control or reference:** `mmid_det`, `dram-mm`, kv-membench, `peak.py`, `vkasm`,
  `isa-pipe`, `graphdump` + `camp.py`, `topkdump` + `moe-route.py stats`, `model-census.py`, `gguf-head.py`,
  `sync-census.py`, `batch-validate.sh` (with `SUITE_ARGS`), `run-bench.sh`, `run-tbo.sh`, `run-quality.sh`,
  `run-batched.sh`, `repeat_gate.py --multiturn`, the samplers (therm, clk, flt, flt-io, fincore, pagecache, thrsample,
  memtrace, trace-run + analyze, cooldown, ttm-drain), `cpustress bw`.
- **Gates that reported a difference on upstream master, cause not established:** `rollback_gate.py` (a discarded
  branch changed a later logprob; its controls behaved) and `repeat_gate.py --subtoken` (logprob streams of identical
  requests differ while the text is identical). Read their output before treating a FAIL / DETECT as a regression.
- **Not run on hardware:** `QOS_US` (only `check`'s refusal), SQTT timing mode (needs a patched Mesa), `tracecmp.py`
  and `perflog.py xbackend` (need a HIP trace), `fusion-misses.py` / `split-bisect.py` / `fusion-family-bisect.sh`
  (need patches; `fusion-census.py` was run on an existing dump), `percommit.sh`, `vkgemm`, `gcea-channels.py`,
  `hip-loop-histo.py`, `balloon`, `cpustress` modes other than `bw`.

## Licence

MIT, see [LICENSE](LICENSE). Third-party code keeps its own licence (below).

## Third-party code

`sqtt/third_party/` vendors an RGP / SQTT trace decoder under the MIT licence; see its
`LICENSE` and `PROVENANCE`. The bundled `decoder-python.zip` carries its own licence files.

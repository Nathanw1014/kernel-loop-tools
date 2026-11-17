# Tooling for GPU kernel optimisation loops

Rules and tools for running kernel optimisation loops on llama.cpp / ggml (written against
Vulkan on an AMD RDNA 3.5 iGPU under Linux, but most of it is backend-neutral), with AI
agents or people doing the iteration.

In a kernel loop, writing the kernel is the cheap part. What costs you is measuring the wrong
thing with confidence. Every tool below exists to catch one specific way a plausible number
turns out to be an artefact of the harness, the cache, the clock, or a test that ran nothing.

**Tags:** **[AMD]** marks tools that read amdgpu sysfs, RADV or ROCm. The idea applies
elsewhere; the data source differs. **[patch]** marks tools whose input is not in upstream
llama.cpp; the tool's header says what patch produces it. Every tool named here lives in
this repo; the [README](README.md) maps names to paths.

**Thresholds below are example defaults.** Measure your own noise floor and set your bars
from it. The method transfers; the numbers do not.

---

## The loop

```
idea -> pre-screen -> read source -> build -> correctness -> engagement -> op timing
                                                                              |
record <- verdict <- A/B in model (ABBA, launches) <- in-model per-op <-------+
```

Each stage has one job and one way to lie to you. The sections follow that order.

---

## 1. Pre-screen the idea (no GPU)

**Write down the predicted number before you run anything.** If you cannot say "this should
take op X from N us to M us, because the source says Y", the run is exploration, not a test,
and on a single shared GPU exploration is the most expensive thing you can do. Read the
dispatch condition, the GGUF hparam, the test's input generator, the upstream commit message.
A wrong parameter does not fail loudly: it silently routes your test to a different code path,
which then comes back clean.

**Read path-selecting values from the artifact, never derive them.** If a number (k, shape,
quant type, threshold, context size) decides which kernel runs, print it from the GGUF, the
config or the code. Then assert the test reached the intended pipeline.

**Dead list + idea linter** (`stage-check`). A TSV of `KIND  regex  citation` lines (`DEAD` or
`WARN`), plus a script that matches an idea (from your ideas table, or `--text "..."`)
against it and against the minimum-lever bars, and prints `DEAD (cite)`, `WARN`,
`MEASURE-FIRST` or `OK`. An idea that states no number is `MEASURE-FIRST`. Bars are flags.

- Guards against: re-running an idea that was already measured dead, because the verdict
  lived in prose nobody re-read.
- `grep -iE` over a TSV is most of it. The value is the discipline of adding a line, with its
  citation, every time a verdict lands, including nulls.

**Minimum-lever bar.** Size the win from the target op's own serial timing rows (section 6),
not from graph totals:

| bar (example) | meaning |
|---|---|
| net >= 0.5% of the step | worth building as a speed lever |
| >= 1% | worth turning on by default |
| below the bar | only if exact and free; record as "op-level win, wall not provable" |

Set the build bar at or above your across-launch noise (the 95% CI of an ABBA on the full
model). A win smaller than the noise cannot be proven, so building it buys nothing.

---

## 2. Build without perturbing anyone

- **Builds take the GPU lock.** On a unified-memory machine a compile steals memory
  bandwidth from a running benchmark's host feeder thread, and `nice` does not help: it lowers
  CPU priority, not bandwidth. Builds also heat the package; a cell started hot reads low.
  `job-run` build cells take the lock; `percommit.sh` and `bisect-pred.sh` build outside it,
  so run them as job cells or wrap them in `flock "$KLT_LOCK"`.
- **Cool down before a cell** [AMD]: `cooldown.sh [tctl_max] [edge_max] [timeout]` blocks
  until CPU and GPU temperatures are under the limits and prints one token for the cell header
  (`cooldown=<waited>s(tctl A->B, edge C->D)`). Never fatal; a timeout is reported.
- **Compile every commit in a stack**: `percommit.sh <worktree> <builddir> <commit-list>
  <out.tsv>`, one line per commit `sha rc seconds first-error`. Do a full build, not a narrow
  `--target`, before pushing; a narrow target can hide a duplicate definition.
- **Shader compiler and driver are part of the build.** Base and candidate must use the same
  `glslc` and the same driver (ICD); different compiler builds can produce different numerics.
  Never compare against an old baseline measured on another driver, compiler or power profile.
  Record both in every run's metadata.

---

## 3. Correctness: prove the test could have failed

**Assert the executed count, not the exit code.** `test-backend-ops -b Vulkan` matches
nothing (the backend is `Vulkan0`), prints `Skipping`, then `2/2 backends passed`, `OK`, exit
0. A misspelled `-o` filter does the same. Every correctness cell parses the `N/M tests
passed` line and **fails on M = 0**.

**Fresh-input repeat loop** (`job-run` `hazard` cell, `jobs/job-hazard`): run the same `test-backend-ops` filter N times. It draws new
random inputs per launch, so N launches are N input sets. Fail on rc != 0 or any `: FAIL`
line; exit 2 if no case ran in any launch. Catches races that pass once.

**Self-vs-self repeat** (`job-run` `repeat` cell): run a deterministic command twice on the same binary and compare the
`sha256` of its output (greedy token stream, or a logits dump), on a deep-context corpus.
Candidate-vs-base KLD cannot tell an unsynchronised write from rounding; a nonzero
self-vs-self difference can.

**Positive control for every "nothing found".** Any check whose good result is zero, absent
or identical gets paired with an input that must produce a hit, in the same environment. Then
check that the **correct** implementation actually emits a signal for your control. A control
that cannot fire manufactures false suspicion of working code. State the expected magnitude
before running, so an unexpected value cannot be rationalised afterwards.

**Model-level gates:**

- *Repeat gate* (`repeat_gate.py`): identical greedy requests to `llama-server`, compare token streams; with
  `n_probs`, compare per-position logprob streams exactly (`QUIET` / `DETECT`). Catches
  ULP-scale non-repeatability before it flips a token. Run it against a known-broken build
  once to see it `DETECT`. `--multiturn` runs a conversation (appends plus one edit-and-regenerate turn) through the
  prompt cache and compares every cached turn with a full re-prefill of the same messages; it fails if an append turn
  did not hit the cache (the path under test never ran). `--selftest` runs its classifier controls without a server.
- *Rollback gate* (`rollback_gate.py`): prompt-cache rollback must be bit-exact. Arms differ only in a discarded
  branch, each on a fresh server launch, with a null arm (same arm twice) and a positive
  control (one prompt token changed). `--p-tokens N` moves the rollback point deep into the context.
- *Multi-slot consistency* (`consist-gate.py`): each prompt served alone twice, then all at once on distinct
  slots, twice. Any mismatch is reported with the top-5 logprobs at the first differing
  position; a near tie reads as batch-shape numerics, a wide gap as a wrong-result candidate.
  Fail as `NOT ENGAGED` if the concurrent phase did not actually run concurrently.

**KLD relative to a same-binary base.** The same model can read a different absolute KLD on
two binaries. Judge a quant or speed change by its KLD *delta* over a base cell on the same
binary, and know your floor: any change to f32 rounding order in a recurrent state costs a
measurable KLD on its own.

**Where two runs differ, not only how much** (`kldtool.py`): `diff` compares two `--kl-divergence-base` files row by
row and reports the positions, the position within `-ub` and the ubatch of every differing row (a band at ubatch tails,
one ubatch shape, one chunk); `log` parses the padded KLD block of a log and compares two runs against their error
bars; `peek` shows a file's header. A mean hides a defect confined to one ubatch shape.

**Backend probes with built-in positive controls** (no model, any ggml backend by name; run `CPU` as the reference):

- `fa_masked_leak`: changes only K/V cells the mask hides; outputs must stay bit-identical. A visible-cell change per
  case must differ (exit 3 if it does not).
- `topk_repeat`: the same `TOP_K` input many times in one process, with ties at the selection boundary; the selected
  set must repeat (`ggml_top_k` promises no order, so an order change is reported and fails only with
  `TOPK_STRICT_ORDER=1`). A changed row must change its set (exit 3 if not).

**First divergent node** (`evcb-diff.py`, upstream `llama-eval-callback` dumps of two builds): aligns nodes by name within
a window (scheduler reorders and `node_N` renumbering are not differences) and prints the first shape / value
differences. `--selftest` checks the alignment. Resolution is the printed 6-decimal sum.

**Shader identity** (`shader-diff.sh <build-A> <build-B>`): a host-only change must leave every generated SPIR-V and embed
identical; a shader change must touch only the files you meant. Same glslc on both sides, or everything differs.

**Zero-output ubatches.** `llama-bench` outputs the last token of every decode, so it never
builds a ubatch with zero outputs. `llama-perplexity` with a long context, and `llama-server`
with a prompt longer than the batch, do. Fusion matchers that assume `n_tokens >= 1` crash
there. Every gate includes one deep perplexity run.

**Bisect on failure** (`bisect-pred.sh`): a `git bisect run` predicate with an `ops` mode (zero failures on a
test filter) and a `ppl` mode (perplexity under a bar, not NaN). Exit codes `0` good, `1`
bad, `125` skip (build failed), `128` abort on a device error. Never bisect through a GPU
fault.

**Graph-level bisects** for "whole graph wrong, every op right":

- *Prefix-isolation bisect* [patch] (`split-bisect.py`): asking the eval callback for a node makes the scheduler split
  the graph there. Bisect over node order for the smallest set of isolated nodes that
  restores output parity; that brackets the offending op (or the relayout around it).
- *Fusion-family bisect* [patch] (`fusion-family-bisect.sh`): disable one fused family at a time on both builds and compare
  against an unfused reference. The family whose removal restores parity carries the
  difference.

---

## 4. Engagement: prove the new path actually ran

A fast path that never fires produces *exactly* the output a correct one does. Exactness tests
cannot see non-engagement.

- **Count dispatches, not log lines.** Use the backend's fusion counters or the perf logger's
  pipeline names. Print the raw counter line, never a regex over names: a stale regex hides
  fusions.
- **Engagement > 0 is not engagement.** Compare the fused count against the number of
  *eligible* sites and print the ratio.
- **Fusion census** [patch] (`fusion-census.py`, `fusion-misses.py`): dump the graph, then
  for every producer -> consumer edge type count pairs that were fused, adjacent but not
  fused, or split by other nodes. Shows which fusions miss and why (graph optimiser
  reordering, shape checks).
- **Check the log channel is visible.** `llama-server` drops libllama INFO lines without
  `-lv 4`; `llama-bench` drops them without `-v`. Count some other INFO line before reading a
  missing engagement line as "not engaged".
- **Prove it in the deployment config, not just the harness.** `llama-bench` sizes
  `n_ctx = n_prompt + n_gen + n_depth`, so the KV cache is always exactly full, and
  `llama-perplexity` fills every chunk. A layout guard that only holds for a full cache passes
  both and is dead in `llama-server`. Standing check: a small model at `-c 32768` with a ~13k
  prompt; count the gate decision.
- **Some paths need depth to exist.** A sparse-attention or top-k selection path with a key
  budget cannot engage below that many keys. Run its engagement cells deeper than the budget.
- **MoE routing census** (`topkdump` + `moe-route.py`, upstream tensor `ffn_moe_topk`): rows per active expert per ubatch
  and tile fill (is the MUL_MAT_ID tile you tuned ever full at this `-ub`?), expert-set agreement between two builds of
  the same prompt and split, and expert sharing within small batches against random routing.
- **Sync census** (`sync-census.py`, reads upstream `GGML_VK_SYNC_LOGGER` output): count
  syncs per graph eval and histogram the op right after each sync. Unexpected syncs are
  where overlap is lost.
- **Batch validator** (`batch-validate.sh <out> <base-bin> <tip-bin> <knobs-file> <model>
  <corpus> [model flags]`) for a
  stack of env-gated features, in one GPU pass: full test suite base vs tip as a failing-*set*
  diff; knob isolation only for new failures; perplexity base / tip / tip-minus-each-knob;
  ABBA base vs tip; a perf-logger run for op and fusion counts. One gate per stack.

---

## 5. Op-level timing: useful, and wrong in three specific ways

`test-backend-ops perf` is the fastest feedback you have. Treat it as a smoke test, never as
the verdict.

1. **Warm cache.** It replays the same node on the same buffers, so any shape whose operands
   fit the last-level cache runs from cache; in the model every weight is read once per graph,
   cold. Rerun "losing" cells with several distinct A/B sets per op. A cache flush between
   reps must be **read-only** (sum over a buffer larger than the LLC): a write flush leaves
   dirty lines whose write-back lands inside the timed op.
2. **Low clock.** Single-node graphs leave idle gaps that the power manager reads as low
   load, so op loops can run well under the in-model clock. Sample the clock (section 8) and
   compare per GHz.
3. **Whole-graph cases time only the out node.** `eval_perf` builds the graph once, then
   appends `n_runs - 1` copies of the final node and reports total / n_runs. For a fused chain
   test (matmul, then mul, then cast) us/run is the cast. Time fused chains with a whole-graph
   microbench.

**Match the in-model operand types.** A test case with an f32 B dispatches a different
pipeline from a model that feeds f16 B into a fused epilogue, and the two can rank
alternatives in opposite order. Read the op *name* in the model's perf log first, then pick or
add a test case that dispatches the same pipeline.

**Before building any per-type kernel change, read its in-model per-call time.** Op harness
ratios can reverse sign in the model.

**Op microbenches worth having:**

- *Whole-graph op bench* (`mmid_det <type> <m> <k> <n> <bcast> <fused> [reps]`): one op or
  fused chain at model shapes; runs twice for bit-identity, compares against an f32 CPU
  reference built from the same dequantised weights, reports median us/graph.
- *Steady-state op loop* (`dram-mm <type_a> <type_b> <m> <n> <k> <secs>`): one matmul repeated
  for N seconds in the same graph form as `test-backend-ops perf`, so slow firmware counters
  reach steady state. Prints us/call and minimum bytes per call, for bandwidth calibration.
- *Memory-pipeline bench* (`kv-membench`, plain Vulkan + one SPIR-V, any GPU): useful
  bandwidth vs stride (coalescing), vs occupancy (in-flight ceiling), and per-hop latency by
  pointer chase.
- *Static resource check* (`lds.py`, mirrors upstream `ggml_vk_matmul_shmem_support()`):
  compute LDS bytes for a matmul warptile and fail over the limit; read register / LDS use
  from the compiled shader (`hip-loop-histo.py` for a HIP `.s` inner loop). Free, no GPU,
  do it first.
- *Plain-Vulkan kit* (`bench/vk/`, no ggml): `vkrun` dispatches any SPIR-V with spec constants; `peak.py` measures the
  cooperative-matrix peak from a slope over two sizes (submit cost cancels), so "% of peak" has a measured denominator;
  `vkgemm` times one `mul_mm` shader variant outside ggml, with a cold-output mode; `vkasm` and `isa-pipe` build one
  pipeline so the driver dumps its ISA (`isa-pipe` also for GPU families you do not have, via RADV's null device);
  `isa-histo.py` counts the main loop's instruction mix; `isa-stats.py` does it for every pipeline a real run compiles
  (upstream `GGML_VK_PIPELINE_STATS`); `vkq` measures whether two queues really overlap (control: a delay longer than one
  job must read SERIAL).
- *Static camping check* (`graphdump` + `camp.py`): every tensor a GEMM / FA / scan node reads, with its row stride and
  `gcd(stride / granule, channels)`. Set granule and channel count to your memory interleave. Cheap, needs only the
  public eval callback; confirm a suspect stride at runtime with the channel sampler (section 8).
- *Header-only model facts* (`gguf-head.py <file-or-url>`): types, bytes per tensor pattern and hparams without downloading
  weights; the weight bytes per token for a roofline.
- *Shape census* (`model-census.py <gguf>`): every dense matmul `(M, K, type)` with its
  multiplicity, so op benches cover the shapes that carry the time.
- *First-principles cost model*: per-ubatch FLOPs and bytes per op from the model dims; the
  roofline any measured time must sit above. A per-op rate above the roofline means the
  instrument is wrong.

---

## 6. In-model per-op timing

`GGML_VK_PERF_LOGGER=1` (serialising, the default) puts a barrier after every node and gives
exclusive per-op times. Their sum exceeds wall time where the real schedule overlaps; that is
fine. Adding `GGML_VK_PERF_LOGGER_CONCURRENT=1` (it only works together with
`GGML_VK_PERF_LOGGER=1`) gives true totals but smears per-op attribution across overlapping
neighbours. Keep `GGML_VK_PERF_LOGGER_FREQUENCY` at 1: higher values merge graphs into one
block. `perflog.py` (`blocks`, `top`, `sum`, `diff`, `shapes`, `ubatch`, `xbackend`)
applies the rules below by default.

- **Rank from the serialising logger, quote totals from the concurrent one.**
- **Use the last block, not the largest.** The first graph carries lazy pipeline compiles and
  buffer growth.
- **A ubatch can span several blocks** (one per graph split). Find the cycle from the op-count
  sequence before treating "the last block" as one ubatch. `perflog.py ubatch` prints op time
  vs wall plus the residual (host / unlogged); `perflog.py diff` compares two binaries per
  class, op type and GEMM shape.
- **Deferred chain starts.** A fused chain's first node that is "not enqueued" is not timed;
  the logger charges the chain to the next logged node. A surprising row right after a chain
  start is suspect.
- **Count the ops in a chained line before computing a rate.** Dividing a multi-op line by one
  op's FLOPs understates the rate.
- **Cross-check every derived number against the tool's own total** (1000 / tg = ms per token;
  the logger's per-block total). A mismatch over a few percent is a parsing bug until proven
  otherwise.
- **Measure how much concurrency hides on your workload** (serial-minus-concurrent sum as a
  share of the graph). If it is small, predicting wall time from the target op's serial rows
  works; if not, use the concurrent group delta.
- **Per-class split**: bucket the last block into classes (expert GEMM, dense GEMM,
  attention, recurrent / SSM, norms, glue) as ms and %, and compare the same classes against
  another backend's profiler output to localise a gap.

**Per-dispatch traces** [patch] (`vktrace.py`, `tracecmp.py`): a Chrome-trace exporter from the backend (one event per timed
interval with op, tensor, graph number, GFLOP/s, GB/s) opens in `ui.perfetto.dev`, and a
small analyser gives per-op summaries, per-layer splits, host-vs-in-graph gaps and a diff of
two traces. Know the asymmetries before comparing backends: a serialising Vulkan logger
reports isolated op times while a single-stream HIP tracer sums to wall clock, and per-shape
rates only compare where dispatch counts match.

---

## 7. The A/B: across launches, counterbalanced, warmed up

**`-r N` inside one process is not an error bar.** In-process spread says nothing about
launch-to-launch variance (page cache, clocks, allocation), which can be many times larger.
Either the across-launch measurement exists or the number does not get quoted.

**ABBA harness** (`abba.sh <out> <preset-file> <label=bin[:ENV=V,...]> ...`; see
`ab/presets.example`):

- one `llama-bench` process per cell, so every rep is an independent launch;
- **one discarded warm-up launch** before cell 1. Palindrome order cancels drift, not a
  one-off cold-cache cost paid at position 1; without warm-up the first arm is penalised by
  construction;
- each round runs arms forward then reversed (A B .. B A); 2 rounds = 4 launches per arm;
- per cell: lock, wait for GPU memory to drain and free memory above the model's need, load
  average low, then settle; stop everything on a device error;
- records `bench-<i>-<arm>.jsonl` and `.err` verbatim, plus `ARMS.txt` with binary, source
  tree, commit and env per arm.

**Rep-level outlier scan** (`abba-outliers.py <dir> [--ratio 0.85]`): reads every repetition,
flags any rep under the ratio times the best rep of the same test, prints the launch's major
page faults, recomputes arm means from clean reps. Stalled reps on an mmap'd model are
usually major faults under memory pressure, so sample `majflt` per launch
(`flt-sample.sh <pid> <out>`).

**Other rules:**

- A small delta from unpaired arms is not a result. Pair every arm.
- Multi-arm lists inside one `llama-bench` call go A, B, B, A, and are read next to the
  per-sample clock: sustained load lowers the clock between the first and second test.
- `tg` measured after a prefill in the same call is a different quantity from `-p 0` decode.
- An A/B of an environment variable must **prove per arm** which value took effect. A runner
  that pins the variable itself silently voids the A/B.
- `llama-bench` does not seed `rand()`: every process sends the same tokens, so a
  page-cache-backed table gets warmer for exactly those rows on later runs.
- A recipe whose warm state lives in the page cache needs the whole machine, not just the GPU
  lock. If `free` shows a small cache after a launch, the number is void.
- Bench one model to completion before loading the next; settling after a model swap takes
  minutes, not seconds.
- When one arm's spread is much larger than the other's, suspect position or a transient
  before the code. A genuinely bimodal binary stays bimodal; check whether outliers cluster
  in time.

---

## 8. Samplers: clock, thermal, DRAM, channels, host

Each runs beside a cell and prints one summary line into the result. Measure each one's
overhead once on your machine.

| tool | what | trap it catches |
|---|---|---|
| `therm-sample.sh <out> [s]` + `therm-summary.py` (load-aware by default) [AMD] | edge / Tctl / W / actual sclk / DPM level / busy %; hwmon resolved by *name* (numbers move across reboots) | throttled or cold cells. Clock tracks load continuously, so an absolute "sclk < max" rule flags every cell; use a load-aware rule (at >= 90% busy, max clock and power must reach a floor). Sample every 2 s. |
| `clk-sample.py <out> <pid> [ms]` [AMD] | sclk, W, busy, fclk / mclk / socclk, voltages, host CPU %, bench process CPU % | memory-clock downshifts and host-feeder starvation that look like kernel regressions |
| `dram-sample.py <out> <pid> [ms]` + `dram-summary.py` [AMD] | DRAM GB/s from `gpu_metrics` sysfs, no root | fields are u16 MB/s and **wrap at 65.5 GB/s**: unwrap by continuity, sample <= 100 ms, start before the GPU phase. Firmware EWMA (~1 s): deconvolve or use >= 3 s phases. Counts after the LLC, so cache hits are invisible. Calibrate against a known stream (`dram-mm`). |
| `gcea-sample` + `chan-run.sh <out> -- cmd` [AMD] | per-memory-channel read / write requests via rocprofiler-sdk device counters, no root | channel camping (one channel takes all the traffic). Counts HIP/KFD work; check whether your driver exposes counters for Vulkan work before trusting a zero. |
| `camp-hip <stride_B> <chunk_B> [secs] [buf_MiB]` [AMD] | reads `chunk` bytes at the start of every row of a `stride` pitch | positive control for the channel sampler: a stride at a multiple of the interleave period with a small chunk must camp; a padded stride must spread |
| `vk-pc-list` | lists `VK_KHR_performance_query` counters for the device | tells you which hardware counters the driver exposes before you plan on one |
| `fincore.py <file>` | page-cache residency per file via `mincore` | an arm that "got faster" because the model became resident |
| `pagecache.py res\|drop\|warm <file>...` | residency, plus per-file drop (`fadvise DONTNEED`) and warm, no root | putting both arms into the same cache state instead of hoping; drop leaves pages a running process maps |
| `host/thrsample.py <pid> <out>` | per-thread CPU time of a process tree beside GPU busy % | a host thread (graph build, sampling, busy-wait submit) that the GPU waits for |
| `host/memtrace.sh <prefix> cmd` | MemAvailable / Cached / Anon, process RSS split, GTT, largest mappings, every second | where a load's memory goes (page cache vs anon copy vs GTT) and the peak that kills |
| `host/flt-io-sample.sh <ppid> <out>` | major / minor faults plus block-layer and requested read bytes of the child | a model re-read from storage per run, or a cold arm passing as warm |
| `host/cpustress <mode> <n>`, `host/balloon <GiB> <ready>` | pinned CPU load (FMA / L2 / DRAM stream / spin); held, touched, locked memory | positive controls for host contention and memory pressure on shared-memory machines: show the job slows when it should before claiming it does not |
| `trace-run [--set S] -- cmd` + `analyze.py` | private tracefs instance around one command: GPU scheduler run/done, amdgpu CS / BO moves / VM flushes, fence waits, reclaim; sets add sched wakeups, ioctls, block I/O and preads of the bench threads. Needs `sudo -n`. | host-side causes: CPU wake latency on decode, storage reads per token. Keep sets narrow; a wide set can overrun its buffer and change what it measures. |

A cell that reads a few percent low with a tight within-run spread is a confound until proven
otherwise: a compile, a hot package, a co-resident model, page cache state.

**Wave / instruction-level traces (SQTT)** [AMD]: a capture script plus a wave-timeline and
per-instruction decoder over RGP captures. Occupancy captures work on a stock recent RADV;
per-instruction timing may need a patched driver. Use it when the perf logger has localised
one kernel and you need to know why it stalls, and capture it **in the model**: the op-test
regime (cache, layout, clock) can be bound by a different resource than the in-model kernel,
so a fix found in the op trace can lose in the model.

---

## 9. Records: every number has a run id

**Runner** (`run-bench.sh <label> <backend> <binary> -- <llama-bench args>`) writes three
files per run plus one `INDEX.tsv` row:

- `<id>.meta`: timestamp, label, binary, build hash, `--version`, every `GGML_* VK_* LLAMA_*
  RADV_* AMD_*` env var and `LD_LIBRARY_PATH`, exact argv, driver, free memory, whether any
  other process held GPU memory (from amdgpu `fdinfo`, never a process-name grep), lock wait;
- `<id>.raw`: verbatim stderr, including device init lines;
- `<id>.json`: `llama-bench -o json` (its stdout), the graphing source.

Nothing in it interprets. Derived tables are separate files that cite run ids.

Siblings with the same discipline (lock, STOP refusal, `.meta` / `.raw`): `run-quality.sh` (llama-perplexity, PPL / KLD
row in `QUALITY.tsv`), `run-tbo.sh` (test-backend-ops, args untouched because its `-o` is the op filter; a test run that
selects nothing exits 3), `run-batched.sh` (llama-batched-bench, `--output-format jsonl`).

**The tell that you have stopped keeping records:** a command line containing both
`llama-bench` and a pipe into `grep` / `sed` / `awk`. One inline sanity probe is fine. The
moment a second arm appears, switch to the runner: an A/B is by definition something someone
will want to check.

**If the `.meta` is published, every field is a public claim.** If a field cannot survive
"what exactly does this number mean", it does not go in. Before writing "we do X" anywhere
public, check for the artifact X would have left.

---

## 10. Agent mechanics: queue, lock, wait

Most of an agent's token spend in a GPU loop goes to polling, not reasoning: every
`tail` / `stat` / `sleep` turn re-reads the whole context. Replace polling with a queue and
one blocking wait.

**Declarative job spec** (`job-run check|run <spec>`). A spec is a few lines:

```bash
LABEL=fa-rowmajor-ab
OUT=$RESULTS/fa-rowmajor
MEM_GB=45 SETTLE_S=30 TIMEOUT=3600
bench base.1 ./base/bin/llama-bench -m $M -p 2048 -ub 2048 -d 0,8192
bench tip.1  ./tip/bin/llama-bench  -m $M -p 2048 -ub 2048 -d 0,8192
tbo   ops    ./tip/bin/test-backend-ops -o FLASH_ATTN_EXT -b Vulkan0
ppl   kld    ./tip/bin/llama-perplexity -m $M --kl-divergence-base base.kld ...
```

The runner does the lock, memory / thermal gates, settle, timeout, sampler attachment,
records, and device-error stop. `check` refuses missing binaries or paths and timing cells
marked shareable; test cells report the `tests passed` line in their summary.

**Queue** (`job-queue submit|ls|cancel|bump|hold|release`): a FIFO per machine run by a user
systemd unit, so the queue drains with no agent attached.

### Starting, stopping, and the fast lane

**Start.** `submit <spec>` runs `check` first, enqueues, and prints the exact wait command to
use. Agents never start their own daemon loop; if the daemon is down, a human starts it.
Every GPU cell gets the same preamble automatically: refuse if a `STOP-GPU` file exists, take
the lock and write an owner line, cool down, wait for GPU memory to drain and free memory to
reach the cell's declared `MEM_GB`, settle, then run under `setsid` + `timeout` with the
sampler attached. Specs that time a fresh build start with a discarded warm-up cell (cold
shader cache).

**Three lanes, one lock:**

| lane | what goes in it | how it shares the GPU |
|---|---|---|
| normal | benches, A/B blocks, anything timed | exclusive lock, one job at a time |
| fast (`FAST=1`) | short decisive checks: cell timeouts sum to <= 20 min, no held block | at each cell boundary a running normal job lets **one** fast job through, then takes its turn. A yield budget stops a stream of fast jobs from starving the long one. |
| share (`SHARE=1`) | non-timing work: correctness suites, repeat loops, builds, small KLD | lock taken shared (`flock -s`), a few jobs side by side, never beside a timing cell. Backfills: starts only while no normal or fast job runs or waits. Admission by a declared-memory ledger, and never two model loads at once. |

`check` refuses `SHARE` on timing cells and a `FAST` spec that is not actually fast.
`HOLD_LOCK=1` keeps the lock across a whole ABBA block so nothing (fast lane included) slots
between arms; use it only where needed, since it blocks every other lane.

The fast lane replaces "cancel the long job, run a 10-minute check, resubmit". It costs the
long job one cell boundary per fast job, and the long job's next cell settles again before it
runs, so its timing stays clean.

**Stop.**

- `cancel <id>` TERMs the job's whole process tree (cells run under `setsid`), so a job still
  waiting on the lock dies at once; KILL after a grace period. Kill by PID or process group,
  never by name.
- `bump <id> [--before|--after <id2>]` reorders pending jobs instead of cancel-and-resubmit.
- `hold [reason]` / `release` drains for a reboot or a daemon change: pending and new submits
  park, the running job finishes. Drained = one background wait on `--lock-free`.
- **Never restart the queue service while a job runs.** systemd kills the unit's cgroup,
  running job included. Hold, wait for lock-free, restart, release. Run the fast and share
  daemons as separate transient units so restarting the main one does not touch them.
- Per-cell guards kill only that cell: its timeout, a host free-memory floor, and a VRAM
  ceiling where the carve-out is small.
- **A device error stops everything.** The cell that sees device-lost, a VM fault or a ring
  timeout writes `STOP-GPU`, the job ends, every lane idles. A human reads it and removes it;
  agents never delete it. Numbers taken after a GPU fault come from a half-broken driver
  state.

**Yield to other GPU users at cell boundaries.** While the file `$KLT_YIELD_FILE` (default
`$KLT_HOME/YIELD`) exists, no new cell starts; a running cell finishes. Whatever else
serves on the same GPU (an interactive model server, say) creates the file before it loads
and removes it when it unloads; that hook is yours to write. A held block keeps the lock
through a yield unless `YIELD_PREEMPT=1`: then it stops at its next boundary, logs
`PREEMPTED`, requeues at the front under the same id and exits 75. The runner writes a
progress file (cell k/N, elapsed, remaining timeouts) so the waiting side can show an ETA.

**One background wait instead of polling** (`job-wait`): launched once as a background call;
it ends on the first of `--marker NAME --file F` (a *new* line ending in the marker),
`--match RE`, `--pid`, `--lock-free`, `--quiet SECS`. Lines already present at start never
count. Exit codes are the contract: `0` done, `1` exited non-zero, `2` device error (stop all
GPU work), `3` timeout, `4` pid gone without marker.

**Status in one call** (`gpu-status`): lock owner and age, GPU busy / memory, free memory,
temperatures, clocks vs top DPM level (flag `LOW` when busy but well under max), newest result
files with their last line.

**Mechanical traps this layer exists to stop:**

- `{ ...; } &` inside a background tool call: the tool reports done while the GPU job runs
  detached, and the next job piles on top of it.
- A completion watcher matching a substring that also appears in the pending text fires
  instantly. Match an exact, anchored signature.
- `exec 9>LOCK; flock 9` held across stages, then "released" by deleting the owner file: the
  fd is still open, and the next stage deadlocks against its own parent. Release with
  `exec 9>&-`, or take the lock per cell.
- `pgrep -f llama-bench` matches the script's own command line. Kill only your own PID.
- `llama-cli` ignores `-no-cnv` plus EOF and spins on an interactive prompt, writing large
  logs. Use `llama-server` or `llama-simple` in loops.
- One request per `llama-server` launch for speculative-decoding A/Bs: later requests on a
  slot can degenerate into token loops that inflate acceptance and t/s together. Read the
  generated text.

**One GPU means no parallelism.** Two agent sessions do not double throughput; they queue on
the lock and evict each other's page cache. Batch questions into one pass, spend tokens
reading source (free), and spend GPU time confirming a written-down prediction.

---

## Checklist per iteration

1. Idea passes the dead list and states a predicted number with its source line.
2. Value that selects the code path (k, shape, type, threshold) read from the artifact, not
   derived.
3. Built under the lock, same compiler and driver as base.
4. Correctness: executed count > 0, fresh-input repeats, self-vs-self, positive control fires.
5. Engagement: fused / eligible ratio printed, in the deployment config.
6. Op timing: cold operands, clock sampled, in-model operand types.
7. In-model per-op: serialising logger, last block, cross-checked against the tool total.
8. A/B: warm-up launch, ABBA, >= 4 launches per arm, outlier scan, paired arms, env proven
   per arm.
9. Every quoted number has a run id; derived tables cite it.
10. Verdict (win, null, dead) written to the dead list with its citation, including nulls.

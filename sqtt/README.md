# sqtt: wave and instruction traces of ggml-vulkan kernels (AMD, RADV)

Use this after the perf logger has narrowed the problem to one kernel and you need to know
why it stalls. RADV records an SQTT thread trace for each submit into an RGP capture.
`rgpwaves.py` splits that capture into per-shader-engine streams and shader ELFs, and the
vendored rocprof-trace-decoder turns them into wave and instruction records. Two readers
summarise the result: `wavetrace.py` (wave timelines) and `timing_report.py` (per-instruction
stalls).

## Requirements

- RADV with SQTT support, a recent Mesa. Captures are driven by Mesa's own env vars
  (`MESA_VK_TRACE=rgp`, `MESA_VK_TRACE_PER_SUBMIT`, `MESA_VK_TRACE_TRIGGER`,
  `RADV_THREAD_TRACE_*`). `capture.sh` sets them for you.
- `sudo -n` for `unshare -m`. Mesa writes captures to a fixed path under `/tmp`, so
  `capture.sh` bind-mounts `<outdir>/tmp` over `/tmp` in a private mount namespace and then
  runs the command as you.
- llama.cpp built with Vulkan. `capture.sh` sets `GGML_VK_DEBUG_MARKERS=1` so that dispatches
  carry ggml labels, which become kernel names.
- docker, plus an image that ships `librocprof-trace-decoder.so` (`DECODE_IMAGE`; see
  `decode.sh`). The decoder's Python driver is vendored in `third_party/` (MIT). Attribution is
  in `third_party/LICENSE` and `third_party/PROVENANCE`.
- `lib/klt-env.sh`. `capture.sh` holds the GPU flock `$KLT_LOCK` and refuses to start while
  `$KLT_STOP` exists.

## Modes (`WT_MODE`)

| mode | what you get | driver |
|---|---|---|
| `occupancy` (default) | wave start and end per CU / SIMD / slot, matched to dispatches | stock recent RADV; `WT_ICD` / `WT_LDP` optional |
| `timing` | per-instruction hits, latency, stall | a Mesa with the token layout the decoder reads (`RADV_THREAD_TRACE_ROCPROF_COMPAT`, not in stock Mesa); `WT_ICD` and `WT_LDP` required |
| `off` | untraced baseline under the same wrapper | any |

Stock RADV with instruction timing on emits token headers the decoder cannot size.
`rgpwaves.py` refuses such captures. With `RGPW_ALLOW_UNKNOWN=1` it keeps going and counts
them, but the decoder can lose sync after each one, so per-instruction data is partial.
`timing_report.py` prints these trust counters next to every result.

## Flow

```sh
WT_TIMEOUT=900 WT_KEEP_LAST=4 sqtt/capture.sh mm out/ -- llama-bench -m MODEL -p 512 -n 0 -r 1
sqtt/rgpwaves.py extract out/tmp/<capture>.rgp out/x > out/x.extract.log
sqtt/decode.sh out/x
sqtt/wavetrace.py check out/x          # clock sync, window, marker match: trust gate
sqtt/wavetrace.py summary out/x        # per-dispatch wave table
sqtt/wavetrace.py trace out/x x.json   # open in ui.perfetto.dev
sqtt/timing_report.py out/x 'MUL_MAT'  # timing mode: stall classes, load latency, coverage
```

Captures are large. Use `WT_KEEP*` so that only the submits you need are kept. The trigger
mode (`WT_TRIGGER_AFTER` / `WT_TRIGGER_EVERY`) also limits captures, but it needs a patched
Mesa: stock Mesa checks the trigger file only at present, and llama.cpp never presents.
`capture.sh` lists every option in its header.

## Caution: capture in the model

Op-test (`test-backend-ops`) traces can mislead. The op-test regime has different cache
state, layout and clocks from the model, so the kernel can be bound by a different resource
there, and a fix found in the op trace can lose in the model. In-model captures are the
reference. Use op-test traces only to iterate quickly, and confirm in the model.

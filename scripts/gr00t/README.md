# GR00T scripts

Run these commands from the repository root. Keep generated results under
`devlocal/gr00t-n1d7/`; scripts do not automatically run other benchmarks.

The MUSA gates check cosine, relative L2 and `max_abs <= 0.125`, and require
the reference noise (`--reference-noise`, for example the Orin `inputs.npz`)
unless `--allow-missing-noise` is explicit. All MUSA paths require warmup >= 1.
The three GPU lifecycle tests are `#[ignore]`; see
[doc/gr00t-n1.7.md](../../doc/gr00t-n1.7.md#musa-bf16) for the serial commands.

| Script | Purpose |
| --- | --- |
| `bench_apxinf_musa_rust.py` | Native MUSA benchmark, W10/R50, strict Graph by default; optional explicit `--reference` |
| `bench_apxinf_musa_wheel.py` | In-process MUSA Python wheel benchmark using `apxinf_py.ModelRunner`, W10/R50, BF16 |
| `../bench_gr00t.py` | Existing native CUDA benchmark, kept at its original location |
| `bench_torch_eager_musa.py` | Optional NVIDIA PyTorch MUSA reference exporter and benchmark |
| `compare_outputs.py` | Compare native benchmark JSON or normalized-action NPZ against an explicit reference |
| `check_gr00t_module_boundaries.py` | Check MUSA model module isolation |
| `calibrate_gr00t_fp8.py` | Collect BF16 activation maxima from local LeRobot observations or explicit fixtures |
| `calibration_data.py` | CPU preprocessing worker: task-balanced LIBERO frames through the checkpoint's NVIDIA processor |

## Representative FP8 calibration

On the local S5000 host, all whole-model benchmarks use physical GPU1
(PCI `0000:3a:00.0`), irrespective of frontend or precision. Both MUSA scripts
default to `--device 1`; do not remap it with `MUSA_VISIBLE_DEVICES`. If occupied,
wait rather than moving to another GPU. Independent operator tests use other
idle GPUs. Cross-card diagnostics run separately, never alongside model timing.

### FP8 diagnostic switches

All accepted FP8 paths remain enabled by default with `--precision fp8`.
Set these variables **before loading the model**; unset them to restore defaults.
`1`/`true` disables the path; `0`/`false` or an unset variable keeps it enabled.
Values are case-insensitive and surrounding whitespace is ignored. Invalid
values (including an empty string) fail explicitly rather than silently selecting
a path. Use a fresh process for each A/B case; changing variables does not
invalidate an already captured graph. Rebuild the wheel/Rust binary for these
value-based semantics; older builds used presence-based disabling.

| Variable suffix after `APXINF_GR00T_DISABLE_FP8_` | Disabled path / fallback |
| --- | --- |
| `GEMM` | All selective FP8 linears; retain BF16 weights and computation |
| `NORM` | All direct FP8 Norm outputs; BF16 Norm and existing GEMM policy |
| `RMS_NORM` | Text RMSNorm direct FP8 output, including shared Q/K/V |
| `VISION_NORM` | Vision LayerNorm/residual-LayerNorm direct FP8 output |
| `DIT_NORM` | DiT LayerNorm/AdaLN/residual Norm and final output Norm direct FP8 |
| `VL_NORM` | Other adapter LayerNorm direct FP8 output |
| `FUSION_ONLY` | Extra C07 fusion-only consumers; retain their BF16 GEMM |
| `ACTIVATION` | Q10 Bias+GELU and SiLU×up direct FP8 output **only**, not Norm/GEMM |
| `BIAS_GELU` | Q10 Bias+GELU direct FP8 output; BF16 producer then independent quantization |
| `VISION_GELU` | Vision block/merger/deepstack Bias+GELU direct FP8 output only; keep DiT GELU and FP8 GEMM unchanged |
| `SILU_MUL` | Q10 SiLU×up direct FP8 output; BF16 producer then independent quantization |
| `KV_REUSE` | C09 cross-attention K/V shared FP8 activation; quantize separately |
| `KV_CROSS_LAYER` | Reuse between DiT layers; retain C09 sharing within each K/V pair |

`APXINF_GR00T_FP8_EXCLUDE` accepts comma-separated consumer-name prefixes,
e.g. `backbone.text.layers.2.down,action_head.dit_blocks.0.attention.query`.
Those consumers fall back to BF16, including their direct-FP8 producer eligibility.
An empty list excludes nothing. The matching names are the calibration site IDs
without the terminal `.input`. This supports per-consumer experiments without
adding hundreds of environment variables. None of these changes BF16 models.

Current Q10 Bias+GELU hooks are in DiT FFNs, not the vision BF16 GELU callback.
The old broad attribution of vision FC2 quantization savings to Q10 is not
supported by the current call graph. `ACTIVATION` now supports the missing
strict Q10 A/B, leaving identical FP8 GEMMs, scales and Norm paths on both sides.
Norm switches can also disable fusion-only GEMMs or coupled sharing and must
not be interpreted as an isolated Norm-kernel A/B. No new kernels are added.

Use the local LIBERO **training demonstration** dataset, not the official
benchmark fixture. The default selects three distinct episodes per task and
three frames per episode (10%, 50%, 90% of its timeline): 90 observations for
LIBERO-10. It reads both RGB cameras, state and the row's task prompt; camera
and state field mapping follows Isaac-GR00T's `examples/LIBERO/modality.json`.
Do not add simulator image flips or reinterpret the stored state rotation.
The checkpoint processor runs in eval mode and supplies normalization,
cropping, tokenization and embodiment metadata. Noise varies deterministically
with `SeedSequence([seed, sample_index])`; the default seed is 9000.

```bash
source ../activate-apxinf-s5000.sh
python scripts/gr00t/calibrate_gr00t_fp8.py \
  --dataset /home/mccxadmin/youyang/datasets/libero_10_no_noops_1.0.0_lerobot \
  --checkpoint ../models/gr00t/libero_10 \
  --gr00t-source ../Isaac-GR00T \
  --processor-python devlocal/gr00t-prompt-gold/env/bin/python \
  --episodes-per-task 3 --frames-per-episode 3 --seed 9000 --margin 1.2 \
  --device musa:1 --tactics PATH_TO_EXISTING_TACTICS.json \
  --output devlocal/gr00t-calibration/results/profile.json
```

Preprocessing needs the pinned GR00T/Transformers/Torch environment plus
PyArrow and PyAV. `--processor-python` isolates these CPU dependencies from the
lightweight ApxInf inference environment; omit it if the current interpreter
already provides them. No model/data download is performed. On this S5000,
the existing Torch reference environment also needs its MUSA 5.1 dependency
directory appended to `LD_LIBRARY_PATH` as in `devlocal/gr00t-prompt-gold/generate.sh`.

Outputs next to `profile.json`:

- `profile.inputs/`: encoded NPZ observations and a manifest with task/episode/
  frame selection, checkpoint binding, source-file hashes and processor versions.
- `profile.samples.jsonl`: **all 476 per-sample maxima**, not just the final maximum.
- `profile.provenance.json`: profile/sample/tool hashes and tactic identity.
- `profile.json`: strict runtime schema; per-site maximum over all samples,
  then `scale=max(amax*margin/448,1e-8)`. It does not merge the old fixture profile.

Use `--prepared-data profile.inputs` with a **new output path** to reuse verified
encoded observations without decoding again. Existing profiles/sample logs are
not overwritten. Explicit positional fixture paths remain supported for
experiments, but no fixture is selected implicitly; they are labeled
`fixed-input-benchmark`, not representative. Sources cannot be combined.

The representative profile loads directly via benchmark `--precision fp8
--calibration profile.json`, without the benchmark-calibration escape hatch.
The schema's `production=true` indicates a representative source rather than a
synthetic/fixed fixture; **it is not a task-success or deployment-quality approval**.
Hold out evaluation inputs, check correctness separately, and never silently
increase the margin or incorporate evaluation maxima to make that check pass.

## Reference selection

The Rust script launches an executable. The wheel script calls the existing
`ModelRunner.load("gr00t", ...)` and `_infer_preprocessed_timed` bindings directly;
it does not launch Cargo or a Rust subprocess. Both use the same fixture and
explicit initial noise. The wheel passes fixture BF16 bits as uint16 NumPy
arrays; the binding constructs BF16 host tensors before timing, matching the
native runner. No floating-point quantization is performed on fixture values.
The wheel's v4 report uses Rust-internal
`infer_host_f32` timing for `latency_ms` and `model_hz`, excluding NumPy/PyO3
input conversion and NumPy output wrapping. This matches the native runner's
model-call boundary, including synchronized model execution and action D2H.
`api_latency_ms` separately measures the entire Python call. Historical v2
wheel reports included the binding overhead in `latency_ms`. The intermediate
v3 reports excluded binding conversion but passed F32 host tensors, so
model-internal F32-to-BF16 conversion was still timed; those are also not
equivalent to the native BF16-input boundary. Rebuild the wheel to enable
`_infer_preprocessed_timed(...)`, which always returns `(actions, model_ms)`.
The existing `_infer_preprocessed(...)` and `_calibrate_preprocessed(...)`
retain their float32 input signatures and array/dictionary return types.
Only the new diagnostic entry accepts uint16 inputs as raw BF16 bits. Existing
policy `timing.model_ms` and WebSocket `policy_timing.infer_ms` remain Python
call timings, including binding conversion; they are not renamed or replaced
by the diagnostic model-core time. The benchmark fails explicitly with an old
wheel rather than silently falling back to API timing.

### Changing the prompt

Both MUSA benchmark scripts accept `--prompt "put the moka pot on the stove now"`.
They retain the fixture's image, state and noise bytes, and rebuild token IDs
and attention masks before loading/timing the model. The helper uses the local
checkpoint's language normalization, Cosmos chat template and image-grid merge
size, plus the wheel's native `HfTokenizer` and `jinja2`; Torch is not needed.
It first reproduces the original fixture tokens exactly and refuses unsupported
or mismatched fixtures rather than guessing a prompt splice.

The derived fixture is saved beside `--output` as `<output-stem>.fixture/`;
choose a fresh output name for each prompt. Both reports record the prompt,
actual token count and fixture path. Adding one word does not necessarily add
one token. Changing tokens while passing `--reference` is refused: first create
the new fixture without a reference, then generate a matching reference and
run against that derived `--fixture` without `--prompt`. The original prompt's
gold output is not a valid reference for a different prompt. Prompt changes
also do not recalibrate FP8 scales; assess calibration coverage separately.

Keep `--warmup 10 --iterations 50` for comparisons. The wheel accepts
`--precision fp8 --calibration <profile>` as well as BF16.

S5000 three-round validation uses a fixed order: **wheel / normal prompt**
(`put the moka pot on the stove`, 145 tokens), **Rust / short prompt**
(`put the pot on the stove`, 143 tokens), then **wheel / long prompt**
(`put the moka pot on the stove now`, 146 tokens). All rounds use W10/R50.
Use the matching Orin reference for the normal input and the packaged local
Torch eager references for short/long inputs. Compare each input against its
own baseline; judge benefit primarily on the normal prompt.

The local S5000 runner is
`bash devlocal/gr00t-norm-fp8/scripts/run_three_prompts.sh <run-label>`.
It checks GPU1 before each round, uses strict Graph execution and writes fresh
logs/results under `devlocal/gr00t-norm-fp8/`. It expects the local reference
package and previously generated calibration/tactics; it does not download,
build or install dependencies.

Build/install the wheel and run it as documented in
[GR00T N1.7](../../doc/gr00t-n1.7.md#musa-python-wheel).

Prefer acceptance reference values generated **on Orin** using
[`python/apxinf_ref/GR00T-N1.7-LIBERO`](../../python/apxinf_ref/GR00T-N1.7-LIBERO/README.md)
and its `generate_GR00T_N1_7_LIBERO_torch_gold.py` script. Use the exact MUSA
preprocessed fixture as the generator's `--input`, and compare against the
generated `debug.npz` (normalized actions) or `reference.json`. Verify fixture
bytes and checkpoint hashes against the generated `manifest.json` before
comparison; see the linked reference README for the Orin generation command.

The optional M1000 PyTorch eager exporter is a diagnostic tool, not the preferred
Orin acceptance reference. No Orin result is bundled or implicitly selected.
Pass the actual reference explicitly:

```bash
python scripts/gr00t/compare_outputs.py \
  --apxinf path_to/musa_output.json \
  --reference path_to/orin_gold/debug.npz \
  --reference-noise path_to/orin_gold/inputs.npz \
  --precision bf16 \
  --output path_to/parity.json
```

NPY contains normalized actions directly. JSON requires `output.values` and
`output.shape`; NPZ requires `normalized_actions`, with optional `actions` and
`noise`. Compare normalized
actions to normalized actions, not decoded actions. The checkpoint, backbone,
processor, input images/prompt/state, embodiment, precision, denoising settings
and initial noise must match. Verify their provenance separately; this tool
checks shapes, finite values, numerical gates and noise equality. Noise is
required by default; `--allow-missing-noise` is an explicit exemption. A
passing comparison alone does not establish matching input provenance.

## Shared numeric formats

The MUSA wheel benchmark, Rust benchmark wrapper, PyTorch MUSA reference script
and comparison CLI use `formats.py` for their numeric inputs. These rules do not
change the CUDA benchmark or the bit-preserving official gold artifact writer.

| Input | Storage / accepted shape | Loaded representation |
| --- | --- | --- |
| Fixture directory | `manifest.json` plus little-endian raw `.bin` tensors; existing `apxinf.gr00t-n1.7.preprocessed-fixture.v1` schema | BF16 tensors decoded to contiguous float32; IDs remain uint32, mask remains uint8 |
| `--noise-input` NPY | Floating array or uint16 BF16 bits, `[40,132]` or `[1,40,132]` | Contiguous float32 `[1,40,132]` |
| `--noise-input` NPZ | Required `noise` array, same encodings/shapes as NPY | Same as NPY; official `inputs.npz` is supported |
| Reference NPY / NPZ `normalized_actions` | Floating array or uint16 BF16 bits, `[horizon,dim]` or `[1,horizon,dim]` | Contiguous float32 `[horizon,dim]` |
| Benchmark/reference JSON | `output.shape` and numeric `output.values`; numbers represent values, not BF16 bit patterns | Float32 normalized actions, optional singleton batch removed |
| Optional reference `noise` | Same array encodings and shapes as `--noise-input` | Decoded and batch-normalized before equality comparison |

For floating tensors, uint16 always means **raw BF16 bits**, not integer values:
`(values.astype(np.uint32) << 16).view(np.float32)`. Thus `0x3f80` becomes `1.0`,
not `16256.0`. Float16/32/64 inputs are converted numerically to float32. Other
integer types, bool, complex, object arrays, empty arrays and non-finite values
are rejected; NumPy pickle loading is disabled. JSON integer numbers are normal
numeric values, not BF16 storage. Token IDs and attention masks are not passed
through the floating-tensor decoder.

The fixture reader checks declared dtype, exact byte count, shape and path
containment before inference. Wheel and Rust wrapper validate available reference
noise against fixture noise before loading/running the model. A reference without
noise remains usable, but cannot prove matching initial noise.

All reference readers require explicit normalized-action semantics. The Rust
wrapper no longer guesses NPZ keys `final_action`, `output`, or `actions`, nor
accepts bare JSON lists or `actions` aliases. Convert such legacy files to the
canonical `normalized_actions` NPZ or `output.shape`/`output.values` JSON only
after confirming the data is normalized, not decoded robot actions.

Wheel/Rust benchmark reports use `.json`; the PyTorch exporter requires `.npz`
and writes decoded floating arrays. Its exported `noise` records the values
actually consumed after the model's BF16 conversion, not the raw uint16 words
or an unrounded float32 input. `debug.npz` need not contain noise: use the matching
`inputs.npz` for `--noise-input` when necessary. Raw `noise.bin` alone has no shape
metadata and is not accepted by `--noise-input`.

CPU-only format regression tests (no checkpoint or GPU required):

```bash
python -m unittest discover -s scripts/gr00t/tests -v
```

## Optional M1000 PyTorch reference

`bench_torch_eager_musa.py` exports a diagnostic reference on MUSA.
It needs the NVIDIA GR00T package, compatible torch/torch_musa, and this
repository's Python processor adapter. Use `--help` for the required checkpoint,
backbone, images, prompt, eight state values, fixed noise and output paths.
Its output NPZ contains `normalized_actions`, decoded `actions`, `noise`, timing
samples and a report. Pass that NPZ via `compare_outputs.py --reference` for
optional diagnostics; `--torch-eager` remains an alias for compatibility.
As with Orin gold capture, `--repeat` must be at least 2. All measured outputs
must be finite and exactly equal across runs (both normalized and decoded
actions); otherwise the script fails before writing the reference NPZ. On
success it exports the first measured output and records `repeat_exact: true`.

## Benchmark options

- The native Rust benchmark and `bench_apxinf_musa_rust.py` require positive
  warmup and iteration counts. Zero warmup is rejected because execution mode
  is checked after warmup, when the prepared plan must already exist.
- The MUSA wrapper requires `--precision` and accepts optional
  `--calibration`, `--tactics`, `--autotune`, `--output`, and `--reference`.
  `--execution` defaults to strict Graph execution.
  `--autotune` requires an explicit `--tactics` output path.
- The MUSA GR00T loader supports BF16 and MP31 selective FP8 with an explicit
  calibration document. `auto` remains BF16; unmatched GEMMs in the FP8 path
  retain BF16. FP8 uses only muDNN NN/NT, with BF16 output and no post-GEMM
  dequantization kernel. Compare activation-quantization-plus-GEMM cost, not
  FP8 GEMM alone. A fixed-input benchmark calibration is experimental and
  requires `APXINF_GR00T_ALLOW_BENCHMARK_CALIBRATION=1`.
  It loads the explicit
  `--tactics` JSON file or the device/checkpoint default. A missing explicit
  file is an error unless `--autotune` requests its creation. Tuning runs on
  real request tensors before Graph capture; capture itself suppresses tuning.
  This uses the current MUSA backend candidate set: exact assembly routes stay
  on their existing dispatch, while remaining BF16 GEMMs tune muBLAS vs muDNN.
- The comparison tool accepts backend-neutral `--reference`;
  `--torch-eager` remains an alias for compatibility.
- Each benchmark runs independently; no combined benchmark launcher is provided.

# Qwen-Drive BF16 benchmark and validation

## Results

Jetson AGX Thor SM110 (20 SMs), CUDA 13.2, BF16 `planner-sft`, direct
planning, batch one and ten flow steps. Request latency includes policy
preprocessing, native execution and the materialized `[50, 3]` host trajectory;
images are already decoded and model loading is excluded. This is an end-to-end request latency
measurement, whereas the PI0.5 README table measures CUDA Graph replay.

| Measurement | Result |
| --- | ---: |
| Request P50 / P95 | 482.60 / 488.78 ms |
| Request throughput | 2.07 Hz |
| Fixed NAVSIM scenes | 242 |
| PDM score, 0–100 | 85.6786 |
| Trajectories identical to accepted padded implementation | 242 / 242 |

The fixed subset's official-code SFT direct reference scores **84.6974**.
The accepted unpadded optimized version scores **85.6723**. Padding changes 125
of its trajectories and leaves 117 exact, without changing any discrete PDM
component. The current implementation matches the accepted padded version
exactly. The final review build repeats all 242 predictions with zero position
or heading difference; its 85.6786 score is carried forward from the already
scored identical trajectories, rather than a newly repeated scorer run.

These are fixed-subset results, not the published full-navtest scores. The
[official release](https://github.com/QwenLM/Qwen-Drive-1.0#planning) reports SFT/RL
88.2/90.7, or 89.3/91.4 with best-of-six selection. Checkpoint, scene selection,
image profile and sampling protocol must match before comparing scores. The
local gain over official code includes three binary metric flips and does not
establish a systematic quality improvement.

### Recorded timing conditions

Measured on Thor2 on 2026-09-28 with CPU min=max=2.601 GHz, GPU min=max=1.575 GHz,
EMC min=max=4.266 GHz, MAXN and fan PWM 255 with automatic fan control disabled.
The GPU lock covered the entire run; configuration was restored afterwards.
The table uses the pooled 60 measured requests from both candidate arms.

| ABBA arm | Request P50 | Request P95 |
| --- | ---: | ---: |
| Pre-review maintained version, first | 481.35 ms | 485.83 ms |
| Final candidate, first | 482.67 ms | 486.40 ms |
| Final candidate, second | 481.99 ms | 488.78 ms |
| Pre-review maintained version, last | 485.08 ms | 490.65 ms |

Reference pooled P50/P95 is 483.04/489.38 ms. The final implementation preserves
performance within this run's variation. Its two arm medians average 482.33 ms;
the historical 482.98 ms result used that average-of-arm-medians convention.
The table above reports the pooled P50 instead, retaining both conventions
explicitly rather than conflating them.

An earlier run measured 498.78 ms. Repeating the same binary on the same Thor3
later gave 484.90 ms P50, so the earlier slowdown was not evidence of a lost
code optimization or a fixed difference between the two boards. That run lacks
continuous resource telemetry; its exact transient cause remains unproven.

## Build and load

Use Linux AArch64, CUDA 13.2 and the native SM110 operator artifacts. The offline
exporter requires CuTe DSL 4.7.0 and pinned upstream sources; normal engine builds
consume the generated artifacts. Follow the
[AOT export instructions](../crates/apxinf-cuda/aot/README.md), then build:

```sh
APXINF_CUDA_ARCH=sm_110 \
APXINF_CUDA_AOT_MANIFEST=/path/to/artifacts/manifest.json \
  maturin build --release --features cuda,extension-module \
  --compatibility linux -m crates/apxinf-py/Cargo.toml
```

Install the resulting wheel and `python/apxinf` in the test environment. The
checkpoint, planner and tokenizer are external assets.

```python
from pathlib import Path
from apxinf import AutoPolicy

model = Path("/path/to/Qwen-Drive-1.0-4B")
policy = AutoPolicy.from_pretrained(
    model, model_type="qwen_drive", model_variant="bf16",
    planner=model / "planner-sft", mode="direct_planning", num_steps=10,
    tactics="/path/to/empty-vendor.json",
)
```

Use the same explicit tactic store for both comparison arms. The empty vendor
store in this acceptance exercises the operator defaults. `reasoning_planning`
is also supported, but is outside this direct-planning latency result.
VQA and BEV perception are not exposed by this planning runtime.

## Latency procedure

The maintained benchmark constructs all RGB images, ego history and initial
noise in memory. There are three cameras and four frames per camera. Historical
images use `(384, 416)` and current images `(720, 799)` target sizes. These
explicit performance shapes differ from official default-resolution accuracy
inputs. Model and planner weights remain real checkpoint weights.

```sh
python scripts/bench_qwen_drive.py \
  --model-dir /models/Qwen-Drive-1.0-4B --precision bf16 \
  --warmup 10 --samples 30 --seed 0 \
  --out devlocal/qwen-drive-bench/results/latency.json
```

The same command in APXinf-robo delegates to this script and uses Robo's
`load_policy`. Neither entry reads saved input tensors. The report includes
raw request timings, P50/P95, output shape/hash and input profile. Fixed-input
repetitions must return identical finite `[50, 3]` trajectories.

Before measuring, exclude other GPU tasks and CPU compilation/scoring. Save and
lock CPU/GPU/EMC clocks and fan, verify readbacks, and restore settings afterwards.
Warm up ten requests and measure thirty requests in each of two runs. Keep both
reports and pool their raw samples. Record source/native-binary, checkpoint,
AOT-manifest and any explicit tactic hashes. Without `--tactics`, the engine
selects a compatible hardware/toolkit database under `configs/tuning` when one
exists; otherwise it uses provider defaults. For a controlled comparison, pass
an explicit `--tactics` path and record its hash and library versions.

The table above retains the previously published recorded-input results. Use
this maintained benchmark for new measurements.

## Accuracy procedure

Accuracy evaluation belongs to APXinf-robo, just as LIBERO evaluation uses the
robot/environment adapter around PI0.5. From the matching Robo checkout, run:

```sh
python scripts/eval_qwen_drive.py \
  --model-dir /models/Qwen-Drive-1.0-4B \
  --scenes /data/navsim/navtest-observed-history.jsonl \
  --image-root /data/navsim/images --seed 42 \
  --metric-cache /data/navsim/metric-cache/metadata/cache.csv --maps /data/nuplan/maps \
  --results-jsonl devlocal/qwen-drive-eval/predictions.jsonl \
  --summary-json devlocal/qwen-drive-eval/summary.json
```

Prediction and scoring may use separate environments. Omit `--metric-cache`
from the prediction command, then score its existing output in the NAVSIM
Python environment without loading CUDA or model weights:

```sh
python scripts/eval_qwen_drive.py --score-only \
  --scenes /data/navsim/navtest-observed-history.jsonl \
  --results-jsonl devlocal/qwen-drive-eval/predictions.jsonl \
  --metric-cache /data/navsim/metric-cache/metadata/cache.csv --maps /data/nuplan/maps \
  --summary-json devlocal/qwen-drive-eval/summary.json
```

This uses real scene JSONL records in Qwen-Drive's `messages`, `trajectory`,
`meta_info` schema. Each scene has twelve image paths and sixteen observed
history samples at 10 Hz; the evaluator never interpolates the history. Keep
raw-data provenance: an array of length sixteen alone does not prove that it
contains observations rather than upstream interpolations. Use the official
NAVSIM/nuPlan evaluator, maps and trusted metric caches for all scene tokens.

Pin the checkpoint, history construction/filtering, image sizes, CUDA noise
seed, ten flow steps and scoring-cache versions together. The historical
242-scene score above used interpolated history and cannot be compared directly
with corrected observed-history evaluation. Exact official Qwen scene-generation
and filtering details are not fully published; do not claim their published
PDM has been reproduced until those input conditions are established.
The latency benchmark's constructed observations are never scored as accuracy.

## Execution notes

- BF16 has one maintained layer composition. Hardware and tensor contracts
  select the native kernels; no experimental layout switch is required.
- Direct prompts of logical length 3383–3387 pad to 3387 with trailing zero
  token IDs and a policy mask. Shorter prompts keep their original length.
  The native boundary validates this mask and derives the real prefix length;
  it does not pass the mask array to the attention kernel. Causal backbone
  attention and per-row norms preserve the real prefix, and action KV assembly
  overwrites the padded suffix before applying its logical key bound.
  Longer prompts are not truncated;
  reasoning uses variable-length execution. Logical length is part of prepared
  plan compatibility, alongside geometry, image-token positions and step count.
- Whole-model preparation owns its input buffers, weights, KV/state and workspace.
  Tactic changes invalidate the plan. Shape-only preparation lacks the required
  geometry; callers use sample-based `prepare_for`.
- Pillow 12.3.0-compatible geometry uses native bicubic resize, normalization
  and patch packing. Other supported inputs retain their preprocessing contract.
- `model_runner/prepare.rs` owns preparation and tests; `model/blocks/bf16.rs`
  owns BF16 computation and direct inputs; `blocks/mod.rs` declares execution
  callbacks. Follow the [model architecture](model-layer-architecture.md).
- This result qualifies direct planning on SM110. It does not establish the
  same latency or downstream accuracy for reasoning, Orin or RTX 4090.

### Native regression commands

Run GPU tests explicitly on an idle Thor with the same AOT build inputs above;
`--include-ignored` enables the tests marked as requiring CUDA. Serialize GPU
work with the shared device lock.

```sh
cargo test --release -p apxinf-cuda --lib gdn_ -- --include-ignored --test-threads=1
cargo test --release -p apxinf-cuda --lib causal_conv_tiling -- --ignored
cargo test --release -p apxinf-cuda --lib weighted_adaln_residual -- --ignored
cargo test --release -p apxinf-cuda --lib pillow_axis_rejects -- --ignored
cargo test --release -p apxinf-py --features cuda --lib packed_rgb
```

The BF16 chunk-state test uploads BF16 operands and compares the handwritten
PTX path with an independent FP64 recurrence, including the deliberate BF16
state/value rounding. The legacy recurrence tests likewise upload V/T/VT/KCD
with their current BF16 storage contracts. Their former FP32 test buffers
produced invalid comparisons and have been corrected. Value-split coverage
selects distinct legacy launch policies directly, without environment mutation.

Retained actions must be copied before the next inference because prepared
output storage may be reused. The benchmark copies each returned trajectory
before its repeatability comparison.

### Maintained runtime controls and ownership

Qwen-Drive has no production `APXINF_QWEN_*` or GDN experiment environment
switches. Direct planning uses its maintained prepared path; reasoning retains
eager execution. Diagnostic branches that changed fusion/capture were removed.
`OnceLock` remains valid for immutable data caches such as the position table.
The AOT manifest variable is a build input. Generic device/tactic infrastructure is shared with
other model families.

Canonical conditioning packs, in order: flattened historical pose (excluding
the final/current entry), flattened historical velocity, flattened historical
acceleration, current velocity and acceleration plus the one-hot command,
and the scalar navigation command. The native boundary validates dimensions
against the checkpoint before any execution.

Tactic-store reuse is scoped to the toolkit/device/kernel identity recorded by
the tuning infrastructure. AOT identity includes artifact and exporter/contract
hashes; use a compatible explicit store, or the empty vendor store used here.
Do not carry an unrelated measured tactic file across toolkit or kernel changes.

The model architecture owns composition and preparation. The
[CUDA adapter boundary](../crates/apxinf-cuda/adapters/README.md) owns exported
module lifetime, tensor descriptors and launch ABI. NVTX markers use the existing
profiling feature and do not select different numerical implementations.

The `Qwen-Drive contracts` GitHub workflow runs the portable Python checks and
model-family boundary check for relevant PR changes. Native-binding checks skip
when the extension is unavailable. CUDA numerical tests, full model preparation
and NAVSIM quality/performance acceptance require the external Thor assets and
are run explicitly; the hosted CPU workflow does not qualify GPU performance.

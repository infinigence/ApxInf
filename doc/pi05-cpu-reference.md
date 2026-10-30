# PI0.5 CPU reference

PI0.5 supports `Device::Cpu` with `model_variant="fp32_reference"` (also selected
by `auto` on CPU). Its self-contained `cpu_pi05` implementation executes unfused
FP32 operations, without a CUDA toolkit or GPU. This is a numerical reference path;
it does not emulate BF16/FP8/INT8 rounding or promise production CPU latency.

## Build and run

Linux and macOS use the same CPU backend with an installed multithreaded
OpenBLAS library:

```bash
cargo test -p apxinf-model --features openblas --test cpu_pi05
```

Without the `openblas` feature, GEMM uses the serial Rust
fallback, useful for small tests but very slow for full PI0.5. For OpenBLAS
outside the system library search path, set `OPENBLAS_LIB_DIR` to its library
directory; runtime dynamic-library lookup must also be configured appropriately.
The root crate and Python extension expose the `openblas` feature.

A pthreads OpenBLAS build manages its threads internally; for example,
`OPENBLAS_NUM_THREADS=8`. An OpenMP OpenBLAS build instead uses
`OMP_NUM_THREADS`. No OpenMP directives or outer parallel loop are required in
ApxInf. Heads, views and model layers are submitted serially, avoiding nested
parallelism. Library build configuration and problem size determine actual
thread usage. A single-thread OpenBLAS build cannot be made multithreaded by an
environment variable.

```rust
use apxinf_core::Device;
use apxinf_model::{AutoModel, LoadOptions};

let model = AutoModel::load_model(
    Device::Cpu,
    "/path/to/pi05-checkpoint",
    &LoadOptions {
        model_name: Some("pi05".into()),
        model_variant: Some("fp32_reference".into()),
        ..LoadOptions::default()
    },
)?;
// Existing VlaRuntime requests and infer_host_f32 are unchanged.
```

With a CPU-built native Python extension, use the existing binding:

```python
from apxinf import ModelRunner

runner = ModelRunner.load(
    "pi05", "/path/to/pi05-checkpoint", device="cpu",
    model_variant="fp32_reference",
)
# runner._infer_patches / runner.infer_rgb use the existing public input contract.
```

`Pi05Policy.from_pretrained(..., device="cpu", model_variant="fp32_reference")`
uses the same native path and skips discovery of checkpoint-local CUDA tactics.
Tokenization, state-to-prompt encoding, resize and action unnormalization remain
policy responsibilities. `pi05-cuda` continues to require CUDA.

## Contract and ownership

The `cpu_pi05/` port owns its implementation independently of `pi05/` (CUDA)
and `hip_pi05/`. It does not import their configuration, weights or model code.
The existing CUDA directory is unchanged. Copies are intentional; independent
reference fixtures establish agreement between ports.

- `config.rs` owns CPU geometry, checkpoint configuration and timestep math.
  `AutoModel` converts the historical `LoadOptions.config` into this local type
  outside both model directories, just as it does for HIP.
- `load.rs` selects CPU assets and assembles the concrete model and runner.
- `weights.rs` owns checkpoint mapping and converts F32/F16/BF16 tensors to FP32
  before norm offset and scale-folding transformations. It validates consumed shapes.
- `model.rs` owns vision, prefix KV, adaptive normalization, attention, gated MLP,
  forward order and the Euler flow loop in `CpuPi05Model`. There is no generic
  Blocks trait or precision-dispatch enum for this single implementation.
- `runner.rs` owns `CpuPi05Runner`: request validation, RGB patchification,
  fixed timestep modulation, noise generation and explicit eager plans.
- `apxinf-cpu` owns generic FP32 math. Dense GQA/MHA packs one head at a time
  and uses BLAS GEMM for both QK-transpose and probability-times-V, with a stable
  Rust softmax. It is non-causal; image views are independent batches, prefix
  queries see the whole prefix, and action queries see prefix plus all action
  tokens. No prefix query sees action tokens.

`apxinf-cpu` is a separate backend crate, depending on `apxinf-core` for
`Backend`, `Tensor`, `KvCache` and sampling contracts. CPU operators, concrete
KV storage, samplers and OpenBLAS linking live in the CPU crate. Core has no BLAS
features or dependency on a backend. The CPU crate exposes only `openblas` for
BLAS selection; no Apple-specific accelerator feature is introduced.

Rust callers import `CpuBackend` and `CpuKVCache` from `apxinf_cpu`.
`Tensor::matmul_cpu` is replaced by `CpuBackend.matmul(&a, &b)` with the
`apxinf_core::Backend` trait in scope. HIP's existing host sampling fallback
explicitly depends on the CPU crate; CUDA uses it only in comparison tests.

Inputs are normalized patch rows or already resized RGB `u8` in NHWC/NCHW.
Patch flattening is channel, patch-y, patch-x. Host FP32/F16/BF16 patches and
provided noise are accepted and converted to FP32. All supplied views/tokens
must be valid; padding masks and extra planning/embodiment metadata are rejected.
State must already be encoded in prompt tokens; direct state and action masks
are not supported by this reference path. Token IDs and tensor shapes are checked.

`Eager` preparation runs the actual computation before reporting ready.
`PreferGraph` returns an eager plan with an explicit fallback reason.
`RequireGraph` fails. No CPU no-op graph is replayed. Explicit plans retain the
model after their original runner is dropped; changed token counts or input
layouts require another plan. There is no implicit CPU plan cache to invalidate.

Only unquantized checkpoint weights and `auto`/`fp32_reference` are supported.
Explicit CUDA calibration/tuning options are rejected. Fixed provided noise
should be used for cross-framework comparisons; equal seeds alone do not imply
identical random tensors across frameworks. FP32 full-model weights require
roughly four bytes per parameter, plus loading temporaries and activations.

## Verification

Run the maintained CPU checks without external BLAS:

```bash
cargo test -p apxinf-cpu --lib
cargo test -p apxinf-model --lib cpu_pi05::
cargo test -p apxinf-model --test cpu_pi05
```

Add `--features openblas` to exercise the accelerated path. See the
[CPU backend guide](../crates/apxinf-cpu/README.md) for linking and threading.

The maintained CPU tests cover public loading, RGB/patch layout equivalence,
changed RNG inputs, retained outputs/plans, eager readiness, graph rejection and
invalid requests. CPU backend tests compare dense grouped attention to scalar FP64
arithmetic. CPU intermediate-stage tests exercise the port's own model.

The end-to-end and intermediate-stage tests use the tracked
[tiny fixture](../crates/apxinf-model/tests/fixtures/pi05_cpu/README.md).
They run by default in CI with no `APXINF_PI05_CPU_REFERENCE` environment variable,
Python installation, network access or external checkpoint. The fixture pins random
weights, two RGB/token/provided-latent requests, and independent PyTorch golden
outputs for vision, prefix, every layer K/V and every flow step. Its generator
verifies the pinned OpenPI helper source hashes. It is a small functional oracle,
not a trained model or a production CUDA workload.

### CPU/GPU regression gate

Three comparisons have separate meanings and acceptance criteria:

| Comparison | Criterion | Coverage |
| --- | --- | --- |
| Tiny CPU vs independent PyTorch oracle | Elementwise `atol=rtol=2e-5` | Default portable CI, final output and intermediate stages |
| Full CPU FP32 vs OpenPI FP32 | Elementwise `atol=rtol=2e-4` | Recorded full-checkpoint validation below |
| GPU precision vs CPU FP32 | Per-device, per-precision, per-case cosine floor | Both eager and graph outputs; L2/max-abs reported as diagnostics |
| GPU eager vs graph | Max-abs = 0; repeated graph output bitwise equal | All required GPU CI lanes |

The full-checkpoint gate uses `AutoModel` and explicit `Eager`/`RequireGraph`
plans. Each backend loads the same checkpoint and receives identical RGB, tokens
and explicit FP32 noise; the GPU's input conversion is included in the measured
precision error. The CPU model is dropped before loading the GPU model. Model
weights, RGB and FP8 calibration are SHA256 checked. The report also pins the
manifest hash, including token IDs, noise, geometry and limits.

The tracked [T10/T21 diagnostic corpus](../crates/apxinf-model/tests/fixtures/pi05_gpu/README.md)
provides reproducible full-size gradient RGB, token IDs and explicit latents:

```sh
python crates/apxinf-model/tests/fixtures/pi05_gpu/materialize.py \
  --output devlocal/pi05-cpu-reference/gpu-diagnostic
```

Prepare an input manifest from existing NPZ requests containing `images`
(`[views,224,224,3]` uint8), `token_ids` and `noise` (`[horizon,32]` float32):

```sh
python scripts/prepare_pi05_reference_gate.py \
  --checkpoint /models/pi05/model.safetensors \
  --calibration /models/pi05/calibration.json \
  --output devlocal/pi05-cpu-reference/gate-inputs \
  /fixtures/case-a.npz /fixtures/case-b.npz
cargo build --release -p apxinf-model --example pi05_reference_gate --features cuda,openblas
```

Omit `--calibration` when only checking BF16/INT8. Use existing real-input cases
for threshold measurements; small random fixtures do not calibrate production
quantization. The generated manifest has an empty `limits` map. First collect
measurements on the appropriate GPU (BF16/FP8 on Thor, BF16/INT8 on Orin):

```sh
OPENBLAS_NUM_THREADS=4 target/release/examples/pi05_reference_gate \
  /models/pi05/model.safetensors \
  devlocal/pi05-cpu-reference/gate-inputs/manifest.json \
  bf16 devlocal/pi05-cpu-reference/bf16-measure.json \
  --tuning devlocal/pi05-cpu-reference/recipes --autotune --measure
```

For `fp8_static`, add `--calibration /models/pi05/calibration.json`.
To use an existing recipe directory, add `--tuning PATH`. To populate a dedicated
run directory, add `--tuning devlocal/pi05-cpu-reference/recipes --autotune`.
The report records this selection. Preserve the qualified recipe directory with
the run evidence; recipe misses use the backend's fallback and can change the
implementation being measured. Measurement mode reports `passed: null`; it is not a passing
regression gate. After reviewing the real-input baseline, pin each variant's
`min_cosine`, `max_relative_l2`, `max_abs`, and `eager_graph_max_abs` under
`limits.bf16`, `limits.fp8_static` or `limits.int8_dynamic`. Re-run without
`--measure` to enforce all limits and return a nonzero exit status on a failure.
Missing limits are an error. Each invocation saves its FP32 reference as
`REPORT.json.cpu.json` before starting GPU work. Subsequent precision runs may
pass `--cpu-reference REPORT.json.cpu.json`; the tool verifies the checkpoint,
geometry, RGB hashes, token IDs and exact latents match. Changing thresholds or
FP8 calibration does not change the CPU input identity. To regenerate the
reference without running a GPU, select variant `cpu`. Regenerate it deliberately
when qualifying a CPU implementation change; cached outputs are frozen evidence,
not proof that a changed CPU model still matches OpenPI.

Keep the pinned manifest, reports and source revision
with the validation evidence under `devlocal/pi05-cpu-reference/` and attach the
results to the PR, following the repository's manual GPU regression process.

This GPU gate requires an external full checkpoint and GPU; the default corpus
contains reproducible generated inputs. It does not run in portable CI. Its tensor-level accuracy checks are separate from the
[formal 500-episode task-success evaluation](pi05-cuda-regression.md#7-accuracy-standard-and-current-reference-results).

### GPU qualification status (2026-10-29)

The new gate was exercised on the same four full-size requests recorded below,
with the unchanged model implementations from PR #176 revision `8449a97c`.
The freshly computed CPU reference still passes all four frozen OpenPI FP32
comparisons. GPU runs used explicit autotuned recipe directories; untuned
attempts stalled during eager preparation (a Thor backtrace reached
`cublasCreate` / `cuMemAlloc`). FP8 used the checkpoint-matching profile with
SHA256 `a7d43c22847f421ea849af1a1c600f169d09fea1420a7d81fe4d12c1976c6f2d`.

The extrema below are across all four cases, not necessarily the same case:

| GPU path | Minimum cosine vs CPU | Maximum relative L2 | Maximum absolute error | Eager/graph max-abs |
| --- | ---: | ---: | ---: | ---: |
| Thor BF16 | 0.996851 | 0.079392 | 0.610382 | 0 |
| Thor FP8 | 0.991483 | 0.133006 | 0.822724 | 0 |
| Orin BF16 | 0.993657 | 0.113234 | 0.866181 | 0 |
| Orin INT8 | -0.562536 | 1.771758 | 2.021155 | 0 |

All graph repeats were bitwise stable. Existing `pi05_bench` aggregate limits
are not transplanted onto the new FP32 reference. The initial CI floors use
actual measured cosine, rounded down to six decimal places for each case;
relative L2 and max-abs remain visible diagnostics. This is a regression floor,
not a claim that the existing quantization error is acceptable for robot deployment.
The two DROID cases remain numerical stress inputs for a LIBERO checkpoint.

### CI policy and monotonic thresholds

[reference-gate.json](../configs/pi05/reference-gate.json) pins the public corpus
hash and separate floors for both eager and graph outputs:

| Required path | T10 cosine floor | T21 cosine floor |
| --- | ---: | ---: |
| Thor BF16 | 0.999991 | 0.999917 |
| Thor FP8 | 0.997492 | 0.991482 |
| Orin BF16 | 0.999977 | 0.999917 |

Orin INT8 runs as an **advisory** lane. Its numerical failures or execution
failures do not fail the GPU workflow; reports and job logs remain available.
Required lanes fail on either case/mode below its cosine floor, invalid/missing
metrics, eager/graph differences or unstable graph replay. L2/max-abs do not
independently fail the CPU/GPU accuracy gate. The standalone Rust example also
supports custom multi-metric limits for investigations; CI runs measurement mode
and applies the maintained board-specific policy using the Python checker.

The portable `PI05 reference contracts` workflow runs the tiny independent CPU
oracle and policy tests on every PR. Once the policy exists on the base branch,
it executes the **base branch's checker** against the candidate policy. Existing
case floors can only increase, required lanes cannot become advisory, graph
tolerance cannot loosen, and corpus identity cannot be substituted. Keep input
identities fixed when raising floors after verified improvements; add a separate
corpus when expanding coverage. Do not lower floors to turn a regression green.

The manually dispatched `PI05 GPU reference` workflow checks out the selected
branch/revision and runs Thor BF16/FP8 and Orin BF16/INT8. Like the previous
`codex/pi05-ci-gate` work, it uses self-hosted `Linux`, `ARM64`, `pi05-ci` runners
with `thor`/`orin` labels, fixed assets and retained reports. Unlike that unfinished
broader task, this initial gate covers two public full-size requests, without
requiring its private image bank or performance budgets.

Runner setup requires CUDA, Cargo, multithreaded OpenBLAS, enough memory/disk,
and repository variables `PI05_CHECKPOINT` and `PI05_CALIBRATION` containing
runner-local paths to the pinned assets. Set `OPENBLAS_LIB_DIR` and runtime
library search paths in the runner environment if required. Workflows do not
register runners or change branch protection. Configure those separately; adding
these workflows does not establish that a hosted GPU run has passed or that merge
protection is enabled.

Run the identical lane locally in a Git checkout:

```sh
OPENBLAS_NUM_THREADS=4 python3 scripts/run_pi05_reference_gate.py \
  --board thor --variant fp8_static \
  --checkpoint /models/pi05/model.safetensors \
  --calibration /models/pi05/calibration.json \
  --output devlocal/pi05-cpu-reference/ci-local-thor-fp8
```

Every run builds current source, recomputes its CPU reference, saves source/binary
identity, a copy and digest of the qualified recipes, and numerical reports, then
applies the policy. CI uses the tracked board-specific recipe banks with autotuning
disabled. To qualify a new bank deliberately, supply `--tuning PATH --autotune`;
use an empty directory to retune every operation. Preserve the resulting recipes
and rerun without autotuning before promoting them. Existing output directories are rejected to preserve evidence. The
initial measurements above precede the main-branch CUDA migration merged at
`9a1dbce7`; the source revision in each new report identifies what was actually
executed. Synthetic corpus coverage complements, but does not replace, real
image/state/view coverage or task-success evaluation.

### Post-merge GPU verification (2026-10-30)

Merged production source at `9a1dbce7` with the committed reference-gate example
was compiled on Thor and Orin. Fresh CPU outputs on both boards are bitwise equal.
With the [qualified recipes](../crates/apxinf-model/tests/fixtures/pi05_gpu/recipes/README.md),
all three required paths reproduce their initial per-case cosine measurements:

| Required path | T10 cosine | T21 cosine | Result |
| --- | ---: | ---: | --- |
| Thor BF16 | 0.999991120 | 0.999917999 | Pass |
| Thor FP8 | 0.997492755 | 0.991482836 | Pass |
| Orin BF16 | 0.999977623 | 0.999917677 | Pass |

Eager/graph max-abs is zero and graph repeats are bitwise equal. A fresh Orin
autotune initially selected different GEMM tactics and produced cosine
0.999938083 / 0.999796691, failing the unchanged floors. Holding source, CPU
reference and inputs fixed, restoring three previously qualified plain-GEMM
choices reproduced the original results. CI therefore freezes the measured
recipes instead of timing/selecting algorithms on every run. Recipe qualification
and performance optimization must continue to satisfy the same cosine floors;
thresholds were not lowered. The advisory Orin INT8 run reproduces cosine
0.509397173 / 0.969470443; meeting that regression floor does not imply adequate
quantized accuracy. Native recipe namespaces include build/library/device
identity. After changing those dependencies, qualify the bank again: stale recipes
may miss and fall back, and are not evidence that the intended tactics ran.

### Recorded numerical validation (2026-10-27)

The Rust fallback and pthreads OpenBLAS (1 and 4 threads)
were locally checked on macOS arm64 with a small independent PyTorch
implementation using unchanged norm/RoPE helpers from OpenPI revision
`215abfb217dbac7d5f1273282331b9b1866c0479`.

Full-checkpoint verification on Linux aarch64 (Thor CPU) also compared the public
native RGB binding against that revision's unchanged `PI0Pytorch`, configured
with `dtype="float32"` and running on CPU. Both loaded the PI0.5 LIBERO checkpoint
with SHA256 `21b8711787c4a75861b02cff6aa81675a3a943d32b435a68262ac4461e476ba4`.
The reference used PyTorch 2.9.1 and Transformers 4.53.2; native GEMM used
OpenBLAS 0.3.26 pthreads with four threads. Each case used two 224x224 views,
50x32 FP32 initial noise, and ten flow steps. The predetermined elementwise
criterion was `atol=rtol=2e-4`.

| Input | Tokens | Maximum absolute error | Relative L2 error |
| --- | ---: | ---: | ---: |
| Gradient images, first prompt | 10 | 6.03e-6 | 3.96e-6 |
| Gradient images, second prompt | 21 | 1.02e-5 | 7.15e-6 |
| Recorded DROID images, first frame | 8 | 1.40e-4 | 1.82e-5 |
| Recorded DROID images, later frame | 8 | 3.04e-6 | 1.61e-6 |

All four cases passed. Replaying the first input after the other cases was
bitwise stable in both implementations. The DROID images are numerical input
coverage for this LIBERO checkpoint, not a matched robot-policy evaluation.
These results do not measure LIBERO task success, quantized CUDA equivalence,
or production CPU latency.


The CPU Python `AutoPolicy.from_pretrained` path was exercised with two fixed
observations through preprocessing and normalized/returned action outputs.
Provided-noise repeats, seeded inference repeats, invalid-RGB rejection and
recovery passed. This checkpoint lacks statistics for some transforms, so the
existing policy uses identity passthrough for them; this validates integration,
not embodiment normalization accuracy. Linux OpenBLAS core/storage tests
(33 tests) and the four regular CPU public-contract tests also passed. The
CPU extension links OpenBLAS without CUDA or OpenMP runtime dependencies.

The four full-checkpoint cases and two Python policy cases were rerun after
CPU model isolation and backend extraction, with bitwise-identical outputs.
GPU hardware execution was not repeated after backend extraction. These are
recorded validation results; the commands above do not reproduce the private
full-checkpoint fixtures.

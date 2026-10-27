# PI0.5 CPU reference

PI0.5 supports `Device::Cpu` with `model_variant="fp32_reference"` (also selected
by `auto` on CPU). Its self-contained `cpu_pi05` implementation executes unfused
FP32 operations, without a CUDA toolkit or GPU. This is a numerical reference path;
it does not emulate BF16/FP8/INT8 rounding or promise production CPU latency.

## Build and run

Linux with a system multithreaded OpenBLAS installation:

```bash
cargo test -p apxinf-model --features apxinf-core/openblas --test cpu_pi05
```

macOS with Apple Accelerate:

```bash
cargo test -p apxinf-model --features apxinf-core/accelerate --test cpu_pi05
```

Choose one BLAS feature. Without either feature, GEMM uses the serial Rust
fallback, useful for small tests but very slow for full PI0.5. For OpenBLAS
outside the system library search path, set `OPENBLAS_LIB_DIR` to its library
directory; runtime dynamic-library lookup must also be configured appropriately.
The root crate and Python extension expose `openblas` and `accelerate` features.

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
- `apxinf-core` owns generic FP32 math. Dense GQA/MHA packs one head at a time
  and uses BLAS GEMM for both QK-transpose and probability-times-V, with a stable
  Rust softmax. It is non-causal; image views are independent batches, prefix
  queries see the whole prefix, and action queries see prefix plus all action
  tokens. No prefix query sees action tokens.

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

The maintained CPU tests cover public loading, RGB/patch layout equivalence,
changed RNG inputs, retained outputs/plans, eager readiness, graph rejection and
invalid requests. Core tests compare dense grouped attention to scalar FP64
arithmetic. CPU intermediate-stage tests exercise the port's own model.

An ignored end-to-end test accepts a private tiny-model checkpoint/fixture via
`APXINF_PI05_CPU_REFERENCE`; a companion ignored unit test compares vision,
embedded prefix, each prefix K/V layer and every flow step. These fixtures use
the fixed small geometry in the tests and are not production checkpoints.
Generated checkpoints, independent reference code and results stay under
`devlocal/pi05-cpu-reference/`, outside Git.

The Rust fallback, Apple Accelerate, and pthreads OpenBLAS (1 and 4 threads)
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


Before backend isolation, native CUDA compilation and a BF16 regression on Thor passed. The same
four full-size inputs produced bitwise-identical outputs between this change
and main baseline `e9da4bd3967e18331cf13ffc44bded61a6b32075`; both builds were
bitwise stable across consecutive repeats. This is historical evidence for the earlier revision; the isolated CPU port
restores the CUDA model directory exactly to main.

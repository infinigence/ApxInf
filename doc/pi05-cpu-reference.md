# PI0.5 CPU reference

PI0.5 supports `Device::Cpu` with `model_variant="fp32_reference"` (also selected
by `auto` on CPU). It executes the existing `Pi05Model` dataflow with unfused
FP32 Blocks, without a CUDA toolkit or GPU. This is a numerical reference path;
it does not emulate BF16/FP8/INT8 rounding or promise production CPU latency.

## Build and run

Linux with a system multithreaded OpenBLAS installation:

```bash
cargo test -p apxinf-model --features apxinf-core/openblas --test pi05_cpu
```

macOS with Apple Accelerate:

```bash
cargo test -p apxinf-model --features apxinf-core/accelerate --test pi05_cpu
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

- `cpu_load.rs` selects CPU assets and assembles the model and runner. CUDA
  variant resolution remains in `load.rs`.
- `weights/cpu.rs` converts unquantized F32/F16/BF16 checkpoint tensors to FP32
  **before** host checkpoint transformations, preserving the FP32 Gemma norm
  offset and scale-folding calculation. It validates CPU-consumed weight shapes.
- `model/blocks/cpu.rs` implements vision, prefix KV, adaptive normalization,
  attention, gated MLP and Euler updates. The shared `model/mod.rs` retains
  forward order and the flow loop. `Blocks::TokenIds` is `[u32]` on CPU and a
  device buffer on CUDA; `model/cuda.rs` contains CUDA preparation contracts.
- `model_runner/cpu.rs` owns fixed timestep modulation, request validation,
  host RGB patchification, noise generation and explicit eager plans. Output
  storage is newly allocated per call and remains valid after subsequent calls.
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
arithmetic. The original shared-flow tests now run without CUDA.

An ignored end-to-end test accepts a private tiny-model checkpoint/fixture via
`APXINF_PI05_CPU_REFERENCE`; a companion ignored unit test compares vision,
embedded prefix, each prefix K/V layer and every flow step. These fixtures use
the fixed small geometry in the tests and are not production checkpoints.
Generated checkpoints, independent reference code and results stay under
`devlocal/pi05-cpu-reference/`, outside Git.

The Rust fallback, Apple Accelerate, and pthreads OpenBLAS (1 and 4 threads)
were locally checked on macOS arm64 with a small independent PyTorch
implementation using unchanged norm/RoPE helpers from OpenPI revision
`215abfb217dbac7d5f1273282331b9b1866c0479`. This evidence does not establish parity
for a trained full-size checkpoint, a particular robot policy, quantized CUDA
execution or GPU performance. Those need the intended checkpoint/reference
inputs and the relevant execution host.

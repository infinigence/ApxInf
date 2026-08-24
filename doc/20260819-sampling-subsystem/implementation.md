# Sampling Subsystem Implementation

Date: 2026-08-19
Status: Implemented

## Source map

| Area | Source |
|---|---|
| Public contracts, CPU sampler, Philox, CPU normals | `crates/apxinf-core/src/sampling.rs` |
| Backend supertrait integration | `crates/apxinf-core/src/backend.rs` |
| CPU backend factory | `crates/apxinf-core/src/op_impls/cpu.rs` |
| CUDA sampler and normal-generator wrappers | `crates/apxinf-cuda/src/sampling.rs` |
| CUDA kernels | `crates/apxinf-cuda/kernels/custom/sampling.cuh` |
| CUB workspace queries and launch policy | `crates/apxinf-cuda/adapters/sampling_adapter.cu` |
| CUDA ABI | `crates/apxinf-cuda/src/ffi/custom.rs` |
| Layered generation settings / JSON loader | `crates/apxinf-model/src/generation_config.rs` |
| Generation driver | `crates/apxinf-model/src/llm_trait.rs` |
| Unified model frontend | `crates/apxinf-model/src/auto.rs` |
| Llama/Qwen device-logit integration | `crates/apxinf-model/src/llama/`, `crates/apxinf-model/src/qwen3vl/` |
| Model-neutral VLA request | `crates/apxinf-model/src/vla/mod.rs` |
| PI0.5 prepared runtime integration | `crates/apxinf-model/src/pi05/vla_runtime.rs` |
| Python compatibility and seeded APIs | `crates/apxinf-py/src/lib.rs` |
| CLI sampling options | `src/main.rs` |

## CPU implementation

`CpuTokenSampler` owns a `Vec<u32>` occurrence table, request length, cloned
parameters, and current RNG key. Each sample converts the selected logits row
to f32, applies penalties, and either scans for greedy selection or sorts a
`(token_id, adjusted_logit)` vector for random selection.

The CPU path is the readable correctness reference for CUDA. It deliberately
uses the same operation order, tie rule, Philox mapping, nucleus boundary, and
log-probability definition.

`CpuNormalGenerator` owns the output `Tensor` and mutates its existing storage
in place. It supports f32, f16, and bf16. Half-precision values are written as
two-byte bit patterns rather than casting a byte allocation to an aligned half
slice.

## Counter-based random generation

CPU and CUDA implement Philox4x32-10 with the Random123 zero-vector test as a
fixed reference. A SplitMix64-derived stream value incorporates request and
draw identity. Uniform values use 23 random mantissa bits and remain strictly
inside `(0, 1)`.

Standard-normal values use Box-Muller in pairs:

```text
r = sqrt(-2 ln(u1))
theta = 2 pi u2
z0 = r cos(theta)
z1 = r sin(theta)
```

The CUDA generator writes f32, f16, or bf16 directly into the supplied device
tensor. No temporary host noise vector is created.

## CUDA categorical path

`CudaTokenSampler` allocates persistent buffers when the request sampler is
created:

- occurrence counts;
- adjusted logits and initial token IDs;
- sorted logits and token IDs;
- softmax weights and cumulative distribution;
- greedy partial values and token IDs;
- CUB radix-sort and inclusive-scan workspaces; and
- one 16-byte `ApxInfSamplingOutput`.

There are seven full-vocabulary four-byte arrays, so fixed array storage is
approximately `28 * vocab_size` bytes, plus CUB workspaces and small reduction
buffers. A 151,669-token vocabulary therefore uses about 4.25 MB before CUB
workspace. Allocations happen once per generation request, not per token.

### Prepare kernel

One grid-stride kernel reads f32, f16, or bf16 logits, applies the full penalty
and temperature policy in f32, and initializes ascending token IDs. NaNs become
negative infinity; positive infinity becomes `FLT_MAX`.

### Fast greedy path

When greedy selection does not request a log-probability, a two-stage
multi-block reduction selects `(maximum logit, lowest token ID)`. The first
stage covers the vocabulary across up to 1024 blocks; the second reduces block
results and atomically updates the selected token's history count.

### Random and log-probability path

Random selection, and greedy selection when a log-probability is requested,
use:

1. `cub::DeviceRadixSort::SortPairsDescending`;
2. a fused exponential-weight kernel that also applies the top-k limit;
3. `cub::DeviceScan::InclusiveSum`;
4. a one-thread binary search for the top-p boundary and random target; and
5. a compact output write plus history update.

CUB's stable sort preserves ascending token IDs for equal keys, matching the
CPU tie rule. The host copies only the 16-byte output structure, which also
provides the synchronization needed before EOS/callback processing.

## Device-logit integration

Llama and Qwen3-VL now return their backend-resident logits instead of copying
the full vocabulary row to the CPU. Their decode graphs expose the stable
logits allocation through the bounds-checked `CudaBuffer::as_tensor` view.

Graph replay, logits preparation, and selection are enqueued on the same CUDA
stream. The old synchronization immediately after replay was removed; the
small result copy is the steady-state synchronization point.

The generation driver constructs one sampler, initializes prompt counts once,
and calls it for both the prefill row and each decode row. Model-specific code
contains no temperature, penalty, filtering, RNG, or token-selection logic.

## VLA implementation

Each prepared PI0.5 plan creates its latent tensor before graph capture and
binds one normal generator to a clone of that allocation. FP8 uses an f16
latent; BF16 and W8A8 use bf16.

Graph variants now have two update paths:

- compatibility updates copy patches/RGB, tokens, and a provided latent;
- generated updates copy only patches/RGB and tokens, then fill the captured
  latent allocation on the CUDA stream.

Eager fallback follows the same `InitialLatent` dispatch. Python `infer_rgb` and
`_infer_patches` construct `VlaRequest::provided` when their optional `noise`
argument is present. When it is absent they advance the model handle's implicit
draw counter and construct `VlaRequest::generated`; explicit seeded methods do
the latter with the caller's complete `RngKey`.

The default `Pi05Policy` input pipeline no longer contains a host
`sample_noise` step. It forwards explicit keyword/observation noise when given
and otherwise delegates generation to the binding. A caller can still install
a `GaussianNoise` processor to preserve or customize the old host path.

## Generation defaults and CLI

`AutoModel` reads `generation_config.json` once for text/VLM models and stores
the resulting partial `GenerationOptions` beside the loaded model. VLA models,
including PI0.5, skip this path. At request entry, ApxInf defaults, model
defaults, deployment overrides, and request options are merged and normalized
into the crate-private `ResolvedGenerationOptions`.

The CLI uses model defaults automatically. Random or greedy selection can also
be forced explicitly:

```bash
cargo run --release --features cuda-no-nvtx -- generate \
  --model /path/to/model \
  --prompt "Describe CUDA graphs." \
  --device cuda --dtype bf16 --max-tokens 50 \
  --sample --temperature 0.8 --top-k 40 --top-p 0.95 \
  --repetition-penalty 1.1 --frequency-penalty 0.0 \
  --presence-penalty 0.0 --seed 42
```

`--generation-config auto|apxinf|PATH` selects the default source and
`--override-generation-config JSON` supplies deployment-level overrides.

## Current optimization opportunities

- Allocate sort/scan buffers lazily for requests that use only greedy without
  log-probabilities.
- Replace full radix sort with a specialized top-k selection when `k` is small.
- Add a batched sampler with one history/RNG stream per sequence.
- Fuse common penalty and top-k cases further when profiling shows a benefit.
- Capture the steady sampling sequence or combine it with decode scheduling if
  launch overhead becomes measurable.
- Keep sample results on device in a future device-resident scheduler; the
  current host callback/EOS loop requires the compact result transfer.

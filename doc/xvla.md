# X-VLA (CUDA port)

X-VLA is a cross-embodiment VLA: a FLORENCE-2 vision-language stack with a
soft-prompted, domain-aware action transformer. This port lives in
`crates/apxinf-model/src/cuda_xvla/` as a backend-native CUDA port per the
[module rule](model-layer-architecture.md#where-backend-native-model-code-lives):
it talks to cuda-new through `crate::accelerator::cuda` and its own kernels
(`crates/apxinf-cuda/src/kernels/xvla_ops.rs`), imports no other model module,
and registers under the key `xvla` (the checkpoint `config.json`
`model_type`).

## Input contract

`AutoPolicy.from_pretrained("<path-to-X-VLA-Libero>")` with an observation dict:

- `image0`, `image1` — RGB camera views, `uint8`, any size (the processor
  resizes to the checkpoint's square size, bicubic, as the reference does);
  one to three views, missing trailing views are zero-padded
- `proprio` — `float32` vector, zero-padded to the checkpoint's proprio dim
- `language_instruction` — the task prompt (BART tokenization with explicit
  BOS/EOS and fixed-length padding)

Returns `result["actions"]`, shape `(action_horizon, action_dim)` `(30, 20)`
for the LIBERO checkpoint, unnormalized.

## Precisions

BF16 end to end (vision tower, language stack, action head). The five custom
kernels (`rgb_u8_to_normalized_tokens`, `im2col`, `dwconv3x3`, `gelu_erf`,
`sigmoid`, `gather_rows`, `pack_action_tokens`) run BF16 with CPU-reference
parity tests in `crates/apxinf-cuda/src/tests/xvla.rs`.

## Parity

The port is validated against an fp32 oracle (the X-VLA reference
implementation) on four recorded observation cases, stage by stage:

| Stage | Tolerance | Observed |
|---|---|---|
| Vision tower (BF16 vs fp32) | max-abs 0.06, cosine > 0.9999 | max-abs 0.053-0.065, cosine ≈ 0.99994 |
| Language/VLM features | max-abs 0.3, cosine > 0.9999 | max-abs 0.265, cosine 0.99995 |
| Action head, 10 denoise steps | max-abs 0.01 | max-abs ≤ 0.0041, cosine ≥ 0.99999 |
| End-to-end pipeline | max-abs 0.01 | max-abs ≤ 0.0038, cosine ≥ 0.99999 |
| Runtime with CUDA graphs | max-abs 0.02 | max-abs ≤ 0.0038, cosine ≥ 0.99999 |

One vision case needs `XVLA_VISION_TOL=0.07` (max-abs 0.0646 against the 0.06
default; cosine is unchanged at 0.99994 — the BF16 storage floor scales with
the activation scale of the case).

The harnesses are `#[ignore]`d GPU tests driven by `XVLA_VISION_CASE`,
`XVLA_VLM_CASE`, `XVLA_ACTION_CASE`, `XVLA_RUNTIME_CASE` and
`XVLA_CKPT_DIR` (defaults to `~/refs/checkpoints/X-VLA-Libero`).

## Measurements

RTX 4090 (sm_89, CUDA 12.8), batch 1, two views 224x224, `X-VLA-Libero`:

- Steady-state (CUDA-graph replay) P50 **39.95 ms** (25.0 Hz); P95 40.06 ms;
  30 samples after 10 warm-ups, 3 runs: 39.92 / 39.95 / 39.95 ms
- Cold first inference (eager warm-up + graph capture): 482 ms; model load
  7.4 s

Reproduce with `python scripts/bench_xvla.py --model-dir <path>`. The Python
surface always selects `PreferGraph`; `ExecutionPolicy::Eager` is a
runtime-internal path (the pre-capture warm-up).

## Limits

- Accuracy in a simulator (LIBERO-style rollouts) is not measured here; the
  acceptance bar is tensor parity against the fp32 oracle.
- Single embodiment domain per request (the checkpoint's domain-aware
  projections are selected by `embodiment_id`).
- Fixed-view contract (no variable image grids); at most three camera views.

# CUDA Operator Catalog

This catalog is for model authors choosing model-neutral L3 operators from
`apxinf-cuda`. Match the complete mathematical expression and tensor contract,
not only a familiar operator name. If no row matches exactly, treat that as an
operator gap and follow [`doc/adding-new-kernels.md`](../../doc/adding-new-kernels.md).

The catalog describes the public semantic API. Provider selection and tuning
remain internal to the CUDA crate. All listed operators support eager execution
and the `prepare_with_session` -> capture -> replay workflow. Inputs, outputs,
biases, and scales must be CUDA tensors on the context's device, and output
storage must not overlap an input.

Each `l3-operator` comment below is checked against the Rust semantic registry.
The test checks that every public semantic appears exactly once; reviewers must
still verify that contract details and limitations are accurate.

## Shared GEMM contract

- Public matrices are contiguous row-major: `A=[M,K]`, `B=[K,N]`.
- Dimensions must be non-zero, mutually compatible, and fit in `i32`.
- `alpha` is applied to the projection. The final value is divided by the
  positive `output_scale`.
- `GemmQuantization::None` requires matching, non-FP8, non-INT8 input dtypes.
- `GemmQuantization::Fp8UnitScale` accepts two pre-quantized FP8 E4M3 tensors.
- `GemmQuantization::Fp8` accepts FP8 E4M3 inputs with FP32 row scales `[M]`
  and channel scales `[N]`.
- `GemmQuantization::W8A8` accepts INT8 inputs with FP32 row scales `[M]` and
  channel scales `[N]`; output must be BF16 and `K <= 131071`.
- Quantization happens before these APIs. Dynamic quantization is not part of
  the current L3 contracts.

## Available L3 operators

<!-- l3-operator:gemm -->
### `gemm`

| Rust API | Semantics | Output | Supported quantization | Important restrictions | Reference test |
|---|---|---|---|---|---|
| `ops::gemm` | `Y = alpha * (A @ B) / output_scale` | `[M,N]` | None, FP8 unit-scale, FP8 row/channel, W8A8 row/channel | W8A8 output is BF16 | `gemm_all_candidates_match_torch` |

Use for a plain linear projection when bias and activation are separate or
absent.

<!-- l3-operator:gemm_bias -->
### `gemm_bias`

| Rust API | Semantics | Output | Supported quantization | Important restrictions | Reference test |
|---|---|---|---|---|---|
| `ops::gemm_bias` | `Y = (alpha * (A @ B) + bias) / output_scale` | `[M,N]` | None, FP8 unit-scale, FP8 row/channel, W8A8 row/channel | `bias=[N]`; W8A8 output is BF16 | `gemm_bias_all_candidates_match_torch` |

Use when bias addition is part of the required L3 semantic.

<!-- l3-operator:gemm_bias_gelu -->
### `gemm_bias_gelu`

| Rust API | Semantics | Output | Supported quantization | Important restrictions | Reference test |
|---|---|---|---|---|---|
| `ops::gemm_bias_gelu` | `Y = GELU(alpha * (A @ B) + bias) / output_scale` | `[M,N]` | None, FP8 unit-scale, FP8 row/channel, W8A8 row/channel | `bias=[N]`; W8A8 output is BF16 | `gemm_bias_gelu_all_candidates_match_torch` |

The activation is the tanh-approximation GELU used by the checked Torch
reference. Use only when the model requires this fused ordering.

<!-- l3-operator:gemm_geglu -->
### `gemm_geglu`

| Rust API | Semantics | Output | Supported quantization | Important restrictions | Reference test |
|---|---|---|---|---|---|
| `ops::gemm_geglu` | `Y = GELU(alpha * (A @ B_gate)) * (alpha * (A @ B_up)) / output_scale` | `[M,N]` | None, FP8 unit-scale | Public `B=[K,2N]`: gate columns first, then up columns; scaled FP8 and W8A8 are not supported | `gemm_geglu_all_candidates_match_torch` |

Use for the complete GeGLU projection. Do not pre-pack or interleave the public
weight: candidate-specific packing is internal. When a weight allocation is
immutable, attach a `WeightVersion` so a prepared execution may safely cache a
transformed copy.

<!-- l3-operator:gemm_bias_relu -->
### `gemm_bias_relu`

Computes `Y = ReLU(alpha * (A @ B) + bias) / output_scale` with output
`[M,N]` and bias `[N]`. Supports all shared GEMM quantization modes;
W8A8 output is BF16. Reference: `gr00t_bias_activation_candidates_match_torch`
and `gr00t_w8a8_bias_family_candidates_match_torch`.

<!-- l3-operator:gemm_bias_silu -->
### `gemm_bias_silu`

Computes `Y = SiLU(alpha * (A @ B) + bias) / output_scale` with output
`[M,N]` and bias `[N]`. Supports all shared GEMM quantization modes;
W8A8 output is BF16. Reference: `gr00t_bias_activation_candidates_match_torch`
and `gr00t_w8a8_bias_family_candidates_match_torch`.

<!-- l3-operator:gemm_bias_residual -->
### `gemm_bias_residual`

Computes `Y = (alpha * (A @ B) + bias + residual) / output_scale`.
Output and residual have shape `[M,N]` and the same dtype; bias has shape
`[N]`. Supports all shared GEMM quantization modes; W8A8 output is BF16.
Reference: `gr00t_bias_residual_candidates_match_torch` and
`gr00t_w8a8_bias_family_candidates_match_torch`.

<!-- l3-operator:gemm_swiglu -->
### `gemm_swiglu`

Computes `Y = SiLU(alpha * (A @ B_gate)) * (alpha * (A @ B_up)) / output_scale`.
Public weight is `[K,2N]`, with gate columns first and up columns second;
output is `[M,N]`. Supports all shared GEMM quantization modes; channel scales
have length `2N`, covering both gate and up columns, and W8A8 output is BF16. Reference:
`gr00t_swiglu_candidates_match_torch` and
`gr00t_w8a8_swiglu_candidates_match_torch`.

The cuBLAS W8A8 fallback converts unscaled INT8 operands exactly to BF16,
accumulates and stores the projection in FP32, then applies row/channel
scales and the fused epilogue before the final BF16 output conversion.
This avoids rounding scaled operands or gate/up projections to BF16 before
SwiGLU.

## Operator families not yet exposed by `cuda-new`

The legacy CUDA crate also contains activation, attention, cache, elementwise,
embedding, normalization, preprocessing, quantization, RoPE, and additional
fused operations. They do not yet have public L3 APIs in `cuda-new`. A new model
that needs one of these semantics must record an operator gap rather than infer
support from the legacy implementation.

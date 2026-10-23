# CUDA L3 Operator Catalog

This document lists the model-independent L3 semantics currently exposed by `apxinf-cuda-new`. Model code must match the complete mathematical semantic and tensor contract, not only the operator name. If no interface matches completely, record an operator gap and follow [`doc/adding-new-kernels.md`](../../doc/adding-new-kernels.md).

This document describes only public contracts and does not promise a specific provider, candidate, or autotune winner. The tunable operators listed here support eager execution and `prepare_with_session` → capture → replay. Fixed model-building kernels are not required to own a session or a graph; the surrounding model/phase captures their direct stream enqueues.

## Shared Constraints

- Input, output, bias, and scale tensors must reside on the CUDA device of the current `CudaContext`.
- Tensors use contiguous row-major layout; output storage must not overlap read-only inputs.
- Shapes must be nonempty, and dimensions passed to the native layer must not exceed `i32::MAX`.
- Policy fields such as workspace, graph-safe, and deterministic affect only candidate eligibility and the recipe; they do not change the L3 mathematical semantic.
- `l3-operator` comments are machine-readable markers. A unit test compares them with the Rust semantic metadata registered by each operator family to ensure that every public semantic appears exactly once; a new family must be added to that set.

## Representation Conversion

`ops::quantization(ctx, QuantizationArgs)` also owns shape-preserving representation conversions used between L3 operators. `CastF16ToBf16` converts F16 to BF16 and `CastBf16ToF32` converts BF16 to F32; both require identical nonempty input/output shapes and distinct device buffers. The latter is the standard boundary for a BF16 backbone feeding an FP32 readout GEMM.

## Shared GEMM Contract

GEMM uses `A=[M,K]` and `B=[K,N]`. `alpha` applies to the projection, and the final result is divided by a finite positive `output_scale`. `projection(A,B)` interprets inputs according to the following quantization contract:

- `None`: A/B have the same dtype, which is neither E4M3 nor INT8.
- `Fp8UnitScale`: A/B are both E4M3 tensors with the expected scaling already applied.
- `Fp8`: A/B are both E4M3; FP32 `row_scales=[M]` and `channel_scales=[N]` dequantize rows of A and columns of B, respectively.
- `W8A8`: A/B are both INT8 and use FP32 row/channel scales with the same shapes; output is BF16, `K <= 131071`, and the mode applies only to `gemm` and `gemm_bias`.
- `Nvfp4`: A/B are both packed FP4 (`DType::E2M1Pair`, two E2M1 values per byte, low nibble first) with one unsigned-E4M3 scale per `block_size` elements along K. Output is BF16, `K` is a multiple of `block_size`, and the mode applies only to `gemm`.

`Nvfp4` is the one contract whose operand storage is not the canonical `A=[M,K]`, `B=[K,N]`: both operands are K-packed, so `A` is stored `[M, K/2]` and `B` is stored `[N, K/2]`. That is the orientation a ModelOpt checkpoint already uses and the one the block-scaled kernel consumes, so requiring the usual layout would force a transpose that buys nothing. Any per-tensor scale a checkpoint carries alongside the block scales (ModelOpt's `weight_scale_2` and `input_scale`) multiplies the whole projection and therefore belongs in `alpha`, which also keeps it out of the tuning key.

Block-scale tensors must be in the layout the candidate reads, which is **not** a plain `[rows, K/block_size]` array. Build them once with `ops::nvfp4_pack_block_scales` into a buffer sized by `ops::nvfp4_scale_buffer_bytes`; the layout depends only on `block_size`, so the result stays valid when the autotuner selects a different configuration.

Quantization occurs before the API call; the current L3 contract does not include dynamic quantization. At least one registered candidate must still support the concrete spec.

<!-- l3-operator:gemm -->
### `gemm`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm(ctx, GemmArgs)` |
| Inputs | `A=[M,K]`, `B=[K,N]`; supports `None`, `Fp8UnitScale`, `Fp8`, `W8A8`, and `Nvfp4` (whose operands are stored `A=[M,K/2]`, `B=[N,K/2]`) |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = alpha * projection(A,B) / output_scale` |
| Constraints | W8A8 and NVFP4 output must be BF16; `WeightVersion` may declare immutable weights and allow prepare to cache an internal prepacked copy |
| Reference test | `gemm_all_candidates_match_torch`, `nvfp4_all_candidates_match_reference` |

<!-- l3-operator:gemm_bias -->
### `gemm_bias`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_bias(ctx, GemmBiasArgs { gemm, bias })` |
| Inputs | `A=[M,K]`, `B=[K,N]`, `bias=[N]`; supports all four GEMM quantization contracts |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = (alpha * projection(A,B) + bias) / output_scale`, with bias broadcast across M |
| Constraints | bias dtype matches the projection dtype; W8A8 output must be BF16 |
| Reference test | `gemm_bias_all_candidates_match_torch` |

<!-- l3-operator:gemm_bias_residual -->
### `gemm_bias_residual`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_bias_residual(ctx, GemmBiasResidualArgs { gemm, bias, residual })` |
| Inputs | `A=[M,K]`, `B=[K,N]`, optional `bias=[N]`, `residual=[M,N]`; supports `Fp8UnitScale` only |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = alpha * projection(A,B) + bias + residual`, with bias broadcast across M |
| Constraints | Requires unit-scale FP8 inputs and F16 output; bias and residual dtypes match the projection dtype; the residual must not alias the output |
| Reference test | `zero_weight_language_layer_is_residual_identity` (model-level; candidate parity is covered by the tuner's capture-replay output validation) |

<!-- l3-operator:gemm_bias_gelu -->
### `gemm_bias_gelu`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_bias_gelu(ctx, GemmBiasGeluArgs { gemm, bias })` |
| Inputs | `A=[M,K]`, `B=[K,N]`, `bias=[N]`; supports `None`, `Fp8UnitScale`, and `Fp8` |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = GELU_tanh(alpha * projection(A,B) + bias) / output_scale` |
| Constraints | bias dtype matches the projection dtype; GELU uses the Torch reference's tanh approximation; W8A8 is unsupported |
| Reference test | `gemm_bias_gelu_all_candidates_match_torch` |

<!-- l3-operator:gemm_geglu -->
### `gemm_geglu`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_geglu(ctx, GemmGegluArgs { gemm })` |
| Inputs | `A=[M,K]`, `B=[K,2N]`; the first N columns of B are `B_gate`, and the last N columns are `B_up`; supports `None` and `Fp8UnitScale` |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = GELU_tanh(alpha*(A@B_gate)) * (alpha*(A@B_up)) / output_scale` |
| Constraints | The second dimension of B is even; FP8 with row/channel scales and W8A8 are unsupported; candidate-specific packing may occur only internally |
| FP8 candidates | Vendor decomposition (FP16 projection for E4M3 output); dual-GeGLU at M522/M533; SM100-family split-GeGLU at M778, 2N32768, K2048; eligible decomposed candidates remain available to autotune |
| Reference tests | `gemm_geglu_all_candidates_match_torch`, `gemm_geglu_three_view_all_candidates_match_torch` (unit/non-unit scales, eager/graph) |

## Shared Attention Contract

Attention computes `softmax(mask(scale * (Q @ K^T))) @ V`. Q/K/V have the same dtype, and `scale` is finite and positive with a default of `1/sqrt(head_dim)`. Dense and KV-cache support MHA, GQA, and MQA and require `query_heads % kv_heads == 0`.

<!-- l3-operator:attention -->
### `attention`

| Item | Contract |
| --- | --- |
| Rust API | `ops::attention(ctx, AttentionArgs)` |
| Inputs | `Q=[B,Tq,Hq,D]`, `K/V=[B,Tk,Hkv,D]`; Q/K/V share F16 or BF16 dtype; mask is `None` or `Causal` |
| Output | `Y=[B,Tq,Hq,D]`; ordinary output uses the input dtype, and F16 input may also be written as E4M3 |
| Mathematical semantic | dense scaled dot-product attention |
| Constraints | Causal requires `Tk>=Tq`, with queries aligned to the final positions of the key sequence; ordinary output requires `output_scale=1`; E4M3 stores `round_to_e4m3(attention/output_scale)` |
| Reference test | `attention_all_candidates_match_reference` |

<!-- l3-operator:kv_cache_attention -->
### `kv_cache_attention`

| Item | Contract |
| --- | --- |
| Rust API | `ops::kv_cache_attention(ctx, KvCacheAttentionArgs)` |
| Inputs | `Q=[B,Tq,Hq,D]`, `K_cache/V_cache=[B,key_capacity,Hkv,D]`; all share F16 or BF16 dtype; mask is `None` or `Causal`; one model-owned `KvCacheDecodeMeta` supplies `[valid_key_tokens, query_start]` from a fixed device-visible address |
| Output | `Y=[B,Tq,Hq,D]`, with the same dtype as the inputs |
| Mathematical semantic | scaled dot-product attention from the query to the first `valid_key_tokens` rows of the cache |
| Constraints | `0<valid_key_tokens<=key_capacity`; when causal, token i is at `query_start+i`, and `query_start+Tq<=valid_key_tokens` is required |
| Graph/tuning identity | For dynamic decode, `key_tokens` in the Spec equals fixed `key_capacity`; exact valid length and query position live only in `decode_meta`, so one execution and graph cover all positions within capacity |
| Reference test | `kv_cache_attention_all_candidates_match_reference` |

<!-- l3-operator:segmented_attention -->
### `segmented_attention`

| Item | Contract |
| --- | --- |
| Rust API | `ops::segmented_attention(ctx, SegmentedAttentionArgs)` |
| Inputs | Q/K/V are all `[total_tokens,H,D]` with the same dtype (F16 or BF16); device U32 offsets match the contents of `host_offsets` |
| Output | `Y=[total_tokens,H,D]`, with the same dtype as the inputs |
| Mathematical semantic | independent non-causal self-attention for each segment of a packed token sequence |
| Constraints | offsets contain at least two elements, are monotonically nondecreasing, begin at 0, and end at `total_tokens`; empty segments are allowed; causal and differing Q/KV head counts are unsupported |
| Reference test | `segmented_attention_all_candidates_match_reference` |

## Testing Responsibilities

The catalog test ensures only that semantics are neither missing nor duplicated; it cannot validate the written contracts. A new L3 semantic must also add a semantic test to `src/ops/tests/l3_behavior.rs` and an independent-reference all-candidate numerical test to `src/ops/tests/precision/precision.rs`.

Tests must run through `crates/apxinf-cuda-new/test-new.sh`; a normal `cargo test` from the repository root does not automatically test this crate.

<!-- l3-operator:Unfold3d -->
### Unfold3d

`PreparedUnfold3d::new` binds a nonempty BF16 THWC volume, optional preceding
THWC history, asymmetric zero padding, positive three-axis kernel/stride and a
contiguous tile of flattened output rows. Columns are ordered kernel T/H/W/C.
This preserves causal history without a host readback or a full im2col allocation;
the caller tiles rows and composes ordinary GEMM. Output must not alias either
source. `enqueue` performs exact BF16 copies with zero padding, allocates no
memory and can be captured after preparation. Captured graphs retain the prepared execution and its
input/output storage even after caller handles are dropped. The single deterministic baseline
uses no device workspace or persistent recipe. Supported dtype is BF16 only.
Target-device numerical/capture acceptance is required before a model claims it.

<!-- l3-operator:ChannelL2Norm -->
### ChannelL2Norm

`channel_l2_norm` accepts contiguous BF16 `[rows, channels]` and learned BF16
`[channels]` weights. It computes the L2 norm in FP32, rounds the norm to BF16,
clamps it by epsilon (normal positive FP32), divides and rounds to BF16, multiplies
by `sqrt(channels)` and rounds again, then multiplies by the learned weight.
The output is BF16 with the input shape. Epsilon is a lower bound on the norm,
not an additive variance epsilon. This distinct numerical semantic preserves
channel normalization used by causal video autoencoders. The fixed kernel has
no selection, workspace or allocation; it supports eager and captured launches.

<!-- l3-operator:Multiply -->
### Multiply

`PointwiseSemantic::Multiply` multiplies same-shaped contiguous `[rows, cols]`
input and secondary tensors, with a single round to the output dtype. F16, BF16
and F32 are supported; all three tensors must have the same dtype. No bias,
activation or quantization scale is accepted. Exact in-place operation is
supported. The existing EulerUpdate semantic also accepts F32 for a full-precision
state boundary; F32 also requires identical shapes and rejects partial aliasing.
Its formula and existing F16/BF16 behavior are unchanged.

<!-- l3-operator:RgbToPatchesStaged -->
### RgbToPatchesStaged

`GatherSemantic::RgbToPatchesStaged` uses the existing RGB patch geometry and
layouts, with a distinct normalization recipe: cast the byte to output F16 or
BF16, divide by 127.5 and round to that dtype, subtract 1 and round again.
The existing `RgbToPatches` single-round recipe is unchanged. The fixed kernel
has no workspace or selection; input bytes and output buffers remain device
resident during capture. Exhaustive byte rounding is checked in
`tests/mixed_precision_ops.rs`.

<!-- l3-operator:F64Recurrence -->
### F64Recurrence

`F64RecurrenceState` owns opaque FP64 state and previous-input storage tied to
one device stream. `prepare_reset` converts matching FP32 input to FP64 state,
zeros history and publishes FP32 output. `prepare_update` computes
`a*state + d*(b*input + c*previous_input)` with separate FP64 rounds, stores the
FP64 state/current input, and publishes rounded FP32 output. Coefficients must
be finite. Inputs and output have identical nonempty shapes; exact in-place
output is supported, partial aliasing is rejected. Call reset at every independent
sequence boundary. Prepared executions and captured graphs retain all storage.
This single fixed implementation has no recipe/tuning; enqueue allocates nothing.
`tests/mixed_precision_ops.rs` checks retained FP64 precision and graph ownership.

<!-- l3-operator:IndexedColumnMean -->
### IndexedColumnMean

`PreparedIndexedMean::new` accepts an FP32 input tensor, FP32 output and a static
row-major host index matrix `[rows, output.numel()]`. It validates and uploads
indices once, then computes each output column's indexed arithmetic mean on GPU.
Every index must be in bounds; output cannot overlap input. Rows are bounded by
2^24 so the FP32 divisor remains exact. The deterministic baseline accumulates
in FP32 with a 128-thread tree; its summation order is part of this implementation,
not a promise of bitwise identity with arbitrary framework reductions. Graphs
retain the input, index map, output and execution; enqueue has no allocation,
host transfer or tuning. Noncontiguous selection and graph lifetime are tested
in `tests/mixed_precision_ops.rs`.

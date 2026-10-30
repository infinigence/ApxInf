# CUDA L3 Operator Catalog

This document lists the model-independent L3 semantics currently exposed by `apxinf-cuda-new`. Model code must match the complete mathematical semantic and tensor contract, not only the operator name. If no interface matches completely, record an operator gap and follow [`doc/adding-new-kernels.md`](../../doc/adding-new-kernels.md).

This document describes only public contracts and does not promise a specific provider, candidate, or autotune winner. The tunable operators listed here support eager execution and `prepare_with_session` → capture → replay. Fixed model-building kernels are not required to own a session or a graph; the surrounding model/phase captures their direct stream enqueues.

## Shared Constraints

- Input, output, bias, and scale tensors must reside on the CUDA device of the current `CudaContext`.
- Tensors use contiguous row-major layout; output storage must not overlap read-only inputs.
- Shapes must be nonempty, and dimensions passed to the native layer must not exceed `i32::MAX`.
- Policy fields such as workspace, graph-safe, and deterministic affect only candidate eligibility and the recipe; they do not change the L3 mathematical semantic.
- `APXINF_GEMM_REPORT=1` and `APXINF_ATTENTION_REPORT=1` print preparation-only implementation summaries, including `source=recipe` for disk restoration and `source=memory-recipe` for same-process reuse. These report choices, not tensor contents, and add no enqueue logging. Unchanged recipe files alone do not prove restoration.
- `l3-operator` comments are machine-readable markers. A unit test compares them with the Rust semantic metadata registered by each operator family to ensure that every public semantic appears exactly once; a new family must be added to that set.

## Representation Conversion

`ops::quantization(ctx, QuantizationArgs)` also owns shape-preserving representation conversions used between L3 operators. `CastF16ToBf16` converts F16 to BF16 and `CastBf16ToF32` converts BF16 to F32; both require identical nonempty input/output shapes and distinct device buffers. The latter is the standard boundary for a BF16 backbone feeding an FP32 readout GEMM.

## Shared GEMM Contract

`GemmPolicy::algorithm` defaults to `Any`, preserving unrestricted candidate
selection. `ColdBaseline` restricts the existing adapter to the first algorithm
that successfully prepares in its cold-fallback traversal, under the same Spec,
alignments, workspace, graph-safety and determinism constraints. Unsupported
configurations may be skipped; allocation/driver failures fail explicitly.
Online tuning still warms up, times and capture-checks this singleton domain
before persisting a recipe. It is not a search for a faster algorithm.
Restoration revalidates the same domain; another supported algorithm under its
key is a miss. Both the persistent recipe and session execution keys include
the policy. The GEMM-specific policy ABI is version 1 and requires Spec version
9; Attention's policy ABI is unchanged.

This preserves the chosen reduction order across cold, tuned and restored runs
within a compatible device/build/library/policy domain. It does not promise
bit-exact outputs across toolkit or device versions. `deterministic` alone
means repeatability within an implementation, not equal rounding between
implementations. Models sensitive to accumulated BF16 rounding must qualify
their whole outputs independently, and report the performance cost of this
restricted domain.

GEMM uses `A=[M,K]` and a weight whose physical interpretation is carried by
`GemmWeightLayout`: `Kn` stores `B=[K,N]`, `OutputMajor` stores the same logical
weight as `[N,K]`, and `GateUpPairs` stores the paired gate/up encoding accepted
only by `gemm_swiglu`. `alpha` applies to the projection, and the final result is
divided by a finite positive `output_scale`. `projection(A,B)` interprets inputs
according to the following quantization contract:

- `None`: A/B have the same dtype, which is neither E4M3 nor INT8.
- `Fp8UnitScale`: A/B are both E4M3 tensors with the expected scaling already applied.
- `Fp8`: A/B are both E4M3; FP32 `row_scales=[M]` and `channel_scales=[N]` dequantize rows of A and columns of B, respectively.
- `W8A8`: A/B are both INT8 and use FP32 row/channel scales with the same shapes; output is BF16, `K <= 131071`, and the mode applies only to `gemm` and `gemm_bias`.
- `Nvfp4`: A/B are both packed FP4 (`DType::E2M1Pair`, two E2M1 values per byte, low nibble first) with one unsigned-E4M3 scale per `block_size` elements along K. Output is BF16, `K` is a multiple of `block_size`, and the mode applies only to `gemm`.

`Nvfp4` is the one contract whose operand storage is not the canonical `A=[M,K]`, `B=[K,N]`: both operands are K-packed, so `A` is stored `[M, K/2]` and `B` is stored `[N, K/2]`. That is the orientation a ModelOpt checkpoint already uses and the one the block-scaled kernel consumes, so requiring the usual layout would force a transpose that buys nothing. Any per-tensor scale a checkpoint carries alongside the block scales (ModelOpt's `weight_scale_2` and `input_scale`) multiplies the whole projection and therefore belongs in `alpha`, which also keeps it out of the tuning key.

Block-scale tensors must be in the layout the candidate reads, which is **not** a plain `[rows, K/block_size]` array. Build them once with `ops::pack_block_scales` into a buffer sized by `ops::block_scale_buffer_bytes`; the layout depends only on `block_size`, so the result stays valid when the autotuner selects a different configuration.

Quantization occurs before the API call; the current L3 contract does not include dynamic quantization. At least one registered candidate must still support the concrete spec.

<!-- l3-operator:gemm -->
### `gemm`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm(ctx, GemmArgs)` |
| Inputs | `A=[M,K]`, logical `B=[K,N]`; `GemmWeightLayout` records canonical `[K,N]`, checkpoint `[N,K]`, or the semantic-specific gate/up-pair encoding; supports `None`, `Fp8UnitScale`, `Fp8`, `W8A8`, and `Nvfp4` (whose operands are stored `A=[M,K/2]`, `B=[N,K/2]`) |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = alpha * projection(A,B) / output_scale` |
| Constraints | W8A8 and NVFP4 output must be BF16; `WeightVersion` may declare immutable weights and allow prepare to cache an internal prepacked copy |
| Numerical policy | `GemmNumerics::Default`, or `LegacyCublas` for the established unit-scale canonical BF16 reduction path; the model still calls `gemm` |
| Reference test | `gemm_all_candidates_match_torch`, `nvfp4_all_candidates_match_reference` |

For unquantized F16/BF16 GEMM with matching output dtype, the
`cublasLt-f32-reduction` candidate admits only unsplit or FP32 partial reduction.
It is the second cold fallback after the existing cuBLAS implementation and
can serve a zero-workspace budget when a supported cuBLASLt heuristic exists.
The default cuBLAS fallback order is unchanged for larger budgets. Selection is
part of the GEMM recipe identity (schema v13); it does not promise a particular
vendor algorithm. `zero_workspace_f32_projection_fallback_capture_and_lifetime`
checks every admitted candidate/configuration against independent f64 sums,
including nonunit alpha/output scaling and retained changed-input graph replay.

<!-- l3-operator:gemm_bias -->
### `gemm_bias`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_bias(ctx, GemmBiasArgs { gemm, bias })` |
| Inputs | `A=[M,K]`, logical `B=[K,N]`, `bias=[N]`; plain BF16 checkpoint weights may use physical `[N,K]`; supports the applicable shared GEMM quantization contracts |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = (alpha * projection(A,B) + bias) / output_scale`, with bias broadcast across M |
| Constraints | bias dtype matches the projection dtype; W8A8 output must be BF16 |
| Reference test | `gemm_bias_all_candidates_match_torch`, `gemm_bias_accepts_output_major_checkpoint_weights_and_replays` |

<!-- l3-operator:gemm_bias_single_round -->
### `gemm_bias_single_round`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_bias_single_round(ctx, GemmBiasArgs { gemm, bias })` |
| Semantics | `cast_output(alpha * (A @ B) + bias)`; F32 accumulation and bias addition, with no intervening F16/BF16 projection rounding |
| Inputs/output | Contiguous `A[M,K]`, `B[K,N]` (or `B[N,K]` with `GemmArgs.b_transposed=true`), `bias[N]`, output `[M,N]`; matching F16 or BF16; unquantized only |
| Compatibility | Existing `gemm_bias` preserves its projection-rounding behavior; this semantic has its own ABI/cache identity |
| Providers | Cold fallback first tries a cuBLASLt bias epilogue with direct output and F32 partial reduction (including split-K), then cuBLAS F32 projection plus typed bias; F16/BF16 partial reductions are excluded; cuBLASLt projection remains a tuning candidate |
| Lifecycle | Scratch owned by prepared execution; graph-safe replay retains operands and scratch; alias and workspace checks follow GEMM |
| Reference test | `gemm_bias_single_round_all_candidates_match_torch` covers both dtypes and weight layouts with nonsquare matrices, all candidates, repeated/changed-input replay and retained resources; `single_round_bias_cold_fallback_fits_without_full_f32_projection` bounds retained scratch |

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

<!-- l3-operator:gemm_swiglu -->
### `gemm_swiglu`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_swiglu(ctx, GemmSwiGluArgs { gemm })` |
| Inputs | BF16 `A=[M,K]`, `B=[K,2N]`; `GemmWeightLayout::GateUpPairs { pair_width }` describes the paired gate/up column encoding |
| Output | BF16 `Y=[M,N]` |
| Mathematical semantic | rounded `SiLU(A@B_gate) * (A@B_up)` for each 256-column pair |
| Constraints | `2N` is divisible by 512; scales are unit; provider padding is private and never changes the output shape |
| Candidates | cuBLAS fallback, cuBLASLt, and fixed SM110 CUTLASS at M3387/N9216/K2560 |

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

FA2 uses upstream `UNFUSE_FMA` maximum subtraction to avoid overflow for large
finite equal logits and is compiled without `--use_fast_math`. These arithmetic
settings participate in the attention
recipe build identity. `attention_large_equal_logits_remain_finite` covers
F16/BF16 finite outputs and graph replay for this case.

For non-causal F16/BF16 dense attention with head width 384 and equal Q/K/V
head counts, the SM80+ memory-efficient CUTLASS provider is the first fallback.
It uses 32×128 online-softmax tiles and an FP32 output accumulator of
`B*Tq*H*D*4` bytes, configured before capture and retained by the execution.
Inputs/output require 16-byte alignment; batch and heads are at most 65535 and
`B*Tq`, `Tq*H*D` and `Tk*H*D` must fit i32. Other shapes retain their existing providers. The vendored
forward source and inference-only compatibility surface are documented in
[`efficient_attention/VENDOR.md`](native/kernels/efficient_attention/VENDOR.md).
`efficient_attention_d384_all_candidates_scalar_oracle_and_retained_graph`
covers both dtypes, multiple batches/heads, tile tails, nondefault scale,
all admitted candidates and changed-input graph replay after owner release.

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

## Pointwise arithmetic

`ops::pointwise(ctx, PointwiseArgs)` uses contiguous `[rows, cols]` tensors.
F16, BF16 and F32 inputs/outputs have matching dtypes. Each binary or scalar
operation rounds once to that dtype. New binary/scalar semantics reject all
input/output overlap and validate the exact secondary shape, including the
broadcast axis. All scalar bindings must be finite. There is no workspace or
tuning recipe; direct stream launches are capture-safe and the graph retains
the input/output allocations.

<!-- l3-operator:pointwise_geglu -->
<!-- l3-operator:pointwise_bias_activation -->
<!-- l3-operator:pointwise_euler_update -->
<!-- l3-operator:pointwise_add -->
<!-- l3-operator:pointwise_subtract -->
<!-- l3-operator:pointwise_multiply -->
<!-- l3-operator:pointwise_broadcast_add -->
<!-- l3-operator:pointwise_broadcast_multiply -->
<!-- l3-operator:pointwise_row_multiply -->
<!-- l3-operator:pointwise_scale -->
<!-- l3-operator:pointwise_scalar_add -->
<!-- l3-operator:pointwise_scalar_divide -->

| Semantic | Input / secondary | Output |
| --- | --- | --- |
| `Geglu` | `[rows, 2*cols]`, gate then up | `gelu_tanh(gate)*up` |
| `BiasActivation` | `[rows,cols]`, optional `bias=[cols]` | optional column bias then selected activation |
| `EulerUpdate` | state and velocity, same element count | `state + dt*velocity` |
| `Add`, `Subtract`, `Multiply` | both `[rows,cols]` | `x+y`, `x-y`, `x*y` |
| `BroadcastAdd`, `BroadcastMultiply` | x `[rows,cols]`, secondary `[1,cols]` | per-column add/multiply |
| `RowMultiply` | x `[rows,cols]`, secondary `[rows,1]` | per-row multiply |
| `Scale`, `ScalarAdd` | x `[rows,cols]`, scalar `dt` | `x*dt`, `x+dt` |
| `ScalarDivide` | x `[rows,cols]`, finite nonzero scalar `dt` | FP32 division `x/dt`, then dtype cast |

`BiasActivation` supports None, tanh GELU, exact erf GELU, SiLU, ReLU and
`GeluRounded`, and `SwishRounded`. None without a bias is rejected. ReLU preserves NaNs.
`SwishRounded` rounds the sigmoid tensor to the input dtype before multiplication;
it is distinct from fused SiLU. Optional bias is rounded before this activation.
`GeluRounded` materializes the cubic power, its coefficient multiplication,
inner addition, sqrt(2/pi) multiplication, tanh, addition of one and `0.5*x`
in the input dtype before the final multiplication. Optional bias is rounded
before this activation. This differs from fused float-compute tanh GELU.

The existing F16-to-E4M3 GeGLU additionally accepts a positive output scale,
requires even cols, four-byte input alignment and two-byte output alignment;
other semantics require `output_scale=1`. Precision and binding tests are in
`pointwise_behavior.rs`; captured changed-input and dropped-owner replay are
required for the direct launch path.

## FP64 complex rotary embedding

<!-- l3-operator:resize_bicubic -->

`resize_bicubic(ctx, input, output)` resamples matching F32/F16/BF16 NHWC
tensors with unchanged batch/channel counts, half-pixel coordinates,
edge replication, cubic coefficient -0.75 and no antialiasing. It accumulates
in FP32 and casts to the output dtype. Positive dimensions must fit INT32;
total storage sizes are checked. Input/output overlap is rejected. No workspace
or tuning; graph capture retains both allocations. The caller owns image
normalization and output geometry.

<!-- l3-operator:complex_rotary -->

`ComplexRotaryTable::new` uploads immutable finite FP64 coefficients shaped
`[tokens,head_dim/2,2]` outside capture. `complex_rotary` rotates adjacent pairs
in F16/BF16 `[tokens,heads,head_dim]` using FP64 separate multiply/add operations,
then casts through FP32 to the output dtype, matching Torch's complex cast.
Input and output may be exactly identical; partial overlap is rejected.
The direct launch retains the coefficient table and activation allocations in
captured graphs. It has no tuning or workspace. Torch CUDA fixtures cover both
dtypes, a double-rounding boundary, changed-input replay and owner release.
Dimensions: nonzero tokens up to INT32_MAX, heads up to 65535, even head_dim
up to 2048. This operation does not infer modality positions or construct tables.

## Row normalization

<!-- l3-operator:rms_norm -->
<!-- l3-operator:layer_norm -->
<!-- l3-operator:adaptive_rms_norm -->
<!-- l3-operator:bias_residual -->
<!-- l3-operator:bias_residual_rms_norm -->
<!-- l3-operator:bias_residual_layer_norm -->
<!-- l3-operator:ada_gate_residual -->
<!-- l3-operator:ada_gate_residual_rms_norm -->
<!-- l3-operator:bias_then_residual -->
<!-- l3-operator:rms_norm_rounded -->
<!-- l3-operator:l2_norm_rounded -->

The row normalization family uses `[rows,cols]` F16/BF16 matrices, FP32
reductions and finite positive epsilon. `rms_norm` and `layer_norm` apply
learned `[cols]` weight (and bias for LayerNorm). Adaptive RMS uses a
`[2*cols]` scale/shift vector; adaptive gate/residual uses the third segment
of a `[3*cols]` gate vector. The corresponding residual variants produce a
hidden sum and optionally a normalized output. `bias_then_residual` rounds
input plus bias to BF16 before adding residual; ordinary bias/residual rounds
the final sum. Existing F16 normalization can optionally emit scaled E4M3.

`rms_norm_rounded(ctx, RmsNormArgs)` additionally rounds `x/√(mean(x²)+eps)`
to the input dtype before multiplying by weight. It does not permit E4M3.
`l2_norm_rounded(ctx, input, output, eps)` computes the FP32 row L2 norm,
rounds that norm to the input dtype, clamps it by epsilon represented in that
dtype, then divides x and rounds the output. Epsilon is a denominator floor;
it is not added to the sum of squares. L2 also admits F32 matrices, preserving
promoted normalization arithmetic through the output. Its channel-row kernel
uses sixteen F32 partials, a fixed four-group combine, and no low-precision
partial reduction. Other normalization
semantics retain their existing dtype admission. Both new semantics require matching
matrix shapes and distinct input/output storage. Graph capture retains their
device allocations. `rounded_norm_matches_independent_reference_and_replays`
checks arithmetic, graph lifetime and shape/alias failures.
`l2_f32_preserves_promoted_precision_and_replays` checks FP32 row normalization
against an independent f64 oracle, zero rows and retained-resource replay.

<!-- l3-operator:rounded_additive_attention -->
## Rounded additive attention

`rounded_additive_attention(ctx, RoundedAttentionBiasArgs)` accepts the dense
Q/K/V layout and policy of `AttentionArgs`, plus bias of the input dtype with
shape `[batch, heads, query_tokens, key_tokens]`. Query/KV head counts must
match. There is no implicit causal mask or output quantization; bias can encode
masking. Input, bias and output storage must be distinct where written.

For input dtype T (F16 or BF16), its exact semantic is
`L=round_T(round_T(scale*QKᵀ)+bias)`,
`P=round_T(softmax(float(L)))`, `Y=round_T(PV)`.
This materialized-logit/probability rounding differs from ordinary fused
attention. FP32 accumulation and softmax computation are used internally.
Prepared resources include scores, compact output and provider workspace;
workspace limits, recipes, cold tuning, graph retention and replay use the
existing Attention execution lifecycle. Bias address is part of the in-memory
binding identity; its values are rebound without retuning. The recipe includes
bias alignment and the loaded cuBLAS version. The baseline candidate covers
the complete admitted semantic; ordinary attention candidates cannot select it.

<!-- l3-operator:permute -->
### `permute`

Physically reorders contiguous rank 1–8 F32/F16/BF16 tensors according to a
bijection of axes, preserving element bits. Exact output shape and disjoint
storage are required. Direct CUDA launch, no temporary workspace or tuning;
capture retains input/output storage. Device parity and graph tests are required.

<!-- l3-operator:unfold3d -->
### `unfold3d`

Extracts zero-padded convolution windows from NTHWC F32/F16/BF16 input into a
matrix whose columns flatten kernel-time/kernel-height/kernel-width/channel.
Stride, dilation and asymmetric padding are explicit. Output can cover a bounded
contiguous row range, so a family can reuse tiled convolution workspace. Output
width is `kernel_elements * padded_channels`; channels beyond the input channel
count are zero at every tap, allowing aligned GEMM reduction layouts.
`unfold3d_blocked` uses `[channel_block,kernel_time,kernel_height,kernel_width,
channel_in_block]` columns; its positive block size must divide the padded
channel count. The default `unfold3d` order is unchanged. No
convolution arithmetic, model-specific caching, host transfers or allocations
at launch. Input/output must be disjoint; capture retains their storage.

<!-- l3-operator:pointwise_exp -->
### `pointwise_exp`

Element-wise exponential of contiguous F32/F16/BF16 matrices, evaluated with
precise CUDA FP32 `expf` and cast once to the output dtype. No secondary operand
or bias; direct launch without workspace or tuning. Capture retains storage.

<!-- l3-operator:pointwise_sin -->
### `pointwise_sin`

Element-wise sine in radians of contiguous F32/F16/BF16 matrices, evaluated with
CUDA FP32 `sinf` and cast once. No secondary operand, bias or workspace;
capture retains storage.

<!-- l3-operator:pointwise_cos -->
### `pointwise_cos`

Element-wise cosine in radians of contiguous F32/F16/BF16 matrices, evaluated
with CUDA FP32 `cosf` and cast once. No secondary operand, bias or workspace;
capture retains storage. These transcendental operations follow the selected
CUDA compiler's device math; cross-toolkit bit identity is not guaranteed.

## Testing Responsibilities

The catalog test ensures only that semantics are neither missing nor duplicated; it cannot validate the written contracts. A new L3 semantic must also add a semantic test to `src/ops/tests/l3_behavior.rs` and an independent-reference all-candidate numerical test to `src/ops/tests/precision/precision.rs`.

Tests must run through `crates/apxinf-cuda-new/test-new.sh`; a normal `cargo test` from the repository root does not automatically test this crate.

<!-- l3-operator:convolution3d -->
### `convolution3d`

Valid, ungrouped 3D cross-correlation on contiguous NCDHW input/output and OIDHW
weights. Matching F16/BF16 storage, F32 accumulation, positive explicit strides;
no bias, padding or dilation. The optional cuDNN 9 provider is compiled when
its development headers are available and loads `libcudnn.so.9` on demand;
`convolution_cudnn_version()` reports availability. Other operators do not need
cuDNN. No external model engine is involved.

A session prepares the first supported cuDNN heuristic excluding nondeterminism,
input downconversion and reduced-precision reduction within caller-provided workspace capacity. There is no persistent
convolution tuning recipe. Plans, stream and operand/scratch owners survive
capture; changed or unprepared bindings fail during capture. Workspace must be
256-byte aligned and disjoint from operands, and may be reused sequentially on
the same stream. Selection/allocation happen outside capture; replay executes
only the prepared convolution. Numerical/provider identity remains dependent on
cuDNN/device versions and must be validated on the target.

The translated fixed-shape implementations are not separate model APIs.
D256 attention and the 624/2200 vision schedule are candidates in the existing
Attention registries. BF16 SwiGLU is the GEMM semantic above. First-prefill GDN
is `ops::gdn_first_prefill`; it owns both the fixed SM110 launch and the generic
decomposition, so candidate misses never escape to model code.

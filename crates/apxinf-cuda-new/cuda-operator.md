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
| Inputs | `Q=[B,Tq,Hq,D]`, `K/V=[B,Tk,Hkv,D]`; Q/K/V share F16, BF16, or F32 dtype; mask is `None` or `Causal` |
| Output | `Y=[B,Tq,Hq,D]`; ordinary output uses the input dtype, and F16 input may also be written as E4M3 |
| Mathematical semantic | dense scaled dot-product attention |
| Constraints | Causal requires `Tk>=Tq`, with queries aligned to the final positions of the key sequence; ordinary output requires `output_scale=1`; E4M3 stores `round_to_e4m3(attention/output_scale)` |
| Reference tests | `attention_all_candidates_match_reference`, `attention_f32_real_shapes_all_candidates_match_reference`, `attention_f32_reduction_all_candidates_match_reference`, `attention_f32_cublas_respects_scratch_budget`, `attention_bf16_vae_all_candidates_match_reference` |

The dense F32 MHA cuBLAS candidate covers up to 1024 keys, including batched
and causal inputs. QK and PV use pedantic F32 GEMMs; masking and softmax share
one device kernel. The existing custom candidates cover GQA and longer keys.
Selection remains native and obeys the workspace/recipe policy.

An additional fused F32 candidate handles B1, Q16–128, K1–160, D128,
equal Q/KV heads (at most 65535), no mask and 16-byte-aligned bindings.
It keeps QK, softmax and PV in one kernel with no global scratch. It is the
first eligible fallback in this domain and preserves the F32 contract.
Boundary cases and `attention_short_f32_needs_no_global_workspace` verify
the candidate and the zero-workspace policy; other shapes retain existing
providers.

<!-- l3-operator:additive_attention -->
### `additive_attention`

| Item | Contract |
| --- | --- |
| Rust API | `ops::additive_attention(ctx, AttentionBiasArgs { attention, bias, intermediate_dtype })` |
| Inputs | Dense Q/K/V as above, F16/BF16/F32; F32 `bias=[1 or B,Hq,Tq,Tk]`, broadcast across batch when its leading dimension is one |
| Output | Same shape and dtype as Q; no E4M3 output |
| Mathematical semantic | `softmax(mask(scale * QK^T + bias)) @ V`; negative infinity masks a key, and an entirely masked row produces zero |
| Arithmetic | `intermediate_dtype=F32` (constructor default) keeps scores/probabilities in F32. Explicit BF16 rounds scaled QK, biased scores and softmax probabilities to BF16; QK/PV accumulation and softmax reductions remain F32. This BF16 option currently supports BF16 MHA with at most 1024 keys. |
| Constraints | Same masks and positive scale as dense attention; bias must not alias output. NaN and positive-infinity bias are outside the contract. |
| Graph/tuning identity | Bias address belongs to the execution key; its values can change between replays. Broadcast and intermediate precision belong to the exact recipe key. |
| Reference tests | `additive_attention_all_candidates_match_reference`; `additive_attention_validates_bias_and_replays_changed_bias` |

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


## HWC image building blocks

<!-- l3-operator:image_rgb_indices -->
`image_rgb_indices` accepts exactly `views*height*width*channels` unsigned bytes
in an owned device buffer and a non-aliasing, four-byte-aligned flat I32 output.
`ImageMemoryLayout::{Nhwc,Nchw}` declares the packed input layout. It concatenates
the views along width and widens each byte without normalization, yielding
logical `[height, views*width, channels]` HWC values in `[0,255]`. Both layout
conversion and dtype conversion execute on CUDA. Prepared graph profiles must
retain the selected layout and geometry; changing source contents needs only
an upload into the retained source allocation. This fixed semantic allocates
no device memory and retains both bindings through graph lifetime.

<!-- l3-operator:image_im2col -->
<!-- l3-operator:image_space_to_depth -->
<!-- l3-operator:image_grouped_average -->
<!-- l3-operator:channel_l2_normalize -->
<!-- l3-operator:channel_standardize -->

`image_im2col` maps contiguous `[H*W,C]` BF16/F32 images to
`[OH*OW,K*K*C]` convolution columns ordered `[kh,kw,channel]`, with zero
padding and explicit output extent (including asymmetric right/bottom padding).
`image_space_to_depth` uses width-first output channels `[channel,dx,dy]`.
`image_grouped_average` rearranges a single frame with explicit left temporal
zero padding into `[C,T,dy,dx,H/f,W/f]`, groups the leading dimension, and averages
in F32 before rounding to the input dtype. It has no temporal cache.

`channel_l2_normalize` computes a per-row F32 L2 norm, clamps its denominator to
the positive epsilon, then rounds division, scalar multiplication and gamma
multiplication individually to the input dtype. It is distinct from RMSNorm's
additive epsilon. `channel_standardize` selects the leading output channels,
converts to F32 and computes `(x-mean)/std` with F32 per-channel parameters.

These fixed, single-implementation primitives perform no tuning, allocation or
synchronization in enqueue. Callers supply non-aliasing device outputs; a CUDA
Graph retains every binding. No recipe selection is involved.

## Elementwise arithmetic and adjacent-pair rotations

These fixed primitives enqueue directly on the context stream, allocate no
resources during enqueue, and are captured by the owning `PreparedPhase`.
They have one native baseline, no autotuning or persisted recipe. Inputs and
outputs are contiguous; arithmetic is deterministic for a fixed GPU build.
Pointwise and ordinary norm retain their baseline storage-lifetime contract:
callers keep operand storage alive through replay. Adjacent-pair RoPE retains
its bindings.

<!-- l3-operator:pointwise-multiply -->
<!-- l3-operator:pointwise-broadcast-multiply -->
<!-- l3-operator:pointwise-row-multiply -->
`pointwise(ctx, PointwiseArgs)` extends the existing pointwise family with
`Multiply` (secondary `[rows, cols]`), `BroadcastMultiply`
(secondary `[cols]`), and `RowMultiply` (secondary `[rows]`). Inputs and output
share F32, F16, or BF16 dtype. Existing GELU and EulerUpdate arithmetic
retain their baseline implementations. Broadcast addition uses the existing
BiasActivation semantic with a bias and no activation; elementwise addition
uses EulerUpdate with dt=1.

### Ordinary LayerNorm dtype extension

`norm(ctx, NormArgs::layer(...))` computes row-wise
`((x - mean(x)) / sqrt(mean((x - mean(x))²) + eps)) * weight + bias`
for contiguous `[rows, cols]` input/output and per-column weight/bias.
Alongside the existing F16/BF16 contract, ordinary LayerNorm accepts F32
input, weight, bias and output together, with FP32 statistics/affine arithmetic.
F32 moments are computed relative to the row's first value, retaining that
centered coordinate system through normalization to avoid loss from a large
common offset. The contract requires
finite positive epsilon and unit output scale. F32 output may not overlap any
read-only binding. Mixed F32/BF16 bindings and F32 quantized output are rejected.
Other Norm contracts retain their existing dtype domains. This fixed kernel
has no tuning recipe or workspace and follows the existing caller-owned operand lifetime during capture.
`layer_norm_f32_real_shapes_match_independent_f64_eager_and_graph` covers
1024/3072 columns, learned affine parameters, true F32 inputs, changed-input
replay against an independent F64 Welford reference.

<!-- l3-operator:pairwise-rope -->
`PairwiseRopeTable::new(ctx, tokens, head_dim, coefficients)` uploads immutable
FP64 `(cos, sin)` pairs shaped `[tokens, head_dim / 2]` before capture.
`pairwise_rope(ctx, PairwiseRopeArgs)` rotates adjacent real/imaginary channels
of BF16 `[tokens, heads * head_dim]` input, using FP64 multiply/add without
contraction, then converts to F32 or BF16 output. The even head dimension must
be at most 2048. Dense and axial position schemes are expressed by the table;
no model family or axis convention enters this operator. Partial overlapping
input/output storage is rejected; exact alias is allowed for BF16 only. The
captured graph retains the immutable table and activation storage through
replay, even after the caller drops its handles.

### RMS intermediate precision

`NormArgs::rms` retains `intermediate_dtype=F32`, using the unchanged fused
baseline kernel. Explicit F16/BF16 (equal to the input dtype) selects a separate
implementation that rounds `x * rsqrt(mean(x*x)+eps)` before weight multiplication.
The reduction accumulator stays F32 in both implementations. This option requires
unquantized output of the same dtype and unit output scale; other combinations
are rejected. Other norm semantics retain their existing arithmetic. No new
mathematical operator or model-specific semantic is introduced.

`rms_intermediate_dtype_preserves_default_and_explicit_boundary` checks independent
reference values, both numerical boundaries, rejected combinations and graph replay.
Attention's BF16 intermediate implementation is covered by the existing additive
all-candidate, cold/warm/corrupt recipe and retained-binding graph tests in both
precision modes; a recipe for one mode must not satisfy the other.
The translated fixed-shape implementations are not separate model APIs.
D256 attention and the 624/2200 vision schedule are candidates in the existing
Attention registries. BF16 SwiGLU is the GEMM semantic above. First-prefill GDN
is `ops::gdn_first_prefill`; it owns both the fixed SM110 launch and the generic
decomposition, so candidate misses never escape to model code.

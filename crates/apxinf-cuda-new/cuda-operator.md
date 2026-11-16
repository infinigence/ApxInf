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

`RepackedWeight::upload_row_channel(ctx, host_weight)` accepts canonical CPU INT8
`[K,N]` storage and returns the device tensor together with its shared layout
state. It uploads fixed NK bytes when the compiled device backend supports the
row/channel W8A8 contract, otherwise canonical bytes. This choice is independent
of M and occurs before prepare; only one device representation is allocated.
Attach the returned state to every consumer, including vendor fallback paths.
The tensor retains its logical `[K,N]` shape and must not be read as canonical
storage after upload. Channel scales retain their original output-channel order.
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
| Constraints | W8A8 and NVFP4 output must be BF16; `WeightVersion` may declare immutable weights and allow prepare to cache an internal prepacked copy; `RepackedWeight` commits a model-owned layout in place and reuses it across executions; packing temporaries belong to preparation, never Execution |
| Numerical policy | `GemmNumerics::Default`, or `LegacyCublas` for the established unit-scale canonical BF16 reduction path; the model still calls `gemm` |
| Reference test | `gemm_all_candidates_match_torch`, `nvfp4_all_candidates_match_reference` |

Plain BF16/FP32 GEMM also enumerates CUDA 13 cuBLASLt configurations
72–79: algorithm 66, 128x224 tile, 64xAUTO stages, 2x1x1 cluster, custom
options 0–7. Preparation checks library capability, actual descriptor legality,
alignment, and workspace before allocating provider resources. Configurations
0–71 retain their existing meaning; selection uses the ordinary fingerprint
and autotuner, with no model-specific dispatch.

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
| Inputs | `Q=[B,Tq,Hq,D]`, `K/V=[B,Tk,Hkv,D]`; Q/K/V share F16 or BF16 dtype; mask is `None` or `Causal` |
| Output | `Y=[B,Tq,Hq,D]`; ordinary output uses the input dtype, and F16 input may also be written as E4M3 |
| Mathematical semantic | dense scaled dot-product attention |
| Constraints | Causal requires `Tk>=Tq`, with queries aligned to the final positions of the key sequence; ordinary output requires `output_scale=1`; E4M3 stores `round_to_e4m3(attention/output_scale)` |
| Reference test | `attention_all_candidates_match_reference` |

The BF16 custom fallback parallelizes score generation for noncausal,
single-head, 384-dimensional inputs with at least 128 query/key tokens and
contiguous K/V rows, without dynamic decode metadata. It tiles K through shared
memory while preserving each dot-product accumulation order, then performs the
reference serial softmax reduction. It uses the existing probability workspace;
other shapes/masks keep the original path. `attention_wide_head.rs` covers a
nontrivial scale, batched tail tiles, CPU reference, and captured storage lifetime.
Kernel changes invalidate old recipes through the attention build fingerprint.

The BF16 D128 dense contract also exposes a persistent CUTLASS FMHA candidate
(provider 3, implementation 2, configuration 0). It preserves BF16 products and
FP32 softmax, uses a 200/64/48 register split, and omits unused inference LSE.
It requires the same 16-byte binding alignment and SM100-family capability as
the original candidate, with zero workspace. Kernel initialization happens in
preparation before Graph capture. Normal reference-gated tuning selects between
candidates; the original implementation remains available. The register variant
has a distinct scheduler/kernel type and translation unit to isolate its policy.
Derived persistent scheduler types retain the persistent shared-memory layout;
its mainloop and epilogue storage must not alias across scheduled tiles.

The existing CUTLASS FMHA candidate also covers BF16 noncausal dense attention
with B=1, Tq=1764, Tk in {512,1764}, Hq=Hkv=16 and D=128, under flexible
numerics and 16-byte binding alignment. Residual key masking excludes padding
from softmax when Tk is not a multiple of 128. Its existing 256x256, D72 domain
is retained. Provider selection stays inside the native registry and autotuner;
models do not select CUTLASS. The real-shape all-candidate reference test covers
non-default scale, nonuniform scores, uniform scores and graph capture/replay.

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

The translated fixed-shape implementations are not separate model APIs.
D256 attention and the 624/2200 vision schedule are candidates in the existing
Attention registries. BF16 SwiGLU is the GEMM semantic above. First-prefill GDN
is `ops::gdn_first_prefill`; it owns both the fixed SM110 launch and the generic
decomposition, so candidate misses never escape to model code.
<!-- l3-operator:Conv3d -->
### Conv3d

`PreparedConv3d::new(ctx, Conv3dArgs)` binds a BF16 cross-correlation with FP32
accumulation and BF16 output rounding. Contiguous input, optional preceding
history and output use THWC; immutable weights use `[KT*KH*KW*Cin,Cout]`.
`Unfold3dGeometry` supplies kernel, stride and independent before/after zero
padding. Bias is a separate operation after convolution output rounding.
Output may not overlap input, history or weight. Weights must remain unchanged
through the execution/graph lifetime; input and history contents may change.

`PreparedConv3d::with_policy` adds optional online tuning and exact recipe
persistence through `Conv3dPolicy`; `new` defaults to no tuning with fallback
allowed. The baseline uses optional system cuDNN 9 direct convolution when
alignment and resources fit, then tiled Unfold3d plus deterministic GEMM.
Tuning compares this baseline with explicit backend engine configurations,
currently qualified for sm110/cuDNN 9.20. Invalid recipes are misses; a miss
with tuning and fallback disabled is an error. `cache_dir=None` uses the
context cache; an empty directory disables disk persistence. Preparation
benchmarks only isolated output and never changes caller-visible results.
`workspace_limit` bounds packed weights, transformed input, columns and provider
workspace. No provider name or selection enters the model API. The diagnostic
`summary()` identifies the implementation and selected algorithm/tile size.
Preparation loads symbols, creates descriptors, packs immutable weights and
allocates scratch before capture. Enqueue is asynchronous and allocation-free;
captured graphs retain the execution and storage. No recipes are persisted.
Aligned channel counts divisible by eight use128-bit input/history transfers
with checked32-bit indices; other materialization layouts keep the scalar path.

<!-- l3-operator:Unfold3d -->
### Unfold3d

`PreparedUnfold3d::new` binds a nonempty BF16 THWC volume, optional preceding
THWC history, asymmetric zero padding, positive three-axis kernel/stride and a
contiguous tile of flattened output rows. Columns are ordered kernel T/H/W/C.
This preserves causal history without a host readback or a full im2col allocation;
the caller tiles rows and composes ordinary GEMM. Output must not alias either
source. `enqueue` performs exact BF16 copies with zero padding, allocates no
memory and can be captured after preparation. Captured graphs retain the prepared execution and its
input/output storage even after caller handles are dropped. Deterministic dispatch
uses vectorized bit copies for supported aligned geometries with proven 32-bit
index bounds, and retains a 64-bit scalar path for all other valid inputs.
This adds no device workspace or persistent recipe and does not strengthen the
public BF16 alignment requirement. Supported dtype is BF16 only.
Target-device numerical/capture acceptance is required before a model claims it.

<!-- l3-operator:ChannelL2Norm -->
### ChannelL2Norm

`channel_l2_norm` uses the unified `NormArgs` contract with
`NormKind::ChannelL2` (plain weighted normalization, BF16 only, without
quantization or fused residuals). It accepts contiguous BF16 `[rows, channels]` and learned BF16
`[channels]` weights. It computes the L2 norm in FP32, rounds the norm to BF16,
clamps it by epsilon (normal positive FP32), divides and rounds to BF16, multiplies
by `sqrt(channels)` and rounds again, then multiplies by the learned weight.
The output is BF16 with the input shape. Epsilon is a lower bound on the norm,
not an additive variance epsilon. This distinct numerical semantic preserves
channel normalization used by causal video autoencoders. The fixed kernel has
no selection, workspace or allocation; it supports eager and captured launches.
The fused SiLU variant uses a packed four-lane subgroup for 96/192 channels and
at least 1024 rows when all three bindings are 16-byte aligned. Its tail guard
supports arbitrary row counts. Other alignments use the original warp path;
plain channel-L2 is unchanged. The packed path retains FP32 reduction order,
each staged BF16 rounding and exact in-place operation.

<!-- l3-operator:Multiply -->
### Multiply

`PointwiseSemantic::Multiply` multiplies same-shaped contiguous `[rows, cols]`
input and secondary tensors, with a single round to the output dtype. F16, BF16
and F32 are supported; all three tensors must have the same dtype. No bias,
activation or quantization scale is accepted. Exact in-place operation is
supported. The existing EulerUpdate semantic also accepts F32 for a full-precision
state boundary; F32 also requires identical shapes and rejects partial aliasing.
Its formula and existing F16/BF16 behavior are unchanged.

BF16 pointwise Multiply and EulerUpdate use eight-element vector loads/stores
for at least 1024 elements, counts divisible by eight and 16-byte aligned input,
secondary and output bindings. Other supported alignments, tail counts and
dtypes retain the scalar path. The scalar FP32 arithmetic and output rounding
are preserved, including exact input/output aliasing; partial overlap remains
rejected. `pointwise_vectors.rs` covers aligned/unaligned bindings, tails, the
real DiT activation shape and in-place Graph replay against a CPU reference.


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

## Precomputed RMSNorm + rotary embedding

`ops::rope::rms_norm_apply_precomputed(ctx, input, weight, table, epsilon, output)`
is the only public precomputed-rotation entry point. It fuses per-head RMSNorm
with half-split rotation over caller-provided FP32 cosine/sine coefficients.
Input and output are matching contiguous BF16 `[tokens, heads, head_dim]`;
weight is BF16 `[head_dim]`; table is FP32 `[tokens, 2, head_dim/2]`, cosine
first and sine second, broadcast over heads. Head dimension is positive, even
and at most 2048. The normalized value rounds to BF16 before each FP32 product,
each product and then the sum round separately, and the result converts to BF16
once. Coefficients are used as supplied, without device trigonometry or FMA
contraction. Exact input/output aliasing is allowed; partial overlap and
weight/table output overlap are rejected. The launcher selects a packed
eight-element path for aligned 128-wide rows and an explicit staged
Norm-then-rotation baseline otherwise; neither tunes, persists a recipe, nor
allocates during enqueue. The caller retains its tensors for graph lifetime.
Reference and in-place Graph coverage lives in `tests/rms_precomputed_rope.rs`.
Existing position-based, packed-QKV and vision rotation-table contracts remain
unchanged.

The bare rotation step is deliberately **not** exported: it is an internal
composition of this operator, and a second public entry point would let callers
compose the rounding order by hand. Tests that need the unfused form build it
from public `norm` plus a CPU reference.

### Grouped adaptive LayerNorm

`norm` accepts `Affine::GroupedAdaptive { style, shift_first }` with explicit
`NormNumerics::StagedBf16`. Input/output are contiguous BF16 `[rows, cols]`;
style is `[groups, parts, cols]`, with positive groups dividing rows and parts
2 or 3. Consecutive `rows/groups` rows share conditioning; the third part is
ignored. `shift_first` selects `[shift, scale]` versus `[scale, shift]`.
The result is `bf16(bf16(bf16(layer_norm(x)) * bf16(1 + scale)) + shift)`.
The existing 256-thread LayerNorm reduction order is retained. Quantization,
RMSNorm and fused residual modes are rejected for this layout/numerics pair.
Exact input/output aliasing is supported; partial input/output overlap and
output/style overlap are rejected. Existing vector-adaptive calls retain their
layout and arithmetic. The Norm ABI version is 9 and includes group geometry
and layout; older specs are rejected. `tests/grouped_adaptive_norm.rs` covers
independent CPU math, exact staged comparison, non-vector-aligned bindings,
shape/numerics/alias rejection, and captured replay with input updates.

For grouped adaptive LayerNorm, the 2048-column implementation retains each
thread's eight input values in registers, preserving the 256-thread reduction
order and staged BF16 rounding. Other column widths use the generic path.

`residual_add` additionally accepts `ResidualGate::GroupedStyle { style, numerics }`.
The BF16 style is `[groups, 3, cols]`; groups evenly divide activation rows,
and the third part is the gate. `NormNumerics::StagedBf16` explicitly requires
`bf16(residual + bf16(input * gate))`. Bias, other numerical modes, non-BF16
and malformed grouping are rejected. Input or residual may exactly alias output;
partial overlap and output/style overlap are rejected. The aligned vector path
loads eight BF16 values per transaction; arbitrary column widths and BF16-aligned
addresses use the scalar fallback. Existing packed-vector and unit-offset gates
retain their contracts. The same Norm ABI grouping fields represent this layout.
`tests/grouped_gated_residual.rs` covers both paths, all binding alignment classes,
CPU exact references, aliasing, rejection and captured binding-content updates.


<!-- l3-operator:RmsNormPrecomputedRoPE -->
### RMSNorm with precomputed RoPE

`rms_norm_apply_precomputed(ctx, input, weight, table, epsilon, output)` applies
per-head weighted RMSNorm, rounds that result to BF16, then applies half-split
rotary embedding with separately rounded FP32 products and sums before the
final BF16 conversion. Input/output are matching contiguous BF16
`[tokens, heads, dim]`; weight is BF16 `[dim]`; table is FP32
`[tokens, 2, dim/2]` with cosine followed by sine, broadcast over heads.
Dimensions are positive, dim is even and at most 2048, tokens*heads fits signed
32 bits, and epsilon is positive and finite. Exact input/output aliasing is
allowed; partial input/output overlap and any output overlap with weight/table
are rejected. Inputs, weights and coefficients may change between replays.

The native adapter selects a single-kernel 128-dimensional implementation,
preserving the existing RMSNorm reduction tree and BF16 stage boundary. With
at least 1024 rows and 16-byte-aligned input, weight and output, it packs eight
adjacent logical lanes per thread, retaining the same reduction order and
handling partial blocks with subgroup masks. Smaller or unaligned 128-dimensional
bindings use the scalar implementation. Other
head dimensions compose the existing Norm and RoPE adapters with the output
as intermediate storage. Both paths allocate no device memory or resources,
work during capture, and retain all bound buffers in the captured graph. The
fixed compound operation has its own ABI version 1; existing RoPE ABI v6 is
unchanged. There is no persisted selection or tuning. Tests in
`tests/rms_precomputed_rope.rs` compare against the unfused operators exactly
and independently check CPU math, aliases, rejection, binding updates and graph
replay after the caller drops its source tensors.

<!-- l3-operator:gemm_gelu_exact -->
### BF16 staged exact GELU GEMM

`gemm_gelu_exact(ctx, GemmArgs)` computes `bf16(GELU_erf(bf16(A @ B)))`
with FP32 accumulation and FP32 GELU evaluation. BF16 A `[M,K]`, KN weights
`[K,N]`, and BF16 output `[M,N]`; no bias, quantization or nonunit scales.
Existing GEMM device, storage, overflow and non-overlap validation applies.
The BF16 projection boundary is mandatory and differs from approximate
`tanh` GELU and an activation applied directly to an unrounded accumulator.
cuBLAS plus exact activation is the baseline. Thor SM110 has six prepared
CUTLASS epilogue configurations for 1764×8192×2048; other valid shapes and
insufficiently aligned bindings use the baseline. Prepared executions retain
bindings, including mutable weights, across capture; enqueue allocates nothing.

### Grouped gated residual followed by adaptive LayerNorm

`fused_add_norm` combines `NormKind::Layer`, `ResidualGate::GroupedStyle` and
`Affine::GroupedAdaptive` with `NormNumerics::StagedBf16` on both stages.
Gate style is `[groups, 3, cols]`; independent normalization style is
`[groups, 2 or 3, cols]`, with the same group count. Both styles remain mutable
bindings. The hidden result rounds the gate product to BF16, then rounds the
residual addition to BF16. Normalization and adaptive affine preserve the
existing staged BF16 rounding boundaries and reduction order.

Both native outputs are required and must be disjoint. Each may exactly alias
an activation input; partial activation overlap or any overlap with either
style is rejected. Bias, quantization and non-BF16 inputs are unsupported.
The 16-byte-aligned 2048-column path assigns one warp to each row and four
rows to a CTA, including a guarded tail. It retains the rounded BF16 hidden
values between passes and preserves the original 256 logical lanes' reduction
tree using warp shuffles. Gate and affine arithmetic use separate rounded BF16 pair multiply/add instructions; normalization and reductions remain FP32. The 4-byte-aligned path still packs pairs across
128 physical threads. Other shapes or
2-byte alignment use the existing residual and normalization kernels with
caller-owned hidden storage. Group geometry, epsilon and affine part order
remain runtime parameters. This extends the ABI9 contract without changing
its layout. `tests/grouped_fused_norm.rs` covers the independent closed-form
reference, staged-chain equivalence, in-place hidden, contract rejection,
mutable styles, capture/replay and retained storage.

### BF16 wide-head attention on SM110

Dense separate-QKV attention has a non-fallback cuBLAS candidate for BF16
input/output, batches 1–4, one query/KV head, dimension 384, noncausal flexible
numerics, and query/key lengths from 1 through 4096. It requires 16-byte
binding alignment. Configuration 0 uses serial FP32 softmax; configuration 1
uses a parallel FP32 reduction. QK GEMM accumulates into FP32, V is widened
exactly to FP32 on every call, and probability-times-V uses pedantic FP32 GEMM
before the final BF16 cast. Probabilities are never rounded to BF16 or TF32.
Reduction order can differ from the scalar fallback.

Preparation owns the cuBLAS handle, 4 MiB vendor workspace, FP32 probabilities,
and FP32 V/output buffers. Enqueue only launches prepared operations. Exact
attention recipe keys now include the runtime cuBLAS version as well as the
operator build fingerprint. Unsupported shapes/alignment/policy retain the
existing fallback. The Thor Cosmos batch-one/batch-four VAE paths select
configuration 1. Official model fixtures, all-candidate tests, cache/resource
checks, graph replay with mutable inputs, and retained-storage tests pass.
The full backend suite reports 201 passed, 0 failed, 20 ignored.

<!-- l3-operator:ChannelL2NormSilu -->
### ChannelL2NormSilu

`channel_l2_norm_silu` composes staged BF16 channel-L2 normalization and
SiLU through the Norm family (additive operation 3, unchanged Spec layout).
The normalized result rounds to BF16 before SiLU, retaining the separate
operator rounding boundary. It accepts the same `ChannelL2NormArgs`, including
exact input/output aliasing, and supports only unquantized weighted BF16
channel-L2. Large 96/192/384-channel rows use the fused warp kernel; other
shapes use the generic fused kernel. Both preserve the existing reduction tree.
No preparation allocation, tuning, or graph replay allocation is introduced.
Thor Cosmos full-model qualification passes with bitwise unchanged VAE and
full-joint outputs. Real AutoPolicy pooled median is 445.39 ms; the full
backend suite reports 202 passed, 0 failed, 20 ignored.

<!-- l3-operator:FusedAddChannelL2NormSilu -->
### FusedAddChannelL2NormSilu

`fused_add_channel_l2_norm_silu(FusedAddChannelL2NormArgs)` materializes both
`hidden = bf16(bf16(input + bias) + residual)` and its staged channel-L2
normalization followed by SiLU. An absent bias or residual omits that addition;
all existing BF16 normalization boundaries are retained. Matrices are contiguous
BF16 `[rows, cols]`; weight and optional bias are `[cols]`, and epsilon is finite
and positive. The two outputs must be disjoint. Either may exactly alias input
or residual; partial activation overlap and output overlap with weight/bias are
rejected by both the safe API and native adapter.

The stateless Norm backend fuses aligned 96/192-channel rows when `rows >= 1024`.
Other legal sizes and alignments use the staged pointwise update followed by the
existing channel normalization kernel. No scratch allocation or persistent
state is required. CUDA Graph callers retain all bound storage through replay.
The Norm Spec version is 10; its layout and the GEMM recipe contract are unchanged.

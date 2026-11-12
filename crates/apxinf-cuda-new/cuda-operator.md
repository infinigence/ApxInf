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

## Shared Convolution Contract

Both convolution semantics take a channel-last input and return a channel-last
output, accumulate the projection in FP32, and write the result in the tensor
dtype. The build serves them through cuDNN when it is compiled in and the shape
is supported, otherwise through the direct temporal kernel; the choice is
private, is made per prepared execution, and never changes the result.

<!-- l3-operator:conv1d -->
### `conv1d`

| Item | Contract |
| --- | --- |
| Rust API | `ops::conv1d(ctx, Conv1dArgs)` |
| Inputs | `input=[B,Lin,Cin]`, `weight=[Cout,Cin/groups,K]`, optional `bias=[Cout]`; input, weight, bias, and output share F32, F16, or BF16 |
| Output | caller-owned `out=[B,Lout,Cout]` with `Lout=(Lin+2*padding-dilation*(K-1)-1)/stride+1`; any other length is rejected |
| Mathematical semantic | temporal cross-correlation, with `groups` splitting the channels into independent groups |
| Constraints | `stride`, `dilation`, and `groups` are nonzero; `Cin` and `Cout` are divisible by `groups`; `output_padding` is zero |
| Numerical policy | `Conv1dAccumulation::Default` lets the provider choose its FP32 accumulation order; `OrderedF32` keeps the direct kernel's sequential order for a result that feeds a discrete decision |
| Deferred bias | `with_direct_channel_last` binds the caller's channel-last buffers in place, takes no bias, and skips both layout transforms; the caller must then add it with this path's rounding, `round_dtype(x+bias)`. It needs a build with the cuDNN provider and a geometry that provider serves (an output channel count above one), and it rejects ordered accumulation; a shape the provider serves with a single deterministic engine keeps its NCHW path and reports `.channel_last_fallback` in the prepared summary |
| Reference test | `conv1d_matches_reference` |

<!-- l3-operator:conv-transpose1d -->
### `conv-transpose1d`

| Item | Contract |
| --- | --- |
| Rust API | `ops::conv1d(ctx, Conv1dArgs)` with `Conv1dSemantic::ConvTranspose1d` |
| Inputs | `input=[B,Lin,Cin]`, `weight=[Cin,Cout/groups,K]`, optional `bias=[Cout]`; same dtype rule as `conv1d` |
| Output | caller-owned `out=[B,Lout,Cout]` with `Lout=(Lin-1)*stride-2*padding+dilation*(K-1)+output_padding+1` |
| Mathematical semantic | adjoint of `conv1d`: each input position scatters its weight along the dilated, strided time axis |
| Constraints | `dilation` is 1 and `0<=output_padding<stride`; every other constraint is that of `conv1d` |
| Numerical policy | the same `Conv1dAccumulation` choice, except that `direct_channel_last` has no ordered-accumulation path and is rejected together with `OrderedF32` |
| Reference test | `conv_transpose1d_matches_reference` |

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

## Shared Speech Pipeline Contract

The speech pipeline needs relative attention plus elementwise, statistics, and
layout semantics that the language-model Pointwise ABI does not expose. Every
operator in the family below accepts F32, F16, or BF16 tensors, requires all
tensors of one call to share the dtype and CUDA device, and writes FP32
accumulation back in the tensor dtype at the rounding points its table names.
The speech models run one batch at a time, so these kernels are sized for that
geometry and are not tuned for large batches.

<!-- l3-operator:relative-attention -->
### `relative_attention`

| Item | Contract |
| --- | --- |
| Rust API | `ops::relative_attention(ctx, RelativeAttentionArgs)` |
| Inputs | `Q/K/V=[batch,length,heads,head_dim]`; head-shared `relative_key`, `relative_value` are `[2*window+1,head_dim]` |
| Output | caller-owned `out=[batch,length,heads,head_dim]`, the shape of `Q` |
| Mathematical semantic | `softmax(scale*(Q@K^T + Q@rk[j-pos+window])) @ (V + rv[j-pos+window])`; a relative row outside `[0,2*window]` contributes nothing and the attention is unmasked |
| Constraints | all five tensors and the output share dtype and device; `scale` is finite; `length<=12288`, the number of FP32 scores that fit the 48 KiB shared window |
| Numerical policy | scores and the weighted sum accumulate in FP32; each relative term is added in FP32 before the scale |
| Reference test | model-level: the official-capture SoVITS encoder comparison covers this path; no crate-level candidate test yet |

<!-- l3-operator:bias-activation -->
### `bias_activation`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::bias_activation(ctx, input, bias, out, activation, round_bias)` |
| Inputs | `input` of any nonempty rank with last-axis width `cols`; optional `bias=[cols]` |
| Output | `out`, the shape of `input` |
| Mathematical semantic | `out = activation(input + bias)` with the bias broadcast over every row; `activation` is a `TtsActivation` variant |
| Constraints | `cols` divides the element count; with `round_bias` the sum is rounded to the tensor dtype before the activation |
| Reference test | `leaky_relu_bias_reproduces_the_ordinary_paths_rounding` (model-level) |

<!-- l3-operator:residual-add-bias -->
### `residual_add_bias`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::residual_add_bias(ctx, input, bias, residual, out)` |
| Inputs | `input` and `residual` with the same shape; `bias=[cols]` |
| Output | `out`, the shape of `input` |
| Mathematical semantic | `out = round(round(input + bias) + residual)` |
| Constraints | the intermediate sum is rounded to the tensor dtype, so this matches a deferred producer bias followed by an ordinary residual add |
| Reference test | `residual_add_bias_reproduces_the_ordinary_paths_rounding` (model-level) |

<!-- l3-operator:affine -->
### `affine`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::affine(ctx, input, style, out)` |
| Inputs | `input` with last-axis width `cols`; `style=[2*cols]` holding gamma then beta |
| Output | `out`, the shape of `input` |
| Mathematical semantic | `out = input * style[col] + style[cols+col]` |
| Reference test | model-level: the official-capture CFM operator isolation compares every intermediate |

<!-- l3-operator:glu-separate -->
### `glu_separate`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::glu_separate(ctx, input, gate, out)` |
| Inputs | `input` and `gate` with the same shape |
| Output | `out`, the shape of `input` |
| Mathematical semantic | `out = input * sigmoid(gate)` |
| Reference test | model-level: `official_ffn_operator_isolation` compares the CFM FFN intermediates |

<!-- l3-operator:tanh-sigmoid-gate -->
### `tanh_sigmoid_gate`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::tanh_sigmoid_gate(ctx, input, secondary, out)` |
| Inputs | `input=[rows,2*cols]`, whose last axis holds the `a` half then the `b` half; optional same-shaped `secondary` |
| Output | `out=[rows,cols]` |
| Mathematical semantic | `out = tanh(a + secondary_a) * sigmoid(b + secondary_b)` |
| Reference test | model-level: `official_ffn_operator_isolation` compares the CFM FFN intermediates |

<!-- l3-operator:grn -->
### `grn`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::grn(ctx, input, style, out)` |
| Inputs | `input=[rows,cols]`; `style=[2*cols]` holding gamma then beta |
| Output | `out=[rows,cols]` |
| Mathematical semantic | with `n_c = sqrt(sum_r input[r,c]^2)` and `m = mean_c(n_c) + 1e-6`, `out = gamma_c * (input * n_c/m) + beta_c`, every intermediate rounded to the tensor dtype |
| Constraints | `cols<=12288`, the FP32 normalizers that fit the 48 KiB shared window |
| Reference test | model-level: the official-capture CFM operator isolation compares every intermediate |

<!-- l3-operator:column-norm -->
### `column_norm`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::column_norm(ctx, input, style, out, eps)` |
| Inputs | `input=[rows,cols]`; `style=[2*cols]` holding gamma then beta |
| Output | `out=[rows,cols]` |
| Mathematical semantic | per-column LayerNorm over rows, `out = gamma_c * (x - mean_c) * rsqrt(var_c + eps) + beta_c` |
| Reference test | model-level: the official-capture CFM operator isolation compares every intermediate |

<!-- l3-operator:rope-apply -->
### `rope_apply`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::rope_apply(ctx, input, cos, sin, out)` |
| Inputs | `input=[batch,time,heads,head_dim]`; `cos`/`sin=[time,head_dim]` |
| Output | `out`, the shape of `input` |
| Mathematical semantic | interleaved rotary embedding over adjacent `(2p, 2p+1)` pairs using the pair's own `cos`/`sin` entry |
| Constraints | every product is rounded to the tensor dtype before the final add or subtract, so the result never depends on FP16/BF16 fused-contract behavior |
| Reference test | `layer_norm_linear_rope_attention_match_reference` |

<!-- l3-operator:sinusoidal-embedding -->
### `sinusoidal_embedding`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::sinusoidal_embedding(ctx, times, output, scale, max_period)` |
| Inputs | `times=[rows]`; `out=[rows,width]` with `width>=4` and even |
| Output | caller-owned `out` |
| Mathematical semantic | `angle = (scale*times[row]) * exp(-(col % (width/2)) * ln(max_period)/(width/2-1))`; the first half of each row is `sin(angle)` and the second half `cos(angle)` |
| Constraints | `scale` is finite and `max_period>1` |
| Reference test | model-level: the official-capture CFM time embedding comparison |

<!-- l3-operator:mean-rows -->
### `mean_rows`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::mean_rows(ctx, input, out)` |
| Inputs | `input=[rows,cols]` |
| Output | `out=[1,cols]` |
| Mathematical semantic | per-column mean over rows |
| Reference test | model-level: the official-capture SoVITS encoder comparison |

<!-- l3-operator:argmax-rows -->
### `argmax_rows`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::argmax_rows(ctx, logits, index)` |
| Inputs | `logits=[rows,cols]` |
| Output | I32 `index=[rows]` |
| Mathematical semantic | per-row argmax with first-index tie breaking |
| Reference test | model-level: `hubert` codebook assignment inside the official capture replay |

<!-- l3-operator:write-kv-cache-token -->
### `write_kv_cache_token`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::write_kv_cache_token(ctx, input, cache, meta)` |
| Inputs | `input=[1,1,heads,head_dim]`; `cache=[1,capacity,heads,head_dim]`; a `KvCacheDecodeMeta` whose second word is the write row |
| Output | `cache` updated in place |
| Mathematical semantic | copy the token into cache row `meta[1]` |
| Constraints | F16 or BF16 with both tensors agreeing; the write row is device state, and a row at or past `capacity` writes nothing, mirroring the clamp `kv_cache_attention` applies to the same metadata |
| Reference test | model-level: the dynamic decode-graph replay tests write and then attend over the same cache |

<!-- l3-operator:fill-zero -->
### `fill_zero`

| Item | Contract |
| --- | --- |
| Rust API | `ops::tts::fill_zero(ctx, tensor)` |
| Inputs | any nonempty supported tensor |
| Output | the same tensor zeroed in place |
| Mathematical semantic | write zero to every element |
| Reference test | model-level: exercised by every decode and capture-replay test that starts from a zeroed buffer |

## Testing Responsibilities

The catalog test ensures only that semantics are neither missing nor duplicated; it cannot validate the written contracts. A new L3 semantic must also add a semantic test to `src/ops/tests/l3_behavior.rs` and an independent-reference all-candidate numerical test to `src/ops/tests/precision/precision.rs`.

Tests must run through `crates/apxinf-cuda-new/test-new.sh`; a normal `cargo test` from the repository root does not automatically test this crate.

The speech-pipeline family is currently validated through the model-level
official-capture comparisons listed in its tables: those tests compare every
intermediate the reference implementation materializes, one operator at a time,
but they run only with captured fixtures present. A crate-level reference test
still has to be added before a second model depends on these semantics.

The translated fixed-shape implementations are not separate model APIs.
D256 attention and the 624/2200 vision schedule are candidates in the existing
Attention registries. BF16 SwiGLU is the GEMM semantic above. First-prefill GDN
is `ops::gdn_first_prefill`; it owns both the fixed SM110 launch and the generic
decomposition, so candidate misses never escape to model code.

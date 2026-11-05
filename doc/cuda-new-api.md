# cuda-new Public API

> The complete public interface that `crates/apxinf-cuda-new` exposes to the model layer.
> Architecture: `doc/cuda-new-design.md`; mathematical contracts: `crates/apxinf-cuda-new/cuda-operator.md`.

This document lists only **what the model layer may call**: the public items of the
crate root and the `ops` module. The internals of `ffi`, `workspace`, and `graph` are
out of scope.

The mechanically discovered `src/ops` public-item inventory is available with:

```bash
python3 scripts/check_cuda_new_architecture.py --list
```

CI runs the same checker without `--list`. It rejects new L3 names that expose dtype,
layout, fixed shape, or implementation details; direct model calls into cuda-new FFI;
and CUDA `__global__` definitions under `native/adapters`. Existing naming debt is an
explicit, shrinking allowlist in the checker and must not grow during migration.

Contents:

1. Overview (§1)
2. Infrastructure: context / stream / buffer (§2)
3. Tunable operators: GEMM family, Attention family (§3)
4. Fixed operators: gdn / attn_ops / mlp / model / NVFP4 quantization (§4)
5. Sessions and graph capture (§5)
6. Build-time constants (§6)

---

## 1. Overview

```rust
use apxinf_cuda_new::{
    // §2 infrastructure
    CudaContext, CudaStream, CudaBuffer, CudaDeviceAddress, HostMappedBuffer,
    // §5 sessions and graphs
    ExecutionSession, GraphWorkspace, PreparedPhase, CapturedGraph, capture,
    // §6 build-time constants: tell dependents whether an optional kernel was
    // compiled into this build, so the model layer can branch between the
    // specialized and generic paths (cfgs do not cross crate boundaries, hence
    // the pub const re-export)
    NVFP4_GEMM_SWIGLU, FA2_DECODE,
};
use apxinf_cuda_new::ops;   // all operators live under ops::
```

Only three calling shapes exist:

| Category | Shape | Synchronization |
|---|---|---|
| Tunable operators (§3) | `ops::gemm(ctx, Args)` — an Args struct + policy | synchronous under eager; enqueue-only under session/capture |
| Fixed operators (§4) | `ops::rms_norm(ctx, tensors, scalars…)` — flat parameters | asynchronous enqueue; the caller synchronizes |
| Host helpers (§4.6) | pure arithmetic / constant queries, no GPU | immediate |

All tensor parameters must live on the current `CudaContext` device, be contiguous
row-major, and output storage must not overlap read-only inputs.

---

## 2. Infrastructure

### 2.1 `CudaContext`

One per device; the first parameter of every operator call.

```rust
CudaContext::new(device_id: usize) -> Result<Self, String>
ctx.device_id() -> usize
ctx.stream() -> &CudaStream          // this context's work stream; operators enqueue on it
ctx.synchronize() -> Result<(), String>
```

### 2.2 `CudaStream`

```rust
CudaStream::new() -> Result<Self, String>
CudaStream::default_stream() -> Self
stream.synchronize() -> Result<(), String>
stream.handle() -> cudaStream_t
stream.device() -> usize
```

### 2.3 `CudaBuffer` — device memory

```rust
CudaBuffer::alloc(num_bytes, device) -> Result<Self, String>
CudaBuffer::alloc_zeros(num_bytes, device) -> Result<Self, String>
CudaBuffer::alloc_zeros_async(...) -> Result<Self, String>
buf.copy_from_host(src: &[u8]) / buf.copy_to_host(dst: &mut [u8])
buf.zero()
buf.len() / buf.is_empty() / buf.device()
buf.address() -> CudaDeviceAddress
buf.view(byte_offset, len) -> Result<Self, String>        // zero-copy slice
CudaBuffer::from_tensor(&Tensor) -> Result<Self, String>  // borrows the Tensor's storage
buf.as_tensor(shape, dtype) -> Result<Tensor, String>
```

### 2.4 `HostMappedBuffer` — fixed-address host-mapped memory

The carrier for a CUDA Graph's "runtime-mutable parameters" (design doc §5.6). The CPU
writes, the kernel reads; the address never changes after capture.

```rust
HostMappedBuffer::alloc(len, device) -> Result<Self, String>
hmb.write_u32(v) / hmb.write_u32s(&[u32])
hmb.address() -> CudaDeviceAddress
hmb.address_at(byte_offset, len) -> Result<CudaDeviceAddress, String>
hmb.len() / hmb.is_empty()
```

---

## 3. Tunable Operators

Multiple implementations, autotune selection, recipes persisted to disk (design doc
§2/§4). Every Args struct carries a `policy` field.

### 3.1 Policy

`GemmPolicy` and `AttentionPolicy` share the same fields (`GemmPolicy` adds
`accumulation_dtype`):

```rust
pub struct GemmPolicy {
    pub accumulation_dtype: DType,   // GEMM only; default F32, I32 for W8A8
    pub workspace_limit: usize,      // default 256 MiB
    pub online_tune: bool,           // default true: measure on first sight of a shape
    pub allow_fallback: bool,        // default true: fall back when no recipe is available
    pub graph_safe: bool,            // default true: accept only CUDA-Graph-safe candidates
    pub deterministic: bool,         // default false
    pub cache_dir: Option<String>,   // recipe directory; None = in-process cache only
}
```

### 3.2 GEMM family (6 semantics)

```rust
ops::gemm(ctx, GemmArgs) -> Result<()>                      // Y = alpha·A@B / output_scale
ops::gemm_bias(ctx, GemmBiasArgs { gemm, bias }) -> Result<()>
ops::gemm_bias_gelu(ctx, GemmBiasGeluArgs { gemm, bias }) -> Result<()>
ops::gemm_bias_residual(ctx, GemmBiasResidualArgs { gemm, bias, residual }) -> Result<()>
ops::gemm_geglu(ctx, GemmGegluArgs { gemm }) -> Result<()>  // B=[K,2N], gate/up halves
ops::gemm_swiglu(ctx, GemmSwiGluArgs { gemm }) -> Result<()> // paired gate/up layout
```

`GemmArgs` fields and constructors:

```rust
pub struct GemmArgs<'a> {
    pub a: &'a Tensor,               // [M, K] ([M, K/2] packed for NVFP4)
    pub b: &'a Tensor,               // [K, N] ([N, K/2] packed for NVFP4)
    pub out: &'a mut Tensor,         // [M, N] row-major
    pub quantization: GemmQuantization<'a>,
    pub weight_layout: GemmWeightLayout,
    pub alpha: f32,
    pub output_scale: f32,
    pub policy: GemmPolicy,
    pub weight_version: Option<WeightVersion>,
}

pub enum GemmNumerics {
    Default,
    LegacyCublas, // established BF16 reduction order; still selected by native dispatch
}

GemmArgs::new(a, b, out)                          // plain floating-point GEMM
GemmArgs::row_channel_quantized(a, row_scales, b, channel_scales, out)
    // FP8 versus INT8 and accumulation dtype are inferred from the operands
GemmArgs::block_quantized(a, a_block_scales, b, b_block_scales,
                          block_size, alpha, out)
args.with_immutable_weight(WeightVersion::new(v)) // asserts the weight is immutable,
                                                  // letting candidates cache a prepacked copy
```

`GemmWeightLayout` is `Kn` for canonical `[K,N]` weights, `OutputMajor` for
checkpoint `[N,K]` weights, or `GateUpPairs { pair_width }` for the
`gemm_swiglu` paired gate/up encoding. Layout is data interpretation in the
contract; provider padding and vectorization remain candidate-private.
`GemmPolicy::numerics` expresses a numerical compatibility requirement rather
than a provider name in the L3 function. `LegacyCublas` is accepted only for
unit-scale canonical BF16 `gemm`; native selection restricts that contract to
the matching cuBLAS candidate.

`GemmQuantization`: `None` / `Fp8UnitScale` / `Fp8 { row_scales, channel_scales }` /
`W8A8 { row_scales, channel_scales }` / `Nvfp4 { a_block_scales, b_block_scales, block_size }`.
Per-mode dtype and shape constraints are in `cuda-operator.md`.

### 3.3 Attention family (3 semantics)

```rust
ops::attention(ctx, AttentionArgs) -> Result<()>            // dense SDPA
ops::kv_cache_attention(ctx, KvCacheAttentionArgs) -> Result<()>
ops::segmented_attention(ctx, SegmentedAttentionArgs) -> Result<()>
```

Constructors and common modifiers:

```rust
pub enum AttentionMask { None, Causal }

AttentionArgs::new(query, key, value, out)        // Q=[B,Tq,Hq,D], K/V=[B,Tk,Hkv,D]; mask defaults to None
    .causal()                                     // mask: None → Causal

KvCacheAttentionArgs::new(query, key_cache, value_cache, out)
    .with_decode_meta(&meta)                      // dynamic decode: one graph covers every position
    .non_causal()
// remaining public fields: valid_key_tokens / query_start / mask / scale / policy

SegmentedAttentionArgs::new(query, key, value, out, offsets, host_offsets)
// offsets: &CudaBuffer (U32 segment boundaries), host_offsets: &[u32] (same content on the host)
```

`KvCacheDecodeMeta` — fixed-address metadata for CUDA Graph decode (design doc §5.6):

```rust
KvCacheDecodeMeta::new(device, key_capacity) -> Result<Self>
meta.update(valid_key_tokens, query_start) -> Result<()>    // call before each replay
```

---

## 4. Fixed Operators

Single implementation, direct enqueue, asynchronous return (design doc §3). Listed by
module.

### 4.1 `ops::` GDN linear attention (`gdn.rs`)

```rust
gdn_recurrent_step(ctx, state, q, k, v, decay, beta, output, k_heads)    // single-step decode recurrence
gdn_prefill(ctx, q, k, v, out, gate_log, beta, cu_seqlens, state, workspace,
            tokens, q_heads, v_heads, num_seqs, scale)                    // batched prefill scan
gdn_prepare_prefill(ctx, fused, q_out, k_out, v_out, g, alpha,
                    tokens, row_width, k_heads, v_heads, dim, epsilon)    // prefill input reordering
gdn_conv_prepare(ctx, input, weight, window, q_out, k_out, v_out, decay, alpha,
                 tokens, k_heads, v_heads, epsilon)
gdn_gated_norm(ctx, GdnGatedNormArgs {                                   // gated RMSNorm
    input,
    gate: GdnGate::Dense(gate)
        | GdnGate::Strided { tensor, column_offset, heads },
    weight, output,
    quantized: Option<GdnQuantizedOutput { tensor, input_scale }>,
    epsilon,
})
gdn_causal_conv_step(ctx, window, input, weight, output)                  // causal conv, one step
gdn_causal_conv_forward(ctx, input, weight, output, window, tokens, channels, kernel_width)
gdn_decay_and_beta(ctx, GdnDecayAndBetaArgs {                            // decay/beta preparation
    a, b, a_log, dt_bias, decay, beta,
})
gdn_l2_normalize_heads(ctx, data, epsilon)                                // per-head L2 norm of q/k
gdn_chunk_scan(ctx, GdnChunkScanArgs {                                    // reference scan
    input: GdnChunkScanInput::Split { q, k, v }
         | GdnChunkScanInput::Interleaved { fused, offsets },
    gate_log, beta, output, state, key_heads, chunk_size,
})
```

`gdn_gated_norm` infers single-step versus sequence execution from rank 2/3,
infers F16/BF16 input handling from the input Tensor, and selects the preserved
quantizing implementation only when `quantized` is present. `GdnGate::Dense`
binds a gate with the normalized input's logical shape;
`GdnGate::Strided` binds the planning layout where gate heads occupy a slice of
a wider row. The old `_seq`, `_seq_f16`, `_quantize`, and planning-specific
gated-RMS names are private L1 implementation details, not L3 APIs.

`gdn_decay_and_beta` infers recurrent-step versus sequence execution from rank 1/2.
`gdn_chunk_scan` expresses split and interleaved Q/K/V as one input-layout binding and
infers sequence length, row width, value heads and head dimension from its tensors.
The preserved `_seq` and `_interleaved` native launch details are not public L3 entries.

### 4.2 RoPE and attention companions

```rust
partial_rope(ctx, data, positions, rotary_width, theta)    // partial rotary embedding, in place
decode_rotation(ctx, DecodeRotationArgs { ... })           // one-token 1-D/multimodal rotation
decode_rope(ctx, DecodeRopeArgs::new(...))                 // rotate Q/K and append head-major K/V
split_qkv_vision(ctx, VisionQkvRopeArgs { ... })           // packed vision QKV + bias + 2-D RoPE
split_qkv_axial_rope(ctx, AxialRopeArgs { ... })           // caller-owned packed QKV split + axial RoPE
split_qkv_mrope_cache(ctx, MropeCacheArgs { ... })         // packed GQA split + mRoPE + token-major cache write
split_qkv_cache(ctx, QkvCacheRopeArgs { ... })             // Q/K norm + partial RoPE + token-major cache write
split_grouped_qkv(ctx, GroupedQkvRopeArgs { ... })         // group-major Q/G/K/V split + Q/K norm + partial RoPE
split_query_and_gate(ctx, fused, query, gate)              // q_proj output split
apply_output_gate(ctx, OutputGateArgs { ... })             // sigmoid output gate, source/numerics in the contract
```

Decode callers bind caller-owned Tensor views. The raw fixed-workspace
launchers are private to the RoPE family. `DecodeRopeArgs` preserves the
legacy head-major cache layout `[1, kv_heads, capacity, head_dim]`; scheduling
and dtype-specific native symbols are not exposed to models.

`OutputGateArgs::source` distinguishes a dense gate tensor from a gate embedded
in an interleaved projection row. `OutputGateNumerics::RoundedSigmoid` preserves
the Qwen-Drive checkpoint's BF16 rounding before multiplication; it is a
numerics field rather than a planning-specific entry point. Head-major backend
decode helpers live under crate-root `runtime::attention`, not `ops`.

`VisionQkvRopeArgs::rotation` selects either position IDs or a precomputed
rotation table as semantic input. The SM87 `[512, 16, 64]` vec2 launcher is an
internal candidate selected by the RoPE family; its shape, hardware gate, and
implementation name are not exposed to model callers.

`AxialRopeArgs::numerics` carries the established pre-rotation BF16 rounding
policy when a model requires it. `MropeCacheArgs` carries Q/KV head geometry,
three rotary sections, and the cache offset; the model does not call a fused
attention ABI or select the underlying fixed kernel.

### 4.3 Norm and related fixed building blocks

```rust
ops::norm::norm(ctx, NormArgs {
    input, output, kind,
    affine: Affine::Weight { weight, bias, offset }
          | Affine::Adaptive { style }
          | Affine::WeightedAdaptive { weight, scale, shift },
    quantized, numerics, eps,
})
ops::fused_add_norm(ctx, FusedAddNormArgs {
    input, residual,
    gate: None | Some(ResidualGate::PackedStyle(style))
               | Some(ResidualGate::UnitOffset(gate)),
    hidden, normalized, kind, affine, quantized, numerics, bias_order, eps,
})
ops::residual_add(ctx, ResidualAddArgs {
    input, residual, gate, hidden, bias_order, ..
})
swiglu(ctx, fused, output)                                 // fused=[.., 2N] → [.., N]
add_into(ctx, addend, accumulator)                         // accumulator += addend
elementwise::add(ctx, left, right, output)                 // caller-owned F32/F16/BF16 output
elementwise::mul(ctx, left, right, output)                 // caller-owned F32/BF16 output
elementwise::scale(ctx, input, output, factor)             // caller-owned F16/BF16 output
quantize_fp8_per_tensor(ctx, input, output, input_scale)
gemv(ctx, GemvArgs {                                       // M=1 projection
    weight, activation, output,
    block_scales: Option<GemvBlockScales {
        weight, activation, block_size,
    }),
    alpha,
})
```

The common norm family ABI uses Spec version 6. `operation`, `statistic`,
`affine`, `gate`, `quantization`, and `numerics` are independent fields; dtype
is inferred from the Tensor bindings. Weight-offset RMSNorm is selected with
`NormArgs::with_weight_offset()`. `NormNumerics::StableDecode` and
`NormNumerics::StablePlanning` preserve the established reduction and BF16
rounding orders without exposing a provider name. Weighted-adaptive affine,
unit-offset residual gating, and strided GDN gating likewise remain behind the
formal semantic entries; their exact kernels are private candidates/launchers,
not additional public functions. Input/output aliasing selects the preserved
head candidate without creating another public function.

`elementwise::add`, `elementwise::mul`, and `elementwise::scale` infer dtype
from their Tensor bindings. They do not allocate output storage; models and
backend adapters use `CudaContext::allocate_output` before invoking the
semantic operation. The former public raw decode add/mul wrappers have been
removed. Llama's fixed decode workspace exposes Tensor views over the same
buffers, preserving the existing kernels and launch parameters.

### 4.4 `ops::preprocess`

```rust
preprocess(ctx, PreprocessArgs {
    images, patches, lookup_table,
    spec: PreprocessSpec { layout, geometry, normalization },
})
pixel_shuffle_4(ctx, input, tokens_per_view)
```

`PatchGeometry` describes uniform images or heterogeneous rectangular frame
grids, including patch size, temporal patch size, and spatial merge size.
`PatchNormalization` selects unit-range, mean/std, or caller-supplied lookup
normalization. Input bytes are semantically RGB u8; output dtype is inferred
from `patches`. The three former implementation-shaped RGB entry points are
private dispatch targets and retain their original kernels and arithmetic.

### 4.5 `ops::` sampling and embedding

```rust
embedding_gather(ctx, table, ids, output)                  // embedding row gather
argmax(ctx, ArgmaxArgs::new(logits, index))                // index is a 1-element I32
argmax(ctx, ArgmaxArgs::new(logits, index).with_remap(remap))
                                                            // optional I32 token-id remap
```

The logits dtype is inferred from the tensor. Unsupported dtypes are rejected by
the sampling family instead of selecting a dtype-named L3 entry point.

### 4.6 `ops::` block quantization (`quantization/block.rs`)

Prepares packed operands and block scales for `GemmArgs::block_quantized` / block-quantized
`gemv`:

```rust
pack_block_scales(ctx, PackBlockScalesArgs {
    source, output, rows, k, block_size,
})
    // rewrites checkpoint row-major block scales into the kernel atom layout; once at load time

block_quantize(ctx, BlockQuantizationArgs::activation(
    activation, packed, scales, input_scale, block_size, layout,
))
block_quantize(ctx, BlockQuantizationArgs::rms_norm(
    input, norm_weight, packed, scales, epsilon, input_scale, block_size, layout,
))
block_quantize(ctx, BlockQuantizationArgs::swiglu(
    fused, packed, scales, input_scale, block_size, layout,
))

pub enum ScaleLayout { RowMajor, GemmAtom }   // which scale layout to emit: GEMV reads RowMajor, GEMM reads GemmAtom

gemm_swiglu(ctx, GemmSwiGluArgs::block_quantized(
    activation, activation_scales, weight, weight_scales,
    output, output_scales, alpha, input_global_scale,
    down_inverse_global_scale,
))
    // fixed NVFP4 GEMM+SwiGLU+requantize path; availability: §6 NVFP4_GEMM_SWIGLU
```

The concrete packed format is inferred from the tensors. NVFP4 remains the
private L2/native implementation name; it is not part of the public L3 function
name.

### 4.7 Host helpers (no GPU work)

```rust
ops::rotary_dim(head_dim, partial_rotary_factor) -> usize
ops::gdn_state_elements(v_heads, v_dim, k_dim) -> usize
ops::gdn_prefill_workspace_bytes(v_heads, num_seqs) -> usize
ops::decode_attention_workspace_bytes() -> usize
ops::block_scale_buffer_bytes(rows, k, block_size) -> Result<usize>
```

### 4.8 Fixed decode attention (`kv_cache_attention.rs`)

```rust
ops::decode_attention(ctx, query, key_cache, value_cache, output, workspace,
                      key_tokens, scale) -> Result<()>
```

Qwen3.8-specialized allocation-free FA2 decode (`[1,1,24,256]` query, BF16 caches).
Note: `key_tokens` is a scalar argument and **gets frozen into a captured graph** — any
path meant for CUDA Graph should use `kv_cache_attention + with_decode_meta` instead
(design doc §5.6). Availability: §6 `FA2_DECODE`.

---

## 5. Sessions and Graph Capture

Concepts and the two-pass mechanism are in design doc §5. The interface:

```rust
// session: handle cache + operator order + workspace arena
ExecutionSession::with_capacity(capacity_bytes, device) -> Result<Self>
ExecutionSession::new(workspace: GraphWorkspace) -> Self
session.workspace() -> &GraphWorkspace

GraphWorkspace::new(capacity_bytes, device) -> Result<Self>
ws.allocate(bytes, device) -> Result<CudaBuffer>
ws.capacity() / ws.used()

// the two passes
ops::prepare_with_session(&session, || forward())   // prepare pass: may allocate/tune, records order
ops::with_session(&session, || forward())           // replay pass: cache hits only, verifies order, async enqueue

// capture (low level)
capture(ctx, || forward()) -> Result<CapturedGraph>
graph.replay() -> Result<()>

// recommended entry: two passes + capture packaged; the phase owns the graph
PreparedPhase::prepare_and_capture(ctx, session, || forward()) -> Result<PreparedPhase>
phase.replay() -> Result<()>
phase.session() -> &ExecutionSession
```

Usage constraints (violations are loud errors, never silent):

- The session and the forward pass must share one thread; nested sessions are not supported.
- During replay pass / capture, no operator call may appear that was absent from the
  prepare pass (cache misses and order mismatches are both errors).
- Capture requires fixed addresses for all inputs and outputs (scratch tensors +
  `HostMappedBuffer` metadata).
- `with_session` / `replay()` are asynchronous submissions; the caller invokes
  `ctx.synchronize()` at the outer boundary.

---

## 6. Build-Time Constants

Two `pub const bool`s that carry build-time cfgs across the crate boundary so
dependents can branch:

| Constant | `true` means | When `false` |
|---|---|---|
| `NVFP4_GEMM_SWIGLU` | the SM110 native NVFP4 GEMM + SwiGLU implementation was compiled | `ops::gemm_swiglu` with `GemmSwiGluArgs::BlockQuantized` returns UNSUPPORTED; take the generic path |
| `FA2_DECODE` | the allocation-free FA2 decode kernel was compiled | `ops::decode_attention` returns UNSUPPORTED; use `kv_cache_attention` |

---

## Appendix: Typical Call Sequences

```rust
let ctx = CudaContext::new(0)?;

// eager: call directly; complete on return
ops::gemm(&ctx, GemmArgs::new(&a, &b, &mut out))?;

// graph one phase: prepare once + capture once, then replay every step
let phase = PreparedPhase::prepare_and_capture(
    &ctx,
    ExecutionSession::with_capacity(64 << 20, ctx.device_id())?,
    || { mlp_forward(&ctx, &weights, &scratch); Ok(()) },
)?;
for _ in 0..num_tokens {
    phase.replay()?;          // asynchronous
    ctx.synchronize()?;
}

// dynamic decode attention: one graph covers every position
let meta = KvCacheDecodeMeta::new(ctx.device_id(), key_capacity)?;
// (at capture time)
let args = KvCacheAttentionArgs::new(&q, &k_cache, &v_cache, &mut out)
    .with_decode_meta(&meta);
ops::kv_cache_attention(&ctx, args)?;
// (before each replay)
meta.update(position + 1, position)?;
```

# cuda-new Public API

> The complete public interface that `crates/apxinf-cuda-new` exposes to the model layer.
> Architecture: `doc/cuda-new-design.md`; mathematical contracts: `crates/apxinf-cuda-new/cuda-operator.md`.

This document lists only **what the model layer may call**: the public items of the
crate root and the `ops` module. The internals of `ffi`, `workspace`, and `graph` are
out of scope.

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
    QWEN38_DENSE_SWIGLU_AOT, FA2_DECODE,
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

### 3.2 GEMM family (4 semantics)

```rust
ops::gemm(ctx, GemmArgs) -> Result<()>                      // Y = alpha·A@B / output_scale
ops::gemm_bias(ctx, GemmBiasArgs { gemm, bias }) -> Result<()>
ops::gemm_bias_gelu(ctx, GemmBiasGeluArgs { gemm, bias }) -> Result<()>
ops::gemm_geglu(ctx, GemmGegluArgs { gemm }) -> Result<()>  // B=[K,2N], gate/up halves
```

`GemmArgs` fields and constructors:

```rust
pub struct GemmArgs<'a> {
    pub a: &'a Tensor,               // [M, K] ([M, K/2] packed for NVFP4)
    pub b: &'a Tensor,               // [K, N] ([N, K/2] packed for NVFP4)
    pub out: &'a mut Tensor,         // [M, N] row-major
    pub quantization: GemmQuantization<'a>,
    pub alpha: f32,
    pub output_scale: f32,
    pub policy: GemmPolicy,
    pub weight_version: Option<WeightVersion>,
}

GemmArgs::new(a, b, out)                          // plain floating-point GEMM
GemmArgs::fp8(a, row_scales, b, channel_scales, out)
GemmArgs::w8a8(a, row_scales, b, channel_scales, out)
GemmArgs::nvfp4(a, a_block_scales, b, b_block_scales, block_size, alpha, out)
args.with_immutable_weight(WeightVersion::new(v)) // asserts the weight is immutable,
                                                  // letting candidates cache a prepacked copy
```

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
gdn_gated_norm(ctx, input, gate, weight, output, epsilon)                 // gated RMSNorm
gdn_gated_norm_seq(ctx, input, gate, weight, output, tokens, heads, head_dim, epsilon)
gdn_gated_norm_seq_f16(ctx, ...same as above...)
gdn_gated_norm_quantize(ctx, input, gate, weight, output, quantized, epsilon, input_scale)
gdn_causal_conv_step(ctx, window, input, weight, output)                  // causal conv, one step
gdn_causal_conv_forward(ctx, input, weight, output, window, tokens, channels, kernel_width)
gdn_decay_and_beta(ctx, a, b, a_log, dt_bias, decay, beta)
gdn_decay_and_beta_seq(ctx, a, b, a_log, dt_bias, decay, beta, tokens, heads)
gdn_l2_normalize_heads(ctx, data, epsilon)                                // per-head L2 norm of q/k
gdn_chunk_scan(ctx, q, k, v, g, beta, out, state, v_heads, k_heads, chunk_size)  // reference impl
gdn_chunk_scan_interleaved(ctx, fused, g, beta, out, state,
                           seq_padded, row_width, offsets, v_heads, k_heads, chunk_size, k_dim)
gdn_widen_f16_to_bf16(ctx, input, output, count)
```

### 4.2 `ops::` attention companions (`attn_ops.rs`)

```rust
partial_rope(ctx, data, positions, rotary_width, theta)    // partial rotary embedding, in place
head_rms_norm(ctx, data, weight, epsilon)                  // per-head RMSNorm, in place
split_query_and_gate(ctx, fused, query, gate)              // q_proj output split
apply_output_gate(ctx, data, gate)                         // sigmoid output gate, in place
```

### 4.3 `ops::` MLP building blocks (`mlp.rs`)

```rust
rms_norm(ctx, input, weight, output, epsilon)
swiglu(ctx, fused, output)                                 // fused=[.., 2N] → [.., N]
add_into(ctx, addend, accumulator)                         // accumulator += addend
quantize_fp8_per_tensor(ctx, input, output, input_scale)
fp8_gemv(ctx, weight, activation, output, alpha)           // M=1 projection
nvfp4_gemv(ctx, weight, weight_scales, activation, activation_scales, output, alpha)
```

### 4.4 `ops::` model level (`model.rs`)

```rust
embedding_gather(ctx, table, ids, output)                  // embedding row gather
argmax(ctx, logits, index)                                 // greedy sampling; index is a 1-element I32
```

### 4.5 `ops::` NVFP4 quantization (`gemm/nvfp4_scales.rs`)

Prepares packed operands and block scales for `GemmArgs::nvfp4` / `nvfp4_gemv`:

```rust
nvfp4_pack_block_scales(ctx, source, destination, rows, k, block_size)
    // rewrites checkpoint row-major block scales into the kernel atom layout; once at load time

nvfp4_quantize_activation(ctx, activation, packed, scales, input_scale, block_size, layout)
nvfp4_quantize_rms_norm(ctx, input, norm_weight, packed, scales,
                        epsilon, input_scale, block_size, layout)   // RMSNorm + quantize fused
nvfp4_quantize_swiglu(ctx, fused, packed, scales, input_scale, block_size, layout)  // SwiGLU + quantize fused

pub enum ScaleLayout { RowMajor, GemmAtom }   // which scale layout to emit: GEMV reads RowMajor, GEMM reads GemmAtom

nvfp4_dense_swiglu_aot(ctx, activation, activation_scales, weight, weight_scales,
                       output, output_scales, alpha, input_global_scale,
                       down_inverse_global_scale, tile_groups, tile_limits,
                       token_map, tile_count)
    // shape-specialized FC1+SwiGLU+requantize AOT path; availability: §6 QWEN38_DENSE_SWIGLU_AOT
```

### 4.6 Host helpers (no GPU work)

```rust
ops::rotary_dim(head_dim, partial_rotary_factor) -> usize
ops::gdn_state_elements(v_heads, v_dim, k_dim) -> usize
ops::gdn_prefill_workspace_bytes(v_heads, num_seqs) -> usize
ops::decode_attention_workspace_bytes() -> usize
ops::nvfp4_scale_buffer_bytes(rows, k, block_size) -> Result<usize>
```

### 4.7 Fixed decode attention (`kv_cache_attention.rs`)

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
| `QWEN38_DENSE_SWIGLU_AOT` | this build linked the Qwen3.8 NVFP4 dense SwiGLU AOT object | `ops::nvfp4_dense_swiglu_aot` returns UNSUPPORTED; take the generic path |
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

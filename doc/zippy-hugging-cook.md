# Plan: Metal GPU Attention and Batched RoPE Kernels

## Context

During Metal inference, the attention computation in `llama.rs` falls back to CPU with GPU→CPU→GPU round-trips per layer. The batched RoPE (`rope_batched`) also runs CPU-only. This kills performance — every layer copies Q/K/V to CPU, runs scalar attention loops, then copies the result back to Metal. The fix requires Metal-native kernels for RoPE, KV cache management, and attention (both decode and prefill).

## Files to Modify

| File | Change |
|------|--------|
| `crates/apxinf-metal/src/shader_source.rs` | Add 4 new MSL kernels |
| `crates/apxinf-metal/src/kernels.rs` | Add 4 Rust dispatch methods |
| `crates/apxinf-metal/src/kv_cache.rs` | **New file**: MetalKVCache struct |
| `crates/apxinf-metal/src/lib.rs` | Export kv_cache module |
| `crates/apxinf-model/src/llama.rs` | Metal path in attention(), MetalKVCache field |

## Step 1: Metal Batched RoPE Kernel

**MSL kernel `rope_batched_f32`** in `shader_source.rs`:
- Input: `[seq_len, n_heads, head_dim]`, output: same shape
- Half-split RoPE: pairs `(i, i + head_dim/2)` matching PyTorch Llama
- Params: `head_dim`, `n_heads`, `rope_theta`, `start_pos`
- Grid: `(head_dim/2, n_heads, seq_len)` via `dispatch_3d`

**Rust method `MetalKernels::rope_batched()`** in `kernels.rs`:
- Signature matches existing `rope()` but with `start_pos` instead of `pos`
- Dispatches `rope_batched_f32` kernel

## Step 2: Metal KV Cache

**New file `crates/apxinf-metal/src/kv_cache.rs`**:
- `MetalKVCache` struct with per-layer `MetalBuffer` for K and V
- Buffer layout: `[n_kv_heads, max_seq_len, head_dim]` (contiguous per head, coalesced reads)
- Methods: `new()`, `append()`, `k_buffer()`, `v_buffer()`, `advance()`, `seq_len()`, `clear()`

**MSL kernel `kv_cache_append_f32`** in `shader_source.rs`:
- Copies K/V from `[seq_len, n_kv_heads, head_dim]` layout into `[n_kv_heads, max_seq_len, head_dim]` cache
- Grid: `(head_dim, n_kv_heads, append_len)` via `dispatch_3d`

## Step 3: Metal Decode Attention Kernel

**MSL kernel `sdpa_decode_f32`** — single query token against full KV cache:
- Input: Q `[n_heads, head_dim]`, K/V from cache buffers
- Output: `[n_heads, head_dim]`
- Each thread handles one Q head, iterates all KV positions with online softmax
- GQA: `kv_h = gid * n_kv_heads / n_heads`
- Grid: `n_heads` via `dispatch_1d`
- Causal masking: not needed (decode token sees all prior positions)

**Rust method `MetalKernels::sdpa_decode()`**:
- Takes Q tensor + raw K/V MetalBuffer references from MetalKVCache
- Returns attention output tensor

## Step 4: Metal Prefill Attention Kernel

**MSL kernel `sdpa_prefill_f32`** — batch of query tokens with causal mask:
- Input: Q `[seq_len, n_heads, head_dim]`, K/V from cache buffers
- Output: `[seq_len, n_heads, head_dim]`
- Grid: `(n_heads, seq_len)` via `dispatch_2d` — each (head, position) is one thread
- Causal masking: position `s` attends only to positions `0..=kv_offset + s`
- Same online softmax as decode, but with per-position KV length limit

**Rust method `MetalKernels::sdpa_prefill()`**:
- Takes Q tensor + raw K/V MetalBuffer references from MetalKVCache
- Returns attention output tensor

## Step 5: Integration in llama.rs

- Add `metal_kv_cache: Option<MetalKVCache>` field to `LlamaModel`
- Initialize in `to_device()` when transferring to Metal
- Rewrite `attention()` Metal path:
  1. Apply `rope_batched` on Metal (instead of CPU `rope_batched`)
  2. Append K/V to MetalKVCache via `kv_cache_append` kernel
  3. Dispatch `sdpa_decode` (seq_len==1) or `sdpa_prefill` (seq_len>1)
  4. Reshape output and project via `self.matmul()` — stays on Metal
  5. Eliminate all `metal_ops::to_cpu()` / `metal_ops::to_metal()` round-trips in attention
- Update `forward()` to call `metal_kv_cache.advance(seq_len)` on Metal
- Update `generate_streaming()` to use MetalKVCache instead of CPU KVCache when on Metal

## Verification

1. **Per-kernel unit tests** (in `apxinf-metal/src/ops.rs` test module):
   - `test_rope_batched`: Compare Metal output against CPU `rope_batched` reference
   - `test_kv_cache_append`: Write and read back, verify layout correctness
   - `test_sdpa_decode`: Small cache (2 KV heads, 3 positions, head_dim=4), compare against CPU reference
   - `test_sdpa_prefill`: 3-token prefill with causal mask, compare against CPU reference

2. **End-to-end test**: Run `apxinf generate --model models/tinyllama --prompt "Where is the capital of Canada?" --device metal --max-tokens 10` and verify correct output ("The capital of Canada is Ottawa.")

3. **Existing tests**: All 46 workspace tests must still pass (CPU path unchanged)

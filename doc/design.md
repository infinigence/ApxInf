# ApxInf Design Principles

Date: 2026-07-04
Status: First draft. Documents the intended layering and known violations.
To be refined.

## The core rule

**Backend is device management + single-kernel APIs. Model structure lives
in `apxinf-model`. The two layers never cross.**

```
┌─────────────────────────────────────────────────────┐
│ apxinf-model                                         │
│  - model architecture (Llama, Qwen3-VL, ...)        │
│  - layer structure (norm → attn → MLP → residual)   │
│  - weight layout / packing                           │
│  - decode workspace orchestration                    │
│  - calls backend kernel wrappers                     │
├─────────────────────────────────────────────────────┤
│ apxinf-cuda / apxinf-core                              │
│  - Backend trait: device mgmt + kernel APIs          │
│  - each op = ONE CUDA kernel (or one cuBLAS call)    │
│  - no knowledge of "transformer" or "layer"          │
│  - no decode_forward_workspace in the trait          │
└─────────────────────────────────────────────────────┘
```

The backend exposes primitives like `matmul`, `rms_norm`, `rope`,
`flash_attn_decode`, `silu_mul`. Each maps to exactly one kernel. The
model layer composes these into a forward pass. The backend never
imports model types, never knows the order of operations, never decides
which fusion to apply.

## Why this matters

1. **Adding a new model shouldn't touch the backend.** If I add a
   Mamba or a T5, I shouldn't need to edit `apxinf-cuda`. Today, adding a
   new architecture that doesn't fit the Llama layer template means
   rewriting `decode_forward_capturable` — that's wrong.

2. **Fusion choices are model decisions.** Whether to fuse QKV, whether
   to use flash attention, whether to tie embeddings — these are
   architecture-specific. Hardcoding them in the backend means every
   model gets the same fusion whether it fits or not.

3. **The workspace buffers are model-specific.** `CudaDecodeWorkspace`
   has fields named `q`, `k`, `v`, `gate`, `up`, `mlp_hidden` — these are
   Llama transformer concepts. A Mamba model would need different
   buffers. The workspace should be owned by the model, not the backend.

4. **Testability.** Backend kernels are tested in isolation (the 25
   unit tests in `kernels.rs`). The layer composition is tested via
   end-to-end greedy-token match. Mixing the two makes both harder to
   debug.

## Current violations (resolved 2026-07-04)

All five violations below have been fixed. The model-structure knowledge
(`decode_forward_capturable`, `DecodeGraphConfig`/`Weights`/`LayerWeights`,
`CudaDecodeWorkspace`, `CudaDecodeGraph`, helpers) moved from `apxinf-cuda`
and `apxinf-core` to `apxinf-model/src/decode_graph.rs`. The `Backend` trait
gained `as_any()` for downcasting; `decode_forward_workspace` was removed.

These are places where model-structure knowledge has leaked into the
backend layer. Each should move to `apxinf-model`.

### 1. `decode_forward_capturable` in `apxinf-cuda/src/backend.rs`

**The big one.** ~200 lines that hardcode the full Llama/Qwen3 transformer
decode forward:
- Pre-attention norm → QKV GEMM → RoPE → KV append → attention → wo →
  residual → pre-FFN norm → Gate/Up GEMM → silu_mul → down → residual.
- Knows about GQA (`n_heads / n_kv_heads`), fusion choices (packed QKV,
  packed Gate/Up, flash attention, rope_k_write, rms_norm_add).
- Knows the residual structure (post-attn add + next-norm fusion, post-
  FFN add + next-layer-norm fusion).

**Where it should go:** `apxinf-model` — a `LlamaDecodeGraph` (or
`TransformerDecodeGraph`) struct that owns the workspace buffers and
calls backend kernel wrappers in the right order. The fusion choices
(packed QKV, flash attention, etc.) become model-level configuration,
not backend hardcoded logic.

### 2. `DecodeGraphConfig` / `DecodeGraphWeights` / `DecodeLayerWeights`
   in `apxinf-core/src/backend.rs`

These types encode "a transformer layer has wq/wk/wv/wo/w_gate/w_up/
w_down + two norm weights + optional q_norm/k_norm + optional packed
weights". That's Llama-specific architecture knowledge in the
backend-agnostic core crate.

**Where they should go:** `apxinf-model` — as part of the model's own
weight/workspace types. The backend shouldn't define "layer" or
"attention norm weight" — it just sees `&Tensor` references.

### 3. `decode_forward_workspace` on the `Backend` trait

The default method on `Backend` that returns `Ok(None)` and the
`CudaBackend` override both encode the concept of a "decode forward"
(a model-level operation). The `Backend` trait should only have
primitive ops + device management.

**Where it should go:** Removed from the trait. The model calls
`backend.matmul(...)`, `backend.rms_norm(...)`, etc. directly. The
workspace + graph capture is orchestrated by the model.

### 4. `CudaDecodeWorkspace` in `apxinf-cuda/src/decode_workspace.rs`

The struct has Llama-specific buffer names (`q`, `k`, `v`, `q_rope`,
`k_rope`, `scores`, `attn_weights`, `attn_out`, `attn_proj`,
`ffn_norm_out`, `gate`, `gate_silu`, `up`, `mlp_hidden`, `mlp_out`).
A different model architecture would need different buffers.

**Where it should go:** `apxinf-model` — the model defines its own
workspace struct with the buffers it needs. The backend provides
`CudaBuffer` (raw GPU memory allocation) as a building block.

### 5. `CudaDecodeGraph` in `apxinf-cuda/src/backend.rs`

Per-bucket captured CUDA Graphs for the decode forward. The graph
capture/replay mechanism is backend-level, but "what to capture" (the
decode forward) is model-level.

**Where it should go:** The graph capture/replay primitive stays in
`apxinf-cuda` (it's a CUDA concept). But the specific decode graph
construction moves to `apxinf-model`.

## What stays in the backend

- `CudaBuffer` — raw GPU memory allocation/free.
- `CudaContext` / `CudaStream` — device + stream management.
- `CublasHandle` — cuBLAS GEMM wrapper.
- Kernel wrappers in `kernels.rs` / `ops.rs` — each one kernel:
  `rms_norm`, `silu`, `silu_mul`, `add`, `mul`, `rope`, `rope_mrope`,
  `rope_k_write`, `flash_attn_decode`, `vision_sdpa`, `layer_norm`,
  `gelu_tanh`, `add_bias`, `embedding`, `softmax`, `kv_cache_append`,
  `rms_norm_add`, `concat_2d`, etc.
- `Backend` trait — the union of the above, callable as `dyn Backend`.

The line: **if it names a CUDA kernel or a CUDA API, it's backend. If it
names a model concept (layer, attention, residual, decode), it's model.**

## The mega-kernel exception

Someday we may introduce a fused mega-kernel like `llama_decode_layer`
or `qwen3_decode_layer` — one kernel that does the entire layer forward.
That would live in `apxinf-cuda` (it's a CUDA kernel) but be called by
`apxinf-model` (the model decides to use it). The kernel itself doesn't
know it's "Llama" — it's just a kernel with a specific signature. The
model knows which kernel to call for which architecture.

This is the "concrete types are the ceiling" philosophy: portable
models use `dyn Backend` + compose primitives; specialized models can
call architecture-specific mega-kernels via the concrete `CudaBackend`
type. But the backend crate doesn't import model types either way.

## Migration path (not yet started)

This is a significant refactor. The steps, roughly:

1. Move `DecodeGraphConfig` / `DecodeGraphWeights` / `DecodeLayerWeights`
   from `apxinf-core` to `apxinf-model`.
2. Move `CudaDecodeWorkspace` from `apxinf-cuda` to `apxinf-model` (it
   becomes `LlamaDecodeWorkspace` or similar).
3. Move `decode_forward_capturable` from `apxinf-cuda` to `apxinf-model`
   (becomes a method on the workspace or the model).
4. Remove `decode_forward_workspace` from the `Backend` trait.
5. Keep the graph capture/replay primitive in `apxinf-cuda` but have the
   model drive it.

Each step is independently committable. The end state: `apxinf-cuda` has
only kernel wrappers + device management; `apxinf-model` has all model
architecture + workspace orchestration.

## Relationship to existing philosophies

This restates and sharpishes philosophies #1 and #2 from
`doc/20260619-qwen3vl/notes.md`:

> 1. Trait is the floor; concrete types are the ceiling.
> 2. Layering is strict: model → backend, never backend → model.

The `decode_forward_capturable` violation existed before the fusion work
(Phase 3 of the graph-support plan put it there), but the fusion work
made it worse by adding more model-specific logic (packed weights,
fusion dispatch, flash attention selection) inside the backend. This
doc is the trigger to fix it.

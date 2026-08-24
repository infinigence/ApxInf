# Adding a New Model to ApxInf

A guide for porting a HuggingFace transformer model to ApxInf. Written
after the Qwen3-VL-2B port; every claim here is based on what actually
worked. See `doc/20260619-qwen3vl/` for the full case study.

## One-page overview

ApxInf gives you for free:
- A `Backend` trait (`apxinf_core::Backend`) with primitive ops: `matmul`,
  `rms_norm`, `silu`, `add`, `mul`, `scale`, `rope`, `embedding`,
  `sdpa_decode`, `sdpa_prefill`, `kv_append`, `to_device`, `to_cpu`,
  `synchronize`. CUDA + CPU backends implement it.
- A `LlmTrait` (`apxinf_model::LlmTrait`) with token-level `forward`, unified
  text/image `prefill(LlmInput)`, backend-bound GPU sampling, and
  `generate_streaming_with_options`.
- A `DecodeGraphConfig` + `DecodeGraphWeights` pattern for the
  allocation-free decode fast path (CUDA workspace + graph capture).
- A safetensors loader (`apxinf_loader::safetensors`) that preserves bf16
  natively (`load_native`) or upcasts to f32 (`load`).
- A tokenizer (`apxinf_tokenizer::Tokenizer`) with chat-template support.

What you always have to write:
- A config struct parsed from the model's `config.json`.
- A weights struct with `from_map(HashMap<String, Tensor>)` that picks the
  right keys and transposes 2D projections `[out, in]` → `[in, out]`.
- A model struct implementing `LlmTrait`; multimodal models override `prefill`
  and advertise `LlmCapabilities`. Every model returns its `backend()` so the
  shared generation loop can create a sampler for device-resident logits.
- A registered loader. `AutoModel` detects Hugging Face `model_type`, so the
  shared CLI generation path does not add a model-specific decode runner.

## The four-file recipe

```
crates/apxinf-model/src/<model>/
  mod.rs         — module wiring + re-exports
  config.rs      — <Model>Config parsed from HF config.json
  weights.rs     — <Model>Weights with from_map() that transposes HF weights
  general.rs     — General<Model> implementing LlmTrait
```

For multimodal models, add:
```
  vision_weights.rs  — vision tower weights
  vision.rs          — vision forward path
```

### config.rs

Parse the HF `config.json` directly with `serde_json`. Do NOT reuse
`apxinf_loader::ModelConfig` — it's Llama-shaped and won't fit models with
nested configs (like Qwen3-VL's `text_config` / `vision_config`).

```rust
pub struct MyConfig {
    pub hidden_size: usize,
    pub n_layers: usize,
    // ...
}

impl MyConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&raw)?;
        Ok(Self {
            hidden_size: v["hidden_size"].as_u64().unwrap_or(2048) as usize,
            // ...
        })
    }
}
```

Add a unit test that parses a minimal config string — catches JSON
schema mismatches early.

### weights.rs

`from_map` takes a `HashMap<String, Tensor>` (from the safetensors loader)
and picks out the model's keys. HF stores 2D Linear weights as
`[out_features, in_features]`; cuBLAS row-major matmul wants
`[in_features, out_features]`, so transpose every 2D projection.

```rust
pub fn from_map(cfg: &MyConfig, mut tensors: HashMap<String, Tensor>) -> Result<Self> {
    let take = |name: &str, m: &mut HashMap<String, Tensor>| -> Result<Tensor> {
        m.remove(name).ok_or_else(|| Error::Other(format!("missing {name}")))
    };
    // ...
    Ok(Self { /* ... */ })
}

fn transpose_2d(t: &Tensor) -> Result<Tensor> {
    // [rows, cols] → [cols, rows], handles both F32 and BF16
}
```

**Gotcha:** if a weight is 4D/5D (e.g. a Conv3d weight
`[out, in, k_t, k_h, k_w]`), reshape to 2D `[out, in*k_t*k_h*k_w]` first,
then transpose. The reshape must flatten in C-contiguous order matching
how the input is laid out.

### general.rs

The model struct holds the config, weights (on the backend's device),
backend, and KV cache. The `forward` method runs the transformer layers
via `dyn Backend` ops.

```rust
pub struct GeneralMyModel {
    config: MyConfig,
    weights: MyWeights,
    backend: Box<dyn Backend>,
    kv: Box<dyn KvCache>,
}

impl LlmTrait for GeneralMyModel {
    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        let x = self.backend.embedding(&self.weights.token_embedding, token_ids)?;
        let mut x = x;
        for i in 0..self.config.n_layers {
            x = self.forward_layer(&x, i, start_pos)?;
        }
        self.kv.advance(seq_len);
        let x = self.backend.rms_norm(&x, &self.weights.norm, eps)?;
        self.backend.matmul(&x, &self.weights.lm_head)
    }

    fn backend(&self) -> &dyn Backend {
        self.backend.as_ref()
    }
}
```

Return logits on the model device. The shared sampler selects from their final
row; copying the complete vocabulary row to the host defeats the CUDA logits
pipeline. See
[`doc/20260819-sampling-subsystem`](20260819-sampling-subsystem/README.md) for
the complete contract and backend implementation.

**Key pattern:** borrow only `self.weights` (immutable) when building the
`DecodeGraphWeights` view, so it can coexist with the `&mut self.kv`
borrow for the KV cache.

### Loader registration

Register the loader and its Hugging Face `model_type`. `AutoModel::load_model`
detects it from `config.json`; the shared CLI and generation pipeline require
no model-specific routing or decoding function.

## Decision tree: does my model need new kernels?

Before adding a new CUDA kernel, check if you can spell it with existing
ops + a reshape:

1. **RMSNorm** — `rms_norm(input, weight, eps)`. If the model uses
   LayerNorm instead, you need a new `layer_norm_bf16` kernel (has bias +
   mean subtraction).
2. **RoPE** — `rope(input, n_heads, head_dim, theta, pos_offset)` does
   1-D rotate_half. If the model uses a different rotation (interleaved
   pairs, mRoPE, 2D vision RoPE), you need a new kernel.
3. **Attention** — `sdpa_decode` / `sdpa_prefill` handle Llama-style
   GQA with causal masking. If the model uses non-causal attention
   (vision tower) or a different head layout, you need a new kernel.
4. **Activation** — `silu` is built-in. For GELU-tanh, you need a new
   kernel. For SwiGLU, it's `silu(gate) * up` composed from `silu` +
   `mul`.
5. **QK-norm** — Don't add a new op. Reshape `[seq, heads, head_dim]` to
   `[seq * heads, head_dim]` and call `rms_norm` with the 128-d weight.
6. **Bias addition** — Linear layers with bias need `add_bias` (broadcast
   a `[cols]` vector over `[rows, cols]`). Llama doesn't have biases;
   Qwen3-VL vision does.

The bar for adding a new kernel: **is there any way to express this with
existing kernels + a reshape?** If yes, don't add the kernel.

## Verification recipe

Set the local HuggingFace model directories used by the reference and debug
tools. When running a single `--only` target, only its corresponding variable
is required.

```bash
export APXINF_TINYLLAMA_MODEL_DIR=/path/to/TinyLlama-1.1B-Chat-v1.0
export APXINF_QWEN3VL_MODEL_DIR=/path/to/Qwen3-VL-2B-Instruct
```

### 1. Write an HF reference dump script

`scripts/hf_reference_dump.py` loads the model in HuggingFace
transformers, runs a fixed prompt, and saves intermediate activations +
greedy tokens as `.npz` files under `tests/<model>_reference/`.

Capture:
- Input token IDs
- Post-embedding (last position)
- Per-layer hidden state at layers 0 / mid / last (last position only —
  keeps files small)
- Post-final-norm
- Full logits (last position)
- First 10 greedy token IDs

Use bf16 for the HF model and dump activations as f32 (numpy doesn't
have native bf16 — **never label bf16 bytes as `'<f2'` in .npy files,
that's float16, a different format**).

### 2. Diff ApxInf's output against the reference

Write an example binary (`crates/apxinf-model/examples/<model>_check.rs`)
that loads the model, runs the forward, and either:
- Saves intermediate tensors as `.npy` for Python comparison, or
- Directly compares greedy tokens against the reference.

The correctness gate: **first 10 greedy tokens must exactly match HF.**

### 3. Debug per-layer, not end-to-end

If the final output is wrong, dump intermediate states at every layer and
compare against HF. A max_abs divergence at layer 0 tells you the bug is
in embedding/RoPE/QK-norm. A divergence that starts at layer 5 tells you
the bug is in attention or MLP. End-to-end token comparison is too weak
to localize bugs.

## Gotchas we hit

1. **`.npy` dtype mismatch.** Writing bf16 bytes with `'descr': '<f2'`
   (numpy float16) silently corrupts every value — bf16 and float16 have
   different exponent/mantissa layouts. Always write f32 (`'<f4'`) for
   debug dumps.
2. **`__shfl_xor_sync` with full mask deadlocks on partial warps.** If
   some threads exit a strided loop early, the remaining threads calling
   `__shfl_xor_sync(0xffffffff, ...)` hang. Fix: make the loop non-strided
   (all threads iterate every element) or use `__activemask()` for the
   reduction.
3. **Tied embeddings need a transposed copy.** `token_embedding` is
   `[vocab, hidden]`; the lm_head matmul needs `[hidden, vocab]`. Cache
   the transpose at load time — doing it per forward is catastrophic
   (622 MB for Qwen3-VL).
4. **mRoPE position IDs are not linear.** Image tokens share 2D grid
   positions, so after the image the linear position diverges from the
   mRoPE position by `rope_delta`. Decode tokens must use
   `linear_pos + rope_delta` for RoPE, or generation diverges after the
   first token.
5. **`--features cuda` is not on `apxinf-cuda`.** The crate always builds
   against CUDA. Run tests as `cargo test -p apxinf-cuda` (no flags).
6. **Kernel launchers must not call `cudaStreamSynchronize`.** It breaks
   CUDA Graph capture and slows the non-graph path. Push sync to the
   caller boundary.
7. **Deepstack injection is at LLM layers 0, 1, 2 — not at the vision
   `deepstack_visual_indexes` [5, 11, 17].** The vision indexes specify
   which VISION blocks produce the embeddings; the LLM injection is at
   the first N LLM layers (where N = number of deepstack embeddings).

## Layering rules

1. **Trait is the floor; concrete types are the ceiling.** Portable
   models use `dyn Backend`. Specialized models can downcast to
   `CudaBackend` for extra perf (e.g. batched GEMM).
2. **Layering is strict: model → backend, never backend → model.** The
   `DecodeGraphConfig` (primitives) + `DecodeGraphWeights` (`&Tensor`
   refs) pattern exists so the backend runs a decode without importing
   model types.
3. **Verify against reference every step, not at the end.** Per-layer
   dumps catch bugs in one step; final-token comparison is too weak.
4. **Reuse the primitives; don't re-invent them per model.** QK-norm is
   `rms_norm` on a reshape. The bar for adding a new kernel: is there any
   way to express this with existing kernels + a reshape?
5. **BF16 is storage/compute default; fp32 is for reductions.** Weights and
   activations are bf16; reductions such as RMSNorm variance use fp32
   accumulation. The backend sampler accepts f32, f16, or bf16 device logits
   and performs probability calculations with fp32 precision.

## Concrete example

The Qwen3-VL port is the reference implementation. See:
- `crates/apxinf-model/src/qwen3vl/` — the four-file recipe (+ vision)
- `crates/apxinf-model/examples/multimodal_check.rs` — verification tool
- `scripts/hf_reference_dump.py` — reference dump script
- `doc/20260619-qwen3vl/results.md` — verification numbers
- `doc/20260619-qwen3vl/notes.md` — live diary of decisions + bugs

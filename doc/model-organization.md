# apxinf-model Organization

Date: 2026-07-04
Status: Design doc for the per-model folder structure.

## The principle

**Each model's structure code lives in its own folder. Shared infrastructure
stays at the top level.**

```
apxinf-model/src/
  ── Shared infrastructure (model-agnostic) ──
  lib.rs              module wiring + re-exports
  llm_trait.rs        LlmTrait (prefill → decode → backend sampling → stream)
  auto.rs             AutoModel (unified frontend, picks best impl)
  registry.rs         model factory registry
  builtin.rs          register_builtin_models()
  profiling.rs        GenerationProfile (TTFT/TPOT tracking)
  debug.rs            DebugCapture / DebugConfig (activation capture)
  nvtx.rs             NVTX no-op stub / re-export

  ── Per-model folders ──
  llama/              Llama model structure
    mod.rs            re-exports
    weights.rs        LlamaWeights, TransformerLayer
    model.rs          LlamaModel (legacy CPU/CUDA impl + its KVCache)
    general.rs        GeneralLlama (dyn Backend impl, decode workspace)
    decode_graph.rs   DecodeGraph, DecodeWorkspace (allocation-free decode)

  qwen3vl/            Qwen3-VL model structure
    mod.rs
    config.rs         Qwen3VLConfig
    weights.rs        Qwen3VLTextWeights
    vision_weights.rs Qwen3VLVisionWeights
    vision.rs         vision tower forward
    general.rs        GeneralQwen3VL (unified text/image prefill + decode)
```

## What's shared vs model-specific

### Shared (top-level)

- **`LlmTrait`** — the shared autoregressive LLM/VLM process. Models
  implement token-level `forward`; request-level `prefill(LlmInput)` accepts
  optional image processor output, and `backend()` binds logits to the matching
  sampler. `generate_streaming_with_options` is shared (validate → prefill →
  penalties/filtering/selection → token-only decode loop → stream). The older
  `generate_streaming` is a greedy compatibility wrapper around that pipeline.
- **`AutoModel`** — unified frontend with one `load_model` entry point. It
  detects `config.json:model_type` by default, accepts an optional registry
  name override in `LoadOptions`, and picks the best device implementation.
- **`registry`** — factory registry for model constructors.
- **`GenerationProfile`** — timing instrumentation (TTFT, TPOT, tok/s).
- **`DebugCapture`** — activation capture for debugging.
- **`nvtx`** — profiling markers (no-op on non-CUDA).

### Model-specific (per-folder)

Each model folder contains:
- **`weights.rs`** — weight struct + `from_map` loader (HF key → tensor).
- **`model.rs`** — the model struct implementing `LlmTrait`. VLMs override
  request-level `prefill`; they do not need a separate generation method.
- **`decode_graph.rs`** (if the model has a fast decode path) — the
  allocation-free workspace + CUDA Graph capture, specific to that
  model's layer structure.
- Additional files as needed (vision tower, config, etc.).

## Why per-model folders

1. **Adding a new model is self-contained.** A new `mamba/` folder
   doesn't touch `llama/` or `qwen3vl/`. The shared infrastructure
   doesn't change.

2. **Model-specific fusion choices stay in the model.** The decode
   graph (packed QKV, flash attention, fused RMSNorm) is Llama-specific
   today — it lives in `llama/decode_graph.rs`. If Qwen3-VL gets its own
   decode graph, it's `qwen3vl/decode_graph.rs`.

3. **Clear boundary between "pipeline" and "architecture".** The shared
   `LlmTrait::generate_streaming_with_options` is the pipeline (prefill →
   decode → backend sampling → stream). The model folder is the architecture
   (how one forward pass works).

## VLM and VLA boundaries

VLM generation uses `LlmTrait` directly. `LlmInput` carries borrowed,
optional processor output to `prefill`; Qwen3-VL keeps mRoPE, deepstack, and
embedding scatter model-specific. See [adding a new model](adding-a-new-model.md)
for the current interface contract.

VLA models remain under the separate `VlaRuntime` interface because their
observation/action contract and generation process are not autoregressive text.
They share `RngKey` and the backend's standard-normal generator with the
sampling infrastructure, but continuous action latents do not pass through the
categorical token sampler. See
[`doc/20260819-sampling-subsystem`](20260819-sampling-subsystem/README.md).

## KVCache

The `KvCache` trait is shared (`apxinf-core`). `CudaKVCache` is shared
(`apxinf-cuda`). The legacy `KVCache` struct in `llama/model.rs` is a CPU
implementation used only by the legacy `LlamaModel` — it stays as an
implementation detail of that model, not a shared type.

# apxinf-model Organization

Status: current source map. Use [Model Layer Architecture](model-layer-architecture.md#current-module-names-and-responsibilities)
for authoritative responsibility/dependency rules and
[Adding a New Model](adding-a-new-model.md) for the integration procedure.

## Shared contracts and construction

Under `crates/apxinf-model/src/`:

| Location | Responsibility |
| --- | --- |
| `lib.rs`, `builtin.rs`, `registry.rs` | Exports, built-in registrations and loader lookup |
| `auto.rs` | `AutoModel` factory, `LoadOptions`, `LoadedModel::{Text,Vla}` result |
| `llm_trait.rs`, `generation_config.rs` | Autoregressive LLM/VLM input, generation, sampling options and output contracts |
| `vla/mod.rs` | `VlaRuntime`, observation/request/action, prepared inference, execution policy/status |
| `accelerator.rs` | Backend creation for shared loading |
| `profiling.rs`, `debug.rs`, `nvtx.rs` | Timing and diagnostic mechanisms |

`AutoModel` selects a loader; `LoadedModel` holds the resulting family interface.
Neither is a worker or another model-forward implementation. Python `AutoPolicy`
selects a policy; the existing PyO3 `ModelRunner` invokes Rust loading/inference.
There is no Python `AutoModel` binding or Python PI0.5 network class.

## Family implementations

| Directory | Current organization |
| --- | --- |
| `pi05/` | `load.rs` and `config.rs` construct `Pi05ModelRunner`; `model/` holds `Pi05Model<B>`, Blocks and `ModelVariant`; `model_runner/` owns preparation/cache/resources; `weights/` owns host mapping and device representations |
| `walloss/` | Existing BF16 runtime/executor, `fp8.rs`, schedule/geometry and weight files |
| `gr00t/` | Existing `vla_runtime.rs`, shared `executor.rs`, precision runtime/executor and weight files, plus a private `backbone/` |
| `llama/` | `GeneralLlama` in `general.rs`, family weights and decode graph; legacy `LlamaModel` remains in `model.rs` |
| `qwen3vl/` | `GeneralQwen3VL` in `general.rs`, text/vision weights, vision computation and family-specific multimodal/decode state |

New VLA code names forward computation `Model` and execution ownership
`ModelRunner`. These are roles inside the family, not new shared base classes.
Use one implementation directly when no variant dispatch is needed. LLM/VLM
continue to implement `LlmTrait`; sharing the word "model" does not turn their
autoregressive generation into VLA action inference.

Each family is self-contained. Inspect/copy a close implementation when useful,
then rename concepts and validate independently. Shared infrastructure stays
above family directories; no family imports another family's private model,
weights, runner, graph or backbone. Existing `*_runtime.rs`/`*_executor.rs`
filenames in WallOSS/GR00T are current code, not a reason to recreate PI0.5's
removed files or claim all families have migrated.

## How to navigate a change

- Forward order, modality connections, flow schedule: model computation.
- Layer fusion and intermediate physical layout: Blocks and its weight representation.
- Checkpoint keys, packing or calibration scales: weights and loading.
- Input/RNG binding, workspace allocation, capture, plan cache or invalidation: runner/preparation.
- Prompt, state/action normalization, tokenizer and output context: Python policy/processors.
- Model-neutral device operation: safe backend API and its provider implementation.

For PI0.5's exact paths and callable interfaces, see the
[current component view](model-lifecycle/architecture.md#implemented-pi05-pilot-stage-2)
and [preparation contract](model-lifecycle/lifecycle.md#implemented-pi05-preparation-contract).
The [current coverage table](model-layer-architecture.md#current-coverage-and-port-decisions)
records which capabilities are still family-specific.

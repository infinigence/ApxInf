# apxinf-model Layer Architecture

This document defines ownership and dependency rules for
`crates/apxinf-model/`. Read it when adding a model, adding a precision path, or
changing model execution topology. For model-instance, session and request lifetimes,
stage contracts and their source ownership, see
[Model Lifecycle and Contracts](model-lifecycle.md).

## Responsibility

The model layer turns a checkpoint into a repeatedly callable inference
function. It owns:

- model structure and layer ordering;
- checkpoint interpretation and device weight layout;
- schedules, caches, workspaces, and execution preparation;
- selection among valid precision and fusion paths.

It does not own raw device kernels, arbitrary application observations, robot
adapter semantics, or policy-level prompt, normalization, and action contracts.
It may own checkpoint-fixed conversion from a declared model input
representation to the tensors consumed by the model.

```text
Policy / application adapter
  container decode, robot field mapping, resize to the declared profile,
  prompt/token construction, policy normalization and action postprocessing
                         |
                         v
apxinf-model
  declared tensor/RGB input, checkpoint-fixed tensor canonicalization,
  architecture, weights, schedules and execution orchestration
                         |
                         v
apxinf-core / apxinf-cuda
  tensors, devices, model-neutral operators, kernel APIs
```

Dependencies flow downward. Backend crates never import model concepts.

## Runtime contracts

`LlmTrait` is the shared autoregressive LLM/VLM process. A VLM extends prefill
semantics but continues through the common categorical generation pipeline.

`VlaRuntime` is the observation-to-action process. It owns continuous action
generation, stochastic inputs, schedules, and prepared inference contracts that
do not fit token sampling.

The contracts may share model-neutral tensor or RNG facilities. They should not
be unified merely because both accept language or images.

## Current module names and responsibilities

Use this table when assigning new model code. The source links show the PI0.5
implementation; they are examples to inspect, not cross-family imports. A small
family may combine files while preserving these owners. Add a trait, enum or
wrapper only when the implementation needs it; matching PI0.5's file count is
not a port requirement.

| Owner / name | Inputs and output; responsibility | PI0.5 or shared source |
| --- | --- | --- |
| Python policy (`<Family>Policy`) | Public observation → canonical native inputs → decoded actions; prompt, tokenizer, state/action normalization and output context | [Pi05Policy](../python/apxinf/apxinf/policies/impls/pi05.py) |
| Native binding (`ModelRunner`) | NumPy/Rust conversion, public shape validation, default sampling keys, host result conversion; delegates family computation to `VlaRuntime` | [PyO3 ModelRunner](../crates/apxinf-py/src/lib.rs) |
| Loading factory (`AutoModel`) and result (`LoadedModel`) | Resolve a registry entry, invoke its loader, return `Text` or `Vla`; these do not add another inference loop | [auto.rs](../crates/apxinf-model/src/auto.rs), [registry.rs](../crates/apxinf-model/src/registry.rs) |
| `config.rs`, `load.rs` | Validate family config; select a supported implementation, materialize weights and fixed embeddings, construct the runner through its constructor | [load.rs](../crates/apxinf-model/src/pi05/load.rs) |
| `model_runner/` (`<Family>ModelRunner`) | Implements `VlaRuntime`; validates requests, owns preparation policy, input/RNG binding, bounded cache and invalidation | [runner.rs](../crates/apxinf-model/src/pi05/model_runner/runner.rs) |
| `model_runner/prepare.rs` and prepared inference | Query requirements, allocate stable inputs/output/workspace, warm up, capture and retain resources; compatible `run` executes without capture/autotune | [prepare.rs](../crates/apxinf-model/src/pi05/model_runner/prepare.rs) |
| `model/` (`<Family>Model<B>`) | Canonical tensors → model-domain result; shared forward order, modality connections and fixed flow schedule | [model/mod.rs](../crates/apxinf-model/src/pi05/model/mod.rs) |
| `model/model.rs` (`ModelVariant`) | Private dispatch among constructed precision-specific models and fixed time embeddings; does not own graph/cache policy | [model.rs](../crates/apxinf-model/src/pi05/model/model.rs) |
| `model/blocks/` (`Blocks`, `PrepareBlocks`) | Backbone/layer composition, physical intermediate representations and precision-specific fusion; reports workspace needs without allocating the graph workspace | [blocks/mod.rs](../crates/apxinf-model/src/pi05/model/blocks/mod.rs) |
| `weights/` | Checkpoint mapping, model-local packing, device weight trees and fixed calibration scales; fixed weight operations stay with their physical representation | [weights/mod.rs](../crates/apxinf-model/src/pi05/weights/mod.rs) |
| `backend.rs` | Concentrates imports/type aliases for safe CUDA resources and operations; it is not an execution engine or provider abstraction | [backend.rs](../crates/apxinf-model/src/pi05/backend.rs) |
| `math.rs` (when needed) | CUDA-independent helpers/reference semantics; PI0.5 loading uses its time embedding, while prompt/Euler helpers are CPU references | [math.rs](../crates/apxinf-model/src/pi05/math.rs) |
| `apxinf-cuda` | Model-neutral operations, dispatch, allocation and CUDA Graph mechanisms; kernel implementations and vendor calls remain behind safe APIs | [CUDA crate](../crates/apxinf-cuda/src/lib.rs) |

`model_variant` is the shared loading field and CLI option `--model-variant`.
`ModelVariantChoice` is a configuration choice; `ModelVariant` is a loaded private
enum; `LoadedModel` is the cross-family loading result. They have different
lifetimes and are not three names for a model. A single-implementation family
does not need a precision-dispatch enum. `StepModulation` is computed per-step
scale/shift/gate data, distinct from learned modulation weights. Keep model
semantics such as `prefix` and `denoise`; flow steps are not token decode steps.

The native binding and the family runner are existing language-boundary and
execution owners. `apxinf.ModelRunner` re-exports `apxinf_py.ModelRunner`; there
is no separate Python `Pi05Model` or Python `AutoModel`. Python `AutoPolicy`
selects a policy; that policy calls the binding, which invokes Rust `AutoModel`.
Reuse these entry points for a new family instead of adding a factory/runner
wrapper merely to reproduce the names in this table.

### Dependency direction within a family

```text
load → weights + model construction + model_runner constructor
model_runner → model interface / ModelVariant → Blocks → weights
model_runner, Blocks, weights → safe backend resources / operations
```

The model does not depend on `VlaRequest`, `InferenceSpec`, graph policy or
runner types. Weights do not depend on model/runner types. The runner invokes
the model interface and typed dispatch; precision-specific matches and concrete
Blocks construction stay in the model construction/variant module. It allocates
from model-reported requirements rather than encoding a second workspace-size
formula. Config and resource descriptions are leaf data. PI0.5 enforces these
directions with [check_pi05_module_boundaries.py](../scripts/check_pi05_module_boundaries.py).
That guard checks PI0.5 only; a new family must enforce its own declared boundaries.

### Current coverage and port decisions

| Family / contract | Current organization and limits |
| --- | --- |
| PI0.5 / `VlaRuntime` | `model/`, `model_runner/`, `weights/`; explicit preparation policy/status, stale-plan checks and retained-resource tests; `model_variant` selects `auto`, `bf16`, `fp8_static`, `int8_dynamic` |
| Qwen-Drive / `VlaRuntime` | Planning-only `model/`, `model_runner/`, `weights/`; one BF16 Blocks file; direct planning prepares the whole model and replays a captured graph, while variable-length reasoning stays eager with local GDN graphs. See [family contract](qwen-drive-planning.md). |
| WallOSS / `VlaRuntime` | Existing `bf16_runtime.rs`, `bf16_executor.rs`, `fp8.rs` and weight files; not migrated to PI0.5's runner/variant or explicit preparation contract |
| GR00T / `VlaRuntime` | Existing `vla_runtime.rs`, `executor.rs`, precision runtime/executor files and private `backbone/`; not migrated to PI0.5's explicit preparation contract |
| Llama, Qwen3-VL / `LlmTrait` | Existing `general.rs` and family-specific state/decode graph paths; shared autoregressive generation remains in `LlmTrait`, not the VLA runner |

New VLA code should use `Model` for forward computation and `ModelRunner` for
execution ownership. Existing family symbols remain their actual names until
separately migrated; do not rename or import them as part of an unrelated port.
`VlaRuntime` default preparation-policy methods return unsupported, and default
`PreparedInference::status` is `RuntimeManaged`. A successful legacy `prepare`
does not prove graph readiness or PI0.5-equivalent guarantees.

The common loader currently accepts `model_variant` for PI0.5 and Qwen-Drive registry
names. Supporting it for a new family requires updating that admission check
and implementing family-local parsing/validation; registration alone is not
enough. `LoadOptions.config`, Python `config_json`/shape overrides and
`ModelRunner.random` are currently PI0.5-specific. A new family's config belongs
in that family and its loader; use existing named assets where applicable and
extend a shared option only for a demonstrated contract, not by copying
`Pi05Config` or adding unrelated fields to it.

See [Adding a New Model](adding-a-new-model.md#registration-and-public-integration)
for the exact integration sequence and
[PI0.5 preparation](model-lifecycle/lifecycle.md#implemented-pi05-preparation-contract)
for implemented signatures and output-lifetime rules. Capture acceptance for a
new accelerator port is defined in
[Model Execution Wiring](model-execution-wiring.md#accelerator-port-acceptance);
it is not a claim that every existing family has already passed those gates.

## VLA input seam and preprocessing ownership

Separate application and policy preprocessing from model tensor
canonicalization. They have different interfaces, owners, and lifecycle
requirements.

The policy or application adapter owns:

- decoding arbitrary image containers and observation objects;
- mapping robot-specific fields, cameras, joint order, units, and masks into
  the policy's canonical observation;
- image resize or padding when it is not part of the runtime's declared input
  representation;
- experimental and user-defined transforms;
- prompt/token construction, state normalization or discretization, and other
  checkpoint policy semantics; and
- trimming, unnormalizing, and adapting model actions for the caller or robot.

The model runtime owns validation and execution after its declared input seam.
A VLA may declare precomputed patches, resized RGB `u8`, or both as supported
vision representations. If it advertises resized RGB through `VlaContract`,
the input is already decoded and resized to the declared profile. The runtime
may then own checkpoint-fixed pixel normalization, temporal duplication,
patchification, spatial merge or ordering, layout conversion, and dtype
conversion. Keeping those operations on the device allows them to share fixed
buffers and CUDA Graph lifecycle with the model.

This ownership does not turn `VlaRuntime` into a high-level policy. In
particular, image normalization that converts declared RGB into model patches
may be runtime-owned, while state/action normalization, prompt construction,
tokenization, robot mapping, and action postprocessing remain policy- or
adapter-owned. PI0.5 and WallOSS both use this distinction: their native RGB
paths perform model tensor canonicalization in Rust/CUDA, while their Python
policies still construct the model inputs and interpret the model actions.

## Five responsibilities inside a model

These are structural responsibilities within the model layer, not sequential
lifecycle stages. The temporal stages are defined in
[Model Lifecycle and Contracts](model-lifecycle.md#seven-stages). Use these roles
to understand a directory rather than requiring a fixed file recipe:

1. **Frontage and contract** — registration, configuration, public runtime
   implementation and capability declaration.
2. **Weight pipeline** — checkpoint keys, transformations, device upload,
   precision-specific representations and tied storage.
3. **Model composition** — model mathematics, layer ordering, residuals,
   attention, conditioning and schedules.
4. **Model runner** — caches, workspaces, prepared shapes, graph capture
   and repeated invocation.
5. **Specification and budget** — derived dimensions, static limits, workspace
   sizing and dispatch invariants.

Files may combine responsibilities in a small first implementation. Split them
when a responsibility becomes independently changeable, not to satisfy a
template.

## Per-model isolation

Each architecture lives under `src/<model>/`. A new model may copy the nearest
implementation to establish a correct vertical slice. It must not grow inside
another model's directory.

Treat a model-family directory as a private architecture module, not as a reuse
library. During an initial port, dependencies follow this matrix:

| Dependency | Allowed |
|---|---|
| `<new_model>` → `apxinf-core` contracts | yes |
| `<new_model>` → safe `apxinf-cuda` kernel interfaces | yes |
| `<new_model>` → explicitly shared top-level model modules | yes |
| `<new_model>` → another model-family directory | no |
| another model-family directory changed to accommodate `<new_model>` | no by default; requires a separate shared-seam design and review |

Use another maintained model implementation as evidence for layer ordering, fusion choices,
workspace lifetimes, and safe CUDA calls. Copy the architecture-specific code
needed by the new model and rename its concepts locally. Reusing an optimized
runtime means reusing those proven patterns and model-neutral interfaces; it
does not mean importing the other family's config, weights, model, runner,
backend seam, cache, or graph modules.

Extract a shared architecture module only after both model implementations are
maintained, independently tested, and demonstrate the same stable semantics and
lifecycle. Make that extraction a separately reviewable design change. A new
port alone is not evidence for moving code out of either family.

When a family already has `backend.rs`, import its concrete CUDA resources
and safe operations through that file. PI0.5 uses it as a re-export/type-alias
module, not an abstract backend provider. A new family need not add a wrapper
with forwarding methods to satisfy this convention. This keeps
accelerator changes from rewriting the directory topology.

## Model/backend boundary

The backend exposes tensors, device movement, and model-neutral operations. The
model composes them into an architecture.

If a name describes a device operation or kernel API, it may belong in the
backend. If it describes a layer, residual path, decode step, action head,
schedule, or model family, it belongs in the model layer.

Portable models use `dyn Backend`. Optimized runtimes may use concrete backend
facilities for fusion, graph capture, or transfers that should not enlarge the
portable trait. Trait is the floor; concrete types are the ceiling.

ApxInf currently concentrates on CUDA backends, and the maintained model set
does not yet provide enough repeated fusion cases to justify a stable abstract
interface for every high-performance composition. Following YAGNI, broadly
useful primitive operations may live on `Backend`, while optimized CUDA model
paths call safe model-neutral fused functions through the model directory's
CUDA seam. This direct safe call is intentional; raw FFI remains forbidden.

Revisit that boundary when multiple maintained models or hardware backends need
the same semantic fusion and lifecycle contract. At that point, extract the
smallest common interface supported by those implementations instead of
forecasting variants through optional flags today. Moving a proven fusion
behind a trait later is an architectural evolution, not a prerequisite for its
first correct optimized use.

A fused mega-kernel still lives in the backend as a kernel implementation, but
the model chooses when its semantics match. The backend does not import the
model type.

## Dependency rules

- Registration may assemble configuration, weights, model, and runtime.
- Weight code may depend on configuration and model-neutral transfer or
  quantization facilities; it does not depend on model/runner types.
- Model composition reads weights and dimensions; it does not depend on a
  runner or capture implementation.
- Model runners invoke a narrow model interface or model-owned function;
  they do not recreate model mathematics.
- Shape and budget definitions remain leaf concepts.
- Debugging and profiling may cross these responsibilities without owning
  correctness semantics.

## YAGNI boundary

One implementation is evidence for a model, not evidence for an abstraction.
Prefer local duplication while architecture, shapes, precision behavior, or
lifecycle are still moving.

Refactor after repeated maintained implementations demonstrate an identical
seam. The extracted module must be model-neutral, reduce dependency surface,
and avoid family switches. If callers need many options to recover their old
behavior, the commonality is not stable enough.

## Review checks

- The new architecture has its own directory.
- The product diff neither imports nor modifies another model-family directory
  for the new architecture. Any exception points to an explicitly shared module
  and a separately approved extraction design.
- No backend crate imports model types or model-family concepts.
- Model code reaches CUDA through its declared seam.
- Weight transformations occur at load time where possible.
- Model mathematics has one owner.
- Prepared execution binds every allocation/dispatch-relevant shape.
- Shared code is backed by repeated maintained use, not a forecast.

Run `scripts/check_model_family_boundaries.sh` to reject direct Rust references
from one model-family directory into another. The check complements review: it
cannot determine why an existing family was modified. The
`apxinf-model` integration test runs the same check during the normal Rust test
suite.

## PI0.5 migration pilot

PI0.5 uses one statically dispatched `Pi05Model` with bf16/fp8_static/int8_dynamic
Blocks. `load.rs` selects model_variant and materializes fixed assets;
`model_runner/runner.rs` owns private request state and plan validity;
`model_runner/prepare.rs` owns the shared
warmup/capture path and graph resources. `weights/` groups checkpoint mapping,
parallel device representations and fixed calibration data. Runtime compatibility
files and aliases have been removed. Typed computation dispatch and BF16 calibration
are private to `model/`. Preparation resource contracts also belong to `model`;
`model_runner` owns allocation/capture and invokes calculation without a reverse
model-to-runner dependency. Python Policy retains encode/decode context and holds
the native `ModelRunner` binding. `AutoPolicy` and Rust `AutoModel` are separate
construction entry points; `LoadedModel` remains the Rust loading-result enum.
See the [implemented component view](model-lifecycle/architecture.md#implemented-pi05-pilot-stage-2),
[callable lifecycle contract](model-lifecycle/lifecycle.md#implemented-pi05-preparation-contract)
and [migration tracker](model-lifecycle/migration.md) for qualification and scope.
WallOSS and GR00T retain their previous implementations pending their own stages.

# Model lifecycle refactor: architecture proposal

Status: proposal; not the current implementation or an accepted ADR.

This proposal separates model semantics from execution resource management.
Policy owns end-to-end input/output semantics; Network owns model mathematics;
ExecutionSession owns preparation and repeated execution. It preserves direct
safe CUDA kernel calls and precision specialization.

The implementation baseline is `7126992` on upstream main. PI0.5 and WallOSS
observations below refer to that baseline. GR00T observations refer separately
to [PR #42](https://github.com/infinigence/ApxInf/pull/42), head
`01ad172474dfcc3d00c8fae3b5970e387ae1eb25`; they are not claims about merged code.
The PR author account is `Casten-Wang`. Its performance evidence has not been
reproduced for this design document.

Read [lifecycle contracts and migration](lifecycle.md) for stage guarantees,
resource lifetimes, acceptance criteria, and implementation slices. Existing
[model-layer architecture](../model-layer-architecture.md) remains the current
reference until a separately reviewed implementation changes it.

## Current logical view

```mermaid
flowchart TB
    O[Images, state, instruction] --> P[PI0.5 Policy and processors]
    O --> W[WallOSS Policy and processor]
    O --> G[GR00T Policy and NVIDIA processor adapter - PR]
    P --> V[Generic VLA input and native Model]
    W --> V
    G --> GI[Gr00tObservation and Gr00tModel - PR]
    V --> PR[PI0.5 runtimes: network, solver, buffers, graph]
    V --> WR[WallOSS runtime: loading, network, buffers, graph]
    GI --> GR[GR00T runtime: weights, network, buffers, graph - PR]
    PR --> PE[Precision executors: mostly layer computations]
    WR --> WE[Executor: layers and larger network computations]
    GR --> Q[Qwen3-VL internals - PR dependency]
    PE --> K[Shared safe CUDA kernels]
    WE --> K
    Q --> K
    GR --> K
```

The directory isolation is useful, but `runtime` has no consistent narrow
meaning. PI0.5 and WallOSS both keep network composition and solver orchestration
inside runtime files. GR00T's roughly 3,052-line runtime also holds device weight
conversion and capture management. Its `execution.rs` describes attention
selection and token grouping: those are network semantics, despite the name.

The generic VLA contract does not express all GR00T inputs. Its PR deliberately
uses a separate input and native entry point instead of widening that contract.
This is evidence for a common lifecycle protocol with typed model-specific
payloads, not for a universal optional-field tensor bag.

## Target logical view

```mermaid
flowchart TB
    O[External observation] --> A[Adapter: external fields and conventions]
    subgraph POLICY[Policy: one end-to-end request]
        A --> E[Input Processor: prompt, tokenizer, images, state]
        E --> I[EncodedInput: model input payload]
        E --> C[DecodeContext: request-local decoding information]
        I --> S[ExecutionSession: prepare, bind, run, invalidate]
        S --> R[ModelOutput]
        R --> D[Output Processor: denormalize and interpret actions]
        C --> D
    end
    L[Load model package] --> M[Model: configuration, device weights, constants]
    L --> E
    L --> D
    M --> S
    S -->|eager or capture| N[Network: topology and layer mathematics]
    N --> T[Schedule: model-specific solver mathematics]
    N --> K[Safe kernel interfaces]
    S -->|captured execution| G[Graph replay]
    D --> OUT[Deployable actions]
```

| Module | Owns | Does not own |
| --- | --- | --- |
| Adapter | External field names, recording conventions and coordinate mappings | Tokenization, neural network mathematics |
| Policy | Encode/run/decode orchestration and request context | CUDA buffer layouts |
| Processor | Prompt templates, tokenizer use, image/state encoding, output decoding | Capture lifecycle or tactic invalidation |
| Loader / Model | Checkpoint interpretation, device weights, fixed assets and capabilities | Mutable per-request state |
| Network / Schedule | Layer ordering, conditioning, solver rules and precision specialization | Session cache policy or capture recovery |
| ExecutionSession | Input binding, workspaces, KV/latent state, RNG execution, capture/replay and invalidation | Prompt meaning or duplicated model mathematics |
| Backend | Model-neutral kernels, device memory and graph facilities | Family-specific scheduling decisions |

`DecodeContext` is request-local. GR00T's processor decodes actions using the
original state from the same request. Preserve this explicit dependency rather
than using mutable global "last observation" state. Other models may use an
empty context or retain state needed by relative-action transforms.

CPU/GPU placement does not determine semantic ownership. A Processor may define
an image normalization/patchification operation implemented by a backend kernel
and scheduled inside a captured Session. The formula has one owner while the
Session owns buffers and execution. Do not force CPU processing or materialize
an unnecessary host intermediate to satisfy the diagram.

## Current development view

Paths below are relative to the repository root. GR00T entries exist in PR #42.

```text
python/apxinf/apxinf/
  policies/impls/
    pi05.py             policy, loading and processor composition
    walloss.py          policy and model processor
    gr00t.py [PR]       policy and NVIDIA processor adapter
  processors/           shared processing utilities
  checkpoints/          checkpoint layouts and assets
  adapters/             external integrations

crates/apxinf-py/src/
  lib.rs                generic VLA binding, PI0.5 options, GR00T class in PR

crates/apxinf-model/src/
  auto.rs / registry.rs / builtin.rs
  vla/mod.rs            common VLA interfaces
  pi05/
    *_executor.rs       layer computations
    *_runtime.rs        network composition, resources, capture
    vla_runtime.rs      input routing, variants, prepared cache
  walloss/
    bf16_executor.rs    BF16 and dynamic FP8 computation
    bf16_runtime.rs     loading, network composition, execution
  gr00t/ [PR]
    input.rs            dedicated observation and inference specification
    execution.rs        attention semantics and token grouping
    checkpoint.rs / weights.rs / math.rs
    runtime.rs          multiple responsibilities
```

## Target development view

This is a responsibility map, not a requirement to create one class, trait or
folder per item. Small implementations can combine files without combining
ownership. Preserve public imports with temporary forwarding exports when needed.

```text
python/apxinf/apxinf/
  policies/
    base.py                      end-to-end request guarantees
    impls/
      <pi05 | walloss | gr00t>/
        policy.py                encode -> run -> decode
        processor.py             model input/output semantics
        assets.py                tokenizer, templates, statistics, adapters
  processors/                    proven shared transformations
  checkpoints/                   shared file-layout detection
  adapters/                      external conventions and integrations

crates/apxinf-py/src/
  lib.rs                         exports and registration
  models/                        thin typed model bindings

crates/apxinf-model/src/
  auto.rs / registry.rs / builtin.rs
  vla/                           common lifecycle guarantees and capabilities
  execution/
    capture.rs                   proven common capture/error-cleanup mechanism
    lifecycle.rs                 readiness, strategy and invalidation
  <pi05 | walloss | gr00t>/
    model.rs                     configuration, weights, capabilities
    loader.rs / weights/         checkpoint -> device representations
    input.rs                     typed inputs, validation, specification derivation
    network/                     layers, full computation, precision variants
    schedule.rs                  solver mathematics where separately useful
    execution.rs                 model session, buffers and request binding

crates/apxinf-cuda/
  kernels/ / workspace.rs / graph.rs   existing device mechanisms
```

```mermaid
flowchart LR
    R[Current runtime] --> L[Loading to loader and weights]
    R --> M[Fixed resource ownership to model]
    R --> N[Mathematics to network and schedule]
    R --> I[Input validation to input]
    R --> E[Mutable resources and execution to session]
    E --> C[Repeated mechanisms to shared execution]
```

Dependency rules:

- Loader assembles Model; weights do not depend on Session.
- Session references Model and invokes Network; Network does not inspect
  serving state, graph cache policy, or capture strategy.
- Eager and captured execution use the same maintained network body.
- Model-local execution owns model-specific buffer layouts. Shared execution
  contains mechanisms supported by multiple implementations, without family switches.
- Raw CUDA mechanisms remain in `apxinf-cuda`; do not build a duplicate allocator
  or graph abstraction solely for this reorganization.
- Keep LLM/VLM token-generation and VLA action-generation contracts distinct.
  The initial implementation scope is these three VLA families.

## GR00T backbone reuse

PR #42 imports Qwen3-VL internals and modifies the family dependency checker to
allow that edge. This differs from main's copy-first family-isolation policy.
Do not silently treat the PR exception as an accepted architecture decision.

The proposed resolution is a deliberately reviewed backbone interface exposing
only required configuration/loading and feature computation. Start by narrowing
exports within Qwen3-VL; extract a shared backbone module only if both maintained
callers justify it. GR00T should not depend on arbitrary Qwen3-VL internal files.

```mermaid
flowchart LR
    subgraph CURRENT[PR implementation]
        G1[GR00T runtime] --> Q1[Qwen3-VL internals]
    end
    subgraph TARGET[Proposed explicit reuse]
        G2[GR00T Network] --> B[Reviewed backbone interface]
        Q2[Standalone Qwen3-VL] --> B
        B --> Q3[Backbone implementation]
    end
```

## Evidence anchors

- `crates/apxinf-model/src/vla/mod.rs`: current Observation, InferenceSpec,
  VlaContract and PreparedInference.
- `crates/apxinf-model/src/pi05/vla_runtime.rs`: prepare-time capture, eager
  fallback, tactic-generation invalidation and cache replacement.
- `crates/apxinf-model/src/walloss/bf16_runtime.rs`: first-run capture, fixed
  workspace budget and captured latent-mode constraint.
- `python/apxinf/apxinf/policies/impls/{pi05,walloss}.py`: current encoding and decoding.
- PR #42 `gr00t/input.rs`, `gr00t/runtime.rs`, `policies/impls/gr00t.py` and
  `scripts/check_model_family_boundaries.sh`: specialized inputs, capture key,
  request-local decode state and the proposed Qwen3-VL dependency exception.

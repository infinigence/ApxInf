# apxinf-model Layer Architecture

This document defines ownership and dependency rules for
`crates/apxinf-model/`. Read it when adding a model, adding a precision path, or
changing model execution topology.

## Responsibility

The model layer turns a checkpoint into a repeatedly callable inference
function. It owns:

- model structure and layer ordering;
- checkpoint interpretation and device weight layout;
- schedules, caches, workspaces, and execution preparation;
- selection among valid precision and fusion paths.

It does not own raw device kernels or Python-side preprocessing and policy
semantics.

```text
Python policy
  preprocessing, normalization, public observation/action contract
                         |
                         v
apxinf-model
  architecture, weights, schedules, execution orchestration
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

## Five responsibilities inside a model

Understand a model directory through the inference lifecycle rather than a
fixed file recipe:

1. **Frontage and contract** — registration, configuration, public runtime
   implementation and capability declaration.
2. **Weight pipeline** — checkpoint keys, transformations, device upload,
   precision-specific representations and tied storage.
3. **Network composition** — model mathematics, layer ordering, residuals,
   attention, conditioning and schedules.
4. **Execution carrier** — caches, workspaces, prepared shapes, graph capture
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

The directory's `backend.rs` is the only CUDA-facing seam when concrete device
facilities are required. Other files import through that seam. This keeps
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

- Registration may assemble configuration, weights, network, and runtime.
- Weight code may depend on configuration and model-neutral transfer or
  quantization facilities; it does not depend on the executor.
- Network composition reads weights and dimensions; it does not depend on a
  captured-runtime implementation.
- Execution carriers invoke a narrow network interface or model-owned function;
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
- No backend crate imports model types or model-family concepts.
- Model code reaches CUDA through its declared seam.
- Weight transformations occur at load time where possible.
- Network mathematics has one owner.
- Prepared execution binds every allocation/dispatch-relevant shape.
- Shared code is backed by repeated maintained use, not a forecast.

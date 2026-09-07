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

It does not own raw device kernels, arbitrary application observations, robot
adapter semantics, or policy-level prompt, normalization, and action contracts.
It may own checkpoint-fixed conversion from a declared model input
representation to the tensors consumed by the network.

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
buffers and CUDA Graph lifecycle with the network.

This ownership does not turn `VlaRuntime` into a high-level policy. In
particular, image normalization that converts declared RGB into model patches
may be runtime-owned, while state/action normalization, prompt construction,
tokenization, robot mapping, and action postprocessing remain policy- or
adapter-owned. PI0.5 and WallOSS both use this distinction: their native RGB
paths perform model tensor canonicalization in Rust/CUDA, while their Python
policies still construct the model inputs and interpret the model actions.

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

Treat a model-family directory as a private architecture module, not as a reuse
library. During an initial port, dependencies follow this matrix:

| Dependency | Allowed |
|---|---|
| `<new_model>` → `apxinf-core` contracts | yes |
| `<new_model>` → safe `apxinf-cuda` kernel interfaces | yes |
| `<new_model>` → explicitly shared top-level model modules | yes |
| `<new_model>` → another model-family directory | no |
| another model-family directory changed to accommodate `<new_model>` | no by default; requires a separate shared-seam design and review |

Use another maintained executor as evidence for layer ordering, fusion choices,
workspace lifetimes, and safe CUDA calls. Copy the architecture-specific code
needed by the new model and rename its concepts locally. Reusing an optimized
runtime means reusing those proven patterns and model-neutral interfaces; it
does not mean importing the other family's config, weights, model, executor,
backend seam, cache, or graph modules.

Extract a shared architecture module only after both model implementations are
maintained, independently tested, and demonstrate the same stable semantics and
lifecycle. Make that extraction a separately reviewable design change. A new
port alone is not evidence for moving code out of either family.

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
- The product diff neither imports nor modifies another model-family directory
  for the new architecture. Any exception points to an explicitly shared module
  and a separately approved extraction design.
- No backend crate imports model types or model-family concepts.
- Model code reaches CUDA through its declared seam.
- Weight transformations occur at load time where possible.
- Network mathematics has one owner.
- Prepared execution binds every allocation/dispatch-relevant shape.
- Shared code is backed by repeated maintained use, not a forecast.

Run `scripts/check_model_family_boundaries.sh` to reject direct Rust references
from one model-family directory into another. The check complements review: it
cannot determine why an existing family was modified. The
`apxinf-model` integration test runs the same check during the normal Rust test
suite.

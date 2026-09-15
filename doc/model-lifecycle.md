# Model Lifecycle and Contracts

This document defines how an inference integration moves from checkpoint assets
to repeated requests and resource release. It applies to LLM, VLM, VLA and future
supported model families. Use it to decide stage ownership, data contracts and
code placement when adding or changing a model.

The stages are architectural responsibilities, not a new shared Rust trait or a
claim that every integration already implements an identical lifecycle API.
Implementations must declare their supported paths and limitations. A model does
not need a separate architecture document or one source file per stage.

Read [Model Layer Architecture](model-layer-architecture.md) for dependency and
family-isolation rules, [Model Execution Wiring](model-execution-wiring.md) for
device execution and allocation details, and [Adding a New Model](adding-a-new-model.md)
for the integration procedure. Those documents remain the detailed authorities
for their respective topics; this document connects their responsibilities over
time. Development-time porting and release qualification are described in
[Porting Workflow](porting-workflow.md).

## Three lifetimes

An integration has several overlapping lifetimes:

| Lifetime | Typical owned state | End or invalidation |
| --- | --- | --- |
| Model instance | validated configuration, weights, tokenizer assets, fixed transforms, device context and reusable execution profiles | unload, replacement of assets, or incompatible device/precision changes |
| Session or sequence | autoregressive KV state, sampling position, robot episode state where supported | explicit reset, sequence end, or replacement of the model instance |
| Request | instruction/messages, images, state, masks, noise, request-local buffers and outputs | completion or failure after outstanding device work is handled |

Prepared profiles belong to the instance but can contain mutable buffers used
by requests. Sharing an instance therefore does not by itself imply concurrent
request safety. Each runtime must state whether calls are serialized or require
independent sessions/profiles. Session reset must not be confused with unloading
weights, and request completion must not imply destruction of reusable graphs.

## Seven stages

Stage order follows data dependencies. In particular, a shape-specialized
runtime may need canonical request metadata before it can prepare an execution
profile. A known profile can also be prepared at startup.

```text
1. Resolve and validate assets -> 2. Load and initialize
                                          |
application input -> 3. Canonicalize request
                                          |
                     4. Select / prepare execution profile
                                          |
                     5. Execute -> 6. Interpret output -> caller
                            ^                 |
                            +-- next token ---+  when generation continues

Requests repeat stages 3-6 while the instance remains loaded.
7. Reset session state or release the instance at the appropriate lifetime end.
```

| Stage | Input -> output contract | Owner | Frequency |
| --- | --- | --- | --- |
| 1. Resolve and validate assets | asset locations and explicit overrides -> compatible asset identities, configuration and selected execution policy | loader mechanisms plus family-specific interpretation | load/reload |
| 2. Load and initialize | validated assets -> usable runtime, device weights and declared capabilities | model runtime; higher-level policy/generation layer loads its own text and output transforms | instance creation |
| 3. Canonicalize request | application input -> family-specific canonical model inputs and routing metadata | application adapter followed by policy/generation input logic | request or prefill |
| 4. Select / prepare execution | runtime plus execution-relevant metadata -> compatible plans, caches, workspace and optionally a captured graph | runtime/execution carrier | profile miss or invalidation; otherwise reuse |
| 5. Execute | prepared state plus current canonical inputs/session state -> model-domain result and state transition | family network and runtime using safe backend operations | request, decode step or solver iteration |
| 6. Interpret output | model-domain result -> public result, stream item or continuation decision | family policy/generation output logic, then application adapter | result or generated step |
| 7. Reset / release | active state plus reset/close intent -> cleared session or retired resources | owning session/runtime; adapters close only what they own | session end, cancellation recovery or unload |

The arrows name conceptual contracts. They do not require one universal
checkpoint object, request struct or output type across families.

## Stage contracts

### 1. Resolve and validate assets

Declare required and optional assets, format versions, checkpoint identity,
configuration precedence and defaults. Generic loaders own file-format parsing;
family logic owns weight-key meaning, tokenizer requirements, normalization
selection and supported architectural profiles.

Check compatible dimensions, dtypes and configuration before allocating large
device buffers where possible. If a compatibility path accepts precomputed
inputs, document which assets it can omit and which validation still applies.
A malformed required asset must produce a contextual error, not silently select
an unrelated default. Legacy conversion belongs outside the inference hot path.

### 2. Load and initialize

Interpret and transform weights once where possible, upload them to the selected
device, and retain the identities needed to validate later requests. Declare
supported precision, input representations, dimensions and execution limits.
Loading weights does not necessarily mean a graph has been captured or a first
request has been warmed up; expose that distinction in startup and latency
reporting.

If initialization fails, partially created resources must remain owned and be
released correctly. Replacing assets must invalidate derived state rather than
reusing tokenizer tables, transformed weights or execution profiles from a
different checkpoint.

### 3. Canonicalize request

Separate application adaptation from family input semantics:

- Application adapters resolve external field names, image containers, transport
  objects, camera sources, units and external ordering.
- Family input logic constructs the checkpoint-defined text/token sequence,
  state representation, modality alignment, normalization and masks.
- A tokenizer module provides loading and encoding mechanisms. Template text,
  state discretization, special-token ordering and multimodal expansion belong
  to the family that defines them.

A canonical input must specify shape, dtype, layout, semantic value domain,
modality ordering, mask meaning, sequence lengths and any stochastic input or
session dependency. State whether input values are raw, normalized or already
quantized, and whether noise is supplied or generated. Define error behavior for
unsupported shapes, missing fields and overlength sequences; do not silently
truncate or drop a modality unless that is part of the declared family contract.

Canonical does not mean fully tensorized on the host. A runtime can accept
resized RGB and perform checkpoint-fixed normalization/patchification on the
device, or accept already computed patches through a separate declared route.
The accepted representation determines where host preparation ends. Capability
checks select the route before execution; failure must not silently change
input semantics.

### 4. Select / prepare execution

Distinguish request values from execution profile identity. Reuse is valid only
when every allocation-, dispatch- and interpretation-relevant property agrees.
Examples include shapes, sequence capacity, input representation/layout,
precision, device, selected tactics and state capacity. These properties may be
part of the instance identity or a profile key; they need not all appear in one
struct. Image bytes, token values and noise normally update buffers without
invalidating a compatible profile.

Allocate stable buffers and initialize native plans before capture. When the
public runtime supports multiple profiles, it must select or rebuild the right
one. An explicitly prepared handle may instead reject incompatible requests.
Do not infer that graph capture is possible merely because eager execution
works. Document capture failures and any supported eager fallback.

### 5. Execute

The family network owns layer order, attention semantics, conditioning,
scheduling and the model's mathematical state transition. The execution carrier
owns allocation reuse, cache/graph lifetime and dispatch; it calls the network
rather than duplicating its mathematics. Backend operations remain model-neutral
and are accessed through safe interfaces.

For accelerator paths, intermediate layer computation remains on the device as
described in [Model Execution Wiring](model-execution-wiring.md). Host-to-device
input transfer and the requested final output transfer are explicit parts of the
contract. Define when an output is ready to read, whether storage is borrowed or
owned, and whether another invocation may overwrite it.

A failed request must state whether session/profile state is still usable or
requires reset/recreation. Mutable prepared state must not be reused concurrently
without an implementation-supported ownership or synchronization policy.

### 6. Interpret output

Declare the model result's semantic domain and the conversion to the public
result. Depending on the family, this can include sampling and stop decisions,
text decoding, action trimming, inverse normalization or structured result
assembly. Streaming generation repeats execution and interpretation until its
stop condition; continuous-action solvers keep their internal numerical steps
inside the model execution owner.

Checkpoint-defined output transforms belong to family inference semantics.
Application packet encoding, transport, robot units and actuator ordering belong
to application adapters. A low-level caller that supplies canonical model inputs
may choose to consume the raw model-domain result and own these later stages.

### 7. Reset and release

Specify what reset clears: for example sequence KV contents, sampling counters
or episode-local state. Reset may retain compatible weights and execution
buffers. It must not accidentally carry values from one logical session into
the next.

Release must respect outstanding device work and ownership dependencies among
graphs, buffers and their context. Shared resources remain alive while borrowed
or referenced. Document the supported close/drop mechanism, whether repeated
close is allowed, and behavior after close. These are required lifecycle
semantics for each implementation, not a claim that all existing frontends have
one common `reset()` or `close()` method.

## Family contracts and language adapters

Shared lifecycle stages do not erase family differences. ApxInf currently uses
`LlmTrait` for autoregressive LLM/VLM execution and `VlaRuntime` for continuous
action inference. VLM image input affects prefill; VLA masks, latent noise and
solver schedules do not become token-sampling options. Reuse mechanisms only
where maintained implementations establish the same semantics.

Language placement follows ownership. Rust owns model execution and native
mechanisms; some existing family input/output orchestration remains Python.
PyO3 translates types, ownership and errors. It must not become the sole owner
of family prompt or normalization rules. A future native high-level policy can
own more of stages 1-3 and 6 without moving application-specific Python adapters
or user callbacks into the runtime.

Custom input processing must terminate at an explicitly supported canonical
input seam. Preserve its separate responsibility for producing valid inputs;
do not silently replace it with built-in processing. Equivalent built-in and
custom paths should be tested against the same intended model-domain inputs.

## Directory organization

Organize by stable ownership within existing packages. The following is a map
of responsibilities, not a mandatory set of filenames or a request to add new
crates:

```text
crates/
  apxinf-core/                 tensors, devices and model-neutral contracts
  apxinf-loader/               checkpoint container/format mechanisms
  apxinf-tokenizer/            tokenizer loading and encoding mechanisms
  apxinf-model/src/
    auto.rs, registry.rs       registration and runtime selection
    llm_trait.rs, vla/        established execution-family contracts
    <model>/
      config / weights        family asset interpretation and weight preparation
      network / executor      network mathematics and composition
      runtime / cache / graph execution preparation, state and resource lifetime
      backend.rs              family CUDA-facing seam when required
  apxinf-cuda/                 safe device operations and kernel implementation
  apxinf-py/                   thin native language adapter
python/apxinf/apxinf/
  checkpoints/                Python compatibility and checkpoint metadata handling
  policies/impls/             existing family policy composition
  processors/                reusable input/output processing and existing steps
  adapters/, robots/          application/robot adaptation
  serving/                   transport and request handling
```

The role names inside `<model>/` are illustrative; use the existing filenames
that express those responsibilities. A small module may combine stages. Split
when a responsibility changes independently, not to obtain seven files. Keep
family-specific input/output semantics local to their owner when reorganizing
existing processors. Do not introduce shared family switches or a new generic
policy crate solely to make the directory tree symmetric.

Follow the existing [dependency matrix](model-layer-architecture.md#per-model-isolation).
Network code must not depend on a concrete captured-runtime implementation;
shape/budget definitions should remain leaf concepts. A family implementation
is not another family's reuse library.

## Verification and documentation

| Seam | Evidence needed when changed |
| --- | --- |
| Assets -> initialized instance | loading/override tests; invalid, missing or mismatched asset failures; ownership on partial failure |
| Application input -> canonical input | reference fixtures for tokens, shapes, layouts, masks and numerical domains, including boundary values and rounding order |
| Profile -> repeated execution | cache invalidation, incompatible-profile handling, repeated calls, changed-input propagation, eager/captured parity where supported |
| Model result -> public result | exact IDs and declared numerical tolerances, output transforms, streaming stop/state behavior where applicable |
| Session/instance end | reset isolation, close/drop ownership and failure-recovery behavior |
| Complete public path | pinned assets and deterministic inputs/stochastic state; precision-qualified output comparisons and separately measured latency |

Tests must exercise the declared public seams. Successful loading, matching
shapes or plausible output are insufficient evidence of numerical equivalence.
Latency reports separate startup/preparation, host processing, model execution
and complete request latency, with device synchronization and concurrency
conditions stated. Closed-loop task quality is a separate qualification claim.

Keep shared lifecycle rules here, dependency rules in Model Layer Architecture,
and device implementation guidance in Model Execution Wiring. Record each
model's actual capabilities, assets, shapes, semantic exceptions and unsupported
cases in its existing module or user documentation. A new model does not need
a copy of this document. Temporary migration plans and generated evidence belong
in private workflow artifacts; summarize the current change and validation in
its PR.

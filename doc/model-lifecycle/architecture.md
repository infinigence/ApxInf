# Model architecture: current implementation and refactor specification

Status: agreed design direction; target interfaces are proposals except in the
explicitly marked implemented PI0.5 section. Reviewed source baseline: upstream/main
`7baa69b281ef862e6afa32c476c58143d3964241` (GR00T N1.7 merged).
This document supersedes the earlier GR00T PR snapshot in this directory.
Stage 1 has since merged upstream/main `ee42185` (documentation-only changes
since the reviewed implementation); see [baseline protocol](baseline.md).
GPU evidence and its limits are recorded in the baseline and migration documents.

Read [lifecycle contracts](lifecycle.md) and the [staged rollout](migration.md).
The existing [model-layer reference](../model-layer-architecture.md) describes
implementation guidance until individual migrations update it.

## Baseline logical view (selected main)

```mermaid
flowchart TB
    VLA[VLA Python Policies and model Processors] --> PY[Generic native Model binding]
    PY --> VR[VlaRuntime: contract / prepare / infer]
    VR --> PI[PI0.5 runtimes: network, solver, resources, graph]
    PI --> PE[Precision executors: mainly layer computation]
    VR --> WA[WallOSS runtime and executor]
    VR --> GA[GR00T VlaRuntime: typed request adaptation]
    GA --> GE[GR00T generic executor: network, resources, graph]
    GE --> GB[GR00T private backbone]
    TEXT[Tokenizer / prepared multimodal input] --> LOOP[Shared LLM and VLM generation loop]
    LOOP --> LM[Llama / Qwen3-VL: network, KV state, decode graph]
    PE --> BE[Shared backend / kernels / memory / graph]
    WA --> BE
    GE --> BE
    GB --> BE
    LM --> BE
```

GR00T now uses the generic Model/VlaRuntime entry and owns its backbone directory.
It is not an independent public Gr00tModel and does not need the earlier proposed
exception for importing sibling Qwen3-VL internals. Do not reintroduce a shared
backbone extraction merely to satisfy the obsolete proposal.

The remaining problems are semantic: `prepare` has different guarantees,
compatibility constraints are partly private, output residency differs, and
`runtime`/`executor` do not identify a consistent responsibility. LLM/VLM already
share a generation loop, but graph preparation and request state remain in models.

## Target logical view

```mermaid
flowchart TB
    U[Caller] --> F[Policy / TextModel facade]
    F --> P[Processor: encode and incremental or final decode]
    P --> I[Typed encoded input]
    P --> C[Request-local output context]
    F --> S[ExecutionSession: resources, readiness, reset, execution]
    I --> S
    M[Loaded Model: config, weights, capabilities] --> S
    S --> D[Native algorithm driver: generation or model algorithm]
    D --> E[Prepared execution regions]
    E --> N[Model Network: stage interfaces and major dataflow]
    N --> B[Semantic Blocks: backbone, attention, action head]
    B --> K[Backend kernels and device weight views]
    E --> G[Graph replay]
    G --> K
    S --> O[Output with device, completion and lifetime contract]
    O --> P
    C --> P
    P --> F
```

The driver is a responsibility, often an existing function, not a mandatory class.
Algorithm control remains in native code; do not round-trip through Python per
network layer or denoising step. A fixed flow loop may be captured as one region.

| Module | Interface role | Owns |
| --- | --- | --- |
| Policy / TextModel | infer / generate | Encode-execute-decode orchestration |
| Processor | encode -> typed input + context; decode -> user output | Prompt, tokenizer, image/state semantics, incremental text or action decoding |
| Loaded Model | load; create_session | Config, resident weights, capabilities and network construction |
| ExecutionSession | prepare; infer/generate; reset_request; clear_plans | Stable buffers, workspace, KV/latent storage, plans, invalidation and completion |
| Algorithm driver | generate or model-specific inference algorithm | Sampling, EOS, iteration/update rules; request progression |
| Network | prefill/decode or encode_condition/predict_velocity | Model-level tensor interfaces and major subnetwork connections |
| Block | Typed tensor/state transformation | Internal layer composition and precision-specific implementation |
| Backend | Kernels, allocation, capture/replay, events | Device mechanisms, not model semantics |

Processor is not synonymous with CPU execution. GPU preprocessing can be captured
without transferring formula ownership to Session. A learned vision encoder is a
Network/Block, not a tokenizer/image Processor. Sampling and EOS belong to the
algorithm; text detokenization belongs to Processor.

## Network / Block seam

Each model has one maintained Network definition where practical. A Block is an
internally cohesive transformation, not necessarily one transformer layer.
Backbones and action heads are large Blocks and may contain smaller Blocks.
Preserve meaningful names such as `vision` and `action`, rather than renaming
all types to generic Block names.

Network connects major subnetworks and exposes computation stages. Blocks hide
local topology, physical layouts and precision-specific fusion. A change to the
vision-to-language connection belongs in Network; changing QKV packing or fused
norm/quantization belongs in the relevant Block and its weight materialization.
If reusable quantized input spans projections, group those projections rather
than exposing that temporary to Network. Do not force conversions at every Block
boundary to make interfaces look uniform. Typed internal values or a larger Block
may preserve a continuous quantized path.

```rust
// Conceptual pseudocode; no mandatory public generic framework.
struct GrootNetwork<V, L, A> { vision: V, language: L, action: A }
fn encode_condition(input, state, ctx) -> Condition {
    images = vision.forward(input.pixels, input.grid, ctx);
    language.prefill(input.tokens, images, input.mask, state, ctx)
}
fn predict_velocity(condition, latent, time, state, ctx) -> Tensor {
    action.forward(condition, latent, time, state, ctx)
}
// A concrete FFN implementation may fuse norm + quantization + projections.
// It exposes the FFN result, not its internal quantized scratch buffers.
```

A Block can describe resource requirements; Session owns their allocation and
lifetime. ExecContext provides bounded device/resource access, not arbitrary
access to the whole Session. Network does not manage graph caching or serving.
Eager and capture must use the same maintained computation semantics. Proven
precision-specific fusion is permitted; a duplicate capture-only network is not
the default architecture. Public Network factories and per-layer dynamic Block
traits are not required. Select the compute variant at construction and retain static
specialization in hot paths.

## Compute implementation selection (agreed target)

Use `compute_variant` for the single user-facing choice of a model's compute
implementation. It selects a compatible bundle of Blocks, physical weight
representations and preparation requirements; it is not merely a dtype or a
checkpoint/model-size variant. Do not add independently combinable quantization
and implementation fields until a real use case requires them.

The field name and selection contract are shared across models. Supported values
belong to each model: do not create one global enum containing every model's
implementations. Within a model module, use `ComputeVariant`; if a flattened
public export is needed, an alias such as `Pi05ComputeVariant` disambiguates it.
The prefix identifies ownership, not a different lifecycle contract.

```rust
// Implemented PI0.5 selection. Shared LoadOptions carries a model-local ID.
let options = LoadOptions {
    compute_variant: Some(pi05::ComputeVariant::Fp8Static.as_str().into()),
    ..LoadOptions::default()
};
// pi05::ComputeVariant::{Auto, Bf16, Fp8Static, Int8Dynamic}
```

Rust and Python use `compute_variant`; canonical values are `auto`, `bf16`,
`fp8_static`, `int8_dynamic`. PI0.5 rejects explicit legacy `precision` and
ambiguous IDs such as `fp8` or `w8a8`. Other models retain their existing precision
interfaces until migrated and reject compute_variant through the current common
loader. Stage 3 extends this support when WallOSS migrates; it does not introduce
a global enum of every model's variants or a registration framework.

`Auto` is resolved once during loading: static FP8 on SM100+ with calibration
(or explicitly supplied uniform diagnostic scales), dynamic INT8 on SM80–SM99,
and BF16 otherwise. Explicit choices retain existing kernel fallback behavior;
this selection rule is not a declaration that all hardware/profile combinations
are qualified. The resolved ID is logged. The loader creates matching Blocks and
injects them into `Pi05Network::from_blocks`. Selection and typed dispatch are
centralized in `load.rs`; Network and Session do not match the variant enum.

Each value selects a complete compute implementation, including numerical
formats and preparation requirements. `fp8_static` means fixed calibration-based
activation scales. `int8_dynamic` means fixed per-output-channel weight scales
and runtime per-row activation scales; it is not a dynamically changing model
or a static-activation INT8 implementation. W8A8 remains useful kernel storage
terminology but is not the model's variant ID. Same-precision alternatives can
add values when actually implemented.

## Weight and precision ownership

| Current file/content | Target responsibility |
| --- | --- |
| runtime loading | Model construction |
| runtime/executor capture, buffers, cache | Session |
| runtime/executor major computation | Network |
| executor attention/FFN computation | Blocks |
| generation loop / flow update | Native algorithm driver or model algorithm function |
| weights.rs checkpoint mappings and validation | Model weights/loading |
| static_*_weights.rs whole-model resident tree | Model resident weights, parameterized when structure matches |
| device_weights.rs matrix representation and compute | Shared or Block-local weight/compute implementation |
| kernel weight views | Backend's non-owning device interface |

Names currently mean different things: PI0.5 device_weights.rs holds FP8 linear
storage and packing; static_weights.rs holds the PI0.5-wide resident weight tree.
GR00T device_weights.rs is a private precision-neutral computation contract.
Backend FP8/W8A8 weight views are already model-neutral. Do not move an entire
model weight tree into shared code just because its filename says static.

Reuse model structure and checkpoint mapping across precision implementations.
Keep genuinely different scales, layouts, quantization, packing and fused compute.
Precision may differ between vision, text and action; one dtype parameter for all
fields is not a requirement. GR00T already has a precision-parameterized executor:
preserve that progress rather than creating three copies of Network.

## Target development view

Prefer semantic grouping before dtype grouping. This is a placement guide, not a
mandatory file checklist. Small Blocks and weights can remain single files.

```text
crates/apxinf-model/src/
  auto.rs / registry.rs / builtin.rs   existing model construction
  llm_trait.rs or generation.rs        shared native generation algorithm
  vla/                                VLA public contracts
  <model>/
    mod.rs                            model entry and capabilities
    network.rs                        one major dataflow definition
    session.rs                        model-specific bindings and plan requirements
    weights.rs                        checkpoint schema and resident weight tree
    blocks/
      vision/                         backbone and its inner blocks
      language/
      action/
        mod.rs                        semantic interface / shared implementation
        bf16.rs / fp8_static.rs / int8_dynamic.rs     only where implementations actually differ
crates/apxinf-cuda*/                   backend mechanisms and kernel weight views
python/apxinf/.../policies/            VLA facade and model processing
crates/apxinf-tokenizer/               existing tokenizer capability
```

Do not create three parallel complete trees under blocks/bf16, blocks/fp8_static,
blocks/int8_dynamic by default. A model-wide precision directory is not required; local
compute specialization belongs beside its semantic Block, common matrix storage
belongs in a demonstrated shared module, and quantization selection belongs in
construction. Keep checkpoint mapping separate from kernel physical layout.

Independent correctness/performance reference implementations are harnesses,
not alternate production Networks. Maintained reusable harnesses belong in the
established tests/benchmark locations; temporary comparisons, scripts and logs
belong in ignored `devlocal/model-lifecycle-refactor/` within the active worktree.
Do not create a shared backbone without multiple maintained consumers and a
reviewed narrow interface. Cross-model reuse is not implied by similar names.

## Change-locality acceptance

| Change | Expected owner |
| --- | --- |
| Prompt or action interpretation | Processor |
| FP8 FFN fusion | Corresponding Block implementation |
| QKV physical layout | Block weight materialization and compute |
| Compatible backbone replacement | Block implementation and model construction |
| Vision/language connection | Network |
| Capture recovery or cache eviction | Session/backend mechanism |
| EOS or sampling policy | Generation driver / sampler |

More files or renamed executors do not prove improvement. Each migration must
show that these changes have predictable owners and that hidden invariants have
become explicit contracts. See migration.md for evidence and documentation gates.

## Implemented PI0.5 pilot (Stage 2)

The extended Stage 2 candidate removes all three PI0.5 runtime files and their
compatibility types. This view describes the refactor branch, not unmigrated
families. The implemented CPU/CUDA checks and native qualification status are
tracked separately in [baseline.md](baseline.md).

```mermaid
flowchart TB
    U[Python Policy: encode / decode context] --> A[AutoModel / LoadedModel]
    A --> L[load: resolve compute_variant, materialize assets, construct Blocks]
    L --> S[Session: execution policy, implicit cache, prepare/run]
    L --> N[One Network: vision → prefix/KV → flow schedule]
    N --> B[Blocks: bf16 / fp8_static / int8_dynamic]
    W[weights: host mapping, device trees, fixed calibration] --> B
    S --> P[PreparedInference: validity, request inputs and RNG]
    S --> R[prepare: requirements → allocation → warmup → capture]
    B -->|layout and workspace requirements| R
    R -->|record same computation| N
    R --> G[CapturedGraph: executable + stable resources + Network]
    P -->|eager| N
    P -->|replay| G
    B --> K[CUDA kernels]
    R --> C[CUDA backend: scoped capture and cleanup]
```

```text
pi05/
  mod.rs                       public entry and registration
  config.rs                    model shape and ComputeVariant names
  load.rs                      asset loading, selection, typed compute dispatch
  session.rs                   policy, prepared requests, validity and implicit cache
  prepare.rs                   one warmup/capture path and CapturedGraph resource owner
  network.rs                   one model schedule, independent of concrete variants
  calibration.rs               BF16 observer and diagnostic Network traversal
  backend.rs                   model-local CUDA/kernel imports
  math.rs                      prompt/state/time mathematical helpers
  blocks/
    mod.rs                     semantic Blocks interface; Prefix and Styles types
    bf16.rs                    BF16 backbone/layers and resource requirements
    fp8_static.rs              static FP8 backbone/layers and resource requirements
    int8_dynamic.rs            dynamic-activation INT8 backbone/layers and requirements
  weights/
    mod.rs                     fixed-asset exports
    host.rs                    checkpoint mapping and common logical weight tree
    packing.rs                 shared host matrix packing
    bf16.rs                    BF16 linear storage and device model tree
    fp8_static.rs              static FP8 linear storage and device model tree
    int8_dynamic.rs            INT8 linear storage and device model tree
    fp8_static_calibration.rs  E4M3 representation, calibration profile and fixed scales
```

The tree has 20 Rust files (22 before slice D); the model root has nine files
(previously eighteen). Each device-weight file groups its linear storage in
an internal module and its aggregate model tree in the same file. Backbone/layer
code also remains grouped per variant instead of expanding into many one-function
files. Cross-model matrix/view reuse remains a later evidence-driven extraction;
PI0.5's own weight organization is complete in this stage.

Network owns the full model order and flow step count/dt. Blocks own backbone
layer loops, fusion, physical layout and fixed weights. The Network source imports
only the Blocks contract; BF16-only calibration traversal lives in calibration.rs.
Rust statically specializes the Network for each implementation. `load.rs` wraps
these types for the public Session; no per-layer virtual calls are introduced.

Blocks report workspace requirements and perform their native input conversion.
`prepare.rs` allocates resources, prepares fixed styles, warms up until tactics
stabilize, captures with the shared CUDA scope, and returns a single CapturedGraph
for every variant. Its erased fixed-resource owner retains the concrete Network
and style tensors; this erases ownership storage only, not computation dispatch.
The executable graph is dropped before the memory it references.

Session owns the preparation policy, request validation, RNG rebinding, tactic
invalidation and implicit cache. It does not choose FP8/INT8 implementations or
manage separate precision graph types. There is no replacement runtime facade.
Low-level diagnostic callers construct a Network with `build_*_network`, call
its computation methods, and explicitly use `capture_patches` or `capture_rgb`.
Ordinary callers use AutoModel and prepare/run.

### Breaking interface migration

| Previous PI0.5 entry | Current entry |
| --- | --- |
| precision=fp8 / bf16 / int8 or w8a8 | compute_variant=fp8_static / bf16 / int8_dynamic |
| Pi05CudaRuntime::new | build_fp8_static_network |
| Pi05Bf16CudaRuntime::new | build_bf16_network |
| Pi05Int8CudaRuntime::new | build_int8_dynamic_network |
| runtime.capture_infer / capture_infer_rgb_u8 | capture_patches(&network, ...) / capture_rgb(&network, ...) |
| Three precision CapturedGraph types | CapturedGraph |
| StaticFp8Pi05Weights / StaticBf16Pi05Weights / StaticInt8Pi05Weights | Fp8StaticWeights / Bf16Weights / Int8DynamicWeights |
| Pi05ActivationScales / StaticFp8Calibration | Fp8StaticActivationScales / Fp8StaticCalibration |
| Unprefixed FP8 layer functions/types | Explicit fp8_static / Fp8Static names |
| Pi05VlaRuntime alias | Pi05Session |
| pi05_bench --dtype fp8; JSON precision key | --compute-variant fp8_static; JSON compute_variant key |
| Python Model.random(precision=...) | Model.random(compute_variant=...) |

Repository callers are migrated. External low-level Rust callers, Python keyword
callers and benchmark parsers must update. Existing checkpoint/calibration/tactic
asset schemas are preserved; operator names such as W8A8 are not renamed globally.
GR00T/WallOSS numerical implementations are unchanged. The shared LoadOptions
still contains legacy precision for those families, not a second PI0.5 selector.
Dedicated PI0.5 benchmark/server tools use compute_variant. The multi-model LIBERO
campaign tool retains its numerical precision ledger category, translating that
category to PI0.5's implementation ID at loading; historical campaign ledgers
are not rewritten. Its websocket boundary recognizes the new server metadata.

### Internal interfaces and change ownership

```rust
// Abbreviated signatures; the callable implementation lives in blocks/mod.rs.
trait Blocks {
    type Prefix;   // precision-specific KV representation
    type Styles;   // fixed per-step modulation tensors
    fn vision(patches, native_representation) -> Tensor;
    fn embed_prefix(vision, token_ids, token_count) -> Tensor;
    fn prefix(embeddings) -> Self::Prefix;
    fn prepare_styles(time_embeddings) -> Vec<Self::Styles>;
    fn eager_styles(time_embeddings) -> Option<Vec<Self::Styles>>;
    fn step(state, time_embedding, prefix, dt) -> Tensor;
    fn step_with_styles(state, styles, prefix, dt) -> Tensor;
}
fn network_infer(input, noise, time_embeddings) {
    styles = blocks.eager_styles(time_embeddings);
    vision = blocks.vision(input.patches, input.is_native);
    prefix = blocks.prefix(blocks.embed_prefix(vision, input.ids, input.count));
    for index in 0..config.num_flow_steps {
        noise = match styles {
            Some(styles) => blocks.step_with_styles(noise, styles[index], prefix, dt),
            None => blocks.step(noise, time_embeddings[index], prefix, dt),
        };
    }
    return noise;
}
```

`Prefix` and `Styles` keep physical representations behind the Block boundary.
The native-input flag is an internal materialization contract: callers already
validate and construct the expected representation. It does not select dtype.
BF16/dynamic INT8 eager styles remain precomputed before vision; static FP8 eager styles remain
computed per flow step after prefix. Capture prepares fixed styles beforehand.
Preserving this order avoids mixing algorithm/rounding changes into migration.
A fusion or backbone implementation change stays in its Block; changing how
vision conditions language/action or the flow schedule belongs in Network.

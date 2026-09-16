# Model lifecycle contracts

Status: target specification plus explicitly marked implemented slices. Current-source baseline and
module responsibilities are in [architecture.md](architecture.md).
Implementation order and tracking are in [migration.md](migration.md).

## Current lifecycle differences at upstream/main 7baa69b

| Family | Preparation and capture | State / output observations |
| --- | --- | --- |
| PI0.5 | Explicit prepare allocates and attempts capture, with eager fallback; automatic infer may tune on a real request first | Prepared plan owns resources; tuning-generation checks; device Action |
| WallOSS | prepare allocates; first run initializes and captures; explicit no-graph path; capture errors otherwise propagate | Private input/noise-mode constraints; device Action |
| GR00T | prepare returns a wrapper sharing engine; infer checks private graph key and captures on demand, with eager fallback | Engine owns graph; explicit noise required; Action already on CPU |
| Llama | Shared generation resets state, attempts decode prewarm, then prefill/decode | Actual decode implementation uses one capacity-bound graph; lazy capture still possible |
| Qwen3-VL | Shared generation; prewarm hook is default no-op; decode captures on new KV-length bucket | Image processing in prefill; KV and rope delta reset; power-of-two decode buckets |

VLA public InferenceSpec currently contains only token_count and image_layout;
private model constraints are richer. Generic Action device-residency comments
are not matched by GR00T's current host-return behavior. These are migration
inputs, not claims that numerical results are incorrect.

```mermaid
flowchart TB
    L[Load model] --> V{VLA implementation}
    V --> P[PI0.5 prepare: allocate and capture]
    V --> W[WallOSS prepare: allocate]
    W --> WR[First run: initialize and capture]
    V --> G[GR00T prepare: shared engine wrapper]
    G --> GR[Infer: graph-key lookup and capture]
    L --> T[LLM/VLM: reset and prewarm hook]
    T --> LL[Llama: attempt decode pre-capture]
    T --> Q[Qwen3-VL: no prewarm implementation]
    LL --> PF[Prefill then token decode]
    Q --> PF
    PF --> QC[Missing graph: capture during decode]
```

## Three lifetimes

```mermaid
flowchart LR
    subgraph ModelAssets[Model assets]
        A[Load and materialize weights] --> B[Loaded] --> C[Unload]
    end
    subgraph Plans[Execution resources]
        D[Describe region and compatibility] --> E[Prepare]
        E --> F[EagerReady / GraphReady]
        F --> G[Repeated execution]
        G --> F
        F --> H[Invalidate or evict]
        H --> D
    end
    subgraph Request[Request state]
        I[Encode] --> J[Initialize state]
        J --> K[Run algorithm]
        K --> O[Decode or stream output]
        O --> R[Finish / cancel / reset]
    end
    B -. fixed assets .-> E
    I -. execution needs .-> D
    F -. execution capability .-> K
```

KV storage is reusable; its valid content/length belongs to a generation.
Latent storage is reusable; latent values belong to an inference. Request reset
is not plan eviction or model unload. Conversation/episode state, when needed,
has an explicit owner and reset scope; it is not inferred from a socket closing.
Sessions are serial by default, not implicitly thread-safe or concurrent.

## Preparation and CUDA Graph contract

Prepare establishes readiness for an execution region and compatible input range,
not one boolean for an entire model. Load normally performs checkpoint parsing,
weight packing/quantization and upload. Processor encodes user input. Prepare
ensures execution resources and choices are ready; already available work is reused.
Calibration profiles are loaded/validated assets, not collected on every prepare.

```mermaid
flowchart TB
    S[Describe region and compatibility] --> A[Allocate stable resources]
    A --> P{Execution policy}
    P -->|Eager| ER[EagerReady]
    P -->|PreferGraph / RequireGraph| W[Warmup / tune if required]
    W --> F[Freeze addresses and execution choices]
    F --> C[Capture and instantiate]
    C --> OK{Success?}
    OK -->|yes| GR[GraphReady]
    OK -->|no, fallback allowed| ER
    OK -->|no, graph required| X[Preparation failure and cleanup]
    ER --> B[Bind compatible current inputs]
    GR --> B
    B --> R[Execute / replay]
    R --> O[Output and completion]
    O --> B
    GR -. dependency change .-> I[Invalidate]
    ER -. dependency change .-> I
    I --> S
```

Required invariants:

- Ready means the selected region will not secretly allocate new capacity, tune
  or capture inside execute. A convenient infer/generate may call ensure_prepared.
- Unknown decode ranges may cause an explicit preparation transition mid-request;
  predictable ranges may be prewarmed. Report preparation separately from execution.
- A whole VLA loop, a decode step, or a smaller supported region may be captured.
  Stage uniformity does not require identical graph topology or graph count.
- Warmup/tuning uses isolated or safely restored state. It must not consume the
  real request RNG sequence, advance KV state or denoising steps. Extending a plan
  during generation must preserve the active request, not reset it.
- Plan validity covers actual bound shape/layout, capacity, model/weight identity,
  device, precision and tuning dependencies. Ordinary data updates should use
  stable buffers, not unnecessarily enter the cache key.
- PreferGraph fallback is observable with its reason; RequireGraph fails on
  capture failure. Failed capture cleans up handles/resources and declares reuse.
- Invalid requests do not automatically destroy an otherwise healthy plan.

Graph storage is an in-process executable handle, not checkpoint serialization.
Current backend wrappers retain graph/exec handles and destroy them on drop.
The owning plan must keep every referenced buffer, workspace, weight and device
context alive. Graph handles alone do not own all user memory they reference.

```rust
// Pseudocode: ownership can be shared rather than duplicated per plan.
struct PreparedPlan {
    compatibility: PlanKey,
    mode: EagerOrGraph,
    inputs_outputs: StableBuffers,
    workspace: Workspace,
    resources: RetainedDependencies, // model/state storage/context references
}
struct Session {
    model: SharedModel,
    plans: BoundedPlanCache,
    state_storage: StateStorage,
    request_state: RequestState,
}
```

Session decides budgets and lifetime; backend supplies allocator/graph mechanisms.
Reuse existing GraphWorkspace initially rather than introducing a second allocator.
Current GR00T and WallOSS reservations are 4 GiB and 12 GiB respectively, not
intrinsic graph requirements. Report reserved, cumulative allocated and peak live
bytes separately. Replacement must respect total budget and wait for safe resource
release. Writable fixed-buffer plans cannot be concurrently reused without isolation.

## Interfaces and module interaction

| Role | Conceptual interface | Contract |
| --- | --- | --- |
| Model | load(options); create_session(limits, policy) | Fixed assets and declared capabilities; no implicit request state sharing |
| Processor | encode(raw) -> typed input + output context | Model-specific semantics, validated layout/units and per-request context |
| Network | describe(region, input, options); compute(region, state_view, ctx) | Execution needs and tensor computation; no plan eviction or prompt interpretation |
| Session | prepare(input, options) -> ReadyReport | Per-region mode, preparation work and fallback reason |
| Session | infer / generate_stream | Native algorithm execution using compatible resources |
| Driver | generate / infer_action | Sampling, EOS and algorithm iteration/update rules |
| Processor | decode(output, context) | Final actions or incremental text; preserve request association |
| Session | reset_request; clear_plans; close/drop | Separate semantic reset, cache eviction and resource release |

These roles do not mandate one universal trait or optional-field-heavy input bag.
VLA action inference and LLM/VLM generation retain distinct public capabilities.
Initial noise or RNG is a run option, not an environment observation field.

```mermaid
sequenceDiagram
    participant U as Caller
    participant F as Policy / TextModel
    participant P as Processor
    participant S as Session
    participant D as Native algorithm
    participant N as Network / Blocks
    participant B as Backend
    U->>F: infer / generate
    F->>P: encode
    P-->>F: typed input + context
    F->>S: prepare(input, options)
    S->>N: describe regions
    S->>B: allocate; optional warmup/capture
    S-->>F: ReadyReport
    F->>S: infer / generate_stream
    S->>D: run algorithm
    loop Required computation
        D->>S: execute region
        alt Eager
            S->>N: tensor computation
            N->>B: kernels
        else Captured
            S->>B: bind and replay
        end
        S-->>D: device result
        D->>D: sample / update / stop
    end
    D-->>F: output or events
    F->>P: decode(output, context)
    P-->>U: actions or text
```

```rust
fn generate(session, prompt, options, emit) {
    session.reset_request();
    sampler.begin(options);
    logits = session.execute(Prefill, prompt);
    for index in 0..options.max_new_tokens {
        token = sampler.sample(logits);
        emit(token);
        if is_eos(token) || index + 1 == options.max_new_tokens { break; }
        session.ensure_decode_ready(next_position); // explicit if needed
        logits = session.execute(Decode, token);
    }
}
// VLM changes prefill (vision + feature merge), not the per-token generation loop.
fn flow_inference(input, options) {
    condition = network.encode_condition(input);
    latent = initialize_latent(options.noise);
    for time in schedule {
        velocity = network.predict_velocity(condition, latent, time);
        latent = solver.update(latent, velocity, time);
    }
    return latent;
}
// A fixed flow body may be captured as one region; no host step loop is required.
```

ModelOutput specifies device, completion ordering and ownership. Device results
must remain available without an unconditional host transfer. A borrowed output
must state when the next run overwrites it; the facade consumes or copies before
reuse. Cross-stream or host consumption observes completion. DecodeContext stays
outside Network; LLM incremental detokenization may retain request-local state.

## Verification contract

Every migrated model/precision declares accepted inputs, exact-noise/seed behavior,
output tolerances and supported hardware. Verify public raw-input paths, eager vs
graph parity, state reset, graph reuse/invalidation, fallback/cleanup, cancellation
and output lifetime as applicable. Measure cold preparation, steady-state latency,
LLM TTFT/TPOT, and device memory separately. A documentation or CPU check does not
qualify native GPU execution; unsupported or untested matrix cells remain explicit.

## Implemented PI0.5 preparation contract

These Rust interfaces are implemented on the refactor branch. Hardware evidence
and model coverage are recorded in the [migration tracker](migration.md).
PI0.5 supports them for all three ComputeVariant choices. Other VLA families
retain their existing prepare path and return an unsupported error for the new
policy methods; their default status is RuntimeManaged, never a fabricated Ready.
Python processing and action decoding are unchanged; PI0.5 loading uses the new
compute_variant field. See the architecture document for breaking entry changes.

```mermaid
sequenceDiagram
    participant C as Rust caller / Policy integration
    participant S as Pi05Session
    participant T as Backend tactics
    participant P as PreparedInference
    participant N as Network / shared prepare
    C->>S: prepare_for(real sample, policy)
    opt backend in AutoTune mode
        S->>N: eager traversal on real input
        N->>T: resolve/tune choices
        S->>S: synchronize and release temporary output
    end
    S->>S: allocate fixed inputs and noise generator
    alt Eager
        S->>P: retain eager input resources
    else PreferGraph or RequireGraph
        S->>N: prepare resources and capture (autotune suppressed)
        alt capture succeeds
            S->>P: retain graph and referenced resources
        else capture fails
            S->>P: PreferGraph: eager + reason; RequireGraph: error
        end
    end
    S-->>C: owned plan, Ready(Eager/Graph)
    C->>P: run(request with compatible spec)
    P->>P: reject stale tactics or mismatched input
    P->>N: eager or replay, no capture/autotune
    P-->>C: device Action
    C->>C: copy output before reusing graph output buffer
```

`prepare_with_policy(spec, policy)` skips real-input tuning and selects existing
or default tactics; it never tunes on placeholder data. `prepare_for(sample,
policy)` permits tuning on the sample first. Legacy `prepare(spec)` maps to
PreferGraph. Legacy `infer(request)` remains the convenience path that can tune
and prepare on a cache miss; it is not the fixed-plan low-latency contract.

```rust
// Callable interfaces, abbreviated result/error types only.
LoadedModel::prepare_with_policy(&InferenceSpec, ExecutionPolicy)
    -> Result<Box<dyn PreparedInference>>;
LoadedModel::prepare_for(&VlaRequest, ExecutionPolicy)
    -> Result<Box<dyn PreparedInference>>;
PreparedInference::status() -> PreparationStatus;
PreparedInference::run(&VlaRequest) -> Result<Action>;
LoadedModel::clear_prepared() -> Result<()>;

// Example: strict captured inference, explicit re-preparation on invalidation.
let plan = model.prepare_for(&sample, ExecutionPolicy::RequireGraph)?;
match plan.status() {
    PreparationStatus::Ready { mode: ExecutionMode::Graph, .. } => {
        let action = plan.run(&request)?;
        // Consume/copy action before another run writes this graph's output.
    }
    PreparationStatus::Invalidated => { /* prepare a replacement */ }
    _ => { /* unsupported or unexpected state for this policy */ }
}
```

| Interface / event | Actual guarantee |
| --- | --- |
| Eager preparation | No graph attempt; native eager allocations may still occur during run |
| PreferGraph | Capture attempted during preparation; failure reason retained in Ready(Eager) |
| RequireGraph | Capture error propagates; no eager plan returned |
| Ready | Fixed spec and current tuning store identity/generation; not readiness for all shapes or models |
| Tactic store replacement or generation changes | Graph and eager plans report Invalidated; run rejects them |
| Invalid input | Run returns an error without automatic plan eviction or recapture |
| execution_mode on Session | Reports the implicit cache only; use plan.status() for explicitly owned plans |
| clear_prepared | Synchronizes and evicts the session's implicit cache; caller-owned plans and output allocation views remain alive |
| Request reset | PI0.5 binds every input and full RNG key per call; no implicit episode counter to reset |
| Output | Device tensor; captured result aliases reusable output storage until next run; Tensor clone is not a value snapshot |
| Concurrency | Fixed-buffer plans run serially; thread-local tuning suppression does not add thread safety |

The pilot does not promise allocation-free eager execution or cancellation, and
resource measurements apply only to the tested profiles. Processor
encode/decode context stays in the existing Python Policy helpers. Broader
lifecycle guarantees above remain targets until separately implemented and tested.


### Stage 2 resource preparation and invalidation details

```mermaid
sequenceDiagram
    participant S as Session
    participant R as Shared prepare
    participant N as Shared Network / Blocks
    participant C as CUDA backend
    S->>R: prepare captured region (autotuning suppressed)
    R->>N: prepare fixed per-step styles
    R->>N: query Block workspace/layout requirements
    R->>R: allocate GraphWorkspace and stable inputs
    R->>N: warm traversal with workspace; synchronize
    R->>C: capture_graph(closure)
    C->>C: begin capture
    C->>N: traverse same Network with workspace
    alt successful capture
        C->>C: end capture and instantiate executable graph
        C-->>R: graph handle and output
        R-->>S: retain graph + workspace + Network + inputs/styles/output
    else closure error or unwind
        C->>C: end capture, discard graph, clear known capture error
        C-->>S: original error / unwind
    end
```

Graph handles are in-process objects, not serialized cache files. The captured
resource owner retains Network and therefore every referenced fixed weight,
plus workspace, style tensors, input buffers and reusable output. Request data
is rebound before replay. Dropping the Session does not invalidate a separately
owned prepared plan. Retained plans may therefore keep considerable GPU memory
alive; explicit cache eviction releases only the Session's implicit plan.
A captured output Tensor also owns a view of the arena allocation. Keeping that
Tensor can retain the entire arena after the plan is dropped. Consume/copy the
value and release old output handles as well as plans when measuring memory
reclamation. Cloning the Tensor preserves ownership and aliasing; it does not
create a compact independent output allocation.

A plan records both the tuning store `Arc` identity and its generation. Replacing
the store with another generation-zero store invalidates old plans just as a
record update does. Compatible prepared execution suppresses tuning; it never
refreshes that identity or quietly captures a replacement. Callers prepare a new
plan after invalidation.

The CUDA scope discards failed captures without instantiating them. Ending an
invalidated capture releases the stream, but CUDA's last-error slot still needs
handling before a later kernel launch check. Cleanup consumes only the known
capture Unsupported/Invalidated errors (900/901); unrelated device errors remain
visible. No device reset is attempted. See NVIDIA's [stream capture API](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html)
and [last-error semantics](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html).

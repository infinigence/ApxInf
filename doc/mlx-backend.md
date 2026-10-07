# MLX backend and Apple Silicon model migration

Status: design contract for an implementation series; this document does not
register a backend or claim a qualified Metal model. The initial execution
target is Apple Silicon macOS. Other Metal platforms require their own build,
runtime and qualification evidence.

ApxInf owns model mathematics, checkpoint interpretation and request state.
The MLX backend supplies model-neutral array operations and prepared compiled
execution on Metal. Python MLX/MLX-LM implementations are reference and migration
tools; an external model provider is not the maintained inference path.

## Relationship to backend contracts

The MLX backend must follow the shared core contracts for mathematical
operations, precision, layout, attention semantics and validation. Reconcile
those interfaces in core before implementing the backend; do not introduce a
second private validator or silently substitute a different numerical contract.
Device implementations remain separate, and unsupported operations fail
explicitly without an implicit host fallback.

The [CUDA operator layers](../crates/apxinf-cuda-new/README.md) remain unchanged.
Their separation also guides MLX: model-visible semantics in safe Rust;
validation/resource lifetime in the backend execution layer; native adaptation
behind a C ABI; MLX/Metal kernels below it. MLX need not duplicate CUDA's
candidate registry or autotuner when it has only one implementation.

## Module ownership and public integration

```text
Rust application / existing Python policy
  -> AutoModel / existing native ModelRunner
  -> family LlmTrait or VlaRuntime implementation
  -> family model composition / selected Blocks
  -> checked model-neutral MLX backend operations
  -> private C ABI / MLX arrays and compiled functions
  -> Metal
```

`apxinf-mlx` is a sibling of the CUDA and future HIP crates. It depends on core
contracts, not CUDA implementation details or a model-family module. A Cargo
`mlx` feature selects its dependency closure without enabling CUDA. Builds
without that feature do not require MLX or Python. Unsupported OS/device/feature
combinations fail at construction with an explicit error.

| Owner | Responsibility |
| --- | --- |
| `apxinf-core` | Device identity, public tensor/storage contracts, portable semantics and validation, sampling contracts |
| `apxinf-mlx` safe Rust API | MLX array ownership, explicit transfers, supported operators, KV storage, prepared callable and stream lifetime |
| `apxinf-mlx` native bridge | MLX C++ adaptation, status/error conversion, array/closure handles and custom Metal operator implementation |
| Family `config`/`load`/`weights` | Checkpoint key mapping, canonical transformations, backend construction and immutable prepared weights |
| Family model/Blocks | Forward order, rounding recipe, attention/position semantics and model-local fused composition |
| Family runner/generation | Request state, preparation, supported profiles, cache invalidation, EOS or flow schedule |
| Existing Python policy / binding | Observation mapping and public input/output conversion, without a second Python network |

`LlmTrait::forward` retains its full `[seq_len, vocab_size]` logits contract.
Last-token-only output-head optimization belongs to an explicit internal
generation path or the existing request-level generation override, not a
shape change to `forward`. Its infallible `reset`/`prewarm_decode` hooks cannot
swallow MLX failures: use a fallible family preparation method, or retain an
invalidated error returned by the next fallible call.

Keep `LlmTrait` and `VlaRuntime` distinct. A small text model may retain the
existing `config.rs`, `weights.rs`, `general.rs` organization. A VLA uses its
own Model and ModelRunner responsibilities. Do not import another family's
model, config or cache merely because equations are similar.

Backend selection is resolved during construction/preparation. Models call
safe mathematical APIs, not raw MLX FFI. Provider/kernel choices stay below
those APIs; family-specific compilation composes those operations rather than
placing a model-name switch in the backend. Reuse the existing registry,
`LoadedModel::text`/`LoadedModel::Vla` and public CLI/binding.

## Native dependency and ABI

The initial bridge can wrap a pinned MLX C++ SDK with a small C ABI. A compatible
`mlx-c` release is an alternative adapter, not a different model architecture.
The implementation must pin the MLX revision/version and linked library/header
pair; do not combine arbitrary system headers with another wheel's library.
A wheel may supply the C++ SDK at build time without embedding Python in the
inference process. SDK discovery must be explicit and reproducible.

Opaque handles own MLX arrays, streams and compiled closures. The bridge uses
fixed-width ABI fields, checked shape conversion and status/error returns;
C++ exceptions and Rust panics cannot cross the ABI. RAII covers partial
construction, callback ownership, asynchronous completion and teardown.
Thread ownership is explicit; do not add `Send`/`Sync` to an opaque handle
without proving the linked MLX runtime and wrapper permit it. Adding a
non-`Send` opaque variant changes the automatic thread traits of every
`Tensor`, including CPU/CUDA tensors; that public compatibility effect requires
an explicit strategy and compilation coverage, not only an MLX-handle test.

Because `Backend` also implements `SamplingBackend`, the MLX backend must
provide the supported categorical sampler on device. A greedy initial profile
may read back only the selected token scalar; unsupported penalties, random
sampling or normal generation must fail rather than run a hidden CPU path.

Ordinary same-process Rust calls may construct a lazy MLX expression. They must
not convert intermediate arrays to host values or evaluate each primitive.
Repeated execution uses prepared block/step closures, avoiding per-token
Rust-to-Python or subprocess calls. Custom Metal kernels are registered once
during preparation and invoked through the same safe operator boundary.

## Tensor storage and portable semantics

Add a distinct Metal device identity. Physical device identity and backend
implementation identity are separate: an MLX handle cannot be consumed by a
different Metal implementation solely because both report `metal:0`.

The current `GpuStorageHandle::from_raw_parts` requires a live device allocation
at `ptr..ptr+len`. A lazy MLX array is not that pointer. Introduce a tagged,
backend-owned opaque storage form with immutable logical extent and retained
owner; preserve existing CPU/CUDA storage and its safety contract. Core may
validate logical capacity/device/dtype without accessing array contents or
forcing evaluation. The backend validates handle identity and physical
representation. Foreign or forged metadata must not reach a native kernel.

Public tensors remain dense, contiguous, row-major. MLX strided views are
backend-private; public permutation, slice and broadcast results honor the
portable contract and explicitly materialize a contiguous value when required.
Reshape may share storage only under the established aliasing rules. The
backend must reconcile a public reshape with the opaque array's own shape;
changing only Tensor metadata cannot leave native operands with stale geometry. New
functional operators cannot mutate inputs, including aliases. Cache updates
and prepared input rebinding are separate explicit mutable operations.

Retain the portable numerical contract:

- F32/F16/BF16 support is declared per operation. No implicit dtype promotion;
  unsupported dtype/device combinations fail explicitly.
- Pointwise operations perform the declared F32 calculation and output cast;
  normalization includes the declared F32 statistics/affine operation. A stock
  MLX fused norm with different intermediate casts is not an equivalent default.
- Matmul preserves canonical `[M,K] @ [K,N]` semantics and the declared
  accumulation/output precision. Packed quantized weights are prepared backend
  artifacts, not replacements for canonical weight meanings.
- Attention uses the portable Q/K/V geometry, GQA head mapping, scale, causal
  offsets, mask and intermediate-precision choices. Fully masked rows return
  zero. A fused SDPA is eligible only where it satisfies that full contract;
  otherwise use an explicit device composition or report unsupported.
- CPU additive-mask content checks happen when the mask is built or updated,
  before upload. Device masks require a guaranteed construction or explicit
  device validation. Structural validation never downloads a tensor.

Use `PortableOps` and `AttentionPlan` once their reconciled core change lands.
Keep immutable structural validation at preparation and device/shape/dtype/
extent checks at invocation. `AttentionPlan` is a device-independent semantic
plan; the MLX executable is device-bound. Do not turn the semantic plan into an
MLX resource owner or add a redundant device field.

Different family rounding recipes must be expressible by explicit casts and
composition. For example, cast-before-affine and cast-after-affine norms cannot
share a fused candidate just because both are called RMSNorm.

## Prepared compiled execution

MLX compilation and CUDA Graph replay are different execution mechanisms.
The existing `Graph`, `RequireGraph` and `ExecutionMode::Graph` retain their
capture/replay meaning; MLX must return unsupported from a requested CUDA-style
capture instead of fabricating a no-op graph.

Add an explicit compiled-function preparation mode in the implementation
series, with prefer/require behavior and an observable coverage descriptor.
This extends the existing prepared owner; it does not introduce a universal
runtime superclass. Readiness identifies the implementation version, supported
profile, selected scope and fallback reason. A required compiled path cannot
silently become eager.

For an MLX target, report each applicable scope independently: local subgraph,
complete decoder block, complete decode/solver step and fixed rollout. A local
norm/RoPE kernel is not proof that full-step compilation was assessed.

| Phase | Required behavior |
| --- | --- |
| Construct | Bind device/runtime identity, weights, public semantics and supported profiles |
| Prepare | Validate structural contracts; materialize immutable weights; create kernels/closures; trace/compile/warm supported profiles; establish memory bounds |
| Run | Check live operands; bind explicit state; call the prepared closure; enqueue device work without retracing, retuning or creating persistent resources |
| Complete | Evaluate required outputs and new state on the correct stream; publish state only after successful completion |
| Reset / invalidate | Make old request state unreachable; preserve retained resources or invalidate dependents before replacing them |
| Release | Retain resources until dependent asynchronous work is complete; release partial preparations on failure |

Compilation keys include callable/build/runtime identity, captured immutable
weights and constants, semantic precision, mask/position rules and all shape
facts that affect tracing. Executable identity also binds the device/stream.
Changing mask kind or fixed causal offsets rebuilds the semantic plan. Dynamic
position/length values are array inputs only where the traced computation
actually consumes them dynamically; otherwise they are specialization facts.

Use exact keys. A documented shape bucket represents an explicitly padded
problem with validity masks, not an approximate recipe hit. `shapeless` is not
proof that a closure is independent of its first observed shape or control flow.
Persistent selection recipes, if later introduced, use implementation/version/
configuration identities and full exact keys; they never store live pointers or
streams. Keep runtime-bound executable caches separate from recipe caches.

CUDA enqueue's allocation-free capture contract is unchanged. MLX compiled
execution may produce temporary array values through its allocator; it must
not be labelled fixed-address or allocation-free replay. Its qualification must
establish bounded transient residency, stable warm memory behavior and no new
long-lived resources, retracing or JIT compilation in the measured hot path.
A latency result does not waive retained-memory growth or stale-state errors.

## State and output lifetime

Weights and constants belong to the loaded model. A prepared object retains
every model/closure/stream resource it uses, even if the original runner is
released. Request state belongs to one request/session; cache eviction,
request reset and model unload are distinct operations.

LLM state includes KV, position, sampling history and any recurrent state.
Append does not independently advance global sequence length in each layer;
the model commits position once after all required layer outputs are complete.
Compiled functions take changing arrays as explicit inputs and return the new
arrays. Failure before completion must discard tentative state or invalidate
the session, never silently reuse a partially updated cache.

VLA state includes observation prefix, latent/noise, conditioning and solver
position. Reuse an observation encoding within its valid request lifetime;
equal shape alone cannot justify reusing it for a new camera frame. Prefix
branching and speculative rollback are explicit capabilities, not consequences
of having a KV container.

Public results state whether they own their data or borrow storage overwritten
by a later call. `Tensor::clone` is not assumed to deep-copy. Tests cover two
live engines, repeated reset/load/unload, retained outputs, cancellation/error
cleanup and capacity exhaustion. Unified memory does not remove ownership,
copy, device-residency or synchronization obligations.

## Migration sequence and acceptance

Pin the source implementation, checkpoint license/revision, dependencies and
supported execution variants in each migration's private workspace. Speculative
modes require their own schedule/state contract and validation, independently
of ordinary decode.

1. Reconcile core/device/storage contracts and build the backend in isolation.
   Test independent feature combinations and existing CPU/CUDA unsupported
   behavior. Do not import either CUDA backend into MLX.
2. Implement and independently check the required model-neutral primitives,
   storage lifecycle and explicit-state eager/compiled execution. Artificial
   shapes establish interface behavior, not complete-model qualification.
3. Migrate the actual optimized forward, weight preparation and schedule into
   each family's own Rust implementation. Preserve supported optimized
   choices: ordered casts, cached constants, compile scopes, native fusions,
   accepted weight layouts and phase-scoped precision. Replacing the portfolio
   with stock MLX-LM inference is not completion of the migration.
4. Register and exercise the public loading and inference path. Keep device
   tensors resident through model computation and sampling; only explicit
   public inputs/outputs cross the host boundary. Rejected implementations
   remain private historical evidence rather than new public runtime switches.
5. Qualify the exact source and public entry on the named hardware. Collect
   numerical, state, full-output semantic and paired performance evidence.
   CPU builds, operator tests, or another project's receipt do not qualify the
   migrated implementation.

Freeze representative real inputs, original complete outputs and the exact
reference protocol before candidate collection. Bind every result to source,
weights, input/settings and dependency hashes. A migration may reuse a source
corpus only after independently verifying its binding to that frozen source;
never overwrite the originals with candidate answers.

For text, use reference-trajectory teacher forcing and the target's fixed
agreement thresholds, plus complete-answer semantic review and corrupt-output
negative control. Free-running token identity is diagnostic rather than a
universal numerical gate. VLA requires its separately declared action/step
error bounds and rollout checks; text top-1 thresholds do not apply to
continuous actions. Quantized paths additionally require precision-matched
implementation checks and the declared model-quality floor.

For each migrated target verify eager/compiled parity, changed-input
propagation, fresh/reused/reset state, supported shape boundaries, output and
prepared-resource lifetimes, and unsupported-case behavior. For fixed MLX VLA
profiles, compiled coverage must include the complete action loop and device
handoff from vision/language, with a whole-model or explicitly justified
prepared stage partition. Eager fallback alone is not completed acceleration.

Measure cold load/compile separately from warm public inference. Drain and
evaluate the actual output/state on the correct stream at the declared
boundary; do not time only lazy graph construction or insert per-layer host
barriers into a different diagnostic loop. Report TTFT, decode or
observation-to-action latency, tail latency and allocator/process memory.
Use prospectively fixed paired/noise criteria for performance claims and
independent confirmation; stable positive gains need no arbitrary 5% floor.

Functional acceptance and optimization status remain separate. Preserved
source semantics do not imply preserved speed on another backend or Mac.
Incomplete compile coverage, omitted accepted optimizations and unexplained
regressions remain explicitly open migration work, not a completed port.

## Required review evidence

| Boundary | Evidence |
| --- | --- |
| Core compatibility | Existing CPU tests and feature builds; CUDA/HIP regressions when their shared contract changes |
| Storage/ABI | Invalid extent/device/dtype rejection, alias/lifetime checks, failure cleanup and thread policy |
| Operators | Independent references for every advertised shape/precision/mask domain, including all-masked attention |
| Prepared execution | No hot retrace/tune; input rebinding; eager/compiled parity; shape invalidation and bounded memory |
| Model | Exact checkpoint load, important intermediate tensors, full real-input gate and state regressions |
| Public path | Registry/CLI or existing binding invocation, real decoded outputs and supported options |
| Performance | Raw paired samples and exact source/environment binding; no cross-run speedup arithmetic |

Temporary references, captures and reports stay in ignored `devlocal/` per
[AGENTS.md](../AGENTS.md). Product code, focused maintained tests, capability
docs and dependency/license provenance belong in their corresponding PRs.

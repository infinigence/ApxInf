# MLX backend and Apple Silicon model migration

The optional `mlx` backend supports native Apple Silicon macOS inference.
See the [selection table](#implemented-text-selections) for supported model
geometry, precision and compilation scopes. Other Metal platforms require
a separate backend implementation.

ApxInf owns model mathematics, checkpoint interpretation and request state.
The MLX backend supplies model-neutral array operations and prepared compiled
execution on Metal. Python MLX/MLX-LM implementations are reference and migration
tools; an external model provider is not the maintained inference path.

## Relationship to backend contracts

MLX uses the core `PortableOps`, `AttentionPlan` and their validators for
checked mathematical semantics, precision and layout. Backend-owned arrays,
prepared weights and compiled resources remain below those shared contracts.
Do not duplicate validators or substitute a different numerical contract.

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
The generation-oriented `prefill(LlmInput)` hook may consume the whole prompt
and return only `[1, vocab_size]`. Qwen3 uses cache-only prefix chunks of at
most 512 followed by a final-token forward; MiniCPM5 projects only the final
normalized prompt row. Call `forward` for every input row's logits. Ordinary
generation uses the shared sampler/EOS loop; DSpark overrides the request-level
generation schedule. The infallible `reset`/`prewarm_decode` hooks cannot
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

### Implemented text selections

The `mlx` Cargo feature registers `qwen3-mlx` and `minicpm5-mlx` with
`AutoModel`. `Device::Metal(N)` selects the `-mlx` implementation; the CLI
accepts `metal[:N]` and `mlx[:N]`, and displays `metal:N`. A missing feature,
unsupported device or absent family implementation fails explicitly. Existing
CPU/CUDA models do not silently execute a Metal request on CPU.

Metal admission requires a registered `-mlx` factory and is resolved before
device construction or checkpoint loading. It does not fall back to a legacy
unsuffixed model factory; CPU/CUDA registry fallback remains unchanged.

| Family selection | Profile admitted by current code | Actual prepared execution |
| --- | --- | --- |
| Qwen3 `bf16-public` | Qwen3-0.6B BF16, B1, up to 2048 positions | Ordered device composition; source public SwiGLU remains locally compiled |
| Qwen3 `bf16-compiled` (default) | Same profile | Local norm/RoPE compilation plus explicit-state B1/L1 decoder blocks after prefix state exists; no whole-step callable |
| Qwen3 `mixed-w8` | Same profile; affine W8/group 64 | Transformer projections use W8 at M1 and BF16 for larger M; fused gate/up; the tied input/output table is packed at every M; local norm/RoPE and fixed Q/K norm-to-RoPE Metal fusion; no BF16 decoder-block callable |
| MiniCPM5 `bf16-public` | Official 2B BF16 geometry, B1, up to 4096 positions | Ordered device composition; source public SwiGLU remains locally compiled |
| MiniCPM5 `bf16-compiled` (default) | Same profile | Local norm/RoPE and complete decode-step compilation, including the packed residual/RMSNorm kernel |
| MiniCPM5 `dspark` | Same target plus the pinned official BF16 drafter; 1–256 output tokens within the context profile | Compiled five-layer draft and sequential Markov-head proposal chain, plus compiled target verification for up to eight rows; explicit accept/trim/pending-token schedule |

These rows describe implemented scopes, not qualified targets. The maintained
owners are [Qwen3](../crates/apxinf-model/src/qwen3/README.md) and
[MiniCPM5](../crates/apxinf-model/src/minicpm5/README.md), with model-neutral
arrays and guarded fusions in [apxinf-mlx](../crates/apxinf-mlx/README.md).
Qwen3 reads `head_dim` explicitly (128, although hidden/heads is 64). MiniCPM5's
official `model_type` is `llama`, so select `--model-name minicpm5` explicitly.
DSpark requires the named `draft` asset and is never selected by default.

```sh
cargo run --features mlx -- generate --model /path/to/Qwen3-0.6B \
  --device metal --dtype bf16 --model-variant bf16-compiled --greedy \
  --chat-options '{"enable_thinking":false}' --max-tokens 80 \
  --prompt 'Explain gravity briefly.'

cargo run --features mlx -- generate --model /path/to/MiniCPM5-2B \
  --model-name minicpm5 --device metal --dtype bf16 --model-variant dspark \
  --asset draft=/path/to/official-DSpark --greedy \
  --chat-options '{"enable_thinking":false}' --max-tokens 80 \
  --prompt 'Explain gravity briefly.'
```

The CLI renders the checkpoint chat template with the supplied options.
Use `--keep-special-tokens` when special-token text carries application meaning;
chat options containing `tools` enable this behavior automatically. MiniCPM's
`<function>`/`<param>` tags must survive decoding for valid tool XML.

## Native dependency and ABI

The implemented bridge wraps the **MLX 0.31.2** C++ SDK with a small C ABI.
`MLX_ROOT` identifies the matching `include/mlx/` and `lib/` distribution;
headers are checked at build time and the linked version at stream construction.
Do not combine arbitrary system headers with another wheel's library. A wheel
may supply native SDK assets without embedding Python in the inference process.
The native feature requires Apple Silicon macOS; see the backend README for
SDK and library-search configuration. No `mlx-c` dependency is currently used.

Opaque handles own MLX arrays, streams and compiled closures. The bridge uses
fixed-width ABI fields, checked shape conversion and status/error returns;
C++ exceptions and Rust panics cannot cross the ABI. RAII covers partial
construction, callback ownership, asynchronous completion and teardown.
Thread ownership is explicit; do not add `Send`/`Sync` to an opaque handle
without proving the linked MLX runtime and wrapper permit it. Adding a
non-`Send` opaque variant changes the automatic thread traits of every
`Tensor`, including CPU/CUDA tensors; that public compatibility effect requires
an explicit strategy and compilation coverage, not only an MLX-handle test.
The implemented strategy is core's opt-in `opaque-storage` feature, enabled by
the native backend: its owner is `Rc<dyn Any>`, making `Tensor` thread-affine
in that feature combination. Default CPU/CUDA builds retain `Send + Sync`.
The existing Python `ModelRunner` stays `#[pyclass(unsendable)]`; no wrapper
introduces an unsafe thread-transfer guarantee.

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

`Device::Metal(usize)` is the distinct Metal device identity. Physical device identity and backend
implementation identity are separate: an MLX handle cannot be consumed by a
different Metal implementation solely because both report `metal:0`.

The current `GpuStorageHandle::from_raw_parts` requires a live device allocation
at `ptr..ptr+len`. A lazy MLX array is not that pointer. `Storage::Opaque` and
`Tensor::from_opaque_parts` retain a tagged owner with checked logical extent;
existing CPU/CUDA storage and its safety contract remain intact. Core may
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

Use the reconciled core `PortableOps` and `AttentionPlan` contracts.
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

The current text implementations select compiled execution through explicit
family variants and fallible family `prepare` methods. `Compiled` owns the pure
Rust callback; `call` is lazy and `call_and_eval` completes all returned arrays.
Preparation uses `Compiled::prepare`, which evaluates twice and rejects any
Rust trace callback on the second invocation. Callable construction rejects
the presence of `MLX_DISABLE_COMPILE`, even when its value is `0`; that MLX
setting otherwise disables compilation silently. The reuse check observes
Rust tracing, not all native shader compilation.

`LlmTrait::preparation_status()` returns a pure-data `TextPreparationStatus`:
`RuntimeManaged`, `Unprepared`, `Ready` or `Invalidated`, implementation/variant,
successfully prepared compilation scopes and sequence lengths, optional prompt
profile, KV capacity, maximum decode rows and a retained error. Qwen3 reports
its actual sorted set of prepared lengths rather than inventing one prompt
profile. Invalidated/unprepared states do not advertise successful scopes.
The default legacy status is `RuntimeManaged`, which proves no preparation.
`TextCompilationScope` distinguishes local subgraphs, decoder block, decode step
and draft proposal; readiness never substitutes for hardware qualification.
The VLA preparation contract remains independent. No universal runtime class
or generic `ExecutionMode::Compiled` is introduced. A future generic
prefer/require interface must extend the existing owner, with explicit profile
and fallback reporting; a required compiled path cannot silently become eager.

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

`Stream::counters` records explicit bridge upload/download bytes and Rust trace
callbacks. It does not count every internal MLX pipeline compilation or physical
DMA. `memory_stats` is the process-wide MLX allocator, not process RSS.
Synchronize before reading allocator statistics, and measure preparation
separately from warm requests. Keep numerical, semantic and performance
validation independent of these diagnostic counters.

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

Follow the [porting workflow](porting-workflow.md) for source/checkpoint
identity, independent numerical and complete-answer checks, public integration
and performance validation. Keep references and captures in `devlocal/`.

MLX ports additionally check explicit-state eager/compiled parity, changed-input
propagation, reset and shape invalidation, retained outputs, failure cleanup and
bounded warm memory. Quantized selections need a precision-matched reference
and the declared model-quality floor. Speculative modes require their own
proposal, verification, stopping and rollback checks.

For fixed VLA profiles, compiled coverage includes the complete action loop and
device handoff from vision/language, through a whole-model callable or a
justified prepared stage partition. Measure construction/compilation separately
from warm inference, evaluate actual outputs and state at the timing boundary,
and avoid per-layer host synchronization introduced only for measurement.

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

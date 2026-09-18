# Model Execution Wiring

Use this guide after model semantics are known and before composing the model
and its runner. Use the [module ownership table](model-layer-architecture.md#current-module-names-and-responsibilities)
for names and placement; this guide defines device execution and acceptance.
It bridges the model equation and ApxInf's safe CUDA interfaces. The purpose is
to design the maintained hot path, not merely to find an implementation that
produces the right answer.

## Accelerator port acceptance

A completed new accelerator port uses ApxInf model code and safe model-neutral
operators as its inference path. Vendor math/kernel libraries belong behind
those APIs. External inference engines (TensorRT, ONNX Runtime, Torch execution,
OpenVINO or equivalents) may run as private references; model graphs, layers or
generated external plans are not a maintained ApxInf implementation.

After public inputs are uploaded, tensor computation stays on the target GPU
until the explicit public output transfer. The host may load/transform weights,
prepare static metadata, submit fixed control flow and receive results. CPU
intermediate computation and D2H/H2D round trips remain private correctness
scaffolds and must be replaced before port completion.

Fixed profiles must support CUDA Graph acceleration. Prefer one graph from
canonical device inputs to outputs. For a VLA where concrete blockers prevent
one graph, the accepted fallback partition covers Vision, Language and Action
with captured graphs, stable device buffers and device-only handoff. Capture
the entire fixed-step action/denoising loop in the Action graph. Intermediate
readback, dynamic allocation and synchronization between segments are not a
completed port. An eager path remains useful for parity and explicit fallback;
it does not satisfy this capture gate by itself.

These are new-port acceptance requirements, not a statement that every existing
family has migrated. Record a missing device/capture path as a blocker. Once
these correctness, residency and capture gates pass, latency optimization is
best effort unless the task specifies a performance release gate. See
[current family coverage](model-layer-architecture.md#current-coverage-and-port-decisions).

## Start from an execution ledger

Write one row for every repeated block and every boundary operation:

| Reference computation | Tensor contract | Frequency | Preferred ApxInf call | Buffers and lifetime | Host traffic | Evidence |
|---|---|---:|---|---|---|---|
| semantic expression, including adjacent operations | shape, dtype, layout, broadcasting, rounding | per request, layer, or solver step | fused call, then primitive alternative | output, scratch, cache, stable address | expected transfer or `none` | replay fixture and tolerance |

The ledger is a design artifact and may stay in the
[private port workspace](porting-workflow.md#private-port-workspace). It must cover
preprocessing-to-output, not only transformer blocks. Update it when the
implementation differs from the plan.

Search for an implementation in this order:

1. the closest maintained model and layer implementation at the requested
   precision and hardware: for PI0.5, inspect `pi05/model/mod.rs` for forward
   order, `pi05/model/blocks/` for fusion, `pi05/model/variant.rs` for precision
   dispatch, and `pi05/model_runner/` for preparation/resource ownership under
   `crates/apxinf-model/src/`. WallOSS/GR00T retain their actual runtime/executor
   filenames; locate them before copying patterns;
2. safe model-neutral interfaces under `crates/apxinf-cuda/src/kernels/`, with
   particular attention to `fused.rs`, `attention.rs`, `rope.rs`, `norm.rs`,
   `activation.rs`, `gemm/`, `cache.rs`, and `elementwise.rs`;
3. the portable `Backend` trait when the operation belongs in a portable path;
4. a new safe, model-neutral operator when required semantics really are absent.

The portable trait is the capability floor, not a catalog of every optimized
CUDA path. A missing `dyn Backend` method does not establish a kernel gap.
Model code may recover the concrete CUDA seam described in
[Adding a New Model](adding-a-new-model.md), but must not call raw FFI.

## Select compositions before primitives

Match sequences of semantics, not isolated framework nodes. Common candidates
include projection plus bias/activation, residual plus normalization, QKV split
plus positional encoding and cache write, gated MLP, adaptive normalization and
residual gating, and solver updates. Prefer an existing fused safe call when its
shape, layout, dtype, broadcast, mask, and intermediate-rounding contracts all
match.

Do not force a nearby fusion onto different semantics. Record why each repeated
sequence uses a fused call, an unfused device composition, or requires a new
operator. Validate a fused choice against the unfused/reference computation at
the fusion boundary as well as at final output.

Equivalent graph rewrites are encouraged. For example, separate Q, K, and V
linear projections may be replaced by one packed QKV GEMM followed by an
existing split, split-plus-RoPE, or split-plus-RoPE-plus-cache-write interface
when the concatenated weight/bias order, head grouping, layout, dtype, and
rounding are proven equivalent. In the current CUDA facade this commonly means
one packed `gemm` call followed by `attention::split_qkv_bias_*` or
`rope::split_qkv_apply_*`; it does not imply that GEMM and split are necessarily
one CUDA launch. Record both the semantic fusion and the actual launch boundary.

## Keep the hot path on the device

After the runtime's declared input representation has been uploaded,
intermediate tensors remain on the target device through the final model
output. The steady-state ledger must have:

- no intermediate device-to-host-to-device round trips;
- no host implementation of activation, indexing, interpolation, scatter,
  masking, positional encoding, or other layer mathematics;
- no synchronization introduced only to inspect or transform an intermediate;
- no per-layer or per-solver-step allocation that could have been prepared.

Host work is appropriate for checkpoint loading, one-time weight conversion,
application/robot preprocessing outside the declared runtime contract,
explicit calibration, and final output transfer. If a runtime declares resized
RGB as an accepted representation, checkpoint-fixed pixel normalization,
patchification, merge ordering, and dtype conversion are part of the maintained
device path rather than host preprocessing. Prepare and capture them with the
fixed-shape model computation when their operators support CUDA Graph capture.
A CPU implementation inside a layer may be used briefly to establish numerical
evidence, but it is a **correctness scaffold**. Mark the affected ledger row,
profile its cost, and use it to validate the replacement boundary. Once the
semantics are established, resolve the row through an existing safe device
composition or the complete [Adding New Kernels](adding-new-kernels.md) path.
For an accelerator target, a steady-state host scaffold remains unfinished
implementation. If a correct device path is unavailable, record the concrete
operator blocker. Long CUDA build times, adapter rebuild scope, or the availability
of a numerically correct host path do not satisfy that blocker.
Ordinary optimization debt may cover an unfused
device composition or untuned tactic after the required device/capture contract
passes.

## Plan tensor lifetime and reuse

Classify each value by when it changes:

- checkpoint/loaded-model lifetime: transformed weights, constant position
  tables and fixed timestep embeddings when the schedule is load-time constant;
- prepared-profile lifetime: masks, index maps, shape metadata, graph workspace
  and precomputed `StepModulation` where the model permits it;
- request lifetime: encoded images and language prefix, reusable cross-attention
  keys and values;
- solver-step lifetime: noisy action state, step output and any conditioning
  that actually changes with the step's inputs;
- layer lifetime: transient projections and normalization scratch.

Compute or upload a value at the widest correct lifetime. In particular, audit
iterative VLA paths for prefix/cross-attention KV, masks, position data, and
timestep data that can be stored in fixed device buffers. A generic cache API
is not proof that the model uses the right cache lifetime.

## Prepare and capture fixed-shape execution

For a fixed target profile, query model/Blocks workspace requirements and let
the runner's preparation code allocate stable inputs, outputs and scratch.
Use `GraphWorkspace` and `prepare_with_workspace` for warmup, then `with_workspace`
during capture through the shared CUDA capture scope. PI0.5's
[`model_runner/prepare.rs`](../crates/apxinf-model/src/pi05/model_runner/prepare.rs)
repeats warmup until tuning generation stabilizes; one invocation is not a
universal readiness guarantee. Capture and eager call the same model semantics.
Update captured input contents in place and replay through the prepared object.

Preparation must exercise the real model computation. A method that only validates a
configuration object is not execution preparation. If graph capture is
unsupported for a required operation, record the exact operation and failure;
an eager fallback is observable but does not satisfy a required capture gate. Compare eager
and replayed outputs before relying on replay latency.

## Wiring review

Before reporting the port, provide evidence for all of the following:

- every ledger row resolves to a safe device call, a named correctness
  scaffold that is still being replaced, or an explicit blocker;
- every repeated adjacency has a documented fused-versus-unfused decision;
- the steady-state host-transfer and synchronization list is empty except for
  public input/output boundaries;
- stable buffers and reusable KV/state are owned by the runtime at the correct
  lifetime;
- the fixed-shape path completes prepare, capture, input update and replay;
  unresolved required capture paths are reported as blockers;
- operator/layer replay, eager end-to-end, captured end-to-end, and public API
  checks pass their declared tolerances;
- wall-clock and graph-replay latency are reported separately, with any gap to
  the stated target attributed to measured operations where possible.

Report two independent outcomes:

- **functional acceptance** requires the reference tolerance, maintained public
  path, native device/capture gates and clear unsupported-case behavior;
- **optimization status** is `target met`, `best effort with performance debt`,
  or `blocked`, with remaining host escapes, unfused hot sequences, missing
  reusable state and measured impact listed explicitly. Device-residency or
  required capture gaps remain blocked rather than best-effort completion.

Optimization is best effort unless the task explicitly defines it as a release
gate. The agent must investigate and attempt the applicable existing paths, but
an honest, measured performance gap does not erase a functionally correct port.
Best effort does not waive the device-residency rule for repeated accelerator
model computation.

Only after this review should a genuinely missing row be handed to
[Adding New Kernels](adding-new-kernels.md). That guide explains how to add a
kernel vertically; this guide decides which safe interface the model should
call and how those calls form the execution path.

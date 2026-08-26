# Model Execution Wiring

Use this guide after model semantics are known and before writing the executor.
It bridges the model equation and ApxInf's safe CUDA interfaces. The purpose is
to design the maintained hot path, not merely to find an implementation that
produces the right answer.

## Start from an execution ledger

Write one row for every repeated block and every boundary operation:

| Reference computation | Tensor contract | Frequency | Preferred ApxInf call | Buffers and lifetime | Host traffic | Evidence |
|---|---|---:|---|---|---|---|
| semantic expression, including adjacent operations | shape, dtype, layout, broadcasting, rounding | per request, layer, or solver step | fused call, then primitive alternative | output, scratch, cache, stable address | expected transfer or `none` | replay fixture and tolerance |

The ledger is a design artifact and may stay in the private port workspace. It
must cover preprocessing-to-output, not only transformer blocks. Update it when
the implementation differs from the plan.

Search for an implementation in this order:

1. the closest maintained optimized executor at the requested precision and
   hardware, especially `crates/apxinf-model/src/pi05/*_executor.rs` and
   `*_runtime.rs` for VLA execution;
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

After canonical inputs have been uploaded, intermediate tensors remain on the
target device through the final model output. The steady-state ledger must have:

- no intermediate device-to-host-to-device round trips;
- no host implementation of activation, indexing, interpolation, scatter,
  masking, positional encoding, or other layer mathematics;
- no synchronization introduced only to inspect or transform an intermediate;
- no per-layer or per-solver-step allocation that could have been prepared.

Host work is appropriate for checkpoint loading, one-time weight conversion,
canonical input preprocessing, explicit calibration, and final output transfer.
A CPU implementation inside a layer may be used briefly to establish numerical
evidence, but it is a **correctness scaffold**. Mark the affected ledger row,
profile its cost, and report it as performance debt until it is replaced. It
does not invalidate functional acceptance by itself, but it must never be an
unreported consequence of an absent backend-trait method.

## Plan tensor lifetime and reuse

Classify each value by when it changes:

- checkpoint lifetime: transformed weights, constant position tables;
- prepared-profile lifetime: masks, index maps, shape metadata, graph workspace;
- request lifetime: encoded images and language prefix, reusable cross-attention
  keys and values;
- solver-step lifetime: timestep embedding, noisy action state, step output;
- layer lifetime: transient projections and normalization scratch.

Compute or upload a value at the widest correct lifetime. In particular, audit
iterative VLA paths for prefix/cross-attention KV, masks, position data, and
timestep data that can be stored in fixed device buffers. A generic cache API
is not proof that the model uses the right cache lifetime.

## Prepare and capture fixed-shape execution

For a fixed target profile, allocate outputs and scratch at stable addresses.
Use `GraphWorkspace`, run the same inference body once through
`prepare_with_workspace` to validate shapes and prepare native plans, then run
it through `with_workspace` during CUDA Graph capture. Update captured input
contents in place and replay the graph through the maintained runtime.

Preparation must exercise the real executor. A method that only validates a
configuration object is not execution preparation. If graph capture is
unsupported for a required operation, record the exact operation and failure;
an eager fallback is a known performance gap, not silent success. Compare eager
and replayed outputs before relying on replay latency.

## Wiring review

Before reporting the port, provide evidence for all of the following:

- every ledger row resolves to a safe device call, a named correctness
  scaffold, or an explicit blocker;
- every repeated adjacency has a documented fused-versus-unfused decision;
- the steady-state host-transfer and synchronization list is empty except for
  public input/output boundaries;
- stable buffers and reusable KV/state are owned by the runtime at the correct
  lifetime;
- the fixed-shape path completes prepare, capture, input update, and replay, or
  records the concrete capture gap as performance debt/blocker;
- operator/layer replay, eager end-to-end, captured end-to-end, and public API
  checks pass their declared tolerances;
- wall-clock and graph-replay latency are reported separately, with any gap to
  the stated target attributed to measured operations where possible.

Report two independent outcomes:

- **functional acceptance** requires the reference tolerance, maintained public
  path, and clear unsupported-case behavior;
- **optimization status** is `target met`, `best effort with performance debt`,
  or `blocked`, with remaining host escapes, unfused hot sequences, missing
  reusable state, capture gaps, and measured impact listed explicitly.

Optimization is best effort unless the task explicitly defines it as a release
gate. The agent must investigate and attempt the applicable existing paths, but
an honest, measured performance gap does not erase a functionally correct port.

Only after this review should a genuinely missing row be handed to
[Adding New Kernels](adding-new-kernels.md). That guide explains how to add a
kernel vertically; this guide decides which safe interface the model should
call and how those calls form the execution path.

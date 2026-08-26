# Model Porting Workflow

This document is the workflow. It guides an agent porting a working reference
model into ApxInf without moving private source code, checkpoints, captures, or
credentials into the repository.

The agent owns investigation and implementation decisions. Commands and small
temporary programs are tools for collecting evidence, not a workflow engine.
The repository accepts a port only when the implementation and its independent
verification evidence agree.

Before implementation, read [Adding a New Model](adding-a-new-model.md). Use
[the model-layer architecture](model-layer-architecture.md) to decide ownership
and refactoring boundaries. Design the target device path with
[Model Execution Wiring](model-execution-wiring.md). When coverage is missing,
switch explicitly to
[Adding New Kernels](adding-new-kernels.md) and return here after the kernel has
been replayed against the original model evidence.

## 1. Define the port

Before editing code, record outside the repository:

- exact model variant, reference repository, and immutable revision;
- checkpoint identity and license constraints;
- target hardware and precision;
- representative input profiles;
- correctness tolerances and performance goals;
- expected public API and deployment path.

### Discover missing inputs and ask early

Treat the list above as a discovery contract, not as an assumption that the
user has already supplied every value. Before implementation, inspect only the
current task's allowed workspace and environment for evidence such as the
reference checkout, checkpoint configuration, configured execution hosts, GPU
capabilities, and CUDA toolchain. Do not infer a source revision, checkpoint
variant, target precision, or hardware target from a similarly named branch or
an unrelated previous experiment.

If required facts remain unknown or ambiguous after that inspection, ask the
user one concise, consolidated question. In particular, explicitly request the
missing parts of:

- reference source location and immutable revision;
- checkpoint location and exact model variant;
- target device or execution host, GPU/compute capability, and CUDA toolkit;
- target inference dtype; and
- mandatory correctness, latency, memory, or capture gates.

Ask as ordinary requirements gathering and name what each missing value
unlocks. Do not replace the question with a generic blocked-status message. A
useful request is, for example: "To run the reference and design the CUDA path,
please provide the reference source/revision, checkpoint, target GPU and CUDA
environment, and inference dtype."

This preflight is best effort rather than an all-or-nothing form:

- continue independent repository inspection and design work while waiting
  when it is safe and useful;
- use an explicit user-provided value even when environment discovery suggests
  a likely default;
- when no performance target is supplied, pursue functional and numerical
  acceptance first and report optimization as best effort;
- do not require the user to repeat facts that are already available in the
  current task or can be verified safely; and
- report a blocker only when a specific missing fact prevents further useful
  progress, after stating what was inspected, what operation is prevented, and
  exactly what information or access would resolve it.

The model variant, checkpoint, dtype, and target hardware are user choices, not
implementation details. Present one recommended tuple and explain material
alternatives. Record the confirmed or explicitly authorized tuple before
implementation; never infer consent from a convenient cache or silently
substitute a different model release.

Keep private paths and generated evidence under the private workspace described
in `AGENTS.md`. Do not commit model weights, captured tensors, environment
snapshots, or generated JSON reports.

## 2. Run the reference implementation

Create an isolated environment from the reference project's own dependency
lock. Disable network access while collecting evidence when practical. Record:

- Python and dependency versions;
- source revision and checkpoint digest;
- random seeds and stochastic inputs;
- preprocessing inputs and outputs;
- final outputs and selected intermediate tensors.

Run more than one representative input. A stochastic input such as action noise
may be an explicit model input even when it is only observable through the
output.

If the reference cannot load or produce a trace, stop and report that failure.
Do not treat a broken reference environment as an unsupported ApxInf model.

## 3. Inventory semantics

Describe computations by semantics, not only by framework operator names.
Include:

- tensor shapes, dtypes, layouts, and broadcasting;
- attention masks, positional encodings, and normalization formulas;
- preprocessing and postprocessing;
- state, timestep, noise, and other conditioning;
- dynamic branches and bounded iteration;
- weight transformations and tied parameters.

Separate three questions:

1. Is the reference computation understood?
2. Can it be represented using ApxInf model-neutral operations?
3. Does the target backend implement those operations for the required shapes?

Do not encode a model-family exception in shared workflow code.

## 4. Prove canonical equivalence

When rewriting the reference into an ApxInf-compatible representation, compare
the original and rewritten paths using the same inputs, weights, seeds, and
stochastic tensors.

For every rewrite, preserve:

- the source and destination tensor identities;
- the transformation applied;
- affected consumers;
- numerical comparisons at the rewrite boundary;
- final observation-to-output comparison.

Use the combined rule

```text
abs(actual - reference) <= atol + rtol * abs(reference)
```

Do not approve a rewrite solely because shapes match or the final output looks
plausible. If a supposedly canonical rewrite changes semantics, stop and report
the gap.

## 5. Design the target execution path

Before implementing the executor, build the execution ledger required by
[Model Execution Wiring](model-execution-wiring.md). Resolve repeated semantic
sequences against maintained optimized executors and the safe CUDA kernel
facade before consulting only the portable backend trait.

The design must identify fusion choices, tensor lifetimes, reusable KV/state,
workspace ownership, host transfers, and CUDA Graph eligibility. Any CPU
implementation inside the steady-state model graph is a temporary correctness
scaffold and must be reported as performance debt.

## 6. Check kernel coverage

For every computation or repeated composition required by the canonical model,
classify it as:

- existing fused implementation;
- existing primitive;
- layout-only adaptation;
- correct but slower fallback;
- missing capability;
- unsupported semantics.

Layout adaptations and device fallbacks require operator-level replay against
captured reference tensors. A declared fallback without replay evidence is a
kernel gap, not a passed capability. A host fallback in the hot path may satisfy
functional correctness, but remains visible performance debt even when its
numerical replay passes.

When a genuine gap is found, follow
[`adding-new-kernels.md`](adding-new-kernels.md). Hand off the operation's full
semantics, shapes, dtype, layout, tolerance, golden tensor identities, frequency,
performance impact, and expected safe Rust interface.

After kernel work returns, replay it against the original model references and
repeat coverage analysis. A returned implementation is not accepted merely
because it builds or passes a synthetic unit test.

## 7. Implement the model

Follow [Adding a New Model](adding-a-new-model.md), including its separate model
directory, runtime-contract choice, and YAGNI rules. Consult
[the model-layer architecture](model-layer-architecture.md) before extracting
shared code or moving behavior across the model/backend boundary.

Keep model code responsible for model structure, weight mapping, scheduling,
and composition. Model code may call safe model-neutral operators; it must not
call raw CUDA, vendor libraries, or FFI directly.

Prefer this order:

1. reuse the execution structure of a maintained optimized runtime;
2. reuse matching fused safe interfaces;
3. compose existing device primitives when fusion semantics do not match;
4. add a model-neutral operator when semantics are genuinely missing.

If implementation reveals that an earlier preflight assumption was wrong,
return to semantic inventory or kernel coverage. Finding more work is progress,
not completion.

## 8. Verify independently

Verification must be runnable independently of the agent's reasoning notes.
Use the repository's existing build, test, example, and benchmark mechanisms
where they apply. Temporary replay programs may live in the private workspace;
they do not need to become maintained repository scripts.

Verify in increasing scope:

1. changed operators on required devices and shapes;
2. transformed weights and important intermediate checkpoints;
3. complete deterministic inference against the reference;
4. the public policy or serving API;
5. requested target/precision tuples;
6. eager-versus-captured output parity;
7. host-transfer and synchronization audit of the steady-state path;
8. wall-clock and graph-replay latency plus memory against stated goals.

Correctness gates are mandatory. Optimization is best effort unless the request
explicitly declares a performance release gate. Report functional acceptance
separately from optimization status. Unmet latency, host escapes, or capture
gaps must include attempted reuse, measured impact where available, and the
next concrete optimization; they must not be hidden behind a generic fallback.

## 9. Prepare the review

The reviewable change should contain only maintained product material:

- model-neutral backend or operator changes;
- model runtime and public integration code;
- focused tests for maintained runtime or kernel behavior;
- concise documentation needed by future maintainers.

Do not commit:

- private reference adapters;
- checkpoint or tensor captures;
- generated evidence JSON;
- one-off replay/export scripts;
- model-specific workflow tests;
- schemas created only to describe a single experiment;
- agent orchestration state.

Summarize the source revision, checkpoint identity, target, precision,
correctness result, performance result, known limitations, and locations of
private reproducibility evidence in the review description.

## Completion criteria

Do not hand off a partial runtime as a completed port. A compilable foundation,
operator subset, checkpoint loader, or list of remaining components is an
intermediate milestone. Continue while the next required step is possible with
the available repository, reference implementation, hardware, and artifacts.

A port is complete only when:

- reference inputs and semantics are understood;
- canonical rewrites have numerical evidence;
- every required computation has validated coverage or a clear blocker;
- the maintained runtime loads the intended checkpoint;
- end-to-end output passes the declared tolerance;
- the public deployment path has been exercised;
- unsupported targets fail clearly;
- optimization opportunities, host escapes, static-buffer/KV reuse, and CUDA
  Graph eligibility have been audited;
- applicable existing optimized paths have been attempted and remaining
  performance debt is itemized;
- performance is measured with the declared metric and reported independently
  from functional acceptance;
- the repository diff contains no private or one-off experiment artifacts.

When the task explicitly makes a latency, memory, or capture target a release
gate, that target remains part of completion. Otherwise a functionally accepted
port may finish with optimization status `best effort with performance debt`.

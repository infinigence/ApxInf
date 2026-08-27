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
and refactoring boundaries. When coverage is missing, switch explicitly to
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

### Resolve missing inputs with the user

The model variant, checkpoint, dtype, and target hardware are user choices, not
implementation details. If any is missing, explicitly ask the user before
implementation. First gather enough evidence to make the question useful:

- inspect the model publisher's official GitHub repository and Hugging Face
  organization for maintained variants, compatible checkpoints, licenses, and
  reference precision;
- inspect the available execution environment for GPU model, compute
  capability, memory, CUDA toolkit, and existing official checkpoint caches;
- compare those facts with ApxInf's currently supported devices and dtypes.

Present one recommended default tuple and explain the tradeoff, plus alternatives
when they materially change scope or feasibility. Recommendations are proposals:
wait for the user to confirm or replace the model, checkpoint, dtype, and target
hardware. Record the confirmed tuple before proceeding. Never infer consent from
a convenient local cache or silently substitute a different model release.

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

## 5. Check kernel coverage

For every computation required by the canonical model, classify it as:

- existing fused implementation;
- existing primitive;
- layout-only adaptation;
- correct but slower fallback;
- missing capability;
- unsupported semantics.

Layout adaptations and fallbacks require operator-level replay against captured
reference tensors. A declared fallback without replay evidence is a kernel gap,
not a passed capability.

When a genuine gap is found, follow
[`adding-new-kernels.md`](adding-new-kernels.md). Hand off the operation's full
semantics, shapes, dtype, layout, tolerance, golden tensor identities, frequency,
performance impact, and expected safe Rust interface.

After kernel work returns, replay it against the original model references and
repeat coverage analysis. A returned implementation is not accepted merely
because it builds or passes a synthetic unit test.

## 6. Implement the model

Follow [Adding a New Model](adding-a-new-model.md), including its separate model
directory, runtime-contract choice, and YAGNI rules. Consult
[the model-layer architecture](model-layer-architecture.md) before extracting
shared code or moving behavior across the model/backend boundary.

Keep model code responsible for model structure, weight mapping, scheduling,
and composition. Model code may call safe model-neutral operators; it must not
call raw CUDA, vendor libraries, or FFI directly.

Prefer this order:

1. reuse an existing runtime abstraction;
2. reuse existing operators and layouts;
3. add a model-neutral operator when semantics are genuinely missing;
4. optimize only after correctness evidence exists.

If implementation reveals that an earlier preflight assumption was wrong,
return to semantic inventory or kernel coverage. Finding more work is progress,
not completion.

## 7. Verify independently

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
6. latency and memory against stated goals.

Correctness gates are mandatory. Performance goals may remain explicitly
unmet, but must not be reported as passed.

## 8. Prepare the review

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
- performance is measured and reported honestly;
- the repository diff contains no private or one-off experiment artifacts.

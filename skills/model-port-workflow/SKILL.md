---
name: model-port-workflow
description: Port a reference LLM, VLM, or VLA model into an ApxInf-native, fully GPU-resident execution path with private evidence, model-layer isolation, kernel-gap handling, and end-to-end verification. Use when adding a model family, migrating a checkpoint/runtime, assessing operator coverage, or preparing a model-port review.
---

# Model Port Workflow

## Required reading

Resolve the ApxInf checkout/worktree being modified first. All repository paths
below refer to that checkout, even when this skill is loaded through a global
symlink pointing at another checkout. Read its `AGENTS.md` for artifact placement,
then read the relevant documents from that same source revision:

1. [`doc/porting-workflow.md`](../../doc/porting-workflow.md) for the complete
   evidence and acceptance sequence.
2. [`doc/adding-a-new-model.md`](../../doc/adding-a-new-model.md) before creating
   or changing model-layer code. Follow its separate-directory and YAGNI rules.
3. [`doc/model-layer-architecture.md`](../../doc/model-layer-architecture.md)
   for the canonical module names, responsibility table, dependency direction
   and current family/option limits before deciding code placement.
4. [`doc/model-execution-wiring.md`](../../doc/model-execution-wiring.md) before
   composing model/Blocks calls, preparing a runner, or deciding that optimized
   coverage is missing. Its accelerator acceptance section defines the native
   GPU and required capture gates.
5. [`doc/adding-new-kernels.md`](../../doc/adding-new-kernels.md) whenever
   operator, dtype, layout, shape, or hardware coverage is missing.

Read each selected document completely. Treat them as instructions, not
background material.

## Assign module ownership before implementation

Use `doc/model-layer-architecture.md#current-module-names-and-responsibilities`
in the selected checkout as the source of truth. The port plan must name the
actual family-local owner for each responsibility:

| Responsibility | Placement / interface |
| --- | --- |
| Checkpoint config, implementation selection and construction | family `config`/`load`; existing Rust registry and `AutoModel` |
| Forward order, modality connections and flow schedule | family `Model`; precision-specific composition in Blocks where needed |
| Request binding, RNG, preparation, graph/cache lifetime | family `ModelRunner` implementing `VlaRuntime` for VLA |
| Host mapping, packing, device weights, fixed calibration | family `weights` |
| Public observation and action semantics | Python `<Family>Policy`, existing native `ModelRunner` binding |
| Device operation or kernel gap | safe model-neutral backend API, then provider implementation |

For PI0.5 these are `model/`, `model_runner/`, `weights/`, `load.rs` and the
existing policy. `ModelVariant` is private precision dispatch; `LoadedModel`
remains the common loader result. `StepModulation` names computed per-step
scale/shift/gate tensors. Use current source symbols for unmigrated families.
A single-implementation model does not need a variant enum, generic Blocks or
additional wrapper just to match PI0.5. Keep model and weights independent of
runner/capture types; keep concrete precision dispatch out of the runner.

New accelerator ports must satisfy
`doc/model-execution-wiring.md#accelerator-port-acceptance`: ApxInf-native GPU
computation, explicit public transfers, and required fixed-profile capture.
External engine and CPU paths are private references only. A required missing
device/capture path is a blocker, not performance debt. Existing-family status
and historical GPU evidence do not prove a new family's acceptance.

## Workflow

1. During preflight, reuse the task's confirmed inputs and execution mode; use
   hands-off for an implementation request unless the user chooses hands-on.
   Ask only for missing choices that affect the port. Offer discovered official
   GitHub source and
   Hugging Face checkpoint candidates; offer detected hardware defaults only
   for local Thor or Orin. Then fix the reference revision, checkpoint
   identity, target, precision, representative inputs, tolerances, performance
   goals, and public API. Do not silently pick user choices: record the
   confirmed or explicitly authorized tuple. Completion: every requested tuple
   and acceptance threshold is explicit.
2. Run the reference privately and capture deterministic inputs, outputs, and
   diagnostic tensors. Completion: the reference loads and repeated captures
   are attributable to the same source, environment, weights, and stochastic
   inputs.
3. Inventory semantics and weight transformations. Completion: every required
   computation is understood independently of its framework operator name.
4. Create an execution ledger from reference semantics through safe CUDA calls.
   Inspect maintained model/Blocks implementations and fused interfaces before the
   portable backend trait. Account for tensor lifetime, reusable KV/state,
   workspace, host traffic, and graph eligibility. Define the intended whole
   graph or Vision/Language/Action graph boundaries, stable buffers,
   input-update mechanism, and capture blockers. Completion: every graph row
   resolves to an ApxInf-native implementation, a private scaffold with a
   named device exit criterion, or a concrete blocker.
5. Classify fused and primitive coverage. If a real gap exists, follow
   `adding-new-kernels.md`, then replay the returned implementation against the
   original references. A CPU layer implementation is a named correctness
   scaffold with an exit criterion. On an accelerator target, replace every
   repeated hot-path scaffold with a safe device path; build cost and passing
   end-to-end values do not turn host round trips into deliverable performance
   debt.
6. Create `crates/apxinf-model/src/<model>/`. Start with a self-contained
   implementation. Inspect and copy a close model when useful, but do not import
   or modify another model-family directory for the new architecture. Defer
   shared extraction to a separate review after repeated maintained
   implementations prove the seam. Run
   `scripts/check_model_family_boundaries.sh` before review.
7. Follow `doc/adding-a-new-model.md#registration-and-public-integration`.
   Use `LlmTrait` and `LoadedModel::text` for LLM/VLM, or `VlaRuntime` on the
   family runner and `LoadedModel::Vla` for VLA. Wire Rust registration; for
   VLA, also wire Python policy discovery and reuse `apxinf.ModelRunner` / `apxinf_py.ModelRunner`
   and `model_runner` injection. Check the common loader's current
   `model_variant` admission and PI0.5-only config overrides before extending
   options. When adopting explicit preparation, read the implemented PI0.5
   section of `doc/model-lifecycle/lifecycle.md`. Test mode/fallback, stale-plan
   rejection, tuning-store replacement/generation, compatible run without
   capture/autotune, input/RNG rebinding and output/resource lifetime. Default
   `RuntimeManaged` status and unsupported policy methods are not readiness.
   Completion: registry loading, native contract, Python policy/public inference
   and family-specific GPU evidence agree in the same change.
8. Verify operators, transformations, intermediate checkpoints, eager and
   captured inference, host-transfer audit, public serving/policy integration,
   and requested performance. Prove that tensor computation between public
   input upload and public output transfer stays on GPU, and verify eager versus
   captured parity, repeated replay, changed-input propagation, stable
   addresses, and capture-safe workspace use. For a VLA, verify the whole-model
   graph or all three minimum Vision/Language/Action graphs. Report functional
   acceptance separately from optimization status (`target met`, `best effort
   with performance debt`, or `blocked`). Performance is best effort unless
   explicitly declared a release gate, but applicable existing optimized paths
   must be investigated.
9. Prepare a product-only diff. Store generated captures, reports, temporary
   adapters, replay scripts, generated plans, and agent state in the ignored
   `<project-root>/devlocal/<feat-name>/` directory according to `AGENTS.md`.
   Reuse existing external checkpoints and reference checkouts in place.
   Do not open a model-port review until correctness scaffolds are removed,
   required capture/replay passes, and the linked accelerator contract passes.
   Update module/capability docs, examples and relevant skills in the same PR;
   validate referenced source paths and keep historical results revision-labelled.

## Stop conditions

In hands-off mode, intermediate builds, numerical checkpoints, progress
summaries, and uncommitted changes are continuation points. Stop only at the
completion criteria, a concrete blocker, or an approval the agent cannot grant.
In hands-on mode, pause at named checkpoints with a concrete question and a
default next action, including whether to commit when that choice is useful.

Stop with a concrete blocker when the reference cannot run, semantics remain
unknown, canonical equivalence fails, a required kernel has no correct path, or
the maintained public integration cannot be exercised. Missing ApxInf-native
GPU coverage is a blocker; it does not authorize a third-party engine or CPU
partition. For a VLA, absence of both a whole-model graph and the complete
Vision/Language/Action fallback partition is also a blocker. A performance gap alone
is not a stop condition unless performance is an explicit release gate; exhaust
applicable existing paths, measure the gap, and report the remaining debt.

A partial foundation is not a completion condition. While safe in-scope work
remains, continue through the runtime, public integration, and end-to-end
reference comparison instead of ending with a list of unfinished components.
Stop early only when the next required action depends on unavailable external
information, authority, hardware, or artifacts, and state that dependency
precisely.

---
name: model-port-workflow
description: Port a reference LLM, VLM, or VLA model into an ApxInf-native, fully GPU-resident execution path with private evidence, model-layer isolation, kernel-gap handling, and end-to-end verification. Use when adding a model family, migrating a checkpoint/runtime, assessing operator coverage, or preparing a model-port review.
---

# Model Port Workflow

## Required reading

Read these repository documents before changing code:

1. [`doc/porting-workflow.md`](../../doc/porting-workflow.md) for the complete
   evidence and acceptance sequence.
2. [`doc/adding-a-new-model.md`](../../doc/adding-a-new-model.md) before creating
   or changing model-layer code. Follow its separate-directory and YAGNI rules.
3. [`doc/model-layer-architecture.md`](../../doc/model-layer-architecture.md)
   when deciding ownership, dependencies, runtime seams, or refactoring.
4. [`doc/model-execution-wiring.md`](../../doc/model-execution-wiring.md) before
   implementing an executor or deciding that optimized coverage is missing.
5. [`doc/adding-new-kernels.md`](../../doc/adding-new-kernels.md) whenever
   operator, dtype, layout, shape, or hardware coverage is missing.

Read each selected document completely. Treat them as instructions, not
background material.

## Non-negotiable execution contract

A completed accelerator port uses ApxInf as the inference runtime and execution
planner. Implement the model through ApxInf model code, safe model-neutral
operators, maintained fused interfaces, and backend kernels. Vendor math or
kernel libraries are allowed only behind ApxInf's safe backend/operator APIs.

Third-party inference engines are not an implementation path. Do not execute or
embed model graphs, subgraphs, layers, or generated plans through TensorRT,
ONNX Runtime, Torch/TorchScript, OpenVINO, or an equivalent external inference
runtime. Reference frameworks may run only in the private evidence workspace.

After canonical inputs enter the ApxInf runtime, the complete tensor
computation graph stays on the target GPU until the explicit public output
transfer. The host may load weights, prepare static metadata, upload public
inputs, issue launches for fixed control flow, and receive final output; it may
not compute intermediates, partition the graph onto CPU, or add intermediate
D2H/H2D transfers or synchronization.

CPU implementations and third-party-engine executions are private correctness
scaffolds only. They require an ApxInf-native GPU replacement and may not be
registered, shipped, benchmarked as the candidate implementation, or included
in a model-port review. If native coverage is missing, follow the kernel
workflow or report a concrete blocker.

The maintained GPU path must support CUDA Graph acceleration for fixed
profiles. Prefer one graph spanning canonical device inputs to outputs. When a
VLA cannot be captured as one graph after concrete blockers are identified, the
minimum partition is three captured graphs: Vision, Language, and Action, with
stable-address device buffers and device-only handoff. The Action graph
includes the complete fixed-step generation or denoising sequence. Dynamic
allocation, host tensor computation, intermediate readback, and synchronization
between graph segments are forbidden.

## Workflow

1. During preflight, ask once for hands-off or hands-on execution and the
   missing port inputs. Recommend hands-off and use it when an implementation
   request does not choose a mode. Offer discovered official GitHub source and
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
   Inspect maintained optimized executors and fused interfaces before the
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
7. Integrate through the appropriate maintained contract: `LlmTrait` for
   LLM/VLM or `VlaRuntime` plus the Python policy layer for VLA.
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
9. Prepare a product-only diff. Keep checkpoints, captures, generated reports,
   temporary adapters, replay scripts, generated engine plans, and agent state
   outside the repository. Do not open a model-port review until correctness
   scaffolds are removed, required capture/replay passes, and this execution
   contract passes.

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
partition. A VLA capture path with fewer than the required Vision/Language/
Action graphs is also a blocker. A performance gap alone
is not a stop condition unless performance is an explicit release gate; exhaust
applicable existing paths, measure the gap, and report the remaining debt.

A partial foundation is not a completion condition. While safe in-scope work
remains, continue through the runtime, public integration, and end-to-end
reference comparison instead of ending with a list of unfinished components.
Stop early only when the next required action depends on unavailable external
information, authority, hardware, or artifacts, and state that dependency
precisely.

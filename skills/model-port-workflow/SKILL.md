---
name: model-port-workflow
description: Port a reference LLM, VLM, or VLA model into ApxInf with private evidence, model-layer isolation, kernel-gap handling, and end-to-end verification. Use when adding a model family, migrating a checkpoint/runtime, assessing operator coverage, or preparing a model-port review.
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
4. [`doc/adding-new-kernels.md`](../../doc/adding-new-kernels.md) whenever
   operator, dtype, layout, shape, or hardware coverage is missing.

Read each selected document completely. Treat them as instructions, not
background material.

## Workflow

1. Fix the exact model variant and reference revision, checkpoint identity,
   target hardware, precision, representative inputs, tolerances, performance
   goals, and public API. When the user omits the model variant, checkpoint,
   dtype, or hardware, inspect the official GitHub/Hugging Face sources and the
   available execution environment, present recommended defaults with reasons,
   and explicitly ask the user to confirm or replace them. Do not silently pick
   these inputs. Completion: the user has confirmed every requested tuple and
   acceptance threshold.
2. Run the reference privately and capture deterministic inputs, outputs, and
   diagnostic tensors. Completion: the reference loads and repeated captures
   are attributable to the same source, environment, weights, and stochastic
   inputs.
3. Inventory semantics and weight transformations. Completion: every required
   computation is understood independently of its framework operator name.
4. Classify kernel coverage. Reuse model-neutral operations first. If a real
   gap exists, follow `adding-new-kernels.md`, then replay the returned
   implementation against the original references.
5. Create `crates/apxinf-model/src/<model>/`. Start with a self-contained
   implementation; copy a close model when useful and defer shared extraction
   until repeated maintained implementations prove the seam.
6. Integrate through the appropriate maintained contract: `LlmTrait` for
   LLM/VLM or `VlaRuntime` plus the Python policy layer for VLA.
7. Verify operators, transformations, intermediate checkpoints, complete
   inference, public serving/policy integration, and requested performance.
8. Prepare a product-only diff. Keep checkpoints, captures, generated reports,
   temporary adapters, replay scripts, and agent state outside the repository.

## Stop conditions

Stop with a concrete blocker when the reference cannot run, semantics remain
unknown, canonical equivalence fails, a required kernel has no correct path, or
the maintained public integration cannot be exercised. Report unmet
performance goals separately from correctness.

A partial foundation is not a completion condition. While safe in-scope work
remains, continue through the runtime, public integration, and end-to-end
reference comparison instead of ending with a list of unfinished components.
Stop early only when the next required action depends on unavailable external
information, authority, hardware, or artifacts, and state that dependency
precisely.

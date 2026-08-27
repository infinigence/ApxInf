# Agentic Optimization Framework

Status: Draft

This document describes the higher-level optimization loop you are building for
turning a user-provided PyTorch model into optimized Rust/C++ implementation
work. It is intentionally broader than the current porting workflow docs.

The core idea is:

> Do not let an agent blindly optimize a full PyTorch model end to end.
> Instead, lower the model into a structured intermediate form, search for
> optimization opportunities at multiple granularities, evaluate against real
> execution, and accumulate reusable knowledge for future tasks.

```text
User PyTorch model
        |
        v
Hierarchical lowering
        |
        v
Stage-level internal form
        |
        v
Multi-granularity search
        |
        v
Candidate generation
        |
        v
Static + execution evaluation
        |
        v
Feedback and knowledge accumulation
        |
        v
Better future optimization
```

## 1. Input to Optimization Space

The process starts from a user model, usually a PyTorch reference model.

The first step is hierarchical lowering:

- identify the model variant, checkpoint, dtype, hardware target, and public
  API;
- freeze the reference implementation and capture representative inputs and
  outputs;
- lower the model into a structured internal representation that exposes
  semantic execution stages.

This stage-level representation is not a node-by-node translation of the
PyTorch graph. Its purpose is to make the model analyzable and optimizable.

## 2. Multi-Granularity Search

Once the model has been lowered, the optimization system searches at multiple
levels.

### Inter-stage optimization

This covers structure between stages:

- scheduling between stage boundaries;
- memory reuse and buffer lifetime;
- producer/consumer layout transitions;
- pipeline opportunities;
- data movement across model components.

### Intra-stage optimization

This covers optimization within a stage:

- operator selection;
- kernel selection;
- layout choices;
- fusion choices;
- tiling and dispatch;
- precision selection.

The search space is therefore not a single global space. It is a set of nested
spaces with different scopes and different optimization levers.

```text
Stage-level internal form
        |
        +-------------------------------+
        |                               |
        v                               v
Inter-stage search               Intra-stage search
        |                               |
        |                               +---------------------------+
        |                               |                           |
        v                               v                           v
Scheduling / layout /         Operator / kernel / fusion /   Precision / tiling /
buffer reuse / pipeline       dispatch / local layout        local memory choice
```

## 3. Candidate Generation

The agent does not directly "write the final code" from the user request.
Instead, it generates candidates in the optimization space.

Examples:

- a different operator composition;
- a different kernel path;
- a different memory layout;
- a different fusion boundary;
- a different precision path;
- a different scheduling strategy.

The candidate can be Rust, C++, CUDA, or a combination, but it must remain
connected to the lowered semantics.

## 4. Evaluation

Every candidate must be evaluated against the physical world.

Evaluation has at least two parts:

- static validation: shapes, dependencies, correctness, buildability, numerical
  equivalence, and interface consistency;
- execution validation: runtime behavior, latency, throughput, memory, and
  target-hardware performance.

Roofline analysis belongs here as an analysis tool. It helps classify whether a
hotspot is likely memory-bound, compute-bound, or fusion-sensitive.

```text
Candidate
   |
   +--> static validation
   |       - shapes
   |       - dependencies
   |       - buildability
   |       - numerical equivalence
   |
   +--> execution validation
           - latency
           - throughput
           - memory
           - target-hardware behavior
           - roofline positioning
```

## 5. Feedback Loop

The key difference from a one-shot autotuner is the feedback loop.

Evaluation results are not just final scores. They become feedback for the next
search iteration.

If a candidate fails, the system should learn:

- which assumption was wrong;
- which semantic boundary was missed;
- which operator gap remains;
- which layout or precision choice was incorrect;
- which kernel path was too slow or unsupported.

If a candidate succeeds, the system should retain:

- the transformation pattern;
- the operator mapping;
- the hardware constraint;
- the performance evidence;
- the validation recipe.

## 6. Knowledge Accumulation and Self-Evolution

The framework is meant to improve across iterations and across tasks.

There are two forms of accumulation:

- task-internal iteration: repeatedly improve the same model port or the same
  hotspot;
- cross-task reuse: transfer successful patterns to later models or later
  hardware targets.

This is the main difference between a normal coding agent and an optimization
agent. The system should not just produce code. It should accumulate reusable
optimization knowledge.

```text
Task A
  |
  v
Search -> Evaluate -> Record useful pattern
  |
  v
Knowledge base
  |
  v
Task B
  |
  v
Reuse pattern -> Faster/better search
```

## 7. Relationship to Existing ApxInf Docs

This framework is conceptually broader than the existing ApxInf porting docs.

Relevant current documents:

- `doc/porting-workflow.md` defines the evidence-driven model port process;
- `doc/adding-a-new-model.md` defines model-layer ownership and boundaries;
- `doc/model-layer-architecture.md` defines what belongs in the model layer
  versus the backend;
- `doc/adding-new-kernels.md` defines the backend gap analysis and validation
  path;
- `scripts/roofline_analysis.py` is a standalone roofline analysis tool.

Those documents describe the mechanics. This document describes the higher-level
optimization loop that sits above them.

## 8. Practical Interpretation

In practice, the pipeline is:

1. accept user model and target constraints;
2. lower the model into a stage-oriented internal form;
3. identify inter-stage and intra-stage optimization candidates;
4. generate one or more implementation candidates;
5. validate correctness and measure performance;
6. feed the result back into the search policy and knowledge base;
7. reuse the learned pattern on future tasks.

If the pipeline is working well, the system becomes better at:

- finding the right abstraction boundary;
- choosing the right kernel or composition path;
- using roofline-style analysis to prioritize hotspots;
- reusing previous optimization experience without starting from zero.

```text
              +-----------------------------+
              |  Knowledge accumulation     |
              |  and self-evolution         |
              +--------------+--------------+
                             ^
                             |
                             |
User model -> Lowering -> Search -> Evaluation
                             |
                             v
                      Feedback to search
```

## 9. Non-Goals

This document is not:

- a replacement for the existing porting workflow;
- a specification of a universal compiler IR;
- a claim that all optimizations can be fully automated;
- a description of any single model family.

It is a framework for organizing optimization work so that agents can improve
through structured search and feedback rather than ad hoc trial and error.

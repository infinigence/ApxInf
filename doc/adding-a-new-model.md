# Adding a New Model to ApxInf

Use this guide after the reference model runs and its semantics are understood.
Follow [the porting workflow](porting-workflow.md) for evidence and acceptance,
[the model-layer architecture](model-layer-architecture.md) for ownership and
dependencies, and [the kernel guide](adding-new-kernels.md) for genuine backend
gaps. For a VLA, [adding an embodiment](adding-an-embodiment.md) covers the
Python-side serving contract this guide leaves to the policy layer: which
observation keys a client sends, how state is routed, and how to register a new
robot.

## Start with a separate model directory

Every model implementation starts in its own directory:

```text
crates/apxinf-model/src/<model>/
```

Do not add the new architecture to `llama/`, `qwen3vl/`, or `pi05/`, even when
most code looks similar. Register the new directory from `lib.rs` and
`builtin.rs`; keep model detection in the existing registry/`AutoModel` path.

The first correct implementation should be self-contained. Copying substantial
code from the closest model is preferable to inventing a shared abstraction
before the second implementation exposes the real common boundary. Preserve
provenance when copying and rename model-specific concepts immediately.

## Choose the runtime contract

### LLM and VLM

Autoregressive text models and vision-language models implement `LlmTrait`.
VLMs use `LlmInput` and override `prefill` for processor output, image-token
placement, multimodal positions, or other model-specific input semantics. They
reuse the shared token generation and sampling pipeline.

A small model commonly starts with:

```text
<model>/
  mod.rs
  config.rs
  weights.rs
  general.rs
```

Add files such as `vision.rs`, `vision_weights.rs`, or `decode_graph.rs` only
when the architecture requires them.

### VLA

VLA models implement `VlaRuntime`; they do not use the categorical token
generation loop. Their public contract is an observation-to-action inference
path with model-specific state, image, language, noise, schedule, and action
semantics.

PI0.5 is the maintained structural reference:

```text
pi05/
  mod.rs                 module wiring and deliberate exports
  backend.rs             the model's only CUDA-facing seam
  config.rs              checkpoint and execution configuration
  weights.rs             source checkpoint representation
  *_weights.rs           device/precision-specific weight forms
  math.rs                model mathematics without device ownership
  schedule.rs            denoising/execution schedule
  *_executor.rs          one precision's layer composition
  *_runtime.rs           device state, workspace and captured execution
  vla_runtime.rs         VlaRuntime adapter and registered loader
```

Create the new model's own directory and equivalent responsibilities. Do not
place its state encoder, action decoder, denoising schedule, embodiment logic,
or workspace inside `pi05/`. A first version may copy PI0.5 runtime or executor
structure extensively; correctness and isolation are the initial goal.

Python preprocessing, normalization, policy metadata, and action
postprocessing belong in the Python policy layer. Rust model code receives the
canonical tensors and owns model structure, weights, and execution.

## Implement by responsibility

### Configuration

Parse the reference checkpoint configuration without forcing it into a
Llama-shaped shared config. Validate dimensions and architecture invariants at
load time. Defaults are acceptable only when the reference format defines the
same defaults.

### Weights

Map checkpoint keys explicitly. Record every transpose, reshape, concatenation,
packing operation, tied weight, and precision conversion. Hugging Face linear
weights commonly require `[out, in]` to `[in, out]` transposition, while
higher-rank convolutional weights require semantics-aware flattening.

Keep source weights, device weights, and precision-specific packed weights as
separate concepts when their lifecycles differ. Perform stable transformations
once during load, not in every inference.

### Model mathematics

Express the architecture as model-level composition of safe backend operators.
The model owns layer ordering, residual structure, attention layout, schedules,
and fusion selection. The backend owns device management and individual kernel
APIs; it never imports model types.

Use `backend.rs` as the model directory's CUDA seam when the implementation
needs concrete CUDA facilities. Portable operations use `dyn Backend`; a
specialized fast path may recover a concrete backend for capabilities that do
not belong on the portable trait. Trait is the floor; concrete types are the
ceiling.

### Execution and preparation

Separate reusable mathematics from execution state. A runtime may own device
weights, caches, workspaces, CUDA graphs, and shape-specific preparation. A
prepared object must bind every shape or condition that changes allocation,
dispatch, or captured execution.

For VLA models, include state shape, image/grid structure, masks, action horizon,
action width, embodiment/category selection, and stochastic input shape where
they affect execution.

### Registration and public integration

Register the loader under stable model identifiers. Ensure the normal
`AutoModel` or `VlaRuntime` entry point can load the checkpoint; a private
example binary is not a deployment integration.

Expose Python policy support only after the Rust runtime contract is stable.
Keep preprocessing and postprocessing outside the low-level runtime.

## YAGNI and refactoring

Apply YAGNI when any of these is true:

- only one maintained model needs the behavior;
- the apparent commonality is based on names rather than identical semantics;
- a second model is still experimental or its shapes and lifecycle are unknown;
- the proposed abstraction would add optional fields, family branches, or a
  configuration language for hypothetical users;
- copying keeps failures local and makes reference comparison easier.

In those cases, keep the implementation in the model directory. Duplication is
an explicit temporary design choice, not an invitation to hide it in the
backend.

Consider refactoring only when:

- at least two maintained models contain the same stable semantics;
- both implementations have independent correctness evidence;
- the shared dependency direction is model → model-neutral helper/backend;
- the extracted API has fewer concepts than either caller and needs no
  model-family switch;
- changes to one copy repeatedly require the identical change in the other.

Extract the smallest proven seam. Good candidates are pure tensor
transformations, checkpoint utilities with identical formats, or genuinely
model-neutral operators. Schedules, layer topology, workspace layouts, and
precision dispatch normally remain model-specific until repeated evidence says
otherwise.

## Kernel decisions

Try existing operators and reshapes first. A missing dtype, layout, shape, mask,
or semantic variant may require backend work, but the new API must be
model-neutral. Follow [Adding New Kernels](adding-new-kernels.md) and return to
the original model references for operator replay.

## Verification

Verify progressively:

1. configuration and checkpoint identity;
2. every weight transformation;
3. representative operator and layer checkpoints;
4. full deterministic output against the reference;
5. registered Rust loading path;
6. Python policy or serving path when in scope;
7. requested device and precision combinations;
8. latency and memory, reported separately from correctness.

Temporary capture and replay programs belong in the private port workspace.
Commit only maintained product tests or examples whose purpose remains after
the port is complete.

## Completion criteria

The model has its own directory, uses the correct runtime contract, respects the
model/backend boundary, loads through the maintained registry, passes declared
numerical tolerances through the public path, reports unsupported cases
clearly, and introduces no speculative shared abstraction.

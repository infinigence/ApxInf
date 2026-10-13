# ApxInf Documentation

Use the current guides below for implementation. Each topic has one authority;
model-specific examples and historical evidence link back to it.

Start with the [architecture codemap](ARCHITECTURE.md) for what exists and
where it lives.

## How docs are organized

- doc/guides/ — how-to: complete a task (add a model, add a kernel, build,
  test, calibrate). Commands are named as justfile recipes only; the
  invocation lives in the justfile.
- doc/reference/ — contracts and catalogs: module responsibilities, model
  lifecycle, execution wiring, cuda-new API, environment variables.
- doc/models/ — one file per family: support matrix, how to run, limits.
  Benchmark numbers link to doc/reports/, they are not repeated here.
- doc/design/ — proposals and design decisions. Every file starts with a
  Status line (Draft / Accepted / Superseded / Historical).
- doc/reports/ — dated measurement reports. Frozen after publication; a
  new run creates a new file. Must record commit, board, and recipe.
- doc/archive/ — historical material (devlogs, retired plans). Kept for
  provenance; never a template for new code.

Do not document here: anything a recipe, type, or script already enforces
(commands go in the justfile, argument lists in --help), per-file tours,
TODOs (GitHub issues), or session notes (devlocal/).

## Add or change a model

| Read when | Guide / authority |
| --- | --- |
| Choosing names, module owners, dependency direction or checking family support | [Model Layer Architecture](reference/model-layer-architecture.md#current-module-names-and-responsibilities) |
| Locating shared contracts and family implementations | [Model Organization](reference/model-organization.md) |
| Implementing loading, a VLA runner/policy or LLM/VLM registration | [Adding a New Model](guides/adding-a-new-model.md) |
| Assigning asset, prepared-profile and request lifetimes | [Model Lifecycle and Contracts](reference/model-lifecycle.md) |
| Mapping model computation to device operations and required capture | [Model Execution Wiring](reference/model-execution-wiring.md) |
| Gathering references and completing independent acceptance evidence | [Porting Workflow](guides/porting-workflow.md) |
| A required safe device operation is missing | [Adding New Kernels](guides/adding-new-kernels.md) |
| Designing the native MLX backend or migrating a qualified Mac model | [Design: MLX Backend](design/mlx-backend.md) — design contract; no runtime support implied |

PI0.5 currently separates model/, model_runner/ and weights/. Its
[component view](archive/model-lifecycle/architecture.md#implemented-pi05-pilot-stage-2)
and [callable preparation contract](archive/model-lifecycle/lifecycle.md#implemented-pi05-preparation-contract)
describe implemented code. WallOSS/GR00T and LLM/VLM retain their own organization;
read the coverage table before assuming identical readiness or config support.

Qwen-Drive follows the same model/ / model_runner/ / weights/ separation as a
planning-only VLA runtime. Its [benchmark and validation guide](models/qwen-drive.md)
records measured performance, accuracy, reproduction steps and runtime limits.

Qwen3.8-27B-NVFP4 is a text LLM behind LlmTrait with the same weights/model/
model_runner owners as single files. [CLI inference, benchmarking and limits](models/qwen38.md)
cover Jetson Thor.

## Build and test

- [Building](guides/building.md) — the one way to build the binding, CLI,
  and platform presets.
- [Testing](guides/testing.md) — tiers, what CI runs, known-broken tests,
  and the GPU evidence rule for PRs.

## Agent workflows

- [model-port-workflow](../.agents/skills/model-port-workflow/SKILL.md): a complete new-family port, including module ownership, registration, public integration and GPU evidence.
- [analyze-model-impl](../.agents/skills/analyze-model-impl/SKILL.md): trace the selected implementation from model/runner through actual backend calls; source inspection alone is not a benchmark.
- [l3-reference-testing](../.agents/skills/l3-reference-testing/SKILL.md): independent numerical tests for apxinf-cuda-new L3 semantics/candidates; these are separate from current PI0.5 backend and model-level acceptance.
- [pre-pr](../.agents/skills/apxinf-pre-pr/SKILL.md): run the pre-PR check and review your diff before opening a PR.

Skills resolve these documents from the checkout/worktree being modified,
even when installed globally through a symlink to another branch. Repository
skills and their linked guides are maintained together; task evidence lives in
ignored devlocal/<feat-name>/ according to [AGENTS.md](../AGENTS.md).

## Design and reports

- [Design principles](design/design-principles.md) — the backend/model
  layering rule and its recorded violations.
- [cuda-new design](design/cuda-new-design.md), [optimization workflow](design/agentic-optimization-workflow.md),
  [MLX backend](design/mlx-backend.md).
- [Reports](reports/) — dated measurement reports (frozen).
- [Archive](archive/) — migration tracker, baseline protocol, Metal kernel
  plan, and the pre-taxonomy architecture/lifecycle pages. Historical
  Session/Network names there are not a template for new code.

## Model operation

- [PI0.5](models/pi05.md) — including the CUDA regression protocol.
- [PI0-FAST](models/pi0fast.md)
- [GR00T N1.7](models/gr00t.md)
- [SmolVLA](models/smolvla.md)
- [Qwen-Drive](models/qwen-drive.md)
- [Qwen3.8 NVFP4](models/qwen38.md)
- [Python API and policies](../python/apxinf/README.md)
- [Native binding](../crates/apxinf-py/README.md)
- [One-step warm-start evaluation](reports/2026-07-04-run-warmstart-with-onestep.md)

# ApxInf Documentation

Use the current guides below for implementation. Each topic has one authority;
model-specific examples and historical evidence link back to it.

## Add or change a model

| Read when | Guide / authority |
| --- | --- |
| Choosing names, module owners, dependency direction or checking family support | [Model Layer Architecture](model-layer-architecture.md#current-module-names-and-responsibilities) |
| Locating shared contracts and family implementations | [Model Organization](model-organization.md) |
| Implementing loading, a VLA runner/policy or LLM/VLM registration | [Adding a New Model](adding-a-new-model.md) |
| Assigning asset, prepared-profile and request lifetimes | [Model Lifecycle and Contracts](model-lifecycle.md) |
| Mapping model computation to device operations and required capture | [Model Execution Wiring](model-execution-wiring.md) |
| Gathering references and completing independent acceptance evidence | [Porting Workflow](porting-workflow.md) |
| A required safe device operation is missing | [Adding New Kernels](adding-new-kernels.md) |

PI0.5 currently separates `model/`, `model_runner/` and `weights/`. Its
[component view](model-lifecycle/architecture.md#implemented-pi05-pilot-stage-2)
and [callable preparation contract](model-lifecycle/lifecycle.md#implemented-pi05-preparation-contract)
describe implemented code. WallOSS/GR00T and LLM/VLM retain their own organization;
read the coverage table before assuming identical readiness or config support.

Qwen-Drive follows the same `model/`, `model_runner/`, `weights/` separation as a
planning-only VLA runtime. Its [loading, request and validation contract](qwen-drive-planning.md)
covers the supported modes and limits; [per-device kernel policy, tactic stores and the
measurement protocol](qwen-drive-devices.md) cover Orin, Thor and the RTX 4090.

## Agent workflows

- [model-port-workflow](../skills/model-port-workflow/SKILL.md): a complete new-family port, including module ownership, registration, public integration and GPU evidence.
- [analyze-model-impl](../skills/analyze-model-impl/SKILL.md): trace the selected implementation from model/runner through actual backend calls; source inspection alone is not a benchmark.
- [l3-reference-testing](../skills/l3-reference-testing/SKILL.md): independent numerical tests for `apxinf-cuda-new` L3 semantics/candidates; these are separate from current PI0.5 backend and model-level acceptance.

Skills resolve these documents from the checkout/worktree being modified,
even when installed globally through a symlink to another branch. Repository
skills and their linked guides are maintained together; task evidence lives in
ignored `devlocal/<feat-name>/` according to [AGENTS.md](../AGENTS.md).

## Historical design and validation

- [Lifecycle migration tracker](model-lifecycle/migration.md): staged scope, source revisions and model/precision qualification.
- [Baseline protocol and GPU evidence](model-lifecycle/baseline.md): historical measurements with their original revisions.
- [Earlier Metal kernel plan](zippy-hugging-cook.md): historical proposal with source paths that no longer exist; not a current backend or model template.
- The architecture and detailed lifecycle pages keep earlier baseline/proposal
  sections at the end. Those sections explain design history; their old
  Session/Network names and proposed APIs are not a template for new code.

## Model operation

- [PI0.5 CUDA regression](pi05-cuda-regression.md)
- [PI0.5 one-step warm-start evaluation](run_warmstart_with_onestep.md)
- [GR00T N1.7](gr00t-n1.7.md)
- [Python API and policies](../python/apxinf/README.md)
- [Native binding](../crates/apxinf-py/README.md)

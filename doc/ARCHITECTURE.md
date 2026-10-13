# ApxInf Architecture

This is the short codemap: what exists, where it lives, and the invariants
that are not expressible as types. It is intentionally small — for contracts
see doc/reference/, for per-family guides see doc/models/, and for design
history see doc/design/. Rows here age quickly; when they disagree with the
code, the code wins and this file must be fixed in the same PR.

## Workspace crates

| Crate | Responsibility | Notes |
|---|---|---|
| apxinf (root) | CLI binary (apxinf generate / test) | thin shell over the workspace |
| apxinf-core | Tensor, Shape, DType, Storage, Device, CPU ops | no CUDA, no model code |
| apxinf-loader | SafeTensors / GGUF parsing, ModelConfig | host-side only |
| apxinf-tokenizer | HF fast tokenizer, chat templates | optional sentencepiece feature |
| apxinf-cuda | Legacy CUDA backend: context, buffers, cuBLAS, custom kernels | new kernels go to apxinf-cuda-new, not here |
| apxinf-cuda-new | Current CUDA backend: L3 semantics, candidates, graphs | kernel rules in its AGENTS.md |
| apxinf-hip | HIP backend skeleton | new; minimal surface |
| apxinf-model | All model architecture: llm_trait, VlaRuntime, families | see families below |
| apxinf-py | PyO3 binding (ModelRunner) | built via just build-py |

## apxinf-model families

Under crates/apxinf-model/src/: shared contracts live at the root
(lib.rs, auto.rs, registry.rs, llm_trait.rs, vla/); each family
(pi05, walloss, gr00t, llama, qwen38, qwen3vl, qwen_drive, smolvla,
pi0fast) is self-contained — no family imports another family's private
model, weights, runner, graph, or backbone. New VLA code names forward
computation Model and execution ownership ModelRunner; those are roles,
not shared base classes. Details and the responsibility table:
doc/reference/model-layer-architecture.md and model-organization.md.

## Layering invariants

1. Backend is device management + single-kernel APIs. Model structure
   lives in apxinf-model. The two never cross (doc/design.md).
2. Families never import each other's private modules — enforced by
   scripts/check_model_family_boundaries.sh (just boundaries).
3. Python calls into Rust through the PyO3 binding; scripts/ are thin
   drivers and must never import just.

## Command entry point

justfile is the single source for build/test/bench commands; docs name
recipes, never full invocations (doc/README.md#how-docs-are-organized).
Platform build presets (arch, nvcc parallelism) live in
configs/platforms/.

## Documentation map

- doc/guides/ — how-to procedures (adding a model/kernel, building,
  testing, calibration).
- doc/reference/ — contracts and catalogs (model-layer architecture,
  lifecycle, execution wiring, cuda-new API).
- doc/models/ — one file per family: support matrix, how to run, limits.
- doc/design/ — proposals and design decisions with explicit Status.
- doc/reports/ — dated measurement reports; frozen after publication.
- doc/archive/ — historical material, kept for provenance only.

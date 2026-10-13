# Repository Instructions

ApxInf is a Rust + CUDA inference engine for VLA models (PI0.5, GR00T,
Qwen-Drive, SmolVLA, ...) with a PyO3 binding and a Python frontend. Read
[doc/ARCHITECTURE.md](doc/ARCHITECTURE.md) for the codemap and
[doc/README.md](doc/README.md) for documentation rules before changing docs.

## Commands

Run `just --list` for the full menu. Common recipes:

- `just build-py` — build and install the Python binding + frontend
  (`--features cuda` is baked in; without it the runtimes silently don't
  exist).
- `just check` — type-check the whole workspace INCLUDING cfg(cuda) code;
  needs no GPU (build scripts skip kernels without a toolkit).
- `just test-cpu` — CPU-only Rust + Python tests. `just test-gpu <family>`
  and `just bench <model>` need an NVIDIA device.
- `just pre-pr` — the full pre-PR check; CI runs the same recipes.

Pitfalls:

- `cargo check` WITHOUT `--features cuda` passes while skipping ~230
  cfg(cuda) sites — that is not a real check. Use `just check`.
- Jetson builds OOM unless nvcc parallelism is capped; the platform
  presets in `configs/platforms/` (selected via
  `APXINF_PLATFORM=<name> just build-py`) already set it.
- When no CUDA toolkit is found, build.rs skips kernel compilation
  silently — a "successful" build may still lack GPU kernels.

## Testing tiers

| Tier | Recipe | Hardware |
|---|---|---|
| Rust unit tests | `just test-rust-cpu` | CPU |
| Python unit tests | `just test-py` | CPU |
| Family GPU tests | `just test-gpu <family>` | NVIDIA + checkpoint |
| L3 candidate tests | `just test-l3` | NVIDIA |
| Benchmarks | `just bench <model>` | NVIDIA |

If you have no suitable device, write "not verified" — never "pass".

## Boundaries

- Always: docs and code change in the same PR; new `APXINF_*` env vars get
  registered in `doc/reference/env-vars.md`; intermediate artifacts go to
  `devlocal/<feat-name>/` (gitignored, never force-add).
- Ask first: changing the PyO3 binding ABI or the Python `Policy` API;
  touching vendored CUTLASS/fa2 code; deleting tests.
- Never: new kernels in legacy `apxinf-cuda` (use `apxinf-cuda-new`);
  claiming GPU results without a device; hardcoding host-specific paths
  into maintained docs.

## Model integration

When adding a family or changing precision/preparation, read
[Model Layer Architecture](doc/reference/model-layer-architecture.md#current-module-names-and-responsibilities)
and [Adding a New Model](doc/guides/adding-a-new-model.md). Use the
`model-port-workflow` skill for a full port. Historical Session/Network
names in `doc/archive/` are not integration APIs.

## Review

Review against [REVIEW.md](REVIEW.md). See  for the full workflow, PR evidence
requirements, and AI-assistance policy.

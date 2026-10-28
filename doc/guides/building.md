# Building ApxInf

All commands go through the justfile; `just --list` shows the menu.

## Prerequisites

- Linux host with an NVIDIA driver and CUDA toolkit (build on the machine you
  deploy to: the build compiles kernels for the visible GPU's compute
  capability).
- A stable Rust toolchain (`rustup default stable`) and
  [just](https://just.systems) (`cargo install just`); verify `just --version`
  in the same shell used for builds. It is needed for the repository commands,
  not for running an already-installed inference package.
- Python 3.10+ with a venv or conda env.
- Install maturin in that environment with `python -m pip install maturin`.

## Build the Python binding and frontend

```bash
just build-py
```

This runs `maturin build --release --features cuda --auditwheel skip` for
`crates/apxinf-py`, installs the wheel, and installs `python/apxinf[serving]`.
`--features cuda` is required: without it the PI0.5/WallOSS runtimes are not
compiled in and loading errors at runtime.

**Tuning databases (tactics).** The default GPU build resolves per-board
tactic JSONs from `configs/tuning/nvidia/<board>/tactics.json` via
`apxinf._tactics.resolve_pi05_tactics`. L2 policies wire this in
automatically; L1 bare-handle callers must pass `tactics=` explicitly to
`ModelRunner.load(...)` or they silently fall back to untuned default GEMM
keys (see `python/apxinf/AGENTS.md`).

**An `undefined symbol` import failure needs a build-and-load diagnosis.**
Identify the extension Python actually loads, the build's Python ABI, and the
missing symbol's caller and definition. Check whether the selected architecture
and template instantiations compile and link that definition. A completed build
alone does not establish that the extension can import; rebuilding cannot fix
a definition excluded by the source configuration.

Stale artifacts are another possibility. The CUDA object cache uses dependency
mtimes and command stamps; restoring source with preserved timestamps can permit
reuse of an object built from different content. File timestamps or hashes alone
do not prove source-to-object provenance. When that provenance is uncertain,
preserve the suspect artifacts and run `build-py` in a separate checkout with
its own `target/wheel` directory. Record the source and build settings, verify
which new extension is loaded, and test import before inference. Import success
and model numerical acceptance are separate checks.

## Platform presets

Per-platform build variables live in `configs/platforms.tsv`
(`thor`, `thor-u`, `orin`, `rtx4090`, `gb10`) and are selected with:

```bash
APXINF_PLATFORM=thor just build-py
```

`build-py` and `test-l3` load the table through `scripts/load_platform.sh`.
Other recipes do not automatically load it. Bash callers can source that script
before invoking their own build command. With no platform selected, inherited
variables and existing command defaults are unchanged; unknown names fail before
any build or installation.

A selected preset overrides `APXINF_CUDA_ARCH` and, on Jetson, caps
`APXINF_NVCC_PARALLELISM` to avoid OOM during parallel nvcc compiles. Set
`CUDA_PATH`/`CUDA_HOME` only if CUDA is not at `/usr/local/cuda`.
The table's `-` leaves the inherited parallelism or build default unchanged.
To choose custom architecture or Jetson parallelism, leave `APXINF_PLATFORM`
unset and set those variables explicitly. These settings affect compilation,
not inference parallelism or speed. Device builds for all presets are not verified.

## Build the CLI

```bash
just build-cli
```

## Cross-compilation

Cross-compiling fails unless `APXINF_CUDA_ARCH` names the target (`sm_87`
Orin, `sm_101` Thor-U, `sm_110` Thor, `sm_89` RTX 4090, `sm_120` GB10).
Use a preset rather than exporting the variables by hand.

## AOT bundles

Qwen-Drive consumes pre-exported AOT operator artifacts. Pass the manifest
through the environment; `just build-py` forwards it:

```bash
APXINF_PLATFORM=thor APXINF_CUDA_AOT_MANIFEST=/path/to/manifest.json just build-py
```

See [crates/apxinf-cuda/aot/README](../../crates/apxinf-cuda/aot/README.md)
for exporting bundles, and [models/qwen-drive](../models/qwen-drive.md) for
the validation procedure.

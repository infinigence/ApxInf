# Building ApxInf

All commands go through the justfile; `just --list` shows the menu.

## Prerequisites

- Linux host with an NVIDIA driver and CUDA toolkit (build on the machine you
  deploy to: the build compiles kernels for the visible GPU's compute
  capability).
- A stable Rust toolchain (`rustup default stable`) and
  [just](https://just.systems) (`cargo install just`).
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

**If `import apxinf_py` fails with `undefined symbol` right after a build
that reported success, the wheel on disk is stale, not your code.** The CUDA
kernel cache keys on source mtime plus a hashed `.cmdline` stamp
(`crates/apxinf-cuda/build_support/nvcc_build.rs:64`,
`crates/apxinf-cuda/build.rs:240`); restoring `.cu` files with preserved
timestamps (`cp -p`, `tar -x`, `scp -p`) can leave objects that look fresh to
Cargo but predate your edit. Confirm with
`ls -l target/wheel/wheels/apxinf_py-*.whl`, then force a clean kernel
rebuild:

```bash
cargo clean -p apxinf-cuda -p apxinf-cuda-new
just build-py
```

## Platform presets

Per-platform build variables live in `configs/platforms/<name>.env`
(`thor`, `thor-u`, `orin`, `rtx4090`, `gb10`) and are selected with:

```bash
APXINF_PLATFORM=thor just build-py
```

Each preset sets `APXINF_CUDA_ARCH` and, on Jetson, caps
`APXINF_NVCC_PARALLELISM` to avoid OOM during parallel nvcc compiles. Set
`CUDA_PATH`/`CUDA_HOME` only if CUDA is not at `/usr/local/cuda`.

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

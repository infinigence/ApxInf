# ApxInf command entry point.
#
# Every recipe is a thin wrapper: 1-3 lines, environment variables fixed
# here instead of being spread across documentation. `just --list` prints
# this comment as the menu, so recipes are self-documenting.
#
# Docs reference recipes by name only (see doc/README.md#how-docs-are-organized).
#
# Platform presets (arch, CUDA path, nvcc parallelism) live in
# configs/platforms/<name>.env — see doc/ARCHITECTURE.md.

# Load a platform preset: APXINF_PLATFORM=thor build-py
# (vars only export to child processes when the recipe itself uses them;
# for other tools use `set -a; source configs/platforms/$APXINF_PLATFORM.env; set +a`.)
export APXINF_PLATFORM := env("APXINF_PLATFORM", "")

# List all recipes.
default:
    @just --list

# Build and install the Python binding (apxinf_py) plus the frontend (requires --features cuda).
build-py:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "{{ APXINF_PLATFORM }}" ]; then
      set -a; source "configs/platforms/{{ APXINF_PLATFORM }}.env"; set +a
    fi
    CARGO_TARGET_DIR=target/wheel \
      maturin build --release --features cuda --auditwheel skip \
      -m crates/apxinf-py/Cargo.toml
    pip install --force-reinstall target/wheel/wheels/apxinf_py-*.whl
    pip install -e "python/apxinf[serving]"

# Build the apxinf CLI binary.
build-cli:
    cargo build --release --features cuda --bin apxinf

# Type-check lib and bin targets including cfg(feature="cuda") code.
# `--all-targets` is excluded while an upstream test target is broken.
check:
    cargo check --workspace --lib --bins --features cuda

# Format all Rust code.
fmt:
    cargo fmt --all

# Check formatting without modifying files.
# Advisory: main carries pre-existing rustfmt drift; a cleanup PR should
# run `just fmt` separately. pre-commit and pre-pr do not include it.
fmt-check:
    cargo fmt --all --check

# Lint (non-blocking in CI until existing warnings are reduced).
clippy:
    cargo clippy --workspace --lib --bins --features cuda

# CPU-only Rust tests.
test-rust-cpu:
    cargo test -p apxinf-core -p apxinf-loader -p apxinf-model -p apxinf-tokenizer --lib

# CPU-only Python tests (two root tests/ modules are broken on main and excluded here).
test-py:
    python3 -m pytest python/apxinf/tests tests \
      --ignore=tests/test_libero_observation.py \
      --ignore=tests/test_pi05_openpi_websocket.py \
      -k "not test_run_episode_obeys_explicit_max_steps \
          and not test_websocket_model_variant_cli_reaches_evaluator \
          and not test_from_pretrained_rejects_options_that_cannot_mean_anything_here" \
      -q -p no:cacheprovider

# All CPU-only tests.
test-cpu: test-rust-cpu test-py

# GPU tests for one model family (requires an NVIDIA device + checkpoint).
# e.g. just test-gpu pi05
test-gpu family:
    cargo test --release -p apxinf-model --features cuda {{ family }}
    cargo test --release -p apxinf-cuda-new --features cuda {{ family }}

# L3 candidate tests for apxinf-cuda-new; arch/target-dir overridable, see doc/guides/testing.md.
test-l3 *args:
    #!/usr/bin/env bash
    set -euo pipefail
    export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
    if [ -n "{{ APXINF_PLATFORM }}" ]; then
      set -a; source "configs/platforms/{{ APXINF_PLATFORM }}.env"; set +a
    fi
    : "${CUDA_PATH:=/usr/local/cuda}"
    : "${APXINF_CUDA_ARCH:=sm_110}"
    : "${CARGO_TARGET_DIR:=$PWD/target-gemm-pilot}"
    export CUDA_PATH APXINF_CUDA_ARCH CARGO_TARGET_DIR
    cargo test -p apxinf-cuda-new {{ args }}

# Run one benchmark script, e.g. just bench pi05 --layer l1 --samples 5.
bench model *args:
    python scripts/bench_{{ model }}.py {{ args }}

# Documentation lint (taxonomy, budget, link and recipe references).
doc-lint:
    python3 scripts/check_docs.py

# Architecture boundary checks.
boundaries:
    bash scripts/check_model_family_boundaries.sh

# Pre-commit quick check (target < 60 s).
pre-commit: doc-lint boundaries

# Pre-PR check; CI runs the same set.
pre-pr: pre-commit check test-cpu

# Run one of the repo skills from just (handy in CI):
#   just skill model-port-workflow
skill name *args:
    echo "skills are invoked by the agent runtime, e.g. /{{ name }} {{ args }}"

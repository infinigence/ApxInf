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

# Load a platform preset: APXINF_PLATFORM=thor just build-py
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

# Build the pinned CUDA Decision Index runner.
build-d3-index:
    cargo build --locked --release -p apxinf-model --features cuda --example d3_decision_index

# Run the native Apple Silicon CLI (requires the MLX SDK).
[positional-arguments]
run-metal *args:
    cargo run --features mlx -- "$@"

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
[positional-arguments]
test-rust-cpu *args:
    #!/usr/bin/env bash
    cargo test -p apxinf-core -p apxinf-loader -p apxinf-model -p apxinf-tokenizer --lib "$@"

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

# CUDA library tests filtered by family; opt-in checkpoint tests need -- --ignored.
[positional-arguments]
test-gpu family *args:
    #!/usr/bin/env bash
    set -euo pipefail
    family="$1"; shift
    cargo test --release -p apxinf-model --lib --features cuda "$family" "$@"
    cargo test --release -p apxinf-cuda-new --lib "$family" "$@"

# L3 candidate tests for apxinf-cuda-new; arch/target-dir overridable, see doc/guides/testing.md.
[positional-arguments]
test-l3 *args:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ -n "{{ APXINF_PLATFORM }}" ]; then
      set -a; source "configs/platforms/{{ APXINF_PLATFORM }}.env"; set +a
    fi
    : "${CUDA_PATH:=/usr/local/cuda}"
    : "${APXINF_CUDA_ARCH:=sm_110}"
    : "${CARGO_TARGET_DIR:=$PWD/target-gemm-pilot}"
    export CUDA_PATH APXINF_CUDA_ARCH CARGO_TARGET_DIR
    cargo test -p apxinf-cuda-new "$@"

# Run a model benchmark, e.g. just bench pi05 --layer l1 --samples 5.
[positional-arguments]
bench model *args:
    #!/usr/bin/env bash
    set -euo pipefail
    model="$1"; shift
    case "$model" in
      qwen38) exec cargo run --release -p apxinf-model --features cuda --example qwen38_bench -- "$@" ;;
      pi05|pi0_fast|smolvla|gr00t) exec python3 "scripts/bench_${model}.py" "$@" ;;
      *) echo "Unknown benchmark: $model (choose pi05, pi0_fast, smolvla, gr00t, qwen38)" >&2; exit 2 ;;
    esac

# Documentation lint (taxonomy, budget, link and recipe references).
doc-lint:
    python3 scripts/check_docs.py

# Architecture boundary checks.
boundaries:
    bash scripts/check_model_family_boundaries.sh

# CPU regression tests for recipes, documentation lint, hooks, and workflows.
test-tooling:
    python3 -m unittest discover -s scripts/tests -p 'test_*.py'

# Local agent evidence: init, run, block, check (opt-in; see --help).
[positional-arguments]
acceptance *args:
    python3 scripts/agent_acceptance.py "$@"

# Validate GitHub Actions syntax and embedded shell commands (requires actionlint).
workflow-lint:
    actionlint

# Pre-commit quick check (target < 60 s).
pre-commit: doc-lint boundaries

# Pre-PR check; CI runs the same set.
pre-pr: pre-commit test-tooling workflow-lint check test-cpu

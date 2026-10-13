# Testing ApxInf

All entry points are justfile recipes — just --list shows the menu. This
guide explains which tier runs where and what evidence to attach to a PR.

## Tiers

| Tier | Recipe | Needs | Runs in CI |
|---|---|---|---|
| Rust unit tests | just test-rust-cpu | CPU only | yes |
| Python unit tests | just test-py | CPU only | yes |
| Both | just test-cpu | CPU only | yes |
| Model family GPU tests | just test-gpu <family> | NVIDIA device + checkpoint | no, attach output to the PR |
| cuda-new L3 candidate tests | just test-l3 [extra args] | NVIDIA device | no |
| Benchmarks | just bench <model> [args] | NVIDIA device | no |

just check type-checks every lib and bin target including
cfg(feature=cuda) code and needs no GPU. It deliberately skips
--all-targets: the qwen38_gdn_batched_ops test target does not compile on
main (pre-existing upstream breakage; the test references
ops::gdn_prepare_flashinfer and ops::convert_f16_to_bf16, which no longer
exist). Fix that in a code PR, then re-enable --all-targets here.

## Known-failing and excluded tests (as of main fde5feb3)

These are excluded from just test-py and just check until fixed in code
PRs; they are not silently green:

- tests/test_libero_observation.py: IndentationError at line 67 upstream.
- tests/test_pi05_openpi_websocket.py: imports the undeclared
  openpi_client package.
- tests/test_eval_libero_norm_stats.py::test_run_episode_obeys_explicit_max_steps
  fails on a CPU-only run.
- python/apxinf/tests/test_examples.py: 3 websocket CLI tests fail
  (bf16, fp8_static, int8_dynamic variants).
- python/apxinf/tests/test_pi0fast_policy.py::test_from_pretrained_rejects_options_that_cannot_mean_anything_here
  fails on CPU.

## Running a specific test

    just test-l3 test -p apxinf-cuda-new -- --nocapture --test-threads=1
    cargo test -p apxinf-core test_matmul_cpu
    pytest python/apxinf/tests/test_tokenize.py

just test-l3 defaults to APXINF_CUDA_ARCH=sm_110 and a dedicated
target-gemm-pilot directory (matching the old test-new.sh behavior);
override either with just test-l3 APXINF_CUDA_ARCH=sm_89 or a platform
preset: APXINF_PLATFORM=orin just test-l3.

## GPU acceptance

just test-gpu <family> runs the family Rust tests with
--features cuda --release. PRs that touch GPU behavior must attach the
recipe output plus board, arch, and commit, per the PR template. If you
have no suitable device, say "not verified" — do not report "pass".

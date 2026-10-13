# Testing ApxInf

All entry points are justfile recipes — just --list shows the menu. This
guide explains which tier runs where and what evidence to attach to a PR.

CI runs Rust tests serially (`RUST_TEST_THREADS=1`) because the synthetic
PI0.5 weight tests allocate full models. It sets
`CARGO_PROFILE_TEST_OPT_LEVEL=1` to speed up their weight generation while
retaining test-profile debug assertions and overflow checks. Neither
setting removes tests or changes their inputs.

## Tiers

| Tier | Recipe | Needs | Runs in CI |
|---|---|---|---|
| Rust unit tests | just test-rust-cpu | CPU only | yes |
| Python unit tests | just test-py | CPU only | yes |
| Both | just test-cpu | CPU only | yes |
| Repository tooling | just test-tooling | CPU + Python test dependencies + just | yes |
| Model family GPU library tests | just test-gpu <family> | NVIDIA device + checkpoint | no, attach output to the PR |
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

    just test-l3 --lib -- --nocapture --test-threads=1
    just test-rust-cpu test_matmul_cpu
    pytest python/apxinf/tests/test_tokenize.py

just test-l3 defaults to APXINF_CUDA_ARCH=sm_110 and a dedicated
target-gemm-pilot directory (matching the old test-new.sh behavior);
override either with APXINF_CUDA_ARCH=sm_89 just test-l3 --lib or a platform
preset: APXINF_PLATFORM=orin just test-l3.

## GPU acceptance

just test-gpu <family> runs matching library tests in apxinf-model (with
--features cuda) and apxinf-cuda-new, both in release mode. Integration
targets are selected explicitly with just test-l3 --test <target>; the
known-broken qwen38_gdn_batched_ops target remains excluded from the
library suite. Verify the output includes the intended tests: a filter
matching zero tests is not evidence of GPU correctness.

PRs that touch GPU behavior must attach the
recipe output plus board, arch, and commit, per the PR template. If you
have no suitable device, say "not verified" — do not report "pass".

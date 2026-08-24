# Sampling Subsystem Evaluation

Date: 2026-08-19
Status: Correctness and integration accepted; release performance profiling is
still a separate follow-up

## Evaluation environment

CUDA validation ran on:

- host alias: `Thor`;
- GPU: NVIDIA Thor, compute capability 11.0, 20 SMs;
- CUDA toolkit: 13.0; and
- build overrides:

```bash
export APXINF_CUDA_ARCH=sm_110
export APXINF_CUDA_ARCH_CUTLASS=sm_110a
```

Portable compilation and CPU tests ran on the local arm64 macOS workspace.
The Mac has no CUDA toolkit, so CUDA-linked tests run on Thor rather than the
local host.

## Automated result

The final Thor command was:

```bash
APXINF_CUDA_ARCH=sm_110 \
APXINF_CUDA_ARCH_CUTLASS=sm_110a \
cargo test --workspace --features cuda -- --test-threads=1
```

Result: **161 passed, 0 failed**.

| Suite | Passed | Sampling-relevant coverage |
|---|---:|---|
| `apxinf-core` | 29 | 13 direct sampler/RNG tests |
| `apxinf-cuda` | 62 | 4 direct CUDA tests, including dtype loops |
| `apxinf-loader` | 11 | Loader integrity |
| `apxinf-model` | 44 | LLM and VLA integrity |
| Unified generation integration | 10 | Request/callback/EOS/error behavior |
| `apxinf-py` | 4 | Binding parsing and construction integrity |
| `apxinf-tokenizer` | 1 | Tokenizer integrity |

Local `cargo check --workspace --all-targets`, `git diff --check`, and focused
format checks for the new Rust files also passed.

## CPU/CUDA agreement

The differential CUDA test runs the same prompt history and five consecutive
draws through CPU and CUDA samplers for f32, f16, and bf16 logits. It requires:

- exact selected-token agreement; and
- absolute selected-token log-probability error below `3e-5`.

Additional coverage verifies a 32,768-token multi-block greedy reduction,
lowest-ID ties, final-row selection, stable log-probability with `f32::MAX`
ties, and generated-token history updates.

The normal generator is checked against the CPU Philox/Box-Muller reference:

- f32 maximum CPU/CUDA error below `2e-5`;
- f16 element error at most `0.002`;
- bf16 element error at most `0.02`;
- repeated generation with one key is exact; and
- 100,000 f32 samples have mean magnitude below `0.02` and variance within
  `0.03` of one.

## Real-model integrity

### Text Qwen path

Checkpoint:
`/home/wwxq/Projects/models/nvidia/Cosmos-Reason2-2B`

Prompt: `What is a robot?`, greedy, ten tokens, BF16 CUDA.

The untouched baseline and new implementation both decoded exactly:

```text
A robot is a machine designed to perform tasks autonom
```

After removing the redundant graph-replay synchronization, the same output was
confirmed again. The final debug-build sanity run reported TTFT 175.8 ms, TPOT
21.8 ms/token, and 45.94 generated tokens/s. This is not a controlled release
benchmark and is not used as a performance acceptance number.

### Qwen VLM path

The Cosmos Qwen3-VL-compatible checkpoint and the repository's bundled image
fixture produced the same first ten tokens as the untouched baseline: 10/10
exact agreement. The bundled Hugging Face golden belongs to a different Qwen
checkpoint, so baseline-to-new parity is the relevant regression comparison.

### PI0.5 VLA path

Checkpoint:
`/home/wwxq/Projects/models/pi05_libero_base`

The BF16 auto smoke test verified:

- prepared, direct, and cached output shapes `[50, 32]`;
- exact same-key reproducibility;
- generated-device-latent versus CPU-reference-latent action cosine `1.0`;
- maximum absolute action difference `0.0`; and
- a different sequence key produces a different action.

The f16 latent path used by FP8 is covered directly by CPU/CUDA RNG tests; this
evaluation did not reload the large FP8 PI0.5 model solely to duplicate that
primitive check.

## Structural performance change

The old decode path copied an entire vocabulary row to the host. The new path
copies a 16-byte `TokenSample` result.

| Example | Old BF16 D2H/token | New D2H/token | Reduction |
|---|---:|---:|---:|
| Qwen, vocab 151,669 | 303,338 B | 16 B | about 18,959x |
| Vocab 32,768 | 65,536 B | 16 B | 4,096x |

The new implementation also removes the separate synchronization after decode
graph replay. Sampling is ordered on the same stream, and the compact result
copy becomes the only steady-state host synchronization.

These are code-path and transfer-volume facts. A controlled release benchmark
and Nsight trace are still required before claiming an end-to-end speedup.

## Remaining performance evaluation

Run separate greedy and random benchmarks with:

1. release binaries and fixed GPU clocks/power mode;
2. one fixed model, prompt, output length, and decode-graph configuration;
3. warmup before at least 100 measured decode steps;
4. median, p90, and p99 TPOT;
5. CUDA allocation tracing to confirm zero per-token allocations;
6. Nsight Systems/CUDA tracing to confirm no vocabulary-sized D2H transfer;
7. kernel-level time for prepare, radix sort, scan, and select; and
8. comparisons for greedy, greedy+logprob, top-k, top-p, and combined
   penalties.

The main expected optimization target is the full CUB sort used by random
sampling. Small-top-k requests should eventually use a selection kernel rather
than ordering the complete vocabulary.

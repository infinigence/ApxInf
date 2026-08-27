# Iteration Report 28 - Exact Prompt Lookup and Marlin Prefill Selection

Date: 2026-08-25 | Artifact revision label: `f4793ee6d7782c61a55fb2db95cc52d438b5d473` plus the measured iteration-28 worktree | Run id: `iterate28-definitive`

## Result

Iteration 28 investigated two remaining ideas without weakening the exact greedy contract:

1. restore the vectorized Marlin-to-BF16 inverse path as the production prefill route, after a compact transient-raw reconstruction path lost end-to-end throughput;
2. verify prompt-lookup draft blocks with the ordinary `seq=1` kernels while snapshotting recurrent state, staging all scalar inputs, and synchronizing once per block.

The vectorized exact prefill route remains the environment-free default. Prompt lookup is exact after rollback/replay fixes but slower, so `APXINF_PROMPT_LOOKUP` remains default-off. No hard-coded case IDs, prompts, or answers are used.

Correctness and reliability pass completely. Against iteration 27, base TTFT decreases 3.78-4.29% and TPOT decreases 1.58-2.41%. The requested 1.2x vLLM threshold is still not met: at 1K, ApxInf is 0.431x vLLM prefill and 0.702x vLLM decode.

## Definitive official evaluation

Canonical evaluator: `benchmarks/qwen38_4090/evaluation/run_evaluation.py`, `public_calibration`, public suite plus the 32,640-token context diagnostic. Production service used no ApxInf feature environment variables.

| Cell | TTFT | Prefill | TPOT | Decode | TTFT vs iter27 | TPOT vs iter27 | VRAM | Prefill/vLLM | Decode/vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| text-perf-1024 | 0.8564 s | 1195.7 tok/s | 28.813 ms | 34.71 tok/s | -4.29% | -1.58% | 23906 MiB | 0.431x | 0.702x |
| text-perf-2048 | 1.7320 s | 1182.5 tok/s | 31.117 ms | 32.14 tok/s | -4.22% | -1.66% | 23906 MiB | 0.405x | 0.651x |
| text-perf-4096 | 3.5392 s | 1157.3 tok/s | 35.655 ms | 28.05 tok/s | -4.11% | -1.85% | 23906 MiB | 0.401x | 0.571x |
| text-perf-8192 | 7.2454 s | 1130.6 tok/s | 44.857 ms | 22.29 tok/s | -4.00% | -1.92% | 23906 MiB | 0.400x | 0.457x |
| text-perf-16384 | 15.0747 s | 1086.9 tok/s | 63.046 ms | 15.86 tok/s | -3.78% | -2.41% | 23906 MiB | 0.403x | 0.329x |

Rates are `prompt_tokens / TTFT` and `1 / TPOT`.

- Protocol: pass.
- Public functional correctness: **6/6**.
- Public trajectory: **256/256**; zero token edit distance at 1K and 8K.
- Every base cell produced the complete 128-token budget.
- Request success rate: **1.0**.
- Reliability: no fallback, NaN, unexpected OOM, or XID; service healthy after the context run.
- Provisional score against the local one-GPU vLLM control: **68.0961** leaderboard / **54.4769** automated course points.
- Raw evidence SHA-256: `bd3b76d8b9c937a825359111a61f4ddc241513d64feb58737bbd28edd8db3ed8`.
- Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate28-definitive/` (`submission.json`, `raw.jsonl`, `environment.json`, `score-vllm.json`).

This is a one-repeat public-calibration run. It is not the official one-warmup/five-repeat private leaderboard run.

## Implementation

### Vectorized exact Marlin prefill retained

The production path reconstructs exact row-major BF16 weights directly from the single Marlin representation with `apxinf_marlin_awq_u4_g32_v1_dequant_bf16`, then uses the established 32 MiB output-row cuBLAS schedule. It preserves the existing `(M,N,K)`, leading dimensions, output partitioning, BF16 boundaries, and token trajectories.

The alternative `APXINF_MARLIN_PREFILL_RAW=1` path reconstructs compact raw qweight/scales/zero-points into caller-owned scratch, expands one row tile to BF16, and invokes the same cuBLAS output tile. It avoids materializing the full dense matrix but adds inverse-layout work and another packed-to-BF16 pass for every tile. End-to-end probes did not beat the vectorized route, so it remains opt-in.

### Exact prompt-lookup verifier retained as opt-in

The generic generator searches prior prompt/generated history for the longest matching suffix and proposes at most eight following tokens. Qwen verification then:

1. stages `current_token`, prior proposals, and every device position before model work;
2. snapshots all cumulatively mutated linear-attention convolution and recurrent state into checked scratch;
3. enqueues each proposal through the ordinary `seq=1` embedding, 64-layer, LM-head, and GPU-argmax path to a distinct mapped output slot;
4. synchronizes once after the block;
5. on mismatch, restores recurrent state and replays only the committed ordinary state transitions;
6. relies on position-indexed full-attention KV rows: uncommitted speculative rows remain invisible and are overwritten when reached.

The verifier preserves serial arithmetic; it does not use `seq>1` kernels for draft validation. Interface checks require `verified.len() == consumed_draft`, an exact accepted prefix, and exactly one mismatching committed token on rejection.

## Negative controls

### Prompt lookup

The first one-wait implementation returned the entire verified block while reporting `consumed_draft=accepted_prefix+1` after a mismatch. The service correctly rejected that inconsistent metadata after 10 emitted tokens. Truncating the returned vector to the committed prefix fixed the invariant.

The corrected one-wait path was exact at both trajectory gates:

| Candidate | 1K trajectory | 1K TPOT | 8K trajectory | 8K TPOT | Decision |
|---|---:|---:|---:|---:|---|
| Production default | 128/128 | 28.81 ms official | 128/128 | 44.86 ms official | retained |
| Prompt lookup, one sync/block | 128/128 | 42.95 ms | 128/128 | 124.62 ms | default-off |

Snapshot traffic and mismatch restore/replay cost more than the removed host synchronizations. Earlier prompt-lookup probes also had worse E2E latency (5.322 s at 1K and 14.913 s at 8K) than the same-binary control probes (4.940 s and 13.449 s).

### Compact transient-raw prefill

The compact route remained trajectory-exact but did not improve end-to-end TTFT. Reconstructing compact qweight, scales, and zero-points before the existing dense expansion adds work without changing the cuBLAS stage. `APXINF_MARLIN_PREFILL_RAW=1` is therefore a rollback/diagnostic gate, not a production default.

## Long-context evaluation

The 32,640-token diagnostic passed and generated the full 128-token budget.

- TTFT: **32.3829 s** (**1008.0 prompt tok/s**).
- TPOT: **99.279 ms** (**10.07 decode tok/s**).
- E2E: **44.9914 s**.
- Peak VRAM: **23906 MiB**.
- Output SHA-256: `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.
- Validator: `normalized_prefix`; expected prefix `KEY-EARLY-767211`; pass.
- Service remained healthy afterward.

### Concrete exact question-answer example

Question decoded from the pretokenized prompt tail:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact generated answer (`completion_tokens=128`; 11 tokens encode the key and 117 tokens encode ` context`; no ellipsis):

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

Input SHA-256: `513c5f019411d089403863774fea53784c76c924da0984045bdaeb447ecd4dcc`. Output token IDs begin `[4631,13044,905,8662,12,22,21,22,17,16,16]` and the remaining 117 IDs are all `2193`; this is the exact 128-token output represented above.

## Nsight Systems profile

Profiler: NVIDIA Nsight Systems with CUDA, NVTX, OS runtime, and CUDA graph-node tracing. Workload: one exact 1,024-token prompt with a 128-token decode request. Client-observed under profiling: 1.1806 s TTFT, 29.323 ms TPOT, 4.9046 s E2E, exact 128/128 trajectory.

Artifacts: `/tmp/iter28-nsys.nsys-rep` (4,170,937 bytes) and `/tmp/iter28-nsys.sqlite` (11,702,272 bytes).

The trace contains the full prefill and 62 LM-head/argmax selections rather than all 128 decode selections. Percentages therefore describe the captured GPU interval, not an extrapolated full request.

| Kernel/group | GPU time share | Instances | Total | Mean |
|---|---:|---:|---:|---:|
| Marlin M=1 main tile | **32.1%** | 16620 | 825.703 ms | 49.68 us |
| CUTLASS BF16 128x64 GEMM | **9.9%** | 2366 | 253.706 ms | 107.23 us |
| tiled prefill delta recurrence | **9.6%** | 96 | 248.464 ms | 2.588 ms |
| exact raw linear-QKV W4 | **9.1%** | 2945 | 233.913 ms | 79.43 us |
| remaining raw tile-alt W4 | **6.5%** | 2945 | 168.554 ms | 57.23 us |
| vectorized Marlin inverse dequant | **6.4%** | 606 | 165.493 ms | 273.09 us |
| dense BF16 LM-head GEMV | **6.4%** | 62 | 164.445 ms | 2.652 ms |
| full-attention flash kernel | **5.7%** | 981 | 146.510 ms | 149.35 us |
| CUTLASS BF16 128x128 GEMM | **3.9%** | 640 | 101.073 ms | 157.93 us |
| RMSNorm | **2.7%** | 8169 | 70.554 ms | 8.64 us |
| packed decode delta recurrence | **1.9%** | 2945 | 49.344 ms | 16.76 us |
| conv + SiLU | **0.8%** | 96 | 21.879 ms | 227.91 us |

### CUDA API and memory traffic

Process-lifetime CUDA API totals include model load and initialization:

- `cudaMemcpy`: 1,896 calls, 3.033 s host API time;
- `cudaMalloc`: 1,927 calls, 1.246 s;
- `cudaLaunchKernel`: 22,378 calls, 447.1 ms host API time;
- `cudaGraphLaunch`: 981 calls, 90.31 ms;
- H2D copies: 21,013.278 MB across 1,896 operations, 2.827 s GPU copy time;
- memsets: 2,843.367 MB across 21,869 operations.

The roughly 21 GB H2D total is model upload, not per-token decode traffic. Token selection uses mapped host output after GPU argmax, so no per-token vocabulary D2H transfer is required.

### Memory-bandwidth utilization and bottlenecks

Privileged GPU hardware counters are unavailable in this environment, so no DRAM/L2/occupancy/stall counter is fabricated. Two bounded indicators are available:

1. The evaluator sampler measured memory-controller utilization at **14.62% mean / 77% max** for the 1K official cell and **8.04% mean / 45% max** for the 32,640-token context case. Sampling includes host gaps and does not equal achieved GB/s.
2. The contract's frozen minimum-weight proxy is 21,017,689,808 bytes/token. Dividing it by the 1K TPOT of 28.813 ms yields a lower-bound sweep rate of **729.4 GB/s**, or **72.4%** of the RTX 4090's 1,008 GB/s theoretical bandwidth. This is a proxy: scale/zero-point, activation, recurrent-state, KV-cache, and repeated traffic make actual bytes larger.

Primary bottlenecks:

1. **Quantized projection sweep.** Marlin plus the exact raw QKV and tile-alt families consume 47.7% of captured GPU time. The exact raw linear-QKV role remains outside Marlin because enabling Marlin there changes the 8K trajectory.
2. **Exact prefill work.** CUTLASS GEMMs, tiled delta recurrence, and vectorized Marlin inverse dequant are all first-order costs. The compact-raw experiment added conversion work instead of removing the dense/cuBLAS stage.
3. **Dense LM head.** Approximately 2.652 ms per selected token; an exact custom replacement remains unproven, and a persistent transposed copy exceeds the memory budget.
4. **Long-context attention.** The full-attention kernel is about 149 us per captured invocation and runs in 16 full-attention layers per token; context growth explains TPOT increasing from 28.81 ms at 1K to 99.28 ms at 32.64K.
5. **Memory headroom.** Peak use is 23,906 MiB. Additional persistent layouts or large workspaces are unsafe on a 24 GiB card.

## vLLM threshold verdict

Local control: `benchmarks/qwen38_4090/evaluation/runs/vllm-one-gpu-control/`, vLLM 0.27.1, one RTX 4090. This public control passes 6/6 functional cases but has a different public token trajectory (144/256), so the ratio is a local diagnostic rather than an official same-round private leaderboard claim.

At 1K:

- vLLM prefill: **2774.2 tok/s**; required 1.2x rate: **3329.1 tok/s**;
- ApxInf prefill: **1195.7 tok/s**, or **0.431x vLLM**; gap to requirement: **2133.3 tok/s**;
- vLLM decode: **49.45 tok/s**; required 1.2x rate: **59.35 tok/s**;
- ApxInf decode: **34.71 tok/s**, or **0.702x vLLM**; gap to requirement: **24.64 tok/s**.

The 1.2x requirement is **not met for either prefill or decode**. No pass claim is made.

## Verification and tooling limitations

- Release CUDA build passed: `cargo build --release --features cuda -p apxinf --bin apxinf -j 40`.
- Canonical direct evaluator passed protocol, 6/6 functional cases, 256/256 trajectories, every base performance cell, and the extended 32,640-token context case.
- `cargo check --workspace --locked -j 2` passed. The first unbounded `test.py check` attempt reached `cargo check` but hit a transient archive mmap error (`memory map must have a non-zero length`); bounded retry passed.
- The documented `test.py run` wrapper currently invokes `run_evaluation.py` without the now-required `--trajectory-reference` and exits with `ValueError: provide --trajectory-reference, or capture one from the vLLM control`. Evaluator files are contract-protected and were not modified. The successful direct canonical run supplied `/tmp/iter25-trajectory-reference.json` explicitly.
- Hidden correctness, official one-warmup/five-repeat medians and CV, multi-request bonus, 65K+ context, and multimodal evaluation were not locally available.
- The artifact revision field records the repository revision supplied to the evaluator; iteration-28 worktree changes require a complete committed SHA before formal submission.

## Reproduction

```bash
RUSTFLAGS='-C link-arg=-fuse-ld=gold' APXINF_CUDA_ARCH=sm_89 \
  cargo build --release --features cuda -p apxinf --bin apxinf -j 40

CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve \
  --model ../model/qwen --host 127.0.0.1 --port 8035

python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8035 \
  --implementation-name apxinf-iter28-definitive \
  --implementation-revision f4793ee6d7782c61a55fb2db95cc52d438b5d473 \
  --backend apxinf --profile public_calibration \
  --trajectory-reference /tmp/iter25-trajectory-reference.json \
  --run-context --run-id iterate28-definitive \
  --output-dir benchmarks/qwen38_4090/evaluation/runs --timeout 1800

python3 benchmarks/qwen38_4090/evaluation/score_submission.py \
  --submission benchmarks/qwen38_4090/evaluation/runs/iterate28-definitive/submission.json \
  --control-submission benchmarks/qwen38_4090/evaluation/runs/vllm-one-gpu-control/submission.json \
  --profile public_calibration \
  --output benchmarks/qwen38_4090/evaluation/runs/iterate28-definitive/score-vllm.json
```

Rollback experiments by leaving `APXINF_PROMPT_LOOKUP` and `APXINF_MARLIN_PREFILL_RAW` unset; those are the production defaults measured above.

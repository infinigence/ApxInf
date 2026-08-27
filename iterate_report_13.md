# Iteration Report 13 — ApxInf Qwen3.8-27B on RTX 4090

Date: 2026-08-24  
Implementation revision: `f4793ee6d7782c61a55fb2db95cc52d438b5d473` plus the measured decode-only 32-column delta recurrence specialization in `crates/apxinf-model/src/qwen35/cuda.rs` (worktree build).  
Run id: `iterate13-tile32`

## Problem and decision

Iteration 7 established the fixed-offset control layout. This iteration changes the delta recurrence state tile from 128 to 64 value columns per block. The shared-memory footprint drops from about 65 KiB to 33 KiB, allowing more resident blocks and exposing 96 independent blocks for the 48 value heads. The q/k broadcast now uses a strided two-elements-per-lane load while preserving values and accumulation order. All fixed-size recurrence loops are explicitly unrolled to expose the inner product and state-update instruction schedule to NVCC.

This does **not** claim the requested 1.2× vLLM threshold: the local repository contains no official vLLM control measurements, and the contract makes the same-round platform control authoritative. The observed ApxInf rates are recorded below; the threshold is `ApxInf >= 1.2 × vLLM` for both phases and cannot be proven locally without that control.

## Official evaluation

Command used for the documented evaluator, extended with the context dataset:

```bash
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8002 \
  --implementation-name apxinf-student \
  --implementation-revision worktree-iter13 --backend apxinf \
  --profile public_calibration \
  --trajectory-reference target/iterate2-trajectory-regression.json \
  --run-context --warmups 0 --repeats 1 --timeout 1800 \
  --run-id iterate6-control \
  --output-dir benchmarks/qwen38_4090/evaluation/runs
```

The README wrapper `test.py run` was exercised in iteration 3 and stopped at the frozen evaluator's required trajectory-reference check. For this iteration, the equivalent documented lower-level command above supplied the available local regression reference without changing the evaluator or contract.

Artifact evidence:

- `benchmarks/qwen38_4090/evaluation/runs/iterate6-control/submission.json`
- `benchmarks/qwen38_4090/evaluation/runs/iterate6-control/raw.jsonl`
- `benchmarks/qwen38_4090/evaluation/runs/iterate6-control/environment.json`
- contract SHA-256: `520349b1279c3bf999a6848b296c23d20cdaeab7420934e9196c90018bac7433`
- public manifest SHA-256: `1ec4f360e8dce8cb366251d9b92f8f91a393e5534bb93277a955f8b9e3e5e1e4`
- context manifest SHA-256: `d8b457ad6b932c0a0503cd5e978d5a16933b987b9c088c95358d9f1d4314a881`
- raw JSONL SHA-256: `23006f881e201c164a33a10e89d456ac58ae0589165ba82378568e25638dc758`

### Correctness and reliability

| Gate | Result |
|---|---:|
| Protocol | pass |
| Public functional cases | 6/6 |
| Public trajectory | 256/256 (local iteration-2 regression reference; not an official platform score) |
| Request success rate | 1.0 |
| No fallback / NaN / unexpected OOM / Xid | all true |
| Health after failure | true |
| Service declaration | `max_model_len=32768`, `parallel_requests=1`, `fallback_active=false` |

### Base performance cells

The run is `public_calibration` (one measured repeat, no warm-up), so CV is structurally 0 and these are diagnostic rather than the contract's five-repeat leaderboard medians.

| Cell | Prompt | TTFT | Prefill rate | TPOT | Decode rate | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| text-perf-1024 | 1,024 | 1.385 s | 739.61 tok/s | 55.10 ms | 18.15 tok/s | 24,012 MiB |
| text-perf-2048 | 2,048 | 2.780 s | 736.65 tok/s | 57.46 ms | 17.40 tok/s | 24,012 MiB |
| text-perf-4096 | 4,096 | 5.610 s | 730.10 tok/s | 62.19 ms | 16.08 tok/s | 24,012 MiB |
| text-perf-8192 | 8,192 | 11.448 s | 715.56 tok/s | 71.63 ms | 13.96 tok/s | 24,012 MiB |
| text-perf-16384 | 16,384 | 23.809 s | 688.15 tok/s | 90.48 ms | 11.05 tok/s | 24,012 MiB |

The reported rate calculations are `prompt_tokens / TTFT` and `1 / TPOT`; official scoring remains client-observed TTFT/TPOT in seconds. All five output budgets completed at 128 tokens.

Compared with iteration 8, loop unrolling preserved correctness and changed latency by less than 0.1% in this single-repeat run; the recurrence remains limited by weight traffic and serial state updates rather than loop-control overhead.

## Long-context evaluation

Generated dataset: `context-iter3`, six exact task categories at 32,640 prompt tokens. The evaluator's staircase deliberately probes only the early retrieval case above the 32K bonus start, then verifies health and a small recovery request. Result:

| Prompt | Cases attempted | Passing | Output budget | TTFT | TPOT | E2E |
|---:|---:|---:|---:|---:|---:|---:|
| 32,640 | 1 required probe | 1/1 | 128 | 51.149 s | 127.92 ms | 67.394 s |

`submission.json` records `max_verified_prompt_tokens=32640`, `pass_rate_at_max_context=1.0`, `verified_output_tokens=128`, `verified_cases_at_max_context=1`, and `service_healthy_after_failure=true`. A separate generated boundary dataset contains all six categories at 32,640 and 32,768 (`target/context-iter3-staircase`, manifest SHA-256 `e08cf458719555611664f2c75e9430939c6bd6a8db86b4098f5f830f4f636605`) but was not substituted into the official result after the completed run; it is retained as a reproducible next probe, not claimed as scored evidence.

### Concrete long-context question–answer example

This is the exact question suffix from `context-32640-retrieval-early` (the preceding input is the frozen 32,640-token pinned corpus prompt):

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact evaluator answer text, including the requested continuation (128 generated tokens; shown verbatim):

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

The answer row reports `functional_pass=true`, `prompt_tokens=32640`, `completion_tokens=128`, `ttft_s=51.14885050803423`, `tpot_s=0.12791563236103284`, and `e2e_s=67.39422058314085`.

## Profiling and bottleneck analysis

### Nsight Systems capture

`nvprof` was attempted as requested. CUDA 11.5 reports that profiling is unsupported for compute capability 7.5 and higher, which includes this RTX 4090 (`sm_89`), so it produced no kernel metrics. The supported replacement, NVIDIA Nsight Systems 2025.1.1, captured:

- `target/iterate13-profile.nsys-rep` (6,073,713 bytes)
- `target/iterate13-profile.sqlite` (Nsight export)

The fixed-duration capture used:

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
  --trace=cuda,cublas,nvtx,osrt --sample=none \
  --cuda-memory-usage=true --stats=true --duration=15 \
  --output=target/iterate13-profile \
  ./target/release/apxinf serve --model ../model/qwen --port 8002
```

The profile contains CUDA API and memory activity but no CUDA kernel activity on this driver/CUPTI combination; Nsight reports `does not contain CUDA kernel data`. The runtime event instrumentation below supplies kernel-level timings for the same binary and request path. This limitation is explicit rather than replacing missing data with estimates.

### Kernel breakdown (runtime CUDA event instrumentation)

With `APXINF_KERNEL_PROF=1 APXINF_GEMM_PROF=1`, one 1,024-token prompt and 8-token decode request produced these repeated steady-state measurements:

| Kernel/path | Observed duration | Role |
|---|---:|---|
| Paired `17408+17408 x 5120` W4A16 TC GEMM | 0.263–0.284 ms | MLP gate/up projection; largest individual decode projection |
| Single `5120 x 17408` TC GEMM | 0.249–0.261 ms | MLP down projection |
| Paired `10240+6144 x 5120` TC GEMM | 0.099–0.110 ms | Linear-attention qkv/z projections |
| Single `5120 x 6144` TC GEMM | 0.081–0.084 ms | Full-attention output projection |
| `delta_step` | ~0.692 ms | Linear-attention recurrence; non-GEMM decode bottleneck |
| `conv_silu` | ~0.063 ms | Linear-attention convolution |
| `flash_prefill` decode path | ~0.153–0.176 ms | Full-attention decode attention |

Real-request Nsight capture shows dequantization 21.8%, cuBLAS prefill GEMM 18.5%, single W4 TC GEMMs 16.1%, paired W4 TC GEMMs 16.0%, and the 64-column prefill delta kernel 11.3%. The decode-only tile32 kernel is 0.4% aggregate across 1,458 launches with 12.75 µs average.

### Memory bandwidth and bottlenecks

The evaluator hardware sampler observed `memory_controller_util_max_pct` 76–77% and `memory_used_peak_mib=24012` during the run. Nsight Systems measured the captured startup/short request's host-to-device traffic at Kernel rows were unavailable in the startup-only export because the profiler terminated during post-capture processing; the persisted `target/iterate9-profile.nsys-rep` is retained. Real-request Nsight capture contains 1,815 H2D copies totalling 21,013.298 MB and 11.706 s aggregate copy time; startup/model loading dominates these transfers. It also contains 6,235 memsets totalling 3,141.969 MB. These copies are dominated by initialization and host transfers, not the steady-state decode loop.

The dominant performance constraints are:

1. **Weight bandwidth / projection launch count.** W4A16 weights are streamed for many small single-row projections. Paired launches remove some launch overhead, but each 0.25–0.28 ms MLP pair still has a large memory footprint and does not approach a full large-GEMM arithmetic regime.
2. **Linear-attention recurrence.** `delta_step` is ~0.692 ms per layer, greater than the attention microkernels and a significant serial component; it limits decode even after projection pairing.
3. **Long-context prefill.** TTFT grows from 1.403 s at 1K to 51.750 s at 32.64K. The fixed attention scratch and 128-row chunks keep memory safe but increase chunk-boundary and long-prefix work. The 24,012 MiB peak leaves only about 552 MiB below nominal 24 GiB, so larger workspaces or multiple requests are unsafe.
4. **Profiler overhead/coverage.** Nsight Systems adds substantial synchronization/OS wait visibility and its CUDA activity export lacks kernel rows in this environment. The official evaluator remains the source of throughput numbers; profiled captures are diagnostic only.

The measured memory-controller utilization is below the 1,008 GB/s theoretical HBM bandwidth ceiling and is not a direct GB/s measurement. It indicates the workload is not saturating the controller continuously; small-grid launch overhead, serial recurrence, and irregular W4A16 scale/zero-point traffic are consistent with the observed gap.

## vLLM threshold status

The contract's authoritative performance reference is the best valid same-round vLLM median per cell. A provided vLLM endpoint at `http://0.0.0.0:8000` was reachable and advertised model `../qwen`, but direct evaluation integration returned no valid official control artifact; direct earlier probes measured approximately 76.8 tok/s decode at 1K and 76.0 tok/s at 8K. Therefore:

- ApxInf prefill: 679–729 tok/s across 1K–16K base cells; 32.64K TTFT corresponds to 631.0 prompt tok/s (`32640 / 51.14885050803423`).
- ApxInf decode: 11.05–18.15 tok/s across base cells; 32.64K context decode is 7.81 tok/s (`1 / 0.12791563236103284`).
- Direct vLLM decode is approximately 76.8 tok/s at 1K; ApxInf reaches 18.15 tok/s, about 0.24×. The requested 40 tok/s and 1.2× vLLM thresholds are not met.

## Negative controls and known limitations

- The decode-only tile32 source built successfully; the official run preserved 256/256 trajectory correctness.
- `nvprof` negative control: correctly refused kernel profiling on RTX 4090 SM89; Nsight Systems was used instead.
- The first `MAX_SEQ_LEN=32768` build with `CHUNK=512` failed model load with CUDA OOM. Reducing `CHUNK` to 128 fixed startup and is the configuration used for both official long-context runs.
- Hidden correctness, official 1-warmup + 5-repeat medians/CV, vLLM control comparisons, C4/C8, 65K+ contexts, and multimodal capability were not locally available.
- The official run is tied to the `worktree-iter13` build. A formal run must use the complete commit SHA. The provided vLLM endpoint at `http://0.0.0.0:8000` was reachable, but its vLLM evaluator adapter returned no successful `text-perf-1024` row, so an authoritative 1.2x comparison remains unavailable.

## Reproduction

```bash
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CUDA_PATH=/usr/local/cuda
export APXINF_CUDA_ARCH=sm_89
cargo check --workspace --locked
cargo build --release --features cuda -p apxinf
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve --model ../model/qwen --host 127.0.0.1 --port 8002
```

Official base plus long-context run:

```bash
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --run-context --warmups 0 --repeats 1 --timeout 1800 \
  --run-id iterate6-control --output-dir benchmarks/qwen38_4090/evaluation/runs
```

Profile one request with Nsight Systems:

```bash
nsys profile --trace=cuda,cublas,nvtx,osrt --sample=none \
  --cuda-memory-usage=true --stats=true --duration=15 \
  --output=target/iterate13-profile \
  ./target/release/apxinf serve --model ../model/qwen --port 8002
```

For named runtime timings:

```bash
APXINF_KERNEL_PROF=1 APXINF_GEMM_PROF=1 \
  ./target/release/apxinf serve --model ../model/qwen --port 8002
```

No model weights, credentials, private evaluation data, or fabricated vLLM results are included.

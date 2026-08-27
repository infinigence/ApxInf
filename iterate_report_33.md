# Iteration Report 33 - Direct M512 Prefill and Position-Safe Decode Graphs

Date: 2026-08-26  
Base repository revision: `f4793ee6d7782c61a55fb2db95cc52d438b5d473` plus the measured iteration-33 worktree  
Run id: `iterate33-candidate`

## Verdict

Iteration 33 is correct and materially improves prefill over iteration 32. It does **not** meet the requested 1.2x vLLM throughput threshold.

- Public functional correctness: **6/6**.
- Public frozen trajectory: **256/256 tokens**.
- Protocol and reliability gates: pass.
- Long context: **32,640 prompt tokens + 128 output tokens**, validator pass.
- Peak VRAM: **22,136 MiB** on one RTX 4090.
- Best measured ApxInf/vLLM throughput ratios: **0.577x prefill** and **0.833x decode**, both at 1K. The required ratio is 1.2x.

The production-default improvements measured in this iteration are:

1. direct Marlin execution for exact 512-row prefill chunks;
2. GQA6 shared-KV decode attention;
3. device-position, full-attention-layer CUDA graphs;
4. retained pair-prefetch raw-W4 scheduling and linear-segment graphs.

## Implementation and candidate decisions

### Promoted

| Change | Source path | Decision evidence |
|---|---|---|
| Direct M=512 Marlin prefill | `crates/apxinf-model/src/qwen35/cuda.rs`, `crates/apxinf-cuda/src/kernels/quantization.rs`, Marlin adapter/kernel sources | Exact at the frozen 1K/8K gates; lowers TTFT across the official matrix. Default for Marlin layout, `seq == 512`, and zero physical N offset. Roll back with `APXINF_DIRECT_MARLIN_PREFILL=0`. |
| GQA6 shared-KV decode attention | `crates/apxinf-cuda/kernels/custom/qwen35.cuh`, CUDA adapter/FFI, `crates/apxinf-model/src/qwen35/cuda.rs` | Exact frozen trajectory and public functional suite. Default `APXINF_GQA_GROUP=6`; `0` restores the prior dispatch. |
| Position-safe full-layer decode graphs | `crates/apxinf-model/src/qwen35/cuda.rs`, CUDA device-address position ABI | Exact through 32,640-token context. Position consumers dereference stable device data instead of capture-time host values. Roll back with `APXINF_FULL_LAYER_GRAPH=0`. |

### Implemented but not promoted

| Candidate | Geometry / contract | Result |
|---|---|---|
| Four-warp prefill GDN recurrence | One 128-thread CTA per value head; 128x128 FP32 shared state; exact per-column K-order | Exact in the candidate matrix but neutral/slower end to end. Remains opt-in with `APXINF_PREFILL_GDN_WARPS=4`. |
| Two-warp prefill GDN recurrence | Two 64-thread CTAs per value head, disjoint 64-column tiles | Initial gate failed because only half of Q/K shared memory was loaded; the load loop and missing token loop were repaired. It was not promoted without a complete post-repair canonical matrix. Opt-in with `APXINF_PREFILL_GDN_WARPS=2`. |
| QKV raw-W4 schedules 0..7 | Exact K-ordered tensor-core variants | No schedule beat the retained default across both 1K and 8K. Remain diagnostic through `APXINF_QKV_SCHED`. |
| GDN projection overlap / merged launches | Side stream and merged projection candidates | Exact variants were neutral or lost to synchronization/launch cost; production remains the proven graph/pair path. |
| Projection-prologue and residual-write fusion candidates | Attention/GDN norm and residual boundaries | No exact measured win large enough for promotion; existing BF16 boundaries remain unchanged. |

The standalone CUDA build initially failed in `qwen35_prefill_delta_step_2w_kernel`: an edit had removed the `for (s...)` token loop and left `qk_base`/`s` undefined. Restoring the complete loop fixed the first NVCC error; all later adapter errors were parser cascades. The release CUDA build then completed.

## Canonical official evaluation

Evaluator: `benchmarks/qwen38_4090/evaluation/run_evaluation.py`  
Profile: `public_calibration`  
Dataset: `benchmarks/qwen38_4090/evaluation/.cache/public`  
Context dataset: `benchmarks/qwen38_4090/evaluation/.cache/context-iter3`  
Warmups / measured repeats: 0 / 1, as required by the public-calibration profile  
Service optimization overrides: none  
Model: `../model/qwen`, revision `63768c10df38c0395e12ef49edac1bd539eaeeea`

| Cell | TTFT | Prefill throughput | TPOT | Decode throughput | Peak VRAM |
|---|---:|---:|---:|---:|---:|
| text-perf-1024 | 0.6401 s | **1,599.7 tok/s** | 24.270 ms | **41.20 tok/s** | 22,136 MiB |
| text-perf-2048 | 1.3020 s | **1,573.0 tok/s** | 25.824 ms | **38.72 tok/s** | 22,136 MiB |
| text-perf-4096 | 2.6790 s | **1,528.9 tok/s** | 28.926 ms | **34.57 tok/s** | 22,136 MiB |
| text-perf-8192 | 5.5254 s | **1,482.6 tok/s** | 35.053 ms | **28.53 tok/s** | 22,136 MiB |
| text-perf-16384 | 11.5981 s | **1,412.6 tok/s** | 47.381 ms | **21.11 tok/s** | 22,136 MiB |

Correctness and reliability from `submission.json`:

- protocol pass: true;
- public functional cases: **6/6**;
- public trajectory: **256/256**, zero edit distance at 1K and 8K;
- request success rate: **1.0**;
- no fallback, NaN, unexpected OOM, or XID;
- service healthy after invalid, capacity, context, and recovery requests;
- all five performance cells completed the exact 128-token output budget.

## Long-context evaluation and exact example

The context staircase verified a 32,640-token prompt with 128 generated tokens.

| Metric | Result |
|---|---:|
| Prompt tokens | 32,640 |
| Output tokens | 128 |
| TTFT | 25.5322 s |
| Prefill throughput | 1,278.4 tok/s |
| TPOT | 72.023 ms/token |
| Decode throughput | 13.88 tok/s |
| End to end | 34.6791 s |
| Validator | pass |
| Output SHA-256 | `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b` |

### Concrete exact question-answer example

Case: `context-32640-retrieval-early`.

The literal question at the end of the 32,640-token input was:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact decoded answer returned by ApxInf:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

This was 128 output tokens and passed the case's normalized-prefix validator.

## Iteration 32 comparison

| Cell | Iteration 33 / iteration 32 prefill | Iteration 33 / iteration 32 decode |
|---|---:|---:|
| 1K | **1.393x** | 1.007x |
| 2K | **1.387x** | 1.010x |
| 4K | **1.377x** | 1.015x |
| 8K | **1.366x** | 1.024x |
| 16K | **1.351x** | 1.032x |

Direct M512 routing produces the large prefill gain. Decode improves only 0.7-3.2%; graph and GQA work reduce overhead but do not change the dominant quantized projection cost.

## vLLM comparison and 1.2x gate

Control: `benchmarks/qwen38_4090/evaluation/runs/vllm-one-gpu-control/submission.json`.

| Cell | ApxInf/vLLM prefill | ApxInf/vLLM decode | Required vLLM-relative ratio | Result |
|---|---:|---:|---:|---|
| 1K | 0.577x | 0.833x | 1.200x | fail |
| 2K | 0.539x | 0.784x | 1.200x | fail |
| 4K | 0.530x | 0.704x | 1.200x | fail |
| 8K | 0.524x | 0.585x | 1.200x | fail |
| 16K | 0.524x | 0.438x | 1.200x | fail |

At 1K, the minimum requested target is 3,329.1 tok/s prefill and 59.35 tok/s decode. Iteration 33 reaches 1,599.7 tok/s and 41.20 tok/s. The user-specified 1.2x threshold is therefore **not achieved**.

## Nsight Systems profiling

Profiler: NVIDIA Nsight Systems 2025.1.1. `nvprof` 11.5 is installed but explicitly refuses profiling on compute capability 8.9, so Nsight Systems was used.

### Capture design

1. **Decode graph-node capture:** frozen `text-perf-8192`, 128 output tokens, `--cuda-graph-trace=node:host-only`. This exposes child kernels inside production CUDA graphs.
2. **Prefill capture:** frozen `text-perf-16384` input with one output token, CUDA/NVTX/OSRT tracing.

Both captures include process startup. Therefore the 19.2 GB host-to-device copies in each whole-process trace are model-weight loading and are **not** steady-state request traffic. Kernel tables and request-window hardware samples are the inference evidence.

### Decode capture kernel breakdown

The table is the capture-wide GPU kernel sum for one 8K prefill plus 128-token decode request. NVTX prefixes identify module ownership.

| Kernel / family | Instances | GPU time | Kernel-time share |
|---|---:|---:|---:|
| Prefill delta recurrence | 768 | 2,029.0 ms | 30.8% |
| `Qwen/MLP` Marlin | 3,072 | 1,729.0 ms | 26.3% |
| CUTLASS `Kernel2` family | 6,144 | 472.6 ms | 7.2% |
| GQA6 flash partial | 511 | 397.0 ms | 6.0% |
| `Qwen/GDN` Marlin | 6,100 | 307.7 ms | 4.7% |
| Other Marlin | 4,328 | 285.5 ms | 4.3% |
| Attention SGEMM 128x128 | 3,456 | 278.3 ms | 4.2% |
| Conv/SiLU | 768 | 178.0 ms | 2.7% |
| Attention Marlin | 1,024 | 175.2 ms | 2.7% |
| Pair-prefetch raw W4 | 1,533 | 163.1 ms | 2.5% |
| Packed GDN recurrence | 1,533 | 28.0 ms | 0.4% |
| LM-head Marlin | 32 | 24.9 ms | 0.4% |

The decode capture issued 1,021 CUDA graph replays. Graph launch API time was only 38.8 ms total; graph replay is not the dominant remaining cost. Quantized projections dominate. GQA6 partial attention is still material at long context: 0.777 ms average per invocation in this 8K capture.

### Prefill capture kernel breakdown

| Kernel / family | Instances | GPU time | Kernel-time share |
|---|---:|---:|---:|
| Marlin M512 main family | 5,009 | 2,351.3 ms | 37.0% |
| Prefill delta recurrence | 888 | 2,343.7 ms | 36.8% |
| CUTLASS BF16 128x64 family | 5,328 | 536.6 ms | 8.4% |
| SGEMM 128x128 | 4,404 | 369.8 ms | 5.8% |
| Conv/SiLU | 888 | 205.1 ms | 3.2% |
| W4 dequant rows | 5,328 | 94.1 ms | 1.5% |
| SGEMM 128x64 | 1,536 | 93.4 ms | 1.5% |
| Attention softmax rows | 7,093 | 55.2 ms | 0.9% |

Marlin plus the serial exact delta recurrence consume **73.8%** of prefill kernel time. Direct M512 avoids the old dense reconstruction path, but Marlin remains projection-bound and the recurrence remains token-serial by mathematical dependency.

CUDA API summary for the whole prefill process recorded 44,672 `cudaLaunchKernel` calls (2.508 s host API time) and 14,215 `cudaEventRecord` calls (1.584 s). These totals include startup and profiling overhead, so they are diagnostic rather than client latency; they still show that launch/event density is the next secondary target after projection and recurrence kernels.

### Memory bandwidth utilization

The official evaluator's 200 ms hardware sampler observed:

| Request | Memory-controller mean | Memory-controller max | GPU-util mean | GPU-util max |
|---|---:|---:|---:|---:|
| 1K / 128 | 13.83% | 81% | 25.00% | 100% |
| 8K / 128 | 9.07% | 68% | 25.00% | 100% |
| 16K / 128 | 6.18% | 57% | 25.00% | 100% |
| 32,640 / 128 | 5.59% | 44% | 24.91% | 100% |

The low means are whole-request, coarse samples and include CPU/service gaps; the high peaks show bursts that approach the memory system during projection. Nsight Systems does not collect SM89 DRAM-throughput counters. No sustained-bandwidth percentage is claimed from its memcpy table. A future Nsight Compute replay of selected Marlin and recurrence invocations is required for per-kernel DRAM throughput, cache hit rate, and achieved occupancy.

### Bottleneck decision

1. **Prefill:** direct Marlin projection and exact recurrent GDN are co-dominant; together 73.8% of kernel time.
2. **Decode:** W4/Marlin projection families dominate; GQA attention grows with context but is secondary at 8K.
3. **Launch overhead:** CUDA graphs have already reduced graph replay overhead; more wrapper-level fusion alone cannot close a 2.1-2.3x prefill gap or a 1.4-2.7x decode gap to the requested target.
4. **Required next architecture:** a faster exact W4 core that consumes the resident packed layout without repeated conversion/staging, plus a recurrence decomposition that preserves token order while increasing value-state parallelism.

## Artifacts and hashes

Official evaluation:

- `benchmarks/qwen38_4090/evaluation/runs/iterate33-candidate/submission.json` — `fbaedab08dd4f2e1fe28a616d6b0d229863056cbb6fc2c0bbd2278f6fd7134bd`
- `benchmarks/qwen38_4090/evaluation/runs/iterate33-candidate/raw.jsonl` — `abb0bc80da5cba57e9ed99dd758413339c500c916c18be22011282399f8f617f`
- `benchmarks/qwen38_4090/evaluation/runs/iterate33-candidate/environment.json` — `591661e399984ececefc1614424a16da06a2121db04f678470cbb984599d52b2`
- trajectory reference — `904fa6f54dec58d9638a2b067c78b145a48c63a4b838fba26e473d866576b793`

Profiler artifacts:

- `/tmp/iterate33-decode-graph.nsys-rep` — `8c83d7af29f1f7788b7711a644fea9b0bc265cfdd5f65efd1fd7368ce564e79f`
- `/tmp/iterate33-decode-graph.sqlite` — `aae9e98e27bfab92413345a58dc273e61ed8ecde4ae17245c83c966a2e695640`
- `/tmp/iterate33-prefill.nsys-rep` — `25cbe61d98cbd35f82888685895513067d07025293e2ef1622c17f08b7999b1d`
- `/tmp/iterate33-prefill.sqlite` — `743ddb940ec647f9b97b78275bb663596889231cae623a5e8114881dcc676c1e`
- `/tmp/iterate33-prefill-kernels_cuda_gpu_kern_sum.csv` — `09df6a305a67149531dfac6309e04b345cc6151b91d6e952b6eac1881b5a4710`

`/tmp` artifacts are workstation-local and must be copied to durable storage before the workstation is stopped if long-term retention is required.

## Reproduction

```bash
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:/usr/bin:/bin
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export APXINF_CUDA_ARCH=sm_89
export RUSTFLAGS='-C link-arg=-fuse-ld=gold'

cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 ./target/release/apxinf serve \
  --model ../model/qwen --host 127.0.0.1 --port 8300

python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8300 \
  --implementation-name apxinf-iteration33 \
  --implementation-revision iterate33-working --backend apxinf \
  --profile public_calibration \
  --trajectory-reference /tmp/iter25-trajectory-reference.json \
  --run-context --timeout 2400 --run-id iterate33-candidate \
  --output-dir benchmarks/qwen38_4090/evaluation/runs
```

The release build and canonical evaluator completed. `git diff --check` passed. No service or profiler process remained live after capture.
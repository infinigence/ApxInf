# Iteration Report 14 — ApxInf Qwen3.8-27B on RTX 4090

Date: 2026-08-24  
Implementation revision: worktree build after iteration 13  
Run id: `iterate14-graph`

## Problem and decision

Iteration 13 left decode dominated by hundreds of single-row projection launches. This iteration tested guarded CUDA Graph replay for the complete one-token Qwen decode body. Position-sensitive RoPE, K-cache append, V-cache append, and fused attention now read a fixed device-resident position value; IDs and position share one fixed-offset control upload. The graph owns embedding, all 64 layers, final RMSNorm, and `lm_head`; token upload remains before replay and mapped GPU argmax remains after replay.

The graph executable is dropped before every referenced allocation. Capture/profiling failures do not retry after state mutation, graph replay errors propagate, and the shared generation loop no longer falls back to eager `forward` after a GPU decode error. Graph capture is disabled under the existing layer/kernel/GEMM/trace instrumentation modes because those paths synchronize, create events, or copy buffers to the host.

The experiment is correct but not retained as a performance win: base TPOT changed by less than 1% versus iteration 13. Graph replay removes host submission overhead, but the GPU still spends about 54 ms per token executing memory-bound W4 projections and attention.

This report does **not** claim the requested 1.2× vLLM threshold. The contract requires a valid same-round platform vLLM control. The locally observed ApxInf decode is also far below the previously measured direct vLLM rate.

## Official evaluation

Command used for the documented evaluator, extended with the generated long-context dataset:

```bash
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8002 \
  --implementation-name apxinf-student \
  --implementation-revision worktree-iter14 --backend apxinf \
  --profile public_calibration \
  --trajectory-reference target/iterate2-trajectory-regression.json \
  --run-context --warmups 0 --repeats 1 --timeout 1800 \
  --run-id iterate14-graph \
  --output-dir benchmarks/qwen38_4090/evaluation/runs
```

Artifact evidence:

- `benchmarks/qwen38_4090/evaluation/runs/iterate14-graph/submission.json`
- `benchmarks/qwen38_4090/evaluation/runs/iterate14-graph/raw.jsonl`
- `benchmarks/qwen38_4090/evaluation/runs/iterate14-graph/environment.json`
- contract SHA-256: `520349b1279c3bf999a6848b296c23d20cdaeab7420934e9196c90018bac7433`
- public manifest SHA-256: `1ec4f360e8dce8cb366251d9b92f8f91a393e5534bb93277a955f8b9e3e5e1e4`
- context manifest SHA-256: `d8b457ad6b932c0a0503cd5e978d5a16933b987b9c088c95358d9f1d4314a881`
- raw JSONL SHA-256: `077f919d3aa67fcd803d577ba81a89bdc7bc27b05a50a40e2be03b1f858a1983`

### Correctness and reliability

| Gate | Result |
|---|---:|
| Protocol | pass |
| Public functional cases | 6/6 |
| Public trajectory | 256/256 against the supplied local regression reference |
| Request success rate | 1.0 |
| No fallback / NaN / unexpected OOM / Xid | all true |
| Health after failure | true |
| Long-context recovery request | pass |

The evaluator reports `output_ids_sha256` equal to the reference for both scored trajectory cells, with edit distance zero. This proves the graph implementation preserved the measured greedy token trajectory.

### Base performance cells

This is the contract's `public_calibration` profile: one measured repeat and no warm-up. CV is structurally zero, so these values are diagnostic rather than five-repeat leaderboard medians.

| Cell | Prompt | TTFT | Prefill rate | TPOT | Decode rate | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| text-perf-1024 | 1,024 | 1.382 s | 740.81 tok/s | 54.49 ms | 18.35 tok/s | 24,026 MiB |
| text-perf-2048 | 2,048 | 2.776 s | 737.88 tok/s | 56.86 ms | 17.59 tok/s | 24,026 MiB |
| text-perf-4096 | 4,096 | 5.601 s | 731.29 tok/s | 61.60 ms | 16.23 tok/s | 24,026 MiB |
| text-perf-8192 | 8,192 | 11.430 s | 716.70 tok/s | 71.07 ms | 14.07 tok/s | 24,026 MiB |
| text-perf-16384 | 16,384 | 23.776 s | 689.09 tok/s | 90.02 ms | 11.11 tok/s | 24,026 MiB |

Rates are `prompt_tokens / TTFT` and `1 / TPOT`. Every base cell completed the required 128 output tokens. Against iteration 13, 1K TPOT moved from 55.10 to 54.49 ms and 8K TPOT from 71.63 to 71.07 ms: approximately 1.1% and 0.8%, respectively. The improvement is too small to justify graph complexity as a throughput solution by itself.

## Long-context evaluation

The official extended run used the generated `context-iter3` dataset. The contract staircase probes the 32,640-token non-scoring diagnostic before the 32,768-token bonus start, then checks service recovery.

| Prompt | Cases attempted | Passing | Output budget | TTFT | Prefill rate | TPOT | Decode rate | E2E |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32,640 | 1 required probe | 1/1 | 128 | 51.079 s | 639.01 tok/s | 127.61 ms | 7.84 tok/s | 67.286 s |

`submission.json` records `max_verified_prompt_tokens=32640`, `pass_rate_at_max_context=1.0`, `verified_output_tokens=128`, `verified_cases_at_max_context=1`, and `service_healthy_after_failure=true`. This is correctness and capacity evidence, not a context-bonus claim: 32,640 is below the contract's 32,768 bonus start and only one of six categories was required at this diagnostic length.

### Concrete long-context question–answer example

Exact question suffix embedded at the end of the frozen 32,640-token `context-32640-retrieval-early` prompt:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact evaluator output text from `raw.jsonl`, shown verbatim (128 output tokens):

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

The row reports `functional_pass=true`, `prompt_tokens=32640`, `completion_tokens=128`, `ttft_s=51.07935217022896`, `tpot_s=0.12760898623410172`, and `e2e_s=67.28576795756817`. Validation detail: normalized prefix expected and observed `KEY-EARLY-767211`.

## Nsight Systems profiling

`nvprof` cannot profile RTX 4090 (`sm_89`); NVIDIA Nsight Systems 2025.1.1 is the supported replacement. Two real-request captures were retained:

- `target/iterate14-profile.nsys-rep` — 2,090,147 bytes; graph-level trace
- `target/iterate14-profile.sqlite` — 5,894,144 bytes
- `target/iterate14-graph-nodes.nsys-rep` — 4,094,505 bytes; node-level graph trace
- `target/iterate14-graph-nodes.sqlite` — 10,899,456 bytes

Representative node-level capture command:

```bash
/opt/nvidia/nsight-compute/2025.1.1/host/target-linux-x64/nsys profile \
  --trace=cuda,cublas,nvtx,osrt --cuda-graph-trace=node \
  --sample=none --cuda-memory-usage=true --stats=true \
  --duration=180 --force-overwrite=true \
  --output=target/iterate14-graph-nodes \
  ./target-iter14/release/apxinf serve \
    --model ../model/qwen --host 127.0.0.1 --port 8014
```

The service received a 1K/8-token request followed by a 1K/128-token request during this capture. SQLite contains 6,741 kernels with non-null CUDA graph node IDs, totaling 378.629 ms. Sixteen fused full-attention kernels occur per token; 112 such instances identify seven fully traced decode replays. Their summed GPU kernel time is 54.09 ms per replay, matching the evaluator's 54.49 ms 1K TPOT closely.

### Graph replay kernel breakdown

Percentages below use only the 378.629 ms of graph-node GPU kernel time, excluding model load, prefill, argmax, and eager activity.

| Kernel/path | Instances | Total | Average | Graph GPU time |
|---|---:|---:|---:|---:|
| Single W4A16 TC projection (`pair=false`) | 1,001 | 162.361 ms | 162.20 µs | 42.88% |
| Paired W4A16 TC projection (`pair=true`) | 896 | 161.793 ms | 180.57 µs | 42.73% |
| cuBLAS BF16 GEMV, large variant | 7 | 18.569 ms | 2.653 ms | 4.90% |
| Fused full-attention decode | 112 | 16.892 ms | 150.82 µs | 4.46% |
| RMSNorm | 903 | 7.927 ms | 8.78 µs | 2.09% |
| Decode-only tile-32 delta recurrence | 336 | 4.127 ms | 12.28 µs | 1.09% |
| cuBLAS BF16 GEMV, small variant | 679 | 3.039 ms | 4.48 µs | 0.80% |
| Convolution + SiLU | 336 | 0.928 ms | 2.76 µs | 0.25% |
| Residual add | 896 | 0.886 ms | 0.99 µs | 0.23% |
| Remaining elementwise/position kernels | 1,456 | 1.108 ms | — | 0.29% |

The two packed W4 projection families consume 85.61% of graph GPU kernel time. CUDA graph replay therefore attacks the wrong dominant term: it reduces CPU submission work but does not reduce projection weight traffic or GPU kernel duration.

### Memory traffic and bandwidth utilization

The graph-level capture recorded 1,775 host-to-device copies totaling 21,013.277 MB and 2,555 memsets totaling 2,982.343 MB. These totals are dominated by model loading and initialization; they are not per-token decode traffic. No D2H copy appears in the Nsight MemOps summary because token selection uses mapped host memory after the GPU argmax kernel.

The contract's frozen minimum-weight proxy is 21,017,689,808 bytes per token and the RTX 4090 theoretical bandwidth is 1,008 GB/s. Dividing this lower-bound byte count by the measured 54.09 ms graph-kernel time gives an effective lower-bound sweep rate of approximately 388.6 GB/s, or 38.5% of theoretical bandwidth. This is a proxy, not a hardware-counter measurement: scale/zero-point reads, activations, cache traffic, and repeated reads make actual bytes larger.

The official evaluator's hardware sampler observed `memory_controller_util_max_pct=78` on the performance and long-context requests, while mean controller utilization was much lower because client/server waits and non-memory phases are included. The two observations are consistent: individual projection kernels exercise memory bandwidth, but the complete token contains many short kernels, scale/zero-point traffic, attention, recurrence, and launch gaps rather than one continuously saturating sweep.

### Bottlenecks

1. **W4 projection execution: 85.61% of graph GPU time.** The dominant work remains one-row asymmetric W4A16 projection kernels. Kernel fusion across dependent projections is not possible without changing the layer dataflow; improvement must come from fewer weight reads, better packed scale/zero-point access, or a faster one-row kernel.
2. **Dense `lm_head`: 2.653 ms/token.** The largest cuBLAS GEMV variant appears once per replay and alone consumes 4.90%. Quantizing or specializing `lm_head` is a discrete remaining opportunity.
3. **Full attention grows with context.** Its 150.82 µs per full-attention layer at 1K becomes increasingly important as the visible KV length grows, explaining TPOT growth from 54.49 ms at 1K to 127.61 ms at 32.64K.
4. **Graph launch overhead is no longer material.** Node-traced GPU execution is 54.09 ms/token versus 54.49 ms client TPOT. At most about 0.40 ms/token remains outside measured graph kernels, including control upload, graph launch, argmax, synchronization, SSE delivery, and measurement overhead.
5. **Memory headroom is unsafe for expansion.** Peak VRAM is 24,026 MiB, leaving roughly 550 MiB below nominal 24 GiB. Larger fixed workspaces, longer KV caches, or parallel requests risk OOM.

## vLLM and requested threshold status

The repository's contract makes the best valid same-round platform vLLM median authoritative. No valid same-round vLLM control artifact was produced locally, so an official 1.2× comparison cannot be claimed. Earlier direct endpoint probes recorded approximately 76.82 tok/s decode at 1K; the iteration-13 report retained that diagnostic value.

- ApxInf 1K prefill: 740.81 tok/s.
- ApxInf 1K decode: 18.35 tok/s.
- Earlier direct vLLM 1K decode: approximately 76.82 tok/s.
- Diagnostic ApxInf/vLLM decode ratio: approximately 0.24×.
- Required 1.2× vLLM decode rate from that diagnostic: approximately 92.18 tok/s.
- Requested absolute decode target: 40 tok/s.
- Result: neither decode threshold is met.

A same-round vLLM prefill control is unavailable; therefore the requested prefill ratio is unproven even though ApxInf measures 689–741 tok/s over the five base cells. No pass claim is fabricated.

## Negative controls and limitations

- Graph capture is skipped when `APXINF_LAYER_PROF`, `APXINF_KERNEL_PROF`, `APXINF_GEMM_PROF`, or `APXINF_TRACE` is enabled; those modes contain capture-hostile synchronization, CUDA events, host copies, or file writes.
- Replay failures are terminal and never fall back to eager execution after partial KV/recurrent mutation.
- The captured graph is destroyed before all allocations it references.
- This measured checkpoint uses the packed W4 path. Dense fallback weights were not graph-qualified.
- The source contains a `prewarm_decode` trait hook, but this Qwen graph is lazily captured on the first decode token. Its capture/instantiate cost lands in TPOT; moving safe capture before prefill requires a non-mutating CUDA prepare path for the decode-only delta kernel's one-time function-attribute setup.
- Public calibration has one repeat; hidden correctness, five-repeat official medians/CV, a valid vLLM control artifact, 32,768+ six-category context scoring, multi-request, and multimodal capability remain unavailable.
- The profile establishes kernel time composition but does not provide Nsight Compute DRAM throughput counters; bandwidth utilization above is explicitly a frozen-byte-proxy calculation.

## Reproduction

```bash
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
export CUDA_PATH=/usr/local/cuda
export APXINF_CUDA_ARCH=sm_89
export CARGO_TARGET_DIR=target-iter14
export RUSTFLAGS='-C link-arg=-fuse-ld=gold'

cargo check --workspace --locked
cargo build --release --features cuda -p apxinf --bin apxinf
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target-iter14/release/apxinf serve \
    --model ../model/qwen --host 127.0.0.1 --port 8002
```

Run the official base and long-context evaluation with the exact command under **Official evaluation**. Generate profiling summaries with:

```bash
nsys stats --force-export=true --report cuda_gpu_kern_sum \
  target/iterate14-graph-nodes.nsys-rep
nsys stats --force-export=true \
  --report cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
  target/iterate14-profile.nsys-rep
```

No model weights, credentials, private evaluation data, or fabricated vLLM results are included.

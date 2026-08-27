# Iteration Report 19 - Final All-at-Once Optimization

Date: 2026-08-25  
Accepted run: `iterate19-final-clean`  
Build: LLD via `cc`, 40 Cargo jobs  
Model: `../model/qwen`

## Scope

Seven remaining tracks were executed concurrently: Marlin-compatible W4 investigation, raw W4 decode, exact GDN graph replay, dense LM-head tactics, full-attention workspace, host/service overhead, and 32K context capacity. Every candidate was measured against the exact eager iteration-18 path; slower or unproven paths are not enabled by default.

## Accepted implementation

1. Physical KV capacity remains 32,768 tokens. Full-attention scratch aliases checked, disjoint regions of the existing dense workspace rather than reserving permanent score/Kt/Vf32/PV slabs.
2. Full-attention prefill uses the current valid prefix for score, Kt, and Vf32 temporary extents and leading dimensions. Physical K/V cache addressing remains at the fixed 32K stride; causal masking and reduction order are unchanged.
3. The original exact raw-layout W4 tensor-core schedule remains the production decode path. Experimental repacked tensors require `APXINF_QWEN35_W4_REPACKED=1`, a complete explicitly suffixed repacked set, and a complete raw fallback set.
4. The eager exact GDN sequence remains the production path. CUDA graph ownership was made safe, but graph replay is opt-in through `APXINF_EXACT_GDN_GRAPH` because it lost measured throughput.
5. Final logits use the established cuBLAS `write_ex` path. The speculative cuBLASLt tactic was removed because a single all-ones probe cannot prove request-wide BF16 equality or a production latency win.
6. The service avoids disabled per-token timing calls, caches immutable health JSON, moves the completions prompt instead of cloning it, uses a stack message array, and removes no-op socket flushes while preserving response bytes and errors.

## Rejected paths

- Physical `W4_REPACKED_N64_K16_V1` decode and fused packed prefill were 256/256 exact but changed 1K TPOT from approximately 55 ms to 80 ms and TTFT from approximately 0.78 s to 1.43 s.
- vLLM Marlin cannot consume ApxInf's raw `[N,K/8]` or experimental N64/K16 tensor directly. It requires GPTQ K-packed repack, padded N/K, permuted scales, unpacked/reordered zero-points, and a Marlin workspace. No safe ABI shortcut exists.
- Per-layer exact GDN graph replay preserved 256/256 but produced approximately 68.18 ms 1K TPOT, versus 55.18 ms for eager execution. It is opt-in only.
- The streamed-activation raw W4 specialization preserved 256/256 but produced approximately 68.30 ms 1K TPOT. It was removed; the faster original schedule is compiled.
- Dense-head cuBLASLt routing was removed: initialization-only equality and timing do not establish arbitrary-request equality or improvement over production cuBLAS.
- TP2 remains capacity-only. The measured 129 reductions per forward dominate PCIe TP2 execution, so TP1 remains the latency configuration.

## Official evaluation

Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate19-final-clean/`

| Prompt | TTFT | Prefill | TPOT | Decode | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.7789 s | 1314.7 tok/s | 55.18 ms | 18.12 tok/s | 23854 MiB |
| 2,048 | 1.5778 s | 1298.0 tok/s | 57.54 ms | 17.38 tok/s | 23854 MiB |
| 4,096 | 3.2304 s | 1267.9 tok/s | 62.26 ms | 16.06 tok/s | 23854 MiB |
| 8,192 | 6.6290 s | 1235.8 tok/s | 71.70 ms | 13.95 tok/s | 23854 MiB |
| 16,384 | 13.8151 s | 1185.9 tok/s | 90.57 ms | 11.04 tok/s | 23854 MiB |

Correctness and reliability:

- Public functional cases: **6/6**
- Public token trajectory: **256/256**
- Protocol: **pass**
- Request success rate: **1.0**
- No fallback, NaN, unexpected OOM, or XID
- 32,640-token context: **pass**, 128 output tokens
- Raw SHA-256: `acd5094647d0a44013a93e6829736a0401f3f965fd0d83f7223f4dc99b9d8de5`

## vLLM comparison

Control: `benchmarks/qwen38_4090/evaluation/runs/vllm-one-gpu-control/`.

| Prompt | ApxInf/vLLM prefill | ApxInf/vLLM decode |
|---:|---:|---:|
| 1,024 | 0.474x | 0.366x |
| 2,048 | 0.445x | 0.352x |
| 4,096 | 0.440x | 0.327x |
| 8,192 | 0.437x | 0.286x |
| 16,384 | 0.440x | 0.229x |

At 1K, ApxInf is 1314.7 prefill tok/s and 18.12 decode tok/s; vLLM is 2774.2 prefill tok/s and 49.45 decode tok/s. The required 1.2x vLLM threshold is **not met** for either phase. Meeting the threshold would require at least 3329.1 prefill tok/s and 59.35 decode tok/s at this cell.

## Nsight Systems profile

Artifacts:

- `target/iterate19-final-clean-profile.nsys-rep`
- `target/iterate19-final-clean-profile.sqlite`

Captured workloads: one exact 1,024-token prompt with 128-token decode and one exact 32,640-token prompt with 128-token decode.

CUDA GPU kernel breakdown:

| Kernel family | GPU time share | Total GPU time | Instances | Median |
|---|---:|---:|---:|---:|
| Raw W4 TC, single projection | 33.7% | 1.2609 s | 7,766 | 130.2 us |
| Raw W4 TC, paired projection | 33.7% | 1.2587 s | 6,952 | 190.4 us |
| Exact delta step | 8.3% | 0.3115 s | 2,703 | 16.7 us |
| BF16 CUTLASS 128x64 prefill GEMM | 6.9% | 0.2592 s | 2,366 | 120.1 us |
| Dense LM-head GEMV | 3.9% | 0.1460 s | 55 | 2.653 ms |
| Qwen flash prefill | 3.6% | 0.1338 s | 869 | 154.0 us |
| BF16 CUTLASS 128x128 prefill GEMM | 2.8% | 0.1029 s | 640 | 158.5 us |
| RMS normalization | 1.8% | 0.0655 s | 7,263 | 8.9 us |
| Tiled W4 dequantization | 1.6% | 0.0580 s | 3,260 | 18.5 us |

The raw W4 projection variants consume **67.4%** of all captured GPU kernel time. This is the primary decode bottleneck. Exact recurrence is the next model-specific target at 8.3%; attention-specific decode kernels are below 0.1% individually.

CUDA API observations:

- 60,318 `cudaLaunchKernel` calls consumed 0.403 s of host API time.
- 55 `cudaStreamSynchronize` calls consumed 2.735 s, corresponding to token-boundary synchronization and readback.
- 1,924 startup/workspace `cudaMalloc` calls consumed 1.751 s in the captured process lifetime.
- GPU memory operations were 99.8% host-to-device by time, dominated by model upload. This is startup cost, not per-token decode traffic.

Memory bandwidth/utilization:

Nsight GPU performance counters were rejected by the host with `ERR_NVGPUCTRPERM`, so privileged achieved-DRAM-bandwidth and SM counter percentages are unavailable. The official evaluator's concurrent hardware sampling for the accepted 32,640-token request measured memory-controller utilization at **32% peak** and **4.75% wall-time mean**, GPU utilization at 100% peak and 25% wall-time mean, and 404.97 W peak power. The low wall-time means include host/token intervals; they must not be interpreted as kernel-only bandwidth efficiency. Kernel timing and the W4 traffic structure still identify raw W4 projection as the dominant optimization target.

## Exact long-context example

Question embedded at the end of the 32,640-token document:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact answer returned by `context-32640-retrieval-early`:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

Exact row evidence: `functional_pass=true`, `prompt_tokens=32640`, `completion_tokens=128`, `ttft_s=29.850403524935246`, `tpot_s=0.12800810175148522`, `e2e_s=46.10750688612461`, output SHA-256 `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.

## Reproduction

```bash
cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve --model ../model/qwen --host 127.0.0.1 --port 8002
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8002 \
  --implementation-name apxinf-final \
  --implementation-revision worktree-final-clean \
  --backend apxinf --profile public_calibration \
  --trajectory-reference target/iterate14-trajectory-reference.json \
  --run-context --warmups 0 --repeats 1 --timeout 1800 \
  --run-id iterate19-final-clean \
  --output-dir benchmarks/qwen38_4090/evaluation/runs
```

# Exact TP2 vLLM versus ApxInf profiling comparison

Date: 2026-08-24  
Model: `../model/qwen`  
vLLM: 0.27.1 from `../model/vllm/.venv`  
Deployment: `CUDA_VISIBLE_DEVICES=2,3`, tensor parallel size 2, max model length 262,144, FP8 KV cache, max 20 sequences

## Deployment profiled

```bash
CUDA_VISIBLE_DEVICES=2,3 \
  ../model/vllm/.venv/bin/vllm serve ../model/qwen \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --gpu-memory-utilization 0.97 \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --max-num-seqs 20 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --tensor-parallel-size 2
```

The profiler service added only the served-model name, loopback host/port, and vLLM torch-profiler configuration. Both requested GPUs were otherwise free before startup.

Artifacts:

- `target/vllm-tp2-profiler/rank0.1787591265918382496.pt.trace.json`
- `target/vllm-tp2-profiler/rank1.1787591265919846343.pt.trace.json`
- `target/vllm-tp2-profiler/profiler_out_0.txt`
- `target/vllm-tp2-profiler/profiler_out_1.txt`
- `target/vllm-tp2-nvprof.txt`

The profiler captured warmed 1K and 8K prompts with 128-token decode budgets. Client elapsed times during profiling were approximately 0.963 s for 1K and 2.597 s for 8K. Profiler overhead means these are not leaderboard latency measurements.

## Module bottlenecks

### Rank 0 / visible GPU 2

| Module family | Aggregate CUDA time | Share of classified kernel time |
|---|---:|---:|
| Marlin W4 projections | 190.850 ms | 57.4% |
| NCCL ring all-reduce | 125.740 ms | 37.8% |
| Dense BF16 GEMV/MM | 7.6 ms GEMV plus other dense MM | 2.3% classified |
| GDN kernels | 6.0 ms | 1.8% |
| FlashInfer full attention | 2.3 ms | 0.7% |
| Argmax | 0.300 ms total | negligible |

Raw dominant rows:

- Marlin context/prefill family: 156.865 ms across 255 calls.
- Marlin decode family: 33.985 ms across 1,020 calls.
- NCCL BF16 ring all-reduce: 125.740 ms across 645 calls.
- `vllm::qwen_gdn_attention_core`: 74.538 ms CPU-total wrapper time, but the classified GDN CUDA kernels total about 6 ms.
- Dense BF16 cuBLAS GEMV: 6.808 ms across nine calls.
- Full attention wrapper: about 5.414 ms including wrapper events; actual FlashInfer kernel share is below 1% of classified kernel time.

### Rank 1 / visible GPU 3

| Module family | Aggregate CUDA time | Share of classified kernel time |
|---|---:|---:|
| Marlin W4 projections | 180.973 ms | 54.6% |
| NCCL ring all-reduce | 135.108 ms | 40.8% |
| Dense BF16 GEMV/MM | 7.6 ms GEMV plus other dense MM | 2.3% classified |
| GDN kernels | 5.6 ms | 1.7% |
| FlashInfer full attention | 2.2 ms | 0.7% |
| Argmax | 0.240 ms total | negligible |

Both ranks agree: **Marlin W4 projection is the largest compute family, and TP2 NCCL communication is the second bottleneck.** Together they account for roughly 95-96% of the classified dominant GPU-kernel time.

## Why TP2 differs from TP1

TP2 halves each rank's projection width, reducing per-rank Marlin work, but introduces 645 BF16 ring all-reduces in the captured range. On two consumer RTX 4090 GPUs without NVLink, that communication consumes 38-41% of classified time. Full attention and GDN are not the limiting modules for this short single-request workload.

Optimization order for this deployment:

1. Marlin W4 projections: improve or reduce quantized projection work; preserve packed layout and fusion.
2. Tensor-parallel communication: fuse/reduce all-reduces, overlap communication with independent work, or prefer TP1 when capacity permits.
3. Dense LM-head/GEMV work.
4. GDN and full attention only after the first two categories.
5. Sampling/frontend last.

## ApxInf comparison

The accepted ApxInf exact run and optimized Nsight trace are:

- `benchmarks/qwen38_4090/evaluation/runs/iterate16-parallel-exact/`
- `target/iterate16-parallel-profile.nsys-rep`
- `target/iterate16-parallel-profile.sqlite`

ApxInf classified profile:

| ApxInf family | GPU time share |
|---|---:|
| Single W4 TC | 21.0% |
| Paired W4 TC | 20.9% |
| Exact delta recurrence | 18.2% |
| Prefill GEMM families | 23.6% |
| Tiled W4 dequantization | 3.8% |
| Dense LM-head GEMV | 2.5% |
| Full attention | 2.2% |
| Conv + SiLU | 1.6% |
| RMSNorm | 1.2% |

The shared conclusion is unchanged: W4 projection and weight handling dominate. ApxInf has no NCCL cost because it runs on one GPU, but its raw checkpoint layout lacks vLLM's Marlin loader repack. ApxInf additionally spends significant time in exact recurrence and prefill dequant/GEMM work.

## nvprof result

`nvprof` was executed with `CUDA_VISIBLE_DEVICES=2,3` and `--devices all`. It produced no GPU profile:

```text
Warning: This version of nvprof doesn't support the underlying device, GPU profiling skipped
Warning: nvprof is not supported on devices with compute capability 8.0 and higher.
Use NVIDIA Nsight Systems for GPU tracing and NVIDIA Nsight Compute for GPU profiling.
```

RTX 4090 is sm_89, so nvprof cannot provide module or kernel metrics. The exact output is retained in `target/vllm-tp2-nvprof.txt`. vLLM's PyTorch/CUDA profiler traces and ApxInf's Nsight Systems traces are the valid comparison evidence.

## Final answer

For the exact two-GPU vLLM command, the bottlenecks are:

1. **Marlin W4 projections: 55-57%** of classified kernel time.
2. **NCCL all-reduce: 38-41%**.
3. Dense GEMV/MM: about 2-3%.
4. GDN kernels: below 2%.
5. Full attention: below 1% for the profiled 1K/8K requests.
6. Argmax/sampling: negligible.

The next material optimization is not GDN or attention. It is reducing W4 projection time and TP2 all-reduce overhead, or using TP1 when the memory/capacity target permits.

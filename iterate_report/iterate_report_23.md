# Iteration Report 23 - Final Paired Activation Round

Date: 2026-08-25  
Accepted run: `iterate23-default`  
Production baseline: unchanged iteration-21 alternate single-tile W4 default  
Build: LLD via `cc`, 40 Cargo jobs

## Scope

Eight final tracks ran concurrently: paired two-tile coarsening, paired activation reuse, paired metadata prefetch, paired warp specialization, pair scheduling, output-store coalescing, long-context attention, and default-gate auditing.

## Candidate result

Every runnable candidate preserved the frozen 128-token trajectory but was slower than the exact production control. No final candidate was promoted.

| Candidate | Stable 1K TPOT | Exactness | Decision |
|---|---:|---:|---|
| Production control | 49.71 ms | 128/128 | Retained |
| Pair coarsening | 65.30 ms | 128/128 | Rejected |
| Pair activation reuse | 68.60 ms | 128/128 | Rejected |
| Pair async prefetch | 53.28 ms | 128/128 | Rejected |
| Pair warp specialization | 52.91 ms | 128/128 | Rejected |
| Alternate output stores | 69.80 ms | 128/128 | Rejected |
| Pair reuse + alternate stores | 69.72 ms | 128/128 | Rejected |

Artifact: `target/iterate23-candidate-matrix.json`.

Pair scheduling was rejected without a gate: all model work and cuBLAS calls use one CUDA stream, so reordering independent launches cannot overlap execution. A second stream would require dependency events and a broader backend redesign. The valid-prefix long-context path is already maximal: workspace uses `seq x visible`, cache views retain physical `MAX_SEQ_LEN` strides, and `start_pos + seq <= 32768` is enforced. Gate auditing added correctness guards so unsupported W4 group geometries fall back instead of entering an incompatible MMA path, and repaired concurrent output-store routing.

## Official evaluation

Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate23-default/`

| Prompt | TTFT | Prefill | TPOT | Decode | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.7827 s | 1307.9 tok/s | 49.86 ms | 20.06 tok/s | 23852 MiB |
| 2,048 | 1.5862 s | 1291.1 tok/s | 52.20 ms | 19.16 tok/s | 23852 MiB |
| 4,096 | 3.2458 s | 1261.7 tok/s | 56.92 ms | 17.57 tok/s | 23852 MiB |
| 8,192 | 6.6625 s | 1229.5 tok/s | 66.39 ms | 15.06 tok/s | 23852 MiB |
| 16,384 | 13.8829 s | 1180.1 tok/s | 85.27 ms | 11.73 tok/s | 23852 MiB |

Correctness and reliability:

- Public functional cases: **6/6**
- Public trajectory: **256/256**
- Protocol: **pass**
- Request success rate: **1.0**
- No fallback, NaN, unexpected OOM, or XID
- 32,640-token context: **pass**, 128 output tokens
- Peak VRAM: **23,852 MiB**
- Raw SHA-256: `2a459a5cdf5fa73a84aa23bbed787f1bf89bf442279a76f6b0043735c528c808`

## vLLM threshold

At 1K, the one-GPU vLLM control is 2774.2 prefill tok/s and 49.45 decode tok/s. ApxInf is approximately 0.471x prefill and 0.406x decode. The required 1.2x threshold remains unmet; the required minimums are 3329.1 prefill tok/s and 59.35 decode tok/s.

## Nsight Systems profile

Artifacts:

- `target/iterate23-default-profile.nsys-rep`
- `target/iterate23-default-profile.sqlite`

Captured workloads: one exact 1K/128-token request and one exact 32,640/128-token request.

| Kernel family | GPU share | Total | Instances | Median |
|---|---:|---:|---:|---:|
| Paired raw W4 TC | 37.4% | 1.4023 s | 7,747 | 120.6 us |
| Alternate-tile raw W4 TC | 29.2% | 1.0942 s | 8,653 | 78.4 us |
| Fused norm/delta/gated GDN | 8.5% | 0.3204 s | 3,002 | 16.8 us |
| BF16 CUTLASS 128x64 prefill GEMM | 6.8% | 0.2560 s | 2,366 | 120.0 us |
| Dense LM-head GEMV | 4.3% | 0.1619 s | 61 | 2.654 ms |
| Qwen flash prefill | 4.0% | 0.1495 s | 968 | 154.3 us |
| BF16 CUTLASS 128x128 prefill GEMM | 2.7% | 0.1017 s | 640 | 157.8 us |

W4 projection families account for **66.6%** of GPU kernel time. Paired W4 remains the dominant optimization target, but all tested exact activation-reuse, coarsening, prefetch, warp-specialized, and output-store variants regress.

Privileged Nsight SM/DRAM counters remain unavailable because the host reports `ERR_NVGPUCTRPERM`. Official 32K hardware sampling measured memory-controller utilization at approximately 31% peak / 4.9% mean, GPU utilization at 100% peak / 25% mean, and 23,852 MiB peak VRAM.

## Exact long-context example

Question:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact answer:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

Exact row evidence: `functional_pass=true`, `prompt_tokens=32640`, `completion_tokens=128`, `ttft_s=29.998726785182953`, `tpot_s=0.12255562466429913`, `e2e_s=45.563349947333336`, output SHA-256 `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.

## Reproduction

```bash
cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve --model ../model/qwen --host 127.0.0.1 --port 8002
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8002 \
  --implementation-name apxinf-iter23 \
  --implementation-revision final-default-unchanged \
  --backend apxinf --profile public_calibration \
  --trajectory-reference target/iterate14-trajectory-reference.json \
  --run-context --warmups 0 --repeats 1 --timeout 1800 \
  --run-id iterate23-default --output-dir benchmarks/qwen38_4090/evaluation/runs
```

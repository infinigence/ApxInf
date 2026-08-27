# Iteration Report 24 - Native W4 Candidate Round

Date: 2026-08-25  
Accepted run: `iterate24-default`  
Production path: unchanged iterate-23 defaults  
Build: LLD via `cc`, 40 Cargo jobs

## Scope

Eight native tracks ran concurrently: CUTLASS int4, fused scale epilogue, persistent transformed-weight cache, persistent layer execution, shared paired activation, native prefill, raw accumulator audit, and final gate audit.

## Candidate matrix

Artifact: `target/iterate24-candidate-matrix.json`.

| Candidate | Stable 1K TPOT | Exactness | Decision |
|---|---:|---:|---|
| Production control | 49.68 ms | 128/128 | Retained |
| Scale epilogue | 55.07 ms | 128/128 | Rejected |
| Transform cache | 80.07 ms | 128/128 | Rejected; startup/cache path slower |
| Shared paired activation | 50.85 ms | 128/128 | Rejected |
| Native prefill | 49.84 ms decode | 128/128 | Opt-in only; no measured win |
| Transform cache + scale epilogue | 80.02 ms | 128/128 | Rejected |

CUTLASS int4 is a definitive source-backed blocker on RTX 4090: vendored SM80 int4 MMA is s4/u4 to s32 only, while the required path needs per-output/per-group asymmetric BF16 operands and ordered BF16-to-F32 accumulation. CUTLASS mixed-input affine conversion is SM90+/SM100 and does not match the current sm89 build or packed zero-point ABI. Persistent layer execution is also blocked: required RMS/GDN/residual/SiLU dependencies prevent a single exact layer kernel, and a host wrapper would only repeat existing pair launches.

The accumulator audit fixed concrete fallback safety: TC paths require complete K=128 tiles; scalar fused W4 requires complete K=256/group-aligned tiles; row-tiled prefill requires group-size alignment. Unsupported geometry now falls back instead of entering an invalid kernel.

## Official evaluation

Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate24-default/`

| Prompt | TTFT | Prefill | TPOT | Decode | Prefill/vLLM | Decode/vLLM |
|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.7839 s | 1306.3 tok/s | 49.87 ms | 20.05 tok/s | 0.471x | 0.405x |
| 2,048 | 1.5878 s | 1289.9 tok/s | 52.23 ms | 19.14 tok/s | 0.442x | 0.388x |
| 4,096 | 3.2498 s | 1260.4 tok/s | 56.95 ms | 17.56 tok/s | 0.437x | 0.358x |
| 8,192 | 6.6677 s | 1228.6 tok/s | 66.40 ms | 15.06 tok/s | 0.434x | 0.309x |
| 16,384 | 13.8940 s | 1179.2 tok/s | 85.24 ms | 11.73 tok/s | 0.438x | 0.244x |

Correctness and reliability:

- Public functional cases: **6/6**
- Public trajectory: **256/256**
- Protocol: **pass**
- Request success rate: **1.0**
- No fallback, NaN, unexpected OOM, or XID
- 32,640-token context: **pass**, 128 output tokens
- Peak VRAM: **23,852 MiB**
- Raw SHA-256: `c95e978e2e6788f2beb3c769f72ebeb772b858b89a4f3866cdab399c06b06b57`

The 1.2x vLLM threshold remains unmet. At 1K, ApxInf is 1306.3 prefill tok/s and 20.05 decode tok/s; vLLM is 2774.2 prefill tok/s and 49.45 decode tok/s. Required minimums are 3329.1 and 59.35 tok/s.

## Nsight Systems profile

Artifacts:

- `target/iterate24-default-profile.nsys-rep`
- `target/iterate24-default-profile.sqlite`

| Kernel family | GPU share | Total | Instances | Median |
|---|---:|---:|---:|---:|
| Paired raw W4 TC | 37.3% | 1.4032 s | 7,760 | 192.3 us |
| Alternate-tile raw W4 TC | 29.1% | 1.0959 s | 8,668 | 77.7 us |
| Fused norm/delta/gated GDN | 8.6% | 0.3229 s | 3,006 | 16.8 us |
| BF16 CUTLASS 128x64 prefill GEMM | 6.9% | 0.2585 s | 2,366 | 120.1 us |
| Dense LM-head GEMV | 4.3% | 0.1618 s | 61 | 2.653 ms |
| Qwen flash prefill | 4.0% | 0.1497 s | 970 | 154.2 us |

W4 projections remain **66.4%** of GPU kernel time. Nsight privileged SM/DRAM counters remain unavailable due to `ERR_NVGPUCTRPERM`. Official 32K sampling measured memory-controller utilization at approximately 31% peak / 4.9% mean and peak VRAM at 23,852 MiB.

## Exact long-context example

Question:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact answer:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

Exact evidence: `functional_pass=true`, `prompt_tokens=32640`, `completion_tokens=128`, `ttft_s=30.012368991971016`, `tpot_s=0.12265109616940416`, `e2e_s=45.58917386829853`, output SHA-256 `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.

## Reproduction

```bash
cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve --model ../model/qwen --host 127.0.0.1 --port 8002
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8002 \
  --implementation-name apxinf-iter24 --implementation-revision native-round-default-unchanged \
  --backend apxinf --profile public_calibration \
  --trajectory-reference target/iterate14-trajectory-reference.json \
  --run-context --warmups 0 --repeats 1 --timeout 1800 \
  --run-id iterate24-default --output-dir benchmarks/qwen38_4090/evaluation/runs
```

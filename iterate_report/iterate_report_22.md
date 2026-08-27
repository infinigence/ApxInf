# Iteration Report 22 - Paired W4 Projection Optimization

Date: 2026-08-25  
Accepted run: `iterate22-default`  
Production path: iteration-21 alternate single-tile W4 default  
Build: LLD via `cc`, 40 Cargo jobs

## Scope

Eight paired-projection tracks ran concurrently: 32-row paired tile, separated alternate launches, two-warp tile, six-warp tile, source-local register bounds, cache-hinted loads, double-buffered activation staging, and multi-row prefill pair batching. All candidates were exact-gated and retained the iteration-21 fallback.

## Result

No paired candidate beat the existing production path. Therefore no paired gate was promoted. The default remains alternate single-tile W4 for eligible single projections, fused exact GDN, and event-scoped argmax completion. All paired implementations remain opt-in for further diagnostics.

## Candidate matrix

Artifact: `target/iterate22-candidate-matrix.json`.

| Candidate | Stable 1K TPOT | Exactness | Decision |
|---|---:|---:|---|
| Production control | 49.77 ms | 128/128 | Retained |
| Paired 32-row/4-warp tile | 51.42 ms | 128/128 | Rejected |
| Separated alternate launches | 49.99 ms | 128/128 | Rejected; slower |
| Paired 2-warp tile | 50.87 ms | 128/128 | Rejected |
| Paired 6-warp tile | 52.53 ms | 128/128 | Rejected |
| Register-bound paired tile | 53.09 ms | 128/128 | Rejected |
| Cache-hinted paired tile | 54.02 ms | 128/128 | Rejected |
| Double-buffered activation paired tile | 53.99 ms | 128/128 | Rejected |
| Prefill pair batch | 49.77 ms decode | 128/128 | Opt-in; no decode benefit |

All probes matched the frozen trajectory SHA-256 `7eedbc78e930361a167ea9dec3f827d5ea9aeb25148e18fb854c5efe36e85bea`.

The paired alternate candidate is slower because the combined CTA geometry does not amortize metadata and synchronization as effectively as the already-optimized single tile. Separated launches avoid that geometry cost but still do not beat the default.

## Official evaluation

Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate22-default/`

| Prompt | TTFT | Prefill | TPOT | Decode | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.7821 s | 1307.9 tok/s | 49.86 ms | 20.06 tok/s | 23852 MiB |
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
- Raw SHA-256: `3bcd12d052b4a4cea3eb380db7c350be4c9a03f8b4298d83f852f68dd7424b29`

## vLLM comparison

The one-GPU vLLM control remains 2774.2 prefill tok/s and 49.45 decode tok/s at 1K. This run is approximately 0.471x prefill and 0.406x decode relative to that control. The required 1.2x target remains unmet; required 1K minimums are 3329.1 prefill tok/s and 59.35 decode tok/s.

## Nsight Systems profile

Artifacts:

- `target/iterate22-default-profile.nsys-rep`
- `target/iterate22-default-profile.sqlite`

Captured workloads: exact 1K/128-token and 32,640/128-token requests on the unchanged production path.

| Kernel family | GPU share | Total | Instances | Median |
|---|---:|---:|---:|---:|
| Paired raw W4 TC | 37.3% | 1.4022 s | 7,763 | 127.8 us |
| Alternate-tile raw W4 TC | 29.2% | 1.0959 s | 8,671 | 76.9 us |
| Fused norm/delta/gated GDN | 8.6% | 0.3216 s | 3,007 | 16.8 us |
| BF16 CUTLASS 128x64 prefill GEMM | 6.9% | 0.2574 s | 2,366 | 120.1 us |
| Dense LM-head GEMV | 4.3% | 0.1619 s | 61 | 2.653 ms |
| Qwen flash prefill | 4.0% | 0.1498 s | 970 | 154.3 us |

Paired and single W4 families together consume **66.5%** of GPU kernel time. Paired W4 is the dominant remaining target, but all tested exact paired variants were slower than the production fallback. Nsight privileged SM/DRAM counters remain unavailable because the host reports `ERR_NVGPUCTRPERM`.

Official 32K hardware sampling measured memory-controller utilization at **31% peak / 4.87% mean**, GPU utilization at 100% peak / 25% mean, 401.53 W peak power, and 23,852 MiB peak VRAM.

## Exact long-context example

Question:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact answer:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

Exact row evidence: `functional_pass=true`, `prompt_tokens=32640`, `completion_tokens=128`, `ttft_s=29.999435625970364`, `tpot_s=0.12261037815978208`, `e2e_s=45.571051344275475`, output SHA-256 `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.

## Reproduction

```bash
cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve --model ../model/qwen --host 127.0.0.1 --port 8002
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8002 \
  --implementation-name apxinf-iter22 \
  --implementation-revision default-iter21-paired-matrix \
  --backend apxinf --profile public_calibration \
  --trajectory-reference target/iterate14-trajectory-reference.json \
  --run-context --warmups 0 --repeats 1 --timeout 1800 \
  --run-id iterate22-default --output-dir benchmarks/qwen38_4090/evaluation/runs
```

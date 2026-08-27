# Iteration Report 27 - Standalone Unified Marlin W4

Date: 2026-08-25 | Implementation revision: `f4793ee6d7782c61a55fb2db95cc52d438b5d473` | Run id: `iterate27-definitive`

## Result

This iteration replaced most of the custom raw W4 decode path with a standalone, Apache-2.0 Marlin implementation compiled directly into ApxInf. It does not link or call vLLM, Python, torch, or libtorch. The GPU retains one physical W4 representation per weight.

Correctness and reliability pass completely. Decode improves 21.5-37.6% over iteration 26. Exact prefill is 12.8-14.2% slower because Marlin-layout weights must be inversely dequantized before the established cuBLAS schedule.

**The requested 1.2x vLLM threshold remains unmet.** At 1K ApxInf reaches 1144.4 tok/s prefill and 34.16 tok/s decode; the required minima are 3329.1 and 59.35 tok/s.

## Definitive official evaluation

Environment-free production defaults; `run_evaluation.py`, `public_calibration`, public suite plus 32,640-token context.

| Cell | TTFT | Prefill | TPOT | Decode | Decode gain vs iter26 | VRAM | Prefill/vLLM | Decode/vLLM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| text-perf-1024 | 0.895 s | 1144.4 tok/s | 29.28 ms | 34.16 tok/s | 37.59% | 23904 MiB | 0.413x | 0.691x |
| text-perf-2048 | 1.808 s | 1132.6 tok/s | 31.64 ms | 31.60 tok/s | 35.77% | 23904 MiB | 0.388x | 0.640x |
| text-perf-4096 | 3.691 s | 1109.8 tok/s | 36.33 ms | 27.53 tok/s | 32.69% | 23904 MiB | 0.385x | 0.561x |
| text-perf-8192 | 7.547 s | 1085.4 tok/s | 45.74 ms | 21.86 tok/s | 27.87% | 23904 MiB | 0.384x | 0.448x |
| text-perf-16384 | 15.667 s | 1045.8 tok/s | 64.60 ms | 15.48 tok/s | 21.49% | 23904 MiB | 0.388x | 0.322x |

- Protocol: pass.
- Public functional correctness: **6/6**.
- Public trajectory: **256/256**, zero edit distance at 1K and 8K.
- Request success rate: **1.0**.
- No fallback, NaN, unexpected OOM, or XID; service healthy after context run.
- Provisional leaderboard score against local vLLM control: **67.2797**; automated course points **53.8238**.
- Raw evidence SHA-256: `05cffba95d3d4fd3c1cb644692045b5aeac42f0b6d883b4fa7cc962ce0f1557d`.
- Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate27-definitive/`.

## Production execution design

### Standalone native Marlin

Vendored source provenance: vLLM tag `v0.27.1`, derived from IST-DASLab Marlin, Apache-2.0. Notices are retained in `crates/apxinf-cuda/kernels/marlin/LICENSE` and source headers.

The compiled path contains only BF16 activation/output/scales, unsigned asymmetric U4 weights, group size 32, integer zero-points, and SM80-family four-stage specializations. Raw C ABI:

- `apxinf_marlin_awq_u4_g32_v1_gemm_bf16`;
- `apxinf_marlin_awq_u4_g32_v1_gemm_batch_bf16`;
- AWQ repack and inverse-dequant helpers.

No torch/vLLM symbols are linked.

### One-layout ownership

At load time, compressed-tensors `[N,K/8]` weights, `[N,K/32]` BF16 scales, and `[N/8,K/32]` zero-points are transformed on CPU into:

- qweight `[padded_K/16,padded_N*2]` I32;
- scales `[padded_K/32,padded_N]` BF16 with upstream scale permutation;
- zero-points `[padded_K/32,padded_N/8]` packed U4 with Marlin permutation/interleave.

Eligible weights upload only this representation. Raw checkpoint tensors remain CPU-owned; raw device weights are uploaded only for roles excluded from Marlin. This avoids the 13.08 GiB duplicate-layout OOM.

### Exact architecture hybrid

Marlin defaults on for `q,k,v,o,out,gate,up,down`. Linear-attention `in_proj_qkv` remains raw because Marlin changes public trajectories. The raw role uses a shape-specific exact `N=10240,K=5120,group32` kernel. `APXINF_MARLIN=0` restores all-raw routing; `APXINF_MARLIN_ROLES=all` enables experimental full Marlin.

### Exact prefill from Marlin layout

Seq>1 never uses Marlin GEMM. The inverse kernel reconstructs the exact row-major BF16 weights from the single Marlin representation, then runs the established 32 MiB output-row tile sequence through cuBLAS. Matching the raw path's individual `(M,N,K)` call shapes was necessary: one full-N cuBLAS call changed tactics and broke trajectories even when dense bytes were correct.

The inverse kernel now emits 16 contiguous BF16 values per lane, reusing packed qwords, scale, and zero-point. Prefill recurrence uses four 32-column blocks per value head, reducing shared memory from about 66 KiB to 17 KiB per block while preserving timestep/K ordering and BF16 boundaries.

## Correctness frontier and rejected candidates

- Full Marlin: 1K initially **127/128**, 26.10 ms/token; after exact prefill, 1K became exact but 8K remained only **29/128**.
- Architecture roles individually at 1K:
  - attention output, MLP gate/up, and MLP down: exact;
  - full-attention q/k/v individually: exact;
  - linear-attention qkv: unsafe.
- Exact role combination `q,k,v,o,out,gate,up,down`: exact at both 1K and 8K.
- Physical Q/K/V concatenation: 1K 127/128 and 8K 29/128; rejected.
- Physical gate/up concatenation: 1K exact but 8K 29/128; rejected.
- Specialized fused gated flash: exact but slightly slower than baseline flash+sigmoid at 1K and 8K; retained opt-in.
- Dense LM head: unchanged. A custom reduction cannot guarantee cuBLAS BF16-logit and tie equivalence; a second transposed 2.37 GiB copy is not acceptable.

## Long-context evaluation

The 32,640-token diagnostic passed with 128 output tokens and successful health/recovery checks.

- TTFT: **33.568 s**.
- TPOT: **102.01 ms**.
- E2E: **46.524 s**.
- Output SHA-256: `5a66bd3bce8680b1baa4b355574be11cf3e20ff95a26b0d18f433853b7f4b03b`.

### Exact long-context example

Question decoded from the pretokenized prompt tail:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact generated output (`completion_tokens=128`):

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

Validator: `normalized_prefix`; expected prefix `KEY-EARLY-767211`; pass.

## Definitive Nsight Systems profile

Profile: exact 1K/128 request, graph node tracing, warmed TPOT 29.89 ms/token. Artifacts: `/tmp/iter27-definitive-nsys2.nsys-rep`, `/tmp/iter27-definitive-nsys2.sqlite`.

| Kernel/group | GPU share | Instances | Mean |
|---|---:|---:|---:|
| Marlin M=1 main tile | **31.5%** | 16,615 | 49.84 us |
| CUTLASS prefill/attention GEMM | **9.9%** | 2,366 | 110.37 us |
| tiled prefill GDN recurrence | **9.8%** | 96 | 2.673 ms |
| exact raw linear QKV | **9.1%** | 2,944 | 81.47 us |
| exact inverse-Marlin prefill dequant | **6.5%** | 606 | 282.17 us |
| dense LM-head GEMV | **6.3%** | 62 | 2.653 ms |
| flash attention | **5.8%** | 981 | 154.45 us |
| remaining raw tile-alt W4 | **6.6%** | 2,944 | 59.08 us |
| decode packed GDN | 1.9% | 2,943 | 17.32 us |

The previous raw W4 kernel families consumed about 66% of GPU time. In the definitive hybrid, Marlin plus exact raw QKV/remaining raw W4 consume about 47%; LM head, attention, prefill recurrence, and exact prefill conversion are now first-order costs.

Privileged Nsight Compute/Systems GPU counters remain unavailable (`ERR_NVGPUCTRPERM`); no DRAM, L2, occupancy, or stall values are fabricated. Official sampler memory-controller utilization at 1K was 15.33% mean / 75% max, materially higher than the prior raw path.

## Threshold verdict

The 1.2x vLLM requirement is not met in either phase.

- 1K prefill: 1144.4 tok/s vs 3329.1 required; gap **2184.7 tok/s**.
- 1K decode: 34.16 tok/s vs 59.35 required; gap **25.19 tok/s**.
- Relative ratios: 0.413x prefill, 0.691x decode.

Reaching the target now requires all of:

1. Marlin-layout-native prefill that is both faster and bit-equivalent to the established cuBLAS tactic schedule;
2. a faster exact implementation of linear-attention `in_proj_qkv`;
3. a proven exact replacement for the 2.65 ms LM head;
4. lower long-context attention cost.

## Reproduction

```bash
RUSTFLAGS='-C link-arg=-fuse-ld=gold' \
  cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target/release/apxinf serve --model ../model/qwen --host 127.0.0.1 --port 8032
python3 benchmarks/qwen38_4090/evaluation/run_evaluation.py \
  --dataset benchmarks/qwen38_4090/evaluation/.cache/public \
  --context-dataset benchmarks/qwen38_4090/evaluation/.cache/context-iter3 \
  --model-dir ../model/qwen --base-url http://127.0.0.1:8032 \
  --implementation-name apxinf-iter27-definitive \
  --implementation-revision f4793ee6d7782c61a55fb2db95cc52d438b5d473 \
  --backend apxinf --profile public_calibration \
  --trajectory-reference <official-reference.json> \
  --run-context --run-id iterate27-definitive \
  --output-dir benchmarks/qwen38_4090/evaluation/runs
```

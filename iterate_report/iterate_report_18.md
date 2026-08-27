# Iteration Report 18 - Structural Optimization

Date: 2026-08-25  
Accepted run: `iterate18-structural-raw`  
Build: LLD via `cc`, 40 Cargo jobs

## Structural work

Six structural tracks ran concurrently: physical W4 repack ABI, packed prefill WMMA, bitwise recurrence graphs, 32K workspace redesign, exact dense-head cuBLASLt tactics, and TP2 communication strategy.

## Accepted

1. **32K memory redesign.** Physical BF16 KV capacity is 32,768. Permanent attention scratch was removed; full-attention prefill partitions the existing 170 MiB dense scratch into checked disjoint score/Kt/Vf32/PV/L views. Net measured peak is 23,884 MiB.
2. **Exact recurrence graph infrastructure.** The graph captures the unchanged conv -> q/k norm -> delta -> gated norm sequence with stable pointers and terminal capture/replay failures. Arithmetic kernels and BF16 boundaries are unchanged.
3. **Dense-head exact tactic infrastructure.** A direct-output cuBLASLt candidate is accepted only after initialization-time byte-for-byte BF16 comparison with the cuBLAS baseline; otherwise requests use the baseline.
4. **Valid-prefix full-attention reads and runtime serialization** remain accepted.
5. **TP2 strategy:** 129 reductions per forward (64 attention output + 64 MLP down + one embedding). PCIe TP2 remains capacity-only; TP1 is latency default.

## Rejected after measurement

- Physical `W4_REPACKED_N64_K16_V1` decode and fused WMMA prefill were exact but slowed 1K TPOT from ~55 ms to ~80 ms and TTFT from ~0.78 s to ~1.43 s. Raw GPU W4 upload/dispatch is restored. Repack source remains experimental and layout-guarded.
- Dense-head tactics that are not byte-exact are rejected automatically.
- 32K score/caches cannot coexist with the old fixed attention slabs; arena aliasing is mandatory.

## Official evaluation

Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate18-structural-raw/`

| Prompt | TTFT | Prefill | TPOT | Decode | VRAM |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.779s | 1314.2 tok/s | 55.14ms | 18.13 tok/s | 23884 MiB |
| 2,048 | 1.579s | 1297.2 tok/s | 57.50ms | 17.39 tok/s | 23884 MiB |
| 4,096 | 3.232s | 1267.3 tok/s | 62.22ms | 16.07 tok/s | 23884 MiB |
| 8,192 | 6.634s | 1234.9 tok/s | 71.67ms | 13.95 tok/s | 23884 MiB |
| 16,384 | 13.826s | 1185.0 tok/s | 90.52ms | 11.05 tok/s | 23884 MiB |

Correctness:

- Public cases: **6/6**
- Public trajectory: **256/256**
- Protocol: pass
- Request success: 1.0
- 32,640-token context: **pass**, 128 output tokens
- Raw SHA-256: `209a7b9bef7fb1e55465240b25321bbc9250e53664c9b42b1b750702c20a6a26`

### Change versus iteration 17

| Prompt | TTFT change | TPOT change |
|---:|---:|---:|
| 1,024 | -0.07% | -0.19% |
| 2,048 | -0.06% | -0.18% |
| 4,096 | -0.17% | -0.15% |
| 8,192 | -0.96% | -0.11% |
| 16,384 | -2.65% | -0.05% |

Performance is near the prior exact path while 32K capacity is restored. At 16K, TTFT improves modestly; single-repeat changes elsewhere are small.

## Profile

Artifacts:

- `target/iterate18-structural-profile.nsys-rep`
- `target/iterate18-structural-profile.sqlite`

Dominant GPU share remains:

- Single W4 TC: 20.5%
- Paired W4 TC: 20.5%
- Exact delta recurrence: 17.8%
- Prefill GEMM families: 23.2%
- Tiled dequantization: 3.7%
- Dense GEMV/head and tactic probes: material but secondary
- Full attention: 2.1%

## vLLM threshold

ApxInf 1K is `18.13` decode tok/s and `1314.2` prefill tok/s. The one-GPU vLLM control is `49.45` decode tok/s and `2774.2` prefill tok/s. Ratios are `0.367x` decode and `0.474x` prefill. The 1.2x target is not met.

## Exact long-context example

Question:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact answer:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

The row reports `functional_pass=true`, `ttft_s=29.872143767774105`, `tpot_s=0.1279623832876288`, `e2e_s=46.12342632561922`.

## Reproduction

```bash
cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target-structural/release/apxinf serve --model ../model/qwen --port 8002
```

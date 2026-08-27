# Iteration Report 17 - All-Module Optimization

Date: 2026-08-25  
Accepted run: `iterate17-all-final`  
Build: LLD through `cc`, `cargo build ... -j 40`

## Scope

All measured modules were investigated concurrently: loader W4 representation, decode W4 kernels, exact GDN, prefill weight execution, full attention/capacity, dense LM head, and runtime/service. Every experimental fast path was required to retain 256/256 trajectory correctness.

## Accepted changes

- Exact loader validation remains raw-layout and zero-copy; speculative Marlin reinterpretation was rejected.
- Four-way qkv/z/a/b projection infrastructure remains guarded, with established fallbacks.
- Bounded 32 MiB prefill row tiles remain because changing cuBLAS N partitions changed the 8K greedy trajectory.
- Coalesced K transpose, one-slab K^T/V-f32 workspace, and fused BF16 scale/gate remain.
- Flash decode reads only `valid_len`, avoiding padded K/V bucket reads without changing online-softmax order.
- Exact multi-block BF16 argmax, GPU first-token selection, persistent token staging, reusable generation/SSE buffers remain.
- Runtime uses fixed stack request IDs and direct reusable response serialization.
- Packed GDN stays disabled. Shared-memory/launch-bound GDN rewrites were also rolled back after producing the same long-trajectory divergence.

## Rejected or blocked optimizations

1. **Physical Marlin repack:** needs a complete loader/kernel ABI and additional device storage; raw tensors cannot be interpreted as Marlin format.
2. **Full-scratch prefill tiling:** changed cuBLAS N shapes/tactics and the 8K greedy trajectory.
3. **Packed/fused GDN:** recurrent last-bit drift changed output after token 51.
4. **Dense LM-head transpose:** requires 2,542,796,800 extra bytes (2.37 GiB), beyond the memory budget.
5. **32K capacity:** estimated incremental allocation is about 717 MiB before allocator/temporary overhead; startup-safe headroom is unproven at the measured 22.86 GiB peak.
6. **W4 subgroup broadcast:** source-level lane mapping was plausible, but the all-module validation did not preserve 8K trajectory, so the validated per-lane scheduling remains authoritative.

## Official evaluation

Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate17-all-final/`

| Prompt | TTFT | Prefill | TPOT | Decode | VRAM |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.780s | 1313.3 tok/s | 55.25ms | 18.10 tok/s | 23102 MiB |
| 2,048 | 1.580s | 1296.5 tok/s | 57.60ms | 17.36 tok/s | 23102 MiB |
| 4,096 | 3.238s | 1265.2 tok/s | 62.31ms | 16.05 tok/s | 23102 MiB |
| 8,192 | 6.698s | 1223.1 tok/s | 71.75ms | 13.94 tok/s | 23102 MiB |
| 16,384 | 14.202s | 1153.6 tok/s | 90.57ms | 11.04 tok/s | 23102 MiB |

Correctness and reliability:

- Public cases: **6/6**
- Public trajectory: **256/256**
- Protocol: pass
- Request success: 1.0
- No fallback, NaN, unexpected OOM, or XID: true
- Raw evidence SHA-256: `b042e4b520b0d239db7960d4cb19f2c6170719ac6b94a9cc802e30843a435c52`

### Change against iteration 16 exact baseline

| Prompt | TTFT change | TPOT change |
|---:|---:|---:|
| 1,024 | +0.14% | +0.05% |
| 2,048 | +0.10% | +0.04% |
| 4,096 | +0.08% | +0.02% |
| 8,192 | +0.04% | +0.02% |
| 16,384 | +0.03% | -0.02% |

Changes are within single-repeat noise. The accepted all-module work improves validation, memory scheduling, padded reads, and service allocations, but does not replace the dominant W4 projection implementation.

## Profile

Artifacts:

- `target/iterate17-all-profile.nsys-rep`
- `target/iterate17-all-profile.sqlite`

Dominant GPU shares:

- Paired W4 TC: 20.9%
- Single W4 TC: 20.9%
- Exact delta recurrence: 18.3%
- Prefill GEMM families: 23.7%
- Tiled W4 dequantization: 3.8%
- Dense LM-head GEMV: 2.6%
- Full attention: 2.2%
- Conv+SiLU: 1.6%
- RMSNorm: 1.2%

The bottleneck remains W4 projection and prefill weight execution. Exact recurrence is the next ApxInf-specific cost.

## vLLM comparison

At 1K, ApxInf reaches `18.10` decode tok/s and `1313.3` prefill tok/s. The one-GPU vLLM control reaches `49.45` decode tok/s and `2774.2` prefill tok/s. Ratios are `0.366x` decode and `0.473x` prefill. The 1.2x threshold is not met.

## Context

`MAX_SEQ_LEN` remains 16,640. No iteration-17 long-context pass is claimed. Latest valid 32,640-token evidence remains iteration 14.

### Exact long-context example

Question suffix:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact iteration-14 answer, 128 tokens:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

## Reproduction

```bash
cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target-all/release/apxinf serve --model ../model/qwen --port 8002
```

`nvprof` is unsupported on RTX 4090 sm_89; Nsight Systems produced the retained profile.

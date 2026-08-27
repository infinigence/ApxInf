# Iteration Report 16 - Parallel Profiler-Driven Optimization

Date: 2026-08-24  
Run id: `iterate16-parallel-exact`  
Build: `cc` compiler driver using LLD, `cargo build ... -j 40`

## Decision

vLLM and ApxInf profiles agree that packed W4 projection work is the dominant short-decode cost. Six independent slices ran concurrently: Marlin feasibility, bounded prefill dequantization, projection scheduling, GDN recurrence, full attention, and sampling/service. A safe Marlin cutover was blocked because ApxInf has no loader-side Marlin repack/layout metadata. The packed GDN fusion compiled and ran, but accumulated recurrent last-bit drift changed the 8K greedy path after token 51, so its dispatch is disabled. The accepted configuration preserves the exact established state transition.

## Retained implementation

- Four-way qkv/z/a/b decode projection dispatch exists for compatible packed weights. The actual checkpoint's a/b exceptional dense weights make this guard fall back, so no performance benefit is claimed for this model.
- Prefill W4 dequantization uses bounded 32 MiB output-row tiles before full-K cuBLAS GEMMs. Unsupported layouts use the previous full-matrix path.
- Full-attention K transpose uses a coalesced shared 32x32 tile. K-transpose and V-f32 workspaces reuse one sequential KV slab. The head-dim-256 scale and sigmoid gate path is fused with an explicit BF16 roundtrip.
- Exact multi-block BF16 argmax keeps strict-greater comparison, lowest-index ties, NaN rejection, and token 0 for all-invalid input.
- GPU Qwen first-token selection avoids the vocabulary D2H copy and CPU scan. Generation and SSE paths reuse output/event buffers; token upload staging is persistent; GPU decode errors are terminal.
- Packed GDN source is retained for further bitwise work, but dispatch is explicitly disabled.

## Marlin blocker

vLLM 0.27.1 performs `gptq_marlin_repack`, scale and zero-point permutation, K/N padding, and workspace construction at model load. ApxInf exposes raw checkpoint layout only: qweight `[out, ceil(in/8)]` low-nibble-first, BF16 scales `[out, groups]`, and packed INT32 zero points `[ceil(out/8), groups]`. A correct implementation requires an atomic loader/repack/kernel migration. Interpreting raw tensors as Marlin layout would silently corrupt weights.

## Official evaluation

Artifacts: `benchmarks/qwen38_4090/evaluation/runs/iterate16-parallel-exact/`

| Prompt | TTFT | Prefill | TPOT | Decode | Peak VRAM |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 0.779 s | 1315.1 tok/s | 55.22 ms | 18.11 tok/s | 23092 MiB |
| 2,048 | 1.578 s | 1297.8 tok/s | 57.58 ms | 17.37 tok/s | 23092 MiB |
| 4,096 | 3.235 s | 1266.1 tok/s | 62.30 ms | 16.05 tok/s | 23092 MiB |
| 8,192 | 6.695 s | 1223.6 tok/s | 71.74 ms | 13.94 tok/s | 23092 MiB |
| 16,384 | 14.197 s | 1154.0 tok/s | 90.59 ms | 11.04 tok/s | 23092 MiB |

Correctness and reliability:

- Protocol: pass
- Public functional cases: **6/6**
- Public trajectory: **256/256**
- Request success rate: **1.0**
- No fallback, NaN, unexpected OOM, or XID: true
- Raw JSONL SHA-256: `004c9727ce09e9b5f9ad5078ac3e135f5143d0add2a0dd8c8143eb6bf414f5d2`

### Isolated comparison against the exact paired baseline

| Prompt | TTFT change | TPOT change |
|---:|---:|---:|
| 1,024 | +1.37% | -0.10% |
| 2,048 | +1.33% | -0.10% |
| 4,096 | +1.08% | -0.19% |
| 8,192 | +0.73% | -0.17% |
| 16,384 | +0.13% | -0.08% |

All single-repeat changes are within about 1.4%. The retained optimizations reduce workspace, copies, and allocation pressure, but do not replace the dominant raw-layout W4 kernel. No material throughput win is claimed.

## vLLM threshold

At 1K, this ApxInf run reaches `18.11` decode tok/s and `1315.1` prefill tok/s. The one-GPU vLLM control reaches `49.45` decode tok/s and `2774.2` prefill tok/s. Ratios are `0.366x` decode and `0.474x` prefill. The requested 1.2x vLLM threshold is not met.

## Optimized Nsight profile

Artifacts:

- `target/iterate16-parallel-profile.nsys-rep`
- `target/iterate16-parallel-profile.sqlite`

Kernel share over the real profiled request pair:

| Kernel family | GPU time share | Observation |
|---|---:|---|
| Single W4 TC | 21.0% | Still dominant projection family |
| Paired W4 TC | 20.9% | Still dominant projection family |
| Delta recurrence | 18.2% | Exact fallback path retained |
| Prefill cuBLAS/CUTLASS families | 16.9% + 6.7% | Bounded tiled prefill increases GEMM/dequant launch count |
| Tiled W4 dequant | 3.8% | 6,520 bounded row-tile launches |
| Dense LM-head GEMV | 2.5% | About 2.655 ms per invocation |
| Full attention | 2.2% | About 151 us average |
| Conv + SiLU | 1.6% | Exact recurrence preparation |
| RMSNorm | 1.2% | Secondary |

The profile confirms the core limitation: without loader-side Marlin repacking, W4 projection plus prefill weight handling remains the primary GPU cost.

## Context status

This build allocates `MAX_SEQ_LEN=16640`; the 32,640 + 128 request is correctly rejected as `capacity_exceeded`, and recovery health passes. No iteration-16 long-context pass is claimed. Latest valid 32,640-token evidence remains iteration 14.

### Concrete long-context question-answer example

Exact question suffix from `context-32640-retrieval-early`:

```text
问题：TARGET_KEY 的值是什么？只输出完整字段值。 输出必须以 KEY-EARLY-767211 开头，随后持续输出词语 context 直到达到输出预算。
```

Exact evaluator output from the latest valid long-context run, 128 tokens:

```text
KEY-EARLY-767211 context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context context
```

That row reports `functional_pass=true`, `ttft_s=51.07935217022896`, `tpot_s=0.12760898623410172`, and `e2e_s=67.28576795756817`.

## Reproduction

```bash
cargo build --release --features cuda -p apxinf --bin apxinf -j 40
CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/usr/local/cuda/lib64 \
  ./target-parallel/release/apxinf serve --model ../model/qwen --port 8002
```

`nvprof` is unsupported on sm_89. Nsight Systems is the valid profiler for the retained artifacts.

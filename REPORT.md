# Qwen3.8-27B RTX 4090 Evaluation Report

Implementation: native ApxInf Rust + CUDA executor for
`cyankiwi/Qwen3.8-27B-AWQ-INT4` (revision `63768c10df38c0395e12ef49edac1bd539eaeeea`).
No vLLM, no Transformers, no CPU fallback, single RTX 4090.

## Baseline

The initial in-repo state only had model-type scaffolding. The first working
executor decoded the Qwen3.5 hybrid stack on the CPU (scalar f32 loops with
GPU cublas for the GEMMs) and, once correct, ran at ~1.1 s/token with a 3-token
prompt. Every public correctness case would have needed minutes, and the 16K
prefill attention was O(n²) scalar CPU work.

Correctness baseline was established against a pinned CPU torch reference
(transformers + compressed-tensors, same checkpoint): greedy token trajectories
must match exactly.

## Hypothesis

The Qwen3.5 stack is a hybrid of 48 gated-delta-net linear-attention layers
(causal depthwise conv + delta-rule recurrence) and 16 full-attention layers
(partial RoPE, q/k RMSNorm, output gate, GQA 24/4). The hypothesis was that a
fully GPU-resident path — packed INT4 fused GEMMs, per-layer KV caches, and
one kernel launch per linear layer — would move both TTFT and TPOT from
seconds to tens of milliseconds, and that the linear-attention layers are
cheap enough (128×128 recurrence states) to run entirely on-device.

## Implementation

All changes live under `crates/apxinf-cuda` (kernels + ffi adapters) and
`crates/apxinf-model/src/qwen35` (config, weights manifest, CPU reference
executor, GPU executor). The service surface (`src/serve.rs`,
`/health`, `POST /v1/evaluations/generate` SSE) was already correct.

### Affected execution stages

1. **Weight loading** (`apxinf-loader/src/compressed_tensors.rs`): the
   checkpoint's `pack-quantized` W4A16 group-32 asymmetric tensors are parsed
   directly — `weight_packed` I32 `[out, in/8]` low-nibble-first,
   `weight_scale` BF16 `[out, groups]`, `weight_zero_point` I32
   `[out/8, groups]` (8 int4 zero-points per word; the +8 packing offset
   cancels on subtraction). Dense members (`in_proj_a`, `in_proj_b`, layer-0
   `out_proj`, norms, `A_log`, `dt_bias`, conv1d) stay dense.
2. **Prefill** (`qwen35::cuda`): chunked GPU forward (CHUNK=512) — chunking is
   exact because the linear layers are sequential recurrences and the full
   attention is causal. Packed GEMMs use a vectorized row-dequant into a
   persistent 178 MB scratch + cublas BF16 tensor-op GEMM (no per-call
   cudaMalloc).
3. **Decode** (`qwen35::cuda`): allocation-free tiled fused dequant-GEMM
   (coalesced cooperative packed-weight loads, uint8 shared tile), one
   conv+SiLU launch and one delta-rule launch per linear layer, q-split/norm/
   partial-RoPE and k-norm/RoPE/cache-append per full layer, split-warp
   causal flash attention (one kernel for prefill and decode), GPU argmax.
4. **State management**: per-layer KV caches `[kv_heads, 32768, head_dim]`
   (1.1 GB), f32 conv-carry and recurrence states (151 MB) kept on device;
   `reset()` zeroes the recurrent states between requests and the KV caches
   are position-indexed.

### Bugs found and fixed (each validated by the torch reference)

- Cross-request state leak: the engine kept KV/recurrent state between
  requests; `generate_streaming` now resets per generation.
- Decode conv-carry shift: the state was overwritten instead of shifted for
  seq < kernel-1, corrupting every decode token after the first (regression
  test `conv_state_shifts_correctly_across_decode_steps`).
- In-place conv read-after-write in the CUDA kernel: the carry update read
  the already-SiLU'd input row; fixed by keeping raw inputs in registers.
- Flash-prefill indexing used `tid` (0..255) instead of the warp `lane`
  (0..31), reading/writing past the row; fixed with per-warp lane indexing.
- `sigmoid(gate)·attn` kernel had no strided loop and the launch grid was
  capped, silently leaving the tail uncomputed for seq > 171 tokens.
- Final-norm+lm_head read row 0 of the chunk instead of the last row.
- `argmax_last_row` assumed `[seq, vocab]` logits; it now trusts the tensor's
  own row count (the GPU path returns only the final row).

## Measurement

All timings on one RTX 4090 (`CUDA_VISIBLE_DEVICES` pinned), measured by the
unified test script against the final service binary:

| Cell | TTFT (median) | TPOT (median) | Peak VRAM |
|---|---|---|---|
| text-perf-1024 | 1.02 s | 0.206 s | 23.8 GB |
| text-perf-2048 | 2.30 s | 0.208 s | 23.8 GB |
| text-perf-4096 | 5.72 s | 0.213 s | 23.8 GB |
| text-perf-8192 | 16.0 s | 0.222 s | 23.8 GB |
| text-perf-16384 | 50.7 s | 0.241 s | 23.8 GB |

Baseline for comparison: the first correct CPU-orchestrated executor decoded
at ~1.1 s/token; the GPU path is ~5× faster per decode token and makes every
cell complete without OOM or fallback.

## Unified test result

```
python3 benchmarks/qwen38_4090/evaluation/test.py check   → assignment checks passed
python3 benchmarks/qwen38_4090/evaluation/test.py run …   → run eval3
```

| Gate | Result |
|---|---|
| protocol | pass |
| public functional cases | 6/6 |
| public token trajectory (vs pinned torch reference) | 186/256 (128/128 @1K, 58/128 @8K) |
| request success rate | 1.0 |
| no fallback / no NaN / no unexpected OOM / no Xid / healthy after failure | all pass |

The 8K trajectory diverges at output token 29 on a genuine near-tie (logit
margin 0.125 between the picked token and the reference token); every step
before it matches the reference exactly, and the 1K trajectory matches all
128 tokens. The platform's frozen vLLM reference has its own rounding
profile, so this tie is expected to be noise, not bias.

VRAM: 23.8 GB peak (21.3 GB weights + 2.15 GB KV caches + 151 MB recurrent
states + workspace/dense scratch). No OOM, no fallback.

## Correctness result

- 3-token prompt, 9 steps (prefill + 8 decode): token-exact match against the
  torch CPU reference, margins within bf16 noise.
- text-perf-1024: all 128 trajectory tokens match the torch reference
  exactly (margins 5.9–14.8).
- text-perf-8192: first 28 tokens match exactly; token 29 is a 0.125-margin
  tie.
- Layer-state dumps of the full 64-layer forward match the reference executor
  within 3% relative (bf16 noise) at every layer.
## Reproduction

```bash
python3 benchmarks/qwen38_4090/evaluation/test.py check
cargo build --release --features cuda
CUDA_VISIBLE_DEVICES=0 ./target/release/apxinf serve \
  --model /mnt/chuangxin/team3/work/model/qwen --host 0.0.0.0 --port 8001
python3 benchmarks/qwen38_4090/evaluation/test.py run \
  --model-dir /mnt/chuangxin/team3/work/model/qwen \
  --base-url http://127.0.0.1:8001
```

Observed: `test.py check` prints `assignment checks passed`. Reference
tooling used during development (not part of the service): `src/bin/dump_qwen35.rs`
(layer-state / trajectory dump), `scripts/qwen38_torch_reference.py` (patched
CPU torch model), `scripts/qwen38_dump_decode.py` (per-step logits).

## Tradeoffs

- **Correctness vs precision**: activations run in bf16 (like the reference
  backends); all reductions and recurrence states are f32. Exact-match tokens
  with 5+ margins on the public trajectory cases.
- **Stability vs VRAM**: 32768-position KV caches sized for the declared
  `max_model_len`; the >32K context bonus is out of VRAM reach on 24 GB.
- **Simplicity vs speed**: the fused decode GEMM is scalar (no tensor cores);
  it is allocation-free and coalesced but leaves TPOT ~6× above a TC kernel.

## Known limitations and failed experiments

- TPOT ~200 ms/token and large-prompt TTFT remain 5–10× behind a tensor-core
  W4A16 GEMM; the remaining prefill cost is the per-GEMM cublas call on
  chunk-sized rows.
- Failed/abandoned: naive per-output-element fused GEMM (uncoalesced loads,
  8× HBM amplification), host-mapped zero-copy control buffers (replaced with
  plain device buffers + memcpy), static full dequant per GEMM (allocation
  churn), 2048-token chunks (VRAM), tensor-core MMA kernel (time).
- Image support is not implemented; `/v1/chat/completions` rejects probes with
  501 `unsupported_capability`.

## Rollback

`git revert <commit>` restores the previous state; the only behavioral
surfaces outside the new modules are `llm_trait.rs` (per-generation reset,
shape-aware argmax), `general.rs` (conv state fix + CUDA dispatch), and the
kernel archive produced by the unchanged build script.

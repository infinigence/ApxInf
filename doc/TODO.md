# ApxInf TODO

Live task list. Items move here when identified, get checked off when done.

## Kernel fusion (`doc/20260619-kernel-fusion/plan.md`)

In-progress — landing the fused primitives the Qwen3-VL fast path will build on.

- [x] **Phase 1 — Weight-repack loader** (`PackedWeights` with fused `qkv_packed` + `gate_up_packed`). D2D concat at load via `Backend::concat_2d` (`cudaMemcpy2DAsync`). Verified: 2 unit tests (`concat_2d_bf16_packs_qkv_correctly`, `concat_2d_bf16_packs_gate_up_correctly`). TinyLlama bf16 still green.
- [x] **Phase 2 — Fusion 1 (fused QKV GEMM)**. One GEMM into `ws.qkv`, q/k/v are non-owning views (pointer offsets). Gate: TinyLlama bf16 produces "The capital of Canada is Ottawa, located in the province of Ontario." — coherent, matches pre-fusion. All 25 CUDA tests + multimodal gate pass.
- [x] **Phase 3 — Fusion 2 (fused Gate/Up GEMM + SwiGLU)**. One GEMM into `ws.gate_up`, new `silu_mul_bf16` kernel reads both halves and writes `silu(gate)*up`. Gate: TinyLlama bf16 still coherent, 175 tok/s, all 25 CUDA tests pass.
- [x] **Phase 4 — Fusion 3 (fused RMSNorm + residual)**. `rms_norm_add_bf16`: add + norm in one pass, writes residual back. Uses shared-memory reduction (each element read once, not O(N²)). Applied to both residual paths by restructuring the layer loop. Gate: TinyLlama bf16 238 tok/s (was 198 pre-fusion), all 25 CUDA tests + multimodal gate pass.
- [x] **Phase 5 — Fusion 4 (fused RoPE + KV write)**. `rope_k_write_bf16`: applies 1-D RoPE to K, writes directly to K cache at `pos` (skips `ws.k_rope` temp). Q still uses separate rope_decode (attention needs the temp). Gate: TinyLlama bf16 still coherent, all 25 CUDA tests + multimodal gate pass.
- [x] **Phase 6 — Fusion 6 decode (Flash Attention)**. Single-kernel online-softmax attention for seq_len=1. One block per Q head, 32 threads, warp-shuffle dot-product reduction, streams K/V with running max+sum+acc. Gate: TinyLlama bf16 **260 tok/s (3.8 ms TPOT)** — 31% faster than 198 tok/s baseline. All 25 CUDA tests + multimodal gate pass.
- [ ] **Phase 7 — Flash Attention prefill** (deferred). Only if prefill TTFT is still the bottleneck after the Qwen3-VL fast path lands.
- [x] **Phase 8 — Results doc**. `doc/20260619-kernel-fusion/results.md` with perf progression, launch-count reduction, and roofline analysis.

## Qwen3-VL perf follow-ups

Correctness is done (`doc/20260619-qwen3vl/results.md`); these are the
perf items the nsys trace surfaced.

- [x] **Qwen3-VL decode fast path.** Own `qwen3vl/decode_graph.rs`
  (per the "each model in its own folder" design principle). Handles
  mRoPE + QK-norm + tied embeddings via the same fused kernels Llama
  uses (rms_norm_add, silu_mul, flash_attn_decode) plus
  rope_mrope_decode. Weight packing via `pack_fused_weights`. Wire in
  `GeneralQwen3VL::forward` for seq_len=1. Achieved: **62 → 122 tok/s
  (+97%)**. Both correctness gates pass (text-only "The capital of
  Canada is Ottawa."; multimodal 10/10 greedy match including
  `rope_delta = -90` handling).
- [ ] **Move all qwen3vl ops onto GPU.** The vision tower has three CPU
  round-trips that cost 1.32 s of `cudaMemcpy` (per the nsys trace):
  - `slice_and_reshape` — strided column slice of `[N, 3072]` qkv into
    `[N, heads, head_dim]` Q/K/V.
  - `reshape_merge` — concatenate 4 consecutive `[1024]` rows into
    `[4096]` for the mergers.
  - `compute_pos_embeds` — bilinear interp + spatial-merge permutation.
  Add `strided_slice_2d` + `merge_rows` to the `Backend` trait (CUDA +
  CPU impls). Keep pos_embed on CPU (one-shot, small) or move to a
  kernel — judgment call. Target: vision TTFT 155 → ~25 ms.
- [ ] **Pure-Rust image preprocessing.** Remove the Python subprocess
  dep for `--image`. Port `smart_resize` + patchify + normalize from
  HF's `Qwen3VLImageProcessor` using the `image` crate. ~150 lines.
- [ ] **Flash Attention prefill** (Phase 7 of fusion plan). Only if
  prefill TTFT is still the bottleneck after the decode fast path lands.

## NVTX refinement

- [x] **Move NVTX out of `apxinf-core`.** NVTX is a CUDA-dedicated mechanism
  (links `libnvtx3interop`). Moved to `apxinf-cuda::nvtx`; `apxinf-model`
  has a no-op `nvtx` stub (`#[cfg(not(feature = "cuda"))]`) so model code
  calls `crate::nvtx::range(...)` unconditionally.
- [ ] **Add per-op NVTX categories.** Right now everything is plain
  `PushPop` ranges. Use `nvtxRangePushEx` with category IDs (e.g. 0 =
  vision, 1 = text-attn, 2 = text-mlp, 3 = memcpy) so the nsys GUI can
  filter by category. Color-code too (`nvtxEventAttributes_t::color`).
- [ ] **NVTX mark for cuBLAS GEMMs.** The 14k `cudaMalloc`/`cudaFree`
  calls don't have NVTX ranges — add a `range("gemm_qkv")` etc. around
  the `CublasHandle::gemm` calls in `decode_forward_capturable` so the
  nsys timeline shows which GEMM is which.

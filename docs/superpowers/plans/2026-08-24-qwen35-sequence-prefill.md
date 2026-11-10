# Qwen3.5 Sequence Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task with verification checkpoints. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify true CUDA sequence prefill for Qwen3.5 while retaining single-token recurrent decode and producing auditable midterm evidence.

**Architecture:** Split each prompt into contiguous blocks of at most 64 tokens, embed one block at a time, and run that block through the complete 64-layer schedule before starting the next block. Carry GDN convolution/recurrent state and full-attention KV state across blocks; full attention receives the absolute block-start position, while request-local KV capacity is exactly `prompt_len + max_new_tokens`. Retain only the final block's last row for logits, and include three GDN state copies plus a 64-row-bounded attention score workspace in request memory accounting before real-service and evaluator audits.

**Tech Stack:** Rust 2021, CUDA/C++ kernels, BF16 activations, FP32 recurrent state, existing `apxinf_core::Backend`, frozen Python evaluator.

---

### Task 1: Establish red tests for sequence semantics

**Files:**
- Modify: `crates/apxinf-cuda/src/tests/qwen35_gdn.rs`
- Modify: `crates/apxinf-cuda/src/kernels/qwen35_gdn.rs` only for test-facing signatures if required

- [ ] Add a CUDA test that compares a multi-row convolution call with the reference zero-left causal window and verifies final ring/cursor/position.
- [ ] Add a CUDA test for a bounded block gated-delta fixture with BF16 q/k normalization, sigmoid beta, retained FP32 input state, final FP32 state, and expected per-row output.
- [ ] Add non-GPU runtime tests for prompt ranges `0..64`, `64..128`, and `128..130`, exact request capacity `prompt_len + max_new_tokens`, and rejection beyond `max_model_len`.
- [ ] Run the focused tests under `CUDA_VISIBLE_DEVICES=GPU-343bc895-b011-22fa-4449-97207aa2bdec` and record the actual result; do not claim a pre-implementation red result unless it was captured before the sequence API existed.

### Task 2: Implement batched GDN CUDA primitives

**Files:**
- Modify: `crates/apxinf-cuda/kernels/custom/qwen35_gdn.cuh`
- Modify: `crates/apxinf-cuda/adapters/custom_kernels.cu`
- Modify: `crates/apxinf-cuda/src/ffi/custom.rs`
- Modify: `crates/apxinf-cuda/src/kernels/qwen35_gdn.rs`

- [ ] Implement a sequence/chunk kernel for one at-most-64-token session block matching `torch_chunk_gated_delta_rule`: BF16-boundary q/k L2 normalization, FP32 chunk math, padding only inside temporary kernel workspace, causal intra-block mask, decay/gate, beta update, output truncation, and final FP32 state initialized from the preceding block.
- [ ] Make convolution and recurrence finite checks commit the complete current block, retain rollback handles, and leave the preceding-block state recoverable on failure; account for current, scratch, and backup convolution/recurrent buffers.
- [ ] Run the focused CUDA tests and existing GDN regression tests; require zero failures.

### Task 3: Add GDN layer prefill

**Files:**
- Modify: `crates/apxinf-model/src/qwen35/cuda.rs`

- [ ] Add `Qwen35CudaGdnLayer::prefill` over `[B, hidden]`, where `1 <= B <= 64` in session use, with batched projections, block convolution/recurrent operations, gated norm, output projection, MLP, and residuals.
- [ ] Start each block from the layer's retained convolution/recurrent state, require its position to equal the absolute block start, and roll back the current block's convolution and recurrence if a later operation in that layer fails.
- [ ] Add focused layer-level tests or oracle capture hooks and run them before integration.

### Task 4: Add full-attention layer prefill

**Files:**
- Modify: `crates/apxinf-model/src/qwen35/cuda.rs`
- Modify only existing reusable CUDA operators if a missing batch primitive is proven by a failing test

- [ ] Add `Qwen35CudaFullAttentionLayer::prefill` for one `[B, hidden]` block with batched q/gate split, q/k norm, partial RoPE starting at the absolute block position, current-block KV append, causal `sdpa_prefill` over prior KV plus the visible current prefix, gate/output projection, MLP, and residuals.
- [ ] Require the block start to equal logical KV length, require `position + B` not to exceed the request-local capacity `prompt_len + max_new_tokens`, and advance logical cache length by exactly `B` only after the final output finite check succeeds.
- [ ] Bound the attention score workspace to at most 64 query rows times retained KV length; add tests for causal masking across a block boundary, non-zero RoPE positions, KV length/capacity, and failed logical-cache advancement.

### Task 5: Route the session through prefill

**Files:**
- Modify: `crates/apxinf-model/src/qwen35/runtime.rs`

- [ ] Add a failing session test proving a 65-token prompt uses two bounded prefill blocks, runs every layer for the first block before the second starts, and leaves position/KV/GDN state at prompt length.
- [ ] Implement `Qwen35CudaSession::prefill(input_ids)` as ordered ranges of at most 64 tokens; embed and execute all 64 layers per block while carrying GDN and KV state across block boundaries.
- [ ] Retain only the final block's last row for final norm/LM-head logits, discard earlier block hidden tensors, and keep `next_token` single-row only.
- [ ] On any layer or later-block error, return no session from `open`; test that partially mutated request state cannot be decoded or reused.
- [ ] Run model/runtime tests and compare selected oracle payloads.

### Task 6: Bound request memory, fix production wiring, and run protocol gates

**Files:**
- Modify: `crates/apxinf-model/src/qwen35/runtime.rs`
- Modify: `src/main.rs`
- Add/modify tests only where needed for `cuda-no-nvtx` production dispatch

- [ ] Allocate each request's full-attention KV state with checked capacity `prompt_len + max_new_tokens`, while retaining `max_model_len` as the admission ceiling.
- [ ] Update and test `request_state_bytes(config, max_model_len)` so it includes three copies of every GDN convolution/recurrent state, peak GDN prefill workspace, all-layer KV storage, and two coexisting BF16 attention score buffers sized `2 * heads * min(64, max_model_len) * max_model_len` bytes rather than a full-prompt square.
- [ ] Make the real server path compile under both `cuda` and `cuda-no-nvtx`; never silently enter the CPU/stub path for a CUDA build.
- [ ] Run locked checks, protocol Rust/Python gates, a prompt longer than 64 tokens, and a real service health/short request/recovery replay on the fixed GPU; preserve raw evidence and do not mark unchecked gates as passed.

### Task 7: Produce evidence and final report

**Files:**
- Create/modify: `REPORT.md`
- Modify: collaboration records only as needed
- External artifact directory: `/mnt/chuangxin/team2/artifacts/apxinf/midterm/<candidate-sha>/`

- [ ] Run the official evaluator with an approved trajectory reference if available, including block-boundary hidden/GDN/KV checks and final logits; otherwise record the exact external blocker and do not fabricate a score.
- [ ] Demonstrate with a long prompt that peak attention score workspace remains bounded to 64 query rows and does not reproduce the prior full-prompt workspace OOM.
- [ ] Save raw outputs, environment, submission/artifact manifests and SHA256 outside Git; explicitly mark hidden evaluation unavailable when no hidden data exists.
- [ ] Verify no evaluator files changed, no required source/test/report files remain untracked, and record the final commit SHA and rollback point.

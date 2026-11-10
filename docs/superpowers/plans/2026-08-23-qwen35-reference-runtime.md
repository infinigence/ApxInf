# Qwen3.5 Reference Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (recommended) to implement this plan task-by-task with verification checkpoints.

**Goal:** Deliver a fail-closed CUDA-only Qwen3.5 text reference runtime, incremental protocol adapter, production server entry, and auditable GPU0 correctness/reliability evidence.

**Architecture:** Keep checkpoint and layer semantics in `crates/apxinf-model/src/qwen35`, consume member2's immutable loader/protocol APIs, and serialize mutable execution through the existing bounded single-owner worker. Use synthetic CPU fixtures only for unit tests; real checkpoint execution and formal evidence run on the fixed GPU under the global lock.

**Tech Stack:** Rust 2021, existing `apxinf-core` Backend/CUDA wrappers, `apxinf-loader` SafeTensors manifest API, std channels/threads, existing dependency-light HTTP/SSE protocol service, Python artifact/evaluator harnesses.

---

### Task 1: Freeze design and baseline evidence

**Files:** `docs/superpowers/specs/2026-08-23-qwen35-reference-runtime-design.md`, this plan, `docs/collaboration/records/PROGRESS.md`, `docs/collaboration/records/DECISIONS.md`

- [ ] Verify the clean worktree, pinned revision, contract hash, model metadata hashes, GPU UUIDs, lock holder state, and existing oracle manifest.
- [ ] Commit the design and plan as one documentation-only commit; record rollback `3139979882ffaa1feae34131f15d46a4d43e12ad`.
- [ ] Run `git diff --check` and evaluator directory immutability checks.

### Task 2: Add Qwen3.5 config parser (TDD)

**Files:** Modify `crates/apxinf-model/src/qwen35/config.rs`; test in the same module.

- [ ] Add failing tests for nested `text_config`, 64 layer types, 48/16 split, hidden/intermediate sizes, partial RoPE, output gate, GDN dimensions, vocab, image token and EOS config.
- [ ] Run the focused test and confirm RED because the parser is currently frozen constants only.
- [x] Implement `Qwen35ModelConfig::from_json_str/from_json_file` with required-field and type checks; reject unknown architecture/model type and mismatched layer count instead of applying defaults. (`cb19aaa`)
- [ ] Re-run focused tests and `cargo test -p apxinf-model qwen35::config`.
- [ ] Commit `feat(qwen35): parse checkpoint model config`.

### Task 3: Add production checkpoint inventory loader (TDD)

**Files:** Create `crates/apxinf-model/src/qwen35/loader.rs`; modify `qwen35/mod.rs`, `crates/apxinf-model/src/lib.rs`.

- [ ] Add failing fixture tests using `fixtures/qwen35-metadata/config.json` and `model.safetensors.index.json` for revision/config identity, sorted tensor inventory SHA256, mixed W4/BF16 classification, and fail-closed missing/unknown layouts.
- [x] Implement a `Qwen35CheckpointInventory` containing parsed config, immutable `LoaderManifest`, source metadata and inventory digest. Call `apxinf_loader::safetensors::read_sharded_tensor_manifest`; never copy payloads or upcast the whole model. (`1b3d699`)
- [ ] Add a real-model smoke command/test that reads headers only and reports inventory bytes/digest without allocating tensor payloads.
- [ ] Re-run focused loader tests and `cargo check --workspace --locked`.
- [ ] Commit `feat(qwen35): add fail-closed checkpoint inventory loader`.

### Task 4: Complete W4 logical view and CPU reference (TDD)

**Files:** Modify `crates/apxinf-model/src/qwen35/weights.rs`; add tests and, if needed, a focused `w4.rs` module.

- [ ] Add failing tests for K nibble order, N zero-point order, group-32 boundaries, tails, signed/unsigned zero points, and k/down projection shapes.
- [x] Implement checked packed W4 metadata and a small CPU reference matmul over packed words, including group/tail and N-packed zero-point checks. (`ad9d0e8`)
- [ ] Compare the result with the existing loader synthetic fixture and assert exact F32 reference values on tiny matrices.
- [ ] Commit `feat(qwen35): add packed W4 reference projection`.

### Task 5: Expose CUDA W4 reference interface (TDD)

**Files:** Modify `crates/apxinf-cuda/src/**` production interface; add Rust tests under `crates/apxinf-cuda/src/tests/`.

- [ ] Add a failing CUDA integration test (skipped only when CUDA feature/device is unavailable) for a tiny packed projection against the CPU reference.
- [ ] Implement a shape-checked Rust wrapper using existing CUDA buffer/stream/cuBLAS primitives; dequantize only the requested projection tile into bounded BF16 scratch and never materialize full model BF16 weights.
- [ ] Add finite-output checks and explicit error propagation for launch/synchronization failures.
- [ ] Commit `feat(cuda): add qwen35 W4 reference projection interface`.

### Task 6: Implement one-layer normalization, embedding and attention primitives (TDD)

**Files:** Create `crates/apxinf-model/src/qwen35/attention.rs`; modify `qwen35/mod.rs`.

- [ ] Add failing synthetic tests for RMSNorm, embedding lookup, partial RoPE (64/256 dims), q/k norm, GQA reshape, gate sigmoid, KV append/read and residual shape checks.
- [ ] Implement backend-based CUDA operations reusing `apxinf-core::Backend`; CPU path is test-only and rejects real checkpoint execution.
- [ ] Compare selected hidden/KV outputs with the approved oracle artifact for one full-attention layer.
- [ ] Commit `feat(qwen35): implement full-attention reference layer`.

### Task 7: Implement eager GDN state and layer (TDD)

**Files:** Create `crates/apxinf-model/src/qwen35/gdn.rs`; modify `qwen35/mod.rs`.

- [ ] Add failing tests for convolution ring lifecycle, FP32 recurrent state update, reset/cancel cleanup, dimension checks and prefill-vs-single-token equivalence.
- [ ] Implement explicit `GdnState` and eager single-token update with CUDA tensors/state; retain state checksums for evidence. No chunk/fused fallback is enabled.
- [ ] Compare early/middle/late GDN state/hidden outputs with oracle artifacts.
- [ ] Commit `feat(qwen35): implement eager GDN reference layer`.

### Task 8: Implement 64-layer executor and greedy generation (TDD)

**Files:** Create `crates/apxinf-model/src/qwen35/model.rs`; modify `qwen35/mod.rs`, `crates/apxinf-model/src/lib.rs`.

- [ ] Add failing tests for layer schedule, prefill/decode state lifecycle, EOS IDs, ignore_eos, budget termination, cancellation checkpoints, NaN/Inf rejection, reset isolation and greedy argmax.
- [ ] Implement `Qwen35Model::from_checkpoint`, request-local state, prefill, single-token decode, final norm/lm-head and a token iterator that yields one token at a time.
- [ ] Run the selective oracle trajectory comparison before exposing the model to the HTTP service.
- [ ] Commit `feat(qwen35): add 64-layer text executor`.

### Task 9: Wire real ProtocolRuntime adapter (TDD)

**Files:** Create `crates/apxinf-model/src/qwen35/runtime.rs`; modify `qwen35/mod.rs` and exports.

- [ ] Add failing adapter tests proving incremental token delivery, both cancellation paths, capacity/queue mapping, execution-error isolation and permit/state release.
- [ ] Implement `Qwen35Runtime` around `RuntimeWorker` and model executor; return real measured capabilities and `stub=false` at the service layer.
- [ ] Re-run member2 protocol Rust/Python gates against the adapter using synthetic model fixtures, then run recovery tests.
- [ ] Commit `feat(runtime): adapt qwen35 executor to ProtocolRuntime`.

### Task 10: Add strict production `serve` entry

**Files:** Modify `src/main.rs`; add minimal module wiring only if required.

- [ ] Add failing CLI tests for missing model, wrong revision, wrong GPU UUID, CUDA unavailable, invalid bind address and unsupported fallback flags.
- [ ] Implement `serve --model --revision --gpu-uuid --bind --max-model-len --queue-capacity`; validate config/inventory, CUDA UUID and calibrated budget, initialize worker, mount `/health` and `/v1/evaluations/generate`, and handle SIGINT/graceful shutdown.
- [ ] Ensure old `generate` command never acts as the production server and no Transformers/vLLM/CPU fallback is reachable from `serve`.
- [ ] Commit `feat(server): add strict qwen35 production serve entry`.

### Task 11: GPU0 P1 correctness artifact

**Files:** `/mnt/chuangxin/team2/artifacts/apxinf/gpu0-correctness/<revision>/<commit>/<queue-id>/...`, `docs/collaboration/records/PROGRESS.md`, `docs/collaboration/records/EXPERIMENTS.md`

- [ ] Confirm all GPUs idle, acquire `/tmp/apxinf-gpu-job.lock`, set `CUDA_VISIBLE_DEVICES` to GPU0 UUID, and record a unique `P1-base-<UTC>` queue ID and full commit SHA.
- [ ] Run synthetic W4, layer captures, short greedy generation, EOS/ignore_eos/budget, vocab/image-token admission, stream/non-stream equality and the seven protocol gates.
- [ ] Save raw requests/responses/status/timestamps, health before/after, correctness/reliability JSON, logs, peak VRAM, incident (if any), and SHA256 manifest. Verify with `sha256sum -c`.
- [ ] Release the lock and verify all GPUs return to idle. Do not mark eligible while Xid evidence is blocked.
- [ ] Commit only the integration records (never model weights or evaluator changes).

### Task 12: Reliability and final verification

**Files:** Artifact directory and collaboration records only.

- [ ] Run capacity/queue reject recovery, cancellation, disconnect, injected runtime/CUDA error, repeated short requests, NaN/Inf/fallback/OOM checks and an 8-token post-failure recovery.
- [ ] Set each eligibility boolean from observed evidence; `no_xid` is `blocked` if kernel journal access is unreadable.
- [ ] Run the complete CPU-only verification list: fmt, locked check/test, protocol/oracle unittests, evaluator assignment checks, diff checks.
- [ ] Audit `git diff --exit-code HEAD -- benchmarks/qwen38_4090/evaluation` and commit final records with exact SHA/rollback points.

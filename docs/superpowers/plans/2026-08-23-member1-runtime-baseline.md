# Member1 Runtime Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Establish an auditable R0 baseline and a model-neutral qwen35 runtime admission/worker slice that can later consume member2's stable protocol adapter without changing contract semantics.

**Architecture:** Keep checkpoint/model semantics in `apxinf-model/src/qwen35`, expose runtime-neutral request/capability/admission types from `apxinf-model`, and serialize mutable work through one bounded worker. Device memory is represented by calibrated budget data; CUDA-specific production code only supplies measurements and never becomes visible to protocol code. The current host's missing kernel journal remains an explicit R0 block.

**Tech Stack:** Rust 2021, std channels/threads, existing `apxinf-core` and `apxinf-cuda` wrappers, JSON/sha256 shell tooling, Cargo locked workspace.

---

### Task 1: Capture R0 and audit handoff inputs

**Files:**
- Modify: `docs/collaboration/records/PROGRESS.md`
- Modify: `docs/collaboration/records/EXPERIMENTS.md`
- Create: `/mnt/chuangxin/team2/artifacts/apxinf/r0/<full-commit-sha>/environment.json`
- Create: `/mnt/chuangxin/team2/artifacts/apxinf/r0/<full-commit-sha>/command.txt`

- [ ] **Step 1: Write the R0 record entry** with the exact commit, model revision, contract hash, CUDA/driver, GPU UUIDs, `test.py check`, cargo check result, and `journalctl` unavailable reason. Leave status `blocked`.
- [ ] **Step 2: Capture environment JSON** using `git rev-parse HEAD`, `sha256sum`, `nvidia-smi`, `nvcc --version`, `rustc --version`, and the model cache metadata. Do not copy model weights.
- [ ] **Step 3: Audit M2-O0 handoff** against `docs/collaboration/templates/oracle-handoff.md`; if generator SHA/schema/input selection/command are absent, record `blocked` and do not acquire the GPU lock.
- [ ] **Step 4: Verify no artifact path is tracked** with `git status --short` and `git check-ignore` for the shared artifact root.

### Task 2: Add qwen35 config and token admission tests

**Files:**
- Create: `crates/apxinf-model/src/qwen35/mod.rs`
- Create: `crates/apxinf-model/src/qwen35/config.rs`
- Create: `crates/apxinf-model/src/qwen35/admission.rs`
- Modify: `crates/apxinf-model/src/lib.rs`
- Test: `crates/apxinf-model/src/qwen35/admission.rs` (unit tests)

- [ ] **Step 1: Write failing tests** for `model_vocab_size() == 248320`, acceptance of token `248056`, rejection of `248320` and `4294967295`, rejection of empty input, and total-budget enforcement `prompt + max_new_tokens <= max_model_len`.
- [ ] **Step 2: Run** `cargo test -p apxinf-model qwen35::admission -- --nocapture`; confirm failures are caused by missing qwen35 API.
- [ ] **Step 3: Implement** constants and checked helpers using `u32` token IDs and `usize` lengths. Do not consult tokenizer vocabulary and do not add fallback behavior.
- [ ] **Step 4: Re-run the focused tests** and then `cargo test -p apxinf-model`.

### Task 3: Add model-neutral runtime capability and request lifecycle types

**Files:**
- Create: `crates/apxinf-model/src/runtime.rs`
- Modify: `crates/apxinf-model/src/lib.rs`
- Test: `crates/apxinf-model/src/runtime.rs` (unit tests)

- [ ] **Step 1: Write failing tests** for a valid request, cancellation before execution, capacity rejection before launch, and RAII release of an in-flight slot after success/error/cancel.
- [ ] **Step 2: Run** `cargo test -p apxinf-model runtime:: -- --nocapture` and verify the expected missing-type failures.
- [ ] **Step 3: Implement** `RuntimeCapabilities`, `RuntimeRequest`, `AdmissionDecision`, `RuntimeError`, `CancellationToken`, and `RequestPermit`. Keep CUDA types out of public signatures; expose only model-neutral data and a calibrated byte budget.
- [ ] **Step 4: Re-run focused and crate tests**.

### Task 4: Implement the bounded single-owner worker

**Files:**
- Modify: `crates/apxinf-model/src/runtime.rs`
- Test: `crates/apxinf-model/src/runtime.rs`

- [ ] **Step 1: Add a failing worker test** proving a bounded queue rejects over-capacity submission and a cancelled request never invokes the executor.
- [ ] **Step 2: Run the focused test and confirm RED.**
- [ ] **Step 3: Implement** a std `sync_channel` worker with one receiver-owning thread, request permits, cancellation checks before execution, and result delivery through a one-shot channel. Ensure all permit paths drop on error.
- [ ] **Step 4: Run** `cargo test -p apxinf-model runtime:: -- --nocapture` and `cargo test --workspace --locked` (record any pre-existing failures without masking them).

### Task 5: Add CUDA-side calibrated budget access

**Files:**
- Modify: `crates/apxinf-cuda/src/context.rs`
- Modify: `crates/apxinf-cuda/src/lib.rs`
- Test: `crates/apxinf-cuda/src/context.rs`

- [ ] **Step 1: Write a unit test** for deterministic conversion of free/total bytes into a calibrated budget with an 8% safety margin.
- [ ] **Step 2: Run the focused test and confirm RED.**
- [ ] **Step 3: Implement** a small public `CudaMemoryInfo`/`calibrated_budget` API backed by existing CUDA FFI calls, with checked arithmetic and no allocator fallback. Keep the API independent of HTTP/protocol types.
- [ ] **Step 4: Run** `cargo test -p apxinf-cuda context:: -- --nocapture`; on a GPU-less test environment, use only pure conversion tests and report skipped CUDA runtime probes.

### Task 6: Integrate exports and record verification

**Files:**
- Modify: `docs/collaboration/records/PROGRESS.md`
- Modify: `docs/collaboration/records/DECISIONS.md` only for a new interface decision

- [ ] **Step 1: Export** the qwen35 config and runtime types from `crates/apxinf-model/src/lib.rs`; leave `src/main.rs` unchanged until member2's stable protocol adapter exists.
- [ ] **Step 2: Run** `python3 benchmarks/qwen38_4090/evaluation/test.py check`, `cargo check --workspace --locked`, focused runtime/admission tests, and `cargo test --workspace --locked`; record the existing `pi05_integrity_probe` failure if it remains.
- [ ] **Step 3: Record** exact commands, commit SHA, GPU UUID/queue ID if a job ran, artifact paths and SHA256, failures, and rollback SHA. Keep R0 `blocked` until equivalent Xid evidence is supplied.
- [ ] **Step 4: Review diff** to confirm no files under `benchmarks/qwen38_4090/evaluation/`, no model weights, and no external fallback were changed.

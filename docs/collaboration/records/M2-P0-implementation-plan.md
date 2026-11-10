# M2-P0 Protocol Stub Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a dependency-light protocol core and standalone HTTP stub that passes every frozen protocol gate and produces raw, reproducible evidence.

**Architecture:** Keep JSON validation, EOS/usage/SSE formatting, runtime adaptation, and HTTP transport in separate `src/server` modules. The binary includes those modules directly and uses the existing `apxinf-model` runtime-neutral types; `Cargo.toml`, `Cargo.lock`, and `src/main.rs` remain unchanged.

**Tech Stack:** Rust 2021, `std::net`, `serde_json`, existing `clap` and `apxinf-model`, Python 3 standard library, Cargo unit tests.

## Execution Status (2026-08-23)

- Tasks 1-4: complete in commits `dec5549`, `30a4007`, and `f989214`.
- Task 5 probe implementation: complete in `c1e05b7`; live evidence is
  recorded separately and binds to that source commit.
- Task 6 verification: complete locally; production runtime integration and
  GPU correctness/reliability evidence remain member1 responsibilities.

---

### Task 1: Add the request schema and admission tests

**Files:**
- Create: `src/server/mod.rs`
- Create: `src/server/schema.rs`
- Create: `src/bin/apxinf_protocol_stub.rs`
- Test: `src/server/schema.rs`

- [ ] **Step 1: Write failing schema tests**

Create module wiring and tests for:

```rust
pub const MODEL_VOCAB_SIZE: u64 = 248_320;

#[derive(Debug, Clone, PartialEq)]
pub struct GenerateRequest {
    pub input_ids: Vec<u32>,
    pub max_new_tokens: usize,
    pub temperature: f64,
    pub ignore_eos: bool,
    pub stream: bool,
}

pub fn parse_generate_request(
    raw: &[u8],
    max_model_len: usize,
) -> Result<GenerateRequest, ProtocolError>;
```

Cover valid input, malformed `{not-json`, missing and unknown fields,
`images`, empty tokens, `-1`, `4294967295`, accepted token `248056`,
temperature `0.1`, zero output budget, non-boolean flags, and total overflow.

- [ ] **Step 2: Run RED**

Run: `cargo test --bin apxinf_protocol_stub schema::tests -- --nocapture`

Expected: compilation fails because the parser and error type are absent.

- [ ] **Step 3: Implement strict parsing and checked admission**

Implement `ProtocolError { error_type, message }`, require exactly
`input_ids`, `max_new_tokens`, `temperature`, `ignore_eos`, and `stream`, parse
tokens as signed integers before `u32`, and enforce:

```rust
if input_ids.is_empty() {
    return Err(ProtocolError::invalid("input_ids must not be empty"));
}
if token < 0 || token as u64 >= MODEL_VOCAB_SIZE {
    return Err(ProtocolError::invalid("token id is outside model vocabulary"));
}
if temperature != 0.0 || max_new_tokens == 0 {
    return Err(ProtocolError::invalid("unsupported generation parameters"));
}
if input_ids
    .len()
    .checked_add(max_new_tokens)
    .is_none_or(|total| total > max_model_len)
{
    return Err(ProtocolError::invalid("request exceeds max_model_len"));
}
```

- [ ] **Step 4: Run GREEN and commit**

Run: `cargo test --bin apxinf_protocol_stub schema::tests -- --nocapture`

```bash
git add src/server/mod.rs src/server/schema.rs src/bin/apxinf_protocol_stub.rs
git commit -m "feat(protocol): add strict request admission"
```

### Task 2: Add response, EOS, usage, and SSE formatting

**Files:**
- Create: `src/server/response.rs`
- Modify: `src/server/mod.rs`
- Test: `src/server/response.rs`

- [ ] **Step 1: Write failing tests** for:

```rust
pub const EOS_TOKEN_IDS: [u32; 2] = [248_046, 248_044];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Usage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
}

pub fn apply_eos(output_ids: Vec<u32>, ignore_eos: bool) -> Vec<u32>;
pub fn result_json(request_id: &str, prompt_tokens: usize, output_ids: &[u32]) -> Vec<u8>;
pub fn sse_frames(request_id: &str, prompt_tokens: usize, output_ids: &[u32]) -> Vec<Vec<u8>>;
pub fn error_json(error: &ProtocolError) -> Vec<u8>;
```

Assert EOS inclusion/truncation, exact usage, consecutive indexes, one request
ID, a done event with usage, and final `data: [DONE]\n\n`.

- [ ] **Step 2: Run RED**

Run: `cargo test --bin apxinf_protocol_stub response::tests -- --nocapture`

- [ ] **Step 3: Implement canonical responses**

Use `serde_json::json!`, compute completion from `output_ids.len()`, and format
each SSE frame as `data: <json>\n\n`.

- [ ] **Step 4: Run GREEN and commit**

Run: `cargo test --bin apxinf_protocol_stub response::tests -- --nocapture`

```bash
git add src/server/mod.rs src/server/response.rs
git commit -m "feat(protocol): add result and SSE responses"
```

### Task 3: Add health and model-neutral service execution

**Files:**
- Create: `src/server/service.rs`
- Modify: `src/server/mod.rs`
- Test: `src/server/service.rs`

- [ ] **Step 1: Write failing service tests** against:

```rust
pub const EVALUATION_CONTRACT: &str =
    "apxinf.qwen38_27b.inference_interface.v1";
pub const MODEL_REVISION: &str =
    "63768c10df38c0395e12ef49edac1bd539eaeeea";

pub trait TokenStream: Send {
    fn next_token(&mut self) -> Result<Option<u32>, RuntimeError>;
    fn cancel(&self);
}

pub trait ProtocolRuntime: Send + Sync {
    fn capabilities(&self) -> RuntimeCapabilities;
    fn start(&self, request: RuntimeRequest)
        -> Result<Box<dyn TokenStream>, RuntimeError>;
}

pub struct ProtocolService<R> {
    runtime: R,
    next_request_id: AtomicU64,
}
```

Require exact health identity, no fallback, required capabilities, correct
non-stream output, EOS behavior, HTTP 400 admission mapping, HTTP 503 capacity
mapping, and health/short-request recovery after a request error.

- [ ] **Step 2: Run RED**

Run: `cargo test --bin apxinf_protocol_stub service::tests -- --nocapture`

- [ ] **Step 3: Implement the service response boundary**

```rust
pub struct ActiveGeneration {
    pub request_id: String,
    pub prompt_tokens: usize,
    pub max_new_tokens: usize,
    pub ignore_eos: bool,
    pub stream: Box<dyn TokenStream>,
}
```

Create `req-<number>`, convert validated input to `RuntimeRequest`, start the
incremental stream, and map errors without exposing CUDA types. Provide one
consumer for non-stream JSON and one iterator-style consumer for SSE; both stop
after `max_new_tokens`, include the first EOS only when `ignore_eos=false`, and
call `cancel()` when stopping before the runtime ends. `ActiveGeneration` owns
the real stream and cancels on EOS, budget exhaustion, disconnect, and drop;
do not precompute SSE frames. Reject service mode mismatch. Before the first
SSE body frame, runtime errors use normal HTTP mapping; after it, emit exactly
one SSE `type:"error"` event, cancel, and close without done/sentinel.

- [ ] **Step 4: Run GREEN and commit**

Run: `cargo test --bin apxinf_protocol_stub service::tests -- --nocapture`

```bash
git add src/server/mod.rs src/server/service.rs
git commit -m "feat(protocol): add runtime-neutral service core"
```

### Task 4: Add HTTP transport and disconnect cancellation

**Files:**
- Create: `src/server/http.rs`
- Modify: `src/server/mod.rs`
- Modify: `src/bin/apxinf_protocol_stub.rs`
- Test: `src/server/http.rs`

- [ ] **Step 1: Write failing loopback tests** for `/health`, malformed raw
POST, structured invalid POST, valid JSON POST, SSE frames arriving one token
at a time, 404, and a client that disconnects after the first SSE frame.

- [ ] **Step 2: Run RED**

Run: `cargo test --bin apxinf_protocol_stub http::tests -- --nocapture`

- [ ] **Step 3: Implement bounded HTTP/1.1**

Read through `\r\n\r\n`, parse `Content-Length`, cap bodies at 1 MiB, and read
exactly that count. JSON uses content length and `Connection: close`. SSE uses
`text/event-stream`, flushes every frame, and cancels on write/flush error.

The binary arguments are:

```rust
#[derive(clap::Parser)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:8001")]
    bind: String,
    #[arg(long, default_value_t = 32768)]
    max_model_len: usize,
}
```

The fake runtime yields token `7` lazily up to `request.max_new_tokens`, records
whether `cancel()` was called, and reports vocab 248320, parallel requests 1,
and the configured length.

- [ ] **Step 4: Run GREEN and commit**

Run: `cargo test --bin apxinf_protocol_stub http::tests -- --nocapture`

```bash
git add src/server/mod.rs src/server/http.rs src/bin/apxinf_protocol_stub.rs
git commit -m "feat(protocol): add standalone HTTP stub"
```

### Task 5: Add the gate probe and raw evidence

**Files:**
- Create: `tools/protocol/README.md`
- Create: `tools/protocol/run_protocol_gates.py`
- Create: `tools/protocol/test_protocol_gates.py`
- Create: `docs/collaboration/records/M2-P0-protocol-evidence.json`

- [ ] **Step 1: Write failing Python tests** for `build_cases()` and
`evaluate_row()`. Require malformed status-only evaluation, six structured
`stream=false` cases with JSON errors, the 8-token result, and both health rows.

- [ ] **Step 2: Run RED**

Run: `python3 -m unittest tools.protocol.test_protocol_gates -v`

- [ ] **Step 3: Implement the standard-library probe**

Use `urllib.request`; record UTC time, commit SHA, exact body, HTTP status,
parsed/raw response, elapsed milliseconds, and pass state. Exit nonzero on any
failed row.

- [ ] **Step 4: Run unit tests and live gate**

```bash
python3 -m unittest tools.protocol.test_protocol_gates -v
python3 tools/protocol/run_protocol_gates.py \
  --base-url http://127.0.0.1:18001 \
  --output docs/collaboration/records/M2-P0-protocol-evidence.json
```

Expected: all ten rows pass.

- [ ] **Step 5: Commit**

```bash
git add tools/protocol docs/collaboration/records/M2-P0-protocol-evidence.json
git commit -m "test(protocol): record evaluator gate evidence"
```

### Task 6: Verify and hand off the protocol branch

**Files:**
- Create: `docs/collaboration/records/M2-P0-handoff.md`

- [ ] **Step 1: Record** source HEAD, prerequisite `58d9fec`, commands, adapter
mapping, raw gate summary, known unrelated workspace-test failure, artifact
hash, and rollback commit.

- [ ] **Step 2: Run final checks**

```bash
cargo fmt --all -- --check
cargo check --workspace --locked
cargo test --bin apxinf_protocol_stub --locked
python3 -m unittest tools.protocol.test_protocol_gates -v
python3 benchmarks/qwen38_4090/evaluation/test.py check
git diff --check
git diff --exit-code HEAD -- benchmarks/qwen38_4090/evaluation
```

- [ ] **Step 3: Commit**

```bash
git add docs/collaboration/records/M2-P0-handoff.md
git commit -m "docs(protocol): add member2 handoff evidence"
```

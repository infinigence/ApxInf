# M2 Protocol, Loader Manifest, and Oracle Design

Date: 2026-08-23
Owner: member2 / protocol-oracle
Prerequisite commit: `58d9fec` (`RuntimeCapabilities`, `AdmissionDecision`, and bounded runtime worker)
Delivery branches: `feat/protocol-stub`, then `feat/oracle-loader`

## Scope

This design covers the member2 CPU-only first batch:

- the evaluator-facing HTTP, JSON, and SSE protocol surface;
- pure protocol admission and error recovery around a model-neutral runtime adapter;
- an independent protocol stub and raw protocol-gate evidence;
- an immutable loader manifest and synthetic W4 directionality fixture;
- a portable oracle job generator, schemas, hashes, and the M2-O0 server handoff.

It does not change the evaluator, scorer, `src/main.rs`, Cargo production entry,
model forward path, CUDA kernels, public answers, or hidden data. It does not
download the checkpoint, expand a full BF16 model, or execute real 8K/16K
oracle workloads. Member1 owns production entry integration, real runtime
capacity checks, GPU1 oracle execution, and GPU0 formal replay.

## Delivery Structure

The work is split into two independently reviewable branches.

1. `feat/protocol-stub` delivers the server protocol core, standalone stub,
   protocol probes, tests, and raw gate evidence.
2. `feat/oracle-loader` delivers loader manifest validation, synthetic W4
   fixture tests, the oracle generator/schema, and the completed handoff
   record. It may be stacked on the protocol branch while both are under
   review, but each commit remains separable.

Both branches declare `58d9fec` as their prerequisite until member1 pushes or
integrates that runtime baseline.

## Protocol Architecture

`src/server/` contains framework-neutral protocol types, validation, response
serialization, SSE event construction, EOS handling, usage accounting, and a
small runtime adapter boundary. HTTP code receives model-neutral capabilities
and an incremental token stream; it never sees CUDA allocators or
device-specific types. The stream exposes `next_token()` and `cancel()`, so a
client disconnect or EOS stop can halt generation before the full budget is
materialized.

The standalone `src/bin/apxinf_protocol_stub.rs` uses existing workspace
dependencies and a minimal standard-library HTTP/1.1 listener, so this branch
does not modify `Cargo.toml`, `Cargo.lock`, or `src/main.rs`. It accepts
`--bind HOST:PORT` (default `127.0.0.1:8001`) and runs until SIGINT/connection
shutdown; the test harness starts it as a child process. Member1 can later
mount the same protocol core under the production HTTP framework and connect it
to the existing bounded runtime worker.

The stub health response is explicitly marked as a fixture while preserving
the frozen contract identity, revision, vocabulary size, maximum model length,
parallel request count, fallback state, and capabilities.

## Admission and Error Semantics

The request parser validates JSON shape and scalar types before conversion to
runtime request types. The accepted request keys are exactly `input_ids`,
`max_new_tokens`, `temperature`, `ignore_eos`, and `stream`; unknown keys are
rejected. It rejects unsupported fields, including the `images` probe, with
HTTP 400 and a JSON `error` object. Raw malformed JSON is also returned as
HTTP 400; JSON error formatting for this one case is useful but is not recorded
as an evaluator hard condition.

Pure protocol admission enforces:

- non-empty integer `input_ids`;
- model vocabulary `[0, 248320)`, accepting `image_token_id=248056` and
  rejecting `4294967295`;
- a positive integer `max_new_tokens`;
- exactly greedy `temperature=0`;
- boolean `ignore_eos` and `stream`;
- `prompt_tokens + max_new_tokens <= max_model_len` using checked arithmetic.

Capacity errors from the runtime adapter remain distinct from malformed
requests. All request-level errors use the shape
`{"error":{"type":"invalid_request","message":"..."}}` for protocol
admission failures and
`{"error":{"type":"capacity","message":"..."}}` for runtime capacity
rejections. They must leave the stub healthy for the next request.

## Generation, SSE, EOS, and Cancellation

Non-streaming success returns HTTP 200 with
`{"type":"result","request_id":"...","output_ids":[...],"usage":{...}}`,
where usage has `prompt_tokens`, `completion_tokens`, and `total_tokens`
computed from the actual returned tokens. Streaming success emits consecutive
token indexes under one request ID, then one done event containing the same
usage object, followed by `data: [DONE]`. The fixture executor returns a
deterministic token (`7`) for a one-token request and never represents this as
model correctness evidence.

The frozen EOS IDs are `248046` and `248044`. With `ignore_eos=false`, the first
EOS token is included once in the generated token sequence and generation then
stops. With `ignore_eos=true`, EOS does not shorten the requested budget.

Every request owns an incremental token stream and cancellation handle. A
socket write failure or client disconnect triggers cancellation and drops the
request guard. Non-streaming requests consume the same token stream into a JSON
result, so EOS and usage behavior cannot diverge between response modes.
Runtime errors are request-scoped; a subsequent health probe and short request
must still succeed. The stub exposes deterministic fake outputs only for
protocol testing and does not claim model correctness.

### Incremental Generation Invariants

`ActiveGeneration` is the only owner of a request's real `TokenStream`. It
contains the request ID, prompt count, requested output budget, EOS policy,
actual output IDs, completion state, and the runtime stream. Its state machine
is `started -> token* -> {eos, eof, budget, cancelled, error} -> finished`.
Both the JSON collector and HTTP SSE writer call the same one-token transition;
the HTTP layer never receives a precomputed frame list or a separate
cancellation token.

The transition includes the first EOS when EOS is honored, then cancels the
real stream. Reaching the requested output budget also cancels the real stream
unless natural EOF had already completed it. Explicit client disconnect and
the `Drop` path cancel any unfinished stream. Therefore each started request
has one observable runtime cancellation route and cannot retain a worker/KV
permit after a response path exits early.

SSE serializes each returned token immediately with its consecutive index. A
runtime error before any body frame maps to the ordinary HTTP error response;
after the `200 text/event-stream` headers and at least one frame, it emits one
`type: "error"` SSE event and closes without a done event or `[DONE]`. It then
cancels the same active generation. The service rejects a non-stream caller
for a `stream=true` request and a stream caller for a `stream=false` request.

The legacy `Fn(RuntimeRequest) -> RuntimeResult` adapter is fixture-only: it
exists for compatibility tests but is not a production streaming adapter.
Production integration must return a genuinely incremental stream and connect
the runtime request cancellation handle to that stream. Health uses one
capability snapshot per service, reports its model vocabulary explicitly, and
marks hard-coded stub-only fields as fixture behavior; production integration
must supply its actual fallback and modality state.

### Design Self-Review Corrections

The 2026-08-23 implementation review made the following invariants explicit:

- `ProtocolService` fails closed if runtime capabilities report a vocabulary
  other than model config vocab `248320`; a tokenizer vocab of `248044` cannot
  silently enter health or admission.
- an SSE runtime error is terminal state. After returning the error once,
  subsequent polling returns no frame and can never produce a done event or
  `[DONE]` sentinel.
- the HTTP disconnect test uses a generation budget much larger than the
  number of emitted frames before socket shutdown, and asserts cancellation
  occurred before natural budget exhaustion.
- the local gate probe reads the over-budget value from the live
  `/health.max_model_len`, records the exact canonical request body, always
  records both health rows, and writes a sibling SHA256 file for the evidence
  JSON.

These corrections do not broaden production claims: local health remains a
stub fixture, and no reliability boolean, correctness result, checkpoint
oracle, memory capacity, or GPU behavior is inferred from it.

## Protocol Evidence

A standard-library probe records timestamp, commit SHA, request body, HTTP
status, parsed JSON or raw response, and pass/fail for:

- `malformed_json`;
- `empty_input_ids`;
- `negative_token_id`;
- `out_of_vocabulary_token_id`;
- `unsupported_temperature`;
- `over_budget`;
- `unsupported_modality_field`;
- `valid_short_nostream_request`;
- `health_after_invalid_requests`;
- `health_contract_identity`.

All six structured negative controls explicitly send `stream=false`. The short
request uses eight input tokens, requests one output token, and requires usage
8/1/9. Evidence distinguishes protocol eligibility from the five global
reliability booleans; it never edits or bypasses scorer logic.

## Loader Manifest and Synthetic W4 Fixture

`apxinf-loader` gains immutable manifest types for checkpoint revision, model
configuration, tensor dtype/shape, quantization role, pack axis, and group
size. Manifest parsing reads metadata and headers without materializing a full
BF16 checkpoint.

The required inventory is validated exactly:

| Tensor | Shape | Dtype | Direction |
| --- | --- | --- | --- |
| `k_proj.weight_packed` | `[1024,640]` | I32 | K packed |
| `k_proj.weight_scale` | `[1024,160]` | BF16 | K group-32 |
| `k_proj.weight_zero_point` | `[128,160]` | I32 | N group-32 |
| `down_proj.weight_packed` | `[5120,2176]` | I32 | K packed |
| `down_proj.weight_scale` | `[5120,544]` | BF16 | K group-32 |
| `down_proj.weight_zero_point` | `[640,544]` | I32 | N group-32 |

Tests declare these production shapes without allocating their full buffers.
Compact synthetic vectors exercise nibble values 0 and 15, K group boundaries,
non-aligned tail blocks, pack/unpack round trips, and directed failures when N
and K axes are swapped. No real checkpoint slice is copied into Git.

## Oracle Generator and Artifact Identity

`tools/oracle/generate_golden.py` uses the Python standard library for local
manifest/schema generation. It accepts explicit `--model-dir`, `--output-dir`,
`--revision`, `--layers`, and `--stages`, with optional input-manifest and
runner arguments. Local tests use a tiny synthetic model directory and token
manifest. Without `--runner`, the command writes only a job manifest and
schema; it never invents hidden/state/logit values. With `--runner`, the
runner receives the canonical job manifest path and must write declared output
files; missing or extra files fail the command before hashes are recorded.

The generator writes canonical JSON for:

- checkpoint and input identity;
- selected layers and stages;
- reference input token IDs plus pending output-token and decoded-text schemas;
- hidden, recurrent state, KV, and logit artifact schemas;
- generation parameters, EOS policy, dtype/shape metadata, and tolerances;
- expected artifact filenames and SHA256 values.

Manifest-only mode prepares a reproducible job bundle without claiming real
golden outputs. Server execution mode delegates checkpoint-specific inference
to an explicit runner command, validates its output against the same schema,
and hashes every produced artifact. The artifact identity binds the generator
source SHA, contract SHA, config and generation-config SHA256 values, model
revision, input-manifest SHA, layer/stage selection, generation parameters,
and schema version. Unchanged identities reuse existing server artifacts. The
generator records its own source SHA when invoked from a Git checkout and fails
closed if the requested revision is empty or the model directory does not
exist.

### Oracle and Loader Self-Review Corrections

The 2026-08-23 loader/oracle implementation review made these invariants
explicit:

- manifest-only output declares every golden artifact as `pending`; it contains
  input token IDs and an output-token schema, but no reference output IDs,
  decoded text, hidden/state/logit values, or completed artifact hashes;
- the generator accepts only the frozen model revision, config vocabulary
  `248320`, EOS list `[248046,248044]`, and a 64-entry Qwen3.5 layer-type map;
- `--layers` without `--stages` means `layer_hidden`. GDN state is emitted only
  for `linear_attention` layers, and KV state only for `full_attention` layers;
- runner execution is shell-free and isolated to `artifacts/`. It fails on
  missing/extra files, directories, symlinks, control-manifest mutation,
  symbolic-shape mismatch, non-finite F32, invalid EOS/stop metadata, model
  metadata drift, schema/dtype/shape/hash mismatch, invalid token IDs, or
  nonzero exit before marking any artifact complete;
- header-only sharded SafeTensors inventory verifies exact index-to-shard
  ownership, rejects shard symlinks, and rejects unindexed tensors rather than
  silently dropping them; production sharded loading has the same behavior;
- loader manifests freeze schema/revision and carry quantization role metadata;
  `build_qwen35_w4_layer_manifest()` is the bridge from header inventory to the
  W4 direction gate;
- synthetic W4 dequantization rejects both weight and zero-point nibbles outside
  `0..=15`.

These checks remain metadata-only and synthetic locally. They do not assert
that the member1 checkpoint runner is implemented, that the real oracle has
run, or that any scorer reliability boolean is true.

Member1 executes the real job under the global lock on GPU1 and records GPU
UUID, peak VRAM, command, raw artifact path, exported file list, and hashes
using `docs/collaboration/templates/oracle-handoff.md`.

## Testing and Rollback

Implementation follows red-green-refactor cycles. Local acceptance includes
focused Rust tests, Python generator/probe tests, formatting, workspace check,
contract self-check, the live stub protocol gate, and a review that the frozen
evaluation directory is unchanged. The known default-feature failure in the
unrelated `pi05_integrity_probe` example is recorded separately.

Rollback points are the parent of each feature branch and the last green commit
within each branch. Protocol integration is additive: member1 can omit the
server module or disconnect the runtime adapter without altering model code.
Loader/oracle changes are isolated to their owned crate and tools directory.

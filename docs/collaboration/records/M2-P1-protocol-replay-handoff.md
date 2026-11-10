# M2-P1 Protocol Replay and Reliability Harness Handoff

Date: 2026-08-23  
Owner: member2 / protocol-oracle  
Branch: `feat/protocol-replay`  
Status: ready for member1 GPU0 production replay

## Goal and Boundary

This phase hardens protocol replay evidence and local fake-runtime reliability
coverage while preserving all seven existing protocol gate meanings and the
frozen evaluator contract. It does not run a real model, request a GPU, unpack
the checkpoint, change evaluator/scorer code, or modify production fallback
paths. GPU correctness, memory capacity, Xid behavior, and real checkpoint
recovery remain member1 responsibilities.

## Changed Files

- `tools/protocol/run_protocol_gates.py`: `--base-url` replay, raw request/
  response capture, timestamps, elapsed time, initial/end health, identity and
  `stub_fixture`/`production_runtime` classification.
- `tools/protocol/test_protocol_gates.py`: frozen health identity and evidence
  schema tests.
- `tools/protocol/README.md`: replay and production command templates.
- `src/server/conformance.rs`: runtime-neutral `ProtocolRuntime` capability,
  incremental stream, and observable cancellation probe.
- `src/server/mod.rs`: exports the conformance helper.
- `src/server/service.rs`: queue-full recovery and SSE request-id/index tests.
- `docs/collaboration/records/M2-P1-stub-replay-evidence.json` and its
  `.sha256`: local fixture evidence only.

## Commit

Implementation commit: `f98cb6ff7e96476c244b0e42db13600c2726520c`  
Evidence capture commit: `c95f3b0e1240f1f7edf6886d732a7a19bf9da52e`

Parent integration baseline: `3139979882ffaa1feae34131f15d46a4d43e12ad`  
Rollback commit: `3139979882ffaa1feae34131f15d46a4d43e12ad`

## Verification Commands and Results

The following commands are required for this handoff:

```bash
cargo fmt --all -- --check
CARGO_TARGET_DIR=/tmp/apxinf-target-m2-p1-check cargo check --workspace --locked
CARGO_TARGET_DIR=/tmp/apxinf-target-m2-p1-test cargo test --workspace --locked
python3 -m unittest tools.protocol.test_protocol_gates -v
python3 -m unittest tools.oracle.test_generate_golden tools.oracle.test_qwen35_checkpoint_runner -v
python3 benchmarks/qwen38_4090/evaluation/test.py check
git diff --check
git diff --exit-code HEAD -- benchmarks/qwen38_4090/evaluation
```

Results:

- `cargo fmt --all -- --check`: blocked by pre-existing formatting drift in
  committed, out-of-scope core/model/CUDA/main files; no prohibited file was
  reformatted. `rustfmt --check src/server/conformance.rs src/server/mod.rs
  src/server/service.rs` passed for every Rust file changed in this phase.
- workspace `cargo check`: passed, warnings only.
- workspace `cargo test`: blocked by the pre-existing default-feature build of
  `crates/apxinf-model/examples/pi05_integrity_probe.rs`, which imports
  CUDA-gated `apxinf_cuda`/pi05 APIs. The protocol binary's full local suite
  passed: 38 passed, 0 failed.
- protocol Python tests: 8 passed, 0 failed.
- oracle Python tests: 21 passed, 0 failed.
- evaluator assignment check: passed.
- `git diff --check`: passed.
- frozen evaluator diff check: passed, no changes.

The stub replay itself was executed with:

```bash
python3 tools/protocol/run_protocol_gates.py \
  --base-url http://127.0.0.1:18013 \
  --output docs/collaboration/records/M2-P1-stub-replay-evidence.json
```

Stub replay: 10/10 gates passed.  
Artifact: `docs/collaboration/records/M2-P1-stub-replay-evidence.json`  
SHA256: `b71df6a884eba3950d11f12d6bf979a84eb7a079c112a23f856dd55605bc013e`

## Production Replay Template

After member1 starts the assigned GPU0 runtime on
`GPU-d074a13d-dbb6-fceb-4caf-a45be9be9281`, run:

```bash
python3 tools/protocol/run_protocol_gates.py \
  --base-url http://127.0.0.1:<PORT> \
  --output docs/collaboration/records/M2-P1-production-replay-evidence.json
sha256sum docs/collaboration/records/M2-P1-production-replay-evidence.json
```

The production artifact must report `runtime_kind=production_runtime` and
`stub=false`; a fixture artifact must never be presented as production
runtime evidence.

## Known Limitations

GPU numerical correctness, actual VRAM capacity, CUDA Xid stability, and real
checkpoint recovery after runtime errors are explicitly out of scope here and
remain with member1.

## Handoff Notes

The conformance helper intentionally does not modify a runtime adapter. It
reports a minimal compatibility failure if capabilities are invalid or if a
stream continues after cancellation. Runtime adapters should preserve the
existing mappings: capacity/queue-full -> capacity protocol error, admission
-> invalid request, cancellation -> cancellation error, and post-SSE runtime
error -> one error frame with no done frame or `[DONE]` sentinel.

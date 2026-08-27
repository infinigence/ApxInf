# M2-P0 Protocol Stub Handoff

Date: 2026-08-23
Owner: member2 / protocol-oracle
Status: review
Branch: `feat/protocol-stub`
Source commit under test: `c1e05b72d6e6c762ad153fde589e5a73c91fcabd`
Prerequisite: `58d9fec`
Rollback point: `30a400774f4d0583ea53d56eee4f0614422947fb`

## Scope and Interface

This branch adds a dependency-light evaluator protocol core and standalone
fixture server. It does not change `Cargo.toml`, `Cargo.lock`, `src/main.rs`,
the evaluator/scorer, model forward code, CUDA kernels, or checkpoint data.

Member1 integration surface:

- implement `ProtocolRuntime::capabilities()` with the real model capability
  snapshot and model vocab `248320`;
- implement `ProtocolRuntime::start(RuntimeRequest)` with a genuinely
  incremental `TokenStream`;
- connect both `RuntimeRequest.cancel` and `TokenStream::cancel()` to worker,
  KV, and admission-permit release;
- mount `ProtocolService` under the production HTTP framework or preserve the
  same JSON/SSE behavior in an equivalent adapter;
- set `stub=false` only for the real runtime and report measured capacity and
  actual modality/fallback state.

The local binary uses `--bind 127.0.0.1:8001` and
`--max-model-len 32768` defaults and emits deterministic token `7`. Its output
is protocol fixture evidence, not model correctness evidence.

## Protocol Gate Evidence

Artifact: `docs/collaboration/records/M2-P0-protocol-evidence.json`

SHA256: `1a74bc05bfd60edf342fc2a9e3d816a50bbb6b97bc414b3c572d94b63ecbece9`

Hash file: `docs/collaboration/records/M2-P0-protocol-evidence.json.sha256`

| Gate | HTTP | Result |
| --- | ---: | --- |
| `malformed_json` | 400 | pass |
| `empty_input_ids` | 400 | pass |
| `negative_token_id` | 400 | pass |
| `out_of_vocabulary_token_id` | 400 | pass |
| `unsupported_temperature` | 400 | pass |
| `over_budget` | 400 | pass |
| `unsupported_modality_field` | 400 | pass |
| `valid_short_nostream_request` | 200 | pass, token `7`, usage `8/1/9` |
| `health_after_invalid_requests` | 200 | pass |
| `health_contract_identity` | 200 | pass |

All six structured negative cases used `stream=false`. The over-budget body
used the live health value `max_new_tokens=32768`. The evidence records exact
canonical request bodies and identifies source commit `c1e05b7`.

## Verification

Commands run from `/mnt/chuangxin/team2/ApxInf`:

```bash
cargo fmt --all -- --check
CARGO_TARGET_DIR=/tmp/apxinf-target-m2-check.<id> cargo check --workspace --locked
CARGO_TARGET_DIR=/tmp/apxinf-target-m2-test.<id> cargo test --bin apxinf_protocol_stub --locked
python3 -m unittest tools.protocol.test_protocol_gates -v
python3 benchmarks/qwen38_4090/evaluation/test.py check
git diff --check
git diff --exit-code HEAD -- benchmarks/qwen38_4090/evaluation
```

Results:

- formatting: pass;
- workspace check: pass with pre-existing warnings only;
- protocol Rust tests: 33 passed, 0 failed;
- protocol Python tests: 6 passed, 0 failed;
- evaluator assignment checks: pass;
- diff checks: pass; frozen evaluator directory unchanged.

## Correctness and Reliability Boundary

This handoff proves only the local protocol gate and cancellation/state-machine
contract. It does not assert checkpoint correctness, GPU memory capacity,
performance, Xid availability, or any of the five global reliability booleans.
Member1 must run real-runtime recovery and campaign evidence; any false
reliability boolean still makes the final submission ineligible.

## Risks and Rollback

- The stdlib HTTP listener is a standalone fixture transport, not the final
  production concurrency architecture.
- Production must preserve incremental streaming; an eager
  `Fn(RuntimeRequest) -> RuntimeResult` adapter is not acceptable.
- Before the first SSE body frame, runtime errors map to normal HTTP errors;
  after streaming starts, one SSE error event is emitted and the connection
  closes without done or `[DONE]`.
- Roll back protocol service/transport work to `30a4007`, or omit the server
  module and runtime adapter during production integration.

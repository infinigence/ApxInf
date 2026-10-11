# HTTP integration checks

`http_gateway.rs` starts the real `apxinf-serve` binary through a Python test runner.
The runner uses Python standard-library HTTP clients and loopback TCP sockets.
Each case starts a separate gateway and an original CPU worker fixture.
The fixture sends the serial protocol through the normal process pipes.
Startup checks the fixture manifest against local files and the actual interpreter.

These checks use the existing [local HTTP profile](../../../doc/serving/local-service-v0.1.md)
and [serial worker profile](../../../doc/serving/serial-profile-v0.1.md).
They do not load MLX, download models, execute tools, or contact external services.
The fixture reports its runtime as `cpu-fixture` and its MLX packages as `absent`.
Its execution labels satisfy the manifest schema only.
They do not establish real inference, model quality, performance, or Claude Code compatibility.

## Coverage

| Boundary | Required observation |
| --- | --- |
| Declared routes | Health, readiness, models, token counting, both generation APIs, and metrics respond through HTTP. |
| Nonstreamed generation | Both APIs preserve output, usage, the model alias, and request identity headers. |
| Anthropic SSE | Event names match payload types, indices remain stable, and one success terminal ends the stream. |
| OpenAI SSE | The stream ends with `[DONE]`. Only the requested final usage chunk contains usage counts. |
| Invalid input | Malformed JSON, duplicate keys, unsupported sampling, and conflicting limits never reach the worker. |
| HTTP routing | Unknown paths return 404. Unsupported methods return 405. Oversized bodies return 413. |
| Tools and stop sequences | Checked arguments become tool blocks. A stop match suppresses the matched text and following text. |
| Recoverable failure | Preparation errors and failed attempts retain readiness after resource settlement. |
| Protocol failure | Token counting preserves `protocol_fault`. Streaming emits one error without a success terminal. |
| Queue and disconnect | Overflow returns 429. Waiting and active disconnects release queue capacity and reservations. |
| Deadline | Expiry emits an error, records one expired outcome, and releases resources after worker settlement. |
| Worker exit | Readiness returns 503. Health remains available. New generation requests fail. |
| CLI shutdown | The binary exits successfully and reaps its worker process. |

## Run the checks

Run this command from the repository root:

```sh
cargo test --locked -p apxinf-serving --test http_gateway -- --nocapture
```

Python 3 must be available as `python3`.
The test adds no Rust or Python package dependency.
The runner uses process groups and therefore requires Linux or macOS.
Each listener selects an available port through port zero.
The runner sends signals only to its own gateway or process group.

Diagnostics remain in unique directories under `devlocal/serving-http-integration/`.
Set `APXINF_HTTP_TEST_OUTPUT` to select another output directory.
The runner retains gateway diagnostics and worker events when a check fails.
The CI workflow uploads this directory after success or failure.

The CPU workflow runs Rust checks and Python unit tests on Linux and macOS.
It fixes the Rust and Python versions and uses `Cargo.lock`.
It leaves `APXINF_SERVING_URL` empty, so the six external model tests remain skipped.
Run those tests separately against an approved real model deployment.

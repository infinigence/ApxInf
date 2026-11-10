# Protocol Gate Probe

This directory contains the dependency-free protocol gate probe for the
standalone stub and the production evaluator endpoint. It intentionally uses
only Python's standard library and does not modify the frozen evaluator.

Start the stub in one terminal:

```bash
cargo run --locked --bin apxinf_protocol_stub -- --bind 127.0.0.1:18001
```

Run the gates in another terminal:

```bash
python3 tools/protocol/run_protocol_gates.py \
  --base-url http://127.0.0.1:18001 \
  --output docs/collaboration/records/M2-P0-protocol-evidence.json
```

The probe reads `max_model_len` from `/health`, validates the frozen contract,
model revision, model vocabulary (`248320`), `fallback_active=false`, and
`capabilities.multimodal=false`, then classifies the evidence as
`stub_fixture` or `production_runtime`. It records the raw request body, HTTP
status, parsed response, raw response text, timestamps, elapsed time, and
source commit for every request, plus both initial and ending health checks.
Structured negative controls all explicitly set `stream=false`.

Production replay should be run only against the assigned real runtime:

```bash
python3 tools/protocol/run_protocol_gates.py \
  --base-url http://127.0.0.1:<PORT> \
  --output docs/collaboration/records/M2-P1-production-replay-evidence.json
```

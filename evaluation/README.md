# Evaluation

Accuracy and performance evaluation tools for source checkouts. Run these
commands from the repository root.

## LIBERO accuracy

The client evaluator owns the LIBERO simulator and sends observations to an
already-running OpenPI-compatible policy server. The server only needs to expose
the `franka_libero` wire contract; it does not need the LIBERO environment.

Start the server from an ApxInf inference environment:

```bash
python scripts/pi05_openpi_websocket_server.py \
  --model-dir /path/to/checkpoint \
  --precision bf16 \
  --host 0.0.0.0 \
  --port 8000
```

Run evaluation from an environment containing LIBERO and `openpi_client`:

```bash
python -m evaluation.libero.client \
  --host <server-host> \
  --port 8000 \
  --precision bf16 \
  --suite libero_10 \
  --tasks all \
  --trials-per-task 10 \
  --results-jsonl out/libero.jsonl \
  --summary-json out/libero.summary.json
```

For a single-process development run, use the dual-backend evaluator:

```bash
python -m evaluation.libero.eval \
  --backend in-process \
  --model-dir /path/to/checkpoint \
  --precision bf16 \
  --suite libero_10 \
  --results-jsonl out/libero.jsonl \
  --summary-json out/libero.summary.json
```

## Performance benchmarks

Run the PI0.5 benchmark with:

```bash
python -m evaluation.benchmarks.pi05 \
  --model-dir /path/to/checkpoint \
  --precision bf16 \
  --interface model,policy
```

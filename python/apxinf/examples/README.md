# apxinf examples

Minimal, dependency-light samples that **teach the interface**. For accuracy and
performance workflows, see [`evaluation/`](../../../evaluation).

Each example is self-contained and runs straight from a source checkout: they
share a tiny `sys.path` shim + synthetic-observation builder in `_common.py`, so
no dataset or simulator is needed to see the API work (the actions are
meaningless — the point is the call shape).

| Example | Shows |
|---|---|
| [`pi05policy_infer.py`](pi05policy_infer.py) | Concrete `Pi05Policy.from_pretrained` — model-specific knobs (`action_dim`, `image_keys`). |
| [`autopolicy_infer.py`](autopolicy_infer.py) | Generic `AutoPolicy.from_pretrained` — dispatch by `config.json` model type. |
| [`openpi_server.py`](openpi_server.py) | Serve a policy in-process over OpenPI's websocket protocol (`apxinf.serving`). |
| [`openpi_client.py`](openpi_client.py) | Connect an OpenPI client, read metadata, send one observation, read actions. |

## Dependencies

- The two `*_infer.py` and `openpi_server.py` need the **`apxinf_py` CUDA
  binding** (`maturin develop --features cuda` in [`crates/apxinf-py`](../../../crates/apxinf-py))
  and a checkpoint directory (model + tokenizer + `norm_stats.json`).
- `openpi_server.py` also needs the transport deps
  ([`scripts/requirements-pi05-websocket.txt`](../../../scripts/requirements-pi05-websocket.txt)).
- `openpi_client.py` needs the upstream `openpi_client` package and a running
  server; it needs **no** CUDA or checkpoint of its own.

## Quick start

```sh
# Terminal 1 — serve a checkpoint
python examples/openpi_server.py --model-dir /path/to/checkpoint --precision bf16

# Terminal 2 — one round trip against it
python examples/openpi_client.py --host 127.0.0.1 --port 8000

# Or skip the network and call the policy in-process
python examples/pi05policy_infer.py --model-dir /path/to/checkpoint
python examples/autopolicy_infer.py --model-dir /path/to/checkpoint
```

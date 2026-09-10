# apxinf examples

Minimal, dependency-light samples that **teach the interface**. Heavy evaluation
harnesses (LIBERO sim rollouts, resumable ledgers, benchmarks) stay in the
repo-level [`scripts/`](../../../scripts) — those are ops/eval tooling, not API
demos.

Each example is self-contained and runs straight from a source checkout: they
share a tiny `sys.path` shim + synthetic-observation builder in `_common.py`, so
no dataset or simulator is needed to see the API work (the actions are
meaningless — the point is the call shape).

| Example | Shows |
|---|---|
| [`pi05policy_infer.py`](pi05policy_infer.py) | Concrete `Pi05Policy.from_pretrained` — model-specific knobs (`action_dim`, `image_keys`). |
| [`autopolicy_infer.py`](autopolicy_infer.py) | Generic `AutoPolicy.from_pretrained` — dispatch by `config.json` model type and pass concrete-policy options as JSON. |
| [`openpi_server.py`](openpi_server.py) | Serve any registered policy over OpenPI's websocket protocol (`apxinf.serving`). |
| [`openpi_client.py`](openpi_client.py) | Connect an OpenPI client, read metadata, send one observation, read actions. |

Examples are organized by user workflow, not model family. A new model should
work through `autopolicy_infer.py` and `openpi_server.py`; use
`--policy-options` for arguments understood by its concrete policy. Add a
model-specific example only when it demonstrates a genuinely distinct workflow.

## Dependencies

- The `*_infer.py` examples and servers need the **`apxinf_py` CUDA
  binding** (see [Build ApxInf](../../../README.md#build-apxinf))
  and a compatible checkpoint directory.
- Using a WallOSS checkpoint additionally needs `pip install -e "python/apxinf[walloss]"`
  for its Qwen2.5-VL tokenizer/image processor and serialized normalizers.
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

# WallOSS uses the same generic commands. Policy-specific options are data.
python examples/autopolicy_infer.py \
  --model-dir /path/to/wall-oss-0.5 \
  --action-dim 7 \
  --policy-options '{"norm_key":"x2_normal"}'

python examples/openpi_server.py \
  --model-dir /path/to/wall-oss-0.5 \
  --image-keys observation/image,observation/wrist_image \
  --state-key observation/state \
  --policy-options '{"norm_key":"x2_normal"}'
```

Without `--action-dim`, both generic launchers keep the action width inferred
from the checkpoint weights (currently deployed WallOSS checkpoints may be either
7 or 26 channels). A user-supplied `--action-dim` wins; otherwise the checkpoint
width is used. Prefix trimming is valid only when the selected normalizer and the
checkpoint's leading-channel layout match the target robot. Non-prefix layouts
need a body-aware adapter around the policy — see
[apxinf-robo](https://github.com/team-mz/APXinf-robo), which pairs named bodies
with named wire dialects — rather than a different `--action-dim`.

WallOSS state-token binning is likewise loaded from checkpoint metadata
(`config.json`, then legacy `config.yml`) and falls back to 256 only when absent.
An explicit `"state_bins"` in `--policy-options` has highest precedence.

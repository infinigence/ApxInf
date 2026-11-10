# Qwen3.5 metadata fixtures

Two files copied verbatim from the pinned leaderboard model. They contain **no
weight values** — only configuration and the safetensors tensor-name / shape /
shard map.

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `config.json` | 20927 | `fece2915d4c8ad4c10877622f04ea5e01cd3ae38768ce5c1edb700dd1de290f6` |
| `model.safetensors.index.json` | 241486 | `82b1bf79f5b61333e83da17ec3bf89c9f178e29395a14c6b3ce3bbc474e1ead8` |

Source: `cyankiwi/Qwen3.8-27B-AWQ-INT4`, revision
`63768c10df38c0395e12ef49edac1bd539eaeeea` (the revision pinned by
`benchmarks/qwen38_4090/evaluation/contract-v1.json`). Upstream license: Apache
License 2.0.

## Why these two are committed and nothing else is

They are read constantly by code and tests — W4 dispatch manifest, dtype
classification per module, layer-type sequence, shape inventory, VRAM ledger —
so those tests must run offline with no network and no model directory. At
262 KB combined they stay reviewable in a diff.

Everything else (tokenizer, chat template, preprocessor configs) is a 12.3 MiB
third-party payload and is **not** committed. Fetch it with:

```bash
python3 scripts/fetch_model_metadata.py
```

That mirrors the policy the contract itself states for the public corpus: fetch
and verify at use time, do not vendor the payload in the repository. The script
pins the same revision and verifies every file by size and SHA256.

Never add `.safetensors` files here, and do not edit these two files. If the
pinned revision ever changes, the contract changes with it — update both this
README and `scripts/fetch_model_metadata.py`, and record the decision in
`docs/collaboration/records/DECISIONS.md`.

## Verified facts derived from these files

Useful numbers that come straight out of `config.json` and the index, recorded
here so nobody has to re-derive them:

- Architecture id is `Qwen3_5ForConditionalGeneration` / `model_type: qwen3_5`,
  even though the repository is named Qwen3.8. Dispatch on the config, never on
  the repository name. Contract identity strings use `qwen38_27b` and are a
  separate namespace — `/health.evaluation_contract` must be exactly
  `apxinf.qwen38_27b.inference_interface.v1`.
- 64 text layers, `full_attention_interval: 4` → 48 linear-attention (GDN)
  layers and 16 full-attention layers.
- `attn_output_gate: true` → `q_proj` emits `2 x 6144`; split into q/gate and
  multiply the attention output by `sigmoid(gate)` before `o_proj`.
- `partial_rotary_factor: 0.25` → RoPE applies to the first 64 of 256 head dims.
- `mamba_ssm_dtype: float32` → GDN recurrent state is FP32, about
  `48 x 128 x 128 x 4 B = 3 MiB` per layer, ~144 MiB per request across 48
  layers.
- Valid token ids are `[0, 248320)` from `embed_tokens` `[248320, 5120]`. Do not
  range-check against `tokenizer.vocab_size` (248044): `image_token_id` is
  248056 and sits between the two.
- Quantization is per-module mixed. From the index: MLP and full-attention
  projections are packed W4; GDN `in_proj_qkv` / `in_proj_z` are W4; GDN
  `out_proj` is packed W4 for 47 of 48 layers (layer 0 is BF16); GDN
  `in_proj_a` / `in_proj_b` / conv / norms, `embed_tokens`, `lm_head`, `mtp.*`
  and `model.visual.*` are BF16.
- Packing directions differ between the two quantization tensors, which is easy
  to get backwards and fails silently:

  | Tensor | Example shape | Packing |
  | --- | --- | --- |
  | `weight_packed` | `[1024, 640]` I32 | 8 int4 along **K** (640 = 5120/8) |
  | `weight_scale` | `[1024, 160]` BF16 | one group per 32 K (160 = 5120/32) |
  | `weight_zero_point` | `[128, 160]` I32 | 8 int4 along **N** (128 = 1024/8) |

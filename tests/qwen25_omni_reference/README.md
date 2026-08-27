# Qwen2.5-Omni reference contract

This directory owns the three pinned BF16 Hugging Face reference artifacts for
`Qwen/Qwen2.5-Omni-3B` revision
`f75b40e3da2003cdd6e1829b1f420ca70797c34e`:

- `qwen25_omni_text.npz`
- `qwen25_omni_image.npz`
- `qwen25_omni_audio.npz`

The artifacts are generated only from a local model snapshot whose five
identity metadata files match the Host manifest. The canonical generator never
resolves a remote model:

```text
python3 scripts/hf_reference_dump.py \
  --only qwen25_omni_text \
  --qwen25-omni-model-dir "$APXINF_QWEN25_OMNI_MODEL_DIR"
```

Repeat with `qwen25_omni_image` and `qwen25_omni_audio`. Each artifact stores
processor tensors, post-embedding and post-media-injection state, Thinker text
layers 0/18/35, final norm, full last-position FP32 logits, and exactly ten
greedy tokens. Image/audio artifacts also store tower output and tower blocks
0/16/31. All BF16 activations are serialized as FP32 arrays.

The native checker consumes one artifact without invoking Transformers model
inference:

```text
cargo run -p apxinf-model --example qwen25_omni_check --features cuda -- \
  "$APXINF_QWEN25_OMNI_MODEL_DIR" \
  tests/qwen25_omni_reference/qwen25_omni_text.npz
```

The checker loads only the native ApxInf Thinker path and exits nonzero unless
the complete ten-token trajectory exactly equals the pinned HF oracle. The
three `.npz` files are intentionally absent until the frozen Host command
materializes the pinned snapshot and generates them on an authorized GPU.

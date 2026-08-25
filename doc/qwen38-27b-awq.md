# Qwen3.8-27B AWQ on RTX 4090

This backend serves the `cyankiwi/Qwen3.8-27B-AWQ-INT4` checkpoint on a
single SM89 GPU. It implements the checkpoint's hybrid decoder (GDN and full
attention layers), asymmetric group-32 AWQ weights, BF16 activations and KV
cache, Marlin prefill, greedy decode, and the native vision path.

The implementation directory is named `qwen35` because the checkpoint declares
`model_type: qwen3_5`. `Qwen3.8-27B` is the concrete model covered by the strict
shape and quantization checks; the module name is not a claim that every Qwen3
or Qwen3.5 checkpoint is supported.

## Build and inspect

CUDA 12.x or 13.x and an RTX 4090-class SM89 target are required for native
execution.

```bash
APXINF_CUDA_ARCH=sm_89 cargo build --release --features cuda

./target/release/apxinf inspect \
  --model /path/to/Qwen3.8-27B-AWQ-INT4 \
  --json
```

`inspect` validates the model identity, layer schedule, shard metadata, tensor
shapes, and AWQ packing before any resident service is started.

## Serve text requests

```bash
./target/release/apxinf serve \
  --model /path/to/Qwen3.8-27B-AWQ-INT4 \
  --host 127.0.0.1 \
  --port 8001 \
  --max-model-len 32768 \
  --enable-experimental-marlin-m64
```

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.8-27B-AWQ-INT4",
    "messages": [{"role": "user", "content": "Explain CUDA graphs briefly."}],
    "max_tokens": 32,
    "temperature": 0,
    "stream": false
  }'
```

The server also exposes `GET /health`, `GET /v1/models`, and the deterministic
`POST /v1/evaluations/generate` endpoint for pre-tokenized evaluation. The
current contract is one resident model, one request at a time, greedy decoding,
and `prompt_tokens + completion_tokens <= max_model_len <= 32768`.

## Enable one-image requests

Set `APXINF_PROCESSOR_PYTHON` to a Python environment containing a
checkpoint-compatible `transformers`, `torch`, `Pillow`, and `numpy`, then add
`--enable-multimodal` to the serve command. Multimodal v1 accepts exactly one
PNG data URL in a user message, non-empty text, `temperature: 0`, and
`stream: false`.

```json
{
  "model": "Qwen3.8-27B-AWQ-INT4",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "Describe the image."}
    ]
  }],
  "max_tokens": 32,
  "temperature": 0,
  "stream": false
}
```

Unsupported checkpoint layouts, architectures, sampling modes, image counts,
or context sizes fail closed instead of silently selecting another runtime.

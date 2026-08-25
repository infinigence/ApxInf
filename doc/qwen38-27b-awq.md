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
execution. The upstream-integrated carrier was built with CUDA 12.3 and Rust
1.97.1.

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

Set `APXINF_PROCESSOR_PYTHON` to a checkpoint-compatible Python environment,
then add `--enable-multimodal` to the serve command. The validated environment
uses Python 3.10.12, Transformers 5.15.0, PyTorch 2.13.0, Pillow 12.3.0,
NumPy 2.2.6, Tokenizers 0.22.2, Safetensors 0.8.0, and Hugging Face Hub 1.27.0.
Older Transformers releases may construct a tokenizer-only batch without
`pixel_values` and are rejected during a real image request. Multimodal v1
accepts exactly one PNG data URL in a user message, non-empty text,
`temperature: 0`, and `stream: false`.

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

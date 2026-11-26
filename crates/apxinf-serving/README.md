# LLM/VLM HTTP serving

`apxinf-serving` exposes a single LLM or VLM through OpenAI Chat Completions
and Anthropic Messages endpoints. It is optional: ordinary CLI builds and the
existing OpenPI WebSocket server do not depend on it.

## Start a server

Build with `server` and the backend required by your model:

```bash
cargo build --release --features server,cuda
./target/release/apxinf serve \
  --model /path/to/checkpoint --device cuda:0 \
  --served-model-name local-model \
  --max-input-tokens 4096 --max-output-tokens 1024
```

For CPU text models use `--features server` and `--device cpu`. For native
Metal text models use `--features server,mlx`, the SDK setup in the
[MLX guide](../../doc/mlx-backend.md), `--device metal`, and the appropriate
`--model-name` / `--model-variant`. Model and backend support is unchanged by
enabling HTTP serving. Image requests require a loaded model that advertises
image input through the shared `LlmCapabilities` contract.

The checkpoint must contain `tokenizer.json` and a chat template supported by
`apxinf-tokenizer`. `--model-name`, `--model-variant`, repeated `--asset NAME=PATH`
and `--dtype auto|fp32|bf16` are forwarded to normal model loading. Sampling
defaults come from the checkpoint's `generation_config.json`; explicit request
settings override them.

OpenAI user content can be a string or an ordered array of text and image
blocks. Assistant content remains text or nullable content with tool calls.

The default bind address is `127.0.0.1:8000`. There is no authentication or TLS
inside this crate; use your deployment's authenticated proxy when exposing it
outside a trusted host. `--host` and `--port` change the listener address.

For CUDA operator preparation, `--autotune` requires an explicit absolute
`--tuning-dir`. Run any tuning pass separately and restart without `--autotune`
for normal serving. HTTP readiness means model loading succeeded; it does not
assert that every requested shape has been warmed up or captured.

## Endpoints

| Method | Path | Behavior |
| --- | --- | --- |
| GET | `/health`, `/healthz` | 200 when loaded; 503 while loading or after startup failure |
| GET | `/v1/models` | The configured model alias; 503 until ready |
| POST | `/v1/chat/completions` | OpenAI-style text/image input, text/tool response or SSE |
| POST | `/v1/messages` | Anthropic-style text/image input, text/tool response or SSE |

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"temperature":0}'

curl -N http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true,"stream_options":{"include_usage":true}}'

curl http://127.0.0.1:8000/v1/messages \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","max_tokens":64,"messages":[{"role":"user","content":"Hello"}]}'
```

OpenAI requests support `max_tokens` or `max_completion_tokens`, `temperature`,
`top_k`, `top_p`, frequency/presence penalties, `seed`, `enable_thinking` and
`chat_template_kwargs`. If no output limit is supplied, the server uses the
smaller of 50 tokens and the deployment limit. Only one completion (`n=1`) is
supported. Stop strings, wire log probabilities and non-text response formats
are rejected. Prompt and output token limits must be positive; prompts over the
configured limit are rejected before generation. Unknown model aliases fail
instead of being silently redirected.

Anthropic requests support text/system content, tool-use/tool-result blocks,
tool choice, `temperature`, `top_p`, and streaming. `max_tokens` is required.
Nonempty stop sequences are rejected. `thinking.type=enabled`
is forwarded to the checkpoint template; reasoning is not returned as separate
Anthropic thinking blocks. This is a supported subset of each protocol, not a
complete provider replacement.

## Image input

The HTTP adapter decodes PNG/JPEG, applies EXIF orientation and converts to RGB
in a bounded blocking pool before enqueueing. Encoded images remain in memory.
The dedicated worker receives RGB bytes and dimensions, not data URLs.

The loaded model declares a `VisionContract`: image count, resize pixel budget,
patch/temporal/merge geometry, image token ID and pixel dtype. Its Rust runtime
performs checkpoint-fixed resize, normalization, patchification and placeholder
expansion using the shared model helper. Text and image conversations use the
same Rust tokenizer and chat template. Vision checkpoints fail loading when
`preprocessor_config.json` is missing or unsupported. No external preprocessing
process or separate preprocessing environment is needed.

OpenAI example (replace `BASE64_PNG` with base64-encoded file bytes):

```json
{
  "model": "local-model",
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "Describe this picture."},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,BASE64_PNG"}}
  ]}],
  "max_tokens": 128
}
```

For Anthropic, use the same user block position with
`{"type":"image","source":{"type":"base64","media_type":"image/png","data":"BASE64_PNG"}}`.
Both protocols support ordinary and SSE responses. Image/text block order and
previous user image turns are retained, including subsequent text-only questions
about that image. Tool-result image blocks are not yet supported.

Transport limits are explicit `ServeConfig.media_limits` values with CLI flags:

| Flag | Default | Scope |
| --- | --- | --- |
| `--max-body-bytes` | 12582912 | Entire JSON/base64 HTTP body |
| `--max-image-bytes` | 4194304 | Encoded file bytes per image |
| `--max-image-dimension` | 4096 | Either decoded image side |
| `--max-decoded-pixels` | 16777216 | All decoded images in one request |

Model limits are separate: serving reads the loaded vision contract and never
narrows the checkpoint's resize budget. Qwen3-VL and Qwen38 currently declare
`max_images=1`; a second image returns 400 naming the model and `max_images`.
Remote/file URLs, animations, tool-result images and image detail other than
`auto` are rejected. Input length is checked after image-placeholder expansion.
Both text and image admission check `prompt_tokens + max_tokens` against the
runtime's actual sequence/KV capacity, including Qwen3-VL's allocated cache cap.
Exceeding either the configured input limit or runtime capacity returns 400
before inference; the next valid request remains usable.

Image prompts bypass the host text-token cache and start a fresh sequence.
There is no image-feature or cross-request KV reuse. A malformed image is a
client error. HTTP draining after SIGTERM/Ctrl-C is bounded to five seconds.

### Image validation matrix

| Date | Backend | Checkpoint / precision | Validation commit and evidence |
| --- | --- | --- | --- |
| 2026-11-26 | Thor / CUDA SM110 | Qwen3-VL-2B-Instruct / BF16 | [Pinned commit, requests and results](https://github.com/infinigence/ApxInf/pull/224#issuecomment-7155887216) |

Other backend × checkpoint combinations, including Metal and HIP image serving,
are **unvalidated**. CPU contract tests use a tiny synthetic model and recorded
processor outputs; they do not qualify CPU vision-model quality. Successful
model loading or a declared image capability does not establish validation.

## Tool calls

The server renders tool definitions and assistant/tool history through the
checkpoint's own chat template. Generated calls must use the Qwen3 XML form
`<tool_call><function=NAME><parameter=NAME>VALUE</parameter></function></tool_call>`.
The parser validates declared names, required arguments and supported schema
types before returning OpenAI `tool_calls` or Anthropic `tool_use` blocks. The
server returns calls to the client; it never executes tools.
Parameters declared as strings keep their XML text, including numeric or
boolean-looking values such as `10001` and `true`, under strict validation too.

`tool_choice` supports automatic, disabled, required and named-function modes.
`parallel_tool_calls=false` rejects multiple calls, and can end generation at
`</tool_call>` when that delimiter is one exact tokenizer token. Tool choice
instructions guide generation; malformed or missing required calls still fail
validation. Tool-use quality requires a checkpoint trained for this format.

Tool-call parsing always validates declared functions, required arguments and
parameter types. There is no process-wide relaxed validation mode.

## Execution and limits

A bounded queue feeds one dedicated worker thread. The thread constructs,
uses and drops the model, so thread-affine device resources never move into
the asynchronous HTTP runtime. Requests execute serially with batch size one;
this is not continuous batching. `--queue-capacity` (default 16) counts waiting
requests behind the active request. A full queue returns 429, and loading or
unavailable workers return 503.
If model construction or inference panics, the worker publishes its failed
state before returning the error. Health/model endpoints and later requests
return 503, and the failed model is dropped on its owning thread. Restart the
server to recover; the worker never retries inference on that model.

Ordinary and streaming text preserve indentation and trailing whitespace.
Streaming forwards stable incremental UTF-8 and ends with the protocol's
terminal event. Queued requests whose clients have disconnected are skipped.
An active disconnect stops publication; the current model generation API may
continue computation until its normal stopping point. Slow readers can apply
backpressure to the single worker.

Four recently rendered prompts are cached as host token IDs. This does not
reuse model KV state; the native adapter reports zero cached prompt tokens.
The Final branch's Qwen3.5 runtime, MTP controls, KV sizing and cross-request
context reuse are separate integrations. `disable_spec_decode=true` is rejected
until request-level speculative control is available in the model runtime.

For application-owned inference, implement `ChatEngine`, construct a
`WorkerClient` with its factory, and pass that client plus `ApiConfig` to
`router`. `EngineOutput` allows such an engine to report its own reused tokens.
Optional `--log-timing` worker timing logs describe queue/worker wall
time, not pure kernel time.

## Validation

```bash
cargo test --locked -p apxinf-serving -p apxinf-tokenizer --no-default-features
cargo test --locked -p apxinf --features server --no-default-features
cargo check --locked -p apxinf --no-default-features
cargo test --locked -p apxinf-model --test vision_preprocessing --no-default-features
```

The HTTP tests use deterministic engines and require no GPU or checkpoint.
They cover protocol conversion, image blocks, tools, streaming, startup errors,
queue overflow, and queued/active client disconnection. A tiny CPU Llama plus
a vision-input probe checks the native RGB/ImageInput/generation boundary;
the model owns the native RGB-to-patches boundary. Recorded fixtures verify
F32 pixels with max_abs=0, image grids, expanded token IDs and structured tool
history in offline CPU-only CI.
These tests do not establish real VLM quality or CUDA/Metal/HIP qualification.

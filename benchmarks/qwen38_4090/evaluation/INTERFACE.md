# Pretokenized inference interface v1

Only this contract and its tests need to be public. A reference implementation,
checkpoint access, private kernels, and hidden evaluation data are not part of
the student handout.

## Health and identity

`GET /health` returns HTTP 200 only when the service can accept a request:

```json
{
  "status": "ok",
  "evaluation_contract": "apxinf.qwen38_27b.inference_interface.v1",
  "model_revision": "63768c10df38c0395e12ef49edac1bd539eaeeea",
  "max_model_len": 32768,
  "parallel_requests": 1,
  "fallback_active": false,
  "capabilities": {
    "pretokenized_input_ids": true,
    "token_id_output": true,
    "multimodal": false
  }
}
```

The evaluator rejects a model revision or contract identifier mismatch. A base
submission may report `parallel_requests=1`; the base TTFT/TPOT workload always
sends one request at a time. A multi-request bonus submission must report at
least `parallel_requests=8` and prove that independent concurrent calls use the
same endpoint without silent serialization or fallback. `max_model_len=32768`
is sufficient for the base leaderboard and the non-scoring 32,640 prompt + 128
output diagnostic. Claiming any positive context bonus requires a larger
advertised and verified total length.

`fallback_active` must describe the currently served path, not a desired
configuration. The exact-token candidate interface requires `false`; a service
that silently delegates to vLLM, Transformers, CPU, or a different model is not
eligible.

## Generate

`POST /v1/evaluations/generate` accepts:

```json
{
  "input_ids": [151644, 872, 198, 9707, 151645, 198, 151644, 77091, 198],
  "max_new_tokens": 128,
  "temperature": 0.0,
  "ignore_eos": true,
  "stream": true
}
```

The requirements are intentionally small:

- `input_ids` is a non-empty array of unsigned 32-bit token IDs;
- `max_new_tokens` is positive and the full request must fit
  `max_model_len`;
- v1 supports greedy sampling only (`temperature=0`);
- performance cells set `ignore_eos=true` so every eligible run produces the
  exact output budget;
- the request body always represents exactly one logical request. Base scoring
  executes one call at a time; the optional serving track executes multiple
  independent calls concurrently.

For a streaming request, the response content type is `text/event-stream`.
Each generated token is one event:

```text
data: {"type":"token","request_id":"eval-apxinf-1","index":0,"token_id":198}
```

The zero-based indexes must be contiguous. The final event is:

```text
data: {"type":"done","request_id":"eval-apxinf-1","usage":{"prompt_tokens":9,"completion_tokens":128,"total_tokens":137}}

data: [DONE]
```

The evaluator measures TTFT from client send to the first token event and TPOT
as `(last token arrival - first token arrival) / (completion_tokens - 1)`.
Server timing fields are diagnostic only and cannot replace client timing.
`request_id` must be unique among all simultaneously active calls so the
multi-request evaluator can prove that token streams never cross.

A non-streaming request returns `type=result`, `request_id`, `output_ids`, and
the same `usage` object. Invalid JSON, IDs, budgets, or sampling parameters
return HTTP 400 with a JSON `error`. Context exhaustion must be recoverable:
the next `/health` call and a small generation request must still succeed.

For the multi-request bonus, the client holds 4 or 8 calls in flight in closed
loop until 32 requests complete. Queueing and scheduler delay are inside every
request's TTFT and the whole batch makespan. The HTTP/SSE schema does not gain a
batch-only variant; concurrency is composition of the same single-request
primitive.

For the context bonus, `input_ids + max_new_tokens` may not exceed the pinned
checkpoint's 262,144 native positions. The largest scored prompt is 262,016
because every capacity probe reserves and must generate 128 output tokens.

Text rendering, chat templates, and tokenizer selection are deliberately
outside this boundary. The teacher evaluator owns tokenization once and passes
the exact same IDs to every implementation. That removes the 40-token template
drift observed between the current ApxInf service and the vLLM reference.

## Optional integrated image capability

`multimodal-contract-v1.json` is an independent capability overlay. It does not
change the 100-point text base score or the two 10-point bonuses. A text-only
submission reports `capabilities.multimodal=false` and remains fully eligible.
It must nevertheless reject an image probe safely with HTTP 400, 415, 422, or
501 and this error shape:

```json
{"error":{"type":"unsupported_capability","message":"native multimodal is not ready"}}
```

HTTP 200 after silently discarding the image, HTTP 500, or delegation to a
fallback backend fails the capability protocol.

A submission that reports `capabilities.multimodal=true` accepts one PNG image
through `POST /v1/chat/completions` using standard OpenAI content parts:

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
      {"type": "text", "text": "图中有多少个绿色实心圆？只输出整数。"}
    ]
  }],
  "temperature": 0.0,
  "max_completion_tokens": 32,
  "stream": false,
  "chat_template_kwargs": {"enable_thinking": false}
}
```

The response must contain `choices[0].message.content`. The capability runner
uses deterministic 448×448 PNG files, verifies every image hash, measures
client-observed end-to-end latency, and applies exact-answer validation. The
server owns the pinned checkpoint processor because this track tests the
integrated media path; its timing is diagnostic, not compared with pretokenized
text TTFT/TPOT.

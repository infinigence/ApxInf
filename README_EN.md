# ApxInf Assignment: Qwen3.8-27B INT4 Inference Optimization on One RTX 4090

English | [中文](README.md)

## 1. Objective

Implement and optimize `cyankiwi/Qwen3.8-27B-AWQ-INT4` in ApxInf so that it
provides stable, correct, and efficient text inference on one NVIDIA RTX 4090.

Submit an implementation PR and a `REPORT.md`. Do not commit model weights or
private evaluation data.

## 2. Fixed Conditions

- GPU: one RTX 4090, compute capability 8.9;
- model revision: `63768c10df38c0395e12ef49edac1bd539eaeeea`;
- weights: W4A16, group size 32, asymmetric;
- input: pretokenized `input_ids`;
- decoding: greedy, `temperature=0`, thinking disabled;
- base scenario: one request at a time with 128 output tokens;
- no fallback to vLLM, Transformers, CPU, another model, or another GPU;
- [`contract-v1.json`](benchmarks/qwen38_4090/evaluation/contract-v1.json) is the single machine-readable authority for workloads, gates, and scores.

You may change the ApxInf Rust, CUDA, scheduling, memory-management, and service
implementations. You may not change the contracts, public-data generators, or
evaluators under `benchmarks/qwen38_4090/evaluation/` to improve a result.

## 3. Service Interface

### Health Check

When the service is ready, `GET /health` must return HTTP 200:

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

### Generation Request

`POST /v1/evaluations/generate` accepts:

```json
{
  "input_ids": [151644, 872, 198],
  "max_new_tokens": 128,
  "temperature": 0.0,
  "ignore_eos": true,
  "stream": true
}
```

Each streamed token is one SSE event:

```text
data: {"type":"token","request_id":"req-1","index":0,"token_id":198}
```

The stream ends with:

```text
data: {"type":"done","request_id":"req-1","usage":{"prompt_tokens":3,"completion_tokens":128,"total_tokens":131}}

data: [DONE]
```

Requirements:

- token indexes start at 0 and increase without gaps;
- concurrent requests never mix `request_id` values;
- performance cases produce the complete output budget;
- invalid parameters return HTTP 400 with a JSON `error`;
- health checks and a small request still succeed after a capacity failure;
- TTFT and TPOT use client receive timestamps; server-reported timings do not score.

Image support is an optional bonus. A supporting service must advertise
`capabilities.multimodal=true` and accept one `data:image/png;base64` image part
followed by one text part at `POST /v1/chat/completions`. The submitted ApxInf
implementation must execute the request without fallback. An unsupported service
keeps the capability `false` and must reject an image probe with HTTP 400, 415,
422, or 501 and `error.type=unsupported_capability`.

## 4. Scoring

The base score is 100 points:

| Section | Points | Requirement |
|---|---:|---|
| Correctness | 30 | Protocol, public cases, and private cases |
| TTFT | 35 | 1K, 2K, 4K, 8K, and 16K prompts |
| TPOT | 25 | 1K and 8K prompts |
| Reliability | 10 | Success rate, no fallback/OOM/NaN/Xid, and recovery |

Optional bonuses:

- long context, 0-10 points: verify steps above 32K up to the native 262,144 positions;
- multiple requests, 0-10 points: C4/C8 closed-loop correct goodput with success-rate, fairness, and tail-latency constraints;
- image capability, 0-10 points: 2 for public correctness, 6 for private correctness,
  and 1 per split for complete integration and reliability. Missing image support
  earns zero image points without invalidating an eligible text-only submission.

The base maximum is 100, the bonus maximum is 30, and the leaderboard maximum is
130. The evaluation platform generates image reports and binds them to the
implementation identity, contract hash, and split. Multimodal fields in
`submission.json` must not be filled in manually.

Each performance cell runs one warm-up followed by five measured repeats and uses
the median. TTFT and TPOT CV must not exceed 10%. The best valid result for each
cell in the same round defines full credit for that cell.

Performance ranking requires 100% public correctness, at least 11/12 private
correctness, at least 99% request success, and passing protocol and reliability
gates. A public run is for local debugging and is not an official score.

## 5. Unified Test Script

This assignment has one test entry point: `test.py`.

Check the code and assignment package:

```bash
python3 benchmarks/qwen38_4090/evaluation/test.py check
```

Prepare public data using a local model directory and `transformers`:

```bash
python3 benchmarks/qwen38_4090/evaluation/test.py prepare \
  --model-dir /path/to/Qwen3.8-27B-AWQ-INT4
```

After starting your service, run the public evaluation:

```bash
python3 benchmarks/qwen38_4090/evaluation/test.py run \
  --model-dir /path/to/Qwen3.8-27B-AWQ-INT4 \
  --base-url http://127.0.0.1:8001
```

Artifacts are written to `benchmarks/qwen38_4090/evaluation/runs/` by default.
The script generates raw request records, environment records, and
`submission.json`. Do not manually fill in or edit aggregate results.

## 6. Suggested Workflow

1. Make `test.py check` pass.
2. Implement `/health` and the generation endpoint, then pass public correctness.
3. Record baseline TTFT, TPOT, VRAM, and failure boundaries.
4. Test one performance hypothesis at a time and retain negative results.
5. Optimize end-to-end bottlenecks; do not substitute isolated kernel numbers for service results.
6. Rerun the unified test after changing shared CUDA/Rust boundaries.
7. Replay the build, service, and public evaluation from a clean checkout.

## 7. Submission Requirements

The PR must include:

- the design change and affected execution stages;
- commands and results for `test.py check` and `test.py run`;
- at least one negative control or regression test;
- tradeoffs among correctness, performance, stability, and VRAM;
- known limitations, failed experiments, and rollback instructions;
- `REPORT.md` with the baseline, hypothesis, implementation, measurement, result, and reproduction steps.

Acceptance uses the complete PR commit SHA. Do not hard-code outputs for case IDs,
public token sequences, or known answers. Do not commit model weights, credentials,
machine addresses, or private evaluation data.

## 8. Minimum Completion Criteria

- a clean checkout builds and starts;
- health declarations are truthful;
- all public functional cases pass;
- base performance cells produce complete outputs without OOM or fallback;
- the service remains available after invalid requests and capacity failures;
- the unified test script generates a complete result;
- the PR report allows another person to reproduce the work independently.

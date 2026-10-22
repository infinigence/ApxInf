# MiniCPM and Claude Code priorities

Status: active acceptance plan, 2026-10-22. This document records the selected workload and its remaining gates.

The [serving contract](contracts-v0.1.md) remains authoritative. The [implementation plan](implementation-plan.md) defines the full roadmap. This workload prioritizes stateless P1a and P2a. It does not require all roadmap features.

## Target workload

One personal computer runs one MiniCPM5-2B model. Claude Code connects through the local Anthropic Messages endpoint. It reads a drawing brief and writes an SVG of a pelican riding a bicycle.

The [pelican task](pelican-acceptance-v0.1.md) separates automated checks from visual acceptance. A successful response alone does not establish task completion. The actual file and matched tool results form the evidence.

A separate context probe checks facts placed near the start, middle, and end of a long prompt. The service reports the actual rendered token count. Character counts do not establish context capacity.

## Observed baseline

The integration branch includes upstream revision `e9da4bd`. The original working directory retains its uncommitted changes. The selected model bundle is `openbmb/MiniCPM5-2B` revision `f97400052a43d642bbc6e9975e2397e3ae6a6b52`.

The test computer has an Apple M4 and 16 GiB of memory. The installed Claude Code version is `2.1.289`. The native profile uses MLX `0.31.2`, checkpoint BF16 weights, and `bf16-compiled` execution.

The native CLI produced the requested short text. It also produced a `Read` function call in the checkpoint's tool grammar. These results do not establish HTTP compatibility or successful tool execution.

A client-only probe set the reported window to 4096 tokens. Claude Code rejected the request before sending it to the model endpoint. The same probe sent a request with a 16384-token window. No model inference ran during this probe.

The captured request exposed `Read` and `Edit` tools. A reference tokenizer counted 710 input tokens, with 1024 requested output tokens. This count describes the captured input rather than a completed native request. The client rejection therefore does not demonstrate that the model input exceeded 4096 tokens.

The baseline serving tests passed 135 Rust checks and 240 Python checks. The runners skipped one Rust check and six Python checks. These checks do not establish native MiniCPM serving acceptance.

## Current acceptance state

The explicit native provider, MiniCPM tool grammar, controlled cancellation, and 16K context path now have runtime evidence.
The [native record](minicpm-native-validation-20261022.md) retains the earlier qualification steps.
The [bounded profile record](native-bounded-thinking-validation-20261023.md) adds worker 2.4 and exact reasoning-budget accounting.
Its real Claude client completed Read, Read, and Edit, but the drawing failed visual acceptance.

The bounded long-context run returned the expected values at 15354 input tokens.
Markdown fences still failed the strict JSON check.
The feedback revision made three unsuccessful Edit attempts and ended at the context limit.
It retained the failed drawing unchanged.

| Open item | Required evidence |
| --- | --- |
| MC-01. Recognizable pelican and bicycle. | The real client produces a visually accepted SVG through actual tools. |
| MC-02. Literal Edit parameters and error recovery. | The model repairs a failed Edit and checks the file after its final successful change. |
| MC-03. Strict JSON in long context. | The model returns the expected facts without Markdown fences or schema errors. |
| MC-04. Client output reservation and portability. | A 4096-token output candidate completes under the default client policy without the private compatibility hook. |

The 4096-token candidate retains the separate 512-token initial reasoning budget.
It has no accepted result yet.
The current MiniCPM5-2B profile remains text-only and rejects image input.
A vision model requires a separate declared provider and interface qualification.
KV reuse remains lower priority than these task gates.

## Baseline gaps and acceptance order

| Priority | Observed gap | Required evidence |
| --- | --- | --- |
| 1 | HTTP execution selects the Python MLX-LM adapter. | An explicit native provider executes the pinned MiniCPM profile without fallback. |
| 1 | The generic tokenizer message type omits tool history fields. | The checkpoint template receives structured calls and results in order. |
| 1 | The existing Qwen parser uses another tool grammar. | MiniCPM function calls survive fragmented output and check declared arguments. |
| 1 | The generation callback cannot request cancellation. | Controlled execution stops at documented boundaries and preserves process recovery. |
| 1 | Model reset retains request arrays. | Checked cleanup releases state or transfers its accounted ownership. |
| 2 | Existing client tasks omit the requested drawing task. | Claude Code reads the brief and saves a visually accepted bicycle drawing. |
| 2 | Native MiniCPM admits at most 4096 consumed positions. | Larger explicit limits pass memory, cancellation, and factual retention checks. |
| 2 | Prefill consumes the complete prompt in one call. | Evaluated chunks bound temporary work and provide cancellation checkpoints. |
| 3 | Each HTTP request starts without reused KV state. | A bounded prefix policy preserves uncached output and accounts for retained memory. |

The first native implementation retained the 4096-position limit. The checkpoint's larger positional limit does not establish engine support. The service must reject excess input instead of removing history.

## Execution and recovery

A native child process owns model execution. The existing supervisor owns admission, deadlines, fencing, and process reaping. An in-process execution thread cannot provide the same recovery boundary when a native call stalls.

The worker renders and tokenizes the prompt during `prepare_input`. It returns the token IDs, effective count, and prompt digest. The coordinator checks that response before reservation and submits those exact tokens.

Provider selection must remain explicit. A load failure must not change the model, precision, provider, or execution variant. Native runtime identities must describe the native dependencies rather than reuse Python version labels.

One execution owner accesses mutable model state. A terminal output event does not prove resource release. Reservations remain charged until checked cleanup or successful process reaping establishes settlement.

## Context and cache policy

Larger context limits require evaluated prefill chunks before acceptance. Each tested range must record actual input tokens, reserved output, runtime identity, observed memory, and factual results. Cancellation and a subsequent request must also succeed.

The first cache candidate retains one idle prefix for one loaded model. Reuse requires matching model identity, template identity, token prefix, and state format. Retained state needs its own capacity owner and an explicit eviction path.

The [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) provides a reference for prompt reuse and idle slots. Its implementation remains external reference material. ApxInf code and fixtures remain original.

Context shifting is a separate feature. It must not silently remove earlier task instructions to fit a request. Prefix reuse is an optimization and does not establish long-context support.

Continuous batching, distributed scheduling, SSD snapshots, arbitrary cache forks, and multiple resident models remain deferred. Their absence does not block this personal-user workload.

## Evidence limits

Anthropic does not officially support Claude Code with non-Claude models, according to its [gateway documentation](https://code.claude.com/docs/en/llm-gateway). Acceptance records observed behavior with the selected client version. It does not claim general client compatibility or general coding quality.

Structural document checks do not establish a complete ASD-STE100 dictionary review. No skill or linter establishes certification.

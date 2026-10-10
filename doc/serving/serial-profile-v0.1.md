# Serial worker profile 0.1

Status: P0 interface for P1a. Date: 2026-10-09.

This profile selects a stateless text subset of `apxinf-worker/2.0`.
It does not complete P1 or add exact session append.
The [main contract](contracts-v0.1.md) defines common ownership and result rules.
All implementation code and fixtures for this profile are handwritten.

## Limits and ownership

The frame limit is 1048576 bytes, including LF.
The maximum JSON depth is 16, with the root object at depth 1.
The worker retains at most 10000 command records per epoch.
The prompt limit is 131072 tokens. The output limit is 65536 tokens.
Token events contain at most 256 tokens.
Messages and tools each contain at most 256 entries.
Each message contains at most 256 tool calls.
Each EOS array contains at most 256 tokens.
Each lease array contains at most 64 unique IDs.
Each general JSON collection contains at most 131072 entries.
String sizes remain inside the frame limit.
Model context and output limits can reduce these protocol limits.

The supervisor passes the protocol and epoch as worker launch arguments.
The worker checks them before it emits `ready`.
The worker owns the authoritative template and incremental decoder for this profile.
Gateway owns stop matching, tool parsing, and public API conversion.
The service must not apply another template or decoder to the prepared result.

## Frame notation

`B` means `protocol`, `kind`, and `worker_epoch`.
`C` means `B` plus `command_id`.
`R` means `B` plus `request_id`, `attempt`, and `event_seq`.
All IDs and primitive types follow the main contract.
Each frame includes all listed fields unless the table says otherwise.
Unknown keys are invalid, including nested keys.

| Kind | Fields after the common fields |
| --- | --- |
| `ready` (`B`) | `model_revision`, `capability_revision`, `model_manifest`, `capabilities`, `limits`, `model_path`, `eos_token_ids`, `vocab_size`, `runtime`, `memory` |
| `prepare_input` (`C`) | `model_revision`, `messages`, `tools`, `template_options` |
| `prepared_input` (`C`) | `model_revision`, `token_ids`, `effective_input_tokens`, `prompt_digest` |
| `submit` (`C`) | `request_id`, `attempt`, `model_revision`, `capability_revision`, `token_ids`, `max_tokens`, `remaining_timeout_ms`, `eos_token_ids`, `capacity_lease_ids` |
| `cancel_request` (`C`) | `request_id`, `attempt`, `reason` |
| `stop_generation` (`C`) | `request_id`, `attempt`, `cause`, `output_token_count`, `text_byte_cutoff` |
| `query_stats`, `drain`, `shutdown` (`C`) | None |
| `accepted` (`R`) | None |
| `rejected` (`R`) | `error` |
| `prefill_progress` (`R`) | `consumed_tokens` |
| `tokens` (`R`) | `output_index`, `token_ids`, `text_delta` |
| `terminal` (`R`) | `status`, `cause`, `usage`, `state_result`, `text_delta`, `metrics`, optional `error` |
| `resources_released` (`R`) | `released_lease_ids`, `retained_lease_ids` |
| `command_result` (`C`) | `status`, optional `error` |
| `stats` (`C`) | `active_requests`, `completed_requests`, `cancelled_requests` |
| `worker_fault` (`B`) | `error` |
| `drained` (`B`) | None |

The worker must check the submit revisions against its ready revisions.
The first token range has index zero.
An empty EOS array explicitly disables EOS stopping.
`remaining_timeout_ms` is between 1 and 3600000.
`max_tokens` can be zero. The worker then emits no token events.

`shutdown` drains accepted work and closes the worker after cleanup.
After queueing a valid `shutdown`, the input reader stops reading commands.
The sender must treat `shutdown` as the final command for that epoch.
The execution owner settles accepted attempts before it emits `drained` and exits.
An invalid shutdown frame follows the normal validation rules.

## Nested fields

`runtime` contains string fields `python`, `mlx`, and `mlx_lm`.
`memory` contains byte counts `active_bytes`, `peak_bytes`, and `cache_bytes`.
`model_path` is an absolute local path.
`vocab_size` is a positive integer no larger than 2147483648.
All input and output token IDs must be below `vocab_size`.

`limits` contains `max_frame_bytes`, `max_depth`, and `max_commands`.
Their values equal the fixed profile limits above.

`capabilities` has the following fields:

| Field | Value |
| --- | --- |
| `text`, `greedy`, `token_events` | `true` |
| `image`, `sampling`, `exact_append` | `false` |
| `cancel_granularity` | `token` or `token_or_prefill_chunk` |
| `max_active_sequences` | `1` |
| `max_context` | Positive integer no larger than 131072 |
| `max_output_tokens` | Integer from 0 through 65536 |
| `state_schema` | `none` |

`model_manifest` is an identity document under `apxinf-canonical-1`.
Its digest uses the `apxinf-model-identity-v1` domain.
The capability digest uses the `apxinf-capability-v1` domain.
The supervisor must check the manifest against approved local artifacts before it trusts the worker.
A matching self-reported digest alone does not prove model identity.

`template_options` contains only `enable_thinking`, a Boolean.
Each message requires `role` and string `content`.
Allowed roles are `system`, `user`, `assistant`, and `tool`.
Optional message fields are `tool_call_id`, `name`, and `tool_calls`.
Each tool call contains `id`, `type`, and `function`.
Its type is `function`. Its function contains `name` and `arguments`.
Arguments can be an object or a JSON string. The validator does not coerce them.

Each tool contains `type` and `function`.
Its type is `function`.
The function requires `name` and permits `description`, `parameters`, and `strict`.
Description is a string. Parameters is an object. Strict is a Boolean.
Within request content, only parameters and object arguments permit arbitrary JSON keys.
All nested values must remain within the common limits.

`prepared_input` contains at least one token.
`effective_input_tokens` equals the token array length.
`prompt_digest` uses `apxinf-token-prefix-v1` from the main contract.

`usage` contains `input_tokens` and `output_tokens`.
Input count equals the submitted prompt length.
Output count equals the number of reported generated token IDs.
It includes hidden EOS and reasoning tokens. It excludes internal lookahead predictions.
Public API profiles must document their usage mapping.

`state_result` contains `validity`, `consumed_position`, and `history_count`.
Validity is `none` in this profile. The worker retains no session state.
History count equals input count plus reported output count.
Consumed position cannot exceed history count.
The worker may report zero after failure when the consumed position is unknown.
An unknown position must never establish a reusable state.

`metrics` contains `elapsed_ns`, `ttft_ns`, and `peak_memory_bytes`.
These values are non-negative safe integers.
TTFT is zero when the worker produces no token.
TTFT cannot exceed elapsed time.

`text_delta` contains new text from the authoritative decoder.
It can be empty. It must not repeat prior text.
The terminal delta contains any remaining decoder text after finalization.

Each text delta must correspond to token IDs in the current event or earlier events.
An execution failure may report pending tokens only when their decoding completed successfully.
After a decoding failure, the worker must suppress finalization text because the decoder state can contain an unreported token.
An output delivery failure must also suppress finalization text.
An earlier execution or decoding failure retains failed status if output delivery also fails.

The gateway processes that delta before it maps a successful terminal result.
Token events cannot contain an empty token array.

An error contains string `code`, string `message`, `scope`, and `state_validity`.
Scope is `request`, `session`, `worker`, or `service`.
State validity is `unchanged`, `committed`, `invalid`, or `none`.
The main contract defines the allowed error codes.

Command status is `accepted`, `already_registered`, `already_terminal`, or `error`.
Only status `error` contains an error record.
Cancel reason is `user_cancel`, `client_disconnect`, `slow_consumer`, or `deadline_exceeded`.
Stop cause is `stop_sequence` or `tool_calls`.
The worker checks that the token boundary refers to output already produced.
The gateway checks the text-byte cutoff against its decoder output.

## Attempt sequence and settlement

The first event is `accepted` or `rejected`, with sequence zero.
A rejected attempt emits no token or prefill events.
It still emits a failed terminal and settles every assigned capacity lease.
Each later event increases the sequence by one.
Each attempt emits one terminal, followed by one `resources_released` event.
No event can follow that cleanup event.

Cleanup reports every submitted capacity lease exactly once.
`retained_lease_ids` is empty in this stateless profile.
Released IDs must equal the submitted lease set.
The supervisor checks cleanup before it changes its capacity ledger.
It continues to charge allocator reserve separately.

A repeated submit produces a command result and no new attempt event.
An identical payload returns `already_registered`.
A conflicting payload returns `error` with code `invalid_request`.
A control after a terminal returns `already_terminal`.
These responses do not change the attempt event sequence.

## Supervisor operation policy

The supervisor counts every command within its worker epoch.
Before preparation, it requires four unused command records.
These records cover preparation, submission, one control, and shutdown.
If fewer records remain, the supervisor rotates the worker before dispatch.
The limit remains 10000 records.
These lifecycle policies retain `apxinf-worker/2.0` and all existing wire fields.

Rotation waits for current operation settlement and old process exit before replacement loading.
The supervisor checks the replacement model and capability revisions against the original approved revisions.
The service publishes the complete replacement `ready` snapshot after readiness checks pass.
This snapshot includes the new worker epoch and current memory values.
Queued requests retain their order and original deadlines during rotation.
Other worker faults still require a service restart.

`command_id` identifies each preparation operation before an attempt exists.
Cancellation or expiry ends its public request without waiting for the preparation response.
The supervisor then retains the execution position for a fixed 20-second settlement grace.
It consumes and discards the matching preparation response before it dispatches another request.
The response can be `prepared_input` or a preparation error in `command_result`.
Grace expiry or a protocol fault fences the worker.

The supervisor checks the public deadline and cancellation while it writes preparation to stdin.
Each stdin write has a 20-second health timeout.
The preparation settlement grace also covers an unfinished write.
An unconfirmed write or response keeps the execution position occupied.
The supervisor fences the worker when safe settlement cannot complete.

The supervisor also checks the public deadline and cancellation while it writes `submit` to stdin.
Cancellation or expiry ends the public request before admission, even if that write remains incomplete.
The supervisor retains the execution position until resource settlement or worker process reaping.
It continues the original write within its health timeout and the 20-second settlement grace.
It never repeats a partially written command.

After that public result, the supervisor suppresses admission success and output from the attempt.
After `accepted`, it sends at most one cancellation control and consumes the remaining attempt events.
The original public result remains final regardless of the transmitted `remaining_timeout_ms` value.
Write failure, grace expiry, or a protocol fault fences the worker without a second public failure count.

Public deadline checks continue during generation, control transmission, and cleanup waiting.
Expiry publishes an error immediately, before resource settlement.
A parser or worker failure already observed at this decision retains failed status.
Before `accepted`, the supervisor reports this error through the admission result.
After `accepted`, it uses the reserved final event slot.
Later admission, content, and worker results cannot replace this public result.

A prior `stop_generation` does not prevent public expiry.
The supervisor sends at most one stop or cancellation control for each attempt.
Cancellation transmission waits for `accepted` to prevent control processing before attempt registration.
An observed terminal needs no additional control.

The first stop request or public expiry establishes the 20-second settlement grace.
The supervisor establishes this grace when it observes cancellation before acceptance or after terminal.
The grace includes control transmission and never restarts after later events.

The supervisor retains its execution position and leases until checked resource settlement or worker process reaping.
It checks all late frames and records worker timing, memory, and cleanup observations.
It suppresses new public output after public expiry.
Later protocol or process failure fences the worker without a second public outcome.
Public success still requires resource settlement before the original public deadline.

The coordinator sweeps waiting requests every 20 milliseconds, independently of worker execution.
It also sweeps before each enqueue operation.
Each sweep removes cancelled or expired requests and returns their queue capacity.
Each removed request updates its public result counters once.
Remaining requests retain FIFO order.

`Service::shutdown` closes admission and the waiting queue.
The active operation retains its original deadline and cancellation rules.
After active settlement, the coordinator stops and reaps its worker, then exits.
The host must call this operation explicitly.
Dropping external `Arc<Service>` references alone does not establish service shutdown.
The CLI calls `Service::shutdown` when it receives Ctrl-C.

The host must then await `Service::wait_stopped` before it stops the async runtime.
This operation returns success only after the coordinator reaps all owned worker processes.
The operation returns an error if the coordinator fails or cannot reap an owned worker process.
The CLI awaits `Service::wait_stopped` after the HTTP server stops.

## Handwritten validator interface

Rust exposes `decode_frame`, `validate_frame`, `encode_frame`, and `AttemptTracker`.
Python exposes the same names and behavior.
Decoders accept a single payload with an optional final LF.
Transport readers must require LF before passing a complete frame to a decoder.
Encoders always append LF.
Integer fields reject negative-zero spelling and floating-point spelling.

`parse_document` checks bounded JSON without a frame schema.
It applies the integer-literal and floating-point rules from the main contract.
`validate_command_envelope` checks command routing, exact outer keys, and capacity lease IDs.
It does not accept request payload values.
The worker can reject invalid content when that envelope check succeeds.
An invalid envelope remains a channel fault.
Repeated prepare commands return `already_registered`. The worker does not retain or replay large prepared results.

The caller constructs `AttemptTracker` from a checked submit frame.
Its `observe` operation checks identity, event order, output ranges, terminal counts, and lease settlement.
It rejects invalid events before changing its state.
Both implementations consume the same original conformance fixtures.
No code generator creates the types, validators, or fixtures.

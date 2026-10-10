# ApxInf serving interface contract

Status: Design draft 0.1. Date: 2026-10-09.

This document defines the proposed serving interfaces.
The [serial profile](serial-profile-v0.1.md) selects the first implementation subset.
The full contract includes future capabilities.
The existing v1 protocols retain their existing behavior.
The [system design](../serving-system-design-20261009.md) explains the design choices.
The [implementation plan](implementation-plan.md) defines the delivery stages.

## Document rules

This document uses ASD-STE100 Issue 9 principles for English structure.
The project does not claim a complete dictionary review or certification.
The [review record](README.md) states the scope of the checks.

Each rule ID identifies one contract obligation.
In this document, `must` states an obligation.
The word `can` states a capability.
The word `may` states a permitted choice.
This requirement notation is a project convention.

## Original implementation

**OR-01.** Authors must write all new first-party code by hand.

**OR-02.** Authors must not copy, translate, port, or lightly rewrite implementation code from another repository.

**OR-03.** Authors must write tests and fixtures from this contract and ApxInf behavior.

**OR-04.** Authors may study external designs and call approved dependencies through public APIs.

**OR-05.** Each dependency decision must identify the library, version, public API, purpose, and replacement boundary.

The MLX adapter can call MLX-LM through its public API.
That call does not make the MLX-LM implementation ApxInf code.
The adapter must not copy the MLX-LM scheduler into this repository.
The same rule applies to oMLX, llama.cpp, Ollama, vLLM, and SGLang.

## Terms

| Term | Meaning |
| --- | --- |
| client | A program that sends a public API request. |
| request | One public inference operation. |
| attempt | One worker execution of a request. |
| sequence | One active autoregressive token stream. |
| session | A named history and its reusable model state. |
| model revision | The immutable identity of model artifacts and their execution profile. |
| model instance | One resident model within a worker epoch. |
| worker | A process that owns a model instance. |
| worker epoch | The identity of one worker lifetime. |
| execution owner | The fixed worker thread that operates model objects. |
| canonical input | The exact tokens and media description that the model consumes. |
| consumed position | The number of input tokens that the model state contains. |
| state snapshot | An immutable model state at a defined input boundary. |
| resume capsule | Model state and control data for a later continuation. |
| capacity reservation | A claim on memory within a resource domain. |
| sequence credit | Permission for one sequence to enter the active set. |
| compute permit | Permission for a worker to submit device work. |
| execution position | Exclusive access for one operation to the serial worker. |
| terminal result | The final logical result of an attempt or request. |
| resource settlement | Release or transfer of all resources that an attempt owns. |
| logical cache block | A unit for state storage and prefix lookup. |
| paged attention | An attention operation that reads cache data through a physical block map. |

**TM-01.** Authors must use these terms with these meanings across documents, types, errors, and tests.

**TM-02.** A logical cache block must not imply support for paged attention.

## Component ownership

| Component | Owns | Does not own |
| --- | --- | --- |
| `Gateway` | Public API input, text output, stop matching, output parsers. | Model state or device tensors. |
| `Coordinator` | Request records, queues, deadlines, attempt records. | Token execution or model tensors. |
| `ModelSupervisor` | Worker processes, model instances, epochs, recovery. | Per-token schedule. |
| `DeviceArbiter` | Capacity reservations and compute permits. | Model operations. |
| `PromptAdapter` | Input format, tokenizer, template, processor identity. | Device schedule. |
| `TextWorker` | Execution owner, active sequences, model state, sequence credits. | Public HTTP responses. |
| `PolicyWorker` | Policy inference and observation order. | The text sequence schedule. |
| `StateStore` | Snapshot metadata, storage, references, compatibility checks. | Mutable active state. |

`Gateway` and `Coordinator` are modules in the service process.
`ModelSupervisor` starts each worker.
The worker constructs each model object on its execution owner.
The service process does not pass model objects between threads.

**OW-01.** Each mutable model state must have one execution owner.

**OW-02.** `Coordinator` must use worker credits instead of a second token scheduler.

**OW-03.** `StateStore` must use the worker adapter for device state export and restore.

**OW-04.** `PolicyWorker` must keep its observation schedule separate from the text sequence schedule.

## Service interfaces

The names below describe semantic operations.
They are not copied library signatures.
Each implementation uses types from this contract.

| Owner | Operation | Result |
| --- | --- | --- |
| `PromptAdapter` | `prepare_input` | `PreparedInput` and its artifact identities. |
| `ModelSupervisor` | `resolve_model` | A fixed model revision. |
| `ModelSupervisor` | `ensure_ready` | A worker epoch and capability revision. |
| `Coordinator` | `submit_request` | A request ID and result stream. |
| `Coordinator` | `cancel_request` | An idempotent control result. |
| `DeviceArbiter` | `reserve_capacity` | A capacity lease or a capacity error. |
| `DeviceArbiter` | `acquire_compute` | A compute permit or a wait result. |
| `TextWorker` | `submit` | Attempt events. |
| `TextWorker` | `stop_generation` | A completion request at an output boundary. |
| `TextWorker` | `cancel_request` | An abort request. |
| `TextWorker` | `drain` | A worker result after active work and cleanup. |
| `StateStore` | `find_prefix` | A candidate handle and a restorable prefix length. |
| `TextWorker` | `restore_prefix` | Model state for a new request. |
| `TextWorker` | `resume_request` | Model state and controls for the same request. |
| `TextWorker` | `commit_session` | A new session version. |
| `PolicyWorker` | `infer_observation` | An action result with observation identity. |

**IF-01.** Each operation must state its input, owner, result, limits, errors, and resource effects before implementation.

**IF-02.** Unsupported operations must return `unsupported_feature` before model mutation.

## Identity and primitive types

| Field | Type | Owner and rule |
| --- | --- | --- |
| `request_id` | Lowercase UUID string. | `Coordinator` creates it once per request. |
| `attempt` | Positive safe integer. | `Coordinator` starts at 1. |
| `worker_epoch` | Lowercase UUID string. | `ModelSupervisor` creates it for each worker start. |
| `command_id` | Lowercase UUID string. | The service creates it for each command. |
| `lease_id` | Lowercase UUID string. | The resource owner creates it. |
| `model_revision` | Lowercase SHA-256 hex string. | The model identity manifest defines it. |
| `capability_revision` | Lowercase SHA-256 hex string. | The capability document defines it. |
| `session_id` | Lowercase UUID string. | The service creates it within a trusted namespace. |
| `session_version` | Non-negative safe integer. | The session owner changes it after a commit. |
| `event_seq` | Non-negative safe integer. | The worker counts events for one attempt. |
| `token_id` | Integer from 0 through 2147483647. | The worker also checks the model vocabulary. |

A safe integer has a value from 0 through 9007199254740991.
A byte count uses bytes.
A timeout uses integer milliseconds.
A token count counts model tokens, not text characters.
Optional fields are absent unless their schema permits `null`.

Public requests and worker frames permit negative integers and finite floating-point numbers in arbitrary JSON content.
Integer literals without a decimal point or exponent must have an absolute value no larger than 9007199254740991.
A decimal point or exponent selects finite floating-point parsing.
The parsers must reject out-of-range integer literals before they return a parsed value.
String values retain their text.
Negative-zero spelling remains valid in arbitrary JSON but invalid in integer fields.

**ID-01.** Every request command and event must carry `(worker_epoch, request_id, attempt)`.

**ID-02.** An admitted request must retain its model revision and capability revision.

**ID-03.** The service must not reuse a session ID after reset or invalidation.

**ID-04.** Validators must reject a Boolean value in an integer field.

## Identity digests

**DG-01.** Implementations must not use language object hashes for persistent identity.

The model identity manifest contains the following data:

- The artifact list with relative paths and content digests.
- The model configuration digest.
- The tokenizer, template, and processor digests.
- The adapter revision.
- The runtime identity.
- The precision and kernel profile.
- The adapter or LoRA identity, if present.
- The position configuration.

The format `apxinf-canonical-1` encodes identity documents as compact UTF-8 JSON.
Identity documents contain strings, Booleans, safe integers, arrays, and objects.
They do not contain floating-point values or `null`.
Object keys use ASCII names and sort by byte value.
Array order remains unchanged.
Artifact entries sort by relative path before encoding.

Strings retain their Unicode code points.
The encoder rejects unpaired surrogates.
It escapes quotation marks and backslashes with a backslash.
It encodes each control character from U+0000 through U+001F as lowercase `\u00xx`.
It writes other characters as UTF-8 without slash escaping.
It adds no byte-order mark, whitespace, or final newline.

The digest input contains the domain name, one zero byte, and the encoded document.
The model domain name is `apxinf-model-identity-v1`.
The capability domain name is `apxinf-capability-v1`.

The token-prefix digest uses a separate binary format.
Its domain name is `apxinf-token-prefix-v1`.
One zero byte follows the domain name.
An unsigned 64-bit little-endian token count follows that byte.
Unsigned 32-bit little-endian token IDs follow the count.
SHA-256 produces the lowercase hexadecimal digest.

**DG-02.** Rust and Python must pass identical digest vectors before a protocol release.

**DG-03.** The legacy v1 digest format must remain unchanged.

## Capability contract

The capability document contains these constraint groups:

| Group | Required meaning |
| --- | --- |
| Operations | The supported operation names. |
| Input | Input variants and effective context rules. |
| Generation | Supported controls and permitted combinations. |
| Execution | Active sequence limit and prefill step limits. |
| Cancellation | The smallest boundary where the worker can stop. |
| State | Session, snapshot, restore, fork, and trim support. |
| Identity | Model, adapter, parser, and state schema revisions. |
| Memory | Estimator revision and resource domains. |
| Quality | The tested execution profile. |

The service capability set intersects four constraint sets:

1. Worker capabilities.
2. `PromptAdapter` capabilities.
3. The public API profile.
4. Deployment limits.

**CP-01.** Capability documents must remain immutable within a worker epoch.

**CP-02.** Live credit counts must remain separate from capability documents.

**CP-03.** The service must reject an unsupported parameter combination before execution.

**CP-04.** A text worker must not claim tool compatibility from token output alone.

**CP-05.** The worker must report the actual provider and quality profile.

## Prepared input

`PreparedInput` binds input data to its model revision and prompt artifacts.
The initial text variant contains `token_ids` and `effective_input_tokens`.
Its token count equals the length of `token_ids`.
The worker checks this equality.

The later vision variant contains tokens and owned media descriptors.
Each descriptor defines content identity, format, shape, position, byte count, and lifetime.
The vision adapter defines the effective input count.
P2c must define that descriptor schema before implementation.

**IN-01.** One `PromptAdapter` must produce the authoritative input for each request.

**IN-02.** The service must not apply a chat template twice.

**IN-03.** The context check must include effective input tokens and the maximum output allowance.

**IN-04.** The context check must use the smaller model limit and deployment limit.

**IN-05.** The service must not normalize Unicode or reorder tools to improve cache matches.

## Request lifecycle

`Coordinator` owns the public request phase.
The worker reports attempt events without changing that record directly.
Cancellation sets a pending control flag until the execution result settles its outcome.

| Phase | Meaning | Next phase |
| --- | --- | --- |
| `received` | The gateway accepts input bytes. | `validated`, `terminal`. |
| `validated` | The gateway checks public input. | `waiting_model`, `queued`, `terminal`. |
| `waiting_model` | The request waits for its fixed model revision. | `queued`, `terminal`. |
| `queued` | The request waits for admission. | `admitted`, `terminal`. |
| `admitted` | The request holds the required reservations. | `prefilling`, `decoding`, `terminal`. |
| `prefilling` | The worker consumes the input prefix. | `decoding`, `terminal`. |
| `decoding` | The worker produces new output tokens. | `terminal`. |
| `terminal` | The public result is final. | None. |

**LC-01.** A request must enter the public terminal phase once.

**LC-02.** Cleanup must use a separate status from the public phase.

**LC-03.** Only an allowed replay may return a non-terminal request to `queued` after an attempt failure.

**LC-04.** Replay must use a new attempt number and preserve the original public deadline.

A complete prefix restore can permit a transition from `admitted` to `decoding`.
That transition requires a valid continuation capsule.
It does not permit the worker to omit state checks.

## Worker protocol

The proposed protocol name is `apxinf-worker/2.0`.
This draft fixes its semantics before implementation.
P0 must add complete field tables and positive and negative fixtures for the first implemented subset.
P0 does not release a protocol with undefined payload fields.

The transport uses UTF-8 JSON Lines.
Each line contains one object followed by LF.
Each frame contains `protocol`, `kind`, and `worker_epoch`.
Each command also contains `command_id`.
Each request event also contains `request_id`, `attempt`, and `event_seq`.

**WR-01.** The worker and supervisor must agree on an exact supported protocol version before `ready`.

**WR-02.** Validators must reject duplicate keys, unknown fields, non-finite numbers, and invalid primitive types.

**WR-03.** The startup contract must fix frame bytes, nesting depth, collection sizes, and retained command records.

**WR-04.** Stdout must contain protocol frames only.

**WR-05.** Stderr must contain diagnostic logs only.

**WR-06.** A framing or envelope fault must invalidate the worker channel.

**WR-07.** Valid envelopes with invalid request content must produce request errors without model mutation.

The first implementation retains the existing bounded transport sizes where possible.
It does not silently increase limits for media payloads.
Oversized payloads require a defined descriptor transport or explicit rejection.

### Commands

| Kind | Meaning |
| --- | --- |
| `submit` | Register one attempt with fixed input, limits, and identities. |
| `cancel_request` | Ask the worker to abort an attempt. |
| `stop_generation` | Ask the worker to finish generation for a completion cause. |
| `drain` | Stop new admission to the worker. |
| `shutdown` | Stop the worker after the selected shutdown policy. |
| `query_stats` | Request a defined statistics snapshot. |

`submit` includes a remaining timeout and acquired capacity lease IDs.
The worker grants a sequence credit before active execution.
Control commands use reserved queue capacity.
The execution owner alone changes model state.

### Events

| Kind | Meaning |
| --- | --- |
| `ready` | The worker reports fixed identities, limits, and capabilities. |
| `accepted` | The worker registers the attempt. |
| `rejected` | The worker refuses the attempt before model mutation. |
| `prefill_progress` | The worker reports a safe consumed position. |
| `tokens` | The worker reports a contiguous range of output tokens. |
| `terminal` | The worker reports the final execution result. |
| `resources_released` | The attempt owns no remaining resources. |
| `command_result` | The worker reports the result of a control command. |
| `stats` | The worker reports defined counters and gauges. |
| `worker_fault` | The worker reports a fault outside one request. |
| `drained` | The worker has no active attempts or incomplete cleanup. |

### Order and duplicate controls

**EV-01.** The first attempt event must use `event_seq = 0`.

**EV-02.** Each next attempt event must increase `event_seq` by one.

**EV-03.** Events from different attempts may interleave.

**EV-04.** Each registered execution key must produce one `terminal` event.

**EV-05.** A rejected attempt must produce a terminal result and settle every assigned lease.

**EV-06.** After `terminal`, the worker must emit no token or prefill events for that attempt.

**EV-07.** Each `tokens` event must contain a non-empty token array and its first `output_index`.

**EV-08.** The next token range must start at the previous range end.

**EV-09.** The receiver must treat sequence gaps, duplicate events, and conflicting token ranges as protocol faults.

**EV-10.** Repeated controls must have idempotent effects.

**EV-11.** A duplicate submit must not start another execution.

A duplicate submit returns a `command_result`, not another attempt event stream.
An identical payload returns `already_registered`.
A conflicting payload returns a command error without changing the original attempt.
The worker checks the execution key as well as the command ID.

**EV-12.** A control after an attempt terminal must return `already_terminal` without another attempt terminal.

The protocol defines a bounded command history per worker epoch.
Before that history reaches its limit, the supervisor drains the worker.
It does not silently forget duplicate detection for a live epoch.

**EV-13.** The supervisor must count commands within each worker epoch.

**EV-14.** Before dispatch, the supervisor must reserve command capacity for the complete operation, controls, and shutdown.

Planned rotation stops the old worker before the supervisor loads its replacement.
The replacement must pass the original model and capability identity checks.
The service exposes its new epoch only after these checks pass.
Rotation does not replay completed requests or authorize recovery from other worker faults.

## Result and stop semantics

`ExecutionResult` reports the worker result.
It contains `status`, `cause`, `usage`, and `state_result`.
A failure also contains `error`.

| Status | Cause examples |
| --- | --- |
| `completed` | `eos`, `length`, `stop_sequence`, `tool_calls`. |
| `cancelled` | `user_cancel`, `client_disconnect`, `slow_consumer`. |
| `expired` | `deadline_exceeded`. |
| `failed` | A stable error code. |

**RS-01.** A completed result must not contain a failure error.

**RS-02.** A failed result must contain a failure error.

**RS-03.** An `accepted` event must not claim that prefill is active.

`Gateway` owns text stop matching and public output parsers.
Its `PublicResult` can differ from the worker execution cause.
For example, a stop match can exist in tokens that precede a worker length result.
The gateway consumes all earlier token events before it maps the terminal result.

**RS-04.** For eligible public success, a text stop match must select completed status.

**RS-05.** `stop_generation` must carry the completion cause and a checked output boundary.

The output boundary contains a token range and a text-byte cutoff.
The cutoff refers to the UTF-8 output of the declared incremental decoder.
The gateway keeps that cutoff separate from the model consumed position.
P0 defines the exact boundary fields and decoder fixtures.

**RS-06.** The gateway must hide output after the selected boundary, including output already in transit.

**RS-07.** A stop inside a token must not imply a restorable model boundary.

**RS-08.** The worker must not trim recurrent state without a tested trim capability.

A stop match can replace a normal `length` or `eos` cause.
It cannot replace worker failure, expiry, or cancellation.
The worker serializes control decisions and terminal decisions on its execution owner.
The first terminal decision ends that attempt.
Later controls cannot change its result.
The gateway waits for the execution result before it publishes a successful final result.

If retained state contains an unwanted tail, the service cannot claim exact session continuation.
It discards or invalidates that state unless a tested operation restores the required boundary.
The public result still reports the completed stop correctly.

## Session contract

A session binds a namespace, model revision, input history, state handle, and version.
The service creates a session with version 0 and no consumed history.
The first successful generation commit creates version 1.
Each later successful commit increases the version by one.
Reset closes the session ID.

An append supplies the following fields:

- `session_id`.
- `expected_version`.
- `expected_prefix_count`.
- `expected_prefix_digest`.
- The full new canonical input.

**SS-01.** The session owner must check identity, version, and exact prefix before model mutation.

**SS-02.** The owner must hold an exclusive write lease during an append.

**SS-03.** Other append requests must receive `session_conflict` while that write lease exists.

**SS-04.** A pre-mutation rejection must preserve the committed session.

**SS-05.** An irreversible partial mutation must invalidate the session after failure or cancellation.

**SS-06.** The worker must complete device work before it publishes a new session version.

**SS-07.** A commit must publish the history, state handle, position, and version as one logical change.

**SS-08.** A missing explicit session must return `session_not_found` without silent recompute.

A stateless prefix lookup can return a miss and compute the full input.
That policy does not change explicit session semantics.
During an in-place append, the prior committed record remains unavailable for use.
Its existence does not imply rollback support.

## State restore contract

The resume capsule reports the following metadata:

| Field | Meaning |
| --- | --- |
| `state_schema` | The version of the state layout. |
| `model_revision` | The bound model identity. |
| `history_digest` | The committed canonical history identity. |
| `history_count` | The number of tokens in that history. |
| `consumed_position` | The number of tokens present in model state. |
| `pending_input_count` | The number of committed tokens not yet consumed. |
| `next_logits_state` | `stored`, `reconstructible`, or `absent`. |
| `logical_state_bytes` | The size of logical state data. |
| `physical_lease_ids` | The capacity owners for actual allocations. |

**ST-01.** The adapter must keep history count and consumed position distinct.

**ST-02.** Each snapshot must align all model-state components at one valid boundary.

**ST-03.** `find_prefix` must return a restorable boundary, not only the longest matching token prefix.

**ST-04.** `restore_prefix` must create fresh request controls for RNG, stop matching, output parsers, and stream progress.

**ST-05.** `resume_request` must restore controls only for the same suspended request.

**ST-06.** Pending tokens must belong to the selected canonical history before the worker consumes them.

**ST-07.** The worker must reject continuation when required state is absent.

**ST-08.** Active state must not share mutable storage with an immutable snapshot.

**ST-09.** State export must report device completion before the store publishes the snapshot.

**ST-10.** A cache hit must pass identity, schema, namespace, and position checks.

The memory store does not require a disk store.
The disk format needs its own version and atomic commit rules before P4.
Neither store implies a paged attention kernel.

## Resource contract

| Resource | Release condition |
| --- | --- |
| Capacity reservation | Memory leaves the domain or transfers to another accounted owner. |
| Sequence credit | The sequence leaves the worker active set. |
| Compute permit | All device work under the permit completes. |

Each lease identifies its kind, owner, epoch, resource domain, amount, and state.
Lease states are `reserved`, `held`, `pending_reclaim`, and `released`.
Shared allocations have one physical capacity charge.
Logical references do not create extra physical charges.

Before dispatch, `Coordinator` assigns submitted capacity leases to the attempt in the resource ledger.
The worker checks these assignments before execution.
A rejection still needs resource settlement even when the worker allocates no tensors.
An uncertain dispatch keeps those reservations until the supervisor resolves the worker state.

**RE-01.** Admission must reserve capacity before it acquires a compute permit.

**RE-02.** The worker must not wait for memory capacity while it holds a compute permit.

**RE-03.** A logical terminal result must not release capacity by itself.

**RE-04.** Pending reclaim must remain inside the device budget.

**RE-05.** Retained session or snapshot memory must transfer to its new budget owner before request cleanup completes.

**RE-06.** `resources_released` must report released leases and retained lease transfers.

**RE-07.** The service must reject a release claim with an unknown lease or wrong owner.

**RE-08.** Repeated settlement must not return capacity twice.

**RE-09.** Allocator-held memory must remain in the runtime reserve after request cleanup.

The serial profile checks memory limits before worker startup.
The memory budget and sequence reservation must each be between 1 and 9007199254740991 bytes.
The sequence reservation must not exceed the memory budget.
Budget arithmetic must reject integer overflow.

The supervisor peak counter retains the largest checked allocator measurement across worker epochs.
Each `ready` contributes the larger of `peak_bytes` and the sum of `active_bytes` and `cache_bytes`.
Each checked worker terminal contributes its reported peak, even when public expiry or later cleanup failure occurs.
This counter does not represent current residency or whole-system memory use.

P1 holds the compute permit for a whole request.
Later adapters can release it after a safe execution step.
An asynchronous submission does not mark that step complete.
Model load, warmup, and restore use the same device policy.
The serial adapter must synchronize its captured generation stream before it releases request state.
Synchronizing another default stream does not establish this completion.
The [memory calibration profile](memory-calibration-v0.1.md) defines offline observations without changing worker frames or serving budgets.

## Host pressure admission

Host pressure is the operating system's memory pressure signal.
It is separate from MLX allocations and capacity reservations.
The local profile defines its sensor, sampling period, freshness limit, and recovery rule.
The worker protocol does not carry this host policy.

**HP-01.** An enabled host policy must reject new work when its pressure sample is missing, invalid, or stale.

**HP-02.** Admission must check pressure before enqueue, before preparation, and before generation dispatch.

**HP-03.** A pressure rejection must use `capacity_unavailable` with HTTP 503.

**HP-04.** Pressure changes must not release resources from active generation or change its terminal result.

**HP-05.** Model loading must require an allowed pressure snapshot before start and after loading.

**HP-06.** Pressure recovery must not change worker identity or restore availability after shutdown or worker failure.

Requests rejected before enqueue do not enter request outcome counters.
An enqueued pressure rejection produces one failed outcome and settles its assigned resources.
Readiness reports worker availability and host admission separately.
A disabled host policy remains explicit in readiness and metrics.
This policy does not establish a hard memory limit or a complete allocation estimate.
Preparation retains its execution position until its matching response or checked worker recovery.

## Deadline and cancellation

**DC-01.** `Coordinator` must start the public deadline at request ingress with its monotonic clock.

**DC-02.** A submit command must carry `remaining_timeout_ms` rather than a foreign monotonic timestamp.

**DC-03.** The worker must start a local timeout when it receives the command.

**DC-04.** The original coordinator deadline must remain authoritative.

The public deadline remains active until the coordinator publishes a public terminal result.
Worker completion and stop matching do not end this deadline.
Expiry must end the public request without waiting for resource settlement.
A parser or worker failure already observed at that decision retains failed status.
Later worker events must not change a published public terminal result.

**DC-05.** Cancellation controls must bypass a full ordinary submit queue.

**DC-06.** The execution owner must process cancellation at a declared safe boundary.

**DC-07.** Every output stream must have byte and token limits.

**DC-08.** One slow client must not block the protocol reader for other attempts.

The service measures control acknowledgement, execution stop, and resource settlement separately.
An unresponsive worker can require process shutdown.
That shutdown invalidates its non-persistent sessions.
The service reports the affected requests explicitly.

Preparation precedes attempt registration and uses `command_id` as its operation identity.
Its public request can end before the preparation operation settles.
The public deadline and cancellation checks include command transmission.
The serial profile defines the settlement grace and command capacity for this operation.

**DC-09.** Request expiry during preparation must not by itself become a worker fault.

**DC-10.** The coordinator must retain the execution position until preparation settles or worker recovery completes.

**DC-11.** The coordinator must remove cancelled and expired waiting requests independently of active execution.

**DC-12.** Queue removal must preserve FIFO order for remaining requests and update counters once.

**DC-13.** Generation must check its public deadline during execution, control transmission, and cleanup waiting.

**DC-14.** Public expiry must suppress later public output without returning unsettled resources.

The serial profile permits at most one stop or cancellation control for each attempt.
The public outcome and control transmission have separate state.
The first observed cancellation starts the settlement grace even when the worker cannot receive a control.
Acceptance and terminal events must not restart that grace.
Late worker events still require protocol checks and resource settlement.
They can update worker measurements but cannot record another public outcome or public output interval.

The serial service defines an explicit shutdown operation for its host.
Shutdown closes admission and the waiting queue before the coordinator exits.
Active work retains its original deadline and cancellation rules.
The coordinator stops and reaps its worker after active settlement.
External reference counts do not define the shutdown policy.

The host must await `Service::wait_stopped` before it stops the async runtime.
This operation returns success only after the coordinator reaps all owned worker processes.
The operation returns an error if the coordinator fails or cannot reap an owned worker process.

## Faults and recovery

The error record contains `code`, `message`, `scope`, and `state_validity`.
The message contains no credentials, private prompts, or traceback.
The scope is `request`, `session`, `worker`, or `service`.
State validity is `unchanged`, `committed`, `invalid`, or `none`.

| Code | Default HTTP status before stream headers |
| --- | --- |
| `invalid_request`, `unsupported_feature`, `context_limit` | 400 |
| `model_not_found`, `session_not_found` | 404 |
| `session_conflict`, `prefix_mismatch` | 409 |
| `queue_full`, `quota_exceeded` | 429 |
| `capacity_unavailable`, `model_unavailable`, `worker_lost` | 503 |
| `deadline_exceeded` | 504 |
| `protocol_fault`, `internal_error`, `state_invalid` | 500 |

**FR-01.** After stream headers, the gateway must use the selected API profile for a stream error.

**FR-02.** The gateway must not report a successful finish for a failed stream.

**FR-03.** The supervisor must fence a lost worker epoch before it creates replacement request results.

**FR-04.** The service must ignore late events from a fenced epoch.

**FR-05.** A retryable error must not authorize replay by itself.

**FR-06.** Automatic replay must require no external output, no committed mutation, and an explicit replay policy.

## Observability contract

These measurements retain `apxinf-worker/2.0` and all existing wire fields.
Service durations start from recorded coordinator events and use its monotonic clock.
Enqueue time follows HTTP input reading and normalization in the serial profile.
It does not represent HTTP ingress or client request start.
The operation label is `generate` or `count_tokens`.

| Histogram | Start | End |
| --- | --- | --- |
| `apxinf_queue_wait_seconds` | Successful enqueue. | Queue removal for execution or a public terminal result. |
| `apxinf_preparation_seconds` | Start of preparation transmission. | Matching preparation response or successful worker process recovery. |
| `apxinf_time_to_first_worker_event_seconds` | Successful enqueue. | First checked, non-empty `tokens` event at the coordinator. |
| `apxinf_time_to_first_public_output_ready_seconds` | Successful enqueue. | First eligible output enters the public result channel. |
| `apxinf_worker_output_event_interval_seconds` | Previous checked `tokens` event. | Next checked `tokens` event for the same attempt. |
| `apxinf_worker_terminal_to_settlement_seconds` | Checked worker terminal event. | Observed request resource settlement. |
| `apxinf_public_terminal_to_settlement_seconds` | Public terminal result. | Observed request resource settlement. |
| `apxinf_service_request_seconds` | Successful enqueue. | Public terminal result. |
| `apxinf_worker_first_token_seconds` | Worker attempt start. | First token, from the existing worker `ttft_ns` value. |

Eligible public output is non-empty `Text` or a complete, checked `Tool` call.
The channel must accept that output before the coordinator records its first occurrence.
Heartbeats, roles, usage records, and terminal markers do not qualify.
Output readiness does not establish an HTTP flush or client receipt.
Worker event intervals include transport and event aggregation effects.
They do not establish model token-step time or client ITL.

Preparation duration includes transmission, template work, tokenization, and response waiting.
Cancellation does not end this duration while preparation still occupies the execution position.
A matching late response ends preparation even when the public request already ended.
Successful process recovery can end preparation when no response arrives.
An unsuccessful reap does not establish an end timestamp.

Settlement occurs after a checked cleanup event, settled preparation, or successful process recovery.
Queue removal establishes settlement when the request owns no execution position or other resources.
If settlement precedes the public terminal result, its public-terminal settlement duration is zero.
Otherwise, that duration is the difference between those two timestamps.
Worker-terminal settlement duration requires an observed worker terminal event.

`apxinf_cleanup_pending` counts public terminal requests whose resource settlement remains incomplete, grouped by operation.
Each such request increments the gauge once and decrements it only after observed settlement.
Failed process recovery leaves its unresolved request in this gauge.
The gauge does not count allocator memory that already transferred to the runtime reserve.

**OB-01.** Implementations must not subtract timestamps from different process clocks.

**OB-02.** Missing events must not produce zero-duration samples.

**OB-03.** A request must record each first-event or terminal duration at most once.

**OB-04.** Each consecutive worker event pair must produce at most one interval sample.

**OB-05.** A failed process reap must not create a resource settlement sample.

The worker first-token histogram converts `ttft_ns` to seconds without mixing clocks.
It records a sample only when the worker terminal reports at least one output token.
The first worker event has no interval sample.
Stages can overlap, so their durations do not form an additive request timeline.

`apxinf_request_outcomes_total` uses operation and public status labels.
Status is `completed`, `cancelled`, `failed`, or `expired`.
It covers the same queued requests as the existing `apxinf_requests_total` counter.
For each status, the sum across operations equals the existing status total.
A later worker fault does not add another public outcome after cancellation or expiry.
Requests rejected before enqueue do not enter these counters.

All new duration histograms use the following fixed upper bounds in seconds:

```text
0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5,
5, 10, 20, 30, 60, 120, 300, 600, 3600, +Inf
```

Each histogram exports cumulative `_bucket` values, `_sum`, and `_count`.
Duration histograms use the operation label and the required bucket label `le`.
The service request histogram also uses the four public status values.
The cleanup gauge uses only the operation label.

Request IDs, epochs, revisions, paths, tool names, and error messages remain outside metric labels.
Existing metric names and meanings remain available for compatibility.
Bucket boundaries define measurement resolution, not accepted latency targets.

Legacy `apxinf_queue_seconds_sum` covers requests selected for preparation.
Legacy `apxinf_request_seconds_sum` covers generation attempts with checked worker settlement.
Their existing coverage remains unchanged, without added histogram buckets or counts under those names.

### Diagnostic output

Diagnostic records describe events but do not establish request outcomes or resource settlement.
Metrics and checked lifecycle events retain those meanings when diagnostic records are missing.
The local profile defines fixed diagnostic buffer limits and loss counters.

**OB-06.** Diagnostic output must not wait for its destination on request, supervision, or worker execution paths.

**OB-07.** A diagnostic write failure must not prevent readiness withdrawal, resource recovery, or shutdown completion.

**OB-08.** Diagnostic queues and record sizes must have fixed bounds.

**OB-09.** The service must expose diagnostic loss and write failures without requiring a diagnostic write.

The supervisor drains worker stderr independently from protocol stdout.
Diagnostic congestion can discard records, including records during shutdown.
The service must not wait indefinitely for diagnostic delivery before process exit.
These rules add no worker fields and do not change public result semantics.

## Interface consistency

**IC-01.** This document must remain the semantic source for the serving interfaces.

**IC-02.** Authors must write Rust and Python types and validators by hand from this contract.

**IC-03.** Both validators must use the same original valid and invalid fixtures.

**IC-04.** Each implementation must decode frames from the other implementation.

**IC-05.** Both implementations must produce identical identity digests.

**IC-06.** Each protocol change must update the contract, both implementations, and the fixture corpus together.

**IC-07.** An additive wire change must use an explicitly supported protocol version.

**IC-08.** A new capability must include its supported parameter combinations and error cases.

**IC-09.** Public API profiles must map to these semantics instead of creating separate execution rules.

### Required conformance cases

| Case | Required observation |
| --- | --- |
| Basic text result | One terminal result and one resource settlement. |
| Duplicate or unknown fields | Both validators reject the same frame. |
| Boolean integer | Both validators reject the value. |
| Token or event gap | The service fences the faulty channel. |
| Repeated cancel | No duplicate terminal result or capacity return. |
| Duplicate submit | A command result without another attempt stream. |
| Rejection after reservation | All assigned capacity leases reach settlement. |
| Stop inside a token | Public completion retains its stop cause without false state alignment. |
| Length result after stop match | The public result preserves the earlier output boundary. |
| Stop with worker failure | The public result preserves the failure. |
| Stop with cancellation | The execution owner's first terminal decision determines the result. |
| Two session appends | Only one request owns the write lease. |
| Partial state mutation | The service does not reuse invalid state. |
| Retained session after terminal | Capacity transfers to the session owner. |
| Worker loss during cleanup | Capacity remains reserved until process recovery completes. |
| Native and MLX positions | Each adapter preserves its true consumed position. |
| Prefix restore | New request controls do not contain previous parser or RNG state. |
| Request resume | The same request continues with its saved controls. |
| Slow client | Other request streams continue. |
| Version mismatch | Startup fails before model requests. |
| Command history threshold | Planned rotation preserves identities and exposes a new epoch without replay. |
| Expiry during preparation | The request expires before operation settlement without immediate worker loss. |
| Late preparation result | The coordinator discards the result before it dispatches another request. |
| Preparation grace expires | The supervisor fences the worker and retains uncertain resources until process exit. |
| Cancelled or expired queue entry | Independent removal returns queue capacity without waiting for active execution. |
| Queue-only termination | Waiting duration and public outcome each have one sample. |
| Preparation expiry before response | Preparation and pending cleanup remain open until safe settlement. |
| Empty or hidden output | Missing public output does not create a first-output sample. |
| Aggregated token events | Intervals count event pairs without claiming token-step latency. |
| Settlement before public success | Public-terminal settlement duration is zero. |
| Failed process reap | Pending cleanup remains visible without a settlement sample. |
| Outcome aggregation | Operation totals equal each legacy status total. |
| Histogram export | Fixed cumulative buckets, count, sum, and labels match this contract. |

## Decisions before code

P0 closes the following details for the first implementation:

1. Complete command and event field tables.
2. Exact frame, queue, byte, and depth limits.
3. Output-boundary fields and stop fixtures.
4. A model capability document and its digest vectors.
5. Request and lease transcripts for success and failure.
6. The first client API profile and target model.

These open details do not permit silent implementation choices.
Each owner records the decision in this contract before dependent code.
Later stages add only the fields and capabilities that their tests cover.

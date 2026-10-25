# Audio serving contract 0.1

Status: A0 contract, 2026-10-25. This document defines the standalone audio profile.
The profile does not change existing text, policy, or v1 protocols.
The service supports one resident model, one task, and one execution position.
The operator selects ASR or TTS before worker startup.

## Terms and owners

| Term | Meaning and owner |
| --- | --- |
| ASR | Automatic speech recognition of one completed recording. |
| TTS | Text-to-speech generation from one complete text input. |
| attempt | One execution, identified by worker epoch, request ID, and attempt number. |
| worker epoch | A lowercase UUID for one worker lifetime. The supervisor creates it. |
| execution owner | The worker thread that loads, operates, and releases model objects. |
| media grant | Private permission to open one service-owned file for one attempt. |
| media lease | Accounted ownership of media bytes. Ownership can transfer to a response. |
| terminal | The final worker result. A terminal does not release the execution position. |
| settlement | Release or transfer of attempt resources. The supervisor checks settlement separately. |
| public outcome | The single final client result, including expiry or cancellation. |

The lifecycle actor owns the child, readiness, queue, deadlines, cancellation, and execution position.
The HTTP handler owns ingress and response transmission.
The worker control reader handles commands without accessing model objects.
Only the execution owner accesses model objects.

## Fixed limits

| Limit | Value |
| --- | --- |
| Protocol frame | 65536 bytes, excluding LF |
| JSON nesting | 12 levels below the outer value |
| JSON object | 64 fields |
| JSON array | 256 entries |
| Command history | 1024 commands per epoch, including reserved cancellation and shutdown commands |
| ASR upload | 2097152 bytes |
| ASR audio | PCM16 little-endian, mono, 16000 Hz, 1 through 480000 frames |
| TTS text | 1 through 4096 Unicode scalar values and at most 16384 UTF-8 bytes |
| TTS output | PCM16 little-endian, mono, 8000 through 48000 Hz |
| TTS duration | 1 frame through 60 seconds |
| Complete output | At most 8388608 bytes, including WAV headers |
| WAV metadata | 65536 bytes, including unknown chunk headers and padding |
| WAV chunks | 32 |
| Waiting requests | 4 |
| Concurrent uploads | 2 |
| Aggregate media | 33554432 reserved bytes |
| HTTP connections | 16 |
| HTTP headers | 16384 bytes |
| Multipart overhead | 65536 bytes beyond the upload limit |
| Ingress deadline | 15000 milliseconds from accepted connection |
| Original request deadline | 120000 milliseconds from accepted connection |
| Queue deadline | 5000 milliseconds after checked submission |
| Cancellation grace | 1000 milliseconds before forced termination |
| Terminal settlement grace | 1000 milliseconds before a protocol fault |
| Send deadline | 15000 milliseconds for the complete response |

The complete output limit includes the RIFF container.
The service reserves ingress capacity before body allocation.
TTS reserves the complete output ceiling before admission.
The service retains uncertain leases until settlement or observed process reaping.
The response retains its output lease until transmission finishes or the connection closes.

The WAV parser checks the exact RIFF length and every chunk extent.
The parser requires one 16-byte `fmt ` chunk and one nonempty `data` chunk.
The parser checks byte rate, block alignment, frame alignment, and zero padding.
The parser rejects duplicate `fmt ` or `data` chunks, compressed formats, truncated chunks, and trailing bytes outside RIFF.
Unknown chunks consume the metadata limit.
Unknown chunk identifiers may repeat within both limits.

These limits bound ingress and retained output.
They do not prove a universal provider device-allocation bound.
Model quality, peak device allocation, and provider cancellation latency require separate hardware qualification.

## Public API matrix

| Route | Accepted fields | Result |
| --- | --- | --- |
| `POST /v1/audio/transcriptions` | Multipart `file`, `model`, optional `language`, optional `response_format=json` | JSON object with `text` |
| `POST /v1/audio/speech` | JSON `model`, `input`, `voice`, optional `response_format=wav` | Complete `audio/wav` body |
| `GET /healthz` | None | Service loop state |
| `GET /readyz` | None | Task, identities, pressure policy, limits, and readiness |
| `GET /v1/models` | None | Configured model and audio task extension |
| `GET /metrics` | None | Active, queued, reserved-media, completion, and fault observations |

The transcription language defaults to the deployment's first declared language.
The speech language uses the deployment's first declared language.
The speech format defaults to WAV.
The service does not infer a model, voice, precision, or provider.
The request model must equal the configured alias.
The request voice must occur in the immutable capability list.

Unknown fields return `invalid_request`.
Known unsupported fields return `unsupported_feature` before worker submission.
Unsupported fields include streaming, timestamps, prompts, temperature, speed, instructions, and reference audio.
Duplicate JSON keys, duplicate form fields, duplicate files, and duplicate content types are invalid.
Hyper normalizes repeated content lengths with equal numeric values.
The service accepts that normalized length and rejects conflicting lengths.
The service rejects transfer encoding and requires an exact content length for POST.
The profile does not implement realtime, SSE, remote URLs, or client file paths.

Errors use `{"error":{"code":"...","message":"..."}}` before response headers.
Public HTTP errors use these exact codes and status values.

| Condition | HTTP status | Public code |
| --- | --- | --- |
| Invalid request structure | 400 | `invalid_request` |
| Unsupported parameter or feature | 400 | `unsupported_feature` |
| Wrong audio task | 400 | `task_mismatch` |
| Unsupported WAV content | 400 | `invalid_audio` |
| Cancelled public request | 400 | `cancelled` |
| Wrong model alias | 404 | `model_not_found` |
| Input limit | 413 | `payload_too_large` |
| Full waiting queue | 429 | `queue_full` |
| Provider failure | 500 | `provider_error` |
| Output limit | 500 | `output_limit` |
| Missing frozen resource | 500 | `missing_resource` |
| Invalid worker protocol | 500 | `protocol_fault` |
| Worker not ready | 503 | `model_unavailable` |
| Host pressure or media quota | 503 | `capacity_unavailable` |
| Worker channel loss | 503 | `worker_lost` |
| Queue or original deadline | 504 | `deadline_exceeded` |

Worker `failed` results use only `provider_error`, `output_limit`, and `missing_resource`.
HTTP framing rejection can occur before the application creates a JSON error body.

Whisper readiness requires its public tokenizer for every declared language.
Each declared language must occur in the loaded tokenizer's supported language codes.
An English-only model must declare only `en`.
Kokoro readiness requires the public English frontend and phonemizer fallback constructors to succeed.
File readers must reject nonregular descriptors without waiting for another process.
Successful worker reaping must retain the recorded failure cause.

## Protocol envelope and primitives

The protocol name is `apxinf-audio-worker/1.0`.
Each UTF-8 JSON object ends with one LF.
The exact outer fields are `protocol`, `worker_epoch`, and `message`.
The `message.kind` field selects the exact schema below.
No schema permits unspecified fields or `null`.
Stdout contains protocol frames only.
Stderr contains bounded diagnostics only.

All integer fields use decimal integer literals from 0 through 9007199254740991.
Attempt numbers start at 1.
Boolean values, decimal points, exponents, and negative zero are invalid in integer fields.
All JSON parsing rejects duplicate keys, unsafe integer literals, nonfinite numbers, and unpaired surrogates.
UUID fields use lowercase canonical hyphenated strings.
Revision and content digests use 64 lowercase SHA-256 hexadecimal characters.
Labels contain 1 through 128 UTF-8 bytes and no control characters.
Strings retain their exact Unicode scalar values.
TTS text must contain a scalar outside the fixed whitespace set below.
The set contains U+0009 through U+000D, U+0020, U+0085, U+00A0, U+1680, and U+2000 through U+200A.
The set also contains U+2028, U+2029, U+202F, U+205F, and U+3000.

`key` has exactly `request_id` and `attempt`.
`MediaGrant` has exactly `handle`, `worker_epoch`, `key`, `bytes`, `sha256`, `frames`, and `sample_rate`.
The handle is a random UUID basename within a private service directory.
The grant binds the epoch, attempt, exact content size, digest, and checked WAV shape.
The service never accepts a handle from a public client.

| Message kind | Exact fields beyond `kind` |
| --- | --- |
| `ready` | `task`, `model`, `model_revision`, `capability_revision`, `manifest_revision`, `provider`, `languages`, `voices`, `cancellation`, `limits` |
| `submit` | `command_id`, `key`, `lease_id`, `model_revision`, `timeout_ms`, `operation` |
| `cancel` | `command_id`, `key` |
| `shutdown` | `command_id` |
| `accepted` | `key`, `event_seq` |
| `terminal` | `key`, `event_seq`, `result` |
| `resources_released` | `key`, `event_seq`, `lease_id` |
| `control_ack` | `command_id`, `status` |
| `stopped` | None |

`ready.task` is `asr` or `tts`.
`provider` is `mlx_audio` or the explicitly identified `cpu_fixture` test provider.
`languages` and `voices` contain at most 32 labels each.
Languages must be nonempty, and TTS voices must be nonempty.
`cancellation` equals `provider_boundary_or_process_reap`.
`limits` contains the nine fields of `AudioLimits` with the fixed values above.
Those fields are `upload_bytes`, `asr_frames`, `text_scalars`, `text_bytes`, `output_bytes`, `output_seconds`, `queue`, `uploads`, and `media_bytes`.

The supervisor independently hashes the configured manifest bytes before startup.
It checks every reported identity and capability against the configured manifest.
The worker checks artifact content before loading.
Missing local artifacts fail startup.
The service never substitutes another revision or downloads missing artifacts.

An ASR operation has exactly `task=asr`, `input`, and `language`.
The input is a `MediaGrant` for checked ASR WAV bytes.
A TTS operation has exactly `task=tts`, `text`, `voice`, `language`, and `output_handle`.
The output handle grants one exclusive file creation within the private namespace.
`timeout_ms` ranges from 1 through 120000 and retains the original request deadline.

The result `status` selects a terminal variant, rather than the separate public outcome.
`transcript` has exactly `status` and `text`, with at most 32768 UTF-8 text bytes.
`speech` has exactly `status` and `output`, where output is a complete WAV grant.
Both variants represent completed execution.
`cancelled` has only `status`.
`failed` has exactly `status`, `code`, and a message of 1 through 512 UTF-8 bytes.
The failure code must occur in the stable error list.

## Attempt state and settlement

| Current state | Required event | Next state |
| --- | --- | --- |
| submitted | `accepted`, sequence 0 | accepted |
| accepted | `terminal`, sequence 1 | terminal |
| terminal | `resources_released`, sequence 2, matching lease | settled |

Every event must retain the attempt key and epoch.
Sequence gaps, duplicate events, double terminals, and double settlement invalidate the worker channel.
The supervisor must not reuse an execution position before settlement.
EOF, partial JSONL, unexpected events, and mismatched identities invalidate readiness and queued requests.
The service does not replay failed requests or automatically substitute a worker.

The supervisor creates each command ID once and retains bounded command history.
Duplicate submissions are protocol faults and cannot execute twice.
Cancellation receipt returns `control_ack` with `cancel_requested` or `already_terminal`.
The acknowledgement never proves settlement.
Shutdown receipt returns `shutdown_requested` and closes admission.
The worker emits `stopped` after cleanup.
The supervisor also observes process exit before releasing epoch ownership.

Public cancellation, disconnect, and expiry stop public output immediately.
The active attempt retains its execution position and media until settlement or kill followed by observed reaping.
The control reader remains active during loading and execution.
The supervisor kills an unresponsive worker after the cancellation grace.
SIGINT and SIGTERM apply during loading and execution.
Shutdown closes admission, cancels work, and reaps the worker before service exit.

## Media transfer and validation

The service creates its private media directory with mode 0700.
Input publication uses exclusive creation, exact bytes, and a read-only file.
Readers reject symlinks and non-regular files.
Readers compare metadata before and after reading and check the content digest.
The worker checks input grant binding before model mutation.
The worker writes output with exclusive creation and a checked cumulative ceiling.
The supervisor reads and checks the complete output before public response headers.
The checked gateway buffer takes ownership of the output lease before worker settlement releases worker ownership.
PCM samples never enter JSONL.

## A0 verification gate

Rust and Python consume `tests/fixtures/audio-serving/cases.json`.
The original corpus contains raw JSON, WAV bytes, and attempt event sequences.
Both implementations must pass the same corpus before runtime implementation.
Runtime tests must additionally exercise the actual CLI, TCP API, and worker pipes.
CPU fixture tests do not establish speech quality or a hardware capacity profile.

## Dependency decisions

| Dependency | Version selection | Role and replacement boundary |
| --- | --- | --- |
| serde | Cargo.lock, major 1 | Handwritten Rust wire types and serialization |
| serde_json | Cargo.lock, major 1 | JSON syntax parsing beneath the strict handwritten visitor |
| sha2 | Cargo.lock, minor 0.10 | SHA-256 content and manifest identities |
| uuid | Cargo.lock, major 1 | Random handle, command, request, and epoch identities |
| libc | Cargo.lock, minor 0.2 | Unix no-follow files, process signals, and exclusive device lock |
| tempfile | Cargo.lock, major 3, tests only | Isolated test directories |

These libraries do not own audio semantics or model scheduling.
The provider dependency decision belongs to the separate operator guide.

This document follows ASD-STE100 Issue 9 structural principles.
Structural checks do not establish a complete dictionary review or certification.

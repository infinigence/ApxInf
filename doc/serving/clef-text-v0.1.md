# Clef text decision profile 0.1

Status: candidate interface. Date: 2026-10-23.
This document defines provider `python-clef`, public profile `apxinf-clef-text-v0.1`, and worker protocol `apxinf-worker/2.6`.
The [main contract](contracts-v0.1.md) defines common identities, transport limits, resource ownership, and errors.
Protocols 2.0 through 2.5 retain their existing behavior.
This interface specification does not establish successful runtime qualification.

## Terms and ownership

A decision evaluates typed questions against one text state.
A record contains the canonical state and questions.
An encoding contains the exact input IDs and question metadata from the selected dependency.
A preparation binds one record and encoding to one worker epoch and attempt.
A probability vector contains one probability for each canonical option.

The existing `Service` coordinator owns text and decision requests through a typed workload variant.
Its decision operation is `execute_decision`.
The coordinator retains admission, queues, deadlines, cancellation, pressure checks, diagnostics, and process supervision.
The decision worker owns its model, preparation, and device operations on one execution owner.
Each worker serves at most one attempt and then exits.
The gateway owns the SystemOne response projection.

**CF-01.** Decision requests must not populate unused text-generation fields or produce token events.
**CF-02.** The service must not publish success or release attempt resources before successful process reaping.
**CF-03.** A replacement worker must retain the approved model, runtime, execution, and capability identities.
**CF-04.** The coordinator must not replay a decision request automatically.

## Public request

`POST /v1/systemone` accepts exactly `model`, `state`, and `questions`.
`model` selects the configured alias and contains 1 through 128 UTF-8 bytes.
`state` is a string with at most 65536 UTF-8 bytes.
`questions` is an object with 1 through 16 entries.
Each question ID uses 1 through 64 ASCII characters from `A-Z`, `a-z`, `0-9`, `_`, `.`, and `-`.
The complete request contains at most 131072 bytes and has JSON depth at most 16.

Each question contains required `type` and `instructions` fields.
`instructions` is a string with 1 through 4096 UTF-8 bytes.
`type` is `noul`, `choice`, or `score`.
The `criteria` rules depend on that type.
Each criterion description is a string with 1 through 2048 UTF-8 bytes.

| Type | Criteria |
| --- | --- |
| `noul` | Optional object containing exactly `true` and `false` descriptions. |
| `choice` | Required object with 2 through 16 entries. Option IDs use the question-ID grammar. |
| `score` | Required array with 2 through 16 descriptions. Array order defines consecutive levels from zero. |

The canonical record preserves absent `noul.criteria`.
The pinned dependency supplies its option descriptions for that case.
Question IDs and choice IDs sort lexicographically by ASCII bytes.
The canonical `noul` option order is `true`, then `false`.
The canonical score option IDs are decimal indices in array order.

The gateway rejects unknown keys, duplicate keys, nulls, non-finite numbers, and unsafe integer literals.
The gateway also rejects a Boolean where a schema requires an integer.
The request schema permits no numeric content.
Media, streaming, generation controls, truncation, and arbitrary structured state remain unsupported.
An admitted request must match the selected model alias.
Text providers reject `/v1/systemone` with `unsupported_feature`.
This provider rejects chat and count endpoints with `unsupported_feature`.
Existing text workers retain their rejection of model bundles with decision-head markers.

Invalid shape or unsupported request fields produce `invalid_request` before preparation.
An unknown model produces `model_not_found` with HTTP status 404.
Context overflow produces `context_limit` with HTTP status 400.
Queue, deadline, cancellation, pressure, and worker errors retain their existing service meanings.

## Public response

The success object contains exactly `model`, `answers`, and `usage`.
`model` equals the admitted alias.
`answers` maps every canonical question ID to its projected answer.
`usage` contains exactly `input_tokens` and `output_tokens`.
`input_tokens` equals the prepared input-ID count, and `output_tokens` equals zero.

The worker returns raw probabilities without decimal rounding.
Each probability must be finite and lie between zero and one, inclusive.
Each vector length must equal its option count.
The absolute difference between each vector sum and one must not exceed `0.00001`.
Validation must not renormalize probabilities or replace invalid values.
The gateway retains float64 precision during projection and JSON serialization.
The gateway must not round results to four decimal places.

| Type | Exact answer fields and values |
| --- | --- |
| `noul` | `type: noul`, and `noul` equal to the `true` probability. |
| `choice` | `type: choice`, selected `choice`, `confidence`, and `probabilities`. |
| `score` | `type: score`, expected `score`, `confidence`, `legend`, and `probabilities`. |

For a choice, the largest probability selects the option.
An exact tie selects the first option in canonical order.
`confidence` equals the selected probability.
`probabilities` maps canonical option IDs to raw values.
For a score, `score` equals the sum of each numeric level multiplied by its probability.
Score `confidence` equals the largest probability.
`legend` maps decimal level IDs to the original descriptions.
JSON object member order does not change response meaning.

## Runtime and dependency boundary

The adapter revision is `apxinf-python-clef-text-v1`.
The checkpoint is `mlx-community/clef-flash-4bit` at revision `b5adbe2c3a95200d4bf04a0727c8ec457157d5e1`.
The backbone uses the bundle's four-bit quantization, and the joint decision head uses BF16.
The external loader remains a separately selected dependency file.
Its SHA-256 is `852223c944819a32fad5cf798d9d1dff30419820eaf5ad1f10cb9698afec97d5`.
The service does not copy its implementation into ApxInf.

| Runtime field | Required version |
| --- | --- |
| `python` | `3.12.10` |
| `mlx` | `0.32.3` |
| `mlx_lm` | `0.32.0` |
| `transformers` | `5.19.0` |
| `numpy` | `2.5.4` |
| `pillow` | `12.3.0` |

Python owns the isolated worker process and dependency imports.
MLX owns device arrays, softmax, synchronization, and allocator controls.
MLX-LM loads the text backbone through the external loader's public API.
Transformers supplies the tokenizer.
NumPy and Pillow identify the selected environment but do not enable media input.
This text path does not select MLX-VLM.

The adapter calls public `clef.load(path, backend="lm")`.
It calls public `clef.encode_record` for preparation.
It calls public `model.logits` for the decision.
Both encoding paths use `max_length=max_context`, `max_state_tokens=None`, `processor=None`, and `truncate=False`.
The adapter passes `processor=None` directly to `encode_record`.
For `model.logits`, the selected LM model supplies its own processor, which must be `None`.
The adapter must compare the actual returned encoding with the preparation before accepting logits.
It converts each logit vector to float32 before public `mlx.core.softmax`.
It evaluates the vectors and converts their values to host floats.
The adapter does not call the dependency's rounded SystemOne projection.

The execution context limit ranges from 1 through 4096 input IDs.
The CLI requires `max_output_tokens=0`, an explicit Python interpreter, and an explicit external loader path.
The CLI also requires the existing positive memory budget.
The worker uses public allocator controls with idle cache limit zero and the configured memory budget.
The worker must not increase the system wired-memory limit.
Context overflow produces `context_limit` without truncation or inference.
No session, prefix reuse, batching, media, or native Clef model implementation belongs to this profile.

## Ready identity

The `ready` frame contains exactly the common base fields and the fields below.
The common base fields are `protocol`, `kind`, and `worker_epoch`.
UUIDs, SHA-256 strings, safe integers, and token IDs retain the main contract's grammar.
All 2.6 frames use the existing 1048576-byte JSONL frame limit, including LF.
All objects reject unknown and duplicate members.

| Field | Shape |
| --- | --- |
| `model_revision`, `capability_revision` | Canonical identity digests. |
| `model_manifest` | The exact manifest below. |
| `capabilities` | The exact capability object below. |
| `limits` | `max_frame_bytes: 1048576`, `max_depth: 16`, `max_commands: 10000`. |
| `model_path` | Absolute configured model-directory path. |
| `runtime` | The six version strings above. |
| `memory` | Safe integer `active_bytes`, `peak_bytes`, and `cache_bytes`. |

The model manifest contains exactly these fields.

| Field | Meaning |
| --- | --- |
| `checkpoint`, `checkpoint_revision`, `adapter_revision` | The fixed values above. |
| `platform` | `macos-metal`. |
| `artifacts` | Sorted artifact file records. |
| `loader` | External loader file record. |
| `runtime` | The ready runtime object. |
| `sources` | Map of first-party source filenames to file records. |
| `python` | Interpreter file record with the selected launch path. |
| `runtime_files` | Runtime module and loaded-library file records. |
| `execution` | The exact execution object below. |

Each artifact record contains exactly `path`, `size_bytes`, and `sha256`.
Artifact paths are unique root filenames with extension `.json`, `.safetensors`, or `.jinja`.
The artifact set includes every regular root file with one of those extensions.
It excludes subdirectories, symlinks, `__pycache__`, Python code, README files, and license files.
It includes the configuration, tokenizer, template, joint-head files, weight index, and every indexed weight shard.
The adapter rejects missing required artifacts or index references outside that set.
Both processes check file lengths and digests before readiness.
The approved artifact records are in `python/apxinf/apxinf/serving/clef_artifacts.json`.
Both implementations use that same maintained record, without replacing its expected digests from the selected model directory.
The loader record contains exactly absolute `path` and the pinned `sha256`.

`sources` has exactly `clef_worker.py`, `clef_runtime.py`, `clef_identity.py`, `clef_contracts.py`, `contracts.py`, and `clef_artifacts.json`.
Each source record contains exactly absolute `path` and `sha256`.
Each filename must match its recorded file and the selected adapter source tree.
The Python record contains exactly absolute `path` and `sha256`.
The launch path preserves the virtual environment, including a final executable symlink.
The digest identifies the resolved executable content.
Runtime file records contain exactly absolute `path` and `sha256`.
`runtime_files` contains exactly `modules` and `libraries`.
Module keys are `mlx_core`, `mlx_lm`, `transformers`, `numpy`, `pillow`, and `tokenizers`.
`libraries` contains one resolved MLX library.
Recorded runtime files must belong to the selected environment.
The supervisor checks the manifest against local files before public readiness.
Version strings alone do not establish executable identity.

`execution` contains exactly these fields.

| Field | Required value |
| --- | --- |
| `provider` | `python-clef`. |
| `precision` | `backbone-4bit-head-bf16`. |
| `backend` | `lm`. |
| `max_context` | Configured integer from 1 through 4096. |
| `max_output_tokens`, `idle_cache_limit_bytes` | Zero. |
| `memory_limit_bytes` | Configured positive safe integer. |
| `truncate`, `kv_reuse` | `false`. |
| `worker_lifetime` | `one-attempt`. |

`capabilities` contains exactly these fields.

| Field | Required value |
| --- | --- |
| `workload` | `decision`. |
| `text`, `noul`, `choice`, `score` | `true`. |
| `image`, `streaming`, `token_events`, `kv_reuse` | `false`. |
| `state_schema` | `none`. |
| `max_active_requests` | One. |
| `vocab_size` | Integer from 1 through 2147483648, equal to the pinned configuration's text vocabulary size. |
| `max_context` | The execution context limit. |
| `max_output_tokens` | Zero. |
| `cancel_granularity` | `process_exit`. |

The model digest uses domain `apxinf-model-identity-v1` over the complete manifest.
The capability digest uses domain `apxinf-capability-v1` over the complete capability object.
Both use the existing `canonical_identity_digest` algorithm without changes.

## Preparation schema

The canonical record contains exactly `state` and `questions`.
Its questions retain the normalized public schema and preserve absent `noul` criteria.
Its digest uses domain `apxinf-clef-record-v1`.
The encoding contains exactly `input_ids` and `questions`.
`input_ids` contains 1 through `max_context` token IDs within the loaded vocabulary.
The metadata array contains exactly one entry for each canonical question.

Each metadata entry contains exactly these fields.

| Field | Meaning |
| --- | --- |
| `question_id` | Canonical question ID. |
| `question_type` | Zero for `noul`, one for `choice`, or two for `score`. |
| `question_span` | Two integer positions, `[start,end]`. |
| `option_spans` | One `[start,end]` pair per canonical option. |
| `option_ids` | The exact canonical option IDs. |

Every span is nonempty, excludes its end, and lies within `input_ids`.
Question spans and their option spans follow input order without overlap.
The next question span follows the previous question's final option span.
The metadata must match the canonical record's IDs, type codes, and option counts.
The encoding digest uses domain `apxinf-clef-encoding-v1` over the encoding object.

The preparation object contains exactly these fields.

| Field | Meaning |
| --- | --- |
| `worker_epoch`, `request_id`, `attempt` | The bound attempt identity. |
| `model_revision`, `capability_revision` | The approved ready identities. |
| `preparation_id` | The original preparation command UUID. |
| `record`, `encoding` | The exact canonical objects above. |
| `effective_input_tokens` | Exactly `len(encoding.input_ids)`. |
| `record_digest`, `encoding_digest` | The canonical digests above. |
| `preparation_digest` | The digest described below. |

The preparation digest uses domain `apxinf-clef-preparation-v1`.
Its identity document contains the first six identity fields, plus `record_digest` and `encoding_digest`.
`effective_input_tokens` adds no hidden or reserved output positions.
The supervisor checks the preparation against the original request before admission or disposal of a late response.
The worker re-encodes at submission and compares the complete encoding before device execution.
The encoding returned with logits must also match exactly before result publication.

## Worker frames

A command adds `command_id` to the common base.
An attempt command also adds `request_id` and positive `attempt`.
An attempt event adds `request_id`, `attempt`, and nonnegative `event_seq` to the common base.
The fields below supplement those envelopes.

| Kind | Envelope | Exact additional fields |
| --- | --- | --- |
| `prepare_decision` | Attempt command. | `model_revision`, `capability_revision`, `record`. |
| `prepared_decision` | Attempt command response. | `model_revision`, `capability_revision`, `preparation`. |
| `submit_decision` | Attempt command. | `model_revision`, `capability_revision`, `preparation`, `capacity_lease_ids`. |
| `accepted` | Attempt event. | `command_id`, `model_revision`, `capability_revision`. |
| `decision_result` | Attempt event. | `command_id`, `preparation_digest`, `probabilities`, `usage`, `metrics`. |
| `failure` | Attempt event. | `command_id`, `code`, `message`. |
| `shutdown` | Command. | None. |
| `drained` | Command response. | `reason`. |

`prepared_decision.command_id` equals `preparation.preparation_id`.
All repeated attempt and revision fields must agree with the preparation.
Submission uses a new command ID and retains the checked preparation unchanged.
The capacity lease array contains unique UUID strings and retains existing serial resource meanings.
`accepted.command_id` equals the submission command ID, and `event_seq` equals zero.
`decision_result.command_id` equals that same ID, and `event_seq` equals one.
Its `preparation_digest` equals the submitted preparation digest.

`probabilities` is an array of objects with exactly `question_id`, `option_ids`, and `values`.
Their order and IDs must match the submitted encoding.
`values` contains the checked raw probability vector.
`usage` contains exactly the public input and output counts above.
`metrics` contains safe integer `elapsed_ms` and `peak_device_bytes`.
These metrics do not establish process RSS or a general memory envelope.

`failure.code` is `invalid_request`, `context_limit`, `model_unavailable`, `worker_lost`, `protocol_fault`, or `internal_error`.
Its nonempty message contains at most 4096 UTF-8 bytes.
A failure before acceptance has sequence zero and identifies the preparation or submission command.
A failure after acceptance has sequence one and identifies the submission command.
Failure ends the attempt and requires process termination and reaping.

After any terminal event, the worker emits `drained` and exits.
For automatic attempt completion, `drained.command_id` identifies the terminating attempt command and `reason` is `attempt_complete`.
For idle shutdown, it identifies the shutdown command and `reason` is `shutdown`.
`drained` does not acknowledge released resources.
The protocol has no resource-release, token, progress, reusable-state, or statistics event.

## Attempt state and settlement

The initial tracker state is `new`.
A valid preparation moves it to `prepared`.
A matching submission moves it to `submitted`.
Acceptance moves it to `accepted`.
A matching result moves it to `terminal_pending_reap`.
Failure moves it to `failed_pending_reap` from any live attempt state.
Invalid frames do not change tracker state.

The supervisor privately retains a valid result until `drained`, clean EOF, exit code zero, and successful process reaping.
Only that combination moves the attempt to `succeeded` and permits the public response.
Premature EOF, duplicate events, extra output, stale identities, invalid probabilities, and nonzero exit discard a pending result.
The supervisor treats those cases as worker failure.
No second attempt enters that worker.

Cancellation, disconnect, deadline, and shutdown suppress any unpublished result.
They retain resource ownership until the existing bounded termination and reaping path completes.
No cooperative cancellation guarantee applies inside dependency calls.
A killed worker cannot return success, even when it emitted a valid result before cancellation.
The existing supervisor reaps the previous worker before it loads a replacement.
Replacement loading retains pressure checks and identity checks.
Readiness remains false while a replacement is absent or loading.

## Shared conformance corpus

`tests/fixtures/serving/clef-v0.1.json` is an original handwritten corpus.
`requests` checks strict public parsing and canonical records.
`encodings` checks metadata, ordering, and span bounds.
`probabilities` checks cardinality, values, normalization, and public projections.
`digests` checks the existing canonical identity algorithm with Clef domains.
`streams` checks event order, identity fences, and process settlement.
Each case states its expected validity or public resource outcome.
Patch cases apply JSON Pointer replacements to the named original template.
`remove` deletes named pointers before `set` replacements.
`raw` supplies exact JSON bytes for duplicate-key and number-parser checks.
An object containing only `$ref` selects a named corpus object before patch application.
References name the top-level `objects` or `frames` entry with a dot separator.
Fixture consumers expand references recursively and reject reference cycles.

Recommended Rust boundaries are `parse_decision_request`, `validate_decision_frame`, `DecisionTracker::observe`, and `DecisionTracker::reaped`.
Recommended Python boundaries are `parse_decision_request`, `validate_frame`, `DecisionTracker.observe`, and `DecisionTracker.reaped`.
Parsing returns a typed request or the existing structured error.
Frame validation accepts no model objects and changes no state.
Tracker observation accepts one checked frame and changes state only on success.
Reaping takes the observed exit code, EOF status, and drained status.
Both implementations consume this same corpus before runtime integration.

Structural writing checks do not establish a complete dictionary review or certification.

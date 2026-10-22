# Serial MiniCPM-V protocol 0.1

Status: P2c interface. Runtime qualification remains pending. Date: 2026-10-23.
This profile defines `apxinf-worker/2.5` for a separately selected MiniCPM-V worker.
It does not replace any text protocol or establish production image readiness.
The [main contract](contracts-v0.1.md) defines common identity, lifecycle, transport, and settlement rules.
The [owned PNG contract](owned-media-v0.1.md) defines source validation and limits.

## Profile and identity

The provider is `mlx-vlm-minicpmv47`. The adapter revision is `apxinf-mlx-minicpmv47-vision-v1`.
The checkpoint is `openbmb/MiniCPM-V-4.7-1B`, revision `379f120c2ff3588e6e06559ec6b9ec15cc05114e`.
The weight profile is BF16. This profile disables model thinking, state reuse, sessions, and batching.
The checkpoint weight SHA-256 is `8c4921ac83182c85190ad10c3aeb118d51e385d2d98ec13b88b514bbf6cac002`.
Its file length is 3109054480 bytes.

`runtime` has exactly four string fields:

| Field | Value |
| --- | --- |
| `python` | `3.14.3` |
| `mlx` | `0.32.4` |
| `mlx_vlm` | `0.7.10` |
| `transformers` | `5.18.0` |

The model manifest binds artifacts, runtime, adapter sources, interpreter, execution settings, and processor identity.
Its `runtime` equals the ready runtime. Its `checkpoint_revision` equals the pinned revision above.
Its `adapter_revision` equals the adapter revision above.
Its `checkpoint` equals the checkpoint name above.
Each `artifacts` entry has exactly `path`, `size_bytes`, and `sha256`.
Paths are relative and contain no empty, dot, or parent components.
Artifact entries sort by unique path.
The `model.safetensors` entry must match the weight digest and file length above.
Its `processor_revision` equals the canonical digest of `processor`, under domain `apxinf-vision-processor-v1`.
The processor identity is an identity document under the main contract's canonical encoding.
The manifest retains additional source and runtime file digests for startup verification.
The supervisor must check those local files before it publishes readiness.
The interpreter launch path must preserve its virtual environment, including a final executable symlink.
The interpreter content digest identifies the resolved executable separately.

`execution` has these exact fields:

| Field | Value or bound |
| --- | --- |
| `provider`, `precision` | `mlx-vlm-minicpmv47`, `bf16` |
| `enable_thinking`, `kv_reuse` | `false`, `false` |
| `max_context` | Integer from 1 through 16384. |
| `max_output_tokens` | Integer from 0 through 8192. |
| `prefill_step_size`, `output_batch_tokens` | `256`, `1` |
| `memory_limit_bytes` | Positive safe integer. |
| `selection` | The fixed selection identity below. |

`selection` has exact fields `algorithm`, `top_k`, `min_p`, `repetition_penalty`, `repetition_context_size`, `presence_penalty`, `frequency_penalty`, and `seed_policy`.
Their values are `mlx-vlm-fixed-controls-v1`, 100, `0`, `1.03`, 8192, `0`, `0`, and `request-id-sha256-v1`.
The numeric values in backticks are strings, except `top_k` and `repetition_context_size`.
Random requests carry temperature, top-p, and seed through the existing sampling object.
Greedy requests carry only `mode: greedy`. Greedy selection retains the fixed repetition controls.
Public defaults are temperature 1 and top-p 0.95. Public seed derivation retains the existing request-ID rule.
The worker must apply the supplied selection mode explicitly.

## Capabilities

The capability object retains all serial text capability fields and adds `media_limits`.
`text`, `image`, `greedy`, `sampling`, and `token_events` are true.
`exact_append` is false. `state_schema` is `none`. `max_active_sequences` is one.
`cancel_granularity` is `yield_or_process_exit`.
Context and output bounds equal the execution bounds.
These are candidate deployment ceilings, rather than evidence of successful execution at those ceilings.

`media_limits` has exact integer fields:

| Field | Value |
| --- | --- |
| `max_images` | 4 |
| `max_image_bytes` | 131072 |
| `max_total_bytes` | 262144 |
| `max_dimension` | 1024 |
| `max_host_array_bytes` | 67108864 |

Image dimensions and file bounds apply across the entire submitted history.
The worker must also check actual processor layouts before device transfer.
An unsupported layout produces an explicit request error.
Neither the model card nor a source-file limit qualifies every processor layout.

## Normalized messages

The public gateway admits only Anthropic base64 `image/png` source blocks.
User content and tool-result content can contain these blocks.
The gateway preserves text and image order and each `tool_call_id`.
It rejects URL sources, other media types, and images in system or assistant content.
OpenAI image inputs remain unsupported. OpenAI text inputs retain their existing behavior.
For tool-result errors, the gateway inserts an ordinary text part before the original content.
Public readiness must state this image input restriction.

Message and tool field names retain their serial meanings and bounds.
System and assistant content remain strings.
User and tool content can be strings or ordered content arrays.
Each array contains from 1 through 256 parts.
A text part has exactly `type: text` and string `text`.
An image part has exactly `type: image` and `media`, an owned PNG descriptor.
Image occurrences follow message order and then content order.
The adapter must preserve repeated images and tool-result identifiers.
Tool results retain their tool role and `tool_call_id`.
The gateway must not replace an image with descriptive text or a file path.

## Preparation and replay fields

All frame envelopes and common fields retain their serial meanings.
`prepare_input` retains its existing fields. `template_options.enable_thinking` must be false.
`prepared_input` retains tokens, effective input count, prompt digest, and identities, and adds `replay`.
`submit` retains all common serial submit fields, adds `sampling`, and adds the same `replay` object.
The complete processor ID sequence includes image placeholders.
For this profile, `effective_input_tokens` equals that sequence's length.
The service checks this count plus the maximum output allowance before submission.
Every frame retains the existing 1048576-byte limit, including LF.
The service must reject an oversized representation without truncation.

`replay` has these exact fields:

| Field | Meaning |
| --- | --- |
| `worker_epoch` | The worker epoch that prepared this input. |
| `model_revision` | The model revision that prepared this input. |
| `capability_revision` | The ready capability revision for this preparation. |
| `preparation_id` | The original prepare command UUID. |
| `rendered_prompt` | The exact formatted prompt, at most 131072 UTF-8 bytes. |
| `processor_revision` | The selected processor identity digest. |
| `processing_digest` | The complete preparation digest below. |
| `media` | Ordered owned PNG descriptors, subject to the batch limits. |
| `image_bindings` | One binding per media occurrence. |
| `host_array_bytes` | Measured unique host array storage, at most 67108864 bytes. |

In a prepared response, `preparation_id` equals `command_id`.
The response belongs to the current worker epoch and model revision.
Replay epoch and model revision must equal the enclosing frame values.
Submit capability revision must equal the replay capability revision.
A later submit uses a new command ID and retains that preparation ID.
No replay plan transfers across worker epochs or automatic retries.

Each binding has exact fields `source_index`, `token_start`, `token_end`, `pixel_shape`, and `target_size`.
`source_index` equals its zero-based array index.
The token interval is nonempty, excludes its end, and lies within the submitted token sequence.
Intervals follow source order and do not overlap.
Each interval contains only the pinned image token ID 248056.
All occurrences of that token must belong to an image interval.
The maximum interval length is 1024 tokens.
This profile does not assume 64 image tokens for every geometry.
`pixel_shape` contains `[3, 14, k]`. `target_size` contains `[h, w]`.
Both target dimensions are between 1 and 1024. Their product is at most 4096.
`k` equals `14 * h * w`.
These fields describe one actual image patch, rather than a requested slice setting alone.
The worker checks the processor's actual one-patch result and NumPy representation.

## Preparation digest

One adapter applies the chat template once.
Replay passes its already rendered string to the processor without applying another template.
Both paths call the selected public processor with NumPy output, 16x downsampling, and one image slice.
The actual processor class must be `mlx_vlm.models.minicpmv4_7.processing_minicpmv4_7.MiniCPMV4_7Processor`.
The worker must not use a helper that drops these options or creates device arrays during preparation.

The processing digest uses the canonical identity domain `apxinf-vision-processing-v1`.
Its document has exactly eight fields.
Four fields bind command identity: `worker_epoch`, `model_revision`, `capability_revision`, and `preparation_id`.
Four fields bind processing: `processor_revision`, `rendered_prompt_sha256`, `source_sha256`, and `processor_output`.
`source_sha256` is the ordered source-digest array.
`processor_output` represents every processor output used by generation, including input IDs and image arrays.
Each value uses the typed representation below.

| Value | Representation |
| --- | --- |
| NumPy array | `kind: array`, `dtype`, `shape`, and `sha256`. |
| Dictionary | `kind: object` and `items`, a dictionary of represented values. |
| List or tuple | `kind: list` and ordered `items`. |
| String | `kind: string` and string `value`. |
| Boolean | `kind: boolean` and Boolean `value`. |
| Integer | `kind: integer` and decimal string `value`. |
| Finite float | `kind: float` and Python hexadecimal float string `value`. |
| None | `kind: null`. |

Array dtype uses NumPy's explicit byte-order string. Shape entries are nonnegative integers.
Array SHA-256 covers contiguous C-order bytes, with little-endian numeric storage.
Object keys are ASCII. Unsupported values and non-finite numbers are invalid preparation output.
Shared backing storage counts once in `host_array_bytes`.
These bytes describe retained arrays, rather than temporary decoder memory or process RSS.
Replay must reproduce token IDs, bindings, processor revision, processing digest, and host array byte count before device transfer.
A mismatch fails the attempt and does not execute the model.

## Lifecycle and output

Preparation releases decoded images and host arrays before it returns a result.
The worker retains no media handle between commands.
Count-only, context rejection, cancellation, and late preparation need no remote release command.
The supervisor retains source bytes and its execution position until operation settlement.
Replay occurs only after context, pressure, and resource admission.
Existing attempt and command identity rules prevent duplicate generation.

Output counts include every yielded ID, including EOS.
The dependency can select tokens before their yield. Yielded counts do not measure internal selections.
`state_result` retains `validity: none` and `history_count`, but `consumed_position` must be null.
The profile does not claim exact consumed positions or reusable model state.
`prefill_progress`, when present, must report zero before the first device operation.
The profile provides no progress or cooperative cancellation guarantee inside dependency calls.
Controls apply before device transfer and at yielded output boundaries.
The supervisor retains its existing public deadline, settlement grace, and process-reaping recovery.

Terminal metrics retain elapsed time, TTFT, and device peak memory, and add `source_media_bytes` and `peak_host_array_bytes`.
`source_media_bytes` is the logical sum of request source-file lengths.
`peak_host_array_bytes` is the maximum measured retained host array storage for the attempt.
These counts exclude queue copies, wire buffers, decoder temporaries, and allocator reserve.
Readiness memory retains the existing device allocator fields.
Resource release follows generator close, device synchronization, and removal of request references.
The worker must not return a capacity lease after uncertain device settlement.

The P2c reference, shape, context, repeated-request, memory, and cancellation gates remain required.
Real Claude tool execution and an inspected pelican riding a bicycle remain separate acceptance requirements.
Structural writing checks do not establish a full dictionary review or certification.

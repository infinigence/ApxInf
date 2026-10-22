# Serial MiniCPM-V protocol 0.1

Status: P2c candidate interface. Date: 2026-10-23.
Selected runtime and HTTP checks pass. Full workload qualification remains pending.
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

### Runtime source identity

These requirements apply to startup verification. Standalone wire checks do not establish startup verification.
The candidate runtime targets macOS Metal. Its manifest `platform` is `macos-metal`.
`adapter_sha256` identifies `vision_worker.py`. `python_sha256` identifies the resolved interpreter.
`sources` maps seven local filenames to their content digests.
They are `vision_worker.py`, `vision_runtime.py`, `vision_identity.py`, `vision_preparation.py`, `text_worker.py`, `contracts.py`, and `owned_media.py`.
The runtime checks all eight checkpoint artifact identities before loading model weights.
`vision_identity.py` contains their pinned lengths and digests.
The dependency record defines the exact package versions and their roles.

`runtime_files` contains exact fields `modules` and `libraries`.
Each file record has exactly `path` and `sha256`.
Module keys are `mlx_core`, `mlx_vlm`, `transformers`, `numpy`, `pillow`, `torch`, `torchvision`, `tokenizers`, `generation`, and `processor`.
They also include `module_state`, `minicpm_model`, `minicpm_language`, `minicpm_language_base`, `qwen_language`, and `rotary_state`.
The library array contains one resolved MLX library.
Its SHA-256 is `d8c9393d584e781393c3effccfccd8ba85ad0c70264edaf9f897dca9875375c6`.
All recorded module and library files belong to the selected virtual environment.
The runtime checks this identity before and after model loading.

The processor document has exact fields `class`, `runtime`, `artifacts`, `options`, and `template_options`.
`class` contains the processor class specified below. `runtime` contains the four pinned runtime versions.
`artifacts` retains every checkpoint artifact except model weights.
`options` contains `return_tensors: np`, `downsample_mode: 16x`, `max_slice_nums: 1`, `padding: true`, and `add_special_tokens: false`.
`template_options` contains `enable_thinking: false`, `tokenize: false`, and `add_generation_prompt: true`.
The loaded model contains 1554476272 BF16 parameters.

### Model request-state ownership

One runtime owner restores model-held transient state after synchronization.
The owner uses the public `Module.state` and attribute interfaces.
The field names below remain qualified dependency internals. They are not a general dependency reset contract.
Startup checks exact model classes and the six state-source digests in `vision_identity.py`.
It also checks initial baselines and captures resident state topology and array handles.

| Location | Field | Baseline | Permitted active value |
| --- | --- | --- | --- |
| Selected language module | `_position_ids` | Readable `None`. Absent public state key. | Int32 array `[3,1,N]`, with `1 <= N <= max_context`. |
| Selected language module | `_rope_deltas` | Readable `None`. Absent public state key. | Int32 array `[1,1]`. |
| Each of 18 qualified convolution modules | `_qwen3_5_decode_conv_weight` | Absent attribute and public state key. | Tuple of resident weight identity and FP32 array `[4,6144]`. |
| Each of 6 qualified rotary modules | `_compiled_apply` | Empty dictionary with its original identity. | Key `3` maps to a callable `mlx.core._gc_func`. |

Each convolution module has resident `conv1d.weight` shape `[6144,4,1]`.
The cached identity equals that resident array's Python identity.
The owner checks every installed field and the remaining state inventory before restoration.
Early cancellation and partial installation can leave individual fields at baseline.
Restoration recovers the exact readable or absent baseline for each field.
The rotary modules have class `mlx_vlm.models.qwen3_5.language.Qwen3_5RotaryEmbedding` and dimension 64.
Their compilation dictionaries retain their original identities and become empty during restoration.
This removal does not clear the dependency's global kernel or compilation caches.
The qualified factory captures kernel configuration. Its callable accepts request arrays as arguments.
Unknown state, replaced resident handles, unsupported values, or uncertain synchronization permanently fence the runtime.

The model manifest adds `request_state` with exact fields:

| Field | Value |
| --- | --- |
| `revision` | `minicpmv47-transient-state-v1` |
| `max_context` | The execution context bound. |
| `position_fields` | Ordered array `[_position_ids, _rope_deltas]`. |
| `convolution_cache_field` | `_qwen3_5_decode_conv_weight` |
| `convolution_module_count` | 18 |
| `rotary_cache_field` | `_compiled_apply` |
| `rotary_module_count` | 6 |

State restoration retains resident array handles. It does not copy or evaluate their contents.
Handle checks do not establish the absence of in-place content changes.
The pinned source path and execution qualification must establish that stronger invariant.
No release event follows uncertain cleanup. Process exit remains the recovery boundary.

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
Random temperature ranges from 0.01 through 2, inclusive.
The dependency increases smaller positive temperatures. This profile rejects them instead.
Greedy requests carry only `mode: greedy`. Greedy selection retains the fixed repetition controls.
Public defaults are temperature 1 and top-p 0.95. Public seed derivation retains the existing request-ID rule.
The worker must apply the supplied selection mode explicitly.

### Service selection

The service selects this worker only with explicit provider `mlx-vlm-minicpmv47`.
This selection requires an explicit Python interpreter path for the separate environment.
The default worker path is `python/apxinf/apxinf/serving/vision_worker.py` for this provider.
Other providers retain their existing worker defaults and interpreter behavior.
The public profile is `apxinf-minicpm-vision-v0.1`.
Readiness reports image support with restriction `anthropic_base64_png_user_and_tool_result` and the media limits below.
It reports the existing Qwen-compatible tool format.

The gateway rejects positive temperatures below 0.01 before conversion to its existing float32 selection type.
Submission serializes the accepted lower bound as at least float64 value 0.01.
This representation rule prevents float32 rounding from violating the worker's lower bound.
Other profiles retain their existing sampling representation.

The supervisor checks every source digest, recorded runtime file, processor document, and request-state schema before public readiness.
Execution limits equal the selected launch configuration.
Prepared media equal the original ordered occurrences, including repeats.
Processor and capability revisions equal the ready worker's selected identities.
These checks also apply to late preparation responses before their disposal.
Submit retains the exact checked replay object. Count-only requests stop before submit.

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
Source images use Pillow RGB conversion before this call.
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
Text decoding preserves generated tool delimiters and excludes the configured EOS IDs.
Final decoding must extend the already emitted text without changing its prefix.
The dependency can select tokens before their yield. Yielded counts do not measure internal selections.
`state_result` retains `validity: none` and `history_count`, but `consumed_position` must be null.
The profile does not claim exact consumed positions or reusable model state.
`prefill_progress`, when present, must report zero before the first device operation.
The profile provides no progress or cooperative cancellation guarantee inside dependency calls.
Controls apply before device transfer and at yielded output boundaries.
The supervisor retains its existing public deadline, settlement grace, and process-reaping recovery.

Terminal metrics retain elapsed time, TTFT, and device peak memory, and add `source_media_bytes` and `peak_host_array_bytes`.
`source_media_bytes` is the logical sum of request source-file lengths.
The attempt tracker must compare this count with submitted source occurrences before it accepts a terminal event.
Repeated images contribute their byte counts separately. A text-only request contributes zero bytes.
`peak_host_array_bytes` is the maximum measured retained host array storage for the attempt.
These counts exclude queue copies, wire buffers, decoder temporaries, and allocator reserve.
Readiness memory retains the existing device allocator fields.
Resource release follows generator close, device synchronization, and removal of request references.
The runtime synchronizes the public generation stream, including failures before the first yield.
The worker must not return a capacity lease after uncertain device settlement.

The P2c reference, shape, context, repeated-request, memory, and cancellation gates remain required.
Real Claude tool execution and an inspected pelican riding a bicycle remain separate acceptance requirements.
Structural writing checks do not establish a full dictionary review or certification.

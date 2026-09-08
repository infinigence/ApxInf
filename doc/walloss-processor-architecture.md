# WallOSS processor architecture

WallOSS supports two processor adapters over one model input seam. The native
RGB route provides the model input seam for a future Rust-only policy. Today,
both built-in policies still require Python orchestration. The Python custom
adapter keeps observation pipelines extensible without changing model execution.

## Canonical seam

The model runtime accepts one of two vision representations together with the
same token and action-mask inputs:

```text
Native adapter:  resized RGB u8 + token_ids + action_mask
Custom adapter:  patch rows f32 + token_ids + action_mask
                                      |
                                      v
                              WallOSS VlaRuntime
                                      |
                                      v
                        normalized-domain action f32
```

`VlaContract.accepts_rgb_u8` advertises the native route. Python selects it for
the built-in processor and keeps `_infer_patches` as the compatibility route for
a user-supplied `processor=` callable. The binding does not contain model
semantics; it only validates arrays and constructs the Rust observation.

The RGB contract deliberately starts after image decoding and smart resize.
Those operations are input and application concerns and are not required to be
inside the CUDA graph. Qwen2-VL channel normalization, still-image temporal
duplication, patchification, and spatial-merge ordering are model tensor
semantics and execute in the captured CUDA path.

## Current PI0.5 and WallOSS flow contracts

This section describes the implemented paths, not a new shared Prompt Builder
interface. Moving family prompt orchestration into Rust is a separate change.

| Stage | PI0.5 built-in policy | WallOSS built-in policy | Current owner |
| --- | --- | --- | --- |
| Observation mapping | configured image keys, instruction, optional state | configured image keys, camera labels, instruction, state and masks | Python policy; robot adapters may map external fields before entry |
| State normalization | checkpoint-selected transform when `discrete_state=True`; computation dtype is configurable | selected legacy proprioception normalizer; normalize and clip to `[-1, 1]` | Python family policy/processors |
| State encoding | optional 256-bin state; preserve signed underflow bin `-1` | required checkpoint-configured bins; clip indices and select active dimensions | Python family prompt logic |
| Prompt assembly | clean task; optional `Task/State/Action` text | system/user/assistant text, camera labels, image placeholders and action tokens | Python family prompt logic |
| Tokenizer execution | native SentencePiece `.model` backend | native HF `tokenizer.json` backend | `apxinf-tokenizer` via PyO3 |
| Sequence assembly | prepend BOS; without state, append separately encoded newline; validate length | add model tokens, check image token ID and expand image placeholders using grid and merge size | Python family logic calling native tokenizer primitives |
| Image preparation | decode/resize with padding; arrange configured views | Pillow smart resize; deployed profile requires two `18x18` grids (`252x252` RGB) | Python processors |
| Model input | resized RGB, token IDs and explicit/generated noise | resized RGB or canonical patches, token IDs, action mask and explicit/generated noise | Python policy to thin PyO3 adapter |
| Tensor preprocessing and inference | native RGB normalization/patchification and model execution | native RGB normalization, temporal duplication, patchify/merge, BF16 conversion and model execution | Rust/CUDA |
| Output interpretation | trim and apply checkpoint action transform | trim and apply selected action normalizer inverse | Python policy; model output is normalized-domain action |

PI0.5 state injection is opt-in at the policy constructor. When enabled,
`state_key` and a matching state normalization transform are required. Robot
presets may enable it. Normalization precision is part of the contract: rounding
near a bin edge can change token IDs. WallOSS uses its own clipping, bin count
and active-state semantics; it must not reuse PI0.5 binning by assumption.

PI0.5's non-state path encodes the cleaned task with BOS and then encodes a
newline separately. Encoding a single string ending in a newline is not an
established equivalent. The existing Rust `pi05_prompt()` helper does not yet
own the Python production sequence-building path.

WallOSS expands each image placeholder to `product(grid_thw) / merge_size^2`
tokens. Image ordering, camera labels and grid metadata must agree. Its action
mask is float32 `[action_horizon, action_dim]`; a one-dimensional DOF mask is
broadcast across the horizon, and the active-state mask supplies the default.
These are family semantics, not generic tokenizer behavior.

### Extension and ownership contracts

- PI0.5 exposes replaceable Python input/output pipelines. Its standard input
  pipeline produces RGB and token IDs for `infer_rgb`.
- WallOSS `processor=` produces `(patches_f32, token_ids_u32, action_mask_f32)`.
  A custom callable does not implicitly opt into native RGB. The built-in route
  uses `VlaContract.accepts_rgb_u8`; legacy checkpoints without processor metadata
  retain patch-only loading.
- Both model runtimes return normalized-domain actions. Policy results expose
  those as `normalized_actions` alongside postprocessed `actions`. These actions
  are in the selected checkpoint's output domain; robot command encoding, units,
  joint ordering and transport remain adapter concerns.
- Checkpoint-defined state/action transforms belong conceptually to the family
  policy even if statistics are embodiment-specific. External robot field names,
  image containers and application protocols belong to adapters.
- `apxinf-py` is required for built-in inference. Python `tokenizers` and
  `sentencepiece` are no longer production dependencies; `walloss` and `tokenizer`
  extras are removed. NumPy, Pillow and PyYAML remain base dependencies, and the
  Python SentencePiece package remains a test reference. The optional `lerobot`
  adapter still has its own Torch dependency.

The next migration should specify family builder inputs, output token sequences,
errors and reference fixtures before changing production callers. Generic
loading/encoding mechanisms can be shared; template text, state binning, special
token ordering and multimodal expansion retain family ownership. A standalone
policy also needs canonical normalizer loading and action postprocessing; native
tokenizers alone do not complete it.

## Ownership

### Reusable ApxInf modules

- `apxinf-tokenizer` owns loading and executing `tokenizer.json`, adding tokens,
  token-to-ID lookup, and ordinary encode/decode behavior. It delegates BPE and
  added-token semantics to Hugging Face's Rust `tokenizers` crate. Its optional
  SentencePiece backend reads native `.model` assets through the C++ library,
  statically linked in the Python extension. Both backends are exposed by thin
  PyO3 bindings; family prompt rules remain outside this crate.
- `apxinf-loader` owns SafeTensors parsing for model and future processor
  sidecars.
- `VlaRuntime`, `Observation`, and `VlaContract` own the model-neutral input and
  capability contract.
- `apxinf-cuda::kernels::preprocess` owns the safe device operation; its CUDA
  implementation is model-neutral because every Qwen2-VL parameter that affects
  the tensor result is explicit.

### WallOSS-family-specific logic

- The exact system/user/assistant prompt and camera labels.
- `<|propri|>` and `<|action|>` token additions and image-token expansion.
- Proprioception quantile binning and the selected normalization key.
- The fixed two-view, 18x18-grid checkpoint profile.
- Action-mask construction and action unnormalization.
- Validation that `preprocessor_config.json` agrees with the model vision
  configuration.

These rules must not be added as branches to generic VLA or tokenizer modules.
A Rust-only policy package should compose them around the same `VlaRuntime`
interface.

### Python-only extension surface

- Mapping arbitrary robot dictionaries or objects to named state/image fields.
- Decoding application-specific image containers.
- User-defined callable processors and experimental transforms.
- Websocket/server integration and Python ecosystem interoperability.

Python is therefore more than a generated binding, but the default model
semantics need not depend on Python. PyO3 remains a thin adapter for canonical
arrays and capability discovery.

## Dependency migration

| Concern | Current default | Rust-only target | Compatibility route |
| --- | --- | --- | --- |
| `.pth` normalizers | restricted Python tensor reader | build-time conversion to a canonical SafeTensors/JSON sidecar, then Rust loading | keep restricted reader for legacy checkpoints |
| Qwen2.5-VL tokenizer | `apxinf-tokenizer` HF backend through PyO3 | same Rust backend called directly | Python callable may emit token IDs itself |
| PI0.5 tokenizer | `apxinf-tokenizer` SentencePiece backend through PyO3 | same Rust backend called directly | custom Python pipeline may emit token IDs itself |
| image resize | Pillow bicubic smart resize | Rust image adapter or caller-provided resized RGB | Python Pillow remains supported |
| normalize/patchify/merge | CUDA for native adapter | same CUDA kernel | NumPy path for custom processors and differential tests |
| prompt construction | WallOSS Python policy | WallOSS Rust policy adapter | custom Python callable |
| inference input | PyO3 arrays | direct Rust `Observation` | `_infer_patches` remains available to policies |

Rust should not parse Python pickle as a long-term deployment format. The
restricted `.pth` reader is a compatibility and conversion tool. A converted
sidecar should carry explicit tensor names, widths, dtype, normalization mode,
and source checkpoint identity so both Rust and Python can reject mismatches.

## Phased delivery

1. Remove Torch and Transformers from the Python runtime while retaining exact
   checkpoint behavior.
2. Move Qwen2-VL normalization and patchification into a graph-capturable CUDA
   operator, expose RGB capability through `VlaContract`, and keep the patch
   adapter.
3. Add the WallOSS prompt/token expansion and canonical normalizer sidecar to a
   Rust policy adapter built on `apxinf-tokenizer` and `VlaRuntime`.
4. Package the Rust adapter as a standalone entry point. Make the Python default
   call that adapter, while preserving `processor=` as the alternate adapter.

Each phase must keep both canonical routes testable; removing the Python patch
route is not a completion criterion.

## Required differential tests

- tokenizer IDs against the pinned Qwen2.5-VL tokenizer, including added model
  tokens and the complete WallOSS prompt;
- resized-image dimensions and bytes against Pillow/Transformers fixtures;
- FP32 NumPy patches against Transformers before BF16 rounding;
- BF16 CUDA patches against the independently generated NumPy golden, for NHWC
  and NCHW, multiple views, multiple merge groups, and the deployed 252x252
  profile;
- legacy `.pth` and converted sidecar normalization values, key selection, and
  invalid/missing metadata failures;
- Python built-in native-RGB routing and custom callable patch routing;
- eager versus captured outputs, repeated replay, and changed-input propagation;
- complete observation-to-action output against the pinned checkpoint with
  identical noise and action mask.

## Validation scope and remaining evidence

The migration requires exact token-ID parity and image-preprocessing parity at
the model input seam, plus fixed-noise action comparisons through the changed
native path. Graph tests must include repeated replay and changed-input
propagation. Offline tests with fake model handles verify policy composition and
routing, not checkpoint accuracy. Checkpoint-gated tests that skip are not passes.

Numerical parity and latency are separate claims. A performance comparison must
use the same checkpoint, precision, tactics, views, token lengths, noise, flow
steps, graph mode, warmup and synchronization. Report CPU preprocessing,
steady-state model inference and full policy latency separately, with repeated
measurements on an otherwise idle device. Cold loading/build time is a separate
measurement. Test pass counts do not establish performance equivalence.

Closed-loop task success rates are a broader qualification exercise. They are
useful when policy semantics or numerical outputs change, but a small rollout
sample cannot replace deterministic input/action differential tests for this
migration. Standalone Rust policy and canonical-sidecar tests above become gates
when those future paths are implemented; they are not claims of current coverage.

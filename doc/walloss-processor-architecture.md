# WallOSS processor architecture

WallOSS supports two processor adapters over one model input seam. The native
adapter makes a Rust-only deployment possible. The Python adapter keeps custom
observation pipelines possible without changing model execution.

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

## Ownership

### Reusable ApxInf modules

- `apxinf-tokenizer` owns loading and executing `tokenizer.json`, adding tokens,
  token-to-ID lookup, and ordinary encode/decode behavior. It delegates BPE and
  added-token semantics to Hugging Face's Rust `tokenizers` crate.
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
| Qwen2.5-VL tokenizer | Python package backed by Rust `tokenizers` | `apxinf-tokenizer` using the same `tokenizer.json` | Python callable may emit token IDs itself |
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

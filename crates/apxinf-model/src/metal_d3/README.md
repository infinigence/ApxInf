# Native d3 on MLX

This Metal port of the d3 decision family owns package interpretation, the
forward composition and its compiled profiles. It calls safe `apxinf-mlx` array
operations, without Python or an external model provider. It is registered as
`metal_d3`; a `d3-code-readout-v1` package detects as `d3` and resolves here on
`Device::Metal`. Target: `vllm-sr/d3-lite` (Qwen3.5-0.8B backbone), BF16 as
stored, batch one, Apple Silicon macOS.

`config.rs` admits only the d3-lite geometry: hidden 1024, 24 layers (18 Gated
DeltaNet, 6 full attention, every fourth), the 255-code FP32 readout with unit
temperature, non-causal full attention, and the package manifest's parameter
counts. Other geometry, including d3-mini, is rejected with the offending field.
`weights.rs` checks every `language_model.*`, `visual.*` and readout tensor.

`auto` and `bf16` select the same composition. CUDA tuning, calibration,
synthetic weights and auxiliary assets are rejected.

## Requests

The runtime implements `CodeDecisionRuntime`: the caller renders and tokenizes
the prompt (the package's `d3_format.py`) and passes token IDs and the
candidate count. The runtime scores the first `candidate_count` package codes
with a masked FP32 softmax at the last position. There is no decode and no KV
cache.

An image request passes one still image through `DecisionInput::with_vision`
(`pixel_values [N, 1536]`, `grid_thw [[1, h, w]]`). `preprocess.rs` reproduces
the checkpoint processor from decoded RGB: `smart_resize` with Python
round-half-even, PyTorch's int16 bicubic resize, `(u8 - 127.5) / 127.5` and
merge-block patchify. `vision.rs` runs the 12-block tower and spatial merger
eagerly. The merged tokens replace the prompt's one contiguous placeholder run,
and the text layers run under mRoPE positions built like the reference's
`get_rope_index`. Multiple images and video are rejected by name.

## Compiled profiles

Each request runs one whole-request `Compiled` callable. A profile is one exact
input shape: the prompt length for text, and the prompt length plus merged
image-token count for images. The candidate count is an input value and reuses
the profile. A new profile is traced and warmed with
`Compiled::prepare`; later requests of that shape replay without a trace
callback.

Profiles are kept least recently used first and bounded by a profile count and
by the sum of the retained prompt lengths. Text profiles honor the shared
`LoadOptions::d3_text_profile_limit` and `d3_text_profile_token_budget`; both
default to 12 profiles and 30,000 tokens. Image profiles use the same default
bound. A request longer than the token budget still runs, alone. Evicting a
profile drops its callable, so its next request traces again. The first request at
a new length also pays for tracing and warming, so a benchmark must warm every
length before timing it.

Retained profiles hold no device memory between requests; the weights do. MLX
keeps the buffers each new shape frees for reuse, though, and that process-wide
cache is not bounded here. On an M2, four 8,000-token lengths grew it to 8.4 GB
while live memory stayed at 1.6 GB. A caller that runs many distinct lengths
must bound it, as `examples/d3_decision_index.rs` does with
`apxinf_mlx::clear_cache` once the cache passes 1 GiB.

## Device tests

The ignored tests run on an Apple Metal GPU with the real package:

| Test | Inputs |
| --- | --- |
| `model::tests::compiled_forward_matches_eager_replays_and_binds_changed_inputs` | `APXINF_D3_CHECKPOINT` |
| `model::tests::text_profiles_stay_within_the_limit_and_evicted_lengths_retrace` | `APXINF_D3_CHECKPOINT` |
| `model::tests::vision_tower_matches_pytorch_reference` | `APXINF_D3_CHECKPOINT`, `APXINF_D3_VISION_REF`, optional `APXINF_D3_VISION_STAGES` |
| `model::tests::image_decision_matches_pytorch_reference_through_the_public_path` | `APXINF_D3_CHECKPOINT`, `APXINF_D3_VISION_REF` |
| `preprocess::tests::preprocessing_reproduces_the_reference_pixel_values` | `APXINF_D3_TEST_IMAGE`, `APXINF_D3_REFERENCE_PIXELS` |

`APXINF_D3_VISION_REF` is a `cases.json` recorded from the checkpoint's own
PyTorch runtime. Each case holds `id`, `token_ids`, `option_count`, `grid_thw`,
the reference `probabilities`, and `arrays.pixel_values.file` and
`arrays.image_features.file`, raw little-endian F32 files relative to the JSON.
`APXINF_D3_REFERENCE_PIXELS` is the processor's raw F32 `pixel_values` for the
PNG in `APXINF_D3_TEST_IMAGE`.

```bash
APXINF_D3_CHECKPOINT=/path/to/d3-lite \
APXINF_D3_VISION_REF=/path/to/reference/cases.json \
cargo test -p apxinf-model --features mlx --lib metal_d3 -- --ignored --test-threads 1
```

The image tolerances come from the reference itself. The vision tower's late
residual stream amplifies BF16 kernel-order differences, and the checkpoint's
PyTorch runtime moves by itself between MPS and CPU. Each test allows three
times that movement: 0.06 absolute probability with no changed decision, and
0.21 relative L2 on the merged image features.

Accuracy on the public text suite is measured with the shared Decision Index
runner: `examples/d3_decision_index.rs --device metal`, fed by
`scripts/d3_prepare_decision_index.py` as in
[the d3-mini guide](../../../../doc/d3-mini-decision-index.md).

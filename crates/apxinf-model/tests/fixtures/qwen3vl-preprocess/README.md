# Qwen3-VL processor fixtures

Captured offline on 2026-11-26 from local `Qwen/Qwen3-VL-2B-Instruct`
assets, using Transformers 5.19.0, CPU torch 2.14.0, NumPy 2.5.3 and Pillow
12.3.0. `manifest.json` records SHA-256 for every source asset and uncompressed
F32 pixel buffer. The checkpoint configuration and tokenizer assets retain the
upstream [Apache-2.0 license](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct).

The original preprocessor budget is unchanged: 65536..16777216 pixels, 16-pixel
patches, temporal duplication 2 and spatial merge 2. Cases are non-square
512x320, upscaled 40x24 and downscaled 4128x4096. The deterministic input RGB
formula is recorded in `tests/vision_preprocessing.rs`; no source images or
weights are needed. Gzip losslessly stores the little-endian F32 arrays and
unmodified tokenizer JSON. The large case compresses well but exercises the
checkpoint's real downscale budget, not a substituted serving cap.

The structured-history case includes system instructions, tools, an assistant
call, its tool result and a subsequent image turn. Expected image and text-only
token IDs come from the checkpoint's own processor/template. Removing the whole
vision span from the image case reproduces text-only tokens exactly.

Run `cargo test -p apxinf-model --test vision_preprocessing --no-default-features`.
CI consumes the recorded assets entirely in Rust, without a GPU, network or
reference framework. The standalone pixel fixture is not evidence of model
quality or that a full-size image fits a model's allocated KV capacity.

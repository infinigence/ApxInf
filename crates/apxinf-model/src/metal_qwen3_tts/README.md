# Native Qwen3-TTS CustomVoice on MLX

This Metal-specific module owns checkpoint interpretation, the generation
schedule and per-request state for Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice. It
implements `SpeechRuntime` with safe `apxinf-mlx` operations, without Python.
The reference is the EngineTailor MLX parity path: mlx-audio revision
`70f4add32911bab6f869b824864ad9f1e24dcb97` with the cast seams of EngineTailor
`282adc1dc5af4c48ffd17ea2f41e75cc5a24b7ee`, eager, on MLX 0.32.3.

The family admits batch one, non-streaming synthesis, the checkpoint's voices
and non-dialect languages (or `auto`), an optional style instruction and a
seed. Sampling follows `generation_config.json`, which must hold the validated
values (temperature 0.9, top-k 50, top-p 1, repetition penalty 1.05 for the
first codebook); `max_new_tokens` is the frame limit. Other model types, sizes
and geometry fail at load.

- `config.rs` checks the fixed geometry of the Talker, code predictor and
  waveform decoder, and reads voices, languages and dialect overrides.
- `weights.rs` checks every key and shape. All floating weights execute in
  BF16, as the reference casts them; codebooks are computed from
  `embedding_sum` and `cluster_usage` in BF16. The waveform encoder and the
  decoder's unused input projections are not read.
- `talker.rs` runs the 28-layer Talker with interleaved MRoPE (text-only
  positions are equal on all axes) and the 5-layer code predictor, with
  mlx_lm-style KV caches. Prefill computes logits for every row, as the
  reference does, so each matmul keeps the reference's shape.
- `sample.rs` reproduces `_sample_token`: control-token suppression, the
  repetition penalty, temperature, compiled top-k and categorical sampling.
  The family owns the MLX key: the seed resets it, and each sample splits it.
- `codec.rs` decodes 300-frame chunks with 25 frames of left context. As in
  mlx-audio, its transformer applies a full causal mask, not the
  checkpoint's 72-frame sliding window.
- `ops.rs` holds the parity arithmetic: FP32 SiLU before the BF16 cast, and
  the reference's compiled GELU and RoPE.

Variants: the default `auto` runs the five code predictor MLPs on
`apxinf_mlx::fusions::SwigluMlpGemv` only on the measured MLX architecture
`applegpu_g16g` (M4). Other architectures retain the reference composition
and report that choice on stderr; a new architecture needs its own numerical
and timing qualification before default enablement. Its reduction order differs
from MLX matmul. On Apple M4 it
measured +1.66% over the parity path (eight utterances, p = 0.013) with
identical codes and waveforms. `reference` keeps the parity path's composed
SwiGLU everywhere.

Verification status: CPU tests cover configuration admission, voices,
languages and dialects. Codes and waveforms against the reference, repeated
requests, memory and latency are measured on Apple M4 and reported in the PR;
other devices are listed as supported only after the same checks pass there.

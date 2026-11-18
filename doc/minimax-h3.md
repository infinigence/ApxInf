# MiniMax-H3: INT8 ConvRot text-to-video-and-audio on CUDA

The `minimax-h3` family generates a clip, RGB8 frames and stereo audio, from a prompt through `AutoModel`,
`LoadedModel::Video` and [`VideoRuntime`](../crates/apxinf-model/src/video.rs). It runs the Comfy-Org INT8 ConvRot release of the
MiniMax-H3 first-and-last-frame-to-video-and-audio DiT, used here for text-only generation: the same 50-block DiT as
[MiniMax-H3-Image](minimax-h3-image.md) over a packed `text | audio | video` sequence, the streamed Qwen3-VL 32B text encoder, a 36-layer
ViT video VAE decoder and a BigVGAN audio VAE decoder, all on the `apxinf-cuda-new` backend. No PyTorch, ComfyUI or diffusers code runs
at inference.

The family lives in [`crates/apxinf-model/src/cuda_minimax_h3/`](../crates/apxinf-model/src/cuda_minimax_h3/mod.rs) and is registered as
`cuda_minimax_h3` and `minimax-h3`. It is a separate copy of the image family's code, as the repository's family isolation requires.

## Checkpoint and package

| Item | Selection |
| --- | --- |
| Source | [`Comfy-Org/MiniMax-H3`](https://huggingface.co/Comfy-Org/MiniMax-H3), the three files below (DiT 20,970,379,616 bytes, video VAE 5,207,808,496, audio VAE 605,254,808); the revision was not recorded when they were downloaded |
| DiT | `diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors` (about 19.5 GiB) |
| Video VAE | `vae/minimax_h3_video_vae_fp16.safetensors` (about 4.8 GiB, FP16) |
| Audio VAE | `vae/minimax_h3_audio_vae_fp32.safetensors` (about 0.56 GiB, FP32) |
| Text encoder | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` of the MiniMax-H3-Image package (25.3 GiB file); the same file serves both families |
| Tokenizer | Qwen2 `vocab.json`, `merges.txt`, `tokenizer_config.json` |
| Precision | Linear layers INT8 ConvRot W8A8; norms, attention, embeddings and residual stream BF16; patch projections and both heads FP32; video VAE BF16 from FP16 weights; audio VAE FP32 |
| Device | CUDA, sm_80 or newer. The direct W8A8 weight path is built for sm_87 and sm_89 only; other devices take the canonical path (a host transpose per layer fill) and were not run. Measured on an RTX 4090 (sm_89, 24 GiB) |

The checkpoints are third-party INT8 requantizations of the MiniMax-H3 weights. The license terms of this release were not reviewed for this
family; the image release's license excludes some regions, so check the one that applies before using the weights. This repository ships
no weights.

A package is one directory with the files above and a manifest named `apxinf_h3.json`, which `AutoModel` recognises the family by:

```json
{
  "schema": "apxinf.minimax-h3.v1",
  "transformer": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
  "video_vae": "vae/minimax_h3_video_vae_fp16.safetensors",
  "audio_vae": "vae/minimax_h3_audio_vae_fp32.safetensors",
  "text_encoder": "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
  "tokenizer": "tokenizer"
}
```

Paths are relative; absolute paths, `..`, unknown fields and another schema string are rejected, and a missing part is named in the error.

## Request contract

| Field | Accepted |
| --- | --- |
| `steps` | 1 to 50. The sigma schedule is ComfyUI's `simple` scheduler over the flow-shifted 1000-point grid (video shift 12, audio shift 3); the parity traces use 4 |
| `width`, `height` | Multiples of 32 in 256 to 1920 |
| `frames` | 5 to 362, snapped up onto the `17 k + 5` grid the model uses (the result reports the real count); 362 is the largest grid value in the trained range, so no accepted count snaps beyond it. 24 fps. ComfyUI's node describes the trained range as about 124 to 362 frames and longer clips as untested; shorter clips are accepted here but their quality was not checked. The check does not look at memory: on a 24 GiB card 768x448 stops between 39 and 73 frames, and the 124 frames of the trained range fit only at 384x256 (see Validation and limits) |
| `prompt` | 1 to 512 tokens (raw text, no chat template). A longer prompt is rejected, not truncated |
| `seed` | Reproduces ComfyUI's `prepare_noise`: the video noise and then the audio noise drawn from one generator, as `torch.randn` would. Only the low 32 bits of the seed are used, as in torch |

The result is `[frames, height, width, 3]` RGB8 (ComfyUI's truncating float-to-byte conversion) and interleaved stereo `f32` at 32 kHz.
There is no guidance scale, negative prompt or keyframe, reference-video or audio-guide input; those segment kinds of the model are not
implemented. Unsupported requests fail before any device work, and so do the load options this family does not implement (`precision` other than `auto`/`w8a8`, `model_variant` other than `auto`/`int8_convrot`, replacement assets, synthetic weights, a config override, calibration, view overrides and a text weight dtype). `autotune` and the tuning path are honoured; the execution policy, layout budget, text-profile limits and generation settings do not apply to this family and are ignored.

## Computation and ownership

One request: tokenize, text encoder (layers 0-49), token refiner, `steps` DiT forwards over the packed sequence with an FP32 Euler update
of each stream, video VAE decode, audio VAE decode.

- **Packed sequence** (`model/layout.rs`): text rows, then the audio latent as `[2 * T, 32]` rows (channel-major), then the patched video
  latent (`2 x 2` patches). Position ids are float64 `(t, h, w)` as ComfyUI builds them (video `t` advances by the 1, 4, 4, 4, 4 frame
  pattern rescaled by 5/3; audio `w` is pinned to the first and last column coordinate).
- **DiT** (`model/blocks/int8.rs`): the image family's block (ConvRot W8A8 linears, partial RoPE as a channel permutation, curve-form AdaLN)
  with three modality rows (visual, text, audio) and a per-stream timestep: text and video rows use the video stream's `t`, audio rows the
  audio stream's, whose sigma is `time_shift_sigma(sigma_video, 12, 3)`. Two FP32 heads (video 96 outputs, audio 32) with their modulation
  folded in. ComfyUI's sampler carries the audio latent in a rescaled form; the rescaling does not reach its output, so this family samples
  each stream in its own scale and the final latents match ComfyUI's (`sampler_final_latents_match_comfyui`).
- **Text encoder**: as in the image family (`weights/text.rs`, `model/blocks/text.rs`): streamed layer by layer by DMA from the page-locked
  checkpoint; `APXINF_H3_TE_PIN` overrides the automatic page-locking decision (see the environment variables below). Prompt conditioning is cached for the last 8
  distinct token sequences.
- **Video VAE** (`weights/vae.rs`, `model/blocks/vae.rs`, `model/vae_host.rs`, `model_runner/video.rs`): 36-layer ViT decoder, dimension 2048,
  32 heads, 3-axis RoPE, over 256-pixel tiles (at least 64 pixels of overlap) of 7-latent-step chunks (5 new steps plus 2 of overlap),
  following ComfyUI's `decode_temporal` and `tiled_decode`. The decoder is 4.5 GiB as BF16 and does not fit beside the DiT, so
  the weights are converted once at load into one page-locked host buffer and streamed block by block through a two-slot device ring while
  the previous block computes; tiles of a chunk are batched (default 2) so one pass over the weights decodes several. A batch runs inside a
  prepared `ExecutionSession`, so intermediates come from a fixed arena and the GEMM and attention executions are bound once.
  `APXINF_H3_VAE_BATCH` raises the tile batch where memory allows; `APXINF_H3_DROP_PLAN` frees the denoising plan before decoding.
- **Audio VAE** (`weights/audio_vae.rs`, `model/blocks/audio_vae.rs`, `model_runner/audio.rs`): BigVGAN decoder, FP32 throughout, on the new
  `apxinf-cuda-new` operators `conv1d`, `conv_transpose1d` and `snake_beta` (below). 260 MB resident.
- **Preparation** (`model_runner/prepare.rs`): a plan per `(layout, steps)` holds the persistent latents, text rows and head outputs and a
  captured CUDA Graph of the 50-block forward; between replays only the step's AdaLN table and folded heads are copied.
- **Environment variables**. `APXINF_H3_EAGER` (run the DiT eagerly instead of replaying the captured graph: debugging, parity checks),
  `APXINF_H3_TIMINGS` (print per-stage wall time; each stage then ends with a device synchronisation, so leave it off for latency numbers)
  and `APXINF_H3_DROP_PLAN` are on when set to any value, including `0`. `APXINF_H3_VAE_SESSION=0|off` decodes the video without a
  prepared session. `APXINF_H3_TE_PIN` is `0|off` (never page-lock), `1|on` (page-lock, and fail the load if that is refused) or unset
  (automatic). `APXINF_H3_VAE_BATCH=<tiles>` must be a positive integer; anything else fails the load.
- **Additions to `apxinf-cuda-new`**: FP32 `ops::conv1d` (zero or replicate padding, stride, dilation, dense or depthwise),
  `ops::conv_transpose1d` (replicate input padding, left crop, output scale, dense or depthwise) and `ops::snake_beta`, with catalog entries
  in `cuda-operator.md` and behaviour tests against FP64 references. They are direct kernels rather than GEMM lowerings.
- **Additions to `apxinf-model`**: `video.rs` (`VideoRequest`, `GeneratedVideo`, `VideoRuntime`), `LoadedModel::Video`
  (`video_mut`, `generate_video`).

Module boundaries are enforced by
[`scripts/check_minimax_h3_module_boundaries.py`](../scripts/check_minimax_h3_module_boundaries.py), called from
`scripts/check_model_family_boundaries.sh`.

## Validation and limits

Reference: ComfyUI `c3fd8b9ab1fd` with comfy-kitchen 0.2.41 on the same files (torch 2.11.0+cu128, Triton backend, BF16 attention because
driver 550 cannot run its INT8 attention), on the same RTX 4090. The accuracy rows use one prompt ("A red fox trotting through a snowy pine
forest, birds singing."), 768x448, 22 frames, 4 steps, seed 42. The parity tests are `#[ignore]`d because they need the checkpoints and a
trace exported from ComfyUI; their environment variables are listed in
[`tests/parity.rs`](../crates/apxinf-model/src/cuda_minimax_h3/tests/parity.rs). The ComfyUI-side scripts that write those reference
directories are named in the test comments and are not part of this repository.

| Check | Result |
| --- | --- |
| Packed position ids vs ComfyUI | Max difference 7e-15 over 2440 rows |
| Seed noise (video then audio) vs ComfyUI | Max abs 4.8e-7 (video), 2.4e-7 (audio) |
| DiT, teacher-forced per step | Video relative L2 0.0265 / 0.0230 / 0.0152 / 0.0147, audio 0.0241 / 0.0264 / 0.0744 / 0.0260; cosine at least 0.9972 |
| DiT residual stream, blocks 0 to 24 | 0.3 % to 2 % on every segment and step; after block 49: video 3.6 % to 4.3 %, audio 3.2 % to 11.6 %. Where the growth in the last blocks starts was not located |
| Sampler from ComfyUI's noise and context | Final video relative L2 0.0598, audio 0.0770 |
| Text encoder (streamed, BF16 and INT8) vs ComfyUI's FP32 stream | Final layer relative L2 0.022; refiner on ComfyUI's encoder output 0.0057; refined context 0.033 |
| Prompt to final latents, whole path | Video 0.1715, audio 0.1124 |
| Video VAE, one tile | Pixels relative L2 0.0118 |
| Video VAE, whole clip on ComfyUI's latents | Relative L2 0.0035, mean abs 0.35 / 255 |
| Audio VAE vs ComfyUI strict FP32 | Waveform relative L2 3.6e-6 / 4.1e-6. ComfyUI's default (TF32) differs from its own strict-FP32 result by 1.2e-3 / 3.2e-3 |
| Frames of the whole path vs ComfyUI, four prompts of 14 tokens | Mean abs 14.0 / 5.9 / 13.8 / 4.9 per 255 (fox, violin player, steam train, hot-air balloon), 22 frames each. With ComfyUI's context and noise fed through APXInf's sampler and decoder, the fox clip differs by 4.9 / 255 |
| Audio of the same four clips, left channel, vs ComfyUI | Waveform correlation 0.96 / 0.96 / 0.98 / 0.99 and log-spectrogram correlation 0.96 / 0.98 / 0.96 / 0.98. For scale, decoding ComfyUI's own audio latents after adding Gaussian noise of 7.7 % and 11 % relative norm gives waveform correlation 0.82 to 0.94 |

The frames are not pixel-identical to ComfyUI's. For the fox prompt the animal is smaller and further right in ComfyUI's frame 11 (about
60 px wide, around x 536 of 768) than in APXInf's (about 90 px, around x 395). The difference follows from the text context. ComfyUI
against itself, with Gaussian noise of relative norm 0.022 (the size of APXInf's text-encoder error) added to its own encoder output,
moves its frames by 11 to 21 / 255 and its final latents by 0.18 to 0.22 in three trials, which is as much as the whole-path difference
to APXInf. This bounds the sensitivity of a 4-step model to a conditioning error of that size. It does not show that APXInf's error is
benign: three trials, one prompt, one seed, Gaussian noise rather than APXInf's structured error, and no perceptual metric. There is no
noise floor for the DiT: at identical inputs ComfyUI's eager and Triton backends return bit-identical outputs.

The output also depends on the tuning-recipe cache. Recipes are cached under `configs/tuning/` relative to the working directory. Same
binary, prompt and seed (768x448, 22 frames, 4 steps), three stored caches:

| Cache | Contents | Frames vs A, mean abs per 255 | Frames vs ComfyUI | Audio vs A, relative L2 |
| --- | --- | --- | --- | --- |
| A | 92 recipe files tuned on the test machine | 0 | 13.97 | 0 |
| B | 46 recipe files | 0.28 | 13.98 | 0 |
| C | 27 recipe files tuned from an empty directory | 9.10 | 9.43 | 0.35 |

Two more runs started from an empty directory and tuned 27 recipes each. Both differ from A by 0.26 and from ComfyUI by 13.97, and their frames are
identical to each other, although 1 of their 27 recipe files differs. `*.recipe` files are not tracked in
the repository, so a clean checkout starts from an empty directory. Each cache is deterministic: a second run from A gave identical
frames. Compare runs only from one cache, and ship or pin the recipes of a deployment. Why a different GEMM or attention choice moves a
4-step result this much was not investigated.

### Latency

RTX 4090, 768x448, 22 frames, 4 steps, seed 42, both sides with BF16 attention. Four prompts of 14 tokens each, so the denoising plan
is reused; APXInf is the mean of 3 steady-state requests (new prompt) and 4 (prompt seen before), ComfyUI of 7 and 3.

| | APXInf | ComfyUI |
| --- | --- | --- |
| New prompt | 4.26 s: text encoder and refiner 1.47, DiT 4 x 0.30, video decode 1.40, audio and rest 0.21 | 6.11 s: text 2.37, sample 2.52, video decode 1.03, audio and rest 0.20 |
| Prompt seen before | 2.80 s: DiT 1.19, video decode 1.41, audio and rest 0.21 | 2.85 s: sample 1.73, video decode 0.88, audio and rest 0.24 |

The video decode is slower than ComfyUI's. Of its 1.4 s, 0.77 s is on the device and the rest is host code (download and unpatchify 0.25,
tile blend 0.26, splice 0.12). The first request after load adds the plan build and capture, 1.6 s, and a request whose prompt has a
different token count rebuilds the plan again: with prompts of 14, 17, 16 and 15 tokens the new-prompt mean is 5.75 s (three
requests after the first) and the seen-prompt mean 4.07 s (four requests). Loading takes about 70 s with the files in the page cache, which includes converting the video VAE into its
page-locked buffer.

### Memory and clip length

At 768x448 and 22 frames the DiT, the text ring, the video decoder's weight ring and both plans use about 23.2 GiB of the card's 24. The tile batch
of the video decoder is therefore 2 (4 runs out of memory with the denoising plan resident). The request check accepts up to 362 frames,
but the card does not. The plan also keeps the folded per-step tables resident, 11.85 MiB per step, so 50 steps hold about 593 MiB
against 47 MiB at 4; the sizes below were measured at 4 steps. One prompt, 4 steps, two requests per size:

| Canvas | Frames | Result |
| --- | --- | --- |
| 768x448 | 22 / 39 | runs (2.8 s / 5.3 s with the prompt cached) |
| 768x448 | 73 | out of memory while the denoising plan is built |
| 896x512, 1024x576 | 22 | runs (4.5 s / 5.2 s) |
| 640x384 | 56 | runs (5.8 s) |
| 512x320 | 73 | runs (6.4 s) |
| 512x320 | 124 | the four steps finish, out of memory in the video decode |
| 384x256 | 124 | runs (4.7 s) |

With `APXINF_H3_DROP_PLAN` the first request of 512x320 x 124 and of 640x384 x 73 finishes (15.5 s and 11.7 s) and the second request in
the same process runs out of memory, so it is not a way to serve long clips; 640x384 x 124 fails while the plan is built. Peak
`nvidia-smi` memory peaked at 23.2 to 23.5 GiB (24.9 to 25.2 GB) of the card's 24 GiB in every run, including the two that ran out of memory, so only a failure tells which size is close to the limit.

### Other limits

- **Page-locking 25.3 GiB for the text encoder file plus about 4.5 GiB for the video VAE buffer.** The text encoder is page-locked only
  when the host (or its container limit) keeps at least 8 GiB beyond the file, and falls back to the pageable path otherwise or when the
  driver refuses. The video VAE buffer is always attempted and falls back only when the driver refuses. The image family measured the
  text encoder's pageable path at about 3x slower per cold prompt; the video VAE's was not measured.
- **Eager glue.** The text stage, the per-step Euler update, tile blending and the RGB conversion run outside the graphs. About 0.6 s of
  each request's video decode is host code.
- **Baselines not run.** ComfyUI's INT8 attention variant needs a CUDA 13 runtime; no diffusers BF16 or SGLang run exists.
- **Not implemented.** Keyframes, reference video or audio, the other segment kinds of the model, and the video and audio encoders.
- **Quality is not measured with a perceptual metric.** There is no LPIPS or CLIP score, and no listening test of the audio.

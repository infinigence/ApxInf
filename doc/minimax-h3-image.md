# MiniMax-H3-Image: INT8 ConvRot text-to-image on CUDA

The `minimax-h3-image` family generates one RGB8 picture from a prompt through
`AutoModel`, `LoadedModel::Image` and [`ImageRuntime`](../crates/apxinf-model/src/image.rs).
It runs the Comfy-Org INT8 ConvRot release of MiniMax-H3-Image: a 50-block DiT,
a streamed Qwen3-VL 32B text encoder and a ViT VAE decoder, all on the
`apxinf-cuda-new` backend. No PyTorch, ComfyUI or diffusers code runs at inference.

The family lives in
[`crates/apxinf-model/src/cuda_minimax_h3_image/`](../crates/apxinf-model/src/cuda_minimax_h3_image/mod.rs)
and is registered as `cuda_minimax_h3_image` and `minimax-h3-image`.

## Checkpoint and package

| Item | Selection |
| --- | --- |
| Source | [`Comfy-Org/MiniMax-H3-Image`](https://huggingface.co/Comfy-Org/MiniMax-H3-Image), revision `0bb149f51e31` |
| DiT | `diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors` (about 19.5 GiB) |
| Text encoder | `text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (about 22.7 GiB of layers 0-49; the whole file is 25.3 GiB with the embedding table and the vision tower) |
| VAE | `vae/minimax_h3_image_vae_fp16.safetensors` |
| Tokenizer | Qwen2 `vocab.json`, `merges.txt`, `tokenizer_config.json` from the same repository |
| Precision | Linear layers INT8 ConvRot W8A8; norms, attention, embeddings and residual stream BF16; `video_patch_proj` and the PDD heads FP32; VAE computes in BF16 from FP16 weights |
| Device | CUDA, sm_80 or newer. The direct W8A8 weight path is built for sm_87 and sm_89 only; other devices take the canonical path (a host transpose per layer fill) and were not run. Measured on an RTX 4090 (sm_89, 24 GiB) |

The checkpoints are third-party INT8 requantizations of the MiniMax-H3-Image
weights, used under the MiniMax-H3-Image Community License. That license excludes
some regions (US, UK, EU, KR at the time of writing); check it before using the
weights. This repository ships no weights.

A package is one directory with the three files above, a tokenizer directory and a
manifest named `apxinf_h3_image.json`. `AutoModel` recognises the family by the
manifest, not by file names:

```json
{
  "schema": "apxinf.minimax-h3-image.v1",
  "transformer": "diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors",
  "vae": "vae/minimax_h3_image_vae_fp16.safetensors",
  "text_encoder": "text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
  "tokenizer": "tokenizer"
}
```

Paths are relative to the package directory. Absolute paths, `..`, unknown
fields and a different schema string are rejected, and a missing part is named in
the error.

## Request contract

| Field | Accepted |
| --- | --- |
| `steps` | Exactly 6. The checkpoint is a 6-step PDD release with the fixed schedule `[1.0, 0.9675237, 0.88754855, 0.75, 0.5327739, 0.23200887, 0.0]`; the loader verifies the schedule stored in the DiT metadata |
| `width`, `height` | Multiples of 32, each at least 256, area from 1024² to 4096² pixels. The check does not look at memory: the plan's arena grows with the canvas (about 160 KiB per packed row plus 384 MiB), and canvases above 2048² were not run on a 24 GiB card |
| `prompt` | At most 512 tokens (raw text, no chat template, no special tokens). A longer prompt is rejected, not truncated. The bound is a deployment choice; the released pipelines set none |
| `seed` | Reproduces the starting noise of `torch.randn` for the same seed on the same latent shape (MT19937 and torch's Box-Muller fill, checked against torch). Only the low 32 bits of the seed are used, as in torch: seeds that differ by a multiple of 2^32 give the same image |

There is no guidance scale, negative prompt or scheduler choice. Unsupported
requests fail before any device work. Load options that this family does not
implement (`precision` other than `auto`/`w8a8`, `model_variant` other than
`auto`/`int8_convrot`, replacement assets, synthetic weights, calibration, view
overrides) are rejected with an error that names the family.

Image editing, the vision tower and BF16/FP8 packages are out of scope.

```bash
cargo run --release -p apxinf-model --features cuda --example minimax_h3_image_generate -- \
  --package /path/to/h3-image-package --prompt "A red fox trotting through a snowy pine forest." \
  --size 1024 --seed 42 --repeat 2 --output fox.ppm
```

`--repeat` runs the same request again to show the cached-prompt latency.
`APXINF_H3_IMAGE_TIMINGS` prints per-stage times, and `APXINF_H3_IMAGE_EAGER`
runs the same operators without CUDA Graph capture for debugging. Both are on when
set to any value, including `0`.

## Computation and ownership

One request runs: tokenize, text encoder (layers 0-49, last hidden state), token
refiner, six DiT steps with an FP32 Euler update, latent denormalization, tiled
VAE decode, RGB8 conversion.

- **DiT** (`model/blocks/int8.rs`): 50 blocks, hidden 5376, 56 heads of 128,
  SwiGLU FFN 14336. Each INT8 linear first applies the ConvRot activation
  transform (256-wide Hadamard groups in FP32, then rowwise INT8) in one kernel,
  then runs a CUTLASS W8A8 GEMM against pre-rotated weights. Partial RoPE (96 of
  128 channels) is expressed as a fixed channel permutation of the q/k rows and
  norm weights plus `rms_norm_apply_precomputed` with an FP32 cos/sin table.
  AdaLN uses the checkpoint's curve form: per-block F16 weights evaluated in FP32
  on the `adaln_t_table` row of the step, rounded to BF16, with separate rows for
  image and text tokens. The PDD heads are `block0 + block_i` with the final
  modulation folded in.
- **Text encoder** (`model/blocks/text.rs`, `weights/text.rs`): Qwen3 with 64/8 grouped heads, head
  dimension 128, FFN 25600, half-split RoPE with theta 5e6, no final norm. The
  checkpoint's `fc1` stores `[gate; up]`, so the first half is the gate; this
  differs from the official diffusers file ordering. The 22.7 GiB of layer weights do
  not fit beside the DiT on a 24 GiB card, so every cold prompt streams them.
  The checkpoint is mapped once at load and page-locked (`cudaHostRegister`), so a
  layer moves from the page cache to the GPU by DMA: no read into the heap, no
  staging copy, no repack, because the file's `[N, K]` bytes are the layout the
  W8A8 kernels execute. Two sets of fixed-address device slots form a ring; a
  copy stream fills one set with layer `i + 1` while the compute stream runs layer
  `i`, ordered by events. A request is then limited by one pass over PCIe.
- **VAE** (`model/blocks/vae.rs`): 2D ViT decoder, 24 layers, dimension 1024,
  4 register tokens, RoPE theta 100, decoded in 256-pixel tiles with 64-pixel
  overlap and linear blending, four tiles per batch.
- **Preparation** (`model_runner/prepare.rs`): a plan per `(text rows, canvas)`
  holds persistent latent, text, velocity and a captured CUDA Graph of the
  50-block forward. Between replays only the per-step AdaLN table and the folded
  head are copied into stable buffers. The VAE batch is also graph-captured.
  Prompt conditioning is cached for the last 8 distinct token sequences, so a new
  seed or canvas with the same prompt skips the text encoder.
- **Additions to `apxinf-cuda-new` and `apxinf-loader`**: `QuantizationSemantic::RotateRowwiseI8`
  (ConvRot rotate and quantize, ABI spec v3), CUTLASS W8A8 output-major candidates
  enabled for sm_89 (previously sm_87 only), `RepackedWeight::upload_row_channel_output_major`,
  `OutputMajorSlot` (a reusable W8A8 weight address that is refilled layer after layer),
  `PinnedRegion` with `CudaBuffer::copy_from_region_async`, `CudaEvent` with
  `CudaStream::{new_on, record, wait_for}`, and `apxinf_loader::MappedFile`.
  `rms_norm_apply_precomputed` no longer pins arena leases for the whole
  execution, which made graph capture fail with a GEMM execution-cache miss; a
  regression test covers it.

Module boundaries are enforced by
[`scripts/check_minimax_h3_image_module_boundaries.py`](../scripts/check_minimax_h3_image_module_boundaries.py),
called from `scripts/check_model_family_boundaries.sh`.

## Validation and limits

Reference: ComfyUI `c3fd8b9ab1fd` with comfy-kitchen 0.2.41 running the same
checkpoint files (torch 2.11.0+cu128, comfy-kitchen Triton backend, BF16
attention), on the same RTX 4090. The tests below are `#[ignore]`d because they
need the checkpoints and a trace exported from ComfyUI; their environment
variables are listed in
[`tests/parity.rs`](../crates/apxinf-model/src/cuda_minimax_h3_image/tests/parity.rs).

| Check | Result |
| --- | --- |
| DiT, teacher-forced per step against the ComfyUI trace (1024², seed 42) | Relative L2 0.014 / 0.025 / 0.046 / 0.030 / 0.015 / 0.019 for steps 0-5, cosine at least 0.9989 |
| ComfyUI eager vs its Triton backend, identical `x`, timestep and text context, all 6 steps | Relative L2 0 (bit-identical). Each backend served every `int8_linear` call (1200 per run). There is no backend noise to compare against, so the DiT rows above are differences between two implementations, not distances inside a noise band |
| ComfyUI's two full runs (eager vs Triton), step 0 | Relative L2 0.0151, caused by the text context: the runs' contexts differ by 0.40 % (relative L2), and feeding one run's context into the other run's own step-0 input reproduces 0.0151 exactly. For scale, a 0.4 % change of the conditioning moves the step-0 output by 1.5 % |
| Text encoder output vs ComfyUI's FP32 stream over the same weights | Relative L2 0.022, cosine 0.99997 |
| VAE decode of the same latent vs ComfyUI | Mean absolute difference 0.29 / 255 |
| Seed noise vs `torch.randn` | Below 1e-4 absolute |
| CUDA Graph vs eager, same recipes | Bit-identical latent and image |
| Final image vs ComfyUI | Mean absolute difference 6.09 / 255 at 1024² and 2048² with the recipe set tuned on the test machine (recipes are not tracked in the repository), 7.44 / 255 at 1024² with a second recipe set. Two full ComfyUI runs (eager and Triton backends) differ from each other by 6.69 / 255. Their DiTs are bit-identical at identical inputs, so that gap comes from the text context (0.40 % apart between the runs), amplified by the trajectory. Which stage of the text encoder differs was not isolated |

Final images are not pixel-identical to ComfyUI. INT8 rounding differs between
implementations, and the free-running latent trajectories of two full ComfyUI runs
(eager and Triton backends, which differ in their text context) already diverge by
about 14 % relative L2 after five steps.

The same sensitivity applies to APXInf's own tuning. GEMM and attention recipes
are cached under `configs/tuning/` relative to the process working directory (or
the `tuning_path` load option). Two recipe sets tuned on the same GPU selected
different candidates and ended with final latents that differ by 12 % relative
L2 (images 5.9 / 255 apart), while a fixed recipe set reproduces latents bit for
bit, eager or graph. For reproducible images, reuse one recipe directory, or load
with `autotune` off after a first tuned run. The VAE decode of one latent is
the same under all four combinations of tuning and capture (0.277 / 255 against
ComfyUI).

Latency on an RTX 4090 (seed 42, 6 steps, steady state). A cold prompt is one the family has not seen among
the last 8 distinct prompts. ComfyUI is shown twice: a full request re-runs its text encoder, and a repeat of the same
prompt skips it (denoise plus VAE only), which is the like-for-like counterpart of APXInf's cached prompt.

| Canvas | APXInf cached prompt | APXInf cold prompt | ComfyUI same prompt | ComfyUI full request |
| --- | --- | --- | --- | --- |
| 1024² | 0.83 s | 3.3 s | 1.4 s | 4.9-5.1 s |
| 2048² | 3.8 s | 6.3 s | 5.3 s | 8.6-9.5 s |

A cold prompt's text stage takes 1.32 to 1.37 s on four different prompts. The floor is one
pass of 22.7 GiB (24.4 GB) over the measured pinned host-to-device rate of about 22.7 GB/s, 1.07 s.
The DiT forward takes 0.116 s at 1024² with the graph (about 1.0 s eager) and 0.55 s
at 2048². The VAE decodes 1024² in 0.13 s. A request whose token count or canvas
differs from the previous request also rebuilds and recaptures the plan (1.0 s at
1024²), and the cold-prompt figures above include that. Loading takes about 42 s with the
files in the page cache.

Limits that remain:

- **Text encoder streaming and host memory.** Streaming 22.7 GiB per cold prompt is still
  a weight transfer, and its speed depends on the file being page-locked. Page-locking
  pins the 25.3 GiB file in host memory for the life of the model. When the host (or its
  container limit) would be left with less than 8 GiB beyond the file, or the driver
  refuses, the family prints a message and streams through the pageable path, which
  is about 3x slower (4 to 5 s instead of 1.3 s per cold prompt) with identical results.
  `APXINF_H3_IMAGE_TE_PIN=0` (or `off`) forces that path and `=1` (or `on`) makes pinning mandatory; any other value leaves the decision automatic. The
  text stage runs eagerly, outside any CUDA Graph; its compute is 0.66 s of the 1.33 s,
  overlapped with the transfer. The file must not be modified while the model is loaded.
- **Eager glue.** The per-step Euler update and the VAE tile blending run
  outside the captured graphs.
- **ConvRot kernel.** The rotate-and-quantize kernel reaches 610 to 680 GB/s on the DiT's
  shapes (the plain rowwise INT8 quantizer 760 to 1430 GB/s) and has not been tuned further.
- **Baselines not run.** ComfyUI's default INT8-attention variant needs a CUDA 13
  runtime and cannot run on driver 550, so only its BF16-attention variant is
  compared. A diffusers BF16 run of the official weights and an SGLang run were
  not done.
- **No noise floor.** At identical inputs ComfyUI's eager and Triton backends
  return bit-identical DiT outputs, so there is no backend-to-backend floor to
  judge the 1.4-4.6 % per-step differences against; they are APXInf-vs-ComfyUI
  differences in full. The 0.0151 between two full ComfyUI runs at step 0 is not
  such a floor either: those runs' text contexts differ by 0.40 %, and that alone
  gives 0.0151. The differences between the full runs (0.0151 at step 0, growing
  with the trajectory) therefore include the text encoder, whose backend
  sensitivity was not isolated.
- **Test environment.** `apxinf-cuda-new --lib` on the 4090, single-threaded:
  on an unmodified `main` 2 to 4 tests fail, a different number from run to run,
  and on this branch 4 failed in each of 4 runs. They are
  `fused_add_channel_l2_norm_silu_all_candidates_match_torch`,
  `rms_rope_matches_unfused_cpu_alias_and_retained_graph`,
  `tiled_unfold_history_padding_stride_and_graph_rebinding` and
  `unfold_accepts_bf16_aligned_views_without_vector_alignment`; each also fails on
  `main` in some runs. The parallel harness segfaults on this card for both trees.
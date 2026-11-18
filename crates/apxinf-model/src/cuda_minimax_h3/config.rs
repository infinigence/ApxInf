/// Geometry of the released MiniMax-H3 (video and audio) DiT. Every other module takes its
/// dimensions from here, and the weight loader rejects any checkpoint that disagrees, so a different
/// model never runs under these shapes by accident.
pub(crate) const HIDDEN: usize = 5376;
pub(crate) const BLOCKS: usize = 50;
pub(crate) const HEADS: usize = 56;
pub(crate) const HEAD_DIM: usize = 128;
pub(crate) const INNER: usize = HEADS * HEAD_DIM;
pub(crate) const FFN: usize = 14336;
pub(crate) const TEXT_DIM: usize = 5120;
pub(crate) const REFINER_BLOCKS: usize = 2;
pub(crate) const LATENT_CHANNELS: usize = 24;
pub(crate) const PATCH: usize = 2;
pub(crate) const PATCH_DIM: usize = LATENT_CHANNELS * PATCH * PATCH;
/// Pixels per latent pixel along each spatial side, and frames per latent step in time.
pub(crate) const VAE_RATIO: usize = 16;
/// Canvas sides are multiples of this (latent patch of 2 times the VAE ratio).
pub(crate) const CANVAS_MULTIPLE: usize = PATCH * VAE_RATIO;
/// Audio latent: 32 channels, 2 stereo channels, `AUDIO_LATENT_FPS` latent frames per second.
pub(crate) const AUDIO_CHANNELS: usize = 32;
pub(crate) const AUDIO_STEREO: usize = 2;
pub(crate) const AUDIO_LATENT_FPS: usize = 40;
/// Video frame rate; the frame count lies on the `17 k + 5` grid.
pub(crate) const FPS: usize = 24;
/// Rotated channels of every head are `2 * 3 * ROPE_FREQS = 96`: three axes of `ROPE_FREQS` frequencies, duplicated.
/// The frequencies themselves (theta 10000) come from the checkpoint's `rope.inv_freq` buffer.
pub(crate) const ROPE_FREQS: usize = 16;
pub(crate) const NORM_EPS: f32 = 1e-5;
/// Rows of the AdaLN curve table and its rank.
pub(crate) const ADALN_GRID: usize = 1025;
pub(crate) const ADALN_RANK: usize = 8;
/// Modalities that address AdaLN rows: visual 0, text 1, audio 2.
pub(crate) const MODALITIES: usize = 3;
pub(crate) const MODALITY_VISUAL: usize = 0;
pub(crate) const MODALITY_TEXT: usize = 1;
pub(crate) const MODALITY_AUDIO: usize = 2;
/// ConvRot group size shared by activation rotation and the weight files.
pub(crate) const CONVROT_GROUP: usize = 256;

/// Flow shifts of the two target streams (`MiniMaxH3Model` defaults, set by ComfyUI's `ModelSamplingMiniMaxH3`).
pub(crate) const SIGMA_SHIFT_VIDEO: f32 = 12.0;
pub(crate) const SIGMA_SHIFT_AUDIO: f32 = 3.0;
/// Temporal position advance per latent step: five-step cycle of frame counts per token, rescaled.
pub(crate) const FRAME_PER_TOKEN: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
pub(crate) const FRAME_RESCALE: f64 = 5.0 / 3.0;

/// Text encoder (Qwen3-VL 32B language stack, first 50 of 64 layers) geometry.
pub(crate) const TE_DIM: usize = 5120;
pub(crate) const TE_LAYERS: usize = 50;
pub(crate) const TE_HEADS: usize = 64;
pub(crate) const TE_KV_HEADS: usize = 8;
pub(crate) const TE_HEAD_DIM: usize = 128;
pub(crate) const TE_Q: usize = TE_HEADS * TE_HEAD_DIM;
pub(crate) const TE_KV: usize = TE_KV_HEADS * TE_HEAD_DIM;
pub(crate) const TE_FFN: usize = 25600;
pub(crate) const TE_VOCAB: usize = 151_936;
pub(crate) const TE_EPS: f32 = 1e-6;
pub(crate) const TE_ROPE_THETA: f32 = 5_000_000.0;
/// Prompt rows are padded up to a multiple of this so the INT8 GEMMs see few distinct shapes; the model is causal,
/// so padding rows at the end never influence the real rows.
pub(crate) const TE_ROW_BUCKET: usize = 64;

/// Longest prompt, in tokens, a request may carry. A deployment bound (it limits the packed sequence and the text stream's
/// scratch), not a property of the checkpoint: longer prompts are rejected rather than truncated.
pub(crate) const MAX_TEXT_TOKENS: usize = 512;

/// Video VAE decoder (`minimax_h3_video_vae_fp16.safetensors`): a ViT over 16x16-pixel, 4-frame patches of the latent grid.
pub(crate) const VV_DIM: usize = 2048;
pub(crate) const VV_HEADS: usize = 32;
pub(crate) const VV_HEAD_DIM: usize = 64;
pub(crate) const VV_FFN: usize = 8192;
pub(crate) const VV_BLOCKS: usize = 36;
pub(crate) const VV_REGISTERS: usize = 4;
/// Register tokens plus one zero token follow the latent tokens of every tile.
pub(crate) const VV_SUFFIX: usize = VV_REGISTERS + 1;
pub(crate) const VV_ROPE_THETA: f32 = 100.0;
/// Rotated channels of a head (`0.75 * VV_HEAD_DIM`): three axes of `VV_ROPE_FREQS` frequencies, split-half pairs `(i, i + 24)`.
pub(crate) const VV_ROPE_DIM: usize = 48;
pub(crate) const VV_ROPE_FREQS: usize = 8;
/// Pixel frames per latent step.
pub(crate) const VV_RATIO_T: usize = 4;
pub(crate) const PIXEL_CHANNELS: usize = 3;
/// Decoder outputs per token: `PIXEL_CHANNELS * VV_RATIO_T * VAE_RATIO^2`.
pub(crate) const VV_OUT_DIM: usize = PIXEL_CHANNELS * VV_RATIO_T * VAE_RATIO * VAE_RATIO;
pub(crate) const VV_TILE_PIXELS: usize = 256;
pub(crate) const VV_TILE_OVERLAP: usize = 64;
pub(crate) const VV_TILE_LATENT: usize = VV_TILE_PIXELS / VAE_RATIO;
/// Temporal chunking: a chunk decodes `VV_CHUNK_LATENT` latent steps (5 new + 2 of overlap) into 28 frames, of which 17 are kept.
pub(crate) const VV_CLIP_FRAMES: usize = 17;
pub(crate) const VV_TOKEN_DROP: usize = 3;
pub(crate) const VV_CHUNK_TOKENS: usize = 5;
pub(crate) const VV_TOKEN_OVERLAP: usize = 2;
pub(crate) const VV_CHUNK_LATENT: usize = VV_CHUNK_TOKENS + VV_TOKEN_OVERLAP;
pub(crate) const VV_FRAME_PRE_PADDING: usize = 3;
pub(crate) const VV_FRAME_OVERLAP: usize = 5;
/// ImageNet statistics the decoder output is denormalised with.
pub(crate) const PIXEL_MEAN: [f32; PIXEL_CHANNELS] = [0.485, 0.456, 0.406];
pub(crate) const PIXEL_STD: [f32; PIXEL_CHANNELS] = [0.229, 0.224, 0.225];

/// Audio VAE decoder (`minimax_h3_audio_vae_fp32.safetensors`): `dec_in_proj` lifts the 32 latent channels to `AUDIO_MEL_DIM`, a BigVGAN
/// vocoder upsamples seven stages (stride, kernel) to 32 kHz audio, each followed by three anti-aliased residual blocks.
pub(crate) const AUDIO_MEL_DIM: usize = 2048;
pub(crate) const AUDIO_BASE_CHANNELS: usize = 1024;
pub(crate) const AUDIO_UPSAMPLE: [(usize, usize); 7] =
    [(5, 9), (5, 9), (2, 4), (2, 4), (2, 4), (2, 4), (2, 4)];
pub(crate) const AUDIO_RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];
pub(crate) const AUDIO_RESBLOCK_DILATIONS: [usize; 3] = [1, 3, 5];
/// Taps of the Kaiser-windowed sinc filters of the x2 up/down sampling inside every activation.
pub(crate) const AUDIO_FILTER: usize = 12;
/// Waveform samples per latent frame (the product of the upsample strides); a test checks it against `AUDIO_UPSAMPLE`.
#[cfg(test)]
pub(crate) const AUDIO_SAMPLES_PER_LATENT: usize = 800;
pub(crate) const AUDIO_SAMPLE_RATE: usize = 32_000;

/// Frame rate of the generated clip, and the bounds a request may ask for. The lower bound is the shortest clip the model's frame grid
/// allows; the upper bound is a deployment limit (the model was trained on roughly 124-362 frames; longer clips are untested here), not a
/// property of the checkpoint, and keeps the packed sequence and the decoder's temporal plan inside the sizes that were measured. It is
/// the largest count on the `17 k + 5` grid that a request may be snapped to: a bound off the grid would let 363 to 365 frames snap to 379.
pub(crate) const FRAME_RATE: u32 = FPS as u32;
pub const MIN_FRAMES: usize = 5;
pub const MAX_FRAMES: usize = 362;
/// Canvas sides in pixels: at least one VAE tile (so the tile plan is the full-tile case the decoder implements) and at most this long.
pub const MIN_SIDE: usize = VV_TILE_PIXELS;
pub const MAX_SIDE: usize = 1920;
/// Steps a request may ask for. The sigma grid is the `simple` scheduler over the flow-shifted 1000-point table, so any positive count is
/// defined; 4 is the count the parity traces use.
pub const MAX_STEPS: usize = 50;

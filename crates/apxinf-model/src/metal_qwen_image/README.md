# Native Qwen-Image-2.1 on MLX

This Metal-specific module owns checkpoint interpretation, the text-to-image
schedule and per-request state for Qwen-Image-2.1. It implements
`ImageRuntime` with safe `apxinf-mlx` operations, without Python. The reference
is the mflux revision `f0bf9cf0506a2ae15115af1c72e3dc0093e41597` `QwenImage21`
txt2img path with `--low-ram`, on MLX 0.32.3.

Initial target: the mflux 4-bit export JoyFusionAI/Qwen-Image-2.1-MLX-4bit
revision `c78c0880dfa969d42a406e6abfadc2375bfa118d`: affine Q4/group-64 DiT
(32 blocks, width 4096), BF16 Qwen3-VL text encoder (36 layers) and F32 VAE
decoder. The weights are under the Qwen Research License, which permits
non-commercial use only; APXinf ships no weights.

The family admits batch one, 40 linear steps at guidance 1, and positive sides
that are multiples of 16 up to the RoPE table limit of 32768 pixels. Both
variants also require `(width / 16) * (height / 16) <= 174762` image tokens.
This is the MLP projection's signed-32-bit flattened-index limit:
`tokens * 12288 - 1 <= i32::MAX`. It also bounds the Q/K fusion grid and keeps
the existing FP32 scheduler's terminal rescale finite and nonzero. These are
execution bounds, not a memory-capacity or output-quality guarantee. Admission
checks the dimensions and checked token product before any request device work.
512 x 512, 1024 x 576 and the default 1024 x 1024 remain admitted; squares with
sides 8192, 16384 or 32768 are rejected. Empty prompts, other step counts,
negative prompts, true CFG, editing, ControlNet and LoRA are rejected or not exposed.

`model_variant=auto`, also the default when omitted, selects per MLX
architecture: on `applegpu_g16g` (M4) the fused image blocks below, elsewhere
the MPP selection described below. `model_variant=reference` skips both and
uses MLX quantized matmul for all Q4 projections. It changes neither
checkpoint interpretation nor the schedule, text encoder, compilation or VAE.
Select it from the CLI with `generate-image --model-variant reference`.

- `weights.rs` checks every DiT and VAE decoder key, dtype and shape at load.
  For the text encoder it checks every tensor header at load and reads one
  decoder layer at a time per request. It loads the stored
  `rotary_emb.inv_freq` as mflux does, and skips the deterministic DiT buffers
  that mflux recomputes.
  All shards are checked for overflow-safe byte extents and complete,
  nonoverlapping SafeTensors data offsets before payload loading. The optional
  mflux export tag `quantization_level` must be the string `4` when present;
  it describes the global export, including its BF16 text and F32 VAE shards,
  rather than overriding individual tensor dtypes. Missing tags remain valid
  when the physical checkpoint contracts match. Text validation reads headers
  and file sizes only, preserving layer-at-a-time payload loading.
- `text.rs` applies the fixed system template, drops the system-prefix states
  and follows Qwen3-VL's FP32 norm, BF16 RoPE and FP32 attention rounding.
- `dit.rs` computes the text-prefix K/V once per request with the t = 0
  modulation row. The pure image-token forward is explicitly prepared twice
  on the actual first-step inputs. Its second evaluated velocity supplies the
  first solver update; steps 1 through 39 call it once each. There are exactly
  40 original BF16 solver updates and one initial noise realization. Preparation
  belongs to the request because the low-RAM path releases the transformer.
  Neither the scheduler nor the rollout is compiled. `ops.rs` keeps MLX
  `nn.silu` and `nn.gelu_approx` as shapeless compiled functions, preparing their
  BF16 rank-2/3 and FP32 rank-4 singleton/nonsingleton classes before first use.
- Image-token block projections decode Q4 to BF16 per call and use the
  MetalPerformancePrimitives GEMM when the token count is a multiple of 64 and
  at least 1024. Loading enables it only
  on the MLX architecture where it was measured faster, `applegpu_g16g` (M4),
  and only when a probe shows that it compiles and matches MLX quantized matmul
  bit for bit for all three projection shapes at 1024 rows, on seeded device data.
  Below that size MLX quantized matmul switches to another reduction (observed
  on M4 at 64 rows), which MPP does not reproduce. The probe does not qualify
  checkpoint weights, larger token counts or whole-image equality; this
  optimization still needs device and workload qualification. In `reference`,
  on other architectures, when the probe fails, below 1024 or for unaligned image
  token counts, and for the text prefix, projections use quantized matmul.
- Fused image blocks (`auto` on `applegpu_g16g`, image-token counts that are
  multiples of 64 in 1024..=4096; other counts use the path above). Each image
  block runs seven kernels from `apxinf-mlx`: `LayerNormModulate`; Q and K
  through `MppGemmHeadRmsRope` (BF16-rounded GEMM, head RMSNorm and RoPE in the
  epilogue, K with the cached text keys copied in front); V through
  `MppGemmHeadsWithPrefix`; SDPA; the output projection through
  `MppGemmDecodeQ4Rows`, which also decodes the MLP gate/up weight;
  `GatedResidualLayerNormModulate`; `MppGemmSwigluPairs`; and the MLP down
  projection through `MppGemmGatedResidual`. The broadcast modulation rows are
  read through their strides. Each block's gate and up Q4 weights are
  interleaved in 32-row pairs when the DiT is loaded for a request, beside the
  originals that the text prefix reads (about 1.8 GB for 32 blocks). These
  kernels round differently from mflux: images are not bit-identical to
  `reference`, so qualification compares them with the reference
  implementation they reproduce and reports PSNR against mflux.
- `vae.rs` decodes in mflux's NCHW layout, tiled with 512 px tiles and 128 px
  overlap. FP32 tiles, weighted products and ordered overlap sums/counts remain
  on device. Each product is materialized before addition, and each replacement
  sum/count pair is evaluated before decoding another tile. Normalization,
  clipping and ties-to-even channel rounding also stay on device. Only final
  U32 channel values are downloaded and serialized to owned RGB8 bytes on host.
- `tables.rs` reproduces the NumPy float32 sin/cos used for the RoPE tables and
  the blending ramp; tests check recorded NumPy digests.

Lifecycle: the DiT is loaded with the model, released before each VAE decode
and read again by the next request; the VAE decoder stays loaded. As mflux
`--low-ram` does, loading caps the process-wide MLX allocator cache at 1 GB,
and the cache is cleared after text encoding and after the DiT is released.
This matches the reference's weight-release boundaries. Device accumulation and integer
pixel finalization add bounded request-local arrays; their measured peak and
complete-request cost require source-bound qualification, not a claim of
identical memory usage.

Verification status: CPU tests cover the tables, admission, RoPE positions,
text-encoder tensor names and checkpoint detection. Image output against the
reference, memory and latency are measured on Apple M4 and reported in the PR;
other devices are listed as supported only after the same checks pass there.

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

The family admits batch one, 40 linear steps at guidance 1, and sides that are
multiples of 16 up to the RoPE table limit. Empty prompts, other step counts,
negative prompts, true CFG, editing, ControlNet and LoRA are rejected or not
exposed.

- `weights.rs` checks every DiT and VAE decoder key, dtype and shape at load.
  For the text encoder it checks every tensor header at load and reads one
  decoder layer at a time per request. It loads the stored
  `rotary_emb.inv_freq` as mflux does, and skips the deterministic DiT buffers
  that mflux recomputes.
- `text.rs` applies the fixed system template, drops the system-prefix states
  and follows Qwen3-VL's FP32 norm, BF16 RoPE and FP32 attention rounding.
- `dit.rs` computes the text-prefix K/V once per request with the t = 0
  modulation row, then runs one compiled image-token forward per step against
  that cache, as mflux does. `ops.rs` keeps MLX `nn.silu` and
  `nn.gelu_approx` as shapeless compiled functions.
- Image-token block projections decode Q4 to BF16 per call and use the
  MetalPerformancePrimitives GEMM when the token count is a multiple of 64
  (EngineTailor GEMM frontier 427d48fc). Loading enables it only when a probe
  shows that it compiles and reproduces MLX quantized matmul bit for bit, so
  the image does not depend on the path; otherwise, and for the text prefix,
  projections use quantized matmul.
- `vae.rs` decodes in mflux's NCHW layout, tiled with 512 px tiles and 128 px
  overlap, and blends tiles on the host.
- `tables.rs` reproduces the NumPy float32 sin/cos used for the RoPE tables and
  the blending ramp; tests check recorded NumPy digests.

Lifecycle: the DiT is loaded with the model, released before each VAE decode
and read again by the next request; the VAE decoder stays loaded. This matches
the memory profile of the `--low-ram` reference on 16 GB devices.

Verification status: CPU tests cover the tables, admission, RoPE positions,
text-encoder tensor names and checkpoint detection. Image output against the
reference, memory and latency are measured on Apple M4 and reported in the PR;
other devices are listed as supported only after the same checks pass there.

# `metal_inkling2` — Native Inkling-2-Mini on Apple Metal

Native Rust/MLX implementation of `thinkingmachines/Inkling-2-Mini` (`inkling2_loop`),
following [Adding a New Model](../../../../doc/adding-a-new-model.md) and
[MLX Backend](../../../../doc/mlx-backend.md).

## Architecture Summary (`inkling2_loop_text`)
- Hidden size `2304`, `24` physical layers expanded to `48` layer visits via
  `loop_config` (`core_start: 6`, `core_end: 18`, `num_passes: 3`, `gate_rank: 64`).
- Attention: GQA (`24` Q heads, `8` KV heads, `head_dim: 128`) with sliding-window
  local layers (`sliding_window_size: 1024`, `sconv_kernel_size: 4`), relative
  attention (`d_rel: 16`, `rel_extent: 1024`), QK/rel L2 clipping, and log-scaling.
- MoE: `128` routed experts, `top-8`, `intermediate_size: 1152`, `route_scale: 8.0`,
  `mlp_gate_softcap: 4.0`, `mlp_up_softcap: 25.0`.
- Attention residuals: `n_layers_per_attn_res_block: 6` with `AttnResQuery`.
- Quantization: `modelopt` NVFP4 / MXFP8 / affine Q4/Q8 mixed precision + 3-layer MTP draft head.

## Runtime Switches
- `APXINF_INKLING2_MTP=0|1` (default `1`): enable or disable the 3-layer MTP speculative draft/verify loop on greedy decode.
- `APXINF_INKLING2_MTP_DEPTH=<n>`: override the MTP draft depth.
- `APXINF_INKLING2_PREFIX_CACHE=1` (default `0`): opt-in cross-request KV prefix reuse for multi-turn / agentic workloads.
- `APXINF_INKLING2_DECODE_TRACE=1` (default `0`): log per-request decode path and MTP verifier statistics to `stderr`.

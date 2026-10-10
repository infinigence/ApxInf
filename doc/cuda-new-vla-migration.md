# Migrating the remaining families to `apxinf-cuda-new`

Status after the LLM/VLM commit. This file scopes the work left for the VLA
families; it is a work item, not a design document.

## What is done

`CudaNewBackend` (`crates/apxinf-cuda-new/src/backend.rs`) implements the
portable `apxinf_core::Backend` trait on the cuda-new runtime, and
`accelerator::create_cuda_new_backend` hands it to any model that composes
through `dyn Backend`. `auto.rs` routes `llama` / `qwen3_vl` / `qwen3vl` there,
and `qwen38` samples through the same backend. Those families need no model-side
change: they never named a concrete backend.

Verified: `cargo check -p apxinf-cuda-new` and `-p apxinf-model --features cuda`
pass; the model test suite is 185 passed / 0 failed.

## Why the VLA families are a different shape of work

walloss, pi0fast and qwen_drive do **not** go through `dyn Backend`. They reach
the fused kernel surface directly through their `backend.rs` alias:

```rust
pub(crate) use crate::accelerator::cuda::kernels;
```

so migrating them means either (a) giving each family a cuda-new seam with the
same names, or (b) rewriting each call site to the cuda-new operator API. The
call surface is ~60 distinct functions across the three families.

### Function inventory the three families call (usage count in parentheses)

Already present in cuda-new under a different name — a mapping layer suffices:

| legacy | cuda-new |
|---|---|
| `norm::rms_bf16` (11) | `ops::mlp::rms_norm` |
| `norm::layer_bf16` (6) | `ops::layer_norm` |
| `fused::bias_residual_bf16` (11) | `ops::bias_residual` |
| `fused::bias_residual_rms_bf16` (7) | `ops::bias_residual_rms_norm` |
| `fused::bias_residual_layer_bf16` (4) | `ops::bias_residual_layer_norm` |
| `embedding::lookup` (6) | `ops::embedding_gather` |
| `elementwise::scale` (4) | `ops::elementwise_scale` |
| `elementwise::add` (6) | `ops::elementwise_add` |
| `elementwise::bias_bf16` (6) | `ops::elementwise_add_bias` |
| `elementwise::concat_rows_bf16` (2) | `ops::concat_rows` |
| `cache::reserve_prefix_bf16` (4) | `ops::reserve_prefix` |
| `activation::silu` (3) | `ops::elementwise_activation(Silu)` |
| `activation::gelu_tanh` (1) | `ops::elementwise_activation(GeluTanh)` |
| `activation::swiglu_bf16` (2) | `ops::mlp::swiglu` |
| `sampling::argmax_bf16_remapped_into` (2) | `ops::argmax` |
| `gemm::bf16` (38) / `gemm::matmul` (1) | `ops::gemm` |

Missing from cuda-new — must be added as native operators:

- `activation::geglu_bf16` (4), `activation::bias_gelu_bf16` (3),
  `activation::swiglu_quantize_rows_bf16_e4m3` (1)
- `attention::mqa_bf16` (5), `mha_bf16` (2), `causal_gqa_bf16` (3),
  `split_qkv_bias_bf16` (2), `split_vision_qkv_rope_bf16` (2),
  `split_gqa_qkv_mrope_cache_bf16` (2), `segmented_mha_bf16` (2),
  the `try_*_fa4_*` candidates
- `elementwise::gather_rows_bf16` (4), `replace_rows_bf16` (3),
  `euler_update_bf16` (1)
- `embedding::add_position_f32_bf16` (1), `sinusoidal_bf16` (1)
- `rope::apply_q_write_kv_bf16` (3), `split_qkv_apply_bf16` (2)
- `gemm::bf16_bias` (10), `bf16_addmv` (2), `bf16_geglu_fused` (3),
  `fp8_bf16` (5), `write_ex` (3), the checkpoint variants
- `quantization::quantize_bf16_e4m3` (1), `quantize_rows_bf16_e4m3*` (2),
  `slice_columns_bf16` (1)
- `fused::adaln_gate_residual_rms_bf16` (2),
  `bias_residual_rms_quantize_rows_bf16_e4m3` (1)
- `preprocess::*` (4 call sites) — the patchification path, absent in cuda-new

qwen_drive additionally uses `linear_attention::*`, `fixed_profile::*` and
`pillow_bicubic::*`, none of which exist in cuda-new.

## Recommended order

1. pi0fast — smallest VLA surface (27 functions, most already mapped). Pilot
   the shape of the seam and the differential test.
2. walloss — 26 functions, adds `preprocess`.
3. qwen_drive — largest, needs GDN re-wiring plus three novel kernel families.

Each step needs a numerical oracle. With no external reference outputs, the
legacy backend is the oracle: run the same weights and inputs on both runtimes
and assert equality. That harness does not exist yet and should be built before
the second family is attempted.

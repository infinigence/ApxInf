# Environment variable reference

Generated from the source tree (`grep -rE "APXINF_"`). Names are
classified by where they are consumed; a variable absent from this
list is not a supported knob — grep the source for it instead.

## Agent infrastructure

| Name | Where read |
|---|---|
| `APXINF_AGENT_ACCEPTANCE_PLAN` | scripts/hooks/guard_agent_stop.py; opt-in absolute path to a task acceptance JSON under devlocal; see [agent acceptance](../guides/agent-acceptance.md) |

## Build-time (read in build.rs / build_support)

| Name | Where read |
|---|---|
| `APXINF_PLATFORM` | justfile; selects a preset from configs/platforms/ |
| `APXINF_CUDA_AOT_MANIFEST` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_nvfp4_swiglu_aot_sm100.cu, cra |
| `APXINF_CUDA_ARCH` | crates/apxinf-cuda-new/native/adapters/runtime.cpp, crates/apxinf-cuda-new/build.rs, crate |
| `APXINF_CUDA_ARCH_CUTLASS` | crates/apxinf-cuda-new/build.rs, crates/apxinf-cuda-new/build_support/cuda_arch.rs, crates |
| `APXINF_CUDA_DEVICE` | crates/apxinf-cuda-new/build_support/cuda_arch.rs, crates/apxinf-cuda/build_support/cuda_a |
| `APXINF_FA2_HEAD64_MAXREG` | crates/apxinf-cuda/build.rs |
| `APXINF_KERNEL_BUILD_ID` | crates/apxinf-cuda-new/build.rs, crates/apxinf-cuda/build.rs, crates/apxinf-cuda/src/tunin |
| `APXINF_NVCC_PARALLELISM` | crates/apxinf-cuda/build_support/nvcc_build.rs |
| `APXINF_STATUS_UNSUPPORTED` | crates/apxinf-cuda-new/native/framework/autotune.h, crates/apxinf-cuda-new/native/include/ |

## User-facing (documented, checkpoint/test input)

| Name | Where read |
|---|---|
| `APXINF_CUDA_ALLOC_CACHE` | crates/apxinf-cuda/src/buffer.rs |
| `APXINF_DISABLE_FA2_SPLITKV` | crates/apxinf-cuda/src/kernels/attention.rs |
| `APXINF_FAST_TOKENIZER` | python/apxinf/apxinf/policies/impls/pi0fast.py |
| `APXINF_PALIGEMMA_TOKENIZER` | python/apxinf/apxinf/policies/impls/pi0fast.py |
| `APXINF_PI05_CALIBRATION` | crates/apxinf-py/tests/conftest.py |
| `APXINF_PI05_CHECKPOINT` | crates/apxinf-py/tests/conftest.py |
| `APXINF_PI05_DEVICE` | crates/apxinf-py/tests/conftest.py, python/apxinf/tests/test_policy_layering.py |
| `APXINF_PI05_MODEL_DIR` | python/apxinf/tests/conftest.py, python/apxinf/tests/test_policy_layering.py |
| `APXINF_PI05_MODEL_VARIANT` | crates/apxinf-py/tests/conftest.py, python/apxinf/tests/test_policy_layering.py |
| `APXINF_PI05_TACTICS` | crates/apxinf-py/tests/conftest.py |
| `APXINF_PI05_TEST_CHECKPOINT` | crates/apxinf-model/src/pi05/model_runner/runner.rs |
| `APXINF_PI0FAST_FP8_ACTIVATION_SCALE` | crates/apxinf-model/src/pi0fast/vla_runtime.rs |
| `APXINF_QWEN38_CHECKPOINT` | crates/apxinf-model/src/qwen38/model_runner.rs, crates/apxinf-cuda-new/tests/qwen38_mlp_la |
| `APXINF_QWEN38_GEMM_TUNE_CACHE` | crates/apxinf-model/src/qwen38/model.rs, crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_TUNE_CACHE` | crates/apxinf-model/src/qwen38/model.rs, crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN3VL_MODEL_DIR` | crates/apxinf-model/examples/vision_check.rs, crates/apxinf-model/examples/multimodal_chec |
| `APXINF_QWEN_DRIVE_TEST_INPUTS` | crates/apxinf-model/src/qwen_drive/model_runner/prepare.rs |
| `APXINF_QWEN_DRIVE_TEST_MODEL` | crates/apxinf-model/tests/qwen_drive_loading.rs, crates/apxinf-model/src/qwen_drive/model_ |
| `APXINF_SMOLVLA_NO_GRAPH` | crates/apxinf-model/src/smolvla/runtime.rs |
| `APXINF_TINYLLAMA_MODEL_DIR` | scripts/hf_reference_dump.py |
| `APXINF_TOKENIZER` | scripts/pi05_openpi_websocket_server.py, python/apxinf/tests/conftest.py, python/apxinf/te |
| `CUDA_HOME` | crates/apxinf-cuda-new/build.rs, crates/apxinf-cuda/build.rs |
| `CUDA_PATH` | crates/apxinf-cuda-new/build.rs, crates/apxinf-cuda/build.rs |
| `CUDA_VISIBLE_DEVICES` | crates/apxinf-cuda-new/build.rs, crates/apxinf-cuda/build.rs |
| `HF_HOME` | python/apxinf/tests/test_pi0fast_policy.py, python/apxinf/apxinf/policies/impls/pi0fast.py |
| `NVIDIA_VISIBLE_DEVICES` | crates/apxinf-cuda/build.rs |

## Debug / tuning (scattered per-family overrides)

| Name | Where read |
|---|---|
| `APXINF_ATTENTION_BUILD_ID` | crates/apxinf-cuda-new/native/adapters/attention/tuning_key.cpp, crates/apxinf-cuda-new/bu |
| `APXINF_ATTENTION_CUTLASS` | crates/apxinf-cuda-new/native/adapters/attention/internal.h, crates/apxinf-cuda-new/native |
| `APXINF_ATTENTION_FA2` | crates/apxinf-cuda-new/native/adapters/attention/internal.h, crates/apxinf-cuda-new/native |
| `APXINF_ATTENTION_FA2_E4M3` | crates/apxinf-cuda-new/native/adapters/attention/providers/fa2.cpp, crates/apxinf-cuda-new |
| `APXINF_ATTENTION_MASK_CAUSAL` | crates/apxinf-cuda-new/native/include/apxinf_cuda/attention_types.h, crates/apxinf-cuda-ne |
| `APXINF_ATTENTION_MASK_NONE` | crates/apxinf-cuda-new/native/kernels/attention/fa2_fwd_split_hdim256_f16.cu, crates/apxin |
| `APXINF_ATTENTION_SEMANTIC_DENSE` | crates/apxinf-cuda-new/native/include/apxinf_cuda/attention_types.h, crates/apxinf-cuda-ne |
| `APXINF_ATTENTION_SEMANTIC_KV_CACHE` | crates/apxinf-cuda-new/native/include/apxinf_cuda/attention_types.h, crates/apxinf-cuda-ne |
| `APXINF_ATTENTION_SEMANTIC_PACKED_QKV` | crates/apxinf-cuda-new/native/include/apxinf_cuda/attention_types.h, crates/apxinf-cuda-ne |
| `APXINF_ATTENTION_SEMANTIC_SEGMENTED` | crates/apxinf-cuda-new/native/include/apxinf_cuda/attention_types.h, crates/apxinf-cuda-ne |
| `APXINF_ATTN_COMPOSED_DECODE` | crates/apxinf-cuda/src/kernels/attention.rs |
| `APXINF_ATTN_COMPOSED_PREFILL` | crates/apxinf-cuda/src/kernels/attention.rs |
| `APXINF_BF16_DUAL_GEGLU_PRODUCTION` | crates/apxinf-cuda-new/build.rs, crates/apxinf-cuda/build.rs |
| `APXINF_CUDA_ALLOC_CACHE_MB` | crates/apxinf-cuda/src/buffer.rs |
| `APXINF_CUDA_POISON_ALLOC` | crates/apxinf-cuda/src/buffer.rs |
| `APXINF_DEFAULT_RECIPE_CHILD` | crates/apxinf-cuda-new/src/ops/tests/backend_framework.rs |
| `APXINF_DEVICE_INFO_VERSION` | crates/apxinf-cuda-new/native/include/apxinf_cuda/runtime.h, crates/apxinf-cuda-new/native |
| `APXINF_DTYPE_BF16` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_DTYPE_E2M1_PAIR` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_DTYPE_E4M3` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_DTYPE_F16` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_DTYPE_F32` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_DTYPE_I32` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_DTYPE_I8` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_DUAL_GEGLU_CLUSTER_M` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_CLUSTER_N` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_DOUBLE_BUFFER` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_EPI_M` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_EPI_N` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_FAST_TANH` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_GEMM_ALPHA` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_MAINLOOP_SCHEDULE` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_MONOLITHIC` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_OUTPUT_SCALE` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_PAIRED_EPILOGUE_IMPLEMENTED` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_REGISTER_RENDEZVOUS` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_ROUND_UP_HALF` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_SATURATE_TANH` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_STAGE_COUNT` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_STREAMED_EVT` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_TILE_K` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_TILE_M` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_TILE_N` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_TRACE_OWNERS` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_WIDE_PAIR_IMPLEMENTED` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_WIDE_STAGE_COUNT` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_FA2_DECODE_SPLITS` | crates/apxinf-model/src/qwen38/model_runner.rs, crates/apxinf-cuda-new/native/kernels/fa2/ |
| `APXINF_FA2_DIRECT_E4M3` | crates/apxinf-cuda-new/build.rs, crates/apxinf-cuda/kernels/cutlass/fa2/flash_attn/flash_f |
| `APXINF_FA2_HEAD_SPECIAL` | crates/apxinf-cuda/build.rs, crates/apxinf-cuda/adapters/fa2_adapter.cu |
| `APXINF_FA2_SM80` | crates/apxinf-cuda/build.rs |
| `APXINF_FA2_SPLITKV` | crates/apxinf-cuda/kernels/cutlass/fa2_bf16_sm80.cu, crates/apxinf-cuda/build.rs, crates/a |
| `APXINF_FA4_SPLIT_BATCH_UNSUPPORTED_TOPOLOGY` | crates/apxinf-cuda/adapters/fa4_d256_split_batch_adapter.cu |
| `APXINF_FIXED_SCENE_TOKENS` | crates/apxinf-cuda/tests/aot_bundle.rs, crates/apxinf-cuda/adapters/fixed_profile.h, crate |
| `APXINF_FMHA_UNDEF_SM100A` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/attention/fmha_sm100.cu, crates/apxinf-c |
| `APXINF_FMHA_UNDEF_SM100F` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/attention/fmha_sm100.cu, crates/apxinf-c |
| `APXINF_FP8_DUAL_GEGLU_PRODUCTION` | crates/apxinf-cuda-new/build.rs, crates/apxinf-cuda/build.rs |
| `APXINF_FP8_GEMV_SHARED` | crates/apxinf-model/src/qwen38/model_runner.rs, crates/apxinf-cuda-new/native/kernels/cust |
| `APXINF_FP8_NATIVE_PAIR` | crates/apxinf-model/src/qwen38/model_runner.rs, crates/apxinf-cuda-new/native/kernels/cust |
| `APXINF_FP8_NN_LAYOUT` | crates/apxinf-cuda/build.rs, crates/apxinf-cuda/adapters/cublaslt_adapter.cu |
| `APXINF_GATHER_SEMANTIC_BIAS_POSITION` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gather_types.h, crates/apxinf-cuda-new/n |
| `APXINF_GATHER_SEMANTIC_EMBEDDING_LOOKUP` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gather_types.h, crates/apxinf-cuda-new/n |
| `APXINF_GATHER_SEMANTIC_RGB_TO_PATCHES` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gather_types.h, crates/apxinf-cuda-new/n |
| `APXINF_GATHER_SPEC_VERSION` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gather_types.h, crates/apxinf-cuda-new/n |
| `APXINF_GDN_CONV_PAIR` | crates/apxinf-model/src/qwen38/model_runner.rs, crates/apxinf-cuda-new/native/kernels/cust |
| `APXINF_GDN_FUSED_CONV_PREPARE` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_GDN_FUSED_WIDEN_NORM` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_GDN_LAUNCH` | crates/apxinf-cuda-new/native/kernels/custom/gdn_ops.cu |
| `APXINF_GDN_LOG_GATE` | crates/apxinf-cuda-new/native/kernels/flashinfer_gdn/flashinfer_gdn.h, crates/apxinf-cuda- |
| `APXINF_GDN_PREFILL_TILE` | crates/apxinf-model/src/qwen38/model_runner.rs, crates/apxinf-cuda-new/native/kernels/cust |
| `APXINF_GDN_PREFILL_TILED` | crates/apxinf-model/src/qwen38/model_runner.rs, crates/apxinf-cuda-new/native/kernels/cust |
| `APXINF_GDN_PREPARE_PARALLEL` | crates/apxinf-model/src/qwen38/model_runner.rs, crates/apxinf-cuda-new/native/kernels/cust |
| `APXINF_GDN_RECURRENT_SPLIT` | crates/apxinf-cuda/adapters/custom_kernels.cu |
| `APXINF_GDN_SMEM_VARIANT` | crates/apxinf-cuda-new/native/kernels/custom/gdn_ops.cu |
| `APXINF_GDN_WMMA_LOSSY` | crates/apxinf-cuda/kernels/custom/gdn_policy.h |
| `APXINF_GDN_WMMA_OFF` | crates/apxinf-cuda/kernels/custom/gdn_policy.h, crates/apxinf-cuda/adapters/custom_kernels |
| `APXINF_GDN_WMMA_UNSUPPORTED` | crates/apxinf-cuda/build.rs, crates/apxinf-cuda/adapters/custom_kernels.cu |
| `APXINF_GEMM_BUILD_ID` | crates/apxinf-cuda-new/native/adapters/gemm/tuning_key.cpp, crates/apxinf-cuda-new/build.r |
| `APXINF_GEMM_CUBLASLT_NVFP4` | crates/apxinf-model/src/qwen38/model_runner.rs, crates/apxinf-cuda-new/native/adapters/gem |
| `APXINF_GEMM_CUTLASS` | crates/apxinf-cuda-new/native/adapters/gemm/providers/cutlass.cu, crates/apxinf-cuda-new/n |
| `APXINF_GEMM_CUTLASS_SM87_W8A8` | crates/apxinf-cuda-new/native/adapters/gemm/providers/cutlass.cu, crates/apxinf-cuda-new/n |
| `APXINF_GEMM_CUTLASS_SM89` | crates/apxinf-cuda-new/native/adapters/gemm/providers/cutlass.cu, crates/apxinf-cuda-new/n |
| `APXINF_GEMM_DUMP_SHAPES` | crates/apxinf-cuda/src/kernels/gemm/bf16.rs |
| `APXINF_GEMM_FP8_F32_PROJECTION` | crates/apxinf-cuda-new/native/adapters/gemm/providers/custom.cu |
| `APXINF_GEMM_FP8_NO_CUTLASS_BF16` | crates/apxinf-cuda-new/native/adapters/gemm/candidates.cpp |
| `APXINF_GEMM_LAYOUT_GATE_UP_INTERLEAVED_256` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h |
| `APXINF_GEMM_LAYOUT_KN` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_GEMM_LAYOUT_NK` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h |
| `APXINF_GEMM_QUANT_FP8_ROW_CHANNEL` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_GEMM_QUANT_FP8_UNIT_SCALE` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_GEMM_QUANT_NONE` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_GEMM_QUANT_NVFP4_BLOCK` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_GEMM_QUANT_W8A8_ROW_CHANNEL` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_GEMM_REPORT` | crates/apxinf-cuda-new/native/adapters/gemm/execution.cpp, crates/apxinf-cuda-new/tests/fp |
| `APXINF_GEMM_SEMANTIC_GEMM` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_GEMM_SEMANTIC_GEMM_BIAS` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_GEMM_SEMANTIC_GEMM_BIAS_RESIDUAL` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_GEMM_SEMANTIC_GEMM_GEGLU` | crates/apxinf-cuda-new/native/include/apxinf_cuda/gemm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_MHA_NAIVE` | crates/apxinf-cuda/src/kernels/attention.rs |
| `APXINF_NORM_SEMANTIC_ADAPTIVE_RMS` | crates/apxinf-cuda-new/native/include/apxinf_cuda/norm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL` | crates/apxinf-cuda-new/native/include/apxinf_cuda/norm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS` | crates/apxinf-cuda-new/native/include/apxinf_cuda/norm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_NORM_SEMANTIC_BIAS_RESIDUAL` | crates/apxinf-cuda-new/native/include/apxinf_cuda/norm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER` | crates/apxinf-cuda-new/native/include/apxinf_cuda/norm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS` | crates/apxinf-cuda-new/native/include/apxinf_cuda/norm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL` | crates/apxinf-cuda-new/native/include/apxinf_cuda/norm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_NORM_SEMANTIC_LAYER` | crates/apxinf-cuda-new/native/include/apxinf_cuda/norm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_NORM_SEMANTIC_RMS` | crates/apxinf-cuda-new/native/include/apxinf_cuda/norm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_NORM_SPEC_VERSION` | crates/apxinf-cuda-new/native/include/apxinf_cuda/norm_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_NO_GRAPH` | crates/apxinf-model/src/llama/decode_graph.rs |
| `APXINF_NVFP4_VECTOR_QUANT` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_nvfp4_sm100.cu |
| `APXINF_PI05_ACTION_ADA_PACKED8` | crates/apxinf-cuda/adapters/custom_kernels.cu |
| `APXINF_PI05_CONSISTENCY_ATOL` | crates/apxinf-py/tests/test_l0_l1_consistency.py |
| `APXINF_PI05_EAGER_ONLY` | crates/apxinf-model/examples/pi05_bench.rs |
| `APXINF_PI05_IMAGE_INPUT` | crates/apxinf-model/examples/pi05_bench.rs |
| `APXINF_PI05_PROFILE_REPLAY` | crates/apxinf-model/examples/pi05_bench.rs |
| `APXINF_PKG` | scripts/validate_pi05_calibration.py, scripts/calibrate_pi05.py, scripts/bench_pi0_fast.py |
| `APXINF_POINTWISE_ACTIVATION_GELU` | crates/apxinf-cuda-new/native/include/apxinf_cuda/pointwise_types.h |
| `APXINF_POINTWISE_ACTIVATION_NONE` | crates/apxinf-cuda-new/native/include/apxinf_cuda/pointwise_types.h, crates/apxinf-cuda-ne |
| `APXINF_POINTWISE_ACTIVATION_SILU` | crates/apxinf-cuda-new/native/include/apxinf_cuda/pointwise_types.h, crates/apxinf-cuda-ne |
| `APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION` | crates/apxinf-cuda-new/native/include/apxinf_cuda/pointwise_types.h, crates/apxinf-cuda-ne |
| `APXINF_POINTWISE_SEMANTIC_EULER_UPDATE` | crates/apxinf-cuda-new/native/include/apxinf_cuda/pointwise_types.h, crates/apxinf-cuda-ne |
| `APXINF_POINTWISE_SEMANTIC_GEGLU` | crates/apxinf-cuda-new/native/include/apxinf_cuda/pointwise_types.h, crates/apxinf-cuda-ne |
| `APXINF_POINTWISE_SPEC_VERSION` | crates/apxinf-cuda-new/native/include/apxinf_cuda/pointwise_types.h, crates/apxinf-cuda-ne |
| `APXINF_QUANTIZATION_SEMANTIC_CAST_F16_BF16` | crates/apxinf-cuda-new/native/include/apxinf_cuda/quantization_types.h, crates/apxinf-cuda |
| `APXINF_QUANTIZATION_SEMANTIC_FIXED_E4M3` | crates/apxinf-cuda-new/native/include/apxinf_cuda/quantization_types.h, crates/apxinf-cuda |
| `APXINF_QUANTIZATION_SEMANTIC_ROWWISE_E4M3` | crates/apxinf-cuda-new/native/include/apxinf_cuda/quantization_types.h, crates/apxinf-cuda |
| `APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8` | crates/apxinf-cuda-new/native/include/apxinf_cuda/quantization_types.h, crates/apxinf-cuda |
| `APXINF_QUANTIZATION_SEMANTIC_SLICE_BF16` | crates/apxinf-cuda-new/native/include/apxinf_cuda/quantization_types.h, crates/apxinf-cuda |
| `APXINF_QUANTIZATION_SPEC_VERSION` | crates/apxinf-cuda-new/native/include/apxinf_cuda/quantization_types.h, crates/apxinf-cuda |
| `APXINF_QWEN38_ALLOW_NO_REFERENCE` | crates/apxinf-cuda-new/tests/qwen38_gdn_prefill.rs |
| `APXINF_QWEN38_ATTENTION_MLP_GRAPHS` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_DENSE_SWIGLU_AOT` | crates/apxinf-cuda-new/native/adapters/gemm/execution.cpp, crates/apxinf-cuda-new/build.rs |
| `APXINF_QWEN38_DIRECT_FA2_DECODE` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_FIXED_DECODE` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_FLASHINFER_GDN` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_FORCED_DECODE_TOKENS` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_FP16_PROBE` | crates/apxinf-model/src/qwen38/model.rs, crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_FUSED_FC1_CLUSTERS` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_nvfp4_swiglu_aot_sm100.cu |
| `APXINF_QWEN38_GDN_AB_MERGED` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_GDN_GRAPHS` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_GDN_NORM_QUANTIZED` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_GDN_STAGES` | crates/apxinf-model/src/qwen38/model.rs, crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_GRAPH_REPORT` | crates/apxinf-model/src/qwen38/model_runner.rs |
| `APXINF_QWEN38_LAYER_TIMING` | crates/apxinf-model/src/qwen38/model.rs, crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_LOGITS_DUMP` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_MAX_NEW` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_MLP_GRAPHS` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_NVFP4_DP4A` | crates/apxinf-cuda-new/native/kernels/custom/mlp_ops.cu |
| `APXINF_QWEN38_PER_TOKEN_PREFILL` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_PREFILL_SESSION` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_PROFILE_DECODE` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_PROFILE_PREFILL` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_PROMPT` | crates/apxinf-model/examples/qwen38_bench.rs, crates/apxinf-cuda-new/tests/qwen38_end_to_e |
| `APXINF_QWEN38_PROMPT_LEN` | crates/apxinf-model/examples/qwen38_bench.rs, crates/apxinf-cuda-new/tests/qwen38_end_to_e |
| `APXINF_QWEN38_REUSE_FP8` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_VECTOR_ELEMENTWISE` | crates/apxinf-model/src/qwen38/model_runner.rs, crates/apxinf-cuda-new/native/kernels/cutl |
| `APXINF_QWEN38_VERIFY_CAKE` | crates/apxinf-model/src/qwen38/model.rs, crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_VERIFY_DECODE_GRAPHS` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_VERIFY_FINITE` | crates/apxinf-model/src/qwen38/model.rs, crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_VERIFY_GDN_GRAPHS` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN38_VERIFY_PREFILL_SESSION` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_QWEN_PREPROC_THREADS` | python/apxinf/apxinf/policies/impls/qwen_drive.py |
| `APXINF_RECIPE_RESTORE_CHILD` | crates/apxinf-cuda-new/src/ops/tests/backend_framework.rs |
| `APXINF_REF_SEQ` | crates/apxinf-cuda-new/tests/qwen38_layer_compare.rs, crates/apxinf-cuda-new/tests/qwen38_ |
| `APXINF_ROPE_SEMANTIC_DECODE_QKV_CACHE` | crates/apxinf-cuda-new/native/include/apxinf_cuda/rope_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_ROPE_SEMANTIC_SPLIT_QKV_BIAS` | crates/apxinf-cuda-new/native/include/apxinf_cuda/rope_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_ROPE_SEMANTIC_SPLIT_QKV_ROPE` | crates/apxinf-cuda-new/native/include/apxinf_cuda/rope_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_ROPE_SPEC_VERSION` | crates/apxinf-cuda-new/native/include/apxinf_cuda/rope_types.h, crates/apxinf-cuda-new/nat |
| `APXINF_ROWWISE_CASE` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_e4m3_f16_sm100.cu, crates/apxi |
| `APXINF_ROWWISE_SCHEDULED_CASE` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_e4m3_f16_sm100.cu, crates/apxi |
| `APXINF_STATUS_CACHE_MISS` | crates/apxinf-cuda-new/native/include/apxinf_cuda/status.h, crates/apxinf-cuda-new/native/ |
| `APXINF_STATUS_CUDA_ERROR` | crates/apxinf-cuda-new/native/framework/runtime_internal.h, crates/apxinf-cuda-new/native/ |
| `APXINF_STATUS_INTERNAL_ERROR` | crates/apxinf-cuda-new/native/framework/runtime_internal.h, crates/apxinf-cuda-new/native/ |
| `APXINF_STATUS_INVALID_ARGUMENT` | crates/apxinf-cuda-new/native/include/apxinf_cuda/status.h, crates/apxinf-cuda-new/native/ |
| `APXINF_STATUS_OK` | crates/apxinf-cuda-new/native/framework/runtime_internal.h, crates/apxinf-cuda-new/native/ |
| `APXINF_STATUS_PROVIDER_ERROR` | crates/apxinf-cuda-new/native/framework/runtime_internal.h, crates/apxinf-cuda-new/native/ |
| `APXINF_TEST_BF16_TACTICS` | crates/apxinf-cuda/src/tests/bf16.rs |
| `APXINF_TF_FUSED` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_TF_HIDDEN` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_TF_OUT` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_TF_TOKENS` | crates/apxinf-cuda-new/tests/qwen38_end_to_end.rs |
| `APXINF_VISION_FA2` | crates/apxinf-cuda/src/kernels/attention.rs, crates/apxinf-cuda/src/tests/operators.rs |
| `APXINF_VISION_SCORES_FP32` | crates/apxinf-cuda/src/kernels/attention.rs |
| `APXINF_VISION_V3_WARPS` | crates/apxinf-cuda/kernels/custom/attention.cuh, crates/apxinf-cuda/adapters/core_kernels_ |
| `APXINF_WALLOSS_NO_GRAPH` | crates/apxinf-model/src/walloss/bf16_runtime.rs |

## Legacy (still read, scheduled for removal)

| Name | Where read |
|---|---|
| `APXINF_DUAL_GEGLU_LEGACY_EPI_M` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_DUAL_GEGLU_LEGACY_EPI_N` | crates/apxinf-cuda-new/native/kernels/cutlass/ops/gemm/gemm_bf16_geglu_interleaved_sm100.c |
| `APXINF_GR00T_BF16_LEGACY_BIAS_RESIDUAL_ADAPTIVE_LN` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_BF16_LEGACY_BIAS_RESIDUAL_LN` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_BF16_LEGACY_PACKED8_BIAS_ACTIVATION` | crates/apxinf-model/src/gr00t/backend.rs |
| `APXINF_GR00T_BF16_LEGACY_QK_MROPE_THREADS` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_BF16_LEGACY_QK_RMS_MROPE` | crates/apxinf-model/src/gr00t/bf16_executor.rs |
| `APXINF_GR00T_BF16_LEGACY_VISION_MLP_RESIDUAL_NORM` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_BF16_LEGACY_VISION_POINTWISE_PACK` | crates/apxinf-model/src/gr00t/bf16_executor.rs |
| `APXINF_GR00T_BF16_LEGACY_VISION_RESIDUAL_LN` | crates/apxinf-model/src/gr00t/backbone/vision.rs |
| `APXINF_GR00T_BF16_LEGACY_VISION_ROPE` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_BF16_LEGACY_VISION_SEGMENT_OUTPUT` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_FP8_LEGACY_DIT_RESIDUAL_LN` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_FP8_LEGACY_M41_FC1_EPILOGUE` | crates/apxinf-model/src/gr00t/fp8_weights.rs |
| `APXINF_GR00T_FP8_LEGACY_M41_FC2_EPILOGUE` | crates/apxinf-model/src/gr00t/fp8_weights.rs |
| `APXINF_GR00T_FP8_LEGACY_M41_FFN_DOWN` | crates/apxinf-model/src/gr00t/fp8_weights.rs |
| `APXINF_GR00T_FP8_LEGACY_QK_RMS_MROPE` | crates/apxinf-model/src/gr00t/fp8_executor.rs |
| `APXINF_GR00T_FP8_LEGACY_STATIC_QUANT` | crates/apxinf-model/src/gr00t/fp8_weights.rs |
| `APXINF_GR00T_FP8_LEGACY_VISION_CONCAT_QUANT` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_FP8_LEGACY_VISION_RESIDUAL_NORM` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_LEGACY_FA2_BM64` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_LEGACY_QK_MROPE_THREADS` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_LEGACY_VISION_BATCHED_ATTENTION` | crates/apxinf-model/src/gr00t/backbone/vision.rs |
| `APXINF_GR00T_LEGACY_VISION_FC2_RESIDUAL` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_LEGACY_VISION_ROPE_POSITIONS` | crates/apxinf-model/src/gr00t/backbone/vision.rs |
| `APXINF_GR00T_W8A8_LEGACY_ATTENTION_BIAS_RESIDUAL` | crates/apxinf-model/src/gr00t/executor.rs |
| `APXINF_GR00T_W8A8_LEGACY_BIAS_RESIDUAL` | crates/apxinf-model/src/gr00t/int8_weights.rs |
| `APXINF_GR00T_W8A8_LEGACY_FC1_GELU_QUANT` | crates/apxinf-model/src/gr00t/int8_weights.rs |
| `APXINF_GR00T_W8A8_LEGACY_LAYER_NORM_QUANT` | crates/apxinf-model/src/gr00t/int8_weights.rs |
| `APXINF_GR00T_W8A8_LEGACY_QKV_GEMM_BIAS` | crates/apxinf-model/src/gr00t/int8_weights.rs |
| `APXINF_GR00T_W8A8_LEGACY_QK_RMS_MROPE` | crates/apxinf-model/src/gr00t/int8_executor.rs |
| `APXINF_GR00T_W8A8_LEGACY_SILU_MUL_PACKED4` | crates/apxinf-model/src/gr00t/int8_weights.rs |
| `APXINF_GR00T_W8A8_LEGACY_VISION_POINTWISE_PACK` | crates/apxinf-model/src/gr00t/int8_executor.rs |
| `APXINF_PREFILL_SDPA_LEGACY` | crates/apxinf-cuda/src/kernels/attention.rs, crates/apxinf-cuda/src/tests/operators.rs |
| `APXINF_VISION_SDPA_LEGACY` | crates/apxinf-cuda/src/kernels/attention.rs |

## Registering a new variable

Add the name here and in the same PR record where it is read. `just doc-lint`
fails on an `APXINF_*` name that appears in source but not in this file.

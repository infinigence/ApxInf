// Copyright 2026 apxinf contributors.
// Stable C ABI and CUDA launch adapter for custom static-inference operators.

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

// BF16 SiLU intermediate rounding matches unfused activation-then-multiply.

namespace {
#include "../kernels/custom/gdn_policy.h"
#include "../kernels/custom/math.cuh"
#include "../kernels/custom/reduction.cuh"
#include "../kernels/custom/quantization.cuh"
#include "../kernels/custom/w4a16.cuh"
#include "../kernels/custom/w8a16.cuh"
#include "../kernels/custom/qwen35_common.cuh"
#include "../kernels/custom/qwen35_gdn.cuh"
#include "../kernels/custom/qwen35_attention.cuh"
#include "../kernels/custom/preprocess.cuh"
#include "../kernels/custom/attention.cuh"
#include "../kernels/custom/normalization.cuh"
#include "../kernels/custom/activation.cuh"
#include "../kernels/custom/embedding.cuh"
#include "../kernels/custom/elementwise.cuh"
#include "../kernels/custom/fused.cuh"
#include "../kernels/custom/cache.cuh"
#include "../kernels/custom/linear_attention.cuh"
#include "../kernels/custom/gdn_chunk_state_wmma.cuh"
#include "../kernels/custom/gdn_chunk_gemm_wmma.cuh"
#include "../kernels/custom/gdn_attn_raw_wmma.cuh"
}  // namespace

extern "C" cudaError_t apxinf_sinusoidal_embedding_bf16(const void* positions,void* output,
    int rows,int dim,float scale,float frequency_step,cudaStream_t stream) {
  if(!positions||!output||rows<=0||dim<=0||dim%2||!std::isfinite(scale)||!std::isfinite(frequency_step))
    return cudaErrorInvalidValue;
  sinusoidal_embedding_bf16_kernel<<<rows,128,0,stream>>>(
      (const float*)positions,(__nv_bfloat16*)output,dim,scale,frequency_step);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_relu_bf16(const void* x,void* y,int64_t count,cudaStream_t stream) {
  if(!x||!y||count<=0) return cudaErrorInvalidValue;
  relu_bf16_kernel<<<256,256,0,stream>>>((const __nv_bfloat16*)x,(__nv_bfloat16*)y,count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_swiglu_bf16_rounded(
    const void* gate_up, void* output, int rows, int inner, cudaStream_t stream) {
  if (!gate_up || !output || rows <= 0 || inner <= 0) return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(rows) * inner;
  const int blocks = static_cast<int>((count + 255) / 256 > 65535 ? 65535 : (count + 255) / 256);
  if (swiglu_vec8_ok(gate_up, output, inner)) {
    const int vblocks = static_cast<int>((count / 8 + 255) / 256 > 65535
                                             ? 65535
                                             : (count / 8 + 255) / 256);
    swiglu_bf16_vec8_kernel<true><<<vblocks, 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<__nv_bfloat16*>(output), rows, inner);
    return cudaGetLastError();
  }
  swiglu_bf16_kernel<true><<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate_up), static_cast<__nv_bfloat16*>(output), rows, inner);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_channel_layer_norm_bf16_rounded(const void* x,
    const void* weight, const void* bias, void* out, int batches, int channels,
    int spatial, float eps, cudaStream_t stream) {
  if(!x||!weight||!bias||!out||batches<=0||channels<=0||spatial<=0||!std::isfinite(eps)||eps<=0
      ||static_cast<int64_t>(batches)*spatial>2147483647) return cudaErrorInvalidValue;
  channel_layer_norm_bf16_rounded_kernel<<<batches*spatial,256,0,stream>>>(
      (const __nv_bfloat16*)x,(const __nv_bfloat16*)weight,(const __nv_bfloat16*)bias,(__nv_bfloat16*)out,channels,spatial,eps);
  return cudaGetLastError();
}

namespace {

// Resolve the validated action Ada packed8 route before CUDA graph capture.
// Auto enables it only for the exact supported shape.
const int kActionAdaPacked8Mode = [] {
    const char* value = std::getenv("APXINF_PI05_ACTION_ADA_PACKED8");
    if (value == nullptr || std::strcmp(value, "auto") == 0) {
      return 2;
    }
    if (std::strcmp(value, "0") == 0 || std::strcmp(value, "off") == 0) {
      return 0;
    }
    if (std::strcmp(value, "1") == 0 || std::strcmp(value, "on") == 0) {
      return 1;
    }
    return -1;
  }();

}  // namespace

extern "C" cudaError_t apxinf_static_evict_l2(
    void* buffer, size_t bytes, uint32_t seed, cudaStream_t stream) {
  if (buffer == nullptr || bytes < sizeof(uint32_t) ||
      bytes % sizeof(uint32_t) != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  int blocks = static_cast<int>((bytes / sizeof(uint32_t) + threads - 1) /
                                threads);
  blocks = blocks > 4096 ? 4096 : blocks;
  l2_cache_evict_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<volatile uint32_t*>(buffer), bytes / sizeof(uint32_t), seed);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_rmsnorm_offset_bf16(
    const void* input, const void* weight, void* output, int rows, int columns,
    float epsilon, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      rows < 1 || rows > 8 || columns != 5120 || !(epsilon > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  qwen35_rmsnorm_offset_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), columns, epsilon);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_residual_add_rmsnorm_offset_bf16(
    void* residual, const void* delta, const void* weight, void* output,
    int rows, int columns, float epsilon, cudaStream_t stream) {
  if (residual == nullptr || delta == nullptr || weight == nullptr ||
      output == nullptr || rows < 1 || rows > 8 || columns != 5120 ||
      !(epsilon > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t shared_bytes = static_cast<size_t>(columns) * sizeof(float);
  qwen35_residual_add_rmsnorm_offset_bf16_kernel<<<
      rows, 256, shared_bytes, stream>>>(
      static_cast<__nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(delta),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), columns, epsilon);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_attention_prepare_bf16(
    const void* q_projection, const void* k_projection,
    const void* v_projection, const void* q_norm_weight,
    const void* k_norm_weight, void* query, void* key, void* value,
    void* gate, const void* position, cudaStream_t stream) {
  if (q_projection == nullptr || k_projection == nullptr ||
      v_projection == nullptr || q_norm_weight == nullptr ||
      k_norm_weight == nullptr || query == nullptr || key == nullptr ||
      value == nullptr || gate == nullptr || position == nullptr) {
    return cudaErrorInvalidValue;
  }
  qwen35_attention_prepare_bf16_kernel<<<28, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q_projection),
      static_cast<const __nv_bfloat16*>(k_projection),
      static_cast<const __nv_bfloat16*>(v_projection),
      static_cast<const __nv_bfloat16*>(q_norm_weight),
      static_cast<const __nv_bfloat16*>(k_norm_weight),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value),
      static_cast<__nv_bfloat16*>(gate),
      static_cast<const uint32_t*>(position));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_attention_prepare_m8_bf16(
    const void* q_projection, const void* k_projection,
    const void* v_projection, const void* q_norm_weight,
    const void* k_norm_weight, void* query, void* key, void* value,
    void* gate, const void* positions, int tokens, cudaStream_t stream) {
  if (q_projection == nullptr || k_projection == nullptr ||
      v_projection == nullptr || q_norm_weight == nullptr ||
      k_norm_weight == nullptr || query == nullptr || key == nullptr ||
      value == nullptr || gate == nullptr || positions == nullptr ||
      tokens < 1 || tokens > 8) {
    return cudaErrorInvalidValue;
  }
  qwen35_attention_prepare_m8_bf16_kernel<<<tokens * 28, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(q_projection),
      static_cast<const __nv_bfloat16*>(k_projection),
      static_cast<const __nv_bfloat16*>(v_projection),
      static_cast<const __nv_bfloat16*>(q_norm_weight),
      static_cast<const __nv_bfloat16*>(k_norm_weight),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value),
      static_cast<__nv_bfloat16*>(gate),
      static_cast<const uint32_t*>(positions), tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_attention_gate_bf16(
    const void* input, const void* gate, void* output, int count,
    cudaStream_t stream) {
  if (input == nullptr || gate == nullptr || output == nullptr ||
      count < 6144 || count > 8 * 6144 || count % 6144 != 0)
    return cudaErrorInvalidValue;
  qwen35_attention_gate_bf16_kernel<<<(count + 255) / 256, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_attention_flash_split_cta_bf16(
    const void* query, const void* key_cache, const void* value_cache,
    void* partial_max, void* partial_sum, void* partial_accumulator,
    void* output, int split_count, int bucket_kv_len, int max_seq_len,
    float scale, const void* position, cudaStream_t stream) {
  if (query == nullptr || key_cache == nullptr || value_cache == nullptr ||
      partial_max == nullptr || partial_sum == nullptr ||
      partial_accumulator == nullptr || output == nullptr ||
      position == nullptr || split_count < 2 || split_count > 16 ||
      (split_count & (split_count - 1)) != 0 || bucket_kv_len <= 0 ||
      bucket_kv_len > max_seq_len || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  dim3 stage_grid(24, split_count, 1);
  qwen35_attention_flash_split_cta_bf16_kernel<<<
      stage_grid, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key_cache),
      static_cast<const __nv_bfloat16*>(value_cache),
      static_cast<float*>(partial_max), static_cast<float*>(partial_sum),
      static_cast<float*>(partial_accumulator), split_count, bucket_kv_len,
      max_seq_len, scale, static_cast<const uint32_t*>(position));
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  qwen35_attention_flash_split_cta_reduce_bf16_kernel<<<
      24, 256, 0, stream>>>(
      static_cast<const float*>(partial_max),
      static_cast<const float*>(partial_sum),
      static_cast<const float*>(partial_accumulator),
      static_cast<__nv_bfloat16*>(output), split_count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_w8a16_gemv_bf16(
    const void* activation, const void* weight, const void* scales,
    void* output, int input_dim, int output_dim, cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || scales == nullptr ||
      output == nullptr || input_dim <= 0 || output_dim <= 0 ||
      input_dim % 8 != 0 || output_dim % 8 != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int kRowsPerBlock = 8;
  const size_t shared_bytes =
      static_cast<size_t>(input_dim) * sizeof(__nv_bfloat16) +
      kRowsPerBlock * sizeof(float);
  w8a16_gemv_bf16_kernel<<<
      output_dim / kRowsPerBlock, 256, shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int8_t*>(weight), static_cast<const float*>(scales),
      static_cast<__nv_bfloat16*>(output), input_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_recurrent_bf16(
    const void* query, const void* key, const void* value, const void* g,
    const void* beta, void* recurrent_state, void* output, int heads,
    int key_dim, int value_dim, cudaStream_t stream) {
  if (query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
      beta == nullptr || recurrent_state == nullptr || output == nullptr ||
      heads != 48 || key_dim != 128 || value_dim != 128) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_recurrent_bf16_kernel<<<heads, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key),
      static_cast<const __nv_bfloat16*>(value), static_cast<const float*>(g),
      static_cast<const float*>(beta), static_cast<float*>(recurrent_state),
      static_cast<__nv_bfloat16*>(output));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_conv4_silu_bf16(
    const void* input, const void* weight, void* conv_state, void* output,
    int channels, int kernel_size, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || conv_state == nullptr ||
      output == nullptr || channels != 10240 || kernel_size != 4) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (channels + 255) / 256;
  qwen35_gdn_conv4_silu_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(conv_state),
      static_cast<__nv_bfloat16*>(output), channels);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_prepare_bf16(
    const void* convolved_qkv, const void* a, const void* b,
    const void* a_log, const void* dt_bias, void* query, void* key,
    void* value, void* g, void* beta, cudaStream_t stream) {
  if (convolved_qkv == nullptr || a == nullptr || b == nullptr ||
      a_log == nullptr || dt_bias == nullptr || query == nullptr ||
      key == nullptr || value == nullptr || g == nullptr || beta == nullptr) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_prepare_bf16_kernel<<<48, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(convolved_qkv),
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value), static_cast<float*>(g),
      static_cast<float*>(beta));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_gated_rmsnorm_bf16(
    const void* input, const void* gate, const void* weight, void* output,
    int heads, int dimension, float epsilon, cudaStream_t stream) {
  if (input == nullptr || gate == nullptr || weight == nullptr ||
      output == nullptr || heads != 48 || dimension != 128 ||
      !(epsilon > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_gated_rmsnorm_bf16_kernel<<<heads, dimension, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), epsilon);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_conv4_prepare_bf16(
    const void* projected_qkv, const void* conv_weight, void* conv_state,
    const void* projected_ab, const void* a_log, const void* dt_bias,
    void* a_output, void* b_output, void* query, void* key, void* value,
    void* g, void* beta, cudaStream_t stream) {
  if (projected_qkv == nullptr || conv_weight == nullptr ||
      conv_state == nullptr || projected_ab == nullptr || a_log == nullptr ||
      dt_bias == nullptr || a_output == nullptr || b_output == nullptr ||
      query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
      beta == nullptr) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_conv4_prepare_bf16_kernel<<<40, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projected_qkv),
      static_cast<const __nv_bfloat16*>(conv_weight),
      static_cast<__nv_bfloat16*>(conv_state),
      static_cast<const __nv_bfloat16*>(projected_ab),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<__nv_bfloat16*>(a_output),
      static_cast<__nv_bfloat16*>(b_output),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value), static_cast<float*>(g),
      static_cast<float*>(beta));
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_conv4_prepare_m8_bf16(
    const void* projected_qkv, const void* conv_weight, void* conv_state,
    const void* projected_ab, const void* a_log, const void* dt_bias,
    void* a_output, void* b_output, void* query, void* key, void* value,
    void* g, void* beta, int tokens, cudaStream_t stream) {
  if (projected_qkv == nullptr || conv_weight == nullptr ||
      conv_state == nullptr || projected_ab == nullptr || a_log == nullptr ||
      dt_bias == nullptr || a_output == nullptr || b_output == nullptr ||
      query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
      beta == nullptr || tokens < 1 || tokens > 8) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_conv4_prepare_m8_bf16_kernel<<<40, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(projected_qkv),
      static_cast<const __nv_bfloat16*>(conv_weight),
      static_cast<__nv_bfloat16*>(conv_state),
      static_cast<const __nv_bfloat16*>(projected_ab),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias),
      static_cast<__nv_bfloat16*>(a_output),
      static_cast<__nv_bfloat16*>(b_output),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value), static_cast<float*>(g),
      static_cast<float*>(beta), tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_recurrent_m8_bf16(
    const void* query, const void* key, const void* value, const void* g,
    const void* beta, void* recurrent_state, void* output, int tokens,
    cudaStream_t stream) {
  if (query == nullptr || key == nullptr || value == nullptr || g == nullptr ||
      beta == nullptr || recurrent_state == nullptr || output == nullptr ||
      tokens < 1 || tokens > 8) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_recurrent_m8_bf16_kernel<<<48, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(query),
      static_cast<const __nv_bfloat16*>(key),
      static_cast<const __nv_bfloat16*>(value), static_cast<const float*>(g),
      static_cast<const float*>(beta), static_cast<float*>(recurrent_state),
      static_cast<__nv_bfloat16*>(output), tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qwen35_gdn_gated_rmsnorm_m8_bf16(
    const void* input, const void* gate, const void* weight, void* output,
    float epsilon, int tokens, cudaStream_t stream) {
  if (input == nullptr || gate == nullptr || weight == nullptr ||
      output == nullptr || !(epsilon > 0.0f) || tokens < 1 || tokens > 8) {
    return cudaErrorInvalidValue;
  }
  qwen35_gdn_gated_rmsnorm_m8_bf16_kernel<<<
      tokens * 48, 128, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), epsilon, tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_w4a16_gemv_bf16(
    const void* activation, const void* packed, const void* scales,
    const void* packed_zero_points, void* output, int input_dim,
    int output_dim, int group_size, cudaStream_t stream) {
  if (activation == nullptr || packed == nullptr || scales == nullptr ||
      packed_zero_points == nullptr || output == nullptr || input_dim <= 0 ||
      output_dim <= 0 || group_size != 32 || input_dim % 32 != 0 ||
      output_dim % 8 != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int kThreads = 256;
  constexpr int kRowsPerBlock = 8;
  const int blocks = (output_dim + kRowsPerBlock - 1) / kRowsPerBlock;
  w4a16_gemv_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(packed),
      static_cast<const __nv_bfloat16*>(scales),
      static_cast<const int32_t*>(packed_zero_points),
      static_cast<__nv_bfloat16*>(output), input_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_w4a16_gemv_bf16_staged(
    const void* activation, const void* packed, const void* scales,
    const void* packed_zero_points, void* output, int input_dim,
    int output_dim, int group_size, cudaStream_t stream) {
  if (activation == nullptr || packed == nullptr || scales == nullptr ||
      packed_zero_points == nullptr || output == nullptr || input_dim <= 0 ||
      output_dim <= 0 || group_size != 32 || input_dim % 32 != 0 ||
      output_dim % 8 != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int kThreads = 256;
  constexpr int kRowsPerBlock = 8;
  const int groups = input_dim / group_size;
  const size_t shared_bytes =
      static_cast<size_t>(input_dim) * sizeof(__nv_bfloat16) +
      static_cast<size_t>(kRowsPerBlock * groups) * sizeof(__nv_bfloat16) +
      static_cast<size_t>(groups) * sizeof(int32_t);
  const int blocks = output_dim / kRowsPerBlock;
  w4a16_gemv_bf16_staged_kernel<<<
      blocks, kThreads, shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(packed),
      static_cast<const __nv_bfloat16*>(scales),
      static_cast<const int32_t*>(packed_zero_points),
      static_cast<__nv_bfloat16*>(output), input_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_w4a16_gemm_m8_bf16(
    const void* activation, const void* packed, const void* scales,
    const void* packed_zero_points, void* output, int tokens, int input_dim,
    int output_dim, int group_size, cudaStream_t stream) {
  if (activation == nullptr || packed == nullptr || scales == nullptr ||
      packed_zero_points == nullptr || output == nullptr || tokens < 1 ||
      tokens > 8 || input_dim <= 0 || output_dim <= 0 || group_size != 32 ||
      input_dim % 32 != 0 || output_dim % 8 != 0) {
    return cudaErrorInvalidValue;
  }
  constexpr int kThreads = 256;
  constexpr int kRowsPerBlock = 8;
  const int blocks = output_dim / kRowsPerBlock;
  w4a16_gemm_m8_bf16_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const int32_t*>(packed),
      static_cast<const __nv_bfloat16*>(scales),
      static_cast<const int32_t*>(packed_zero_points),
      static_cast<__nv_bfloat16*>(output), tokens, input_dim, output_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_quantize_f16_e4m3(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 || !(scale > 0.0f))
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const float inverse_scale = 1.0f / scale;
  const bool aligned =
      (reinterpret_cast<uintptr_t>(input) & 3U) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 3U) == 0;
  int64_t vector_count = aligned ? count & ~int64_t{3} : 0;
  if (vector_count != 0) {
    const int64_t groups = vector_count / 4;
    int blocks = static_cast<int>((groups + threads - 1) / threads);
    blocks = blocks > 1024 ? 1024 : blocks;
    quantize_f16_e4m3_packed4_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const half*>(input),
        static_cast<__nv_fp8_e4m3*>(output), vector_count, inverse_scale);
  }
  const int64_t tail = count - vector_count;
  if (tail != 0) {
    int blocks = static_cast<int>((tail + threads - 1) / threads);
    blocks = blocks > 1024 ? 1024 : blocks;
    quantize_f16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const half*>(input) + vector_count,
        static_cast<__nv_fp8_e4m3*>(output) + vector_count,
        tail, inverse_scale);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_dequantize_e4m3_f16(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  dequantize_e4m3_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_fp8_e4m3*>(input),
      static_cast<half*>(output), count, scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_quantize_bf16_e4m3(
    const void* input, void* output, int64_t count, float scale,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  quantize_bf16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_fp8_e4m3*>(output), count, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_dynamic_quantize_rows_bf16_e4m3(
    const void* input, void* output, void* scales, int rows,
    int input_cols, int output_cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || input_cols <= 0 || output_cols < input_cols) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  if (input_cols % 8 == 0 && output_cols % 8 == 0) {
    constexpr int rows_per_block = threads / 32;
    const int blocks = (rows + rows_per_block - 1) / rows_per_block;
    quantize_rows_bf16_e4m3_vec8_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, output_cols);
  } else {
    quantize_rows_bf16_e4m3_kernel<<<rows, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, output_cols);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_dynamic_rms_norm_quantize_rows_bf16_e4m3(
    const void* input, const void* weight, void* output, void* scales,
    int rows, int input_cols, int output_cols, float eps,
    cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      scales == nullptr || rows <= 0 || input_cols <= 0 ||
      output_cols < input_cols || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  constexpr int rows_per_block = threads / 32;
  const int blocks = (rows + rows_per_block - 1) / rows_per_block;
  if (input_cols % 8 == 0 && output_cols % 8 == 0) {
    rms_norm_quantize_rows_bf16_e4m3_vec8_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, output_cols, eps);
  } else {
    rms_norm_quantize_rows_bf16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, output_cols, eps);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_dynamic_swiglu_quantize_rows_bf16_e4m3(
    const void* gate_up, const void* bias, void* output, void* scales,
    int rows, int input_cols, int inner, int output_cols,
    cudaStream_t stream) {
  if (gate_up == nullptr || output == nullptr || scales == nullptr ||
      rows <= 0 || inner <= 0 || input_cols < 2 * inner ||
      output_cols < inner) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  const size_t shared_bytes = static_cast<size_t>(inner) * sizeof(float);
  if (input_cols % 8 == 0 && inner % 8 == 0 && output_cols % 8 == 0) {
    swiglu_quantize_rows_bf16_e4m3_vec8_kernel
        <<<rows, threads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, inner, output_cols);
  } else if (input_cols % 4 == 0 && inner % 4 == 0 &&
             output_cols % 4 == 0) {
    swiglu_quantize_rows_bf16_e4m3_vec4_kernel
        <<<rows, threads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, inner, output_cols);
  } else {
    swiglu_quantize_rows_bf16_e4m3_kernel
        <<<rows, threads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(gate_up),
        static_cast<const __nv_bfloat16*>(bias),
        static_cast<__nv_fp8_e4m3*>(output), static_cast<float*>(scales),
        rows, input_cols, inner, output_cols);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t
apxinf_dynamic_bias_residual_rms_norm_quantize_rows_bf16_e4m3(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized, void* scales,
    int rows, int cols, int output_cols, float eps, cudaStream_t stream) {
  if (projection == nullptr || residual == nullptr || weight == nullptr ||
      hidden == nullptr || normalized == nullptr || scales == nullptr ||
      rows <= 0 || cols <= 0 || output_cols < cols || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  constexpr int threads = 256;
  if (rows <= 64) {
    constexpr int rows_per_block = threads / 32;
    const int blocks = (rows + rows_per_block - 1) / rows_per_block;
    if (cols % 8 == 0 && output_cols % 8 == 0) {
      bias_residual_rms_norm_quantize_rows_bf16_e4m3_vec8_kernel
          <<<blocks, threads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(projection),
              static_cast<const __nv_bfloat16*>(bias),
              static_cast<const __nv_bfloat16*>(residual),
              static_cast<const __nv_bfloat16*>(weight),
              static_cast<__nv_bfloat16*>(hidden),
              static_cast<__nv_fp8_e4m3*>(normalized),
              static_cast<float*>(scales), rows, cols, output_cols, eps);
    } else {
      bias_residual_rms_norm_quantize_rows_bf16_e4m3_kernel
          <<<blocks, threads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(projection),
              static_cast<const __nv_bfloat16*>(bias),
              static_cast<const __nv_bfloat16*>(residual),
              static_cast<const __nv_bfloat16*>(weight),
              static_cast<__nv_bfloat16*>(hidden),
              static_cast<__nv_fp8_e4m3*>(normalized),
              static_cast<float*>(scales), rows, cols, output_cols, eps);
    }
  } else {
    if (cols % 8 == 0 && output_cols % 8 == 0) {
      bias_residual_rms_norm_quantize_rows_bf16_e4m3_large_vec8_kernel
          <<<rows, threads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(projection),
              static_cast<const __nv_bfloat16*>(bias),
              static_cast<const __nv_bfloat16*>(residual),
              static_cast<const __nv_bfloat16*>(weight),
              static_cast<__nv_bfloat16*>(hidden),
              static_cast<__nv_fp8_e4m3*>(normalized),
              static_cast<float*>(scales), rows, cols, output_cols, eps);
    } else {
      bias_residual_rms_norm_quantize_rows_bf16_e4m3_large_kernel
          <<<rows, threads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(projection),
              static_cast<const __nv_bfloat16*>(bias),
              static_cast<const __nv_bfloat16*>(residual),
              static_cast<const __nv_bfloat16*>(weight),
              static_cast<__nv_bfloat16*>(hidden),
              static_cast<__nv_fp8_e4m3*>(normalized),
              static_cast<float*>(scales), rows, cols, output_cols, eps);
    }
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_slice_columns_bf16(
    const void* input, void* output, int rows, int input_cols,
    int output_cols, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || rows <= 0 ||
      output_cols <= 0 || output_cols > input_cols) {
    return cudaErrorInvalidValue;
  }
  const int64_t count = static_cast<int64_t>(rows) * output_cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  slice_columns_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), rows, input_cols, output_cols);
  return cudaGetLastError();
}

// FIX (implement_r8): block-per-row tiled softmax for the route-3 composed
// vision attention: fp32 scores in, bf16 probabilities out (round-to-nearest,
// matching the cast kernels). The stock thread-per-element softmax is
// structurally unviable at the [16*3472, 3472] vision-segment geometry.
__global__ void row_softmax_f32_bf16_kernel(
    const float* input, __nv_bfloat16* output, uint32_t cols, uint32_t rows) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  const float* x = input + static_cast<size_t>(row) * cols;
  __shared__ float scratch[32];
  float local = -INFINITY;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    local = fmaxf(local, x[i]);
  }
  const float max_val = block_max_parallel_unsafe(local, scratch);
  float partial = 0.0f;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    partial += expf(x[i] - max_val);
  }
  // Finish all maximum reads before the sum overwrites shared scratch.
  __syncthreads();
  const float sum = block_sum_parallel_unsafe(partial, scratch);
  __nv_bfloat16* y = output + static_cast<size_t>(row) * cols;
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    y[i] = __float2bfloat16(expf(x[i] - max_val) / sum);
  }
}

extern "C" cudaError_t apxinf_static_row_softmax_f32_bf16(
    const void* input, void* output, uint32_t cols, uint32_t rows,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || cols == 0 || rows == 0) {
    return cudaErrorInvalidValue;
  }
  row_softmax_f32_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const float*>(input), static_cast<__nv_bfloat16*>(output),
      cols, rows);
  return cudaGetLastError();
}

// FIX (implement_final_r20): Option A budget repair for the composed hdim256 text
// path -- tiled block-per-row causal fp32 softmax replacing the thread-per-element
// attention_softmax_f32_kernel (measured ~3.9-4.1s per composed full-attention layer,
// ~31.6s of the ~43s prefill). Row contract identical to the stock kernel
// (kernels/custom/attention.cuh): row = blockIdx.x, seq_pos = row / n_heads,
// valid = min(seq_pos + kv_offset + 1u, cols); valid cells expf(x-max)/sum, masked
// cells exact 0.0f. In-place safe with input == output: every index is owned by
// exactly one thread and the block reductions (block_max/block_sum) synchronize
// between the read and write passes. Revert in the acceptance-bound revision.
__global__ void row_softmax_causal_f32_kernel(
    const float* input, float* output, uint32_t cols, uint32_t rows,
    uint32_t kv_offset, uint32_t n_heads) {
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;
  const float* x = input + static_cast<size_t>(row) * cols;
  float* y = output + static_cast<size_t>(row) * cols;
  const uint32_t seq_pos = row / n_heads;
  const uint32_t valid = min(seq_pos + kv_offset + 1u, cols);
  __shared__ float scratch[32];
  float local = -INFINITY;
  for (uint32_t i = threadIdx.x; i < valid; i += blockDim.x) {
    local = fmaxf(local, x[i]);
  }
  const float max_val = block_max_parallel_unsafe(local, scratch);
  float partial = 0.0f;
  for (uint32_t i = threadIdx.x; i < valid; i += blockDim.x) {
    partial += expf(x[i] - max_val);
  }
  // Finish all maximum reads before the sum overwrites shared scratch.
  __syncthreads();
  const float sum = block_sum_parallel_unsafe(partial, scratch);
  for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
    y[i] = (i < valid) ? (expf(x[i] - max_val) / sum) : 0.0f;
  }
}

extern "C" cudaError_t apxinf_static_row_softmax_causal_f32(
    const void* input, void* output, uint32_t cols, uint32_t rows,
    uint32_t kv_offset, uint32_t n_heads, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || cols == 0 || rows == 0 ||
      n_heads == 0 || n_heads > rows) {
    return cudaErrorInvalidValue;
  }
  row_softmax_causal_f32_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const float*>(input), static_cast<float*>(output), cols, rows,
      kv_offset, n_heads);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_cast_f16_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0)
    return cudaErrorInvalidValue;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  cast_f16_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_rgb_u8_to_patches_e4m3(
    const void* images, void* patches, int views, int image_size,
    int patch_size, int layout, float scale, cudaStream_t stream) {
  if (images == nullptr || patches == nullptr || views <= 0 ||
      image_size <= 0 || patch_size <= 0 || image_size % patch_size != 0 ||
      (layout != 0 && layout != 1) || !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const int patches_per_side = image_size / patch_size;
  const int64_t count = static_cast<int64_t>(views) * patches_per_side *
                        patches_per_side * 3 * patch_size * patch_size;
  constexpr int threads = 256;
  int blocks = static_cast<int>((count + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  if (layout == 0) {
    rgb_u8_to_patches_e4m3_kernel<true><<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(images),
        static_cast<__nv_fp8_e4m3*>(patches), views, image_size, patch_size,
        1.0f / scale);
  } else {
    rgb_u8_to_patches_e4m3_kernel<false><<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(images),
        static_cast<__nv_fp8_e4m3*>(patches), views, image_size, patch_size,
        1.0f / scale);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_rgb_u8_to_normalized_temporal_merged_patches_bf16(
    const void* images, void* patches, int views, int image_size,
    int patch_size, int temporal_patch_size, int merge_size, int layout,
    double rescale_factor, float mean0, float mean1, float mean2,
    float std0, float std1, float std2,
    cudaStream_t stream) {
  if (images == nullptr || patches == nullptr || views <= 0 ||
      image_size <= 0 || patch_size <= 0 || temporal_patch_size <= 0 ||
      merge_size <= 0 || (layout != 0 && layout != 1) ||
      !std::isfinite(rescale_factor) || !(rescale_factor > 0.0f) ||
      !std::isfinite(mean0) || !std::isfinite(mean1) ||
      !std::isfinite(mean2) || !std::isfinite(std0) ||
      !std::isfinite(std1) || !std::isfinite(std2) || !(std0 > 0.0f) ||
      !(std1 > 0.0f) || !(std2 > 0.0f)) {
    return cudaErrorInvalidValue;
  }

  constexpr int64_t kMaxKernelInt = std::numeric_limits<int>::max();
  constexpr int64_t kMaxKernelIndex = std::numeric_limits<int64_t>::max();
  const int64_t patch_merge =
      static_cast<int64_t>(patch_size) * merge_size;
  if (patch_merge > image_size || image_size % patch_merge != 0) {
    return cudaErrorInvalidValue;
  }

  const int64_t grid_size64 = image_size / patch_size;
  const int64_t rows_per_view64 = grid_size64 * grid_size64;
  const int64_t patch_area64 =
      static_cast<int64_t>(patch_size) * patch_size;
  if (rows_per_view64 > kMaxKernelInt || patch_area64 > kMaxKernelInt) {
    return cudaErrorInvalidValue;
  }
  const int64_t three_patch_area64 = 3 * patch_area64;
  if (temporal_patch_size > kMaxKernelInt / three_patch_area64 ||
      views > kMaxKernelInt / rows_per_view64) {
    return cudaErrorInvalidValue;
  }
  const int64_t patch_width64 =
      static_cast<int64_t>(temporal_patch_size) * three_patch_area64;
  const int64_t patch_rows64 =
      static_cast<int64_t>(views) * rows_per_view64;
  if (patch_rows64 > kMaxKernelIndex / patch_width64) {
    return cudaErrorInvalidValue;
  }
  const int64_t pixels_per_view64 =
      static_cast<int64_t>(image_size) * image_size;
  if (pixels_per_view64 > kMaxKernelIndex / 3 ||
      views > kMaxKernelIndex / (3 * pixels_per_view64)) {
    return cudaErrorInvalidValue;
  }

  const int grid_size = static_cast<int>(grid_size64);
  const int patch_width = static_cast<int>(patch_width64);
  const int64_t count = patch_rows64 * patch_width64;
  constexpr int threads = 256;
  // Avoid overflowing at the accepted int64_t element-count boundary.
  const int64_t requested_blocks =
      count / threads + (count % threads != 0 ? 1 : 0);
  const int blocks =
      static_cast<int>(requested_blocks > 1024 ? 1024 : requested_blocks);
  if (layout == 0) {
    rgb_u8_to_normalized_temporal_merged_patches_bf16_kernel<true><<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(images),
        static_cast<__nv_bfloat16*>(patches), views, image_size, patch_size,
        temporal_patch_size, merge_size, rescale_factor, mean0, mean1, mean2,
        std0, std1, std2);
  } else {
    rgb_u8_to_normalized_temporal_merged_patches_bf16_kernel<false><<<blocks, threads, 0, stream>>>(
        static_cast<const uint8_t*>(images),
        static_cast<__nv_bfloat16*>(patches), views, image_size, patch_size,
        temporal_patch_size, merge_size, rescale_factor, mean0, mean1, mean2,
        std0, std1, std2);
  }
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_mqa_flash_f16(
    const void* q, const void* prefix_k, const void* prefix_v,
    const void* suffix_k, const void* suffix_v, void* output,
    int suffix_tokens, int heads, int head_dim, int prefix_tokens,
    cudaStream_t stream) {
  if (suffix_tokens <= 0 || heads <= 0 || head_dim <= 0 || head_dim > 256 ||
      prefix_tokens < 0) return cudaErrorInvalidValue;
  int threads = 256;
  int warps = threads / 32;
  size_t shared_bytes =
      static_cast<size_t>(prefix_tokens + suffix_tokens + warps) * sizeof(float);
  mqa_flash_f16_kernel<<<dim3(suffix_tokens, heads), threads, shared_bytes, stream>>>(
      static_cast<const half*>(q), static_cast<const half*>(prefix_k),
      static_cast<const half*>(prefix_v), static_cast<const half*>(suffix_k),
      static_cast<const half*>(suffix_v), static_cast<half*>(output),
      suffix_tokens, heads, head_dim, prefix_tokens);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_rms_norm_quant_f16_e4m3(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  rms_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(weight),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_layer_norm_quant_bf16_e4m3(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, float scale, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || !(eps > 0.0f) ||
      !(scale > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
  const int threads = 256;
  layer_norm_quant_bf16_e4m3_kernel<<<rows, threads, shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_layer_norm_quant_f16_e4m3(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  layer_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(weight),
      static_cast<const half*>(bias), static_cast<__nv_fp8_e4m3*>(output),
      rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_gelu_quant_f16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_gelu_quant_f16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), count, cols, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_gelu_quant_bf16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, cudaStream_t stream) {
  if (!input || !bias || !output || rows <= 0 || cols <= 0 || cols % 4 != 0 ||
      !(scale > 0.0f) || reinterpret_cast<uintptr_t>(input) % alignof(Bf16x4) != 0 ||
      reinterpret_cast<uintptr_t>(bias) % alignof(Bf16x4) != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(uint32_t) != 0)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const int64_t quad_count = static_cast<int64_t>(rows) * cols / 4;
  int blocks = static_cast<int>((quad_count + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_gelu_quant_bf16_e4m3_packed4_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), quad_count, cols, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_silu_quant_f16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_silu_quant_f16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), count, cols, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_silu_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_silu_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_f16(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(bias),
      static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_embedding_f16(
    const void* table, const void* ids, void* output, int tokens,
    int width, int vocab_size, cudaStream_t stream) {
  if (tokens <= 0 || width <= 0 || vocab_size <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(tokens) * width;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  embedding_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(table), static_cast<const uint32_t*>(ids),
      static_cast<half*>(output), tokens, width, vocab_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_concat_rows_f16(
    const void* first, const void* second, void* output, int first_rows,
    int second_rows, int cols, cudaStream_t stream) {
  if (first_rows <= 0 || second_rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t first_count = static_cast<int64_t>(first_rows) * cols;
  int64_t total_count = static_cast<int64_t>(first_rows + second_rows) * cols;
  int blocks = static_cast<int>((total_count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  concat_rows_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(first), static_cast<const half*>(second),
      static_cast<half*>(output), first_count, total_count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_euler_update_f16(
    const void* state, const void* velocity, void* output, int64_t count,
    float dt, cudaStream_t stream) {
  if (count <= 0) return cudaErrorInvalidValue;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  euler_update_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(state), static_cast<const half*>(velocity),
      static_cast<half*>(output), count, dt);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_geglu_quant_f16_e4m3(
    const void* gate_up, void* output, int rows, int inner, float scale,
    cudaStream_t stream) {
  if (rows <= 0 || inner <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  if ((inner & 1) != 0) return cudaErrorInvalidValue;
  const bool packed8 = (inner & 7) == 0 &&
      (reinterpret_cast<uintptr_t>(gate_up) & 7U) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 7U) == 0;
  if (packed8) {
    int group_count = rows * (inner / 8);
    int blocks = (group_count + 255) / 256;
    blocks = blocks > 1024 ? 1024 : blocks;
    geglu_quant_f16_e4m3_packed8_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const half*>(gate_up),
        static_cast<__nv_fp8_e4m3*>(output), rows, inner, 1.0f / scale);
    return cudaGetLastError();
  }
  const bool packed4 = (inner & 3) == 0 &&
      (reinterpret_cast<uintptr_t>(gate_up) & 3U) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 3U) == 0;
  if (packed4) {
    int group_count = rows * (inner / 4);
    int blocks = (group_count + 255) / 256;
    blocks = blocks > 1024 ? 1024 : blocks;
    geglu_quant_f16_e4m3_packed4_kernel<<<blocks, 256, 0, stream>>>(
        static_cast<const half*>(gate_up),
        static_cast<__nv_fp8_e4m3*>(output), rows, inner, 1.0f / scale);
    return cudaGetLastError();
  }
  int pair_count = rows * (inner / 2);
  int blocks = (pair_count + 255) / 256;
  blocks = blocks > 1024 ? 1024 : blocks;
  geglu_quant_f16_e4m3_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(gate_up), static_cast<__nv_fp8_e4m3*>(output),
      rows, inner, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_f16(
    const void* projection, const void* bias, const void* residual, void* output,
    int rows, int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_residual_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(bias),
      static_cast<const half*>(residual), static_cast<half*>(output), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_rms_norm_quant_f16_e4m3(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  bias_residual_rms_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(bias),
      static_cast<const half*>(residual), static_cast<const half*>(weight),
      static_cast<half*>(hidden), static_cast<__nv_fp8_e4m3*>(normalized),
      rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_residual_layer_norm_quant_f16_e4m3(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, float scale,
    cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  bias_residual_layer_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(projection_bias),
      static_cast<const half*>(residual), static_cast<const half*>(norm_weight),
      static_cast<const half*>(norm_bias), static_cast<half*>(hidden),
      static_cast<__nv_fp8_e4m3*>(normalized), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_ada_rms_norm_quant_f16_e4m3(
    const void* input, const void* style, void* output, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  ada_rms_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(input), static_cast<const half*>(style),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_ada_gate_residual_f16(
    const void* projection, const void* residual, const void* style,
    void* output, int rows, int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  ada_gate_residual_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(residual),
      static_cast<const half*>(style), static_cast<half*>(output), rows, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_ada_gate_residual_rms_norm_quant_f16_e4m3(
    const void* projection, const void* residual, const void* gate_style,
    const void* norm_style, void* hidden, void* normalized, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || !(scale > 0.0f)) return cudaErrorInvalidValue;
  const int packed8_mode = kActionAdaPacked8Mode;
  if (packed8_mode < 0) return cudaErrorInvalidValue;
  const bool packed8_exact_shape = rows == 10 && cols == 1024;
  if (packed8_mode == 1 && !packed8_exact_shape) return cudaErrorInvalidValue;
  if (packed8_mode != 0 && packed8_exact_shape) {
    if (!std::isfinite(scale) ||
        projection == nullptr || residual == nullptr || gate_style == nullptr ||
        norm_style == nullptr || hidden == nullptr || normalized == nullptr) {
      return cudaErrorInvalidValue;
    }
    ada_gate_residual_rms_norm_quant_f16_e4m3_packed8_kernel
        <<<rows, 256, 0, stream>>>(
            static_cast<const half*>(projection),
            static_cast<const half*>(residual),
            static_cast<const half*>(gate_style),
            static_cast<const half*>(norm_style), static_cast<half*>(hidden),
            static_cast<__nv_fp8_e4m3*>(normalized), eps, 1.0f / scale);
    return cudaGetLastError();
  }
  ada_gate_residual_rms_norm_quant_f16_e4m3_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(residual),
      static_cast<const half*>(gate_style), static_cast<const half*>(norm_style),
      static_cast<half*>(hidden), static_cast<__nv_fp8_e4m3*>(normalized),
      rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qkv_rope_f16(
    const void* qkv, const void* bias, void* q, void* k, void* v, int tokens, int q_heads,
    int kv_heads, int head_dim, float theta, int position_offset,
    int kv_output_offset, cudaStream_t stream) {
  if (tokens <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      head_dim > 256 || (head_dim & 1) != 0) return cudaErrorInvalidValue;
  qkv_rope_f16_kernel<<<dim3(tokens, q_heads + 2 * kv_heads), head_dim / 2, 0, stream>>>(
      static_cast<const half*>(qkv), static_cast<const half*>(bias),
      static_cast<half*>(q), static_cast<half*>(k),
      static_cast<half*>(v), tokens, q_heads, kv_heads, head_dim, theta,
      position_offset, kv_output_offset);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_qkv_split_bias_f16(
    const void* qkv, const void* bias, void* q, void* k, void* v,
    int tokens, int projection_width, cudaStream_t stream) {
  if (tokens <= 0 || projection_width <= 0) return cudaErrorInvalidValue;
  qkv_split_bias_f16_kernel<<<tokens, 256, 0, stream>>>(
      static_cast<const half*>(qkv), static_cast<const half*>(bias),
      static_cast<half*>(q), static_cast<half*>(k), static_cast<half*>(v),
      tokens, projection_width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_mha_flash_f16(
    const void* q, const void* k, const void* v, void* output,
    int tokens_per_batch, int batches, int heads, int head_dim, cudaStream_t stream) {
  if (tokens_per_batch <= 0 || batches <= 0 || heads <= 0 ||
      head_dim <= 0 || head_dim > 256)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  size_t shared_bytes = static_cast<size_t>(tokens_per_batch + threads / 32) * sizeof(float);
  mha_flash_f16_kernel<<<dim3(tokens_per_batch, heads, batches), threads, shared_bytes, stream>>>(
      static_cast<const half*>(q), static_cast<const half*>(k),
      static_cast<const half*>(v), static_cast<half*>(output),
      tokens_per_batch, heads, head_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_bias_position_f16(
    const void* projection, const void* bias, const void* position,
    void* output, int rows, int cols, int tokens_per_view, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0 || tokens_per_view <= 0 ||
      rows % tokens_per_view != 0) return cudaErrorInvalidValue;
  int64_t count = static_cast<int64_t>(rows) * cols;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_position_f16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const half*>(projection), static_cast<const half*>(bias),
      static_cast<const half*>(position), static_cast<half*>(output),
      count, cols, tokens_per_view);
  return cudaGetLastError();
}

// ── linear_attention.cuh launch adapters ───────────────────────────────────

extern "C" cudaError_t apxinf_static_cast_f32_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0)
    return cudaErrorInvalidValue;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  cast_f32_to_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const float*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_cast_bf16_f32(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0)
    return cudaErrorInvalidValue;
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 4096 ? 4096 : blocks;
  cast_bf16_to_f32_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<float*>(output), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_causal_conv1d_silu_bf16(
    const void* x, const void* weight, const void* state, void* out,
    void* new_state, int channels, int seq, int kernel_size,
    int64_t x_row_stride, cudaStream_t stream) {
  if (x == nullptr || weight == nullptr || out == nullptr ||
      new_state == nullptr || channels <= 0 || seq <= 0 || kernel_size <= 0 ||
      kernel_size > 8 || x_row_stride < channels) {
    return cudaErrorInvalidValue;
  }
  const int channel_blocks = (channels + 255) / 256;
  causal_conv1d_silu_bf16_kernel<<<dim3(seq, channel_blocks), 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(state),
      static_cast<__nv_bfloat16*>(out),
      static_cast<__nv_bfloat16*>(new_state),
      channels, seq, kernel_size, x_row_stride);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_qk_prep_bf16(
    const void* conv_out, void* q_out, void* k_out,
    int seq, int seq_pad, int conv_dim, int key_dim,
    int num_v_heads, int head_k_dim, float scale, float eps, int recurrent,
    cudaStream_t stream) {
  if (conv_out == nullptr || q_out == nullptr || k_out == nullptr ||
      seq <= 0 || seq_pad < seq || conv_dim < 2 * key_dim || key_dim <= 0 ||
      num_v_heads <= 0 || head_k_dim <= 0 ||
      (head_k_dim & (head_k_dim - 1)) != 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  if (key_dim % head_k_dim != 0) return cudaErrorInvalidValue;
  const int num_k_heads = key_dim / head_k_dim;
  if (num_v_heads % num_k_heads != 0) return cudaErrorInvalidValue;
  const size_t smem = static_cast<size_t>(2 * head_k_dim) * sizeof(float);
  gdn_qk_prep_kernel<<<dim3(seq, num_k_heads), head_k_dim, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(conv_out),
      static_cast<float*>(q_out), static_cast<float*>(k_out),
      seq, seq_pad, conv_dim, key_dim, num_v_heads, head_k_dim, scale, eps, recurrent != 0);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_vb_prep_bf16(
    const void* conv_out, const void* b_proj, const void* a_proj,
    const void* dt_bias, const void* a_log,
    void* v_out, void* beta_out, void* g_out,
    int seq, int seq_pad, int conv_dim, int v_offset,
    int num_v_heads, int ba_row_stride, int head_v_dim,
    cudaStream_t stream) {
  if (conv_out == nullptr || b_proj == nullptr || a_proj == nullptr ||
      dt_bias == nullptr || a_log == nullptr || v_out == nullptr ||
      beta_out == nullptr || g_out == nullptr || seq <= 0 || seq_pad < seq ||
      num_v_heads <= 0 || head_v_dim <= 0 || ba_row_stride < num_v_heads) {
    return cudaErrorInvalidValue;
  }
  gdn_vb_prep_kernel<<<dim3(seq, num_v_heads), head_v_dim, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(conv_out),
      static_cast<const __nv_bfloat16*>(b_proj),
      static_cast<const __nv_bfloat16*>(a_proj),
      static_cast<const float*>(dt_bias), static_cast<const float*>(a_log),
      static_cast<float*>(v_out), static_cast<float*>(beta_out),
      static_cast<float*>(g_out),
      seq, seq_pad, conv_dim, v_offset, ba_row_stride, head_v_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_cumsum_f32(
    const void* g, void* g_cum, int seq_pad, int num_v_heads, int chunk_size,
    cudaStream_t stream) {
  if (g == nullptr || g_cum == nullptr || seq_pad <= 0 ||
      num_v_heads <= 0 || chunk_size <= 0 || seq_pad % chunk_size != 0) {
    return cudaErrorInvalidValue;
  }
  const int chunks = seq_pad / chunk_size;
  gdn_cumsum_kernel<<<dim3(chunks, num_v_heads), 32, 0, stream>>>(
      static_cast<const float*>(g), static_cast<float*>(g_cum),
      seq_pad, chunk_size);
  return cudaGetLastError();
}


extern "C" cudaError_t apxinf_static_gdn_attn_raw_f32(
    const void* q, const void* k, const void* beta, const void* g_cum,
    void* a_out, void* t_out, int seq_pad, int num_v_heads, int head_k_dim,
    int chunk_size, const ApxinfGdnPolicy* policy, cudaStream_t stream) {
  if (policy == nullptr) return cudaErrorInvalidValue;
  if (q == nullptr || k == nullptr || beta == nullptr || g_cum == nullptr ||
      a_out == nullptr || t_out == nullptr || seq_pad <= 0 ||
      num_v_heads <= 0 || head_k_dim <= 0 || chunk_size <= 0 ||
      seq_pad % chunk_size != 0) {
    return cudaErrorInvalidValue;
  }
  const int chunks = seq_pad / chunk_size;
  if (policy->attn_raw_wmma != APXINF_GDN_WMMA_OFF && head_k_dim == 128 &&
      chunk_size == 64) {
    const size_t smem =
        static_cast<size_t>(64 * 128) * sizeof(__nv_bfloat16) * 4 +
        static_cast<size_t>(64 * 64) * sizeof(float) * 2;
    const bool split = policy->attn_raw_wmma >= APXINF_GDN_WMMA_SPLIT2;
    const void* entry =
        split ? reinterpret_cast<const void*>(gdn_attn_raw_wmma_kernel<true>)
              : reinterpret_cast<const void*>(gdn_attn_raw_wmma_kernel<false>);
    static const void* attn_wmma_opted = nullptr;
    if (attn_wmma_opted != entry) {
      const cudaError_t attr = cudaFuncSetAttribute(
          entry, cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem));
      if (attr != cudaSuccess) {
        return attr;
      }
      attn_wmma_opted = entry;
    }
#define GDN_ATTN_WMMA_ARGS                                                     \
  static_cast<const float*>(q), static_cast<const float*>(k),                  \
      static_cast<const float*>(beta), static_cast<const float*>(g_cum),       \
      static_cast<float*>(a_out), static_cast<float*>(t_out), seq_pad
    if (split) {
      gdn_attn_raw_wmma_kernel<true>
          <<<dim3(chunks, num_v_heads), 256, smem, stream>>>(GDN_ATTN_WMMA_ARGS);
    } else {
      gdn_attn_raw_wmma_kernel<false>
          <<<dim3(chunks, num_v_heads), 256, smem, stream>>>(GDN_ATTN_WMMA_ARGS);
    }
#undef GDN_ATTN_WMMA_ARGS
    return cudaGetLastError();
  }
  // One K tile and one Q tile for the chunk, rows padded by one float to keep
  // the warp off a single shared-memory bank (see the kernel comment).
  const size_t attn_smem =
      2u * static_cast<size_t>(chunk_size) * (head_k_dim + 1) * sizeof(float);
  if (attn_smem > 48u * 1024u) {
    static bool attn_opted_in = false;
    if (!attn_opted_in) {
      const cudaError_t attr = cudaFuncSetAttribute(
          reinterpret_cast<const void*>(gdn_attn_raw_kernel),
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(attn_smem));
      if (attr != cudaSuccess) {
        return attr;
      }
      attn_opted_in = true;
    }
  }
  gdn_attn_raw_kernel<<<dim3(chunks, num_v_heads), 256, attn_smem, stream>>>(
      static_cast<const float*>(q), static_cast<const float*>(k),
      static_cast<const float*>(beta), static_cast<const float*>(g_cum),
      static_cast<float*>(a_out), static_cast<float*>(t_out),
      seq_pad, head_k_dim, chunk_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_tri_solve_f32(
    void* a, int matrices, int chunk_size, cudaStream_t stream) {
  if (a == nullptr || matrices <= 0 || chunk_size <= 0 || chunk_size > 128) {
    return cudaErrorInvalidValue;
  }
  if (chunk_size == 64) {
    gdn_block_inverse64_kernel<<<matrices, 256, 0, stream>>>(static_cast<float*>(a));
    return cudaGetLastError();
  }
  const size_t smem =
      (static_cast<size_t>(chunk_size) * chunk_size + chunk_size) * sizeof(float);
  gdn_tri_solve_kernel<<<matrices, 64, smem, stream>>>(
      static_cast<float*>(a), chunk_size);
  return cudaGetLastError();
}



#define CHUNK_GEMM_ARGS                                                    \
  chunks, num_v_heads, gemm_smem, stream, a, v, k, beta, g_cum, vt_out,    \
      kcd_out, seq_pad, head_k_dim, head_v_dim, chunk_size

template <int TILE>
static cudaError_t launch_chunk_gemm(
    int chunks, int num_v_heads, size_t gemm_smem, cudaStream_t stream,
    const void* a, const void* v, const void* k, const void* beta,
    const void* g_cum, void* vt_out, void* kcd_out, int seq_pad,
    int head_k_dim, int head_v_dim, int chunk_size) {
  if (gemm_smem > 48u * 1024u) {
    static bool gemm_opted_in = false;
    if (!gemm_opted_in) {
      const cudaError_t attr = cudaFuncSetAttribute(
          reinterpret_cast<const void*>(gdn_chunk_gemm_kernel<TILE>),
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(gemm_smem));
      if (attr != cudaSuccess) {
        return attr;
      }
      gemm_opted_in = true;
    }
  }
  gdn_chunk_gemm_kernel<TILE><<<dim3(chunks, num_v_heads), 256, gemm_smem, stream>>>(
      static_cast<const float*>(a), static_cast<const float*>(v),
      static_cast<const float*>(k), static_cast<const float*>(beta),
      static_cast<const float*>(g_cum),
      static_cast<float*>(vt_out), static_cast<float*>(kcd_out),
      seq_pad, head_k_dim, head_v_dim, chunk_size);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_chunk_gemm_f32(
    const void* a, const void* v, const void* k, const void* beta,
    const void* g_cum, void* vt_out, void* kcd_out,
    int seq_pad, int num_v_heads, int head_k_dim, int head_v_dim,
    int chunk_size, const ApxinfGdnPolicy* policy, cudaStream_t stream) {
  if (policy == nullptr) return cudaErrorInvalidValue;
  if (a == nullptr || v == nullptr || k == nullptr || beta == nullptr ||
      g_cum == nullptr || vt_out == nullptr || kcd_out == nullptr ||
      seq_pad <= 0 || num_v_heads <= 0 || head_k_dim <= 0 ||
      head_v_dim <= 0 || chunk_size <= 0 || seq_pad % chunk_size != 0) {
    return cudaErrorInvalidValue;
  }
  const int chunks = seq_pad / chunk_size;
  // vb and kb tiles, precomputed once per chunk and held in BF16 because every
  // value in them is already on the BF16 grid (see the kernel comment). 32KB at
  // the shipped shape rather than 64KB.
  const size_t gemm_smem = static_cast<size_t>(chunk_size) *
                           (head_v_dim + head_k_dim) * sizeof(__nv_bfloat16);
  // Tensor-core form. Off by default, unlike the chunk-state scan: this
  // kernel rounds both outputs to BF16, so the scalar form lands within
  // 1.369621e-7 of an fp64 reference and the split form within 2.959516e-6 --
  // 21.6x more, though both sit far below the BF16 grid the results are
  // written on. It is worth 1.9% of the scene and the end-to-end probe moves
  // with it, so it is the owner's call rather than a default.
  if (policy->chunk_gemm_wmma != APXINF_GDN_WMMA_OFF && head_k_dim == 128 &&
      head_v_dim == 128 &&
      chunk_size == 64) {
    const size_t smem = static_cast<size_t>(64 * 128) * sizeof(__nv_bfloat16) * 2 +
                        static_cast<size_t>(64 * 64) * sizeof(__nv_bfloat16) * 3 +
                        static_cast<size_t>(64 * 128) * sizeof(float);
    const int passes = policy->chunk_gemm_wmma;
    const void* entry =
        passes == 3 ? reinterpret_cast<const void*>(gdn_chunk_gemm_wmma_kernel<3>)
        : passes == 2 ? reinterpret_cast<const void*>(gdn_chunk_gemm_wmma_kernel<2>)
                      : reinterpret_cast<const void*>(gdn_chunk_gemm_wmma_kernel<1>);
    static const void* gemm_opted = nullptr;
    if (gemm_opted != entry) {
      const cudaError_t attr = cudaFuncSetAttribute(
          entry, cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem));
      if (attr != cudaSuccess) {
        return attr;
      }
      gemm_opted = entry;
    }
#define GDN_GEMM_WMMA_ARGS                                                     \
  static_cast<const float*>(a), static_cast<const float*>(v),                  \
      static_cast<const float*>(k), static_cast<const float*>(beta),           \
      static_cast<const float*>(g_cum), static_cast<float*>(vt_out),           \
      static_cast<float*>(kcd_out), seq_pad
    if (passes == 3) {
      gdn_chunk_gemm_wmma_kernel<3>
          <<<dim3(chunks, num_v_heads), 256, smem, stream>>>(GDN_GEMM_WMMA_ARGS);
    } else if (passes == 2) {
      gdn_chunk_gemm_wmma_kernel<2>
          <<<dim3(chunks, num_v_heads), 256, smem, stream>>>(GDN_GEMM_WMMA_ARGS);
    } else {
      gdn_chunk_gemm_wmma_kernel<1>
          <<<dim3(chunks, num_v_heads), 256, smem, stream>>>(GDN_GEMM_WMMA_ARGS);
    }
#undef GDN_GEMM_WMMA_ARGS
    return cudaGetLastError();
  }
  switch (policy->chunk_gemm_tile) {
    case 1: return launch_chunk_gemm<1>(CHUNK_GEMM_ARGS);
    case 2: return launch_chunk_gemm<2>(CHUNK_GEMM_ARGS);
    case 4: return launch_chunk_gemm<4>(CHUNK_GEMM_ARGS);
    case 8: return launch_chunk_gemm<8>(CHUNK_GEMM_ARGS);
    case 32: return launch_chunk_gemm<32>(CHUNK_GEMM_ARGS);
    default: return launch_chunk_gemm<16>(CHUNK_GEMM_ARGS);
  }
}



#define CHUNK_STATE_ARGS                                                      \
  num_v_heads, v_split, block_threads, smem, stream, q, k, g_cum, t_in,       \
      vt_in, kcd_in, state, out, seq, seq_pad, head_k_dim, head_v_dim,        \
      chunk_size, total_chunks, out_row_width

template <int TILE, bool V_SPLIT>
static cudaError_t launch_chunk_state(
    int num_v_heads, int v_split, int block_threads, size_t smem,
    cudaStream_t stream,
    const void* q, const void* k, const void* g_cum, const void* t_in,
    const void* vt_in, const void* kcd_in, void* state, void* out,
    int seq, int seq_pad, int head_k_dim, int head_v_dim, int chunk_size,
    int total_chunks, int out_row_width) {
  if (smem > 48u * 1024u) {
    // Per instantiation, and each instantiation is its own function.
    static bool opted_in = false;
    if (!opted_in) {
      const cudaError_t attr = cudaFuncSetAttribute(
          reinterpret_cast<const void*>(gdn_chunk_state_kernel<TILE, V_SPLIT>),
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem));
      if (attr != cudaSuccess) {
        return attr;
      }
      opted_in = true;
    }
  }
  gdn_chunk_state_kernel<TILE, V_SPLIT>
      <<<dim3(num_v_heads, v_split), block_threads, smem, stream>>>(
          static_cast<const float*>(q), static_cast<const float*>(k),
          static_cast<const float*>(g_cum), static_cast<const float*>(t_in),
          static_cast<const float*>(vt_in), static_cast<const float*>(kcd_in),
          static_cast<float*>(state), static_cast<__nv_bfloat16*>(out),
          seq, seq_pad, head_k_dim, head_v_dim, chunk_size, total_chunks,
          out_row_width,
          static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_k_dim))),
          v_split);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gdn_chunk_state_f32(
    const void* q, const void* k, const void* g_cum, const void* t_in,
    const void* vt_in, const void* kcd_in, void* state, void* out,
    int seq, int seq_pad, int num_v_heads, int head_k_dim, int head_v_dim,
    int chunk_size, int total_chunks, int out_row_width,
    const ApxinfGdnPolicy* policy, cudaStream_t stream) {
  if (policy == nullptr) return cudaErrorInvalidValue;
  if (q == nullptr || k == nullptr || g_cum == nullptr || t_in == nullptr ||
      vt_in == nullptr || kcd_in == nullptr || state == nullptr ||
      out == nullptr || seq <= 0 || seq_pad < seq || num_v_heads <= 0 ||
      head_k_dim <= 0 || head_v_dim <= 0 || chunk_size <= 0 ||
      total_chunks <= 0 || seq_pad != total_chunks * chunk_size ||
      out_row_width != num_v_heads * head_v_dim) {
    return cudaErrorInvalidValue;
  }
  // The chunk loop is sequential because the state is recurrent, so the only
  // parallelism is inside a chunk: chunk_size*head_v_dim cells against one
  // block per head. At the shipped shape that is 32 blocks of 256 threads on
  // 16 SMs, a third of the threads an SM can hold. A wider block raises
  // occupancy without touching the arithmetic -- each cell is independent and
  // keeps its own accumulation order. Must stay a multiple of head_v_dim so a
  // thread's column index remains fixed (see the kernel comment), and no
  // larger than the per-thread attn_inter budget allows.
  //
  // A head's scan may also be split across v_split blocks along the value
  // dimension, which changes no arithmetic (see the kernel) but multiplies the
  // grid. The policy asks for a width from the device's multiprocessor count;
  // the shape decides what it can have, and a slice that would leave partial
  // warps of columns falls back towards one block per head.
  int v_split = policy->chunk_state_v_split > 0 ? policy->chunk_state_v_split : 1;
  while (v_split > 1 &&
         (head_v_dim % v_split != 0 || (head_v_dim / v_split) % 32 != 0)) {
    v_split /= 2;
  }
  const int v_cols = head_v_dim / v_split;
  const int cells_per_chunk = chunk_size * v_cols;
  auto block_ok = [&](int threads) {
    return threads >= 32 && threads <= 1024 && threads % 32 == 0 &&
           v_cols % 32 == 0 &&
           cells_per_chunk % threads == 0 && cells_per_chunk / threads <= 32 &&
           // Keep the tiled path reachable: with a trip count the accumulator
           // array is indexed dynamically and nvcc spills it, which costs more
           // than the wider block wins. The tiles run along the columns, so
           // the slice has to hold a whole number of them.
           v_cols % policy->chunk_state_tile == 0 &&
           cells_per_chunk % (threads * policy->chunk_state_tile) == 0;
  };
  // 1024. That is not what an earlier sweep found -- on the CUDA 12.6 board,
  // before this kernel was tiled, 512 won at 33.27ms per layer against 36.35ms
  // at 1024, and the reasoning was that two 1024-thread blocks exceed the 1536
  // threads an SM holds. The tiling changed which side of that trade wins: with
  // eight cells carried per thread there is enough work in flight for one block
  // per SM to keep the pipes busy, and the wider block cuts the per-block
  // prologue. Re-measured on orin2, four interleaved pairs, every 1024 sample
  // faster than every 512 sample: 5.9319 s/scene mean against 5.9564, 0.41%.
  //
  // Re-sweep after changing this kernel; the optimum has moved once already
  // and will again. The width comes from the policy table, and anything the
  // shape cannot take falls back rather than launching something invalid --
  // 1024 exceeds the register budget for this kernel on some boards.
  int block_threads = policy->chunk_state_threads;
  if (!block_ok(block_threads)) {
    block_threads = 256;
    for (const int candidate : {1024, 512, 256, 128, 64, 32}) {
      if (block_ok(candidate)) {
        block_threads = candidate;
        break;
      }
    }
  }
  // v_new, a BF16 copy of the carried state, and a BF16 tile holding v_new's
  // round trip for the chunk (see the kernel comment). 32KB + 32KB + 16KB at
  // the shipped shape, so two blocks still fit in an SM's 164KB.
  const size_t smem =
      static_cast<size_t>(chunk_size) * v_cols * sizeof(float) +
      static_cast<size_t>(head_k_dim) * v_cols * sizeof(__nv_bfloat16) +
      static_cast<size_t>(chunk_size) * v_cols * sizeof(__nv_bfloat16);
  // At the shipped shape this lands at 80KB, past the 48KB a kernel receives
  // without asking. Opt in once per instantiation; a device that refuses keeps
  // the error rather than launching with too little shared memory.
  // Tensor-core form, for the shipped shape only.
  //   1 / on / true -> split left operand, two BF16 passes, fp32-class error
  //   lossy         -> one pass, left operand rounded to BF16
  //   unset         -> the scalar kernel
  // Default on where the tensor cores are worth this much: on Thor the split
  // form is 1.4623 -> 1.1909 s of fixed cost at the same error.
  const bool wmma_split = policy->chunk_state_wmma >= APXINF_GDN_WMMA_SPLIT2;
  if (policy->chunk_state_wmma != APXINF_GDN_WMMA_OFF && head_k_dim == 128 &&
      head_v_dim == 128 && chunk_size == 64) {
    const size_t wmma_smem =
        static_cast<size_t>(128 * 128) * sizeof(__nv_bfloat16) +  // state
        static_cast<size_t>(64) * 1024 +                          // left operands
        static_cast<size_t>(64 * 128) * sizeof(__nv_bfloat16) +   // v_round
        static_cast<size_t>(64 * 128) * sizeof(float) * 2;        // v_new, inter/out
    const void* entry =
        wmma_split
            ? reinterpret_cast<const void*>(gdn_chunk_state_wmma_kernel<true>)
            : reinterpret_cast<const void*>(gdn_chunk_state_wmma_kernel<false>);
    static const void* opted_in = nullptr;
    if (opted_in != entry) {
      const cudaError_t attr = cudaFuncSetAttribute(
          entry, cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(wmma_smem));
      if (attr != cudaSuccess) {
        return attr;
      }
      opted_in = entry;
    }
    const float chunk_scale =
        static_cast<float>(1.0 / std::sqrt(static_cast<double>(head_k_dim)));
#define GDN_WMMA_ARGS                                                          \
  static_cast<const float*>(q), static_cast<const float*>(k),                  \
      static_cast<const float*>(g_cum), static_cast<const float*>(t_in),       \
      static_cast<const float*>(vt_in), static_cast<const float*>(kcd_in),     \
      static_cast<float*>(state), static_cast<__nv_bfloat16*>(out), seq,       \
      seq_pad, total_chunks, out_row_width, chunk_scale
    if (wmma_split) {
      gdn_chunk_state_wmma_kernel<true>
          <<<num_v_heads, 1024, wmma_smem, stream>>>(GDN_WMMA_ARGS);
    } else {
      gdn_chunk_state_wmma_kernel<false>
          <<<num_v_heads, 1024, wmma_smem, stream>>>(GDN_WMMA_ARGS);
    }
#undef GDN_WMMA_ARGS
    return cudaGetLastError();
  }

  const int tile = policy->chunk_state_tile;
  cudaError_t launched = cudaErrorInvalidValue;
  // One block per head compiles to the kernel as it was before the split
  // existed; see the V_SPLIT comment in the kernel for why that matters.
  if (v_split == 1) {
    switch (tile) {
      case 1: launched = launch_chunk_state<1, false>(CHUNK_STATE_ARGS); break;
      case 2: launched = launch_chunk_state<2, false>(CHUNK_STATE_ARGS); break;
      case 4: launched = launch_chunk_state<4, false>(CHUNK_STATE_ARGS); break;
      case 8: launched = launch_chunk_state<8, false>(CHUNK_STATE_ARGS); break;
      default: launched = launch_chunk_state<16, false>(CHUNK_STATE_ARGS); break;
    }
  } else {
    switch (tile) {
      case 1: launched = launch_chunk_state<1, true>(CHUNK_STATE_ARGS); break;
      case 2: launched = launch_chunk_state<2, true>(CHUNK_STATE_ARGS); break;
      case 4: launched = launch_chunk_state<4, true>(CHUNK_STATE_ARGS); break;
      case 8: launched = launch_chunk_state<8, true>(CHUNK_STATE_ARGS); break;
      default: launched = launch_chunk_state<16, true>(CHUNK_STATE_ARGS); break;
    }
  }
  return launched;
}


extern "C" cudaError_t apxinf_static_gdn_recurrent_f32(
    const void* q, const void* k, const void* v, const void* beta,
    const void* g, void* state, void* out,
    int num_v_heads, int head_k_dim, int head_v_dim,
    const ApxinfGdnPolicy* policy, cudaStream_t stream) {
  if (policy == nullptr) return cudaErrorInvalidValue;
  if (q == nullptr || k == nullptr || v == nullptr || beta == nullptr ||
      g == nullptr || state == nullptr || out == nullptr ||
      num_v_heads <= 0 || head_k_dim <= 0 || head_v_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  // Split width for the decode recurrence. The scalar kernel is SPLIT=1 and
  // stays available for any shape the split form is not instantiated for, and
  // for A/B on one binary: APXINF_GDN_RECURRENT_SPLIT=1 selects it.
  // 8 would ask for 1024 threads each holding 16 registers of state and the
  // launcher rejects it with "too many resources requested", so the policy
  // does not offer it.
  const int split = policy->recurrent_split;
  if (split > 1 && head_k_dim == 128) {
    const size_t smem =
        (static_cast<size_t>(2 * head_k_dim) +
         static_cast<size_t>(head_v_dim) * split) * sizeof(float);
    const dim3 block(static_cast<unsigned>(head_v_dim * split));
    if (block.x > 1024u) return cudaErrorInvalidValue;
#define GDN_SPLIT_ARGS                                                        \
  static_cast<const float*>(q), static_cast<const float*>(k),                 \
      static_cast<const float*>(v), static_cast<const float*>(beta),          \
      static_cast<const float*>(g), static_cast<float*>(state),               \
      static_cast<__nv_bfloat16*>(out), head_k_dim, head_v_dim
    switch (split) {
      case 2:
        gdn_recurrent_split_kernel<2, 128><<<num_v_heads, block, smem, stream>>>(GDN_SPLIT_ARGS);
        return cudaGetLastError();
      case 4:
        gdn_recurrent_split_kernel<4, 128><<<num_v_heads, block, smem, stream>>>(GDN_SPLIT_ARGS);
        return cudaGetLastError();
      default:
        break;
    }
#undef GDN_SPLIT_ARGS
  }
  const size_t smem = static_cast<size_t>(2 * head_k_dim) * sizeof(float);
  gdn_recurrent_kernel<<<num_v_heads, head_v_dim, smem, stream>>>(
      static_cast<const float*>(q), static_cast<const float*>(k),
      static_cast<const float*>(v), static_cast<const float*>(beta),
      static_cast<const float*>(g), static_cast<float*>(state),
      static_cast<__nv_bfloat16*>(out), head_k_dim, head_v_dim);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gated_rms_silu_bf16(
    const void* x, const void* z, const void* weight, void* out,
    int rows, int cols, int z_heads, int64_t z_row_stride,
    int64_t z_col_offset, float eps, cudaStream_t stream) {
  if (x == nullptr || z == nullptr || weight == nullptr || out == nullptr ||
      rows <= 0 || cols <= 0 || z_heads <= 0 || rows % z_heads != 0 ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  if (cols == 128) {
    gated_rms_silu_bf16_rowfit_kernel<<<rows, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(z),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(out),
        cols, z_heads, z_row_stride, z_col_offset, eps);
    return cudaGetLastError();
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  gated_rms_silu_bf16_kernel<<<rows, 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(z),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(out),
      cols, z_heads, z_row_stride, z_col_offset, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_rms_norm_plus1_bf16(
    const void* input, const void* weight, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  rms_norm_plus1_bf16_kernel<<<rows, 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_add_rms_norm_plus1_bf16(
    const void* a, const void* b, const void* weight, void* sum_out,
    void* output, int rows, int cols, float eps, cudaStream_t stream) {
  if (a == nullptr || b == nullptr || weight == nullptr || sum_out == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  add_rms_norm_plus1_bf16_kernel<<<rows, 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(sum_out),
      static_cast<__nv_bfloat16*>(output), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_full_attn_prepare_bf16(
    const void* fused, const void* q_norm_w, const void* k_norm_w,
    const void* cos, const void* sin, void* q_out, void* k_cache, void* v_cache,
    int seq, int cache_offset, int q_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t fused_width, int64_t cache_width, float eps,
    cudaStream_t stream) {
  if (fused == nullptr || q_norm_w == nullptr || k_norm_w == nullptr ||
      cos == nullptr || sin == nullptr || q_out == nullptr ||
      k_cache == nullptr || v_cache == nullptr || seq <= 0 ||
      cache_offset < 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
  full_attn_prepare_bf16_kernel
      <<<dim3(seq, q_heads + 2 * kv_heads), 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<const __nv_bfloat16*>(q_norm_w),
      static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<const __nv_bfloat16*>(cos),
      static_cast<const __nv_bfloat16*>(sin),
      static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(k_cache),
      static_cast<__nv_bfloat16*>(v_cache),
      cache_offset, q_heads, kv_heads, head_dim, rotary_dim, fused_width,
      cache_width, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_sigmoid_gate_mul_bf16(
    void* attn, const void* fused, int rows, int heads, int head_dim,
    int64_t fused_width, cudaStream_t stream) {
  if (attn == nullptr || fused == nullptr || rows <= 0 || heads <= 0 ||
      head_dim <= 0) {
    return cudaErrorInvalidValue;
  }
  sigmoid_gate_mul_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(attn),
      static_cast<const __nv_bfloat16*>(fused), heads, head_dim, fused_width);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_adaln_rms_norm_bf16(
    const void* x, const void* weight, const void* scale, const void* shift,
    void* out, int rows, int cols, float eps, cudaStream_t stream) {
  if (x == nullptr || weight == nullptr || scale == nullptr ||
      shift == nullptr || out == nullptr || rows <= 0 || cols <= 0 ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  adaln_rms_norm_bf16_kernel<<<rows, 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(scale),
      static_cast<const __nv_bfloat16*>(shift),
      static_cast<__nv_bfloat16*>(out), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_adaln_gate_residual_bf16(
    const void* proj, const void* residual, const void* gate, void* out,
    int64_t count, int cols, cudaStream_t stream) {
  if (proj == nullptr || residual == nullptr || gate == nullptr ||
      out == nullptr || count <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  adaln_gate_residual_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(proj),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<__nv_bfloat16*>(out), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_expert_qkv_prepare_bf16(
    const void* fused, const void* q_norm_w, const void* k_norm_w,
    const void* cos, const void* sin, void* q_out, void* gate_out, void* k_out,
    void* v_out, int seq, int q_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t fused_width, float eps, cudaStream_t stream) {
  if (fused == nullptr || q_norm_w == nullptr || k_norm_w == nullptr ||
      cos == nullptr || sin == nullptr || q_out == nullptr ||
      gate_out == nullptr || k_out == nullptr || v_out == nullptr ||
      seq <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim ||
      q_heads % kv_heads != 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
  expert_qkv_prepare_bf16_kernel
      <<<dim3(seq, q_heads + 2 * kv_heads), 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<const __nv_bfloat16*>(q_norm_w),
      static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<const __nv_bfloat16*>(cos),
      static_cast<const __nv_bfloat16*>(sin),
      static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(gate_out),
      static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out),
      q_heads, kv_heads, head_dim, rotary_dim, fused_width, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_expert_sigmoid_gate_mul_bf16(
    void* attn, const void* gate, int64_t count, cudaStream_t stream) {
  if (attn == nullptr || gate == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  expert_sigmoid_gate_mul_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<__nv_bfloat16*>(attn),
      static_cast<const __nv_bfloat16*>(gate), count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_fourier_features_bf16(
    const void* waypoints, const void* freqs, void* out,
    int rows, int point_dim, int num_features, cudaStream_t stream) {
  if (waypoints == nullptr || freqs == nullptr || out == nullptr ||
      rows <= 0 || point_dim <= 0 || num_features <= 0) {
    return cudaErrorInvalidValue;
  }
  fourier_features_bf16_kernel<<<rows, 64, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(waypoints),
      static_cast<const __nv_bfloat16*>(freqs),
      static_cast<__nv_bfloat16*>(out), point_dim, num_features);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_concat7_cols_bf16(
    const void* s0, const void* s1, const void* s2, const void* s3,
    const void* s4, const void* s5, const void* s6, void* dst,
    int rows, int cols, int broadcast_mask, cudaStream_t stream) {
  if (s0 == nullptr || s1 == nullptr || s2 == nullptr || s3 == nullptr ||
      s4 == nullptr || s5 == nullptr || s6 == nullptr || dst == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  concat7_cols_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(s0),
      static_cast<const __nv_bfloat16*>(s1),
      static_cast<const __nv_bfloat16*>(s2),
      static_cast<const __nv_bfloat16*>(s3),
      static_cast<const __nv_bfloat16*>(s4),
      static_cast<const __nv_bfloat16*>(s5),
      static_cast<const __nv_bfloat16*>(s6),
      static_cast<__nv_bfloat16*>(dst), rows, cols, broadcast_mask);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_flow_update_f32(
    void* w, const void* endpoint, float remaining, float step, int64_t count,
    cudaStream_t stream) {
  if (w == nullptr || endpoint == nullptr || count <= 0 ||
      !(remaining > 0.0f) || !std::isfinite(step)) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  flow_update_f32_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<float*>(w), static_cast<const float*>(endpoint),
      remaining, step, count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_suppress_logits_bf16(
    void* logits, const uint32_t* ids, int count, cudaStream_t stream) {
  if (logits == nullptr || ids == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  const int blocks = (count + 63) / 64;
  suppress_logits_bf16_kernel<<<blocks, 64, 0, stream>>>(
      static_cast<__nv_bfloat16*>(logits), ids, count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_static_gelu_exact_bf16(
    const void* input, void* output, int64_t count, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  gelu_exact_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError();
}

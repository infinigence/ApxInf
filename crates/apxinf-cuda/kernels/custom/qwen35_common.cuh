#pragma once

// Qwen3.5 decoder RMSNorm uses gamma=(1+checkpoint_weight), unlike the
// ordinary Llama-style weight contract.  Keep that numerical order explicit
// instead of pre-rounding shifted weights during model load.
__global__ void qwen35_rmsnorm_offset_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output, int columns, float epsilon) {
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * columns;
  input += row_offset;
  output += row_offset;
  const int thread = threadIdx.x;
  float partial = 0.0f;
  for (int column = thread; column < columns; column += blockDim.x) {
    const float value = __bfloat162float(input[column]);
    partial += value * value;
  }
  __shared__ float scratch[8];
  const float inverse = rsqrtf(block_sum(partial, scratch) / columns + epsilon);
  for (int column = thread; column < columns; column += blockDim.x) {
    const float value = __bfloat162float(input[column]);
    const float gamma = 1.0f + __bfloat162float(weight[column]);
    output[column] = __float2bfloat16(value * inverse * gamma);
  }
}

// Fuses the two decoder residual edges with the next RMSNorm.  The updated
// residual is written in place and the normalized value is written to stable,
// caller-owned storage for the following mixer/MLP.
__global__ void qwen35_residual_add_rmsnorm_offset_bf16_kernel(
    __nv_bfloat16* residual, const __nv_bfloat16* delta,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    int columns, float epsilon) {
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * columns;
  residual += row_offset;
  delta += row_offset;
  output += row_offset;
  extern __shared__ float updated[];
  const int thread = threadIdx.x;
  float partial = 0.0f;
  for (int column = thread; column < columns; column += blockDim.x) {
    // PyTorch materializes the BF16 residual add before the following
    // RMSNorm.  Preserve that seam instead of normalizing the unrounded FP32
    // sum inside the fusion.
    const __nv_bfloat16 rounded = __float2bfloat16(
        __bfloat162float(residual[column]) + __bfloat162float(delta[column]));
    residual[column] = rounded;
    const float value = __bfloat162float(rounded);
    updated[column] = value;
    partial += value * value;
  }
  __shared__ float scratch[8];
  const float inverse = rsqrtf(block_sum(partial, scratch) / columns + epsilon);
  for (int column = thread; column < columns; column += blockDim.x) {
    const float value = updated[column];
    const float gamma = 1.0f + __bfloat162float(weight[column]);
    output[column] = __float2bfloat16(value * inverse * gamma);
  }
}

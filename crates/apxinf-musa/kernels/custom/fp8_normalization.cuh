__device__ __forceinline__ __mt_fp8_e4m3 norm_e4m3(float value, float inverse_scale) {
  const float rounded = __bfloat162float(__float2bfloat16(value));
  return __mt_fp8_e4m3(fminf(448.0f, fmaxf(-448.0f, rounded * inverse_scale)));
}

template <bool ADAPTIVE>
__global__ void layer_norm_e4m3_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* weight,
    const __mt_bfloat16* bias, __mt_fp8_e4m3* output,
    int cols, float eps, float inverse_scale) {
  __shared__ float scratch[8];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[base + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = __bfloat162float(input[base + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std = rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float normalized = (__bfloat162float(input[base + col]) - mean) * inverse_std;
    const float factor = __bfloat162float(weight[col]);
    const float shift = __bfloat162float(bias[col]);
    const float value = ADAPTIVE ? normalized * (1.0f + factor) + shift
                                : normalized * factor + shift;
    output[base + col] = norm_e4m3(value, inverse_scale);
  }
}

__global__ void rms_norm_e4m3_2048_kernel(
    const __mt_bfloat16* input, const __mt_bfloat16* weight,
    __mt_fp8_e4m3* output, float eps, float inverse_scale) {
  __shared__ float scratch[8];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * 2048 + threadIdx.x * 8;
  const Fp8SourcePack source = *reinterpret_cast<const Fp8SourcePack*>(input + base);
  float values[8];
  float square_sum = 0.0f;
#pragma unroll
  for (int lane = 0; lane < 8; ++lane) {
    values[lane] = __bfloat162float(source.values[lane]);
    square_sum += values[lane] * values[lane];
  }
  const float mean = block_sum_parallel_unsafe(square_sum, scratch) / 2048;
  const float inverse_rms = rsqrtf(__fadd_rn(mean, eps));
  const Fp8SourcePack weights = *reinterpret_cast<const Fp8SourcePack*>(weight + threadIdx.x * 8);
  Fp8OutputPack destination;
#pragma unroll
  for (int lane = 0; lane < 8; ++lane)
    destination.values[lane] = norm_e4m3(values[lane] * inverse_rms *
        __bfloat162float(weights.values[lane]), inverse_scale);
  *reinterpret_cast<Fp8OutputPack*>(output + base) = destination;
}

template <int COLS, bool ADAPTIVE, bool ROUND_BIAS>
__global__ void residual_layer_norm_e4m3_kernel(
    const __mt_bfloat16* projection, const __mt_bfloat16* projection_bias,
    const __mt_bfloat16* residual, const __mt_bfloat16* weight,
    const __mt_bfloat16* bias, __mt_bfloat16* hidden, __mt_fp8_e4m3* output,
    float eps, float inverse_scale) {
  __shared__ float scratch[8];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * COLS;
  constexpr int kValues = COLS / 256;
  float cache[kValues];
#pragma unroll
  for (int index = 0; index < kValues; ++index) {
    const int col = threadIdx.x + index * 256;
    float value;
    if (ROUND_BIAS) {
      const __mt_bfloat16 biased = __float2bfloat16(__bfloat162float(projection[base + col]) +
          __bfloat162float(projection_bias[col]));
      value = __bfloat162float(biased) + __bfloat162float(residual[base + col]);
    } else {
      value = __bfloat162float(projection[base + col]) + __bfloat162float(residual[base + col]);
      if (projection_bias != nullptr) value += __bfloat162float(projection_bias[col]);
    }
    const __mt_bfloat16 rounded = __float2bfloat16(value);
    hidden[base + col] = rounded;
    cache[index] = __bfloat162float(rounded);
  }
  float sum = 0.0f;
#pragma unroll
  for (int index = 0; index < kValues; ++index) sum += cache[index];
  const float mean = block_sum_parallel_unsafe(sum, scratch) / float(COLS);
  float variance_sum = 0.0f;
#pragma unroll
  for (int index = 0; index < kValues; ++index) {
    const float centered = cache[index] - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std = rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / float(COLS) + eps);
#pragma unroll
  for (int index = 0; index < kValues; ++index) {
    const int col = threadIdx.x + index * 256;
    const float normalized = (cache[index] - mean) * inverse_std;
    const float factor = __bfloat162float(weight[col]);
    const float shift = __bfloat162float(bias[col]);
    const float value = ADAPTIVE ? normalized * (1.0f + factor) + shift
                                : normalized * factor + shift;
    output[base + col] = norm_e4m3(value, inverse_scale);
  }
}

#pragma once

// Qwen3.5 recurrent gated-delta core for one decode token.
// Fixed head dimensions are part of the fail-closed host contract.

__global__ void qwen35_gdn_recurrent_bf16_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const float* g, const float* beta,
    float* recurrent_state, __nv_bfloat16* output) {
  constexpr int kDim = 128;
  constexpr float kEpsilon = 1.0e-6f;
  constexpr float kQueryScale = 0.08838834764831845f;  // 1/sqrt(128)
  __shared__ float scratch[8];
  __shared__ float normalized_query[kDim];
  __shared__ float normalized_key[kDim];

  const int head = blockIdx.x;
  const int dimension = threadIdx.x;
  const int64_t vector_offset = static_cast<int64_t>(head) * kDim;
  const float query_value =
      __bfloat162float(query[vector_offset + dimension]);
  const float key_value = __bfloat162float(key[vector_offset + dimension]);
  const float query_sum = block_sum(query_value * query_value, scratch);
  const float key_sum = block_sum(key_value * key_value, scratch);
  const float query_normalizer = rsqrtf(query_sum + kEpsilon) * kQueryScale;
  const float key_normalizer = rsqrtf(key_sum + kEpsilon);
  normalized_query[dimension] = query_value * query_normalizer;
  normalized_key[dimension] = key_value * key_normalizer;
  __syncthreads();

  const float qk = block_sum(
      normalized_query[dimension] * normalized_key[dimension], scratch);
  const float decay = expf(g[head]);
  const float beta_value = beta[head];
  const int value_dimension = dimension;
  const int64_t state_base = static_cast<int64_t>(head) * kDim * kDim;
  float key_memory = 0.0f;
  float query_memory = 0.0f;
#pragma unroll 4
  for (int key_dimension = 0; key_dimension < kDim; ++key_dimension) {
    const float state = recurrent_state[
        state_base + static_cast<int64_t>(key_dimension) * kDim +
        value_dimension];
    key_memory = fmaf(state, normalized_key[key_dimension], key_memory);
    query_memory = fmaf(state, normalized_query[key_dimension], query_memory);
  }
  const float value_value =
      __bfloat162float(value[vector_offset + value_dimension]);
  const float delta =
      (value_value - decay * key_memory) * beta_value;
  const float output_value = decay * query_memory + delta * qk;

#pragma unroll 4
  for (int key_dimension = 0; key_dimension < kDim; ++key_dimension) {
    const int64_t index =
        state_base + static_cast<int64_t>(key_dimension) * kDim +
        value_dimension;
    recurrent_state[index] = fmaf(
        normalized_key[key_dimension], delta,
        recurrent_state[index] * decay);
  }
  output[vector_offset + value_dimension] =
      __float2bfloat16(output_value);
}

__global__ void qwen35_gdn_conv4_silu_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* conv_state, __nv_bfloat16* output, int channels) {
  int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) return;
  const int64_t offset = static_cast<int64_t>(channel) * 4;
  const __nv_bfloat16 state0 = conv_state[offset + 1];
  const __nv_bfloat16 state1 = conv_state[offset + 2];
  const __nv_bfloat16 state2 = conv_state[offset + 3];
  const __nv_bfloat16 state3 = input[channel];
  conv_state[offset] = state0;
  conv_state[offset + 1] = state1;
  conv_state[offset + 2] = state2;
  conv_state[offset + 3] = state3;
  float sum = 0.0f;
  sum = fmaf(__bfloat162float(state0), __bfloat162float(weight[offset]), sum);
  sum = fmaf(__bfloat162float(state1), __bfloat162float(weight[offset + 1]), sum);
  sum = fmaf(__bfloat162float(state2), __bfloat162float(weight[offset + 2]), sum);
  sum = fmaf(__bfloat162float(state3), __bfloat162float(weight[offset + 3]), sum);
  output[channel] = __float2bfloat16(sum / (1.0f + expf(-sum)));
}

__global__ void qwen35_gdn_prepare_bf16_kernel(
    const __nv_bfloat16* convolved_qkv, const __nv_bfloat16* a,
    const __nv_bfloat16* b, const __nv_bfloat16* a_log,
    const __nv_bfloat16* dt_bias, __nv_bfloat16* query,
    __nv_bfloat16* key, __nv_bfloat16* value, float* g, float* beta) {
  constexpr int kKeyHeads = 16;
  constexpr int kValueHeads = 48;
  constexpr int kDim = 128;
  constexpr int kKeyWidth = kKeyHeads * kDim;
  const int head = blockIdx.x;
  const int dimension = threadIdx.x;
  const int source_head = head / (kValueHeads / kKeyHeads);
  const int64_t output_offset = static_cast<int64_t>(head) * kDim + dimension;
  query[output_offset] = convolved_qkv[source_head * kDim + dimension];
  key[output_offset] = convolved_qkv[kKeyWidth + source_head * kDim + dimension];
  value[output_offset] = convolved_qkv[2 * kKeyWidth + head * kDim + dimension];
  if (dimension == 0) {
    const float a_value = __bfloat162float(a[head]);
    const float b_value = __bfloat162float(b[head]);
    const float dt = a_value + __bfloat162float(dt_bias[head]);
    const float softplus = dt > 20.0f ? dt : log1pf(expf(dt));
    g[head] = -expf(__bfloat162float(a_log[head])) * softplus;
    beta[head] = 1.0f / (1.0f + expf(-b_value));
  }
}

__global__ void qwen35_gdn_gated_rmsnorm_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output, float epsilon) {
  constexpr int kDim = 128;
  __shared__ float scratch[8];
  const int head = blockIdx.x;
  const int dimension = threadIdx.x;
  const int64_t offset = static_cast<int64_t>(head) * kDim + dimension;
  const float value = __bfloat162float(input[offset]);
  const float square_sum = block_sum(value * value, scratch);
  const float inverse_rms = rsqrtf(square_sum / kDim + epsilon);
  const __nv_bfloat16 normalized = __float2bfloat16(value * inverse_rms);
  const __nv_bfloat16 weighted = __float2bfloat16(
      __bfloat162float(normalized) * __bfloat162float(weight[dimension]));
  const float gate_value = __bfloat162float(gate[offset]);
  const float silu_gate = gate_value / (1.0f + expf(-gate_value));
  output[offset] = __float2bfloat16(__bfloat162float(weighted) * silu_gate);
}

__global__ void qwen35_gdn_conv4_prepare_bf16_kernel(
    const __nv_bfloat16* projected_qkv, const __nv_bfloat16* conv_weight,
    __nv_bfloat16* conv_state, const __nv_bfloat16* projected_ab,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    __nv_bfloat16* a_output, __nv_bfloat16* b_output,
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    float* g, float* beta) {
  constexpr int kKeyHeads = 16;
  constexpr int kValueHeads = 48;
  constexpr int kDim = 128;
  constexpr int kKeyWidth = kKeyHeads * kDim;
  constexpr int kChannels = 2 * kKeyWidth + kValueHeads * kDim;
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel < kChannels) {
    const int64_t state_offset = static_cast<int64_t>(channel) * 4;
    const __nv_bfloat16 state0 = conv_state[state_offset + 1];
    const __nv_bfloat16 state1 = conv_state[state_offset + 2];
    const __nv_bfloat16 state2 = conv_state[state_offset + 3];
    const __nv_bfloat16 state3 = projected_qkv[channel];
    conv_state[state_offset] = state0;
    conv_state[state_offset + 1] = state1;
    conv_state[state_offset + 2] = state2;
    conv_state[state_offset + 3] = state3;
    float sum = 0.0f;
    sum = fmaf(__bfloat162float(state0),
               __bfloat162float(conv_weight[state_offset]), sum);
    sum = fmaf(__bfloat162float(state1),
               __bfloat162float(conv_weight[state_offset + 1]), sum);
    sum = fmaf(__bfloat162float(state2),
               __bfloat162float(conv_weight[state_offset + 2]), sum);
    sum = fmaf(__bfloat162float(state3),
               __bfloat162float(conv_weight[state_offset + 3]), sum);
    const __nv_bfloat16 convolved =
        __float2bfloat16(sum / (1.0f + expf(-sum)));
    if (channel < kKeyWidth) {
      const int source_head = channel / kDim;
      const int dimension = channel % kDim;
#pragma unroll
      for (int repeat = 0; repeat < 3; ++repeat) {
        query[(source_head * 3 + repeat) * kDim + dimension] = convolved;
      }
    } else if (channel < 2 * kKeyWidth) {
      const int local = channel - kKeyWidth;
      const int source_head = local / kDim;
      const int dimension = local % kDim;
#pragma unroll
      for (int repeat = 0; repeat < 3; ++repeat) {
        key[(source_head * 3 + repeat) * kDim + dimension] = convolved;
      }
    } else {
      value[channel - 2 * kKeyWidth] = convolved;
    }
  }
  if (channel < kValueHeads) {
    const __nv_bfloat16 a = projected_ab[channel];
    const __nv_bfloat16 b = projected_ab[kValueHeads + channel];
    a_output[channel] = a;
    b_output[channel] = b;
    const float dt = __bfloat162float(a) + __bfloat162float(dt_bias[channel]);
    const float softplus = dt > 20.0f ? dt : log1pf(expf(dt));
    g[channel] = -expf(__bfloat162float(a_log[channel])) * softplus;
    beta[channel] = 1.0f / (1.0f + expf(-__bfloat162float(b)));
  }
}

__global__ void qwen35_gdn_conv4_prepare_m8_bf16_kernel(
    const __nv_bfloat16* projected_qkv, const __nv_bfloat16* conv_weight,
    __nv_bfloat16* conv_state, const __nv_bfloat16* projected_ab,
    const __nv_bfloat16* a_log, const __nv_bfloat16* dt_bias,
    __nv_bfloat16* a_output, __nv_bfloat16* b_output,
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    float* g, float* beta, int tokens) {
  constexpr int kKeyHeads = 16;
  constexpr int kValueHeads = 48;
  constexpr int kDim = 128;
  constexpr int kKeyWidth = kKeyHeads * kDim;
  constexpr int kValueWidth = kValueHeads * kDim;
  constexpr int kChannels = 2 * kKeyWidth + kValueWidth;
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel < kChannels) {
    const int64_t state_offset = static_cast<int64_t>(channel) * 4;
    __nv_bfloat16 state0 = conv_state[state_offset];
    __nv_bfloat16 state1 = conv_state[state_offset + 1];
    __nv_bfloat16 state2 = conv_state[state_offset + 2];
    __nv_bfloat16 state3 = conv_state[state_offset + 3];
    for (int token = 0; token < tokens; ++token) {
      state0 = state1;
      state1 = state2;
      state2 = state3;
      state3 = projected_qkv[static_cast<int64_t>(token) * kChannels + channel];
      float sum = 0.0f;
      sum = fmaf(__bfloat162float(state0),
                 __bfloat162float(conv_weight[state_offset]), sum);
      sum = fmaf(__bfloat162float(state1),
                 __bfloat162float(conv_weight[state_offset + 1]), sum);
      sum = fmaf(__bfloat162float(state2),
                 __bfloat162float(conv_weight[state_offset + 2]), sum);
      sum = fmaf(__bfloat162float(state3),
                 __bfloat162float(conv_weight[state_offset + 3]), sum);
      const __nv_bfloat16 convolved =
          __float2bfloat16(sum / (1.0f + expf(-sum)));
      const int64_t token_offset = static_cast<int64_t>(token) * kValueWidth;
      if (channel < kKeyWidth) {
        const int source_head = channel / kDim;
        const int dimension = channel % kDim;
#pragma unroll
        for (int repeat = 0; repeat < 3; ++repeat)
          query[token_offset + (source_head * 3 + repeat) * kDim + dimension] =
              convolved;
      } else if (channel < 2 * kKeyWidth) {
        const int local = channel - kKeyWidth;
        const int source_head = local / kDim;
        const int dimension = local % kDim;
#pragma unroll
        for (int repeat = 0; repeat < 3; ++repeat)
          key[token_offset + (source_head * 3 + repeat) * kDim + dimension] =
              convolved;
      } else {
        value[token_offset + channel - 2 * kKeyWidth] = convolved;
      }
    }
    conv_state[state_offset] = state0;
    conv_state[state_offset + 1] = state1;
    conv_state[state_offset + 2] = state2;
    conv_state[state_offset + 3] = state3;
  }
  if (channel < kValueHeads) {
    for (int token = 0; token < tokens; ++token) {
      const int64_t parameter_offset =
          static_cast<int64_t>(token) * kValueHeads + channel;
      const __nv_bfloat16 a =
          projected_ab[static_cast<int64_t>(token) * 2 * kValueHeads + channel];
      const __nv_bfloat16 b = projected_ab[
          static_cast<int64_t>(token) * 2 * kValueHeads + kValueHeads + channel];
      a_output[parameter_offset] = a;
      b_output[parameter_offset] = b;
      const float dt = __bfloat162float(a) + __bfloat162float(dt_bias[channel]);
      const float softplus = dt > 20.0f ? dt : log1pf(expf(dt));
      g[parameter_offset] = -expf(__bfloat162float(a_log[channel])) * softplus;
      beta[parameter_offset] =
          1.0f / (1.0f + expf(-__bfloat162float(b)));
    }
  }
}

__global__ void qwen35_gdn_recurrent_m8_bf16_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const float* g, const float* beta,
    float* recurrent_state, __nv_bfloat16* output, int tokens) {
  constexpr int kHeads = 48;
  constexpr int kDim = 128;
  constexpr int kWidth = kHeads * kDim;
  constexpr float kEpsilon = 1.0e-6f;
  constexpr float kQueryScale = 0.08838834764831845f;
  __shared__ float scratch[8];
  __shared__ float normalized_query[kDim];
  __shared__ float normalized_key[kDim];
  const int head = blockIdx.x;
  const int dimension = threadIdx.x;
  const int64_t state_base = static_cast<int64_t>(head) * kDim * kDim;
  for (int token = 0; token < tokens; ++token) {
    const int64_t vector_offset =
        static_cast<int64_t>(token) * kWidth + head * kDim;
    const float query_value = __bfloat162float(query[vector_offset + dimension]);
    const float key_value = __bfloat162float(key[vector_offset + dimension]);
    const float query_sum = block_sum(query_value * query_value, scratch);
    const float key_sum = block_sum(key_value * key_value, scratch);
    const float query_normalizer =
        rsqrtf(query_sum + kEpsilon) * kQueryScale;
    const float key_normalizer = rsqrtf(key_sum + kEpsilon);
    normalized_query[dimension] = query_value * query_normalizer;
    normalized_key[dimension] = key_value * key_normalizer;
    __syncthreads();
    const float qk = block_sum(
        normalized_query[dimension] * normalized_key[dimension], scratch);
    const float decay = expf(g[token * kHeads + head]);
    const float beta_value = beta[token * kHeads + head];
    float key_memory = 0.0f;
    float query_memory = 0.0f;
#pragma unroll 4
    for (int key_dimension = 0; key_dimension < kDim; ++key_dimension) {
      const float state = recurrent_state[
          state_base + static_cast<int64_t>(key_dimension) * kDim + dimension];
      key_memory = fmaf(state, normalized_key[key_dimension], key_memory);
      query_memory = fmaf(state, normalized_query[key_dimension], query_memory);
    }
    const float value_value =
        __bfloat162float(value[vector_offset + dimension]);
    const float delta = (value_value - decay * key_memory) * beta_value;
    const float output_value = decay * query_memory + delta * qk;
#pragma unroll 4
    for (int key_dimension = 0; key_dimension < kDim; ++key_dimension) {
      const int64_t index =
          state_base + static_cast<int64_t>(key_dimension) * kDim + dimension;
      recurrent_state[index] = fmaf(
          normalized_key[key_dimension], delta,
          recurrent_state[index] * decay);
    }
    output[vector_offset + dimension] = __float2bfloat16(output_value);
    __syncthreads();
  }
}

__global__ void qwen35_gdn_recurrent_m8_hybrid_bf16_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const float* g, const float* beta,
    float* recurrent_state, __nv_bfloat16* output, int tokens) {
  constexpr int kHeads = 48;
  constexpr int kDim = 128;
  constexpr int kSharedKeyDims = 92;
  constexpr int kRegisterKeyDims = kDim - kSharedKeyDims;
  constexpr int kWidth = kHeads * kDim;
  constexpr float kEpsilon = 1.0e-6f;
  constexpr float kQueryScale = 0.08838834764831845f;
  __shared__ float scratch[8];
  __shared__ float normalized_query[kDim];
  __shared__ float normalized_key[kDim];
  __shared__ float staged_state[kSharedKeyDims * kDim];
  const int head = blockIdx.x;
  const int dimension = threadIdx.x;
  const int64_t state_base = static_cast<int64_t>(head) * kDim * kDim;
#pragma unroll 4
  for (int key_dimension = 0; key_dimension < kSharedKeyDims;
       ++key_dimension) {
    staged_state[key_dimension * kDim + dimension] = recurrent_state[
        state_base + static_cast<int64_t>(key_dimension) * kDim + dimension];
  }
  float resident_tail[kRegisterKeyDims];
#pragma unroll
  for (int tail = 0; tail < kRegisterKeyDims; ++tail) {
    const int key_dimension = kSharedKeyDims + tail;
    resident_tail[tail] = recurrent_state[
        state_base + static_cast<int64_t>(key_dimension) * kDim + dimension];
  }
  __syncthreads();
  for (int token = 0; token < tokens; ++token) {
    const int64_t vector_offset =
        static_cast<int64_t>(token) * kWidth + head * kDim;
    const float query_value = __bfloat162float(query[vector_offset + dimension]);
    const float key_value = __bfloat162float(key[vector_offset + dimension]);
    const float query_sum = block_sum(query_value * query_value, scratch);
    const float key_sum = block_sum(key_value * key_value, scratch);
    const float query_normalizer =
        rsqrtf(query_sum + kEpsilon) * kQueryScale;
    const float key_normalizer = rsqrtf(key_sum + kEpsilon);
    normalized_query[dimension] = query_value * query_normalizer;
    normalized_key[dimension] = key_value * key_normalizer;
    __syncthreads();
    const float qk = block_sum(
        normalized_query[dimension] * normalized_key[dimension], scratch);
    const float decay = expf(g[token * kHeads + head]);
    const float beta_value = beta[token * kHeads + head];
    float key_memory = 0.0f;
    float query_memory = 0.0f;
#pragma unroll 4
    for (int key_dimension = 0; key_dimension < kSharedKeyDims;
         ++key_dimension) {
      const float state = staged_state[key_dimension * kDim + dimension];
      key_memory = fmaf(state, normalized_key[key_dimension], key_memory);
      query_memory = fmaf(state, normalized_query[key_dimension], query_memory);
    }
#pragma unroll
    for (int tail = 0; tail < kRegisterKeyDims; ++tail) {
      const int key_dimension = kSharedKeyDims + tail;
      const float state = resident_tail[tail];
      key_memory = fmaf(state, normalized_key[key_dimension], key_memory);
      query_memory = fmaf(state, normalized_query[key_dimension], query_memory);
    }
    const float value_value =
        __bfloat162float(value[vector_offset + dimension]);
    const float delta = (value_value - decay * key_memory) * beta_value;
    const float output_value = decay * query_memory + delta * qk;
#pragma unroll 4
    for (int key_dimension = 0; key_dimension < kSharedKeyDims;
         ++key_dimension) {
      const int shared_index = key_dimension * kDim + dimension;
      staged_state[shared_index] = fmaf(
          normalized_key[key_dimension], delta,
          staged_state[shared_index] * decay);
    }
#pragma unroll
    for (int tail = 0; tail < kRegisterKeyDims; ++tail) {
      const int key_dimension = kSharedKeyDims + tail;
      resident_tail[tail] = fmaf(
          normalized_key[key_dimension], delta, resident_tail[tail] * decay);
    }
    output[vector_offset + dimension] = __float2bfloat16(output_value);
    __syncthreads();
  }
#pragma unroll 4
  for (int key_dimension = 0; key_dimension < kSharedKeyDims;
       ++key_dimension) {
    recurrent_state[
        state_base + static_cast<int64_t>(key_dimension) * kDim + dimension] =
        staged_state[key_dimension * kDim + dimension];
  }
#pragma unroll
  for (int tail = 0; tail < kRegisterKeyDims; ++tail) {
    const int key_dimension = kSharedKeyDims + tail;
    recurrent_state[
        state_base + static_cast<int64_t>(key_dimension) * kDim + dimension] =
        resident_tail[tail];
  }
}

__global__ void qwen35_gdn_recurrent_m8_hybrid_pairnorm_bf16_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const float* g, const float* beta,
    float* recurrent_state, __nv_bfloat16* output, int tokens) {
  constexpr int kHeads = 48;
  constexpr int kDim = 128;
  constexpr int kSharedKeyDims = 92;
  constexpr int kRegisterKeyDims = kDim - kSharedKeyDims;
  constexpr int kWidth = kHeads * kDim;
  constexpr float kEpsilon = 1.0e-6f;
  constexpr float kQueryScale = 0.08838834764831845f;
  __shared__ float scratch[8];
  __shared__ float normalized_query[kDim];
  __shared__ float normalized_key[kDim];
  __shared__ float staged_state[kSharedKeyDims * kDim];
  const int head = blockIdx.x;
  const int dimension = threadIdx.x;
  const int64_t state_base = static_cast<int64_t>(head) * kDim * kDim;
#pragma unroll 4
  for (int key_dimension = 0; key_dimension < kSharedKeyDims;
       ++key_dimension) {
    staged_state[key_dimension * kDim + dimension] = recurrent_state[
        state_base + static_cast<int64_t>(key_dimension) * kDim + dimension];
  }
  float resident_tail[kRegisterKeyDims];
#pragma unroll
  for (int tail = 0; tail < kRegisterKeyDims; ++tail) {
    const int key_dimension = kSharedKeyDims + tail;
    resident_tail[tail] = recurrent_state[
        state_base + static_cast<int64_t>(key_dimension) * kDim + dimension];
  }
  __syncthreads();
  for (int token = 0; token < tokens; ++token) {
    const int64_t vector_offset =
        static_cast<int64_t>(token) * kWidth + head * kDim;
    const float query_value = __bfloat162float(query[vector_offset + dimension]);
    const float key_value = __bfloat162float(key[vector_offset + dimension]);
    const float2 norm_sums = block_sum_pair(
        query_value * query_value, key_value * key_value, scratch);
    const float query_normalizer =
        rsqrtf(norm_sums.x + kEpsilon) * kQueryScale;
    const float key_normalizer = rsqrtf(norm_sums.y + kEpsilon);
    normalized_query[dimension] = query_value * query_normalizer;
    normalized_key[dimension] = key_value * key_normalizer;
    __syncthreads();
    const float qk = block_sum(
        normalized_query[dimension] * normalized_key[dimension], scratch);
    const float decay = expf(g[token * kHeads + head]);
    const float beta_value = beta[token * kHeads + head];
    float key_memory = 0.0f;
    float query_memory = 0.0f;
#pragma unroll 4
    for (int key_dimension = 0; key_dimension < kSharedKeyDims;
         ++key_dimension) {
      const float state = staged_state[key_dimension * kDim + dimension];
      key_memory = fmaf(state, normalized_key[key_dimension], key_memory);
      query_memory = fmaf(state, normalized_query[key_dimension], query_memory);
    }
#pragma unroll
    for (int tail = 0; tail < kRegisterKeyDims; ++tail) {
      const int key_dimension = kSharedKeyDims + tail;
      const float state = resident_tail[tail];
      key_memory = fmaf(state, normalized_key[key_dimension], key_memory);
      query_memory = fmaf(state, normalized_query[key_dimension], query_memory);
    }
    const float value_value =
        __bfloat162float(value[vector_offset + dimension]);
    const float delta = (value_value - decay * key_memory) * beta_value;
    const float output_value = decay * query_memory + delta * qk;
#pragma unroll 4
    for (int key_dimension = 0; key_dimension < kSharedKeyDims;
         ++key_dimension) {
      const int shared_index = key_dimension * kDim + dimension;
      staged_state[shared_index] = fmaf(
          normalized_key[key_dimension], delta,
          staged_state[shared_index] * decay);
    }
#pragma unroll
    for (int tail = 0; tail < kRegisterKeyDims; ++tail) {
      const int key_dimension = kSharedKeyDims + tail;
      resident_tail[tail] = fmaf(
          normalized_key[key_dimension], delta, resident_tail[tail] * decay);
    }
    output[vector_offset + dimension] = __float2bfloat16(output_value);
    __syncthreads();
  }
#pragma unroll 4
  for (int key_dimension = 0; key_dimension < kSharedKeyDims;
       ++key_dimension) {
    recurrent_state[
        state_base + static_cast<int64_t>(key_dimension) * kDim + dimension] =
        staged_state[key_dimension * kDim + dimension];
  }
#pragma unroll
  for (int tail = 0; tail < kRegisterKeyDims; ++tail) {
    const int key_dimension = kSharedKeyDims + tail;
    recurrent_state[
        state_base + static_cast<int64_t>(key_dimension) * kDim + dimension] =
        resident_tail[tail];
  }
}

__global__ void qwen35_gdn_gated_rmsnorm_m8_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* gate,
    const __nv_bfloat16* weight, __nv_bfloat16* output,
    float epsilon, int tokens) {
  constexpr int kHeads = 48;
  constexpr int kDim = 128;
  constexpr int kWidth = kHeads * kDim;
  __shared__ float scratch[8];
  const int token = blockIdx.x / kHeads;
  const int head = blockIdx.x % kHeads;
  const int dimension = threadIdx.x;
  if (token >= tokens) return;
  const int64_t offset =
      static_cast<int64_t>(token) * kWidth + head * kDim + dimension;
  const float value = __bfloat162float(input[offset]);
  const float inverse_rms =
      rsqrtf(block_sum(value * value, scratch) / kDim + epsilon);
  const __nv_bfloat16 normalized = __float2bfloat16(value * inverse_rms);
  const __nv_bfloat16 weighted = __float2bfloat16(
      __bfloat162float(normalized) * __bfloat162float(weight[dimension]));
  const float gate_value = __bfloat162float(gate[offset]);
  const float silu_gate = gate_value / (1.0f + expf(-gate_value));
  output[offset] = __float2bfloat16(__bfloat162float(weighted) * silu_gate);
}

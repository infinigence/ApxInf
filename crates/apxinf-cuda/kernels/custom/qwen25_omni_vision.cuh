#pragma once

// Qwen2.5-Omni vision Q/K/V projection epilogue. The projection output is
// first bias-added and rounded to BF16, then Q/K apply the incumbent 2-D RoPE
// pair order while V publishes the rounded value directly.
template <int kHeads, int kHeadDim, bool kGrouped, bool kPackedProjection>
__global__ void qwen25_omni_vision_qkv_bias_rope_bf16_kernel(
    const __nv_bfloat16* query, const __nv_bfloat16* key,
    const __nv_bfloat16* value, const __nv_bfloat16* query_bias,
    const __nv_bfloat16* key_bias, const __nv_bfloat16* value_bias,
    __nv_bfloat16* query_output, __nv_bfloat16* key_output,
    __nv_bfloat16* value_output, int sequence, float theta,
    const uint32_t* positions, const uint32_t* group_indices) {
  constexpr int kHidden = kHeads * kHeadDim;
  constexpr int kHalf = kHeadDim / 2;
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int output_token = blockIdx.y;
  const int projection = blockIdx.z;
  if (output_token >= sequence) return;
  const int token = kGrouped ? group_indices[output_token] : output_token;
  const int input_stride = kPackedProjection ? 3 * kHidden : kHidden;
  const int projection_offset = kPackedProjection ? projection * kHidden : 0;

  if (projection == 2) {
    if (item >= kHidden) return;
    const int64_t input_index =
        static_cast<int64_t>(token) * input_stride + projection_offset + item;
    const int64_t output_index =
        static_cast<int64_t>(output_token) * kHidden + item;
    value_output[output_index] = __float2bfloat16(
        __bfloat162float((kPackedProjection ? query : value)[input_index]) +
        __bfloat162float(value_bias[item]));
    return;
  }

  if (item >= kHeads * kHalf) return;
  const int head = item / kHalf;
  const int pair = item - head * kHalf;
  const int input_base =
      token * input_stride + projection_offset + head * kHeadDim;
  const int output_base = output_token * kHidden + head * kHeadDim;
  const int input_index0 = input_base + pair;
  const int input_index1 = input_base + kHalf + pair;
  const int output_index0 = output_base + pair;
  const int output_index1 = output_base + kHalf + pair;
  const __nv_bfloat16* input = kPackedProjection
      ? query
      : (projection == 0 ? query : key);
  const __nv_bfloat16* bias = projection == 0 ? query_bias : key_bias;
  __nv_bfloat16* output = projection == 0 ? query_output : key_output;
  const __nv_bfloat16 rounded0 = __float2bfloat16(
      __bfloat162float(input[input_index0]) +
      __bfloat162float(bias[head * kHeadDim + pair]));
  const __nv_bfloat16 rounded1 = __float2bfloat16(
      __bfloat162float(input[input_index1]) +
      __bfloat162float(bias[head * kHeadDim + kHalf + pair]));
  const int axis = pair < kHalf / 2 ? 0 : 1;
  const int pair_in_axis = pair < kHalf / 2 ? pair : pair - kHalf / 2;
  const uint32_t position = positions[token * 2 + axis];
  const float frequency =
      1.0f / powf(theta, 2.0f * static_cast<float>(pair_in_axis) / kHalf);
  const float angle = static_cast<float>(position) * frequency;
  const float cosine = cosf(angle);
  const float sine = sinf(angle);
  const float value0 = __bfloat162float(rounded0);
  const float value1 = __bfloat162float(rounded1);
  output[output_index0] = __float2bfloat16(value0 * cosine - value1 * sine);
  output[output_index1] = __float2bfloat16(value0 * sine + value1 * cosine);
}

template <int kHidden>
__global__ void qwen25_omni_vision_bias_residual_exact_bf16_kernel(
    const __nv_bfloat16* projection, const __nv_bfloat16* bias,
    const __nv_bfloat16* residual, __nv_bfloat16* output, int sequence) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int token = blockIdx.y;
  if (token >= sequence || item >= kHidden) return;
  const int64_t index = static_cast<int64_t>(token) * kHidden + item;
  const __nv_bfloat16 rounded_projection = __float2bfloat16(
      __bfloat162float(projection[index]) + __bfloat162float(bias[item]));
  output[index] = __float2bfloat16(
      __bfloat162float(rounded_projection) + __bfloat162float(residual[index]));
}

template <int kIntermediate, bool kPackedProjection>
__global__ void qwen25_omni_vision_gate_up_bias_silu_mul_exact_bf16_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* gate_bias,
    const __nv_bfloat16* up, const __nv_bfloat16* up_bias,
    __nv_bfloat16* output, int sequence) {
  const int item = blockIdx.x * blockDim.x + threadIdx.x;
  const int token = blockIdx.y;
  if (token >= sequence || item >= kIntermediate) return;
  const int64_t output_index =
      static_cast<int64_t>(token) * kIntermediate + item;
  const int64_t packed_base =
      static_cast<int64_t>(token) * 2 * kIntermediate;
  const int64_t gate_index =
      kPackedProjection ? packed_base + item : output_index;
  const int64_t up_index =
      kPackedProjection ? packed_base + kIntermediate + item : output_index;
  const __nv_bfloat16 rounded_gate = __float2bfloat16(
      __bfloat162float(gate[gate_index]) + __bfloat162float(gate_bias[item]));
  const __nv_bfloat16 rounded_up = __float2bfloat16(
      __bfloat162float((kPackedProjection ? gate : up)[up_index]) +
      __bfloat162float(up_bias[item]));
  const float gate_value = __bfloat162float(rounded_gate);
  const __nv_bfloat16 activated = __float2bfloat16(
      gate_value / (1.0f + expf(-gate_value)));
  output[output_index] = __float2bfloat16(
      __bfloat162float(activated) * __bfloat162float(rounded_up));
}

#pragma once

__global__ void concat7_cols_bf16_kernel(
    const __nv_bfloat16* s0, const __nv_bfloat16* s1,
    const __nv_bfloat16* s2, const __nv_bfloat16* s3,
    const __nv_bfloat16* s4, const __nv_bfloat16* s5,
    const __nv_bfloat16* s6, __nv_bfloat16* dst, int rows, int cols,
    int broadcast_mask) {
  const int token = blockIdx.x;
  const int64_t dst_base = static_cast<int64_t>(token) * 7 * cols;
  for (int i = threadIdx.x; i < cols; i += blockDim.x) {
    const int64_t src_idx = static_cast<int64_t>(token) * cols + i;
    const int64_t bcast_idx = i;
    dst[dst_base + i] = (broadcast_mask & 1) ? s0[bcast_idx] : s0[src_idx];
    dst[dst_base + cols + i] =
        (broadcast_mask & 2) ? s1[bcast_idx] : s1[src_idx];
    dst[dst_base + 2 * cols + i] =
        (broadcast_mask & 4) ? s2[bcast_idx] : s2[src_idx];
    dst[dst_base + 3 * cols + i] =
        (broadcast_mask & 8) ? s3[bcast_idx] : s3[src_idx];
    dst[dst_base + 4 * cols + i] =
        (broadcast_mask & 16) ? s4[bcast_idx] : s4[src_idx];
    dst[dst_base + 5 * cols + i] =
        (broadcast_mask & 32) ? s5[bcast_idx] : s5[src_idx];
    dst[dst_base + 6 * cols + i] =
        (broadcast_mask & 64) ? s6[bcast_idx] : s6[src_idx];
  }
}

__global__ void flow_update_f32_kernel(
    float* w, const float* endpoint, float remaining, float step,
    int64_t count) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const float delta = __fsub_rn(endpoint[index], w[index]);
    const float velocity = __fmul_rn(delta, __fdiv_rn(1.0f, remaining));
    w[index] = __fadd_rn(w[index], __fmul_rn(velocity, step));
  }
}

__global__ void vision_position_kernel(
    const __nv_bfloat16* __restrict__ table,
    __nv_bfloat16* __restrict__ embeddings,
    int32_t* __restrict__ position_ids, int source_side, int height, int width,
    int hidden, int merge_size) {
  const int token = static_cast<int>(blockIdx.x);
  const int merge_area = merge_size * merge_size;
  const int merged_index = token / merge_area;
  const int inner_index = token % merge_area;
  const int merged_width = width / merge_size;
  const int row = (merged_index / merged_width) * merge_size +
                  inner_index / merge_size;
  const int column = (merged_index % merged_width) * merge_size +
                     inner_index % merge_size;

  if (threadIdx.x == 0) {
    position_ids[2LL * token] = row;
    position_ids[2LL * token + 1] = column;
  }

  const float row_position =
      height > 1 ? static_cast<float>(row) * static_cast<float>(source_side - 1) /
                       static_cast<float>(height - 1)
                 : 0.0f;
  const float column_position =
      width > 1 ? static_cast<float>(column) *
                          static_cast<float>(source_side - 1) /
                      static_cast<float>(width - 1)
                : 0.0f;
  const int row0 = static_cast<int>(floorf(row_position));
  const int column0 = static_cast<int>(floorf(column_position));
  const int row1 = min(row0 + 1, source_side - 1);
  const int column1 = min(column0 + 1, source_side - 1);
  const float row_weight = row_position - static_cast<float>(row0);
  const float column_weight = column_position - static_cast<float>(column0);
  const float row_weight0 = 1.0f - row_weight;
  const float column_weight0 = 1.0f - column_weight;

  for (int channel = threadIdx.x; channel < hidden; channel += blockDim.x) {
    const long long offset00 =
        (static_cast<long long>(row0) * source_side + column0) * hidden +
        channel;
    const long long offset01 =
        (static_cast<long long>(row0) * source_side + column1) * hidden +
        channel;
    const long long offset10 =
        (static_cast<long long>(row1) * source_side + column0) * hidden +
        channel;
    const long long offset11 =
        (static_cast<long long>(row1) * source_side + column1) * hidden +
        channel;
    const float value =
        row_weight0 * column_weight0 * __bfloat162float(table[offset00]) +
        row_weight0 * column_weight * __bfloat162float(table[offset01]) +
        row_weight * column_weight0 * __bfloat162float(table[offset10]) +
        row_weight * column_weight * __bfloat162float(table[offset11]);
    embeddings[static_cast<long long>(token) * hidden + channel] =
        __float2bfloat16(value);
  }
}

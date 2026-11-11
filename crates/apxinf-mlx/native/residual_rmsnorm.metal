// Adapted from EngineTailor. Copyright 2026 Haiyan Qin. MIT; see NOTICE.
constexpr uint columns = (uint(WIDTH) + 255u) / 256u;
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint group = simdgroup_index_in_threadgroup;
uint row_start = threadgroup_position_in_grid.x * uint(WIDTH);
uint norm_start = threadgroups_per_grid.x * uint(WIDTH) + row_start;
threadgroup float partials[8];
float values[columns];
float total = 0.0f;
#pragma unroll
for (uint i = 0; i < columns; ++i) {
    uint column = tid * columns + i;
    if (column < uint(WIDTH)) {
        T rounded = static_cast<T>(float(x[row_start + column]) + float(delta[row_start + column]));
        packed[row_start + column] = rounded;
        values[i] = float(rounded);
        float squared = values[i] * values[i];
        total = squared + total;
    }
}
total = simd_sum(total);
if (lane == 0u) partials[group] = total;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (group == 0u) {
    float value = lane < 8u ? partials[lane] : 0.0f;
    float sum = simd_sum(value);
    if (lane == 0u) partials[0] = sum;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float variance = partials[0] / float(WIDTH);
float inverse_rms = metal::precise::rsqrt(variance + EPSILON);
#pragma unroll
for (uint i = 0; i < columns; ++i) {
    uint column = tid * columns + i;
    if (column < uint(WIDTH)) {
        float scale = float(weight[column]);
        if (BEFORE_WEIGHT) {
            T normalized = static_cast<T>(values[i] * inverse_rms);
            packed[norm_start + column] = static_cast<T>(scale * float(normalized));
        } else {
            packed[norm_start + column] = static_cast<T>(values[i] * inverse_rms * scale);
        }
    }
}

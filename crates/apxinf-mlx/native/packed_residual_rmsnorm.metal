// Adapted from EngineTailor. Copyright 2026 Haiyan Qin. MIT; see NOTICE.
uint tid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint group = simdgroup_index_in_threadgroup;
threadgroup float partials[8];
float values[8];
float total = 0.0f;
#pragma unroll
for (uint i = 0; i < 8u; ++i) {
    uint column = tid * 8u + i;
    T rounded = static_cast<T>(float(x[column]) + float(delta[column]));
    packed[column] = rounded;
    values[i] = float(rounded);
    float squared = values[i] * values[i];
    total = squared + total;
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
float variance = partials[0] * 0.00048828125f;
float inverse_rms = metal::precise::rsqrt(variance + EPSILON);
#pragma unroll
for (uint i = 0; i < 8u; ++i) {
    uint column = tid * 8u + i;
    T normalized_round = static_cast<T>(values[i] * inverse_rms);
    packed[2048u + column] = static_cast<T>(float(weight[column]) * float(normalized_round));
}

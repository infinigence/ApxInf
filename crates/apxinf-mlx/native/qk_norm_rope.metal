// EngineTailor (MIT), Copyright 2026 Haiyan Qin.
// Fixed constants expanded from candidate_metal_qk_norm_rope.py::_kernel_source.
uint lid = thread_position_in_threadgroup.x;
uint lane = thread_index_in_simdgroup;
uint simd_group = simdgroup_index_in_threadgroup;
uint simd_groups = simdgroups_per_threadgroup;
uint head = threadgroup_position_in_grid.x;
bool is_q = head < 16u;
uint local_head = is_q ? head : head - 16u;
uint row_offset = local_head * 128u;

threadgroup float partials[32];
threadgroup T normalized_row[128];
float total = 0.0f;

for (uint column = lid * 4u; column < 128u; column += 1024u) {
    #pragma unroll
    for (uint j = 0; j < 4u; ++j) {
        uint c = column + j;
        if (c < 128u) {
            T activation = is_q ? q_input[row_offset + c] : k_input[row_offset + c];
            float value = static_cast<float>(activation);
            float squared = value * value;
            total = total + squared;
        }
    }
}
total = simd_sum(total);
if (lane == 0u) {
    partials[simd_group] = total;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (simd_group == 0u) {
    float group_value = lane < simd_groups ? partials[lane] : 0.0f;
    group_value = simd_sum(group_value);
    if (lane == 0u) {
        partials[0] = group_value;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
float inverse_rms = metal::precise::rsqrt(partials[0] * 0.0078125f + 0.000001f);
if (lid < 128u) {
    uint column = lid;
    T activation = is_q ? q_input[row_offset + column] : k_input[row_offset + column];
    float value = static_cast<float>(activation);
    float scaled = value * inverse_rms;
    float weight = is_q ? q_weight[column] : k_weight[column];
    float weighted = scaled * weight;
    normalized_row[column] = static_cast<T>(weighted);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (lid < 128u) {
    uint column = lid;
    uint partner = column < 64u ? column + 64u : column - 64u;
    T normalized = normalized_row[column];
    T rotate_half = column < 64u ? static_cast<T>(-normalized_row[partner]) : normalized_row[partner];
    T cos_value = static_cast<T>(cos_row[column]);
    T sin_value = static_cast<T>(sin_row[column]);
    T left = static_cast<T>(normalized * cos_value);
    T right = static_cast<T>(rotate_half * sin_value);
    T rotated = static_cast<T>(left + right);
    if (is_q) {
        q_output[row_offset + column] = rotated;
    } else {
        k_output[row_offset + column] = rotated;
    }
}

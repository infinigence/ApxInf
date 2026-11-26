// Unweighted RMSNorm + scalar scale reproducing MLX 0.32.3
// `fast.rms_norm(x, None, eps).mul(scalar)` (rms_norm.metal rms_single_row,
// axis_size 128 <= RMS_LOOPED_LIMIT, one simdgroup; then the BF16 Multiply
// binary op). Two rounding points, as the unfused path: RMSNorm output rounds
// to BF16, then the BF16 * BF16 scale rounds to BF16. Portions adapted from
// MLX (MIT, Apple). See NOTICE.
// template args: T (dtype), AXIS_SIZE (int); input x; input scale; output dst.
    constexpr int SIMD_SIZE = 32;
    constexpr int N_READS = 4;
    threadgroup float local_inv_mean[1];
    threadgroup float local_sums[SIMD_SIZE];
    const uint gid = threadgroup_position_in_grid.x;
    const uint lid = thread_position_in_threadgroup.x;
    const uint simd_lane_id = thread_index_in_simdgroup;
    const uint simd_group_id = simdgroup_index_in_threadgroup;
    x += gid * size_t(AXIS_SIZE) + lid * N_READS;
    float acc = 0.0f;
    float thread_x[N_READS];
    if (lid * N_READS + N_READS <= uint(AXIS_SIZE)) {
        for (int i = 0; i < N_READS; ++i) {
            thread_x[i] = (float)x[i];
            acc += thread_x[i] * thread_x[i];
        }
    } else {
        for (int i = 0; i < N_READS; ++i) {
            thread_x[i] = (lid * N_READS + i < uint(AXIS_SIZE)) ? (float)x[i] : 0.0f;
            acc += thread_x[i] * thread_x[i];
        }
    }
    acc = simd_sum(acc);
    if (simd_group_id == 0u) local_sums[simd_lane_id] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_lane_id == 0u) local_sums[simd_group_id] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0u) {
        acc = simd_sum(local_sums[simd_lane_id]);
        if (simd_lane_id == 0u) {
            local_inv_mean[0] = metal::precise::rsqrt(acc / float(AXIS_SIZE) + EPSILON);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dst += gid * size_t(AXIS_SIZE) + lid * N_READS;
    const T scalar_t = static_cast<T>(scale);
    if (lid * N_READS + N_READS <= uint(AXIS_SIZE)) {
        for (int i = 0; i < N_READS; ++i) {
            const T normalized = static_cast<T>(thread_x[i] * local_inv_mean[0]);
            dst[i] = normalized * scalar_t;
        }
    } else {
        for (int i = 0; i < N_READS; ++i) {
            if (lid * N_READS + i < uint(AXIS_SIZE)) {
                const T normalized = static_cast<T>(thread_x[i] * local_inv_mean[0]);
                dst[i] = normalized * scalar_t;
            }
        }
    }

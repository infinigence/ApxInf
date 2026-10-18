// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
// Affine Q4 group-64 rows R0..R0+N decoded with BF16 arithmetic, widened to FP16, written as [K, N].

    threadgroup half tile[32][65];
    const uint tid = thread_position_in_threadgroup.x;
    const uint k0 = threadgroup_position_in_grid.x * 64;
    const uint n0 = threadgroup_position_in_grid.y * 32;
    const uint row = tid / 8;
    const uint word = tid % 8;
    const uint packed = weight[(R0 + n0 + row) * (K / 8) + k0 / 8 + word];
    const uint group = (R0 + n0 + row) * (K / 64) + k0 / 64;
    const bfloat16_t scale = scales[group];
    const bfloat16_t bias = biases[group];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const uint8_t d = (packed >> (4 * i)) & 0x0f;
        const bfloat16_t w = scale * d + bias;
        tile[row][word * 8 + i] = half(float(w));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint out_row = (tid % 8) * 4;
    const uint out_col = tid / 8;
    device half4* vectors = reinterpret_cast<device half4*>(out);
    #pragma clang loop unroll(full)
    for (int j = 0; j < 2; ++j) {
        const uint col = out_col + j * 32;
        const half4 value = half4(tile[out_row][col], tile[out_row + 1][col],
                                 tile[out_row + 2][col], tile[out_row + 3][col]);
        vectors[((k0 + col) * N + n0 + out_row) / 4] = value;
    }

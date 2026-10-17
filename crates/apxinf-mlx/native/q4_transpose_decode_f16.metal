// Adapted from EngineTailor feat/clef-g20-engine 6edc065c (g20). Copyright 2026 Haiyan Qin. MIT; see NOTICE.
// Affine Q4 group-64 rows R0..R0+N decoded with FP16 scale/bias arithmetic, written as [K, N].

    threadgroup half tile[32][65];
    const uint tid = thread_position_in_threadgroup.x;
    const uint k0 = threadgroup_position_in_grid.x * 64;
    const uint n0 = threadgroup_position_in_grid.y * 32;
    const uint row = tid / 8;
    const uint word = tid % 8;
    const uint packed = weight[(R0 + n0 + row) * (K / 8) + k0 / 8 + word];
    const uint group = (R0 + n0 + row) * (K / 64) + k0 / 64;
    const half scale = half(float(scales[group]));
    const half bias = half(float(biases[group]));
    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const uint8_t nibble = (packed >> (4 * i)) & 0x0f;
        tile[row][word * 8 + i] = scale * nibble + bias;
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

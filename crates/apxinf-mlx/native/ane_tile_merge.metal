// Adapted from EngineTailor M4 ANE campaign frontier stk011-m4a012-fp16-1248 (A14 boundary kernels). Copyright 2026 Haiyan Qin. MIT; see NOTICE.
// Restores the Neural Engine share (rows times both scales in FP32), adds the GPU share and rounds once to OT.
    const uint lane = thread_position_in_threadgroup.x;
    const uint row = thread_position_in_threadgroup.y;
    const uint channel_base = threadgroup_position_in_grid.x * 32;
    const uint token_base = threadgroup_position_in_grid.y * 32;
    // TILE is a multiple of 32, so these 32 tokens come from one ANE tile.
    const uint tile_base = (token_base / TILE) * C * TILE + token_base % TILE;
    threadgroup float transposed[32][33];
    for (uint i = 0; i < 8; ++i) {
        const uint channel = channel_base + row + 4 * i;
        const uint token = token_base + lane;
        transposed[row + 4 * i][lane] = token < N
            ? static_cast<float>(ane_y[tile_base + channel * TILE + lane]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 8; ++i) {
        const uint token = token_base + row + 4 * i;
        const uint channel = channel_base + lane;
        if (token < N) {
            const float restored = (transposed[lane][row + 4 * i]
                * static_cast<float>(gate_scale[token]))
                * static_cast<float>(up_scale[token]);
            out[token * C + channel] = static_cast<OT>(
                static_cast<float>(gpu_y[token * C + channel]) + restored);
        }
    }

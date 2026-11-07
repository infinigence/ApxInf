// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
// One pass over the layer input: the GPU share's FP16 rows padded to ROWS and every Neural Engine tile
// channels first, zero past the N real tokens.
    const uint lane = thread_position_in_threadgroup.x;
    const uint row = thread_position_in_threadgroup.y;
    const uint channel_base = threadgroup_position_in_grid.x * 32;
    const uint token_base = threadgroup_position_in_grid.y * 32;
    threadgroup half block[32][33];
    for (uint i = 0; i < 8; ++i) {
        const uint token = token_base + row + 4 * i;
        const uint channel = channel_base + lane;
        const half value = token < N ? half(float(h[token * C + channel])) : half(0);
        block[row + 4 * i][lane] = value;
        if (token < ROWS) {
            x[token * C + channel] = value;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 8; ++i) {
        const uint channel = channel_base + row + 4 * i;
        const uint token = token_base + lane;
        packed[((token / TILE) * C + channel) * TILE + token % TILE] = block[lane][row + 4 * i];
    }

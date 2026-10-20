// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
    const int groups = (N / BN) * (M / BM);
    const int group = int(threadgroup_position_in_grid.y) * (N / BN)
        + int(threadgroup_position_in_grid.x);
    device uint4* wout = (device uint4*)wgp;
    for (int r = group; r < GR; r += groups) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int word = int(tid) + 128 * j;
            const uint val = gq[r * 512 + word];
            const bfloat16_t scale = gs[r * 64 + word / 8];
            const bfloat16_t bias = gb[r * 64 + word / 8];
            uint packed[4];
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const uint8_t d0 = (val >> (8 * e)) & 0x0f;
                const uint8_t d1 = (val >> (8 * e + 4)) & 0x0f;
                const bfloat16_t v0 = scale * d0 + bias;
                const bfloat16_t v1 = scale * d1 + bias;
                packed[e] = uint(as_type<ushort>(v0)) | (uint(as_type<ushort>(v1)) << 16);
            }
            wout[r * 512 + word] = uint4(packed[0], packed[1], packed[2], packed[3]);
        }
    }

// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = tid % 32;
    const uint sg = tid / 32;
    threadgroup float partial[16];
    const device uint2* xr = (const device uint2*)x + row * 1024;
    const device uint2* ar = (const device uint2*)a + row * 1024;
    device uint2* x1r = (device uint2*)x1 + row * 1024;
    device uint2* outr = (device uint2*)out + row * 1024;
    float v[16];
    float sum = 0.0f;
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        const uint col = 4 * (tid + 256 * c);
        const uint2 xp = xr[tid + 256 * c];
        const uint2 ap = ar[tid + 256 * c];
        const uint xw[4] = {xp.x << 16, xp.x & 0xffff0000u, xp.y << 16, xp.y & 0xffff0000u};
        const uint aw[4] = {ap.x << 16, ap.x & 0xffff0000u, ap.y << 16, ap.y & 0xffff0000u};
        uint h[4];
        #pragma unroll
        for (uint e = 0; e < 4; ++e) {
            const float g = float(gate_mod[row * gate_mod_strides[1]
                                           + (4096 + col + e) * gate_mod_strides[2]]);
            const float p = float(bfloat(float(bfloat(metal::precise::tanh(g)))
                                         * as_type<float>(aw[e])));
            const float r = float(bfloat(as_type<float>(xw[e]) + p));
            v[4 * c + e] = r;
            sum += r;
            h[e] = as_type<uint>(r) >> 16;
        }
        x1r[tid + 256 * c] = uint2(h[0] | (h[1] << 16), h[2] | (h[3] << 16));
    }

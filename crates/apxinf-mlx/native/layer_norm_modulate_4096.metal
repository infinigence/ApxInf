// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
    const uint row = threadgroup_position_in_grid.y;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = tid % 32;
    const uint sg = tid / 32;
    threadgroup float partial[16];
    const device uint2* xr = (const device uint2*)x + row * 1024;
    device uint2* outr = (device uint2*)out + row * 1024;
    float v[16];
    float sum = 0.0f;
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        const uint2 xp = xr[tid + 256 * c];
        v[4 * c] = as_type<float>(xp.x << 16);
        v[4 * c + 1] = as_type<float>(xp.x & 0xffff0000u);
        v[4 * c + 2] = as_type<float>(xp.y << 16);
        v[4 * c + 3] = as_type<float>(xp.y & 0xffff0000u);
        sum += v[4 * c] + v[4 * c + 1] + v[4 * c + 2] + v[4 * c + 3];
    }

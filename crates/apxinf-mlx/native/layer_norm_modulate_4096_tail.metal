// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
    sum = simd_sum(sum);
    if (lane == 0) {
        partial[sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = simd_sum(lane < 8 ? partial[lane] : 0.0f) / 4096.0f;
    float square = 0.0f;
    #pragma unroll
    for (uint i = 0; i < 16; ++i) {
        square += (v[i] - mean) * (v[i] - mean);
    }
    square = simd_sum(square);
    if (lane == 0) {
        partial[8 + sg] = square;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = metal::precise::rsqrt(
        simd_sum(lane < 8 ? partial[8 + lane] : 0.0f) / 4096.0f + 1.0e-6f);
    #pragma unroll
    for (uint c = 0; c < 4; ++c) {
        const uint col = 4 * (tid + 256 * c);
        uint h[4];
        #pragma unroll
        for (uint e = 0; e < 4; ++e) {
            const float s = float(mod[row * mod_strides[1] + (col + e) * mod_strides[2]]);
            const float n = float(bfloat((v[4 * c + e] - mean) * inv));
            h[e] = as_type<uint>(float(bfloat(n * float(bfloat(1.0f + s))))) >> 16;
        }
        outr[tid + 256 * c] = uint2(h[0] | (h[1] << 16), h[2] | (h[3] << 16));
    }

// Adapted from EngineTailor campaign/clef-flash9b-m1pro 4030047b (C14). Copyright 2026 Haiyan Qin. MIT; see NOTICE.
// One packed word (eight weights) per thread, [N, K] FP16 out, MLX affine_dequantize with FP16 scales (s22).
    const uint i = thread_position_in_grid.x;
    const uint packed = weight[i];
    const uint group = i / 8;
    const half scale = half(float(scales[group]));
    const half bias = half(float(biases[group]));
    half4 lo, hi;
    #pragma clang loop unroll(full)
    for (int j = 0; j < 4; ++j) {
        lo[j] = scale * half((packed >> (4 * j)) & 0x0f) + bias;
        hi[j] = scale * half((packed >> (4 * j + 16)) & 0x0f) + bias;
    }
    device half4* vectors = reinterpret_cast<device half4*>(out);
    vectors[2 * i] = lo;
    vectors[2 * i + 1] = hi;

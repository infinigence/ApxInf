// Adapted from the EngineTailor Qwen3-TTS predictor MLP frontier 944fd18d.
// Copyright (c) 2026 Haiyan Qin. MIT; see NOTICE.
// One row: row4-packed down [256, 3072, 4] times activation [3072] -> [1024].
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint group = threadgroup_position_in_grid.x;
    const int row_quad = int(group) * 4 + int(simd_group);
    const int row_base = row_quad * 4;

    const device vec<bfloat16_t, 4>* packed4 =
        (const device vec<bfloat16_t, 4>*)packed_down;

    vec<float, 4> down_acc = vec<float, 4>(0.0f);

    for (int block = 0; block < 24; ++block) {
        const int k_base = block * 128 + int(lane) * 4;
        MLX_MTL_PRAGMA_UNROLL
        for (int item = 0; item < 4; ++item) {
            const int k = k_base + item;
            const float xv = static_cast<float>(activation[k]);
            down_acc += vec<float, 4>(packed4[row_quad * 3072 + k]) * xv;
        }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (ushort offset = 16; offset >= 1; offset >>= 1) {
        MLX_MTL_PRAGMA_UNROLL
        for (int row_item = 0; row_item < 4; ++row_item) {
            down_acc[row_item] +=
                metal::simd_shuffle_down(down_acc[row_item], offset);
        }
    }

    if (lane == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int row_item = 0; row_item < 4; ++row_item) {
            output[row_base + row_item] =
                static_cast<bfloat16_t>(down_acc[row_item]);
        }
    }

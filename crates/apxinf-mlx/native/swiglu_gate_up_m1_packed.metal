// Adapted from the EngineTailor Qwen3-TTS predictor MLP frontier 944fd18d.
// Copyright (c) 2026 Haiyan Qin. MIT; see NOTICE.
// One row: SwiGLU of row4-packed gate/up [768, 1024, 8] times x [1024] -> [3072].
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint group = threadgroup_position_in_grid.x;
    const int row_quad = int(group) * 4 + int(simd_group);
    const int row_base = row_quad * 4;

    const device vec<bfloat16_t, 4>* packed4 =
        (const device vec<bfloat16_t, 4>*)packed_gate_up;

    vec<float, 4> gate_acc = vec<float, 4>(0.0f);
    vec<float, 4> up_acc = vec<float, 4>(0.0f);

    for (int block = 0; block < 8; ++block) {
        const int k_base = block * 128 + int(lane) * 4;
        MLX_MTL_PRAGMA_UNROLL
        for (int item = 0; item < 4; ++item) {
            const int k = k_base + item;
            const int packed_index = (row_quad * 1024 + k) * 2;
            const float xv = static_cast<float>(x[k]);
            gate_acc += vec<float, 4>(packed4[packed_index]) * xv;
            up_acc += vec<float, 4>(packed4[packed_index + 1]) * xv;
        }
    }

    MLX_MTL_PRAGMA_UNROLL
    for (ushort offset = 16; offset >= 1; offset >>= 1) {
        MLX_MTL_PRAGMA_UNROLL
        for (int row_item = 0; row_item < 4; ++row_item) {
            gate_acc[row_item] +=
                metal::simd_shuffle_down(gate_acc[row_item], offset);
            up_acc[row_item] +=
                metal::simd_shuffle_down(up_acc[row_item], offset);
        }
    }

    if (lane == 0) {
        MLX_MTL_PRAGMA_UNROLL
        for (int row_item = 0; row_item < 4; ++row_item) {
            activation[row_base + row_item] = QwenTTSActivateMultiply()(
                gate_acc[row_item], up_acc[row_item]);
        }
    }

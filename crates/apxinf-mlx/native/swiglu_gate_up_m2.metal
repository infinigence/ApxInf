// Adapted from the EngineTailor Qwen3-TTS predictor MLP frontier 944fd18d.
// Copyright (c) 2026 Haiyan Qin. MIT; see NOTICE.
// Two rows: SwiGLU of gate/up [3072, 1024] times x [2, 1024] -> [2, 3072].
    const uint lane = thread_index_in_simdgroup;
    const uint simd_group = simdgroup_index_in_threadgroup;
    const uint group = threadgroup_position_in_grid.y;
    const int row = int(group) * 4 + int(simd_group);

    const device vec<bfloat16_t, 4>* gate4 =
        (const device vec<bfloat16_t, 4>*)(w_gate + row * 1024);
    const device vec<bfloat16_t, 4>* up4 =
        (const device vec<bfloat16_t, 4>*)(w_up + row * 1024);
    const device vec<bfloat16_t, 4>* x4 =
        (const device vec<bfloat16_t, 4>*)x;

    vec<float, 4> gate_weights[8];
    vec<float, 4> up_weights[8];
    MLX_MTL_PRAGMA_UNROLL
    for (int item = 0; item < 8; ++item) {
        const int index = item * 32 + int(lane);
        gate_weights[item] = vec<float, 4>(gate4[index]);
        up_weights[item] = vec<float, 4>(up4[index]);
    }

    float gate_acc[2] = {0.0f};
    float up_acc[2] = {0.0f};
    MLX_MTL_PRAGMA_UNROLL
    for (int vector = 0; vector < 2; ++vector) {
        float gate_local = 0.0f;
        float up_local = 0.0f;
        MLX_MTL_PRAGMA_UNROLL
        for (int item = 0; item < 8; ++item) {
            const int index = vector * 256 + item * 32 + int(lane);
            const vec<float, 4> xv = vec<float, 4>(x4[index]);
            gate_local += metal::dot(gate_weights[item], xv);
            up_local += metal::dot(up_weights[item], xv);
        }
        gate_acc[vector] += gate_local;
        up_acc[vector] += up_local;
    }

    MLX_MTL_PRAGMA_UNROLL
    for (int vector = 0; vector < 2; ++vector) {
        MLX_MTL_PRAGMA_UNROLL
        for (ushort offset = 16; offset >= 1; offset >>= 1) {
            gate_acc[vector] +=
                metal::simd_shuffle_down(gate_acc[vector], offset);
            up_acc[vector] +=
                metal::simd_shuffle_down(up_acc[vector], offset);
        }
    }

    if (lane == 0) {
        activation[row] = QwenTTSActivateMultiply()(
            gate_acc[0], up_acc[0]);
        activation[3072 + row] = QwenTTSActivateMultiply()(
            gate_acc[1], up_acc[1]);
    }

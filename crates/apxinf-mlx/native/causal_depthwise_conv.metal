// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
// Causal depthwise convolution over [1, L, C] with zero history, FP32 accumulation in
// tap order, one rounding to InT. Follows MLX 0.32.3 depthwise_conv_1d (MIT, Apple).
            const uint c = thread_position_in_grid.x;
            const uint t = thread_position_in_grid.y;
            if (t >= uint(length)) return;
            // MLX binds the small weight and the two scalars as `constant`
            // buffers and `x` as `device`, so the pointer is deduced.
            auto w = weight + c * kernel_size;
            float acc = 0.0f;
            for (int i = 0; i < kernel_size; ++i) {
                const int row = int(t) + i - (kernel_size - 1);
                const InT value = row >= 0 ? x[row * C + c] : InT(0);
                acc += static_cast<float>(value) * static_cast<float>(w[i]);
            }
            out[t * C + c] = static_cast<InT>(acc);

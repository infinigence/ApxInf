// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
// Causal depthwise convolution over [1, L, C] with zero history, FP32 accumulation.
    const int c = thread_position_in_grid.x;
    const int t = thread_position_in_grid.y;
    if (t >= length) return;
    auto w = weight + c * kernel_size;
    float acc = 0.0;
    for (int i = 0; i < kernel_size; ++i) {
        const int row = t + i - (kernel_size - 1);
        const InT value = row >= 0 ? x[row * C + c] : InT(0);
        acc += static_cast<float>(value) * w[i];
    }
    out[t * C + c] = static_cast<InT>(acc);

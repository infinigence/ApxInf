// Adapted from EngineTailor campaign/clef-flash9b-m1pro 4030047b (C14). Copyright 2026 Haiyan Qin. MIT; see NOTICE.
// s31: causal depthwise convolution over [1, L, C] with zero history, FP32 accumulation.
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

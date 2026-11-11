// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
    constexpr int BM = 64;
    constexpr int BN = 128;
    constexpr int BK = 64;
    constexpr int HD = 128;
    const int M = int(a_shape[1]);
    const int text = WITH_TEXT ? int(kt_shape[2]) : 0;
    const int keys = text + M;
    const int m_base = int(threadgroup_position_in_grid.y) * BM;
    const int head = int(threadgroup_position_in_grid.x);
    const int n_base = head * HD;
    const uint tid = thread_position_in_threadgroup.x;
    const int lane = int(tid % 32);
    const int sg = int(tid / 32);
    alignas(16) threadgroup float ctile[BM * BN];

    metal::tensor<device bfloat, metal::dextents<int32_t, 2>, metal::tensor_inline>
        ta((device bfloat*)a, metal::dextents<int32_t, 2>(K, M));
    metal::tensor<device bfloat, metal::dextents<int32_t, 2>, metal::tensor_inline>
        tw((device bfloat*)w, metal::dextents<int32_t, 2>(K, N));
    auto ma = ta.slice(0, m_base);
    auto mb = tw.slice(0, n_base);
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN, BK, false, true, false,
        mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<8>> op;
    auto accum = op.template get_destination_cooperative_tensor<
        metal::remove_addrspace_t<decltype(ma)>,
        metal::remove_addrspace_t<decltype(mb)>, float>();
    #pragma unroll
    for (uint16_t i = 0; i < accum.get_capacity(); ++i) {
        accum[i] = 0.0f;
    }
    for (int kk = 0; kk < K; kk += BK) {
        auto left = ta.slice(kk, m_base);
        auto right = tw.slice(kk, n_base);
        op.run(left, right, accum);
    }
    metal::tensor<threadgroup float, metal::dextents<int32_t, 2>, metal::tensor_inline>
        tcf(ctile, metal::dextents<int32_t, 2>(BN, BM));
    accum.store(tcf);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int d = lane * 4;
    for (int row = sg; row < BM; row += 8) {
        const int t = m_base + row;
        float values[4];
        float square = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            values[j] = float(bfloat(ctile[row * HD + d + j]));
            square += values[j] * values[j];
        }
        const float inv = metal::precise::rsqrt(simd_sum(square) / 128.0f + 1.0e-6f);
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            values[j] = values[j] * inv * float(nw[d + j]);
        }
        const int f = t * 64 + d / 2;
        const float c0 = rc[f];
        const float s0 = rs[f];
        const float c1 = rc[f + 1];
        const float s1 = rs[f + 1];
        const int dst = (head * keys + text + t) * HD + d;
        out[dst] = bfloat(values[0] * c0 - values[1] * s0);
        out[dst + 1] = bfloat(values[0] * s0 + values[1] * c0);
        out[dst + 2] = bfloat(values[2] * c1 - values[3] * s1);
        out[dst + 3] = bfloat(values[2] * s1 + values[3] * c1);
    }
    if (WITH_TEXT && m_base == 0) {
        for (int i = int(tid); i < text * HD; i += 256) {
            out[head * keys * HD + i] = kt[head * text * HD + i];
        }
    }

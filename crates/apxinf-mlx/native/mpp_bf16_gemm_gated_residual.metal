// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    const int M = int(a_shape[1]);
    const int m_base = int(threadgroup_position_in_grid.y) * BM;
    const int n_base = int(threadgroup_position_in_grid.x) * BN;
    const uint tid = thread_position_in_threadgroup.x;
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
    mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<4>> op;
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
    for (int i = int(tid); i < BM * BN; i += 128) {
        const int row = m_base + i / BN;
        const int col = n_base + i % BN;
        const float g = float(mod[row * mod_strides[1] + (N + col) * mod_strides[2]]);
        const float gate = float(bfloat(metal::precise::tanh(g)));
        const float product = float(bfloat(gate * float(bfloat(ctile[i]))));
        const float r = float(res[row * res_strides[1] + col * res_strides[2]]);
        out[row * N + col] = bfloat(r + product);
    }

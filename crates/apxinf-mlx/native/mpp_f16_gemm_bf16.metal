// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
// [P, K] x [K, N] FP16 GEMM in Metal 4 MPP matmul2d 64 x 64 x 64 tiles, FP32 accumulation, BF16 output rows < T.
    constexpr int BM = 64, BN = 64, BK = 64;
    const int M = int(x_shape[1]);
    const int row0 = int(threadgroup_position_in_grid.y) * BM;
    const int col0 = int(threadgroup_position_in_grid.x) * BN;
    const uint tid = thread_position_in_threadgroup.x;
    alignas(16) threadgroup float ctile[BM * BN];

    metal::tensor<device half, metal::dextents<int32_t, 2>, metal::tensor_inline>
        ta((device half*)x, metal::dextents<int32_t, 2>(K, M));
    metal::tensor<device half, metal::dextents<int32_t, 2>, metal::tensor_inline>
        tw((device half*)weight, metal::dextents<int32_t, 2>(WS, K));
    auto ma = ta.slice(0, row0);
    auto mb = tw.slice(col0, 0);

    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(
        BM, BN, BK, false, false, false,
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
        auto left = ta.slice(kk, row0);
        auto right = tw.slice(col0, kk);
        op.run(left, right, accum);
    }
    metal::tensor<threadgroup float, metal::dextents<int32_t, 2>, metal::tensor_inline>
        tc(ctile, metal::dextents<int32_t, 2>(BN, BM));
    accum.store(tc);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < BM * BN; i += 128) {
        const int row = row0 + i / BN;
        if (row < T) {
            out[row * N + OFF + col0 + i % BN] = static_cast<bfloat16_t>(ctile[i]);
        }
    }

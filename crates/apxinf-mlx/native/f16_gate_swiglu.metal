// Adapted from EngineTailor campaign/clef-flash9b-m1pro 4030047b (C14). Copyright 2026 Haiyan Qin. MIT; see NOTICE.
// a13: Steel FP16 gate GEMM x W^T with the SwiGLU epilogue silu(gate) * up, FP16 out.
    using namespace mlx::steel;
    using loader_a_t = BlockLoader<half, 64, 32, 40, true, 128>;
    using loader_b_t = BlockLoader<half, 32, 32, 40, true, 128>;
    using mma_t = BlockMMA<half, half, 64, 32, 32, 2, 2,
                          false, true, 40, 40, float, A13SwiGLU>;
    const ushort simd_lane_id = thread_index_in_simdgroup;
    const ushort simd_group_id = simdgroup_index_in_threadgroup;
    const int c_row = int(threadgroup_position_in_grid.y) * 64;
    const int c_col = int(threadgroup_position_in_grid.x) * 32;
    threadgroup half As[64 * 40];
    threadgroup half Bs[32 * 40];
    threadgroup_barrier(mem_flags::mem_none);
    thread mma_t mma_op(simd_group_id, simd_lane_id);
    thread loader_a_t loader_a(x + size_t(c_row) * K, K, As, simd_group_id, simd_lane_id);
    thread loader_b_t loader_b(weight + size_t(c_col) * K, K, Bs, simd_group_id, simd_lane_id);
    const int gemm_k_iterations = iterations[0];
    for (int k = 0; k < gemm_k_iterations; k++) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      loader_a.load_unsafe();
      loader_b.load_unsafe();
      threadgroup_barrier(mem_flags::mem_threadgroup);
      mma_op.mma(As, Bs);
      loader_a.next();
      loader_b.next();
    }
    threadgroup_barrier(mem_flags::mem_none);
    const size_t offset = size_t(c_row) * N + c_col;
    thread const A13SwiGLU epilogue = {};
    mma_op.store_result(out + offset, N, up + offset, N, 1, epilogue);

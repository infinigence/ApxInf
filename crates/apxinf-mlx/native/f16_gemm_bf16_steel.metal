// Adapted from EngineTailor campaign/clef-flash9b-m1pro 4030047b (C14). Copyright 2026 Haiyan Qin. MIT; see NOTICE.
// [P, K] x [K, N] FP16 GEMM, FP32 accumulation, BF16 output rows < T; Steel's unpipelined K loop (s13).
    constexpr int BK = 16, LA = 24, LB = 72;
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int tid = thread_index_in_threadgroup;
    const int row0 = int(threadgroup_position_in_grid.y) * 64;
    const int col0 = int(threadgroup_position_in_grid.x) * 64;
    const int qid = lane / 4;
    const int lm = (qid & 4) + ((lane / 2) % 4);
    const int ln = (qid & 2) * 2 + (lane % 2) * 2;
    const int wm = (sg / 2) * 8;
    const int wn = (sg % 2) * 8;
    const int ar = tid / 2, ac = (tid % 2) * 8;
    const int br = tid / 8, bc = (tid % 8) * 8;
    threadgroup half as[64 * LA];
    threadgroup half bs[BK * LB];
    float2 c[4][4] = {};

    // Steel's K loop: copy one K tile between two barriers, then multiply.
    for (int k0 = 0; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        *reinterpret_cast<threadgroup half4*>(as + ar * LA + ac) =
            *reinterpret_cast<const device half4*>(x + (row0 + ar) * K + k0 + ac);
        *reinterpret_cast<threadgroup half4*>(as + ar * LA + ac + 4) =
            *reinterpret_cast<const device half4*>(x + (row0 + ar) * K + k0 + ac + 4);
        *reinterpret_cast<threadgroup half4*>(bs + br * LB + bc) =
            *reinterpret_cast<const device half4*>(weight + (k0 + br) * N + col0 + bc);
        *reinterpret_cast<threadgroup half4*>(bs + br * LB + bc + 4) =
            *reinterpret_cast<const device half4*>(weight + (k0 + br) * N + col0 + bc + 4);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk = 0; kk < BK; kk += 8) {
            float2 a[4], b[4];
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                const int offset = (wm + lm + i * 16) * LA + kk + ln;
                a[i] = float2(as[offset], as[offset + 1]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j) {
                const int offset = (kk + lm) * LB + wn + ln + j * 16;
                b[j] = float2(bs[offset], bs[offset + 1]);
            }
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int i = 0; i < 4; ++i) {
                #pragma clang loop unroll(full)
                for (int jj = 0; jj < 4; ++jj) {
                    const int j = (i % 2) ? 3 - jj : jj;
                    c[i][j] = g07_mma(a[i], b[j], c[i][j]);
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (int i = 0; i < 4; ++i) {
        const int row = row0 + wm + lm + i * 16;
        if (row < T) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j) {
                const int col = col0 + wn + ln + j * 16;
                out[row * N + col] = static_cast<bfloat16_t>(c[i][j].x);
                out[row * N + col + 1] = static_cast<bfloat16_t>(c[i][j].y);
            }
        }
    }

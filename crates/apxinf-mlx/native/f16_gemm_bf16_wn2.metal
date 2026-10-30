// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
// MLX class-g FP16 NN tile (bm64 bn64 bk16 wm1 wn2), 64 threads; WS (weight stride) and OFF
// (output column offset) are defined by the prefix of the low or high half.
    constexpr int BK = 16, LA = 24, LB = 72;
    const int lane = thread_index_in_simdgroup;
    const int sg = simdgroup_index_in_threadgroup;
    const int tid = thread_index_in_threadgroup;
    const int row0 = int(threadgroup_position_in_grid.y) * 64;
    const int col0 = int(threadgroup_position_in_grid.x) * 64;
    const int qid = lane / 4;
    const int lm = (qid & 4) + ((lane / 2) % 4);
    const int ln = (qid & 2) * 2 + (lane % 2) * 2;
    const int wn = sg * 8;
    const int ar = tid;
    const int br = tid / 4, bc = (tid % 4) * 16;
    threadgroup half as[64 * LA];
    threadgroup half bs[BK * LB];
    float2 c[8][4] = {};

    // Steel's K loop: copy one K tile between two barriers, then multiply.
    for (int k0 = 0; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        #pragma clang loop unroll(full)
        for (int v = 0; v < 16; v += 4) {
            *reinterpret_cast<threadgroup half4*>(as + ar * LA + v) =
                *reinterpret_cast<const device half4*>(x + (row0 + ar) * K + k0 + v);
            *reinterpret_cast<threadgroup half4*>(bs + br * LB + bc + v) =
                *reinterpret_cast<const device half4*>(weight + (k0 + br) * WS + col0 + bc + v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (int kk = 0; kk < BK; kk += 8) {
            float2 a[8], b[4];
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; ++i) {
                const int offset = (lm + i * 8) * LA + kk + ln;
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
            for (int i = 0; i < 8; ++i) {
                #pragma clang loop unroll(full)
                for (int jj = 0; jj < 4; ++jj) {
                    const int j = (i % 2) ? 3 - jj : jj;
                    c[i][j] = g07_mma(a[i], b[j], c[i][j]);
                }
            }
        }
    }

    #pragma clang loop unroll(full)
    for (int i = 0; i < 8; ++i) {
        const int row = row0 + lm + i * 8;
        if (row < T) {
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j) {
                const int col = col0 + wn + ln + j * 16;
                out[row * N + OFF + col] = static_cast<bfloat16_t>(c[i][j].x);
                out[row * N + OFF + col + 1] = static_cast<bfloat16_t>(c[i][j].y);
            }
        }
    }

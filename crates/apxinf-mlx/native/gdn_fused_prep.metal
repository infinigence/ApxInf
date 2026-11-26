// Graft (2026-10-23): the GDN operand chain of d3-lite -- the causal depthwise
// convolution, SiLU, the q/k/v split and the per-head q/k l2norm -- over one
// read of the raw in_proj_qkv output.
//
// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
// The convolution is ApxInf's `causal_depthwise_conv.metal` (same tap order:
// tap `i` reads the row `t + i - (S - 1)`, rows before 0 are zero, and the
// accumulation is FP32 in tap order). The chain then stays in FP32 and each
// output is rounded once: SiLU on the accumulated value, the l2norm
// `x * rsqrt(sum(x^2) + EPSILON)` with q's `Dk ** -0.5` and k's unit scale over
// the SiLU values, and the BF16 write. A separately composed chain rounds
// between those steps as well -- conv, then SiLU, then each normed output --
// because each is its own BF16 node; this kernel removes those two
// intermediate roundings.
//
// Placeholders are substituted by `GdnFusedPrep::new` in Rust: the templates
// come from `MetalKernel::call` (`InT`, `C`, `Dk`, `Dv`, `Hk`, `Hv`, `S`) and
// `EPSILON`, `QSCALE`, `KSCALE` are replaced with f32 literals.
//
// Grid (32, T, Hv), threadgroup (32, TT, 1): one SIMD group per (prompt row,
// value head), each lane owning Dk / 32 columns of q and k and Dv / 32 columns
// of v. The value head `hv` selects its key head `hv / (Hv / Hk)`, so Hv may be
// a multiple of Hk; d3 has Hv == Hk == 16.
    const uint lane = thread_position_in_threadgroup.x;          // 0..31
    const int t = thread_position_in_grid.y;                     // the prompt row
    if (t >= length) return;
    const int hv = thread_position_in_grid.z;                    // the value head
    const int hk = hv / (Hv / Hk);
    const int key_dim = Hk * Dk;
    const int q_col = hk * Dk + int(lane) * (Dk / 32);           // this lane's first q channel
    const int k_col = key_dim + q_col;                            // the same position in k
    const int v_col = 2 * key_dim + hv * Dv + int(lane) * (Dv / 32);

    float q_sum = 0.0f, k_sum = 0.0f;
    float q_silu[Dk / 32], k_silu[Dk / 32], v_silu[Dv / 32];
#pragma unroll
    for (int i = 0; i < Dk / 32; ++i) {
        float q_acc = 0.0f, k_acc = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const int row = t + s - (S - 1);
            if (row >= 0) {
                q_acc += static_cast<float>(x[row * C + q_col + i]) * static_cast<float>(weight[(q_col + i) * S + s]);
                k_acc += static_cast<float>(x[row * C + k_col + i]) * static_cast<float>(weight[(k_col + i) * S + s]);
            }
        }
        q_silu[i] = q_acc / (1.0f + exp(-q_acc));
        k_silu[i] = k_acc / (1.0f + exp(-k_acc));
        q_sum += q_silu[i] * q_silu[i];
        k_sum += k_silu[i] * k_silu[i];
    }
#pragma unroll
    for (int i = 0; i < Dv / 32; ++i) {
        float v_acc = 0.0f;
#pragma unroll
        for (int s = 0; s < S; ++s) {
            const int row = t + s - (S - 1);
            if (row >= 0) {
                v_acc += static_cast<float>(x[row * C + v_col + i]) * static_cast<float>(weight[(v_col + i) * S + s]);
            }
        }
        v_silu[i] = v_acc / (1.0f + exp(-v_acc));
    }
    q_sum = simd_sum(q_sum);
    k_sum = simd_sum(k_sum);
    const float q_inv = metal::precise::rsqrt(q_sum + EPSILON);
    const float k_inv = metal::precise::rsqrt(k_sum + EPSILON);
    const uint q_base = (uint(t) * Hk + uint(hk)) * Dk + uint(lane) * (Dk / 32);
    const uint v_base = (uint(t) * Hv + uint(hv)) * Dv + uint(lane) * (Dv / 32);
#pragma unroll
    for (int i = 0; i < Dk / 32; ++i) {
        q[q_base + i] = static_cast<InT>(q_silu[i] * q_inv * QSCALE);   // the one rounding of q
        k[q_base + i] = static_cast<InT>(k_silu[i] * k_inv * KSCALE);
    }
#pragma unroll
    for (int i = 0; i < Dv / 32; ++i) {
        v[v_base + i] = static_cast<InT>(v_silu[i]);
    }

// Portions adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin; see NOTICE.
// Four value rows per SIMD group; adapts the MLX-VLM 0.7.4 gated delta kernel (MIT, Prince Canuma).

            uint n = thread_position_in_grid.z;
            uint b = n / Hv, hv = n % Hv, hk = hv / (Hv / Hk);
            uint lane = thread_position_in_threadgroup.x;
            uint dv = 4 * thread_position_in_grid.y;
            int L = length;
            float4 s[Dk / 32];
            for (int i = 0; i < Dk / 32; ++i) {
                uint dk = (Dk / 32) * lane + i;
                s[i] = float4(state[(n * Dv + dv) * Dk + dk],
                              state[(n * Dv + dv + 1) * Dk + dk],
                              state[(n * Dv + dv + 2) * Dk + dk],
                              state[(n * Dv + dv + 3) * Dk + dk]);
            }
            for (int t = 0; t < L; ++t) {
                uint qbase = ((b * L + t) * Hk + hk) * Dk;
                uint vbase = ((b * L + t) * Hv + hv) * Dv + dv;
                float decay = float(g[(b * L + t) * Hv + hv]);
                float4 kv = 0.0f;
                for (int i = 0; i < Dk / 32; ++i) {
                    uint dk = (Dk / 32) * lane + i;
                    s[i] = s[i] * decay;
                    kv += s[i] * float(k[qbase + dk]);
                }
                kv = simd_sum(kv);
                float4 value = float4(v[vbase], v[vbase + 1], v[vbase + 2], v[vbase + 3]);
                float4 delta = (value - kv) * float(beta[(b * L + t) * Hv + hv]);
                float4 out = 0.0f;
                for (int i = 0; i < Dk / 32; ++i) {
                    uint dk = (Dk / 32) * lane + i;
                    s[i] = s[i] + float(k[qbase + dk]) * delta;
                    out += s[i] * float(q[qbase + dk]);
                }
                out = simd_sum(out);
                if (lane == 0) {
                    y[vbase] = InT(out.x);
                    y[vbase + 1] = InT(out.y);
                    y[vbase + 2] = InT(out.z);
                    y[vbase + 3] = InT(out.w);
                }
            }
            for (int i = 0; i < Dk / 32; ++i) {
                uint dk = (Dk / 32) * lane + i;
                next_state[(n * Dv + dv) * Dk + dk] = s[i].x;
                next_state[(n * Dv + dv + 1) * Dk + dk] = s[i].y;
                next_state[(n * Dv + dv + 2) * Dk + dk] = s[i].z;
                next_state[(n * Dv + dv + 3) * Dk + dk] = s[i].w;
            }

// Adapted from EngineTailor campaign/clef-flash9b-m1pro 4030047b (C14). Copyright 2026 Haiyan Qin. MIT; see NOTICE.
// s12: eight value rows per SIMD group, each lane holding two rows of Dk / 8 keys; adapts the
// MLX-VLM 0.7.4 gated delta kernel (MIT, Prince Canuma).
            uint n = thread_position_in_grid.z;
            uint b = n / Hv, hv = n % Hv, hk = hv / (Hv / Hk);
            uint lane = thread_position_in_threadgroup.x;
            uint kg = lane % 8;
            uint dv = 8 * thread_position_in_grid.y + 2 * (lane / 8);
            int L = length;
            float2 s[Dk / 8];
            for (int i = 0; i < Dk / 8; ++i) {
                uint dk = (Dk / 8) * kg + i;
                s[i] = float2(state[(n * Dv + dv) * Dk + dk],
                              state[(n * Dv + dv + 1) * Dk + dk]);
            }
            // Each word holds two BF16 values, widened exactly at use.
            const device uint* kw = reinterpret_cast<const device uint*>(k);
            const device uint* qw = reinterpret_cast<const device uint*>(q);
            uint kc[Dk / 16], qc[Dk / 16];
            uint qbase0 = ((b * L + 0) * Hk + hk) * Dk + (Dk / 8) * kg;
            uint vbase0 = ((b * L + 0) * Hv + hv) * Dv + dv;
            for (int i = 0; i < Dk / 16; ++i) {
                kc[i] = kw[qbase0 / 2 + i];
                qc[i] = qw[qbase0 / 2 + i];
            }
            float gc = float(g[(b * L + 0) * Hv + hv]);
            InT bc = beta[(b * L + 0) * Hv + hv];
            InT vc0 = v[vbase0];
            InT vc1 = v[vbase0 + 1];
            for (int t = 0; t < L; ++t) {
                int tn = min(t + 1, L - 1);
                uint qnext = ((b * L + tn) * Hk + hk) * Dk + (Dk / 8) * kg;
                uint vnext = ((b * L + tn) * Hv + hv) * Dv + dv;
                uint kn[Dk / 16], qn[Dk / 16];
                for (int i = 0; i < Dk / 16; ++i) {
                    kn[i] = kw[qnext / 2 + i];
                    qn[i] = qw[qnext / 2 + i];
                }
                float gn = float(g[(b * L + tn) * Hv + hv]);
                InT bn = beta[(b * L + tn) * Hv + hv];
                InT vn0 = v[vnext];
                InT vn1 = v[vnext + 1];

                uint vbase = ((b * L + t) * Hv + hv) * Dv + dv;
                float decay = gc;
                float2 kv = 0.0f;
                for (int i = 0; i < Dk / 8; ++i) {
                    s[i] = s[i] * decay;
                    kv += s[i] * as_type<float>((i & 1) ? (kc[i / 2] & 0xffff0000u) : (kc[i / 2] << 16));
                }
                for (ushort m = 1; m < 8; m <<= 1) {
                    kv.x += simd_shuffle_xor(kv.x, m);
                    kv.y += simd_shuffle_xor(kv.y, m);
                }
                float2 value = float2(vc0, vc1);
                float2 delta = (value - kv) * float(bc);
                float2 out = 0.0f;
                for (int i = 0; i < Dk / 8; ++i) {
                    s[i] = s[i] + as_type<float>((i & 1) ? (kc[i / 2] & 0xffff0000u) : (kc[i / 2] << 16)) * delta;
                    out += s[i] * as_type<float>((i & 1) ? (qc[i / 2] & 0xffff0000u) : (qc[i / 2] << 16));
                }
                for (ushort m = 1; m < 8; m <<= 1) {
                    out.x += simd_shuffle_xor(out.x, m);
                    out.y += simd_shuffle_xor(out.y, m);
                }
                if (kg == 0) {
                    y[vbase] = InT(out.x);
                    y[vbase + 1] = InT(out.y);
                }
                for (int i = 0; i < Dk / 16; ++i) {
                    kc[i] = kn[i];
                    qc[i] = qn[i];
                }
                gc = gn;
                bc = bn;
                vc0 = vn0;
                vc1 = vn1;
            }
            for (int i = 0; i < Dk / 8; ++i) {
                uint dk = (Dk / 8) * kg + i;
                next_state[(n * Dv + dv) * Dk + dk] = s[i].x;
                next_state[(n * Dv + dv + 1) * Dk + dk] = s[i].y;
            }

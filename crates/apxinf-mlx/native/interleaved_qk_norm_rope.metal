// Adapted from mflux f0bf9cf0 models/qwen21/model/qwen21_transformer/qwen21_fused_kernels.py.
// Copyright (c) 2026 Filip Strand. MIT; see NOTICE.
constexpr float eps = 1e-6;
constexpr uint blocks = (D + 127) / 128;
constexpr uint pairs_per_lane = 2 * blocks;
uint row = thread_position_in_grid.x / 32;
if (row >= B * L * H) return;
uint lane = thread_index_in_simdgroup;
uint h = row % H;
uint l = (row / H) % L;
uint b = row / (H * L);
size_t qb = b * q_strides[0] + l * q_strides[1] + h * D * q_strides[2];
size_t kb = b * k_strides[0] + l * k_strides[1] + h * D * k_strides[2];

float qv[4 * blocks], kv[4 * blocks];
float sq = 0.0f, sk = 0.0f;
// Match native RMSNorm: four adjacent values per lane, then subgroup totals.
for (uint block = 0; block < blocks; ++block) {
    float bq = 0.0f, bk = 0.0f;
    for (uint j = 0; j < 4; ++j) {
        uint c = block * 128 + lane * 4 + j;
        uint i = block * 4 + j;
        qv[i] = c < D ? float(TQ(q[qb + c * q_strides[2]])) : 0.0f;
        kv[i] = c < D ? float(TK(k[kb + c * k_strides[2]])) : 0.0f;
        bq += qv[i] * qv[i];
        bk += kv[i] * kv[i];
    }
    bq = simd_sum(bq);
    bk = simd_sum(bk);
    if (blocks == 1 || lane == block) {
        sq = bq;
        sk = bk;
    }
}
float rrq = metal::precise::rsqrt((blocks == 1 ? sq : simd_sum(sq)) / float(D) + eps);
float rrk = metal::precise::rsqrt((blocks == 1 ? sk : simd_sum(sk)) / float(D) + eps);
size_t out = ((size_t(b) * H + h) * L + l) * D;

for (uint pair = 0; pair < pairs_per_lane; ++pair) {
    #pragma STDC FP_CONTRACT OFF
    // Composed RoPE rounds the products before its separate add/subtract.
    uint c = (pair / 2) * 128 + lane * 4 + (pair % 2) * 2;
    if (c >= D) continue;
    uint p = c / 2;
    // MLX RMSNorm promotes x/weight together, rounds normalized x, then
    // rounds the weighted result before the FP32 rotary arithmetic.
    float nqa = float(TQ(float(TQ(qv[2 * pair] * rrq)) * float(TQ(wq[c * wq_strides[0]]))));
    float nqb = float(TQ(float(TQ(qv[2 * pair + 1] * rrq)) * float(TQ(wq[(c + 1) * wq_strides[0]]))));
    float nka = float(TK(float(TK(kv[2 * pair] * rrk)) * float(TK(wk[c * wk_strides[0]]))));
    float nkb = float(TK(float(TK(kv[2 * pair + 1] * rrk)) * float(TK(wk[(c + 1) * wk_strides[0]]))));
    float cosine = float(cos_t[l * cos_t_strides[0] + p * cos_t_strides[1]]);
    float sine = float(sin_t[l * sin_t_strides[0] + p * sin_t_strides[1]]);
    out_q[out + c] = TQ(nqa * cosine - nqb * sine);
    out_q[out + c + 1] = TQ(nqa * sine + nqb * cosine);
    out_k[out + c] = TK(nka * cosine - nkb * sine);
    out_k[out + c + 1] = TK(nka * sine + nkb * cosine);
}

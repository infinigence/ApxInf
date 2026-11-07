// Adapted from MLX-VLM 0.7.4 rope_utils.py _mrope_apply_kernel: half-split pairing,
// two-dimensional positions. Copyright (c) 2025 Prince Canuma. MIT; see NOTICE.
uint elem = thread_position_in_grid.x;

const int half_dim = ROTARY_DIM / 2;
const int q_bsz = x_shape[0];
const int q_heads = x_shape[1];
const int q_len = x_shape[2];
const int q_dim = x_shape[3];
const int slots = half_dim + q_dim - ROTARY_DIM;
const int work_size = q_bsz * q_heads * q_len * slots;

if (elem >= uint(work_size)) {
    return;
}

int local = int(elem);
int slot = local % slots;
int tmp = local / slots;
int t = tmp % q_len;
tmp = tmp / q_len;
int h = tmp % q_heads;
int b = tmp / q_heads;
int base = ((b * q_heads + h) * q_len + t) * q_dim;

if (slot >= half_dim) {
    int pass_d = ROTARY_DIM + slot - half_dim;
    int pass_idx = base + pass_d;
    x_out[pass_idx] = x[pass_idx];
    return;
}

int freq_idx = slot;
int d = freq_idx;
int pair_d = d + half_dim;
float pos = static_cast<float>(position_ids[b * q_len + t]);
float angle = pos * static_cast<float>(inv_freq[freq_idx]);
float c = metal::cos(angle);
float s = metal::sin(angle);

int idx = base + d;
float xv = static_cast<float>(x[idx]);
float xp = static_cast<float>(x[base + pair_d]);
x_out[idx] = static_cast<T>(xv * c - xp * s);
x_out[base + pair_d] = static_cast<T>(xp * c + xv * s);

const uint pair = thread_position_in_grid.x;
const uint half_dim = DIM / 2;
if (pair >= (Q_HEADS + K_HEADS) * half_dim) return;
const uint head = pair / half_dim;
const bool is_q = head < Q_HEADS;
const uint column = pair % half_dim;
const uint offset = (is_q ? head : head - Q_HEADS) * DIM + column;
const T first = is_q ? q[offset] : k[offset];
const T second = is_q ? q[offset + half_dim] : k[offset + half_dim];
const T negative_second = static_cast<T>(-second);
const T first_left = static_cast<T>(first * cos_row[column]);
const T first_right = static_cast<T>(negative_second * sin_row[column]);
const T second_left = static_cast<T>(second * cos_row[column + half_dim]);
const T second_right = static_cast<T>(first * sin_row[column + half_dim]);
const T first_result = static_cast<T>(first_left + first_right);
const T second_result = static_cast<T>(second_left + second_right);
if (is_q) {
    q_out[offset] = first_result;
    q_out[offset + half_dim] = second_result;
} else {
    k_out[offset] = first_result;
    k_out[offset + half_dim] = second_result;
}

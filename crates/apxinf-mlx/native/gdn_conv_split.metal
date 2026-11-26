// Split the gated-delta convolution output [1, L, 2*key_dim + value_dim]
// into q [1, L, Hk, Dk], k [1, L, Hk, Dk], v [1, L, Hv, Dv] in one pass.
// Pure layout: every output element is the same BF16 bit as the input element
// it comes from, so the composed q/k RMSNorm+scale and the recurrence see
// exactly the values the slice+reshape path produced. One gather kernel
// replaces the three strided gathers the consumers would otherwise each do
// (RMSNorm `set_output`, and the recurrence's row-contiguous input copy).
// template args: InT (dtype), KEY_DIM, VALUE_DIM, CONV_DIM (ints);
// input x (device), input length (constant int&); outputs q, k, v.
    const uint idx = thread_position_in_grid.x;
    const uint conv_dim = uint(CONV_DIM);
    const uint total = uint(length) * conv_dim;
    if (idx >= total) return;
    const uint t = idx / conv_dim;
    const uint c = idx % conv_dim;
    const InT value = x[idx];
    if (c < uint(KEY_DIM)) {
        q[t * uint(KEY_DIM) + c] = value;
    } else if (c < 2u * uint(KEY_DIM)) {
        k[t * uint(KEY_DIM) + c - uint(KEY_DIM)] = value;
    } else {
        v[t * uint(VALUE_DIM) + c - 2u * uint(KEY_DIM)] = value;
    }

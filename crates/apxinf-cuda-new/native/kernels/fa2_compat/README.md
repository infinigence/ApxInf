# FlashAttention build compatibility

The vendored FlashAttention files remain byte-for-byte copies of
`Dao-AILab/flash-attention` tag `v2.7.4.post1`, except for the documented
direct-E4M3 operator extension. ApxInf does not link libtorch, so this
directory provides only the three declarations that the upstream headers
reference during a dropout-disabled inference build:

- the `at::PhiloxCudaState` carrier type;
- device-side extraction of its seed and offset;
- CUDA error checking used by the launch templates.

This code is maintained by ApxInf and is deliberately kept outside the
vendored source tree. It is not a general PyTorch compatibility layer.

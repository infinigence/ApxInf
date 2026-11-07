// MUSA 5.1 cannot reliably terminate an empty capture after an operator
// refuses it. A real no-op node makes error cleanup independent of that case.
__global__ void capture_anchor_kernel() {}

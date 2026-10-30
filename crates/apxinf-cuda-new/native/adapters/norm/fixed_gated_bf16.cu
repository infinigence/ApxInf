#include "../../kernels/norm/fixed_gated_bf16.cuh"

extern "C" cudaError_t apxinf_norm_gated_rms_silu_bf16(
    const void* x, const void* z, const void* weight, void* out,
    int rows, int cols, int z_heads, int64_t z_row_stride,
    int64_t z_col_offset, float eps, cudaStream_t stream) {
  if (x == nullptr || z == nullptr || weight == nullptr || out == nullptr ||
      rows <= 0 || cols <= 0 || z_heads <= 0 || rows % z_heads != 0 ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  if (cols == 128) {
    gated_rms_silu_bf16_warp4_kernel<<<(rows + GRS_ROWS - 1) / GRS_ROWS, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(z),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(out),
        rows, cols, z_heads, z_row_stride, z_col_offset, eps);
    return cudaGetLastError();
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  gated_rms_silu_bf16_kernel<<<rows, 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(z),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(out),
      cols, z_heads, z_row_stride, z_col_offset, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_rms_norm_plus1_bf16(
    const void* input, const void* weight, void* output,
    int rows, int cols, float eps, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  rms_norm_plus1_bf16_kernel<<<rows, 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_add_rms_norm_plus1_bf16(
    const void* a, const void* b, const void* weight, void* sum_out,
    void* output, int rows, int cols, float eps, cudaStream_t stream) {
  if (a == nullptr || b == nullptr || weight == nullptr || sum_out == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  // The tested prefill shape uses only the exact vector mean branch.
  if (rows >= 16 && cols == 2560) {
    add_rms_norm_plus1_bf16_kernel<true><<<rows, 256, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(a),
        static_cast<const __nv_bfloat16*>(b),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(sum_out),
        static_cast<__nv_bfloat16*>(output), cols, eps);
    return cudaGetLastError();
  }
  add_rms_norm_plus1_bf16_kernel<false><<<rows, 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(sum_out),
      static_cast<__nv_bfloat16*>(output), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_adaln_rms_norm_bf16(
    const void* x, const void* weight, const void* scale, const void* shift,
    void* out, int rows, int cols, float eps, cudaStream_t stream) {
  if (x == nullptr || weight == nullptr || scale == nullptr ||
      shift == nullptr || out == nullptr || rows <= 0 || cols <= 0 ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(cols) * sizeof(float);
  adaln_rms_norm_bf16_kernel<<<rows, 256, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(x),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(scale),
      static_cast<const __nv_bfloat16*>(shift),
      static_cast<__nv_bfloat16*>(out), cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_adaln_gate_residual_bf16(
    const void* proj, const void* residual, const void* gate, void* out,
    int64_t count, int cols, cudaStream_t stream) {
  if (proj == nullptr || residual == nullptr || gate == nullptr ||
      out == nullptr || count <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  adaln_gate_residual_bf16_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(proj),
      static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<__nv_bfloat16*>(out), count, cols);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_adaln_gate_residual_rms_bf16(
    const void* proj, const void* residual, const void* gate, const void* weight,
    const void* scale, const void* shift, void* hidden, void* normalized,
    int rows, int cols, float eps, cudaStream_t stream) {
  if (!proj || !residual || !gate || !weight || !scale || !shift || !hidden || !normalized ||
      rows <= 0 || cols <= 0 || cols > 8192 || !(eps > 0.0f)) return cudaErrorInvalidValue;
  adaln_gate_residual_rms_bf16_kernel<<<rows, 256, cols * sizeof(float), stream>>>(
      static_cast<const __nv_bfloat16*>(proj), static_cast<const __nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(gate), static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(scale), static_cast<const __nv_bfloat16*>(shift),
      static_cast<__nv_bfloat16*>(hidden), static_cast<__nv_bfloat16*>(normalized), cols, eps);
  return cudaGetLastError();
}

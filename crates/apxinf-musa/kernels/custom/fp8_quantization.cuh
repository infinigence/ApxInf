struct alignas(16) Fp8SourcePack { __mt_bfloat16 values[8]; };
struct alignas(8) Fp8OutputPack { __mt_fp8_e4m3 values[8]; };

__global__ void quantize_bf16_e4m3_kernel(const __mt_bfloat16* input,
    __mt_fp8_e4m3* output, size_t count, float inverse_scale) {
  const size_t start = (size_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  const size_t stride = size_t(gridDim.x) * blockDim.x * 8;
  for (size_t offset = start; offset < count; offset += stride) {
    if (offset + 8 <= count) {
      const Fp8SourcePack source = *reinterpret_cast<const Fp8SourcePack*>(input + offset);
      Fp8OutputPack destination;
      #pragma unroll
      for (int lane = 0; lane < 8; ++lane) {
        destination.values[lane] = __mt_fp8_e4m3(
            fminf(448.0f, fmaxf(-448.0f, float(source.values[lane]) * inverse_scale)));
      }
      *reinterpret_cast<Fp8OutputPack*>(output + offset) = destination;
    } else {
      for (size_t index = offset; index < count; ++index) {
        output[index] = __mt_fp8_e4m3(
            fminf(448.0f, fmaxf(-448.0f, float(input[index]) * inverse_scale)));
      }
    }
  }
}

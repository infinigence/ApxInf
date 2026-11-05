// Pillow 12.3.0 RGB8 bicubic axis arithmetic. Bounds and 22-bit integer
// coefficients are prepared and validated by the model-neutral Rust facade.
__global__ void pillow_bicubic_u8_axis_kernel(
    const uint8_t* input, uint8_t* output,
    const int64_t* input_offsets, const int64_t* output_offsets,
    const int32_t* bounds, const int32_t* weights, int ksize,
    int in_w, int in_h, int out_w, int out_h, bool horizontal) {
  const int frame = blockIdx.y;
  const int64_t pixel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (pixel >= static_cast<int64_t>(out_w) * out_h) return;
  const int y = static_cast<int>(pixel / out_w);
  const int x = static_cast<int>(pixel - static_cast<int64_t>(y) * out_w);
  const int coord = horizontal ? x : y;
  const int first = bounds[coord * 2], taps = bounds[coord * 2 + 1];
  const int32_t* w = weights + coord * ksize;
  const uint8_t* src = input + input_offsets[frame];
  uint8_t* dst = output + output_offsets[frame] + static_cast<int64_t>(pixel) * 3;
  int r = 1 << 21, g = r, b = r;
  for (int k = 0; k < taps; ++k) {
    const int yy = horizontal ? y : first + k;
    const int xx = horizontal ? first + k : x;
    const uint8_t* p = src + (static_cast<int64_t>(yy) * in_w + xx) * 3;
    const int v = w[k];
    r += p[0] * v; g += p[1] * v; b += p[2] * v;
  }
  r >>= 22; g >>= 22; b >>= 22;
  dst[0] = static_cast<uint8_t>(r < 0 ? 0 : r > 255 ? 255 : r);
  dst[1] = static_cast<uint8_t>(g < 0 ? 0 : g > 255 ? 255 : g);
  dst[2] = static_cast<uint8_t>(b < 0 ? 0 : b > 255 ? 255 : b);
}

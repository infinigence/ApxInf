#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace apxinf::norm::kernels {

template <class T>
__device__ float to_float(T value);

template <>
__device__ inline float to_float(__half value) {
  return __half2float(value);
}

template <>
__device__ inline float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T>
__device__ T from_float(float value);

template <>
__device__ inline __half from_float(float value) {
  return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 from_float(float value) {
  return __float2bfloat16(value);
}

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = 16; offset > 0; offset >>= 1)
    value += __shfl_down_sync(0xffffffff, value, offset);
  return value;
}

// One block per row; `scratch` must hold blockDim.x / 32 floats. The final
// barrier publishes scratch[0], but does not protect the reads that follow it.
// Callers must consume the result and synchronize all block threads before
// reusing the same scratch storage.
__device__ __forceinline__ float block_sum_parallel_unsafe(float value,
                                                            float* scratch) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;
  value = warp_sum(value);
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  if (warp == 0) {
    value = lane < warps ? scratch[lane] : 0.0f;
    value = warp_sum(value);
    if (lane == 0) scratch[0] = value;
  }
  __syncthreads();
  return scratch[0];
}

// Every kernel below takes (rows, cols) in that order.  The legacy backend
// carries two same-named overloads that differ only by `int` versus
// `uint32_t`, with the row and column arguments swapped between them; the port
// deliberately keeps a single signature so that mistake cannot recur.

template <class T, bool Silu = false>
__global__ void channel_l2_norm(const T* input, const T* weight, T* output,
                               int rows, int cols, float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float x = to_float(input[int64_t(row) * cols + col]);
    sum = __fadd_rn(sum, __fmul_rn(x, x));
  }
  sum = block_sum_parallel_unsafe(sum, scratch);
  const float norm = to_float(from_float<T>(__fsqrt_rn(sum)));
  const float denominator = to_float(from_float<T>(fmaxf(norm, eps)));
  const float scale = __fsqrt_rn(static_cast<float>(cols));
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t offset = int64_t(row) * cols + col;
    const float divided = to_float(from_float<T>(__fdiv_rn(to_float(input[offset]), denominator)));
    const float scaled = to_float(from_float<T>(__fmul_rn(divided, scale)));
    const T rounded = from_float<T>(__fmul_rn(scaled, to_float(weight[col])));
    if constexpr (Silu) {
      const float value = to_float(rounded);
      output[offset] = from_float<T>(value / (1.0f + expf(-value)));
    } else {
      output[offset] = rounded;
    }
  }
}

// Preserve the 256-thread channel-L2 reduction tree with one warp per row.
// Small row counts use the original kernel to avoid underfilling the GPU.
template <class T, int Cols, bool Silu = false>
__global__ void channel_l2_norm_warp(const T* input, const T* weight,
                                     T* output, int rows, float eps) {
  const int64_t row = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  float partial[8];
#pragma unroll
  for (int warp = 0; warp < 8; ++warp) {
    float sum = 0.0f;
    if (warp * 32 < Cols) {
      for (int c = warp * 32 + lane; c < Cols; c += 256) {
        const float x = to_float(input[row * Cols + c]);
        sum = __fadd_rn(sum, __fmul_rn(x, x));
      }
      partial[warp] = warp_sum(sum);
    } else {
      partial[warp] = 0.0f;
    }
  }
  float sum = __fadd_rn(
      __fadd_rn(__fadd_rn(partial[0], partial[4]), __fadd_rn(partial[2], partial[6])),
      __fadd_rn(__fadd_rn(partial[1], partial[5]), __fadd_rn(partial[3], partial[7])));
  sum = __shfl_sync(0xffffffff, sum, 0);
  const float norm = to_float(from_float<T>(__fsqrt_rn(sum)));
  const float denominator = to_float(from_float<T>(fmaxf(norm, eps)));
  const float scale = __fsqrt_rn(static_cast<float>(Cols));
  for (int c = lane; c < Cols; c += 32) {
    const int64_t index = row * Cols + c;
    const float divided = to_float(from_float<T>(__fdiv_rn(to_float(input[index]), denominator)));
    const float scaled = to_float(from_float<T>(__fmul_rn(divided, scale)));
    const T rounded = from_float<T>(__fmul_rn(scaled, to_float(weight[c])));
    if constexpr (Silu) {
      const float value = to_float(rounded);
      output[index] = from_float<T>(value / (1.0f + expf(-value)));
    } else {
      output[index] = rounded;
    }
  }
}

// Four lanes per 96/192-channel row. Packed high shuffle levels and local low
// levels preserve the original warp32 sum and every BF16 rounding boundary.
struct alignas(16) ChannelL2Pack8 { __nv_bfloat16 x[8]; };
template<int Cols>
__global__ void channel_l2_silu_packed_bf16(const __nv_bfloat16* input,const __nv_bfloat16* weight,__nv_bfloat16* output,int rows,float eps){
 static_assert(Cols == 96 || Cols == 192);
 const int64_t row=int64_t(blockIdx.x)*64+threadIdx.x/4;
 const int lane=threadIdx.x%4;
 const unsigned mask=__ballot_sync(0xffffffff,row<rows);
 if(row>=rows)return;
 float x[Cols/32][8],p[Cols/32][8];
 #pragma unroll
 for(int q=0;q<Cols/32;++q){
  auto a=*reinterpret_cast<const ChannelL2Pack8*>(input+row*Cols+q*32+lane*8);
  #pragma unroll
  for(int j=0;j<8;++j){x[q][j]=__bfloat162float(a.x[j]);p[q][j]=__fadd_rn(0.f,__fmul_rn(x[q][j],x[q][j]));}
  #pragma unroll
  for(int off=2;off>0;off/=2){
   #pragma unroll
   for(int j=0;j<8;++j)p[q][j]+=__shfl_down_sync(mask,p[q][j],off,4);
  }
  #pragma unroll
  for(int off=4;off>0;off/=2){
   #pragma unroll
   for(int j=0;j<8-off;++j)p[q][j]+=p[q][j+off];
  }
 }
 float sum;
 if constexpr (Cols == 96) {
  sum=__fadd_rn(__fadd_rn(__fadd_rn(p[0][0],0.f),__fadd_rn(p[2][0],0.f)),__fadd_rn(__fadd_rn(p[1][0],0.f),0.f));
 } else {
  sum=__fadd_rn(__fadd_rn(__fadd_rn(p[0][0],p[4][0]),__fadd_rn(p[2][0],0.f)),__fadd_rn(__fadd_rn(p[1][0],p[5][0]),__fadd_rn(p[3][0],0.f)));
 }
 sum=__shfl_sync(mask,sum,0,4);
 const float norm=__bfloat162float(__float2bfloat16(__fsqrt_rn(sum)));
 const float denominator=__bfloat162float(__float2bfloat16(fmaxf(norm,eps)));
 const float scale=__fsqrt_rn(float(Cols));
 #pragma unroll
 for(int q=0;q<Cols/32;++q){
  auto w=*reinterpret_cast<const ChannelL2Pack8*>(weight+q*32+lane*8);ChannelL2Pack8 out;
  #pragma unroll
  for(int j=0;j<8;++j){
   float divided=__bfloat162float(__float2bfloat16(__fdiv_rn(x[q][j],denominator)));
   float scaled=__bfloat162float(__float2bfloat16(__fmul_rn(divided,scale)));
   float value=__bfloat162float(__float2bfloat16(__fmul_rn(scaled,__bfloat162float(w.x[j]))));
   out.x[j]=__float2bfloat16(value/(1.f+expf(-value)));
  }
  *reinterpret_cast<ChannelL2Pack8*>(output+row*Cols+q*32+lane*8)=out;
 }
}

template<int Cols, bool Residual>
__global__ void channel_l2_silu_update_packed_bf16(const __nv_bfloat16* input,const __nv_bfloat16* bias,const __nv_bfloat16* residual,__nv_bfloat16* hidden,const __nv_bfloat16* weight,__nv_bfloat16* output,int rows,float eps){
 static_assert(Cols == 96 || Cols == 192);
 const int64_t row=int64_t(blockIdx.x)*64+threadIdx.x/4;
 const int lane=threadIdx.x%4;
 const unsigned mask=__ballot_sync(0xffffffff,row<rows);
 if(row>=rows)return;
 float x[Cols/32][8],p[Cols/32][8];
 #pragma unroll
 for(int q=0;q<Cols/32;++q){
  auto a=*reinterpret_cast<const ChannelL2Pack8*>(input+row*Cols+q*32+lane*8);
  ChannelL2Pack8 b{},rv{};
  if(bias)b=*reinterpret_cast<const ChannelL2Pack8*>(bias+q*32+lane*8);
  if constexpr(Residual)rv=*reinterpret_cast<const ChannelL2Pack8*>(residual+row*Cols+q*32+lane*8);
  #pragma unroll
  for(int j=0;j<8;++j){
   float value=__bfloat162float(a.x[j]);
   if(bias)value=__fadd_rn(value,__bfloat162float(b.x[j]));
   a.x[j]=__float2bfloat16_rn(value);
   if constexpr(Residual)a.x[j]=__float2bfloat16_rn(__fadd_rn(__bfloat162float(a.x[j]),__bfloat162float(rv.x[j])));
  }
  *reinterpret_cast<ChannelL2Pack8*>(hidden+row*Cols+q*32+lane*8)=a;
  #pragma unroll
  for(int j=0;j<8;++j){x[q][j]=__bfloat162float(a.x[j]);p[q][j]=__fadd_rn(0.f,__fmul_rn(x[q][j],x[q][j]));}
  #pragma unroll
  for(int off=2;off>0;off/=2){
   #pragma unroll
   for(int j=0;j<8;++j)p[q][j]+=__shfl_down_sync(mask,p[q][j],off,4);
  }
  #pragma unroll
  for(int off=4;off>0;off/=2){
   #pragma unroll
   for(int j=0;j<8-off;++j)p[q][j]+=p[q][j+off];
  }
 }
 float sum;
 if constexpr (Cols == 96) {
  sum=__fadd_rn(__fadd_rn(__fadd_rn(p[0][0],0.f),__fadd_rn(p[2][0],0.f)),__fadd_rn(__fadd_rn(p[1][0],0.f),0.f));
 } else {
  sum=__fadd_rn(__fadd_rn(__fadd_rn(p[0][0],p[4][0]),__fadd_rn(p[2][0],0.f)),__fadd_rn(__fadd_rn(p[1][0],p[5][0]),__fadd_rn(p[3][0],0.f)));
 }
 sum=__shfl_sync(mask,sum,0,4);
 const float norm=__bfloat162float(__float2bfloat16(__fsqrt_rn(sum)));
 const float denominator=__bfloat162float(__float2bfloat16(fmaxf(norm,eps)));
 const float scale=__fsqrt_rn(float(Cols));
 #pragma unroll
 for(int q=0;q<Cols/32;++q){
  auto w=*reinterpret_cast<const ChannelL2Pack8*>(weight+q*32+lane*8);ChannelL2Pack8 out;
  #pragma unroll
  for(int j=0;j<8;++j){
   float divided=__bfloat162float(__float2bfloat16(__fdiv_rn(x[q][j],denominator)));
   float scaled=__bfloat162float(__float2bfloat16(__fmul_rn(divided,scale)));
   float value=__bfloat162float(__float2bfloat16(__fmul_rn(scaled,__bfloat162float(w.x[j]))));
   out.x[j]=__float2bfloat16(value/(1.f+expf(-value)));
  }
  *reinterpret_cast<ChannelL2Pack8*>(output+row*Cols+q*32+lane*8)=out;
 }
}

// Bias-only materialization for the staged fallback; no scratch allocation.
__global__ void channel_bias_bf16(const __nv_bfloat16* input,const __nv_bfloat16* bias,
                                __nv_bfloat16* hidden,int64_t count,int cols) {
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
  float value=__bfloat162float(input[i]);
  if(bias)value=__fadd_rn(value,__bfloat162float(bias[i%cols]));
  hidden[i]=__float2bfloat16_rn(value);
 }
}
__global__ void channel_bias_packed_bf16(const ChannelL2Pack8* input,const ChannelL2Pack8* bias,
                                       ChannelL2Pack8* hidden,int64_t count,int cols) {
 for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
  auto a=input[i];ChannelL2Pack8 b{};if(bias)b=bias[i%cols];
  #pragma unroll
  for(int j=0;j<8;++j){float value=__bfloat162float(a.x[j]);if(bias)value=__fadd_rn(value,__bfloat162float(b.x[j]));a.x[j]=__float2bfloat16_rn(value);}
  hidden[i]=a;
 }
}

template <class T>
__global__ void rms_norm(const T* input, const T* weight, T* output, int rows,
                         int cols, float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = to_float(input[static_cast<int64_t>(row) * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    output[index] =
        from_float<T>(to_float(input[index]) * inverse_rms * to_float(weight[col]));
  }
}

// Eight independent rows per CTA. Keep the four 32-element partial sums
// and their combination in the same order as the 256-thread RMS kernel.
// Eight BF16 values per lane; preserve the scalar 128-column reduction tree.
// Call only with 16-byte aligned input, weight, and output.
template<int V> struct alignas(V*2) PlainRmsPack { __nv_bfloat16 x[V]; };
template<int V> __global__ void rms_norm_128_packed(const __nv_bfloat16* input,const __nv_bfloat16* weight,__nv_bfloat16* output,int rows,float eps){
 constexpr int W=32/V;
 const int64_t row=int64_t(blockIdx.x)*(256/W)+threadIdx.x/W;
 const int lane=threadIdx.x%W;
 const unsigned mask=__ballot_sync(0xffffffff,row<rows);
 if(row>=rows)return;
 float x[4][V],p[4][V],y[4][V];
 #pragma unroll
 for(int q=0;q<4;++q){
  auto a=*reinterpret_cast<const PlainRmsPack<V>*>(input+row*128+q*32+lane*V);
  #pragma unroll
  for(int j=0;j<V;++j){x[q][j]=__bfloat162float(a.x[j]);p[q][j]=x[q][j]*x[q][j];}
  #pragma unroll
  for(int off=W/2;off>0;off/=2){
   #pragma unroll
   for(int j=0;j<V;++j)p[q][j]+=__shfl_down_sync(mask,p[q][j],off,W);
  }
  #pragma unroll
  for(int off=V/2;off>0;off/=2){
   #pragma unroll
   for(int j=0;j<V-off;++j)p[q][j]+=p[q][j+off];
  }
 }
 float sum=__fadd_rn(__fadd_rn(p[0][0],p[2][0]),__fadd_rn(p[1][0],p[3][0]));
 sum=__shfl_sync(mask,sum,0,W);
 const float inv=rsqrtf(sum/128+eps);
 #pragma unroll
 for(int q=0;q<4;++q){
  auto w=*reinterpret_cast<const PlainRmsPack<V>*>(weight+q*32+lane*V);
  #pragma unroll
  for(int j=0;j<V;++j)y[q][j]=__bfloat162float(__float2bfloat16(x[q][j]*inv*__bfloat162float(w.x[j])));
 }
 #pragma unroll
 for(int q=0;q<4;++q){PlainRmsPack<V> out;
  #pragma unroll
  for(int j=0;j<V;++j)out.x[j]=__float2bfloat16(y[q][j]);
  *reinterpret_cast<PlainRmsPack<V>*>(output+row*128+q*32+lane*V)=out;
 }
}

template <class T>
__global__ void rms_norm_128(const T* input, const T* weight, T* output,
                             int rows, float eps) {
  const int64_t row = int64_t(blockIdx.x) * 8 + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (row >= rows) return;
  const int64_t base = row * 128 + lane;
  const float x0 = to_float(input[base]);
  const float x1 = to_float(input[base + 32]);
  const float x2 = to_float(input[base + 64]);
  const float x3 = to_float(input[base + 96]);
  const float p0 = warp_sum(x0 * x0);
  const float p1 = warp_sum(x1 * x1);
  const float p2 = warp_sum(x2 * x2);
  const float p3 = warp_sum(x3 * x3);
  float sum = __fadd_rn(__fadd_rn(p0, p2), __fadd_rn(p1, p3));
  sum = __shfl_sync(0xffffffff, sum, 0);
  const float inverse_rms = rsqrtf(sum / 128 + eps);
  output[base] = from_float<T>(x0 * inverse_rms * to_float(weight[lane]));
  output[base + 32] = from_float<T>(x1 * inverse_rms * to_float(weight[lane + 32]));
  output[base + 64] = from_float<T>(x2 * inverse_rms * to_float(weight[lane + 64]));
  output[base + 96] = from_float<T>(x3 * inverse_rms * to_float(weight[lane + 96]));
}

template <class T>
__global__ void layer_norm(const T* input, const T* weight, const T* bias,
                           T* output, int rows, int cols, float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += to_float(input[static_cast<int64_t>(row) * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        to_float(input[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  // Finish reading the mean before the variance reduction overwrites scratch.
  __syncthreads();
  const float inverse_std = rsqrtf(
      block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    output[index] = from_float<T>(
        (to_float(input[index]) - mean) * inverse_std * to_float(weight[col]) +
        to_float(bias[col]));
  }
}

// Grouped gate: retain the BF16 product boundary before residual addition.
__global__ void grouped_gate_residual_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* residual,
    const __nv_bfloat16* style, __nv_bfloat16* output,
    int64_t count, int cols, int group_rows) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < count; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t offset = (i / cols / group_rows * 3 + 2) * cols + i % cols;
    const auto product = __float2bfloat16(__fmul_rn(to_float(input[i]), to_float(style[offset])));
    output[i] = __float2bfloat16(to_float(residual[i]) + to_float(product));
  }
}

__global__ void grouped_gate_residual_bf16_vec8(
    const uint4* input, const uint4* residual, const uint4* style, uint4* output,
    uint32_t vectors, uint32_t cols, uint32_t group_rows) {
  union Pack { uint4 words; unsigned short lanes[8]; };
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < vectors; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const uint32_t index = static_cast<uint32_t>(i);
    const uint32_t row = index / cols;
    const int64_t offset = (static_cast<int64_t>(row / group_rows) * 3 + 2) * cols + index % cols;
    Pack x, r, g, result;
    // All inputs loaded before storing, including exact in-place aliases.
    x.words = input[i]; r.words = residual[i]; g.words = style[offset];
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
      const auto product = __float2bfloat16(__fmul_rn(
          __bfloat162float(__ushort_as_bfloat16(x.lanes[lane])),
          __bfloat162float(__ushort_as_bfloat16(g.lanes[lane]))));
      const auto value = __float2bfloat16(
          __bfloat162float(__ushort_as_bfloat16(r.lanes[lane])) + __bfloat162float(product));
      result.lanes[lane] = __bfloat16_as_ushort(value);
    }
    output[i] = result.words;
  }
}

// Same reduction order as layer_norm, with explicit BF16 affine boundaries.
// Each consecutive group_rows rows reads one [parts, cols] style group.
__global__ void grouped_adaptive_layer_norm_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* style,
    __nv_bfloat16* output, int rows, int cols, int group_rows, int parts,
    bool shift_first, float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  const int64_t style_base = static_cast<int64_t>(row / group_rows) * parts * cols;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += to_float(input[base + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = to_float(input[base + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std = rsqrtf(
      block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    // The original unit affine adds +0 after normalization, including signed zero.
    const auto normalized = __float2bfloat16(__fadd_rn(
        (to_float(input[base + col]) - mean) * inverse_std, 0.0f));
    const auto scale = __float2bfloat16(
        to_float(style[style_base + (shift_first ? cols : 0) + col]) + 1.0f);
    const auto product = __float2bfloat16(__fmul_rn(to_float(normalized), to_float(scale)));
    output[base + col] = __float2bfloat16(
        to_float(product) + to_float(style[style_base + (shift_first ? 0 : cols) + col]));
  }
}

template <int V> struct GroupedNormWords;
template <> struct GroupedNormWords<2> { using T = unsigned; };
template <> struct GroupedNormWords<8> { using T = uint4; };
template <int V> union GroupedNormPack {
  typename GroupedNormWords<V>::T words;
  unsigned short lanes[V];
};

template <int V>
__device__ float grouped_norm_packed_reduce(float (&sums)[V], float* scratch) {
#pragma unroll
  for (int delta = 16; delta >= V; delta /= 2) {
#pragma unroll
    for (int j = 0; j < V; ++j)
      sums[j] += __shfl_down_sync(0xffffffff, sums[j], delta / V, 32 / V);
  }
#pragma unroll
  for (int delta = V / 2; delta; delta /= 2) {
#pragma unroll
    for (int j = 0; j < delta; ++j) sums[j] += sums[j + delta];
  }
  if (threadIdx.x % (32 / V) == 0)
    scratch[threadIdx.x / (32 / V)] = sums[0];
  __syncthreads();
  if (threadIdx.x < 32) {
    float v = threadIdx.x < 8 ? scratch[threadIdx.x] : 0.f;
    v = warp_sum(v);
    if (threadIdx.x == 0) scratch[0] = v;
  }
  __syncthreads();
  return scratch[0];
}

// Two adjacent logical lanes share one thread. Preserve the original 256-lane
// reduction tree while using aligned BF16-pair loads and stores.
__global__ void grouped_gate_residual_layer_norm_packed_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* residual,
    const __nv_bfloat16* gate, __nv_bfloat16* hidden,
    const __nv_bfloat16* style, __nv_bfloat16* output,
    int group_rows, int parts, bool shift_first, float eps) {
  using B = __nv_bfloat16;
  constexpr int V = 2;
  constexpr int cols = 2048;
  const int row = blockIdx.x, group = row / group_rows;
  const int64_t base = int64_t(row) * cols, style_base = int64_t(group) * parts * cols;
  __shared__ float scratch[8];
  float values[8][V], sums[V] = {};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int col = threadIdx.x * V + i * 256;
    GroupedNormPack<V> x, r, g, h;
    x.words = *reinterpret_cast<const typename GroupedNormWords<V>::T*>(input + base + col);
    r.words = *reinterpret_cast<const typename GroupedNormWords<V>::T*>(residual + base + col);
    g.words = *reinterpret_cast<const typename GroupedNormWords<V>::T*>(gate + (int64_t(group) * 3 + 2) * cols + col);
#pragma unroll
    for (int j = 0; j < V; ++j) {
      B prod = __float2bfloat16(__fmul_rn(
          __bfloat162float(__ushort_as_bfloat16(x.lanes[j])),
          __bfloat162float(__ushort_as_bfloat16(g.lanes[j]))));
      B val = __float2bfloat16(
          __bfloat162float(__ushort_as_bfloat16(r.lanes[j])) + __bfloat162float(prod));
      h.lanes[j] = __bfloat16_as_ushort(val);
      values[i][j] = __bfloat162float(val);
      sums[j] += values[i][j];
    }
    *reinterpret_cast<typename GroupedNormWords<V>::T*>(hidden + base + col) = h.words;
  }
  float mean = grouped_norm_packed_reduce<V>(sums, scratch) / cols;
#pragma unroll
  for (int j = 0; j < V; ++j) sums[j] = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < V; ++j) {
      float x = values[i][j] - mean;
      sums[j] += x * x;
    }
  }
  __syncthreads();
  float inv = rsqrtf(grouped_norm_packed_reduce<V>(sums, scratch) / cols + eps);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int col = threadIdx.x * V + i * 256;
    GroupedNormPack<V> s, b, o;
    s.words = *reinterpret_cast<const typename GroupedNormWords<V>::T*>(style + style_base + (shift_first ? cols : 0) + col);
    b.words = *reinterpret_cast<const typename GroupedNormWords<V>::T*>(style + style_base + (shift_first ? 0 : cols) + col);
#pragma unroll
    for (int j = 0; j < V; ++j) {
      B norm = __float2bfloat16(__fadd_rn((values[i][j] - mean) * inv, 0.f));
      B scale = __float2bfloat16(__bfloat162float(__ushort_as_bfloat16(s.lanes[j])) + 1.f);
      B prod = __float2bfloat16(__fmul_rn(__bfloat162float(norm), __bfloat162float(scale)));
      o.lanes[j] = __bfloat16_as_ushort(__float2bfloat16(
          __bfloat162float(prod) + __bfloat162float(__ushort_as_bfloat16(b.lanes[j]))));
    }
    *reinterpret_cast<typename GroupedNormWords<V>::T*>(output + base + col) = o.words;
  }
}

// Four independent rows per CTA; each warp preserves the original reduction
// tree. Save only the already-rounded BF16 hidden values between passes.
__device__ __forceinline__ float grouped_norm_warp_reduce(float (&sums)[8]) {
#pragma unroll
 for(int delta=16;delta>=8;delta/=2) {
#pragma unroll
  for(int j=0;j<8;++j)sums[j]+=__shfl_down_sync(0xffffffff,sums[j],delta/8,4);
 }
#pragma unroll
 for(int delta=4;delta;delta/=2) {
#pragma unroll
  for(int j=0;j<delta;++j)sums[j]+=sums[j+delta];
 }
 float value=sums[0];
 // The eight original warp leaders occupy lanes0,4,...,28.
#pragma unroll
 for(int delta=16;delta>=4;delta/=2)value+=__shfl_down_sync(0xffffffff,value,delta);
 return __shfl_sync(0xffffffff,value,0);
}
__global__ void grouped_gate_residual_layer_norm_warp_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* residual,
    const __nv_bfloat16* gate, __nv_bfloat16* hidden,
    const __nv_bfloat16* style, __nv_bfloat16* output,
    int group_rows, int parts, bool shift_first, float eps, int rows) {
  using B = __nv_bfloat16;
  union WarpNormPairPack { uint4 words; unsigned short lanes[8]; __nv_bfloat162 pairs[4]; };
  constexpr int V = 8;
  constexpr int cols = 2048;
  const int lane = threadIdx.x & 31;
  const int64_t row_index = int64_t(blockIdx.x) * 4 + (threadIdx.x >> 5);
  if (row_index >= rows) return;
  const int row = static_cast<int>(row_index), group = row / group_rows;
  const int64_t base = int64_t(row) * cols, style_base = int64_t(group) * parts * cols;

  WarpNormPairPack values[8];
  float sums[V] = {};
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int col = lane * V + i * 256;
    WarpNormPairPack x, r, g, h;
    x.words = *reinterpret_cast<const typename GroupedNormWords<V>::T*>(input + base + col);
    r.words = *reinterpret_cast<const typename GroupedNormWords<V>::T*>(residual + base + col);
    g.words = *reinterpret_cast<const typename GroupedNormWords<V>::T*>(gate + (int64_t(group) * 3 + 2) * cols + col);
#pragma unroll
    for (int j = 0; j < V; j += 2) {
      h.pairs[j/2] = __hadd2_rn(r.pairs[j/2], __hmul2_rn(x.pairs[j/2], g.pairs[j/2]));
      sums[j] += __bfloat162float(__ushort_as_bfloat16(h.lanes[j]));
      sums[j+1] += __bfloat162float(__ushort_as_bfloat16(h.lanes[j+1]));
    }
    values[i].words = h.words;
    *reinterpret_cast<typename GroupedNormWords<V>::T*>(hidden + base + col) = h.words;
  }
  float mean = grouped_norm_warp_reduce(sums) / cols;
#pragma unroll
  for (int j = 0; j < V; ++j) sums[j] = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
#pragma unroll
    for (int j = 0; j < V; ++j) {
      float x = __bfloat162float(__ushort_as_bfloat16(values[i].lanes[j])) - mean;
      sums[j] += x * x;
    }
  }

  float inv = rsqrtf(grouped_norm_warp_reduce(sums) / cols + eps);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    int col = lane * V + i * 256;
    WarpNormPairPack s, b, o;
    s.words = *reinterpret_cast<const typename GroupedNormWords<V>::T*>(style + style_base + (shift_first ? cols : 0) + col);
    b.words = *reinterpret_cast<const typename GroupedNormWords<V>::T*>(style + style_base + (shift_first ? 0 : cols) + col);
#pragma unroll
    for (int j = 0; j < V; j += 2) {
      B n0 = __float2bfloat16(__fadd_rn((__bfloat162float(__ushort_as_bfloat16(values[i].lanes[j])) - mean) * inv, 0.f));
      B n1 = __float2bfloat16(__fadd_rn((__bfloat162float(__ushort_as_bfloat16(values[i].lanes[j+1])) - mean) * inv, 0.f));
      auto norm = __halves2bfloat162(n0,n1);
      auto scale = __hadd2_rn(s.pairs[j/2],__float2bfloat162_rn(1.f));
      o.pairs[j/2] = __hadd2_rn(__hmul2_rn(norm,scale),b.pairs[j/2]);
    }
    *reinterpret_cast<typename GroupedNormWords<V>::T*>(output + base + col) = o.words;
  }
}

__global__ void grouped_adaptive_layer_norm_cached_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* style,
    __nv_bfloat16* output, int rows, int cols, int group_rows, int parts,
    bool shift_first, float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int64_t base = static_cast<int64_t>(row) * cols;
  const int64_t style_base = static_cast<int64_t>(row / group_rows) * parts * cols;
  // Dispatch guarantees cols2048 and blockDim256. Keep the same per-thread
  // summation order as the generic kernel while loading each input only once.
  float values[8];
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    values[i] = to_float(input[base + threadIdx.x + i * 256]);
    sum += values[i];
  }
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float centered = values[i] - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std = rsqrtf(
      block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int col = threadIdx.x + i * 256;
    // The original unit affine adds +0 after normalization, including signed zero.
    const auto normalized = __float2bfloat16(__fadd_rn(
        (values[i] - mean) * inverse_std, 0.0f));
    const auto scale = __float2bfloat16(
        to_float(style[style_base + (shift_first ? cols : 0) + col]) + 1.0f);
    const auto product = __float2bfloat16(__fmul_rn(to_float(normalized), to_float(scale)));
    output[base + col] = __float2bfloat16(
        to_float(product) + to_float(style[style_base + (shift_first ? 0 : cols) + col]));
  }
}

// norm_style is [2 * cols]: scale in [0, cols), shift in [cols, 2 * cols).
template <class T>
__global__ void ada_rms_norm(const T* input, const T* norm_style, T* output,
                             int rows, int cols, float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = to_float(input[static_cast<int64_t>(row) * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float normalized = to_float(input[index]) * inverse_rms;
    output[index] =
        from_float<T>(normalized * (1.0f + to_float(norm_style[col])) +
                      to_float(norm_style[cols + col]));
  }
}

template <class T>
__global__ void bias_residual(const T* projection, const T* bias,
                              const T* residual, T* output, int64_t count,
                              int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = to_float(projection[index]) + to_float(residual[index]);
    if (bias != nullptr) value += to_float(bias[index % cols]);
    output[index] = from_float<T>(value);
  }
}

// Preserve the observable result of two BF16 kernels (bias, then residual) in
// one launch. The explicit T round-trip is semantically required: folding all
// three operands into one float expression produces different BF16 values at
// rounding boundaries. The L3 contract only registers this kernel for BF16.
template <class T>
__global__ void bias_then_residual(const T* projection, const T* bias,
                                   const T* residual, T* output,
                                   int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    float value = to_float(projection[index]);
    if (bias != nullptr) value += to_float(bias[index % cols]);
    const T biased = from_float<T>(value);
    output[index] =
        from_float<T>(to_float(biased) + to_float(residual[index]));
  }
}

// The combined value is rounded to T in `hidden` and read back before the
// reduction.  The legacy kernel does the same; keeping the round-trip is what
// makes this port bit-comparable against the old backend.
template <class T>
__global__ void bias_residual_rms_norm(const T* projection, const T* bias,
                                       const T* residual, const T* weight,
                                       T* hidden, T* normalized, int rows,
                                       int cols, float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = to_float(projection[index]) + to_float(residual[index]);
    if (bias != nullptr) value += to_float(bias[col]);
    const T rounded = from_float<T>(value);
    hidden[index] = rounded;
    const float stored = to_float(rounded);
    square_sum += stored * stored;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    normalized[index] = from_float<T>(to_float(hidden[index]) * inverse_rms *
                                      to_float(weight[col]));
  }
}

template <class T>
__global__ void bias_residual_layer_norm(const T* projection,
                                         const T* projection_bias,
                                         const T* residual,
                                         const T* norm_weight,
                                         const T* norm_bias, T* hidden,
                                         T* normalized, int rows, int cols,
                                         float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = to_float(projection[index]) + to_float(residual[index]);
    if (projection_bias != nullptr) value += to_float(projection_bias[col]);
    const T rounded = from_float<T>(value);
    hidden[index] = rounded;
    sum += to_float(rounded);
  }
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        to_float(hidden[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  // Finish reading the mean before the variance reduction overwrites scratch.
  __syncthreads();
  const float inverse_std = rsqrtf(
      block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    normalized[index] = from_float<T>(
        (to_float(hidden[index]) - mean) * inverse_std *
            to_float(norm_weight[col]) +
        to_float(norm_bias[col]));
  }
}

// gate_style is [3 * cols]; the gate is the third segment.
template <class T>
__global__ void ada_gate_residual(const T* projection, const T* residual,
                                  const T* gate_style, T* output,
                                  int64_t count, int cols) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    output[index] = from_float<T>(
        to_float(residual[index]) +
        to_float(projection[index]) * to_float(gate_style[2 * cols + col]));
  }
}

// Reads the gate from this layer's gate_style and normalizes with the next
// layer's norm_style; see the semantic table in norm_types.h.
template <class T>
__global__ void ada_gate_residual_rms_norm(const T* projection,
                                           const T* residual,
                                           const T* gate_style,
                                           const T* norm_style, T* hidden,
                                           T* normalized, int rows, int cols,
                                           float eps) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  if (row >= rows) return;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const T rounded = from_float<T>(
        to_float(residual[index]) +
        to_float(projection[index]) * to_float(gate_style[2 * cols + col]));
    hidden[index] = rounded;
    const float value = to_float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float value = to_float(hidden[index]) * inverse_rms;
    normalized[index] =
        from_float<T>(value * (1.0f + to_float(norm_style[col])) +
                      to_float(norm_style[cols + col]));
  }
}

// FP8-output variants preserve the legacy PI0.5 contract: normalization is
// accumulated in FP32 and quantized directly, without materializing and then
// re-reading a full F16 normalized tensor.
__global__ void rms_norm_quant_f16_e4m3(
    const half* input, const half* weight, __nv_fp8_e4m3* output,
    int rows, int cols, float eps, float inverse_scale) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __half2float(input[row * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = __half2float(input[row * cols + col]) * inverse_rms *
                  __half2float(weight[col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[row * cols + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void layer_norm_quant_f16_e4m3(
    const half* input, const half* weight, const half* bias,
    __nv_fp8_e4m3* output, int rows, int cols, float eps,
    float inverse_scale) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __half2float(input[row * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = __half2float(input[row * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std = rsqrtf(
      block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = (__half2float(input[row * cols + col]) - mean) * inverse_std;
    value = value * __half2float(weight[col]) + __half2float(bias[col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[row * cols + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void ada_rms_norm_quant_f16_e4m3(
    const half* input, const half* style, __nv_fp8_e4m3* output,
    int rows, int cols, float eps, float inverse_scale) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __half2float(input[row * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float normalized =
        __half2float(input[row * cols + col]) * inverse_rms;
    float value = normalized * (1.0f + __half2float(style[col])) +
                  __half2float(style[cols + col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[row * cols + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void bias_residual_rms_norm_quant_f16_e4m3(
    const half* projection, const half* bias, const half* residual,
    const half* weight, half* hidden, __nv_fp8_e4m3* normalized,
    int rows, int cols, float eps, float inverse_scale) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value =
        __half2float(projection[index]) + __half2float(residual[index]);
    if (bias != nullptr) value += __half2float(bias[col]);
    const half rounded = __float2half(value);
    hidden[index] = rounded;
    value = __half2float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __half2float(hidden[index]) * inverse_rms *
                  __half2float(weight[col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    normalized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void bias_residual_layer_norm_quant_f16_e4m3(
    const half* projection, const half* projection_bias, const half* residual,
    const half* norm_weight, const half* norm_bias, half* hidden,
    __nv_fp8_e4m3* normalized, int rows, int cols, float eps,
    float inverse_scale) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value =
        __half2float(projection[index]) + __half2float(residual[index]);
    if (projection_bias != nullptr)
      value += __half2float(projection_bias[col]);
    const half rounded = __float2half(value);
    hidden[index] = rounded;
    sum += __half2float(rounded);
  }
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;
  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __half2float(hidden[static_cast<int64_t>(row) * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std = rsqrtf(
      block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = (__half2float(hidden[index]) - mean) * inverse_std;
    value = value * __half2float(norm_weight[col]) +
            __half2float(norm_bias[col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    normalized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void ada_gate_residual_rms_norm_quant_f16_e4m3(
    const half* projection, const half* residual, const half* gate_style,
    const half* norm_style, half* hidden, __nv_fp8_e4m3* normalized,
    int rows, int cols, float eps, float inverse_scale) {
  __shared__ float scratch[32];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float gate = __half2float(gate_style[2 * cols + col]);
    const half rounded = __float2half(
        __half2float(residual[index]) + __half2float(projection[index]) * gate);
    hidden[index] = rounded;
    const float value = __half2float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __half2float(hidden[index]) * inverse_rms;
    value = value * (1.0f + __half2float(norm_style[col])) +
            __half2float(norm_style[cols + col]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    normalized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

// Exact [10, 1024] action specialization from the legacy production path.
// It retains the same 256-lane reduction tree while caching rounded hidden
// values and storing four E4M3 outputs per thread.
__global__ void ada_gate_residual_rms_norm_quant_f16_e4m3_10x1024(
    const half* projection, const half* residual, const half* gate_style,
    const half* norm_style, half* hidden, __nv_fp8_e4m3* normalized,
    float eps, float inverse_scale) {
  constexpr int cols = 1024;
  __shared__ float scratch[32];
  __shared__ half cached[cols];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    const float gate = __half2float(gate_style[2 * cols + col]);
    const half rounded = __float2half(
        __half2float(residual[index]) + __half2float(projection[index]) * gate);
    hidden[index] = rounded;
    cached[col] = rounded;
    const float value = __half2float(rounded);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);

  union Half4 {
    uint2 packed;
    half values[4];
  };
  union Bytes4 {
    uint32_t packed;
    uint8_t values[4];
  };
  const int col = threadIdx.x * 4;
  Half4 h;
  Half4 scale;
  Half4 shift;
  h.packed = *reinterpret_cast<const uint2*>(cached + col);
  scale.packed = *reinterpret_cast<const uint2*>(norm_style + col);
  shift.packed = *reinterpret_cast<const uint2*>(norm_style + cols + col);
  Bytes4 output;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float value = __half2float(h.values[i]) * inverse_rms;
    value = value * (1.0f + __half2float(scale.values[i])) +
            __half2float(shift.values[i]);
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    const __nv_fp8_e4m3 quantized = static_cast<__nv_fp8_e4m3>(value);
    output.values[i] = *reinterpret_cast<const uint8_t*>(&quantized);
  }
  *reinterpret_cast<uint32_t*>(normalized + row * cols + col) = output.packed;
}

}  // namespace apxinf::norm::kernels

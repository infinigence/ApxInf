#pragma once

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace apxinf::rope::kernels {

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

// Load both partners before either write so exact in-place operation is safe.
template <int StaticHeads = 0, int StaticDim = 0>
__global__ void apply_precomputed_bf16(const __nv_bfloat16* input,
    const float* table, __nv_bfloat16* output, int64_t tokens,
    int runtime_heads, int runtime_dim) {
  const int heads = StaticHeads == 0 ? runtime_heads : StaticHeads;
  const int dim = StaticDim == 0 ? runtime_dim : StaticDim;
  const int half = dim / 2;
  const int64_t count = tokens * heads * half;
  for (int64_t pair = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       pair < count; pair += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = pair / half;
    const int c = pair % half;
    const int64_t t = row / heads;
    const int64_t a = row * dim + c, b = a + half;
    const float x = __bfloat162float(input[a]);
    const float y = __bfloat162float(input[b]);
    const float co = table[t * dim + c], si = table[t * dim + half + c];
    output[a] = __float2bfloat16(__fadd_rn(__fmul_rn(x, co), __fmul_rn(y, -si)));
    output[b] = __float2bfloat16(__fadd_rn(__fmul_rn(y, co), __fmul_rn(x, si)));
  }
}

// Matches norm::rms_norm_128's reduction order exactly.
__device__ __forceinline__ float rms_rope_warp_sum(float value) {
  for(int offset=16;offset>0;offset>>=1)
    value += __shfl_down_sync(0xffffffff,value,offset);
  return value;
}
__global__ void rms_precomputed_bf16_128(const __nv_bfloat16* input,
    const __nv_bfloat16* weight, const float* table, __nv_bfloat16* output,
    int rows, int heads, float eps) {
  const int64_t row=int64_t(blockIdx.x)*8+(threadIdx.x>>5);
  const int lane=threadIdx.x&31;
  if(row>=rows)return;
  const int64_t base=row*128+lane;
  const float x0=__bfloat162float(input[base]);
  const float x1=__bfloat162float(input[base+32]);
  const float x2=__bfloat162float(input[base+64]);
  const float x3=__bfloat162float(input[base+96]);
  const float p0=rms_rope_warp_sum(x0*x0),p1=rms_rope_warp_sum(x1*x1);
  const float p2=rms_rope_warp_sum(x2*x2),p3=rms_rope_warp_sum(x3*x3);
  float sum=__fadd_rn(__fadd_rn(p0,p2),__fadd_rn(p1,p3));
  sum=__shfl_sync(0xffffffff,sum,0);
  const float inv=rsqrtf(sum/128+eps);
  const float a=__bfloat162float(__float2bfloat16(x0*inv*__bfloat162float(weight[lane])));
  const float b=__bfloat162float(__float2bfloat16(x1*inv*__bfloat162float(weight[lane+32])));
  const float c=__bfloat162float(__float2bfloat16(x2*inv*__bfloat162float(weight[lane+64])));
  const float d=__bfloat162float(__float2bfloat16(x3*inv*__bfloat162float(weight[lane+96])));
  const int64_t t=int64_t(int(row)/heads)*128+lane;
  const float co0=table[t],si0=table[t+64],co1=table[t+32],si1=table[t+96];
  output[base]=__float2bfloat16(__fadd_rn(__fmul_rn(a,co0),__fmul_rn(c,-si0)));
  output[base+64]=__float2bfloat16(__fadd_rn(__fmul_rn(c,co0),__fmul_rn(a,si0)));
  output[base+32]=__float2bfloat16(__fadd_rn(__fmul_rn(b,co1),__fmul_rn(d,-si1)));
  output[base+96]=__float2bfloat16(__fadd_rn(__fmul_rn(d,co1),__fmul_rn(b,si1)));
}

// Adjacent logical lanes are packed into one physical thread. Preserve the
// original four independent warp32 sums: subgroup shuffles perform the high
// reduction levels and local halving performs the remaining low levels.
// A whole subgroup owns each row, including the final partial block.
template<int V> struct alignas(V*2) RmsRopePacked { __nv_bfloat16 x[V]; };
template<int V> __global__ void rms_precomputed_packed_bf16_128(const __nv_bfloat16* input,const __nv_bfloat16* weight,const float* table,__nv_bfloat16* output,int rows,int heads,float eps){
 constexpr int W=32/V;
 const int64_t row=int64_t(blockIdx.x)*(256/W)+threadIdx.x/W;
 const int lane=threadIdx.x%W;
 const unsigned mask=__ballot_sync(0xffffffff,row<rows);
 if(row>=rows)return;
 float x[4][V],p[4][V],y[4][V];
 #pragma unroll
 for(int q=0;q<4;++q){
  auto a=*reinterpret_cast<const RmsRopePacked<V>*>(input+row*128+q*32+lane*V);
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
  auto w=*reinterpret_cast<const RmsRopePacked<V>*>(weight+q*32+lane*V);
  #pragma unroll
  for(int j=0;j<V;++j)y[q][j]=__bfloat162float(__float2bfloat16(x[q][j]*inv*__bfloat162float(w.x[j])));
 }
 #pragma unroll
 for(int q=0;q<2;++q){
  RmsRopePacked<V> a,b;
  #pragma unroll
  for(int j=0;j<V;++j){
   const int64_t t=int64_t(int(row)/heads)*128+q*32+lane*V+j;
   const float co=table[t],si=table[t+64];
   a.x[j]=__float2bfloat16(__fadd_rn(__fmul_rn(y[q][j],co),__fmul_rn(y[q+2][j],-si)));
   b.x[j]=__float2bfloat16(__fadd_rn(__fmul_rn(y[q+2][j],co),__fmul_rn(y[q][j],si)));
  }
  *reinterpret_cast<RmsRopePacked<V>*>(output+row*128+q*32+lane*V)=a;
  *reinterpret_cast<RmsRopePacked<V>*>(output+row*128+(q+2)*32+lane*V)=b;
 }
}

// Splits a packed [tokens, q_heads*head_dim + 2*kv_heads*head_dim] projection,
// adds the optional fused bias, and rotates Q and K.  V is copied unrotated.
//
// `kv_output_offset` is what makes this one kernel serve two call sites: at 0
// it writes K/V into fresh buffers (prefill), and at the current prefix length
// it appends them into a KV cache (decode).  The caller owns which buffer the
// k/v pointers refer to.
//
// Launch: grid (tokens, q_heads + 2 * kv_heads), block >= head_dim / 2.
template <class T>
__global__ void split_qkv_rope(const T* qkv, const T* bias, T* q, T* k, T* v,
                               int tokens, int q_heads, int kv_heads,
                               int head_dim, float theta, int position_offset,
                               int kv_output_offset) {
  const int token = blockIdx.x;
  const int projection_head = blockIdx.y;
  const int half_dim = head_dim / 2;
  const int pair = threadIdx.x;
  if (token >= tokens || pair >= half_dim) return;
  const int q_width = q_heads * head_dim;
  const int kv_width = kv_heads * head_dim;
  const int fused_width = q_width + 2 * kv_width;
  const int position = position_offset + token;
  const float frequency = powf(theta, -static_cast<float>(pair) / half_dim);
  float sine, cosine;
  sincosf(position * frequency, &sine, &cosine);

  if (projection_head < q_heads) {
    const int source = token * fused_width + projection_head * head_dim;
    float first = to_float(qkv[source + pair]);
    float second = to_float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += to_float(bias[projection_head * head_dim + pair]);
      second += to_float(bias[projection_head * head_dim + half_dim + pair]);
    }
    const int destination = (token * q_heads + projection_head) * head_dim;
    q[destination + pair] = from_float<T>(first * cosine - second * sine);
    q[destination + half_dim + pair] =
        from_float<T>(second * cosine + first * sine);
  } else if (projection_head < q_heads + kv_heads) {
    const int head = projection_head - q_heads;
    const int source = token * fused_width + q_width + head * head_dim;
    float first = to_float(qkv[source + pair]);
    float second = to_float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += to_float(bias[q_width + head * head_dim + pair]);
      second += to_float(bias[q_width + head * head_dim + half_dim + pair]);
    }
    const int destination =
        ((kv_output_offset + token) * kv_heads + head) * head_dim;
    k[destination + pair] = from_float<T>(first * cosine - second * sine);
    k[destination + half_dim + pair] =
        from_float<T>(second * cosine + first * sine);
  } else {
    const int head = projection_head - q_heads - kv_heads;
    const int source =
        token * fused_width + q_width + kv_width + head * head_dim;
    const int destination =
        ((kv_output_offset + token) * kv_heads + head) * head_dim;
    float first = to_float(qkv[source + pair]);
    float second = to_float(qkv[source + half_dim + pair]);
    if (bias != nullptr) {
      first += to_float(bias[q_width + kv_width + head * head_dim + pair]);
      second +=
          to_float(bias[q_width + kv_width + head * head_dim + half_dim + pair]);
    }
    v[destination + pair] = from_float<T>(first);
    v[destination + half_dim + pair] = from_float<T>(second);
  }
}

// Splits a packed [tokens, 3 * projection_width] projection with an optional
// fused bias and no rotation; this is the vision tower's MHA layout, where Q,
// K and V all have the same width.
//
// Launch: grid (tokens), block any.
template <class T>
__global__ void split_qkv_bias(const T* qkv, const T* bias, T* q, T* k, T* v,
                               int tokens, int projection_width) {
  const int token = blockIdx.x;
  if (token >= tokens) return;
  const int fused_width = 3 * projection_width;
  for (int col = threadIdx.x; col < fused_width; col += blockDim.x) {
    float value = to_float(qkv[static_cast<int64_t>(token) * fused_width + col]);
    if (bias != nullptr) value += to_float(bias[col]);
    const int64_t row = static_cast<int64_t>(token) * projection_width;
    if (col < projection_width) {
      q[row + col] = from_float<T>(value);
    } else if (col < 2 * projection_width) {
      k[row + col - projection_width] = from_float<T>(value);
    } else {
      v[row + col - 2 * projection_width] = from_float<T>(value);
    }
  }
}

// Packed BF16 QKV split with height/width axial rotary positions.
__global__ void split_qkv_axial_rope(
    const __nv_bfloat16* qkv, const __nv_bfloat16* bias,
    const uint32_t* positions, __nv_bfloat16* query, __nv_bfloat16* key,
    __nv_bfloat16* value, int tokens, int heads, int head_dim, float theta) {
  const int token = blockIdx.x;
  const int half = head_dim / 2;
  const int projection_width = heads * head_dim;
  const int fused_width = 3 * projection_width;
  const int pairs_per_projection = heads * half;

  extern __shared__ float rope[];
  const int quarter = half / 2;
  float* sine = rope;
  float* cosine = rope + 2 * quarter;
  for (int slot = threadIdx.x; slot < 2 * quarter; slot += blockDim.x) {
    const int axis = slot / quarter;
    const int pair = slot - axis * quarter;
    const float position = static_cast<float>(positions[token * 2 + axis]);
    const float frequency =
        powf(theta, -2.0f * static_cast<float>(pair) / half);
    sincosf(position * frequency, &sine[slot], &cosine[slot]);
  }
  __syncthreads();

  for (int work = threadIdx.x; work < 2 * pairs_per_projection;
       work += blockDim.x) {
    const bool is_key = work >= pairs_per_projection;
    const int local = is_key ? work - pairs_per_projection : work;
    const int head = local / half;
    const int pair = local - head * half;
    const int projection_offset = is_key ? projection_width : 0;
    const int source = token * fused_width + projection_offset + head * head_dim;
    const int bias_base = projection_offset + head * head_dim;
    float first = __bfloat162float(qkv[source + pair]);
    float second = __bfloat162float(qkv[source + half + pair]);
    if (bias != nullptr) {
      first += __bfloat162float(bias[bias_base + pair]);
      second += __bfloat162float(bias[bias_base + half + pair]);
    }
    first = __bfloat162float(__float2bfloat16(first));
    second = __bfloat162float(__float2bfloat16(second));
    const int axis = pair < quarter ? 0 : 1;
    const int slot = axis * quarter + (pair < quarter ? pair : pair - quarter);
    __nv_bfloat16* destination =
        (is_key ? key : query) + (token * heads + head) * head_dim;
    destination[pair] =
        __float2bfloat16(first * cosine[slot] - second * sine[slot]);
    destination[half + pair] =
        __float2bfloat16(first * sine[slot] + second * cosine[slot]);
  }

  for (int column = threadIdx.x; column < projection_width;
       column += blockDim.x) {
    const int source = token * fused_width + 2 * projection_width + column;
    float result = __bfloat162float(qkv[source]);
    if (bias != nullptr) {
      result += __bfloat162float(bias[2 * projection_width + column]);
    }
    value[token * projection_width + column] = __float2bfloat16(result);
  }
}

__device__ __forceinline__ int multimodal_axis(int pair, int section_h,
                                                int section_w) {
  const int remainder = pair % 3;
  if (remainder == 1 && pair < section_h * 3) return 1;
  if (remainder == 2 && pair < section_w * 3) return 2;
  return 0;
}

__global__ void partial_multimodal_rope(
    __nv_bfloat16* data, const uint32_t* positions, int tokens, int heads,
    int head_dim, int rotary_dim, int section_h, int section_w, float theta) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  if (token >= tokens || head >= heads) return;
  const int half = rotary_dim / 2;
  const int64_t base = (static_cast<int64_t>(token) * heads + head) * head_dim;
  for (int pair = threadIdx.x; pair < half; pair += blockDim.x) {
    const int axis = multimodal_axis(pair, section_h, section_w);
    const float position = static_cast<float>(positions[token * 3 + axis]);
    const float frequency =
        __powf(theta, -2.0f * static_cast<float>(pair) / rotary_dim);
    float sine;
    float cosine;
    __sincosf(position * frequency, &sine, &cosine);
    const float low = __bfloat162float(data[base + pair]);
    const float high = __bfloat162float(data[base + half + pair]);
    data[base + pair] = __float2bfloat16(low * cosine - high * sine);
    data[base + half + pair] =
        __float2bfloat16(high * cosine + low * sine);
  }
}

}  // namespace apxinf::rope::kernels

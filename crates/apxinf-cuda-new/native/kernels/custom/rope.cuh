#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "norm_reduction.cuh"

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

template <class T> __device__ T from_double(double value);
template <> __device__ inline __half from_double(double value) {
  return __float2half(__double2float_rn(value));
}
template <> __device__ inline __nv_bfloat16 from_double(double value) {
  return __float2bfloat16(__double2float_rn(value));
}

template <class T>
__global__ void complex_table(const T* input, const double* coefficients,
                             T* output, int heads, int dim) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  for (int pair = threadIdx.x; pair < dim / 2; pair += blockDim.x) {
    const int64_t offset = (static_cast<int64_t>(token) * heads + head) * dim + 2 * pair;
    const int64_t coeff = static_cast<int64_t>(token) * dim + 2 * pair;
    const double real = static_cast<double>(to_float(input[offset]));
    const double imag = static_cast<double>(to_float(input[offset + 1]));
    const double c = coefficients[coeff], s = coefficients[coeff + 1];
    // Match unfused complex multiplication, including the FP64 boundaries.
    output[offset] = from_double<T>(__dsub_rn(__dmul_rn(real, c), __dmul_rn(imag, s)));
    output[offset + 1] = from_double<T>(__dadd_rn(__dmul_rn(real, s), __dmul_rn(imag, c)));
  }
}

__device__ inline unsigned short complex_bits(__half value) {
  return __half_as_ushort(value);
}
__device__ inline unsigned short complex_bits(__nv_bfloat16 value) {
  return __bfloat16_as_ushort(value);
}

__device__ inline float2 complex_product_bounds(float a, float lo, float hi) {
  return a >= 0 ? make_float2(__fmul_rd(a, lo), __fmul_ru(a, hi))
                : make_float2(__fmul_rd(a, hi), __fmul_ru(a, lo));
}

template<class T> struct ComplexValue { T real, imag; };

template<class T>
__device__ __forceinline__ ComplexValue<T> certified_complex_value(
    float a, float b, const double* coefficients, int64_t coeff, float4 cs) {
  const float2 ac = complex_product_bounds(a, cs.x, cs.y);
  const float2 bs = complex_product_bounds(b, cs.z, cs.w);
  const float2 as = complex_product_bounds(a, cs.z, cs.w);
  const float2 bc = complex_product_bounds(b, cs.x, cs.y);
  const float rl = __fsub_rd(ac.x, bs.y), rh = __fsub_ru(ac.y, bs.x);
  const float il = __fadd_rd(as.x, bc.x), ih = __fadd_ru(as.y, bc.y);
  T real = from_float<T>(rl), imag = from_float<T>(il);
  // Each directed FP32 interval also encloses the separately rounded FP64
  // product/sum: its endpoints are exact FP64 values and RN64 is monotone.
  // RN16(RN32(x)) is monotone too. Equal endpoint encodings therefore certify
  // the reference bits. Exclude zeros/nonfinite values to preserve sign and NaNs.
  const bool certified =
      isfinite(rl) && isfinite(rh) && isfinite(il) && isfinite(ih) &&
      isfinite(to_float(real)) && isfinite(to_float(imag)) &&
      to_float(real) != 0 && to_float(imag) != 0 &&
      complex_bits(real) == complex_bits(from_float<T>(rh)) &&
      complex_bits(imag) == complex_bits(from_float<T>(ih));
  if (!certified) {
    const double c = coefficients[coeff], s = coefficients[coeff + 1];
    real = from_double<T>(__dsub_rn(__dmul_rn(double(a), c), __dmul_rn(double(b), s)));
    imag = from_double<T>(__dadd_rn(__dmul_rn(double(a), s), __dmul_rn(double(b), c)));
  }
  return {real, imag};
}

// Launch 256 threads: four heads share the 64 coefficient pairs. This translation
// unit is compiled with FTZ disabled so directed FP32 operations enclose subnormals.
template <class T>
__global__ void complex_table_certified_128(const T* input,
                                           const double* coefficients,
                                           T* output, int heads) {
  __shared__ float4 bounds[64];
  const int token = blockIdx.x, lane = threadIdx.x;
  const int pair = lane % 64, head = blockIdx.y * 4 + lane / 64;
  const int64_t coeff = static_cast<int64_t>(token) * 128 + 2 * pair;
  if (lane < 64) {
    const double c = coefficients[coeff], s = coefficients[coeff + 1];
    bounds[pair] = make_float4(__double2float_rd(c), __double2float_ru(c),
                              __double2float_rd(s), __double2float_ru(s));
  }
  __syncthreads();
  if (head >= heads) return;
  const int64_t offset = (static_cast<int64_t>(token) * heads + head) * 128 + 2 * pair;
  const float a = to_float(input[offset]), b = to_float(input[offset + 1]);
  const float4 cs = bounds[pair];
  const auto value = certified_complex_value<T>(a, b, coefficients, coeff, cs);
  output[offset] = value.real;
  output[offset + 1] = value.imag;
}

// Staged RMS over the complete flattened head row, then complex rotation.
// The baseline covers every complex-table geometry. Warp reduction is chosen
// with exactly the same structural predicate/tree as NormNumerics::Staged.
// No input values are inspected for dispatch, and no workspace is allocated.
template<class T, bool WarpReduction, bool CertifiedRotation>
__global__ void rms_complex_table(const T* input, const T* weight,
                                  const double* coefficients, T* output,
                                  int heads, int head_dim, float eps) {
  const int dim = CertifiedRotation ? 128 : head_dim;
  __shared__ float scratch[32];
  __shared__ float factor;
  __shared__ float4 bounds[64];
  const int token = blockIdx.x, thread = threadIdx.x, cols = heads*dim;
  const int64_t base = static_cast<int64_t>(token)*cols;
  if constexpr (WarpReduction) {
    if (thread < 32) {
      float sums[4] = {0.f,0.f,0.f,0.f};
      for (int start=thread*4; start<cols; start+=128) {
        #pragma unroll
        for (int j=0; j<4; ++j) {
          const float x=to_float(input[base+start+j]);
          sums[j]=__fadd_rn(sums[j],__fmul_rn(x,x));
        }
      }
      float sum=__fadd_rn(__fadd_rn(__fadd_rn(sums[0],sums[1]),sums[2]),sums[3]);
      for (int offset=16; offset; offset/=2)
        sum=__fadd_rn(sum,__shfl_down_sync(0xffffffff,sum,offset));
      sum=__shfl_sync(0xffffffff,sum,0);
      if (thread==0) factor=rsqrtf(__fadd_rn(__fmul_rn(sum,1.f/cols),eps));
    }
  } else {
    float sum=0.f;
    for (int64_t col=thread; col<cols; col+=blockDim.x) {
      const float x=to_float(input[base+col]);
      sum=__fadd_rn(sum,__fmul_rn(x,x));
    }
    const float square_sum=apxinf::norm::kernels::block_sum_parallel_unsafe(sum,scratch);
    if (thread==0) factor=rsqrtf(square_sum/cols+eps);
  }
  if constexpr (CertifiedRotation) {
    if (thread<64) {
      const int64_t coeff=static_cast<int64_t>(token)*128+2*thread;
      const double c=coefficients[coeff], sn=coefficients[coeff+1];
      bounds[thread]=make_float4(__double2float_rd(c),__double2float_ru(c),
                                __double2float_rd(sn),__double2float_ru(sn));
    }
  }
  // Every input has contributed to the row reduction before any in-place store.
  __syncthreads();
  for (int64_t work=thread; work<cols/2; work+=blockDim.x) {
    const int64_t off=base+2*work;
    const int pair=work%(dim/2);
    const int64_t coeff=static_cast<int64_t>(token)*dim+2*pair;
    const float an=to_float(from_float<T>(__fmul_rn(to_float(input[off]),factor)));
    const float bn=to_float(from_float<T>(__fmul_rn(to_float(input[off+1]),factor)));
    const float a=to_float(from_float<T>(__fmul_rn(an,to_float(weight[2*work]))));
    const float b=to_float(from_float<T>(__fmul_rn(bn,to_float(weight[2*work+1]))));
    if constexpr (CertifiedRotation) {
      const auto value=certified_complex_value<T>(a,b,coefficients,coeff,bounds[pair]);
      output[off]=value.real;
      output[off+1]=value.imag;
    } else {
      const double c=coefficients[coeff], sn=coefficients[coeff+1];
      output[off]=from_double<T>(__dsub_rn(__dmul_rn(double(a),c),__dmul_rn(double(b),sn)));
      output[off+1]=from_double<T>(__dadd_rn(__dmul_rn(double(a),sn),__dmul_rn(double(b),c)));
    }
  }
}

}  // namespace apxinf::rope::kernels

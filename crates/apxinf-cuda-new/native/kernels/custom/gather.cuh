#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <cutlass/fast_math.h>
#include "../../include/apxinf_cuda/gather_types.h"

namespace apxinf::gather::kernels {

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

template <>
__device__ inline float to_float(float value) {
  return value;
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

template <>
__device__ inline float from_float(float value) {
  return value;
}

// Token-embedding gather, optionally scaled by sqrt(width).
//
// The scale is not optional: the Gemma backbone folds it into the embedding
// lookup.  The legacy backend also carries an unscaled same-named overload
// that differs only by `uint32_t` versus `int` parameters. `Scale=true`
// preserves the Gemma path; `Scale=false` serves the portable backend's plain
// embedding contract. Out-of-range ids produce zeros rather than reading past
// the table.
template <class T, bool Scale = true>
__global__ void embedding_lookup(const T* table, const uint32_t* ids, T* output,
                                 int tokens, int width, int vocab_size) {
  const int64_t count = static_cast<int64_t>(tokens) * width;
  const float normalizer = Scale ? sqrtf(static_cast<float>(width)) : 1.0f;
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int token = static_cast<int>(index / width);
    const int col = static_cast<int>(index % width);
    const uint32_t id = ids[token];
    output[index] =
        id < static_cast<uint32_t>(vocab_size)
            ? from_float<T>(
                  to_float(table[static_cast<int64_t>(id) * width + col]) *
                  normalizer)
            : from_float<T>(0.0f);
  }
}

// Vision patch-embedding epilogue: projection + learned position embedding,
// plus an optional per-column bias.  `position` is [tokens_per_view, cols] and
// repeats across views.
template <class T>
__global__ void bias_position(const T* projection, const T* bias,
                              const T* position, T* output, int64_t count,
                              int cols, int tokens_per_view) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    const int token = static_cast<int>((index / cols) % tokens_per_view);
    float value = to_float(projection[index]) +
                  to_float(position[static_cast<int64_t>(token) * cols + col]);
    if (bias != nullptr) value += to_float(bias[col]);
    output[index] = from_float<T>(value);
  }
}

// Cuts RGB images into flattened patches and maps [0, 255] to [-1, 1].
// Output is [views * patches_per_view, 3 * patch_size * patch_size].
template <class T, bool kNhwc>
__global__ void rgb_u8_to_patches(const uint8_t* images, T* patches, int views,
                                  int image_size, int patch_size) {
  const int patches_per_side = image_size / patch_size;
  const int patches_per_view = patches_per_side * patches_per_side;
  const int patch_area = patch_size * patch_size;
  const int patch_width = 3 * patch_area;
  const int64_t count =
      static_cast<int64_t>(views) * patches_per_view * patch_width;
  int64_t output_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; output_index < count; output_index += stride) {
    const int patch_element = static_cast<int>(output_index % patch_width);
    const int patch_index = static_cast<int>(output_index / patch_width);
    const int view = patch_index / patches_per_view;
    const int patch_in_view = patch_index - view * patches_per_view;
    const int patch_y = patch_in_view / patches_per_side;
    const int patch_x = patch_in_view - patch_y * patches_per_side;
    const int channel = patch_element / patch_area;
    const int pixel_in_patch = patch_element - channel * patch_area;
    const int dy = pixel_in_patch / patch_size;
    const int dx = pixel_in_patch - dy * patch_size;
    const int y = patch_y * patch_size + dy;
    const int x = patch_x * patch_size + dx;
    const int64_t input_index =
        kNhwc ? ((static_cast<int64_t>(view) * image_size + y) * image_size +
                 x) * 3 + channel
              : ((static_cast<int64_t>(view) * 3 + channel) * image_size + y) *
                        image_size + x;
    const float normalized =
        (static_cast<float>(images[input_index]) / 255.0f) * 2.0f - 1.0f;
    patches[output_index] = from_float<T>(normalized);
  }
}

template <class T>
__global__ void replace_rows(const T* input, const T* replacement,
                             const uint32_t* row_map, T* output,
                             int64_t count, int cols,
                             uint32_t replacement_rows) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int row = static_cast<int>(index / cols);
    const uint32_t source_row = row_map[row];
    output[index] = source_row < replacement_rows
        ? replacement[static_cast<int64_t>(source_row) * cols + index % cols]
        : input[index];
  }
}

// The normalized geometry contains no pointers or runtime host state.
template<class T>
__global__ void permute(const T* input,T* output,apxinf_gather_spec_t spec) {
  for(int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
      i<spec.rows;i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
    int64_t residual=i,source=0;
    for(int axis=static_cast<int>(spec.rank)-1;axis>=0;--axis) {
      source+=(residual%spec.layout_shape[axis])*spec.layout_strides[axis];
      residual/=spec.layout_shape[axis];
    }
    output[i]=input[source];
  }
}

// Copy in scalar or aligned 16-byte units. Host-precomputed exact integer
// dividers avoid per-element runtime 64-bit division; no floating arithmetic
// or convolution accumulation order changes. The launcher proves the index
// domain first and retains the general 64-bit kernel for larger geometry.
struct UnfoldIndex32 {
  int count, start, channels, shape[5], stride[3], padding[3], dilation[3], block;
  cutlass::FastDivmod cols, block_div, taps, taps_block, kernel[3], spatial[3];

  explicit UnfoldIndex32(const apxinf_gather_spec_t& s, int unit)
      : count(s.rows*s.cols/unit), start(s.row_start),
        channels(s.layout_shape[4]/unit),
        block((s.channel_block ? s.channel_block
               : s.cols/(s.kernel[0]*s.kernel[1]*s.kernel[2]))/unit),
        cols(s.cols/unit), block_div(block),
        taps(s.kernel[0]*s.kernel[1]*s.kernel[2]), taps_block(taps.divisor*block) {
    for (int a=0; a<5; ++a) shape[a]=s.layout_shape[a];
    for (int a=0; a<3; ++a) {
      stride[a]=s.stride[a];
      padding[a]=s.padding[a];
      dilation[a]=s.dilation[a];
      kernel[a]=cutlass::FastDivmod(s.kernel[a]);
      spatial[a]=cutlass::FastDivmod(s.output_spatial[a]);
    }
  }
};

template<class V>
__global__ void unfold3d_index32(const V* input, V* output, UnfoldIndex32 g) {
  // Unsigned iteration also keeps the last grid-stride increment defined when
  // the proven count is close to INT_MAX.
  for (uint32_t i=blockIdx.x*blockDim.x+threadIdx.x;
       i<static_cast<uint32_t>(g.count); i+=gridDim.x*blockDim.x) {
    int row, col;
    g.cols(row,col,i);
    row+=g.start;
    int group, ignored, lane;
    g.taps_block(group,ignored,col);
    g.block_div(ignored,lane,col);
    const int channel=group*g.block+lane;
    int tap;
    g.taps(ignored,tap,g.block_div.div(col));
    int coord[3];
    #pragma unroll
    for (int a=2; a>=0; --a) {
      int kernel_offset, center;
      g.kernel[a](tap,kernel_offset,tap);
      g.spatial[a](row,center,row);
      coord[a]=center*g.stride[a]-g.padding[a]+kernel_offset*g.dilation[a];
    }
    bool inside=channel<g.channels;
    #pragma unroll
    for (int a=0; a<3; ++a)
      inside &= coord[a]>=0 && coord[a]<g.shape[a+1];
    if (inside) {
      const int64_t offset=(((int64_t(row)*g.shape[1]+coord[0])*g.shape[2]
          +coord[1])*g.shape[3]+coord[2])*g.channels+channel;
      output[i]=input[offset];
    } else {
      output[i]=V{};
    }
  }
}

template<class T>
__global__ void unfold3d(const T* input,T* output,apxinf_gather_spec_t s) {
  const int64_t count=s.rows*s.cols;
  const int64_t channels=s.cols/(s.kernel[0]*s.kernel[1]*s.kernel[2]);
  for(int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;
      i<count;i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
    int64_t row=s.row_start+i/s.cols, col=i%s.cols;
    const int64_t block=s.channel_block>0?s.channel_block:channels;
    const int64_t taps=s.kernel[0]*s.kernel[1]*s.kernel[2];
    const int64_t c=(col/(taps*block))*block+col%block;
    col=(col/block)%taps;
    int64_t coord[3];
    for(int axis=2;axis>=0;--axis) {
      const int64_t tap=col%s.kernel[axis];col/=s.kernel[axis];
      const int64_t center=row%s.output_spatial[axis];row/=s.output_spatial[axis];
      coord[axis]=center*s.stride[axis]-s.padding[axis]+tap*s.dilation[axis];
    }
    bool inside=c<s.layout_shape[4];
    for(int axis=0;axis<3;++axis) inside &= coord[axis]>=0 && coord[axis]<s.layout_shape[axis+1];
    if(inside) {
      const int64_t offset=(((row*s.layout_shape[1]+coord[0])*s.layout_shape[2]+coord[1])*s.layout_shape[3]+coord[2])*s.layout_shape[4]+c;
      output[i]=input[offset];
    } else output[i]=from_float<T>(0.0f);
  }
}

__device__ inline float cubic_inner(float x) {return (1.25f*x-2.25f)*x*x+1.f;}
__device__ inline float cubic_outer(float x) {return ((-0.75f*x+3.75f)*x-6.f)*x+3.f;}
__device__ inline void cubic_coefficients(float t,float* c) {
  c[0]=cubic_outer(t+1.f);c[1]=cubic_inner(t);
  const float opposite=1.f-t;
  c[2]=cubic_inner(opposite);c[3]=cubic_outer(opposite+1.f);
}
template<class T>
__global__ void resize_bicubic(const T* input,T* output,apxinf_gather_spec_t spec) {
  const int ih=spec.layout_shape[1],iw=spec.layout_shape[2],channels=spec.layout_shape[3];
  const int oh=spec.output_spatial[0],ow=spec.output_spatial[1];
  const int64_t count=spec.rows*spec.cols;
  const float sy=static_cast<float>(ih)/oh,sx=static_cast<float>(iw)/ow;
  for(int64_t i=static_cast<int64_t>(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=static_cast<int64_t>(gridDim.x)*blockDim.x) {
    const int c=i%channels,x=(i/channels)%ow,y=(i/channels/ow)%oh;
    const int64_t batch=i/channels/ow/oh;
    if(ih==oh && iw==ow) {output[i]=input[i];continue;}
    const float fy=sy*(y+0.5f)-0.5f,fx=sx*(x+0.5f)-0.5f;
    const int iy=static_cast<int>(floorf(fy)),ix=static_cast<int>(floorf(fx));
    float cy[4],cx[4],v[4];cubic_coefficients(fy-iy,cy);cubic_coefficients(fx-ix,cx);
    for(int j=0;j<4;++j) {
      const int yy=min(max(iy+j-1,0),ih-1);
      float pixel[4];
      for(int k=0;k<4;++k) {
        const int xx=min(max(ix+k-1,0),iw-1);
        pixel[k]=to_float(input[((batch*ih+yy)*iw+xx)*channels+c]);
      }
      v[j]=pixel[0]*cx[0]+pixel[1]*cx[1]+pixel[2]*cx[2]+pixel[3]*cx[3];
    }
    output[i]=from_float<T>(v[0]*cy[0]+v[1]*cy[1]+v[2]*cy[2]+v[3]*cy[3]);
  }
}

}  // namespace apxinf::gather::kernels

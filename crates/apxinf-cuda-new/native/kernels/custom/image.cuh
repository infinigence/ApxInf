#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>

namespace apxinf::image::kernels {
__global__ void rgb_indices(const uint8_t* x,int32_t* y,int64_t count,int views,int height,int width,int channels,bool nhwc) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(blockDim.x)*gridDim.x) {
    int64_t t=i;const int ch=t%channels;t/=channels;const int col=t%width;t/=width;
    const int view=t%views;const int row=t/views;
    const int64_t source=nhwc?((int64_t(view)*height+row)*width+col)*channels+ch:
      ((int64_t(view)*channels+ch)*height+row)*width+col;
    y[i]=int32_t(x[source]);
  }
}
template <class T> __device__ float read(T value) { return float(value); }
template <class T> __device__ T round(float value) { return T(value); }

template <class T>
__global__ void columns(const T* x, T* y, int h, int w, int c,
                        int oh, int ow, int k, int stride, int padding) {
  const int64_t count = int64_t(oh)*ow*k*k*c;
  for (int64_t i = int64_t(blockIdx.x)*blockDim.x+threadIdx.x; i < count;
       i += int64_t(blockDim.x)*gridDim.x) {
    int64_t t=i; const int ch=t%c; t/=c;
    const int kw=t%k; t/=k; const int kh=t%k; t/=k;
    const int ix=(t%ow)*stride+kw-padding;
    const int iy=(t/ow)*stride+kh-padding;
    y[i]=(iy>=0 && iy<h && ix>=0 && ix<w) ? x[(int64_t(iy)*w+ix)*c+ch] : round<T>(0.f);
  }
}

template <class T>
__global__ void pack(const T* x,T* y,int h,int w,int c,int f) {
  const int oh=h/f, ow=w/f, oc=c*f*f;
  const int64_t count=int64_t(oh)*ow*oc;
  for (int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(blockDim.x)*gridDim.x) {
    const int ch=i%oc, row=i/oc;
    const int ix=(row%ow)*f+(ch/f)%f;
    const int iy=(row/ow)*f+ch%f;
    y[i]=x[(int64_t(iy)*w+ix)*c+ch/(f*f)];
  }
}

template <class T>
__global__ void grouped_average(const T* x,T* y,int h,int w,int c,int oc,int fs,int ft) {
  const int oh=h/fs,ow=w/fs,group=c*ft*fs*fs/oc;
  const int64_t count=int64_t(oh)*ow*oc;
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(blockDim.x)*gridDim.x) {
    const int outch=i%oc,row=i/oc; float sum=0.f;
    for(int g=0;g<group;++g) {
      int idx=outch*group+g;
      const int dx=idx%fs;idx/=fs;const int dy=idx%fs;idx/=fs;
      const int dt=idx%ft;const int ch=idx/ft;
      if(dt==ft-1) sum+=read(x[(int64_t(row/ow*fs+dy)*w+row%ow*fs+dx)*c+ch]);
    }
    y[i]=round<T>(sum/float(group));
  }
}

template <class T>
__global__ void l2_norm(const T* x,const T* gamma,T* y,int rows,int channels,float scale,float eps) {
  __shared__ float partial[256];
  for(int row=blockIdx.x;row<rows;row+=gridDim.x) {
    float sum=0.f;
    for(int c=threadIdx.x;c<channels;c+=blockDim.x) {float v=read(x[int64_t(row)*channels+c]);sum+=v*v;}
    partial[threadIdx.x]=sum;__syncthreads();
    for(int n=128;n>0;n/=2){if(threadIdx.x<n)partial[threadIdx.x]+=partial[threadIdx.x+n];__syncthreads();}
    const float denom=fmaxf(sqrtf(partial[0]),eps);
    for(int c=threadIdx.x;c<channels;c+=blockDim.x) {
      const int64_t i=int64_t(row)*channels+c;
      // PyTorch normalizes in F32, then casts before each pointwise product.
      T n=round<T>(read(x[i])/denom);
      n=round<T>(read(n)*scale);
      y[i]=round<T>(read(n)*read(gamma[c]));
    }
    __syncthreads();
  }
}

template <class T>
__global__ void standardize(const T* x,const float* mean,const float* std,float* y,int rows,int c,int oc) {
  const int64_t count=int64_t(rows)*oc;
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(blockDim.x)*gridDim.x) {
    const int ch=i%oc;
    y[i]=(read(x[(i/oc)*c+ch])-mean[ch])/std[ch];
  }
}
}

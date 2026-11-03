#include "../include/apxinf_cuda/image.h"
#include "../framework/runtime_internal.h"
#include "../kernels/custom/image.cuh"
#include <algorithm>
#include <climits>
#include <initializer_list>

namespace {
using apxinf::framework::Failure;
template<class T> void launch(const apxinf_image_spec_t& s,const apxinf_image_bindings_t& b) {
  namespace k=apxinf::image::kernels;
  auto stream=static_cast<cudaStream_t>(b.stream);
  const auto* x=static_cast<const T*>(b.input);auto* y=static_cast<T*>(b.output);
  int64_t count=int64_t(s.output_height)*s.output_width*s.output_channels;
  const int blocks=std::min<int64_t>(4096,(count+255)/256);
  switch(s.semantic) {
    case 0:k::columns<<<blocks,256,0,stream>>>(x,y,s.height,s.width,s.channels,s.output_height,s.output_width,s.kernel,s.stride,s.padding);break;
    case 1:k::pack<<<blocks,256,0,stream>>>(x,y,s.height,s.width,s.channels,s.kernel);break;
    case 2:k::grouped_average<<<blocks,256,0,stream>>>(x,y,s.height,s.width,s.channels,s.output_channels,s.kernel,s.temporal_factor);break;
    case 3:k::l2_norm<<<std::min<uint32_t>(4096,s.height*s.width),256,0,stream>>>(x,static_cast<const T*>(b.parameter),y,s.height*s.width,s.channels,b.scale,b.epsilon);break;
    case 4:k::standardize<<<blocks,256,0,stream>>>(x,static_cast<const float*>(b.parameter),static_cast<const float*>(b.parameter2),static_cast<float*>(b.output),s.height*s.width,s.channels,s.output_channels);break;
  }
  apxinf::framework::check_cuda(cudaGetLastError());
}
}

extern "C" apxinf_status_t apxinf_image_launch(apxinf_runtime_t runtime,const apxinf_image_spec_t* s,const apxinf_image_bindings_t* b) {
  return apxinf::framework::abi_boundary([&] {
    if(!runtime||!s||!b||s->version!=1||s->semantic>4||!b->input||!b->output||
       !s->height||!s->width||!s->channels||!s->output_height||!s->output_width||!s->output_channels||
       s->height>INT_MAX||s->width>INT_MAX||s->channels>INT_MAX||
       (s->dtype!=APXINF_DTYPE_F32&&s->dtype!=APXINF_DTYPE_BF16)||
       (s->semantic==4?s->output_dtype!=APXINF_DTYPE_F32:s->output_dtype!=s->dtype)||
       (s->semantic>=3&&!b->parameter)||(s->semantic==4&&!b->parameter2))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"invalid image primitive spec or bindings");
    const size_t input_alignment=s->dtype==APXINF_DTYPE_F32?alignof(float):alignof(__nv_bfloat16);
    const size_t output_alignment=s->semantic==4?alignof(float):input_alignment;
    if(reinterpret_cast<uintptr_t>(b->input)%input_alignment||reinterpret_cast<uintptr_t>(b->output)%output_alignment||
       (b->parameter&&reinterpret_cast<uintptr_t>(b->parameter)%(s->semantic==4?alignof(float):input_alignment))||
       (b->parameter2&&reinterpret_cast<uintptr_t>(b->parameter2)%alignof(float)))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"image primitive binding is not dtype-aligned");
    const uint64_t pixels=uint64_t(s->height)*s->width;
    const uint64_t output_pixels=uint64_t(s->output_height)*s->output_width;
    if(pixels>INT_MAX||output_pixels>INT_MAX||s->output_channels>INT_MAX||
       s->kernel>INT_MAX||s->stride>INT_MAX||s->temporal_factor>INT_MAX)
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"image primitive geometry exceeds index range");
    bool legal=false;
    switch(s->semantic) {
      case 0: legal=s->kernel&&s->stride&&s->padding<s->kernel&&
          uint64_t(s->kernel)*s->kernel*s->channels==s->output_channels&&
          uint64_t(s->output_height-1)*s->stride<uint64_t(s->height)+s->padding&&
          uint64_t(s->output_width-1)*s->stride<uint64_t(s->width)+s->padding;break;
      case 1: legal=s->kernel&&s->height%s->kernel==0&&s->width%s->kernel==0&&
          s->output_height==s->height/s->kernel&&s->output_width==s->width/s->kernel&&
          uint64_t(s->channels)*s->kernel*s->kernel==s->output_channels;break;
      case 2: legal=s->kernel&&s->temporal_factor&&s->height%s->kernel==0&&s->width%s->kernel==0&&
          s->output_height==s->height/s->kernel&&s->output_width==s->width/s->kernel&&
          uint64_t(s->channels)*s->kernel*s->kernel*s->temporal_factor<=INT_MAX&&
          (uint64_t(s->channels)*s->kernel*s->kernel*s->temporal_factor)%s->output_channels==0;break;
      case 3: legal=s->output_height==s->height&&s->output_width==s->width&&s->output_channels==s->channels&&
          std::isfinite(b->scale)&&std::isfinite(b->epsilon)&&b->epsilon>0;break;
      case 4: legal=s->output_height==s->height&&s->output_width==s->width&&s->output_channels<=s->channels;break;
    }
    if(!legal)throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"image primitive geometry is inconsistent");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    if(s->dtype==APXINF_DTYPE_F32)launch<float>(*s,*b);else launch<__nv_bfloat16>(*s,*b);
  });
}

extern "C" apxinf_status_t apxinf_image_rgb_indices_launch(apxinf_runtime_t runtime,const apxinf_image_rgb_indices_spec_t* s,const apxinf_image_rgb_indices_bindings_t* b) {
  return apxinf::framework::abi_boundary([&] {
    if(!runtime||!s||!b||s->version!=1||s->layout>1||!b->input||!b->output||
       !s->views||!s->height||!s->width||!s->channels||s->views>INT_MAX||
       s->height>INT_MAX||s->width>INT_MAX||s->channels>INT_MAX||
       reinterpret_cast<uintptr_t>(b->output)%alignof(int32_t)!=0)
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"invalid image RGB indices specification");
    uint64_t count=1;
    for(uint32_t dimension:{s->views,s->height,s->width,s->channels}) {
      if(count>uint64_t(INT64_MAX)/sizeof(int32_t)/dimension)
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"image RGB indices size overflow");
      count*=dimension;
    }
    const uintptr_t ip=reinterpret_cast<uintptr_t>(b->input),op=reinterpret_cast<uintptr_t>(b->output);
    if(ip>UINTPTR_MAX-count||op>UINTPTR_MAX-count*sizeof(int32_t)||
       (ip<op+count*sizeof(int32_t)&&op<ip+count))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"image RGB indices aliases or overflows storage");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const int blocks=std::min<uint64_t>(4096,(count+255)/256);
    apxinf::image::kernels::rgb_indices<<<blocks,256,0,static_cast<cudaStream_t>(b->stream)>>>(
      b->input,b->output,count,s->views,s->height,s->width,s->channels,s->layout==0);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}

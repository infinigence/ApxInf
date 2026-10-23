#include "../include/apxinf_cuda/reduction.h"
#include "../framework/runtime_internal.h"
#include "../kernels/custom/reduction.cuh"
#include <limits>
#include <memory>
namespace {
using apxinf::framework::Failure;
struct Execution {apxinf_indexed_mean_spec_t s;apxinf_indexed_mean_bindings_t b;int device;};

}
extern "C" apxinf_status_t apxinf_indexed_mean_prepare(apxinf_runtime_t runtime,const apxinf_indexed_mean_spec_t* s,const apxinf_indexed_mean_bindings_t* b,void** result) {
 return apxinf::framework::abi_boundary([&] {
  if(!runtime||!s||!b||!result)throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"null indexed mean argument");
  *result=nullptr;
  if(s->version!=APXINF_INDEXED_MEAN_SPEC_VERSION||s->reserved||s->input_elements<=0||
     s->input_elements>UINT32_MAX||s->rows<=0||s->rows>16777216||s->cols<=0||
     s->cols>std::numeric_limits<int64_t>::max()/s->rows/4||!b->input||!b->indices||!b->output)
   throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"invalid indexed mean contract");
  apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
  cudaStreamCaptureStatus capture;
  apxinf::framework::check_cuda(cudaStreamIsCapturing(static_cast<cudaStream_t>(b->stream),&capture));
  if(capture!=cudaStreamCaptureStatusNone)throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"indexed mean preparation during capture");
  auto e=std::make_unique<Execution>(Execution{*s,*b,runtime->device});*result=e.release();
 });
}
extern "C" apxinf_status_t apxinf_indexed_mean_enqueue(void* opaque) {
 return apxinf::framework::abi_boundary([&] {
  if(!opaque)throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"null indexed mean execution");
  auto& e=*static_cast<Execution*>(opaque);
  apxinf::framework::check_cuda(cudaSetDevice(e.device));
  int blocks=int(e.s.cols<4096?e.s.cols:4096);
  apxinf::reduction::kernels::indexed_mean<<<blocks,128,0,static_cast<cudaStream_t>(e.b.stream)>>>(e.s,e.b);
  apxinf::framework::check_cuda(cudaGetLastError());
 });
}
extern "C" void apxinf_indexed_mean_destroy(void* opaque){delete static_cast<Execution*>(opaque);}

#include "../include/apxinf_cuda/recurrence.h"
#include "../framework/runtime_internal.h"
#include "../kernels/custom/recurrence.cuh"
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
namespace {
using apxinf::framework::Failure;
struct Execution { apxinf_recurrence_spec_t spec; apxinf_recurrence_bindings_t bindings; int device; };

}
extern "C" apxinf_status_t apxinf_recurrence_prepare(apxinf_runtime_t runtime,
 const apxinf_recurrence_spec_t* s,const apxinf_recurrence_bindings_t* b,void** output) {
 return apxinf::framework::abi_boundary([&] {
  if (!runtime||!s||!b||!output) throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"null recurrence argument");
  *output=nullptr;
  if(s->version!=APXINF_RECURRENCE_SPEC_VERSION||s->reset>1||s->count<=0||
     s->count>std::numeric_limits<int64_t>::max()/8||!std::isfinite(s->a)||!std::isfinite(s->b)||
     !std::isfinite(s->c)||!std::isfinite(s->d)||!b->input||!b->output||!b->state||!b->previous)
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"invalid recurrence contract");
  apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
  cudaStreamCaptureStatus capture;
  apxinf::framework::check_cuda(cudaStreamIsCapturing(static_cast<cudaStream_t>(b->stream),&capture));
  if(capture!=cudaStreamCaptureStatusNone) throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"recurrence preparation during capture");
  auto e=std::make_unique<Execution>(Execution{*s,*b,runtime->device}); *output=e.release();
 });
}
extern "C" apxinf_status_t apxinf_recurrence_enqueue(void* opaque) {
 return apxinf::framework::abi_boundary([&] {
  if(!opaque) throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"null recurrence execution");
  auto& e=*static_cast<Execution*>(opaque);
  apxinf::framework::check_cuda(cudaSetDevice(e.device));
  int blocks=int(std::min<int64_t>(4096,(e.spec.count-1)/256+1));
  apxinf::recurrence::kernels::recurrence<<<blocks,256,0,static_cast<cudaStream_t>(e.bindings.stream)>>>(e.spec,e.bindings);
  apxinf::framework::check_cuda(cudaGetLastError());
 });
}
extern "C" void apxinf_recurrence_destroy(void* e) {delete static_cast<Execution*>(e);}

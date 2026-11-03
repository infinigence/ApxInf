#include "../include/apxinf_cuda/volume.h"
#include "../include/apxinf_cuda/gemm.h"
#include "../include/apxinf_cuda/runtime.h"
#include "../framework/runtime_internal.h"
#include "../kernels/custom/conv3d.cuh"
#include <algorithm>
#include <climits>
#include <memory>
#include <map>
#include <vector>
#include <sstream>
#include "../framework/autotune.h"
#include "../framework/registry.h"
#if __has_include(<cudnn.h>) && defined(__linux__)
#define APXINF_OPTIONAL_CUDNN 1
#include <cudnn.h>
#include <dlfcn.h>
#endif

namespace {
using apxinf::framework::Failure;
using apxinf::framework::check_cuda;
void require(bool yes, const char* message) {
  if (!yes) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, message);
}
int64_t mul(int64_t a, int64_t b) {
  require(a >= 0 && b >= 0 && (!b || a <= INT64_MAX / b), "Conv3d extent overflow");
  return a * b;
}
int64_t add(int64_t a, int64_t b) {
  require(a >= 0 && b >= 0 && a <= INT64_MAX - b, "Conv3d extent overflow");
  return a + b;
}
void check_status(apxinf_status_t status) {
  if (status != APXINF_STATUS_OK)
    throw Failure(status, std::string("Conv3d lowering/GEMM: ") + apxinf_last_error());
}
struct Buffer {
  void* ptr = nullptr;
  ~Buffer() { if (ptr) cudaFree(ptr); }
  void allocate(int64_t bytes) { check_cuda(cudaMalloc(&ptr, bytes)); }
};
#ifdef APXINF_OPTIONAL_CUDNN
#include "conv3d/providers/cudnn_library.h"
#include "conv3d/providers/cudnn_backend.h"
struct Direct {
  Library& lib = library();
  cudnnHandle_t handle = nullptr;
  cudnnTensorDescriptor_t x = nullptr, y = nullptr;
  cudnnFilterDescriptor_t w = nullptr;
  cudnnConvolutionDescriptor_t conv = nullptr;
  cudnnConvolutionFwdAlgo_t algorithm{};
  Buffer input, weight, workspace;
  std::unique_ptr<BackendPlan> backend;
  int64_t padded[3]{}, input_count = 0;
  size_t workspace_bytes = 0;
  bool materialize = false;
  ~Direct() {
    backend.reset();
    if (conv) lib.cudnnDestroyConvolutionDescriptor_fn(conv);
    if (w) lib.cudnnDestroyFilterDescriptor_fn(w);
    if (y) lib.cudnnDestroyTensorDescriptor_fn(y);
    if (x) lib.cudnnDestroyTensorDescriptor_fn(x);
    if (handle) lib.cudnnDestroy_fn(handle);
  }
};
#endif
struct Tile {
  void* unfold = nullptr;
  std::shared_ptr<apxinf_gemm_execution> gemm;
  int64_t start = 0, rows = 0;
  ~Tile() { apxinf_unfold3d_destroy(unfold); }
};
struct Candidate {
  const char* name;
  uint32_t provider_id,implementation_id,implementation_version;
  bool fallback,graph_safe,deterministic;
};
const apxinf::framework::Registry<Candidate>& registry() {
  static const apxinf::framework::Registry<Candidate> value{
    {"conv3d/baseline",1,1,1,true,true,true},
    {"conv3d/cudnn-backend",2,1,1,false,true,true}};
  return value;
}
struct Execution {
  const Candidate* implementation=nullptr;
  int configuration=0;
  apxinf_conv3d_spec_t spec{};
  apxinf_conv3d_bindings_t bindings{};
  int device = 0;
  int64_t k = 0;
  Buffer columns, tile_output;
  std::vector<std::unique_ptr<Tile>> tiles;
#ifdef APXINF_OPTIONAL_CUDNN
  std::unique_ptr<Direct> direct;
#endif
  std::string summary;
  ~Execution() {
    cudaSetDevice(device);
    // Preparation may have submitted a weight transpose; enqueue is async.
    cudaStreamSynchronize(static_cast<cudaStream_t>(bindings.volume.stream));
  }
};
void validate(const apxinf_conv3d_spec_t& spec, const apxinf_conv3d_bindings_t& b) {
  const auto& s = spec.geometry;
  require(s.version == APXINF_CONV3D_SPEC_VERSION && !s.reserved && s.row_start == 0 &&
      s.history_frames >= 0 && s.input[3] > 0 && spec.output_channels > 0 &&
      b.volume.input && b.volume.output && b.weight &&
      ((s.history_frames > 0) == (b.volume.history != nullptr)), "invalid Conv3d contract");
  int64_t rows = 1, k = s.input[3], input = s.input[3];
  for (int i = 0; i < 3; ++i) {
    require(s.input[i] > 0 && s.kernel[i] > 0 && s.stride[i] > 0, "invalid Conv3d dimension");
    auto padded = add(add(add(s.input[i], i == 0 ? s.history_frames : 0), s.padding_before[i]), s.padding_after[i]);
    require(padded >= s.kernel[i] && s.output[i] == (padded - s.kernel[i]) / s.stride[i] + 1,
        "Conv3d output extent mismatch");
    rows = mul(rows, s.output[i]); k = mul(k, s.kernel[i]);
    input = mul(input, i == 0 ? add(s.input[0], s.history_frames) : s.input[i]);
  }
  require(s.rows == rows, "Conv3d must cover entire output");
  mul(input, 2); mul(mul(rows, spec.output_channels), 2); mul(mul(k, spec.output_channels), 2);
}
#ifdef APXINF_OPTIONAL_CUDNN
std::unique_ptr<Direct> prepare_direct(Execution& e, uint64_t limit, int backend_configuration = -1) {
  if (!library().available ||
      reinterpret_cast<uintptr_t>(e.bindings.volume.input)%16 ||
      reinterpret_cast<uintptr_t>(e.bindings.volume.output)%16) return nullptr;
  if (backend_configuration >= 0) {
    cudaDeviceProp prop{}; check_cuda(cudaGetDeviceProperties(&prop,e.device));
    if (!library().backend_available || library().cudnnGetVersion_fn()!=92000 ||
        prop.major!=11 || prop.minor!=0 || backend_configuration>4) return nullptr;
  }
  const auto& s = e.spec.geometry;
  auto d = std::make_unique<Direct>();
  int pad[3], stride[3], dilation[] = {1,1,1};
  int64_t count = s.input[3];
  d->materialize = s.history_frames != 0;
  for (int i = 0; i < 3; ++i) {
    auto common = std::min(s.padding_before[i], s.padding_after[i]);
    d->padded[i] = add(add(s.input[i], i == 0 ? s.history_frames : 0),
        std::max(s.padding_before[i], s.padding_after[i]) - common);
    if (d->padded[i] > INT_MAX || s.output[i] > INT_MAX || s.kernel[i] > INT_MAX ||
        s.stride[i] > INT_MAX || common > INT_MAX) return nullptr;
    pad[i] = common; stride[i] = s.stride[i];
    count = mul(count, d->padded[i]);
    d->materialize |= s.padding_before[i] != s.padding_after[i];
  }
  if (s.input[3] > INT_MAX || e.spec.output_channels > INT_MAX) return nullptr;
  // cuDNN descriptor strides have a signed-int range.
  if (count > INT_MAX || mul(s.rows, e.spec.output_channels) > INT_MAX) return nullptr;
  d->input_count = count;
  auto weight_bytes = mul(mul(e.k, e.spec.output_channels), 2);
  auto scratch = add(d->materialize ? mul(count,2) : 0, weight_bytes);
  if (uint64_t(scratch) > limit) return nullptr;
  auto& l = d->lib;
  ck(l.cudnnCreate_fn(&d->handle)); ck(l.cudnnSetStream_fn(d->handle, static_cast<cudaStream_t>(e.bindings.volume.stream)));
  ck(l.cudnnCreateTensorDescriptor_fn(&d->x)); ck(l.cudnnCreateTensorDescriptor_fn(&d->y));
  ck(l.cudnnCreateFilterDescriptor_fn(&d->w)); ck(l.cudnnCreateConvolutionDescriptor_fn(&d->conv));
  int xd[] = {1,int(s.input[3]),int(d->padded[0]),int(d->padded[1]),int(d->padded[2])};
  int yd[] = {1,int(e.spec.output_channels),int(s.output[0]),int(s.output[1]),int(s.output[2])};
  int wd[] = {int(e.spec.output_channels),int(s.input[3]),int(s.kernel[0]),int(s.kernel[1]),int(s.kernel[2])};
  ck(l.cudnnSetTensorNdDescriptorEx_fn(d->x,CUDNN_TENSOR_NHWC,CUDNN_DATA_BFLOAT16,5,xd));
  ck(l.cudnnSetTensorNdDescriptorEx_fn(d->y,CUDNN_TENSOR_NHWC,CUDNN_DATA_BFLOAT16,5,yd));
  ck(l.cudnnSetFilterNdDescriptor_fn(d->w,CUDNN_DATA_BFLOAT16,CUDNN_TENSOR_NHWC,5,wd));
  ck(l.cudnnSetConvolutionNdDescriptor_fn(d->conv,3,pad,stride,dilation,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
  ck(l.cudnnSetConvolutionMathType_fn(d->conv,CUDNN_TENSOR_OP_MATH));
  if (backend_configuration >= 0) {
    d->backend=std::make_unique<BackendPlan>(l,d->handle);
    d->backend->prepare(xd,wd,yd,pad,stride,backend_configuration);
    if (uint64_t(d->backend->workspace_bytes)>limit-uint64_t(scratch)) return nullptr;
    d->workspace_bytes=d->backend->workspace_bytes;
  } else {
  cudnnConvolutionFwdAlgoPerf_t perf[16]; int n = 0;
  ck(l.cudnnGetConvolutionForwardAlgorithm_v7_fn(d->handle,d->x,d->w,d->conv,d->y,16,&n,perf));
  bool found = false;
  for (int i = 0; i < n; ++i) {
    if (perf[i].status != CUDNN_STATUS_SUCCESS || perf[i].determinism != CUDNN_DETERMINISTIC ||
        perf[i].memory > limit - scratch) continue;
    d->algorithm = perf[i].algo; d->workspace_bytes = perf[i].memory;
    ck(l.cudnnSetConvolutionMathType_fn(d->conv,perf[i].mathType));
    found = true; break;
  }
  if (!found) return nullptr;
  }
  if (d->materialize) d->input.allocate(mul(count,2));
  d->weight.allocate(weight_bytes);
  if (d->workspace_bytes) d->workspace.allocate(d->workspace_bytes);
  if (d->backend) d->backend->bind(d->materialize?d->input.ptr:const_cast<void*>(e.bindings.volume.input),
      d->weight.ptr,e.bindings.volume.output,d->workspace.ptr);
  apxinf::volume::kernels::conv3d_weight<<<int(std::min<int64_t>(4096,(weight_bytes/2+255)/256)),256,0,
      static_cast<cudaStream_t>(e.bindings.volume.stream)>>>(
      static_cast<const __nv_bfloat16*>(e.bindings.weight), static_cast<__nv_bfloat16*>(d->weight.ptr),e.k,e.spec.output_channels);
  check_cuda(cudaGetLastError());
  e.summary = "conv3d/cudnn-v9/v1 algorithm=" + std::to_string(int(d->algorithm)) +
      " workspace=" + std::to_string(scratch + d->workspace_bytes);
  if (d->backend) e.summary="conv3d/cudnn-backend/v1 config="+std::to_string(backend_configuration)+
      " engine="+std::to_string(d->backend->engine_id)+" workspace="+std::to_string(scratch+d->workspace_bytes);
  return d;
}
#endif
void prepare_lowered(apxinf_runtime_t runtime, Execution& e, uint64_t limit) {
  const auto& s = e.spec.geometry;
  auto n = e.spec.output_channels;
  auto row_bytes = mul(add(e.k,n),2);
  auto tile_rows = std::min<int64_t>({s.rows,4096,int64_t(std::min<uint64_t>(limit/2/row_bytes,INT64_MAX))});
  require(tile_rows > 0,"Conv3d workspace cannot hold one lowered row");
  e.columns.allocate(mul(mul(tile_rows,e.k),2));
  e.tile_output.allocate(mul(mul(tile_rows,n),2));
  // Cache-only, deterministic GEMM baseline. No tuning or recipe writes are
  // needed for this fixed convolution fallback. Reuse full and final tile plans.
  std::map<int64_t,std::shared_ptr<apxinf_gemm_execution>> plans;
  for (int64_t start = 0; start < s.rows; start += tile_rows) {
    auto tile = std::make_unique<Tile>(); tile->start = start; tile->rows = std::min(tile_rows,s.rows-start);
    auto unfold = s; unfold.row_start = start; unfold.rows = tile->rows;
    auto binding = e.bindings.volume; binding.output = e.columns.ptr;
    check_status(apxinf_unfold3d_prepare(runtime,&unfold,&binding,&tile->unfold));
    auto existing = plans.find(tile->rows);
    if (existing != plans.end()) {
      tile->gemm=existing->second; e.tiles.push_back(std::move(tile)); continue;
    }
    apxinf_gemm_spec_t gs{};
    gs.version=9; gs.a_dtype=gs.b_dtype=gs.output_dtype=APXINF_DTYPE_BF16;
    gs.accumulation_dtype=APXINF_DTYPE_F32; gs.b_is_immutable=1;
    gs.a_alignment=gs.output_alignment=256; gs.b_alignment=2;
    gs.m=tile->rows; gs.n=n; gs.k=e.k; gs.alpha_is_unit=gs.output_scale_is_unit=1;
    apxinf_gemm_bindings_t gb{};
    gb.a=e.columns.ptr;gb.b=e.bindings.weight;gb.b_version=1;gb.b_is_immutable=1;
    gb.output=e.tile_output.ptr;gb.stream=binding.stream;gb.alpha=gb.output_scale=1;
    apxinf_gemm_policy_t gp{};
    gp.workspace_limit=(limit-mul(tile_rows,row_bytes))/2;gp.allow_fallback=gp.graph_safe=gp.deterministic=1;
    gp.cache_dir="";
    apxinf_gemm_execution_t raw=nullptr;
    check_status(apxinf_gemm_prepare(runtime,&gs,&gp,&gb,&raw));
    tile->gemm=std::shared_ptr<apxinf_gemm_execution>(raw,apxinf_gemm_destroy);
    plans[tile->rows]=tile->gemm;
    e.tiles.push_back(std::move(tile));
  }
  e.summary="conv3d/lowered-gemm/v1 tile_rows="+std::to_string(tile_rows);
}
std::unique_ptr<Execution> prepare_candidate(apxinf_runtime_t runtime,
    const apxinf_conv3d_spec_t& spec,const apxinf_conv3d_bindings_t& bindings,
    uint64_t limit,const Candidate& candidate,int configuration) {
  auto e=std::make_unique<Execution>();e->spec=spec;e->bindings=bindings;e->device=runtime->device;
  e->implementation=&candidate;e->configuration=configuration;
  e->k=spec.geometry.input[3];for(int i=0;i<3;++i)e->k=mul(e->k,spec.geometry.kernel[i]);
  if(candidate.provider_id==2) {
#ifdef APXINF_OPTIONAL_CUDNN
    require(configuration>=0 && configuration<=4,"invalid Conv3d configuration");
    e->direct=prepare_direct(*e,limit,configuration);
    if(e->direct)return e;
#endif
    throw Failure(APXINF_STATUS_UNSUPPORTED,"Conv3d backend unavailable or over budget");
  }
  require(candidate.provider_id==1 && configuration==0,"invalid Conv3d baseline recipe");
#ifdef APXINF_OPTIONAL_CUDNN
  try { e->direct=prepare_direct(*e,limit); }
  catch(const Failure& failure) { if(failure.status!=APXINF_STATUS_PROVIDER_ERROR)throw; }
  if(!e->direct)
#endif
    prepare_lowered(runtime,*e,limit);
  return e;
}
#include "conv3d/selection.h"
} // namespace
extern "C" apxinf_status_t apxinf_conv3d_prepare(apxinf_runtime_t runtime,
    const apxinf_conv3d_spec_t* spec,const apxinf_conv3d_policy_t* policy,
    const apxinf_conv3d_bindings_t* bindings,void** execution) {
  return apxinf::framework::abi_boundary([&] {
    require(runtime && spec && policy && bindings && execution,"null Conv3d argument");
    *execution=nullptr; validate(*spec,*bindings);
    check_cuda(cudaSetDevice(runtime->device));
    cudaStreamCaptureStatus capture;
    check_cuda(cudaStreamIsCapturing(static_cast<cudaStream_t>(bindings->volume.stream),&capture));
    require(capture == cudaStreamCaptureStatusNone,"Conv3d preparation during capture");
    require(policy->version==1 && policy->online_tune<=1 && policy->allow_fallback<=1,
        "invalid Conv3d policy");
    auto e=select_execution(runtime,*spec,*policy,*bindings);
    *execution=e.release();
  });
}
extern "C" apxinf_status_t apxinf_conv3d_enqueue(void* opaque) {
  return apxinf::framework::abi_boundary([&] {
    require(opaque,"null Conv3d execution");auto& e=*static_cast<Execution*>(opaque);
    check_cuda(cudaSetDevice(e.device));auto stream=static_cast<cudaStream_t>(e.bindings.volume.stream);
#ifdef APXINF_OPTIONAL_CUDNN
    if(e.direct) {
      auto& d=*e.direct;const void* x=e.bindings.volume.input;
      if(d.materialize) {
        const auto& g=e.spec.geometry;
        const bool vectorized = g.input[3]%8 == 0 &&
            reinterpret_cast<uintptr_t>(e.bindings.volume.history)%16 == 0;
        if(vectorized) {
          const int count=int(d.input_count/8);
          apxinf::volume::kernels::conv3d_input_vec8<<<std::min(4096,(count+255)/256),256,0,stream>>>(
            int(g.input[0]),int(g.input[1]),int(g.input[2]),int(g.input[3]/8),
            int(g.history_frames),int(d.padded[1]),int(d.padded[2]),
            int(g.padding_before[0]-std::min(g.padding_before[0],g.padding_after[0])),
            int(g.padding_before[1]-std::min(g.padding_before[1],g.padding_after[1])),
            int(g.padding_before[2]-std::min(g.padding_before[2],g.padding_after[2])),
            static_cast<const uint4*>(x),static_cast<const uint4*>(e.bindings.volume.history),
            static_cast<uint4*>(d.input.ptr),count);
        } else {
        apxinf::volume::kernels::conv3d_input<<<int(std::min<int64_t>(4096,(d.input_count+255)/256)),256,0,stream>>>(
          e.spec.geometry,d.padded[0],d.padded[1],d.padded[2],
          static_cast<const __nv_bfloat16*>(x),static_cast<const __nv_bfloat16*>(e.bindings.volume.history),
          static_cast<__nv_bfloat16*>(d.input.ptr),d.input_count);
        }
        check_cuda(cudaGetLastError());x=d.input.ptr;
      }
      if(d.backend) { d.backend->enqueue(); return; }
      float alpha=1,beta=0;
      ck(d.lib.cudnnConvolutionForward_fn(d.handle,&alpha,d.x,x,d.w,d.weight.ptr,d.conv,d.algorithm,
          d.workspace.ptr,d.workspace_bytes,&beta,d.y,e.bindings.volume.output));
      return;
    }
#endif
    for(const auto& tile:e.tiles) {
      check_status(apxinf_unfold3d_enqueue(tile->unfold));check_status(apxinf_gemm_enqueue(tile->gemm.get()));
      check_cuda(cudaMemcpyAsync(static_cast<char*>(e.bindings.volume.output)+tile->start*e.spec.output_channels*2,
          e.tile_output.ptr,tile->rows*e.spec.output_channels*2,cudaMemcpyDeviceToDevice,stream));
    }
  });
}
extern "C" void apxinf_conv3d_destroy(void* opaque) { delete static_cast<Execution*>(opaque); }
extern "C" const char* apxinf_conv3d_summary(void* opaque) {
  return opaque?static_cast<Execution*>(opaque)->summary.c_str():"";
}

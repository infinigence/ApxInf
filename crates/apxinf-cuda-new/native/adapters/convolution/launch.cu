#include "../../include/apxinf_cuda/convolution.h"
#include "../../framework/runtime_internal.h"
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

// cuDNN is an optional operator provider, not a mandatory dependency of the
// CUDA backend. Builds without its development headers retain explicit errors.
#if __has_include(<cudnn.h>)
#define APXINF_HAS_CUDNN 1
#include <cudnn.h>
#include <dlfcn.h>
namespace {
using apxinf::framework::Failure;
using D = cudnnBackendDescriptor_t;
struct Api {
  void* library = nullptr;
#define CUDNN_FUNCTIONS(F) \
  F(cudnnCreate) F(cudnnDestroy) F(cudnnSetStream) F(cudnnGetVersion) \
  F(cudnnGetErrorString) F(cudnnBackendCreateDescriptor) \
  F(cudnnBackendDestroyDescriptor) F(cudnnBackendSetAttribute) \
  F(cudnnBackendGetAttribute) F(cudnnBackendFinalize) F(cudnnBackendExecute)
#define DECLARE(name) decltype(&::name) name = nullptr;
  CUDNN_FUNCTIONS(DECLARE)
#undef DECLARE
  Api() {
    library = dlopen("libcudnn.so.9", RTLD_NOW | RTLD_LOCAL);
    if (!library) return;
#define LOAD(name) name = reinterpret_cast<decltype(name)>(dlsym(library, #name));
    CUDNN_FUNCTIONS(LOAD)
#undef LOAD
#define CHECK(name) if (!name) { dlclose(library); library=nullptr; return; }
    CUDNN_FUNCTIONS(CHECK)
#undef CHECK
  }
  // Process lifetime: prepared graphs can retain provider objects beyond the
  // caller's context. Never unload a DSO containing their executable kernels.
};
Api& api() { static Api value; return value; }
void check(cudnnStatus_t status) {
  if (status != CUDNN_STATUS_SUCCESS)
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  std::string("cuDNN: ")+api().cudnnGetErrorString(status));
}
struct Descriptor {
  D raw = nullptr;
  explicit Descriptor(cudnnBackendDescriptorType_t type) {
    check(api().cudnnBackendCreateDescriptor(type, &raw));
  }
  ~Descriptor() { if(raw) api().cudnnBackendDestroyDescriptor(raw); }
};
void set(D d, cudnnBackendAttributeName_t a, cudnnBackendAttributeType_t t,
         int64_t count, const void* value) {
  check(api().cudnnBackendSetAttribute(d,a,t,count,value));
}
void finalize(D d) { check(api().cudnnBackendFinalize(d)); }
void tensor(D d, const int64_t* dims, int64_t id, uint32_t alignment,
            cudnnDataType_t dtype) {
  int64_t strides[5], stride=1, align=alignment;
  for(int i=4;i>=0;--i) { strides[i]=stride;stride*=dims[i]; }
  bool virtual_tensor=false;
  set(d,CUDNN_ATTR_TENSOR_DATA_TYPE,CUDNN_TYPE_DATA_TYPE,1,&dtype);
  set(d,CUDNN_ATTR_TENSOR_DIMENSIONS,CUDNN_TYPE_INT64,5,dims);
  set(d,CUDNN_ATTR_TENSOR_STRIDES,CUDNN_TYPE_INT64,5,strides);
  set(d,CUDNN_ATTR_TENSOR_UNIQUE_ID,CUDNN_TYPE_INT64,1,&id);
  set(d,CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,CUDNN_TYPE_INT64,1,&align);
  set(d,CUDNN_ATTR_TENSOR_IS_VIRTUAL,CUDNN_TYPE_BOOLEAN,1,&virtual_tensor);
  finalize(d);
}
}
#endif

struct apxinf_convolution_execution {
  std::string summary;
  int32_t configuration_count=0;
#if defined(APXINF_HAS_CUDNN)
  cudnnHandle_t handle=nullptr;
  std::vector<std::unique_ptr<Descriptor>> descriptors;
  D plan=nullptr, pack=nullptr;
  D make(cudnnBackendDescriptorType_t type) {
    descriptors.push_back(std::make_unique<Descriptor>(type));
    return descriptors.back()->raw;
  }
  ~apxinf_convolution_execution() {
    // Plans depend on descriptors and the handle; release in reverse order.
    while(!descriptors.empty()) descriptors.pop_back();
    if(handle) api().cudnnDestroy(handle);
  }
#endif
};
namespace {
using apxinf::framework::Failure;
void validate(const apxinf_convolution_spec_t& s,
              const apxinf_convolution_bindings_t& b) {
  if(s.version!=APXINF_CONVOLUTION_SPEC_VERSION || (s.dtype!=1 && s.dtype!=2) ||
     !b.input || !b.weight || !b.output || !b.workspace || !b.workspace_bytes ||
     reinterpret_cast<uintptr_t>(b.workspace)%256)
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"invalid convolution arguments");
  const int64_t* shapes[]={s.input,s.weight,s.output};
  const void* pointers[]={b.input,b.weight,b.output};
  for(int j=0;j<3;++j) {
    int64_t count=1;
    for(int i=0;i<5;++i) {
      if(shapes[j][i]<=0 || shapes[j][i]>INT32_MAX || count>INT64_MAX/2/shapes[j][i])
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"convolution extent overflow");
      count*=shapes[j][i];
    }
    const uint32_t a=s.alignment[j];
    if(a<2 || a>32 || (a&(a-1)) || reinterpret_cast<uintptr_t>(pointers[j])%a)
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"convolution alignment mismatch");
  }
  if(s.input[0]!=s.output[0] || s.input[1]!=s.weight[1] || s.output[1]!=s.weight[0])
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"convolution channel/batch mismatch");
  for(int i=0;i<3;++i)
    if(s.stride[i]<=0 || s.input[i+2]<s.weight[i+2] ||
       s.output[i+2]!=(s.input[i+2]-s.weight[i+2])/s.stride[i]+1)
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"convolution spatial mismatch");
}
}
extern "C" uint64_t apxinf_convolution_cudnn_version() {
#if defined(APXINF_HAS_CUDNN)
  return api().library ? api().cudnnGetVersion() : 0;
#else
  return 0;
#endif
}
extern "C" apxinf_status_t apxinf_convolution_prepare(apxinf_runtime_t runtime,
    const apxinf_convolution_spec_t* spec,
    const apxinf_convolution_bindings_t* bindings, int32_t configuration,
    apxinf_convolution_execution_t* output) {
  return apxinf::framework::abi_boundary([&] {
    if(!runtime || !spec || !bindings || !output || configuration < -1)
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"null convolution prepare argument");
    *output=nullptr;validate(*spec,*bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    if(!apxinf_convolution_cudnn_version())
      throw Failure(APXINF_STATUS_UNSUPPORTED,
          "3D convolution needs cuDNN 9 runtime and cuDNN headers when building ApxInf");
#if defined(APXINF_HAS_CUDNN)
    auto e=std::make_unique<apxinf_convolution_execution>();
    check(api().cudnnCreate(&e->handle));
    check(api().cudnnSetStream(e->handle,static_cast<cudaStream_t>(bindings->stream)));
    const cudnnDataType_t dtype=spec->dtype==1?CUDNN_DATA_HALF:CUDNN_DATA_BFLOAT16;
    D x=e->make(CUDNN_BACKEND_TENSOR_DESCRIPTOR),w=e->make(CUDNN_BACKEND_TENSOR_DESCRIPTOR),y=e->make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    tensor(x,spec->input,'x',spec->alignment[0],dtype);
    tensor(w,spec->weight,'w',spec->alignment[1],dtype);
    tensor(y,spec->output,'y',spec->alignment[2],dtype);
    D conv=e->make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    cudnnDataType_t compute=CUDNN_DATA_FLOAT;
    cudnnConvolutionMode_t mode=CUDNN_CROSS_CORRELATION;
    int64_t dims=3,ones[3]={1,1,1},zero[3]={0,0,0};
    set(conv,CUDNN_ATTR_CONVOLUTION_COMP_TYPE,CUDNN_TYPE_DATA_TYPE,1,&compute);
    set(conv,CUDNN_ATTR_CONVOLUTION_CONV_MODE,CUDNN_TYPE_CONVOLUTION_MODE,1,&mode);
    set(conv,CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,CUDNN_TYPE_INT64,1,&dims);
    set(conv,CUDNN_ATTR_CONVOLUTION_DILATIONS,CUDNN_TYPE_INT64,3,ones);
    set(conv,CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,CUDNN_TYPE_INT64,3,spec->stride);
    set(conv,CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,CUDNN_TYPE_INT64,3,zero);
    set(conv,CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,CUDNN_TYPE_INT64,3,zero);finalize(conv);
    D op=e->make(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    float alpha=1,beta=0;
    set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,CUDNN_TYPE_FLOAT,1,&alpha);
    set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,CUDNN_TYPE_FLOAT,1,&beta);
    set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&x);
    set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&w);
    set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&y);
    set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&conv);finalize(op);
    D graph=e->make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    set(graph,CUDNN_ATTR_OPERATIONGRAPH_HANDLE,CUDNN_TYPE_HANDLE,1,&e->handle);
    set(graph,CUDNN_ATTR_OPERATIONGRAPH_OPS,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&op);finalize(graph);
    D heur=e->make(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
    cudnnBackendHeurMode_t hm=CUDNN_HEUR_MODE_INSTANT;
    set(heur,CUDNN_ATTR_ENGINEHEUR_MODE,CUDNN_TYPE_HEUR_MODE,1,&hm);
    set(heur,CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&graph);finalize(heur);
    int64_t count=0;
    check(api().cudnnBackendGetAttribute(heur,CUDNN_ATTR_ENGINEHEUR_RESULTS,CUDNN_TYPE_BACKEND_DESCRIPTOR,0,&count,nullptr));
    if(count<0 || count>4096) throw Failure(APXINF_STATUS_PROVIDER_ERROR,"invalid convolution heuristic count");
    e->configuration_count=static_cast<int32_t>(count);
    std::vector<D> configs;
    for(int64_t i=0;i<count;++i) configs.push_back(e->make(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
    check(api().cudnnBackendGetAttribute(heur,CUDNN_ATTR_ENGINEHEUR_RESULTS,CUDNN_TYPE_BACKEND_DESCRIPTOR,count,&count,configs.data()));
    int selected=-1;int64_t needed=0,engine_id=0;
    for(int i=0;i<count;++i) {
      if(configuration>=0 && i!=configuration) continue;
      D engine=e->make(CUDNN_BACKEND_ENGINE_DESCRIPTOR);int64_t got=0;
      check(api().cudnnBackendGetAttribute(configs[i],CUDNN_ATTR_ENGINECFG_ENGINE,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&got,&engine));
      cudnnBackendNumericalNote_t notes[32];int64_t note_count=0;
      check(api().cudnnBackendGetAttribute(engine,CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,CUDNN_TYPE_NUMERICAL_NOTE,32,&note_count,notes));
      if(note_count<0 || note_count>32) throw Failure(APXINF_STATUS_PROVIDER_ERROR,"invalid convolution numerical notes");
      bool valid=true;
      for(int64_t j=0;j<note_count;++j)
        if(notes[j]==CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC || notes[j]==CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS ||
           notes[j]==CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION) valid=false;
      if(!valid) continue;
      D plan=e->make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
      set(plan,CUDNN_ATTR_EXECUTION_PLAN_HANDLE,CUDNN_TYPE_HANDLE,1,&e->handle);
      set(plan,CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&configs[i]);
      if(api().cudnnBackendFinalize(plan)!=CUDNN_STATUS_SUCCESS) continue;
      check(api().cudnnBackendGetAttribute(plan,CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,CUDNN_TYPE_INT64,1,&got,&needed));
      if(needed<0 || static_cast<uint64_t>(needed)>bindings->workspace_bytes) continue;
      check(api().cudnnBackendGetAttribute(engine,CUDNN_ATTR_ENGINE_GLOBAL_INDEX,CUDNN_TYPE_INT64,1,&got,&engine_id));
      e->plan=plan;selected=i;break;
    }
    if(selected<0) throw Failure(APXINF_STATUS_UNSUPPORTED,"no deterministic F32 convolution plan within workspace budget");
    e->pack=e->make(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    int64_t ids[3]={'x','w','y'};
    void* ptrs[3]={const_cast<void*>(bindings->input),const_cast<void*>(bindings->weight),bindings->output};
    set(e->pack,CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,CUDNN_TYPE_INT64,3,ids);
    set(e->pack,CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,CUDNN_TYPE_VOID_PTR,3,ptrs);
    set(e->pack,CUDNN_ATTR_VARIANT_PACK_WORKSPACE,CUDNN_TYPE_VOID_PTR,1,&bindings->workspace);finalize(e->pack);
    e->summary="cudnn-v"+std::to_string(apxinf_convolution_cudnn_version())+
        " engine="+std::to_string(engine_id)+" heuristic-rank="+std::to_string(selected)+
        " workspace="+std::to_string(needed);
    if(std::getenv("APXINF_CONVOLUTION_REPORT")) std::fprintf(stderr,"[convolution] %s\n",e->summary.c_str());
    *output=e.release();
#endif
  });
}
extern "C" apxinf_status_t apxinf_convolution_enqueue(apxinf_convolution_execution_t e) {
  return apxinf::framework::abi_boundary([&] {
    if(!e) throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"null convolution execution");
#if defined(APXINF_HAS_CUDNN)
    check(api().cudnnBackendExecute(e->handle,e->plan,e->pack));
#else
    throw Failure(APXINF_STATUS_UNSUPPORTED,"cuDNN provider unavailable");
#endif
  });
}
extern "C" const char* apxinf_convolution_summary(apxinf_convolution_execution_t e) { return e?e->summary.c_str():""; }
extern "C" int32_t apxinf_convolution_configuration_count(apxinf_convolution_execution_t e) { return e?e->configuration_count:0; }
extern "C" void apxinf_convolution_destroy(apxinf_convolution_execution_t e) { delete e; }

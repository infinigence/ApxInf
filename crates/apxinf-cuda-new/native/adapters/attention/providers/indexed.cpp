#include "../internal.h"
#if defined(APXINF_ATTENTION_INDEXED_SM110)
#include "../../../kernels/attention/indexed_sm110.h"
namespace apxinf::attention {
namespace {
namespace kernel=apxinf::cuda_new::indexed_attention;
struct Sizes {
 size_t flags,indices,counts,cumulative,native_lse,fa2_lse;
 size_t total()const{return flags+indices+counts+cumulative+native_lse+fa2_lse;}
};
Sizes sizes(const Spec& s){
 // supports_indexed fixes this domain before resource/preparation callbacks.
 if(s.query_tokens!=1785 || s.query_heads!=40)
  throw Failure(APXINF_STATUS_UNSUPPORTED,"indexed Attention resource domain");
 const size_t rows=size_t(s.query_tokens)*size_t(s.query_heads);
 return {rows*sizeof(uint32_t),rows*sizeof(int),size_t(s.query_heads)*sizeof(int),
         size_t(s.query_heads+1)*sizeof(int),size_t((s.query_tokens+127)/128*128)*size_t(s.query_heads)*sizeof(float),rows*sizeof(float)};
}
struct State {
 kernel::Bindings bindings{};void* native=nullptr;void* mapped=nullptr;int device=0;
 ~State(){
  int previous=device;cudaGetDevice(&previous);cudaSetDevice(device);
  kernel::destroy_native(native);kernel::destroy_mapped(mapped);
  for(void* p:{static_cast<void*>(bindings.flags),static_cast<void*>(bindings.indices),static_cast<void*>(bindings.counts),static_cast<void*>(bindings.cumulative),static_cast<void*>(bindings.native_lse),static_cast<void*>(bindings.fa2_lse)})if(p)cudaFree(p);
  if(previous!=device)cudaSetDevice(previous);
 }
};
template<class T>void allocate(T*& p,size_t bytes){check_cuda(cudaMalloc(reinterpret_cast<void**>(&p),bytes));}
void plan_status(int status){
 if(status!=0)throw Failure(status==-1?APXINF_STATUS_UNSUPPORTED:APXINF_STATUS_PROVIDER_ERROR,"indexed Attention plan status "+std::to_string(status));
}
}
size_t indexed_resource_requirements(const Spec& s,int configuration){
 if(configuration!=0)throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"indexed Attention configuration");
 return sizes(s).total();
}
void prepare_indexed(Execution& e){
 const auto bytes=sizes(e.spec);
 if(e.configuration!=0 || bytes.total()>e.resource_limit)
  throw Failure(APXINF_STATUS_UNSUPPORTED,"indexed Attention workspace exceeds policy");
 auto state=std::make_unique<State>();state->device=e.device;auto& b=state->bindings;
 b.query=e.bindings.query;b.key=e.bindings.key;b.value=e.bindings.value;b.output=e.bindings.output;
 b.queries=int(e.spec.query_tokens);b.keys=int(e.spec.key_tokens);b.heads=int(e.spec.query_heads);b.scale=e.bindings.scale;
 allocate(b.flags,bytes.flags);allocate(b.indices,bytes.indices);allocate(b.counts,bytes.counts);allocate(b.cumulative,bytes.cumulative);allocate(b.native_lse,bytes.native_lse);allocate(b.fa2_lse,bytes.fa2_lse);
 plan_status(kernel::prepare_native(b,&state->native));plan_status(kernel::prepare_mapped(b,&state->mapped));
 e.resource_bytes=bytes.total();e.provider_state=state.release();
}
cudaError_t launch_indexed(Execution& e){
 auto& state=*static_cast<State*>(e.provider_state);const auto& b=state.bindings;
 const auto stream=static_cast<cudaStream_t>(e.bindings.stream);
 auto result=cudaMemsetAsync(b.flags,0,size_t(b.queries)*b.heads*sizeof(uint32_t),stream);if(result!=cudaSuccess)return result;
 result=kernel::enqueue_native(state.native,stream);if(result!=cudaSuccess)return result;
 result=cudaMemsetAsync(b.counts,0,size_t(b.heads)*sizeof(int),stream);if(result!=cudaSuccess)return result;
 result=kernel::compact(b,stream);if(result!=cudaSuccess)return result;
 return kernel::enqueue_mapped(state.mapped,stream);
}
void destroy_indexed(Execution& e)noexcept{delete static_cast<State*>(e.provider_state);e.provider_state=nullptr;}
} // namespace apxinf::attention
#endif

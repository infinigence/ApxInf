#include "../internal.h"
#include <ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h>
#include <limits>

namespace apxinf::attention {
namespace {
// Keep the upstream bias-capable arithmetic: it rounds score scaling before
// online softmax. The public provider always binds a null additive bias.
template<class T> using Kernel = PyTorchMemEffAttention::AttentionKernel<
    T, cutlass::arch::Sm80, true, 32, 128, 65536, false, true>;
struct State {
  float* accumulator = nullptr;
  ~State() { if (accumulator) cudaFree(accumulator); }
};
template<class T>
typename Kernel<T>::Params parameters(const Execution& e) {
  const auto& s=e.spec; const auto& b=e.bindings;
  typename Kernel<T>::Params p;
  p.query_ptr=static_cast<const T*>(b.query);
  p.key_ptr=static_cast<const T*>(b.key);
  p.value_ptr=static_cast<const T*>(b.value);
  p.output_ptr=static_cast<T*>(b.output);
  p.output_accum_ptr=static_cast<State*>(e.provider_state)->accumulator;
  p.scale=b.scale;
  p.head_dim=p.head_dim_value=static_cast<int>(s.head_dim);
  p.num_queries=static_cast<int>(s.query_tokens);
  p.num_keys=static_cast<int>(s.key_tokens);
  p.num_batches=static_cast<int>(s.batch);
  p.num_heads=static_cast<int>(s.query_heads);
  p.q_strideM=p.k_strideM=p.v_strideM=p.o_strideM=p.num_heads*p.head_dim;
  p.q_strideH=p.k_strideH=p.v_strideH=p.head_dim;
  p.q_strideB=s.query_tokens*p.q_strideM;
  p.k_strideB=p.v_strideB=s.key_tokens*p.k_strideM;
  return p;
}
template<class T> void configure(const Execution& e) {
  using K=Kernel<T>;
  const auto kernel=PyTorchMemEffAttention::attention_kernel_batched_impl<K>;
  int limit=0;
  check_cuda(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,e.device));
  if (sizeof(typename K::SharedStorage)>static_cast<size_t>(limit))
    throw Failure(APXINF_STATUS_UNSUPPORTED,"memory-efficient attention shared-memory limit");
  check_cuda(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 sizeof(typename K::SharedStorage)));
  K::check_supported(parameters<T>(e));
}
template<class T> cudaError_t enqueue(Execution& e) {
  using K=Kernel<T>;
  auto p=parameters<T>(e);
  PyTorchMemEffAttention::attention_kernel_batched_impl<K>
      <<<p.getBlocksGrid(),p.getThreadsGrid(),sizeof(typename K::SharedStorage),
         static_cast<cudaStream_t>(e.bindings.stream)>>>(p);
  return cudaGetLastError();
}
} // namespace
size_t efficient_resource_requirements(const Spec& s,int) {
  size_t bytes=sizeof(float);
  for (auto dim:{s.batch,s.query_tokens,s.query_heads,s.head_dim}) {
    if(dim<=0 || static_cast<uint64_t>(dim)>SIZE_MAX/bytes)
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"memory-efficient attention size overflow");
    bytes*=static_cast<size_t>(dim);
  }
  return bytes;
}
void prepare_efficient(Execution& e) {
  auto state=std::make_unique<State>();
  const auto bytes=efficient_resource_requirements(e.spec,e.configuration);
  if(bytes>e.resource_limit)
    throw Failure(APXINF_STATUS_UNSUPPORTED,"memory-efficient attention workspace exceeds policy");
  check_cuda(cudaMalloc(&state->accumulator,bytes));
  e.provider_state=state.get();
  try {
    if(e.spec.dtype==APXINF_DTYPE_BF16) configure<cutlass::bfloat16_t>(e);
    else configure<cutlass::half_t>(e);
  } catch (...) { e.provider_state=nullptr; throw; }
  e.resource_bytes=bytes;
  state.release();
}
cudaError_t launch_efficient(Execution& e) {
  return e.spec.dtype==APXINF_DTYPE_BF16 ? enqueue<cutlass::bfloat16_t>(e)
                                      : enqueue<cutlass::half_t>(e);
}
void destroy_efficient(Execution& e) noexcept {
  delete static_cast<State*>(e.provider_state); e.provider_state=nullptr;
}
} // namespace apxinf::attention

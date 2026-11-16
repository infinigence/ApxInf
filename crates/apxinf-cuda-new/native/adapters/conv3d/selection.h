// Operator-owned exact selection. Included in the adapter translation unit.
#pragma once
#ifndef APXINF_CONV3D_BUILD_ID
#error "Conv3d selection requires a build fingerprint"
#endif
std::string tuning_key(apxinf_runtime_t runtime,const apxinf_conv3d_spec_t& spec,
    const apxinf_conv3d_policy_t& policy,const apxinf_conv3d_bindings_t& b) {
  cudaDeviceProp device{};check_cuda(cudaGetDeviceProperties(&device,runtime->device));
  int cuda=0,driver=0;check_cuda(cudaRuntimeGetVersion(&cuda));check_cuda(cudaDriverGetVersion(&driver));
  uint64_t cudnn=0;
#ifdef APXINF_OPTIONAL_CUDNN
  if(library().cudnnGetVersion_fn)cudnn=library().cudnnGetVersion_fn();
#endif
  int blas_major=0,blas_minor=0,blas_patch=0;
  apxinf::framework::check_cublas(cublasGetProperty(MAJOR_VERSION,&blas_major));
  apxinf::framework::check_cublas(cublasGetProperty(MINOR_VERSION,&blas_minor));
  apxinf::framework::check_cublas(cublasGetProperty(PATCH_LEVEL,&blas_patch));
  std::ostringstream key;
  key<<"conv3d-recipe-v1|"<<APXINF_CONV3D_BUILD_ID<<'|'<<CUDART_VERSION<<'|'<<cuda<<'|'<<driver<<'|'<<cudnn
     <<'|'<<blas_major<<'.'<<blas_minor<<'.'<<blas_patch
     <<'|'<<device.major<<'|'<<device.minor<<'|'<<device.multiProcessorCount<<'|'<<device.name<<'|';
  for(unsigned char c:device.uuid.bytes)key<<std::hex<<int(c)<<',';
  key<<std::dec<<'|'<<spec.geometry.version<<'|'<<spec.output_channels<<'|'<<spec.geometry.history_frames;
  for(auto v:spec.geometry.input)key<<'|'<<v;
  for(auto values:{spec.geometry.kernel,spec.geometry.stride,spec.geometry.padding_before,
      spec.geometry.padding_after,spec.geometry.output})for(int i=0;i<3;++i)key<<'|'<<values[i];
  // Only the minimum vector alignment affects legality. No addresses or streams.
  for(auto pointer:{b.volume.input,b.volume.history,b.weight,static_cast<const void*>(b.volume.output)}) {
    auto address=reinterpret_cast<uintptr_t>(pointer);unsigned alignment=1;
    while(alignment<16 && address%(alignment*2)==0)alignment*=2;
    key<<'|'<<alignment;
  }
  key<<'|'<<policy.workspace_limit<<"|bf16-fp32-deterministic-graph";
  return key.str();
}
struct TuningProblem {
  using Implementation=Candidate;
  using Execution=::Execution;
  apxinf_runtime_t runtime;
  const apxinf_conv3d_spec_t& spec;
  const apxinf_conv3d_policy_t& policy;
  apxinf_conv3d_bindings_t bindings;
  Buffer output;
  TuningProblem(apxinf_runtime_t r,const apxinf_conv3d_spec_t& s,
      const apxinf_conv3d_policy_t& p,const apxinf_conv3d_bindings_t& b):runtime(r),spec(s),policy(p),bindings(b) {
    output.allocate(add(mul(mul(s.geometry.rows,s.output_channels),2),16));
    // Preserve the caller's alignment class, without writing its output.
    bindings.volume.output=static_cast<char*>(output.ptr)+(reinterpret_cast<uintptr_t>(b.volume.output)%16);
  }
  ~TuningProblem(){cudaStreamSynchronize(stream());}
  const auto& registry(){return ::registry();}
  bool supports(const Candidate& candidate,std::string& reason) {
    if(candidate.fallback)return true;
#ifdef APXINF_OPTIONAL_CUDNN
    if(library().backend_available && library().cudnnGetVersion_fn()==92000 &&
       runtime->compute_major==11 && runtime->compute_minor==0 &&
       reinterpret_cast<uintptr_t>(bindings.volume.input)%16==0 &&
       reinterpret_cast<uintptr_t>(bindings.volume.output)%16==0)return true;
#endif
    reason="backend device/library/alignment";return false;
  }
  void configurations(const Candidate& candidate,std::vector<int>& values) {
    values=candidate.fallback?std::vector<int>{0}:std::vector<int>{0,1,2,3,4};
  }
  std::unique_ptr<Execution> prepare(const Candidate& candidate,int config) {
    return prepare_candidate(runtime,spec,bindings,policy.workspace_limit,candidate,config);
  }
  cudaError_t enqueue(Execution& execution) {check_status(apxinf_conv3d_enqueue(&execution));return cudaSuccess;}
  cudaStream_t stream(){return static_cast<cudaStream_t>(bindings.volume.stream);}
  bool graph_safe(){return true;}
};
std::unique_ptr<Execution> select_execution(apxinf_runtime_t runtime,
    const apxinf_conv3d_spec_t& spec,const apxinf_conv3d_policy_t& policy,
    const apxinf_conv3d_bindings_t& bindings) {
  require(!policy.reserved,"invalid Conv3d policy reserved field");
  const std::string directory=policy.cache_dir?policy.cache_dir:runtime->default_cache_dir+"/conv3d";
  const auto key=tuning_key(runtime,spec,policy,bindings);
  // Preparation only. Serialize same-process persistent recipe writes without
  // adding family state to the shared runtime or invalidating unrelated keys.
  static std::mutex selection_mutex;
  std::lock_guard<std::mutex> lock(selection_mutex);
  using apxinf::framework::Recipe;
  auto restore=[&](const Recipe& recipe) {
    const auto* candidate=registry().find(recipe.provider_id,recipe.implementation_id,recipe.implementation_version);
    if(!candidate || recipe.configuration<0 || recipe.configuration>(candidate->fallback?0:4))
      throw Failure(APXINF_STATUS_UNSUPPORTED,"invalid Conv3d recipe identity");
    return prepare_candidate(runtime,spec,bindings,policy.workspace_limit,*candidate,recipe.configuration);
  };
  Recipe recipe{};bool found=false;
  std::istringstream in(apxinf::framework::read_recipe(directory,key));
  if(in>>recipe.provider_id>>recipe.implementation_id>>recipe.implementation_version>>recipe.configuration){in>>std::ws;found=in.eof();}
  if(found)try {auto e=restore(recipe);e->summary+=" cache=hit";return e;}
  catch(const Failure& f){if(f.status==APXINF_STATUS_CUDA_ERROR)throw;}
  if(policy.online_tune && runtime->allow_online_tune) {
    try {
      std::string report;TuningProblem problem(runtime,spec,policy,bindings);
      recipe=apxinf::framework::autotune(problem,report);
      auto e=restore(recipe);
      std::ostringstream serialized;serialized<<recipe.provider_id<<' '<<recipe.implementation_id<<' '
        <<recipe.implementation_version<<' '<<recipe.configuration;
      apxinf::framework::write_recipe(directory,key,serialized.str());
      e->summary+=" "+report;return e;
    }catch(const Failure& f){if(f.status==APXINF_STATUS_CUDA_ERROR || !policy.allow_fallback)throw;}
  }
  if(!policy.allow_fallback)throw Failure(APXINF_STATUS_UNSUPPORTED,"Conv3d exact recipe miss and fallback disabled");
  auto e=prepare_candidate(runtime,spec,bindings,policy.workspace_limit,*registry().begin(),0);
  e->summary+=" fallback=untuned";return e;
}

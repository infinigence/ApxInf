// Optional cuDNN backend provider. No library calls or allocations at enqueue.
// Included inside the Conv3d adapter namespace after cudnn_library.h.
#pragma once
#if CUDNN_VERSION >= 92000
struct BackendPlan {
  using Descriptor = cudnnBackendDescriptor_t;
  Library& lib;
  cudnnHandle_t handle; // Owned by Direct, destroyed after this plan.
  std::vector<Descriptor> descriptors;
  Descriptor plan=nullptr, pack=nullptr;
  int64_t workspace_bytes=0, engine_id=0;
  BackendPlan(Library& library,cudnnHandle_t owner):lib(library),handle(owner) {}
  ~BackendPlan() {
    for(auto it=descriptors.rbegin();it!=descriptors.rend();++it)
      lib.cudnnBackendDestroyDescriptor_fn(*it);
  }
  Descriptor make(cudnnBackendDescriptorType_t type) {
    Descriptor d=nullptr;ck(lib.cudnnBackendCreateDescriptor_fn(type,&d));
    try { descriptors.push_back(d); }
    catch(...) { lib.cudnnBackendDestroyDescriptor_fn(d); throw; }
    return d;
  }
  void set(Descriptor d,cudnnBackendAttributeName_t name,
      cudnnBackendAttributeType_t type,int64_t count,const void* value) {
    ck(lib.cudnnBackendSetAttribute_fn(d,name,type,count,value));
  }
  void finish(Descriptor d) { ck(lib.cudnnBackendFinalize_fn(d)); }
  Descriptor tensor(int64_t id,const int* dimensions) {
    auto d=make(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
    int64_t dims[5],strides[5],alignment=16;
    for(int i=0;i<5;++i)dims[i]=dimensions[i];
    strides[1]=1;strides[4]=dims[1];strides[3]=mul(strides[4],dims[4]);
    strides[2]=mul(strides[3],dims[3]);strides[0]=mul(strides[2],dims[2]);
    cudnnDataType_t dtype=CUDNN_DATA_BFLOAT16;
    set(d,CUDNN_ATTR_TENSOR_DATA_TYPE,CUDNN_TYPE_DATA_TYPE,1,&dtype);
    set(d,CUDNN_ATTR_TENSOR_DIMENSIONS,CUDNN_TYPE_INT64,5,dims);
    set(d,CUDNN_ATTR_TENSOR_STRIDES,CUDNN_TYPE_INT64,5,strides);
    set(d,CUDNN_ATTR_TENSOR_UNIQUE_ID,CUDNN_TYPE_INT64,1,&id);
    set(d,CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT,CUDNN_TYPE_INT64,1,&alignment);
    finish(d);return d;
  }
  // Configurations are immutable engine/knob identities, not heuristic indices.
  // Qualified only for sm110 and cuDNN92000 by the caller. Numerical notes are
  // checked independently before constructing a plan. Unsupported descriptors
  // are rejected during preparation, never by changing an enqueued execution.
  void prepare(const int* xd,const int* wd,const int* yd,
      const int* padding,const int* stride,int configuration) {
    require(configuration>=0 && configuration<=4,"invalid Conv3d backend configuration");
    auto x=tensor(1,xd),w=tensor(2,wd),y=tensor(3,yd);
    auto conv=make(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
    cudnnDataType_t dtype=CUDNN_DATA_FLOAT;
    cudnnConvolutionMode_t mode=CUDNN_CROSS_CORRELATION;
    int64_t spatial=3,pad[3],step[3],unit[]={1,1,1};
    for(int i=0;i<3;++i){pad[i]=padding[i];step[i]=stride[i];}
    set(conv,CUDNN_ATTR_CONVOLUTION_COMP_TYPE,CUDNN_TYPE_DATA_TYPE,1,&dtype);
    set(conv,CUDNN_ATTR_CONVOLUTION_CONV_MODE,CUDNN_TYPE_CONVOLUTION_MODE,1,&mode);
    set(conv,CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS,CUDNN_TYPE_INT64,1,&spatial);
    set(conv,CUDNN_ATTR_CONVOLUTION_DILATIONS,CUDNN_TYPE_INT64,3,unit);
    set(conv,CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES,CUDNN_TYPE_INT64,3,step);
    set(conv,CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS,CUDNN_TYPE_INT64,3,pad);
    set(conv,CUDNN_ATTR_CONVOLUTION_POST_PADDINGS,CUDNN_TYPE_INT64,3,pad);finish(conv);
    auto op=make(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
    float one=1,zero=0;
    set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&x);
    set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&w);
    set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&y);
    set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&conv);
    set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA,CUDNN_TYPE_FLOAT,1,&one);
    set(op,CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA,CUDNN_TYPE_FLOAT,1,&zero);finish(op);
    auto graph=make(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
    set(graph,CUDNN_ATTR_OPERATIONGRAPH_HANDLE,CUDNN_TYPE_HANDLE,1,&handle);
    set(graph,CUDNN_ATTR_OPERATIONGRAPH_OPS,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&op);finish(graph);
    auto engine=make(CUDNN_BACKEND_ENGINE_DESCRIPTOR);engine_id=configuration==0?75:73;
    set(engine,CUDNN_ATTR_ENGINE_OPERATION_GRAPH,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&graph);
    set(engine,CUDNN_ATTR_ENGINE_GLOBAL_INDEX,CUDNN_TYPE_INT64,1,&engine_id);finish(engine);
    cudnnBackendNumericalNote_t notes[32];int64_t count=0;
    ck(lib.cudnnBackendGetAttribute_fn(engine,CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
        CUDNN_TYPE_NUMERICAL_NOTE,32,&count,notes));
    for(int64_t i=0;i<count;++i)if(notes[i]==CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC ||
        notes[i]==CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS ||
        notes[i]==CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION)
      throw Failure(APXINF_STATUS_UNSUPPORTED,"Conv3d backend numerical contract");
    using Knob=std::pair<cudnnBackendKnobType_t,int64_t>;
    std::vector<Knob> values;
    if(configuration==0)values={{CUDNN_KNOB_TYPE_TILEK,3},{CUDNN_KNOB_TYPE_TILE_CGA_M,1},
      {CUDNN_KNOB_TYPE_TILE_CGA_N,1},{CUDNN_KNOB_TYPE_SPLIT_P_SLC,4},
      {CUDNN_KNOB_TYPE_TILE_M,1},{CUDNN_KNOB_TYPE_TILE_N,4}};
    else values={{CUDNN_KNOB_TYPE_TILEK,3},{CUDNN_KNOB_TYPE_SPLIT_K_SLC,1},
      {CUDNN_KNOB_TYPE_TILE_CGA_M,4},{CUDNN_KNOB_TYPE_TILE_CGA_N,1},
      {CUDNN_KNOB_TYPE_CTA_COUNT,configuration==1?0:1},
      {CUDNN_KNOB_TYPE_STREAM_K,configuration==2?1:0},
      {CUDNN_KNOB_TYPE_TILE_M,configuration==1?3:4},
      {CUDNN_KNOB_TYPE_TILE_N,(configuration==1||configuration==4)?3:4}};
    std::vector<Descriptor> knobs;
    for(auto value:values){auto knob=make(CUDNN_BACKEND_KNOB_CHOICE_DESCRIPTOR);
      set(knob,CUDNN_ATTR_KNOB_CHOICE_KNOB_TYPE,CUDNN_TYPE_KNOB_TYPE,1,&value.first);
      set(knob,CUDNN_ATTR_KNOB_CHOICE_KNOB_VALUE,CUDNN_TYPE_INT64,1,&value.second);
      finish(knob);knobs.push_back(knob);}
    auto config=make(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR);
    set(config,CUDNN_ATTR_ENGINECFG_ENGINE,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&engine);
    set(config,CUDNN_ATTR_ENGINECFG_KNOB_CHOICES,CUDNN_TYPE_BACKEND_DESCRIPTOR,knobs.size(),knobs.data());finish(config);
    plan=make(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    set(plan,CUDNN_ATTR_EXECUTION_PLAN_HANDLE,CUDNN_TYPE_HANDLE,1,&handle);
    set(plan,CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,CUDNN_TYPE_BACKEND_DESCRIPTOR,1,&config);finish(plan);
    ck(lib.cudnnBackendGetAttribute_fn(plan,CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
        CUDNN_TYPE_INT64,1,&count,&workspace_bytes));
    if(count!=1 || workspace_bytes<0)
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,"invalid Conv3d backend workspace");
  }
  void bind(void* x,void* w,void* y,void* workspace) {
    pack=make(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
    void* pointers[]={x,w,y};int64_t ids[]={1,2,3};
    set(pack,CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS,CUDNN_TYPE_INT64,3,ids);
    set(pack,CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS,CUDNN_TYPE_VOID_PTR,3,pointers);
    set(pack,CUDNN_ATTR_VARIANT_PACK_WORKSPACE,CUDNN_TYPE_VOID_PTR,1,&workspace);finish(pack);
  }
  void enqueue() { ck(lib.cudnnBackendExecute_fn(handle,plan,pack)); }
};

#else
// Older development headers retain legacy cuDNN and lowering; they must not
// compile references to newer engine knob enums even with a newer runtime.
struct BackendPlan {
  int64_t workspace_bytes=0,engine_id=0;
  BackendPlan(Library&,cudnnHandle_t) {}
  void prepare(const int*,const int*,const int*,const int*,const int*,int) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,"Conv3d backend needs cuDNN9.20 headers");
  }
  void bind(void*,void*,void*,void*) {}
  void enqueue() {throw Failure(APXINF_STATUS_UNSUPPORTED,"Conv3d backend unavailable");}
};
#endif

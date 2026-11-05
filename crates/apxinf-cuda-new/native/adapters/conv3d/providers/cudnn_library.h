// Included inside the Conv3d adapter namespace.
#pragma once
// Optional system library. No model or binary link-time dependency on cuDNN.
// Symbols and the library stay loaded for the process; executions own handles.
#define CUDNN_SYMBOLS(X) \
 X(cudnnCreate) X(cudnnDestroy) X(cudnnSetStream) \
 X(cudnnCreateTensorDescriptor) X(cudnnDestroyTensorDescriptor) X(cudnnSetTensorNdDescriptorEx) \
 X(cudnnCreateFilterDescriptor) X(cudnnDestroyFilterDescriptor) X(cudnnSetFilterNdDescriptor) \
 X(cudnnCreateConvolutionDescriptor) X(cudnnDestroyConvolutionDescriptor) \
 X(cudnnSetConvolutionNdDescriptor) X(cudnnSetConvolutionMathType) \
 X(cudnnGetConvolutionForwardAlgorithm_v7) X(cudnnConvolutionForward)
#define CUDNN_BACKEND_SYMBOLS(X) \
 X(cudnnGetVersion) X(cudnnBackendCreateDescriptor) X(cudnnBackendDestroyDescriptor) \
 X(cudnnBackendSetAttribute) X(cudnnBackendGetAttribute) X(cudnnBackendFinalize) X(cudnnBackendExecute)
struct Library {
  void* library = nullptr;
  bool available = false, backend_available = false;
#define FIELD(name) decltype(&name) name##_fn = nullptr;
  CUDNN_SYMBOLS(FIELD)
  CUDNN_BACKEND_SYMBOLS(FIELD)
#undef FIELD
  Library() {
    library = dlopen("libcudnn.so.9", RTLD_NOW | RTLD_LOCAL);
    if (!library) return;
#define LOAD(name) name##_fn = reinterpret_cast<decltype(&name)>(dlsym(library, #name)); if (!name##_fn) return;
    CUDNN_SYMBOLS(LOAD)
#undef LOAD
    available = true;
#define LOAD_BACKEND(name) name##_fn = reinterpret_cast<decltype(&name)>(dlsym(library, #name)); if (!name##_fn) return;
    CUDNN_BACKEND_SYMBOLS(LOAD_BACKEND)
#undef LOAD_BACKEND
    backend_available = true;
  }
};
Library& library() { static Library lib; return lib; }
void ck(cudnnStatus_t status) {
  if (status != CUDNN_STATUS_SUCCESS)
    throw Failure(APXINF_STATUS_PROVIDER_ERROR, "Conv3d cuDNN status " + std::to_string(status));
}

#undef CUDNN_SYMBOLS
#undef CUDNN_BACKEND_SYMBOLS

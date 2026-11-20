#include "../../include/apxinf_cuda/conv1d.h"
#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/conv1d.cuh"

#include <cstdint>
#include <memory>
#include <string>
#include <algorithm>
#include <vector>
#ifdef APXINF_CONV1D_CUDNN
#include <cudnn.h>
#endif

namespace {
#include "../../kernels/primitives/math.cuh"

using apxinf::framework::Failure;
namespace kernels = apxinf::conv1d::kernels;

constexpr int kConvThreads = 256;
constexpr int kMaxBlocks = 65535;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

int64_t conv_output_length(const apxinf_conv1d_spec_t& spec) {
  const int64_t span = spec.dilation * (spec.kernel - 1) + 1;
  const int64_t numerator = spec.in_length + 2 * spec.padding - span;
  if (numerator < 0) return -1;
  return numerator / spec.stride + 1;
}

int64_t transpose_output_length(const apxinf_conv1d_spec_t& spec) {
  return (spec.in_length - 1) * spec.stride - 2 * spec.padding +
         spec.dilation * (spec.kernel - 1) + spec.output_padding + 1;
}

void validate_spec(const apxinf_conv1d_spec_t& spec) {
  const bool supported_dtype = spec.dtype == APXINF_DTYPE_F32 ||
                               spec.dtype == APXINF_DTYPE_F16 ||
                               spec.dtype == APXINF_DTYPE_BF16;
  const bool supported_semantic =
      spec.semantic == APXINF_CONV1D_SEMANTIC_CONV1D ||
      spec.semantic == APXINF_CONV1D_SEMANTIC_CONV_TRANSPOSE1D;
  if (spec.version != APXINF_CONV1D_SPEC_VERSION || !supported_semantic ||
      !supported_dtype || spec.has_bias > 1 || spec.batch <= 0 ||
      spec.direct_channel_last > 1 ||
      (spec.direct_channel_last != 0 && spec.has_bias != 0) ||
      spec.in_length <= 0 || spec.in_channels <= 0 || spec.out_channels <= 0 ||
      spec.kernel <= 0 || spec.stride <= 0 || spec.dilation <= 0 ||
      spec.groups <= 0 || spec.padding < 0 || spec.output_padding < 0 ||
      spec.in_channels % spec.groups != 0 ||
      spec.out_channels % spec.groups != 0 || spec.batch > INT32_MAX ||
      spec.in_length > INT32_MAX || spec.in_channels > INT32_MAX ||
      spec.out_channels > INT32_MAX || spec.kernel > INT32_MAX ||
      spec.stride > INT32_MAX || spec.padding > INT32_MAX ||
      spec.dilation > INT32_MAX || spec.groups > INT32_MAX ||
      spec.output_padding > INT32_MAX || spec.out_length <= 0 ||
      spec.out_length > INT32_MAX ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.weight_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.output_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Conv1d Spec");
  }
  const int64_t element_bytes = spec.dtype == APXINF_DTYPE_F32 ? 4 : 2;
  auto checked_count = [element_bytes](int64_t a, int64_t b, int64_t c) {
    if (a > INT64_MAX / b || a * b > INT64_MAX / c ||
        a * b * c > INT64_MAX / element_bytes) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Conv1d tensor size overflow");
    }
  };
  checked_count(spec.batch, spec.in_length, spec.in_channels);
  checked_count(spec.batch, spec.out_length, spec.out_channels);
  checked_count(spec.out_channels, spec.in_channels / spec.groups, spec.kernel);
  if (spec.semantic == APXINF_CONV1D_SEMANTIC_CONV1D && spec.output_padding != 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Conv1d output_padding must be zero");
  }

  const int64_t expected =
      spec.semantic == APXINF_CONV1D_SEMANTIC_CONV1D
          ? conv_output_length(spec)
          : transpose_output_length(spec);
  if (spec.semantic == APXINF_CONV1D_SEMANTIC_CONV_TRANSPOSE1D &&
      spec.dilation != 1) {
    /* PyTorch allows dilation here, and the kernel implements it, but the
     * model could only ever use dilation 1 for the current graph. Keep the
     * contract explicit rather than silently accepting an untested path. */
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "ConvTranspose1d currently requires dilation 1");
  }
  if (expected <= 0 || expected != spec.out_length) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Conv1d Spec out_length does not match the geometry");
  }
  if (spec.semantic == APXINF_CONV1D_SEMANTIC_CONV_TRANSPOSE1D &&
      spec.output_padding >= spec.stride) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "ConvTranspose1d output_padding must be smaller than stride");
  }
}

void validate_bindings(const apxinf_conv1d_spec_t& spec,
                       const apxinf_conv1d_bindings_t& bindings) {
  if (bindings.input == nullptr || bindings.weight == nullptr ||
      bindings.output == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Conv1d input, weight, or output");
  }
  const uintptr_t output_begin = reinterpret_cast<uintptr_t>(bindings.output);
  const uintptr_t output_bytes = uintptr_t(spec.batch * spec.out_length * spec.out_channels) *
      (spec.dtype == APXINF_DTYPE_F32 ? 4 : 2);
  auto reject_overlap = [&](const void* input, uint64_t count) {
    if (!input) return;
    const uintptr_t input_begin = reinterpret_cast<uintptr_t>(input);
    const uint64_t bytes = count * (spec.dtype == APXINF_DTYPE_F32 ? 4 : 2);
    if (output_begin > UINTPTR_MAX - output_bytes || input_begin > UINTPTR_MAX - bytes ||
        (output_begin < input_begin + bytes && input_begin < output_begin + output_bytes)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Conv1d output overlaps an input");
    }
  };
  reject_overlap(bindings.input, spec.batch * spec.in_length * spec.in_channels);
  reject_overlap(bindings.weight, spec.out_channels * (spec.in_channels / spec.groups) * spec.kernel);
  reject_overlap(bindings.bias, spec.out_channels);

  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Conv1d bias binding disagrees with Spec.has_bias");
  }
}

int blocks_for(int64_t count) {
  const int64_t blocks = (count + kConvThreads - 1) / kConvThreads;
  if (blocks < 1) return 1;
  return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

template <class T>
cudaError_t launch_typed(const apxinf_conv1d_spec_t& spec,
                         const apxinf_conv1d_bindings_t& bindings) {
  const auto* input = static_cast<const T*>(bindings.input);
  const auto* weight = static_cast<const T*>(bindings.weight);
  const auto* bias = static_cast<const T*>(bindings.bias);
  auto* output = static_cast<T*>(bindings.output);
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int64_t total = spec.batch * spec.out_length * spec.out_channels;
  const int grid = blocks_for(total);
  if (spec.semantic == APXINF_CONV1D_SEMANTIC_CONV1D) {
    kernels::conv1d_forward<T><<<grid, kConvThreads, 0, stream>>>(
        input, weight, bias, output, spec.batch, spec.in_length,
        spec.in_channels, spec.out_channels, spec.kernel, spec.stride,
        spec.padding, spec.dilation, spec.groups, spec.out_length);
  } else {
    kernels::conv_transpose1d_forward<T><<<grid, kConvThreads, 0, stream>>>(
        input, weight, bias, output, spec.batch, spec.in_length,
        spec.in_channels, spec.out_channels, spec.kernel, spec.stride,
        spec.padding, spec.dilation, spec.groups, spec.out_length);
  }
  return cudaGetLastError();
}

cudaError_t launch(const apxinf_conv1d_spec_t& spec,
                   const apxinf_conv1d_bindings_t& bindings) {
  switch (spec.dtype) {
    case APXINF_DTYPE_F32:
      return launch_typed<float>(spec, bindings);
    case APXINF_DTYPE_F16:
      return launch_typed<__half>(spec, bindings);
    case APXINF_DTYPE_BF16:
      return launch_typed<__nv_bfloat16>(spec, bindings);
    default:
      return cudaErrorInvalidValue;
  }
}

#ifdef APXINF_CONV1D_CUDNN
// Backend graph objects and bindings have the same lifetime as the prepared
// convolution. Enqueue uses the finalized plan and never creates descriptors.
struct CudnnBackendState {
  std::vector<cudnnBackendDescriptor_t> descriptors;
  cudnnBackendDescriptor_t plan = nullptr;
  cudnnBackendDescriptor_t pack = nullptr;

  cudnnBackendDescriptor_t create(cudnnBackendDescriptorType_t type) {
    cudnnBackendDescriptor_t descriptor = nullptr;
    const auto status = cudnnBackendCreateDescriptor(type, &descriptor);
    if (status != CUDNN_STATUS_SUCCESS) {
      throw Failure(APXINF_STATUS_PROVIDER_ERROR, cudnnGetErrorString(status));
    }
    try {
      descriptors.push_back(descriptor);
    } catch (...) {
      cudnnBackendDestroyDescriptor(descriptor);
      throw;
    }
    return descriptor;
  }

  ~CudnnBackendState() {
    for (auto it = descriptors.rbegin(); it != descriptors.rend(); ++it) {
      cudnnBackendDestroyDescriptor(*it);
    }
  }
};
#endif

}  // namespace

extern "C" apxinf_status_t apxinf_conv1d_launch(
    apxinf_runtime_t runtime, const apxinf_conv1d_spec_t* spec,
    const apxinf_conv1d_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Conv1d argument");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    if (spec->direct_channel_last != 0) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "direct channel-last layout requires a prepared execution");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    apxinf::framework::check_cuda(launch(*spec, *bindings));
  });
}

// All handles, descriptors, transforms and workspace belong to one prepared
// execution. Enqueue is asynchronous and allocation-free, including capture.
struct apxinf_conv1d_execution {
  apxinf_conv1d_spec_t spec{};
  apxinf_conv1d_bindings_t bindings{};
  int device = 0;
  std::string summary = "direct.conv1d.v1";
  bool use_temporal = false;
  void* input_temporal = nullptr;
  void* output_temporal = nullptr;
#ifdef APXINF_CONV1D_CUDNN
  bool use_cudnn = false;
  /* Whether this execution really runs in the direct channel-last layout. A
   * requested layout is a preference: a shape the provider serves with a single
   * deterministic channel-last engine keeps the NCHW path instead (see
   * prepare_cudnn), so enqueue dispatches on this and not on the spec. */
  bool channel_last = false;
  std::unique_ptr<CudnnBackendState> backend;
  cudnnHandle_t handle = nullptr;
  cudnnTensorDescriptor_t input_desc = nullptr;
  cudnnTensorDescriptor_t output_desc = nullptr;
  cudnnFilterDescriptor_t filter_desc = nullptr;
  cudnnConvolutionDescriptor_t conv_desc = nullptr;
  cudnnConvolutionFwdAlgo_t forward_algo{};
  cudnnConvolutionBwdDataAlgo_t backward_algo{};
  void* input_ncl = nullptr;
  void* output_ncl = nullptr;
  /* Direct channel-last mode only: the model's [K, C, 1, kernel] weights in the
   * provider's [K, 1, kernel, C] order. Owned here because the permutation is a
   * property of the prepared execution, not of each call; it is written once in
   * prepare, on the same stream enqueue later uses. */
  void* weight_krsc = nullptr;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
#endif
  ~apxinf_conv1d_execution() {
    cudaSetDevice(device);
    if (input_temporal) cudaFree(input_temporal);
    if (output_temporal) cudaFree(output_temporal);
#ifdef APXINF_CONV1D_CUDNN
    backend.reset();
    if (workspace) cudaFree(workspace);
    if (input_ncl) cudaFree(input_ncl);
    if (output_ncl) cudaFree(output_ncl);
    if (weight_krsc) cudaFree(weight_krsc);
    if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
    if (filter_desc) cudnnDestroyFilterDescriptor(filter_desc);
    if (output_desc) cudnnDestroyTensorDescriptor(output_desc);
    if (input_desc) cudnnDestroyTensorDescriptor(input_desc);
    if (handle) cudnnDestroy(handle);
#endif
  }
};

namespace {
int layout_blocks(int64_t batch, int64_t rows, int64_t cols) {
  const int64_t count = batch * ((rows + 31) / 32) * ((cols + 31) / 32);
  return static_cast<int>(count < kMaxBlocks ? count : kMaxBlocks);
}


bool prepare_temporal(apxinf_conv1d_execution& e, uint64_t workspace_limit) {
  const auto& s = e.spec;
  // Small reductions do not amortize two layout transforms. This condition
  // depends only on the operation geometry, never the model or input values.
  if (s.semantic != APXINF_CONV1D_SEMANTIC_CONV1D ||
      s.in_channels / s.groups < 32 || s.out_length < 32) return false;
  const uint64_t bytes_per_element = s.dtype == APXINF_DTYPE_F32 ? 4 : 2;
  const uint64_t input_bytes = uint64_t(s.batch) * s.in_length * s.in_channels * bytes_per_element;
  const uint64_t output_bytes = uint64_t(s.batch) * s.out_length * s.out_channels * bytes_per_element;
  if (input_bytes > workspace_limit || output_bytes > workspace_limit - input_bytes) return false;
  apxinf::framework::check_cuda(cudaMalloc(&e.input_temporal, input_bytes));
  apxinf::framework::check_cuda(cudaMalloc(&e.output_temporal, output_bytes));
  e.use_temporal = true;
  e.summary = "direct.conv1d.temporal.v3.workspace=" + std::to_string(input_bytes + output_bytes);
  return true;
}

template<class T>
void enqueue_temporal(apxinf_conv1d_execution& e) {
  const auto& s = e.spec;
  const auto& b = e.bindings;
  const auto stream = static_cast<cudaStream_t>(b.stream);
  kernels::transpose_layout<T, false><<<layout_blocks(s.batch, s.in_length, s.in_channels), dim3(32, 8), 0, stream>>>(
      static_cast<const T*>(b.input), nullptr, static_cast<T*>(e.input_temporal),
      s.in_length, s.in_channels, s.batch);
  apxinf::framework::check_cuda(cudaGetLastError());
  const int blocks = blocks_for(s.batch * s.out_length * s.out_channels);
  const int64_t paired_blocks_needed =
      (s.batch * s.out_channels * ((s.out_length + 63) / 64) +
       kConvThreads / 32 - 1) / (kConvThreads / 32);
  const int paired_blocks = static_cast<int>(
      std::min<int64_t>(paired_blocks_needed, kMaxBlocks));
  if (s.kernel == 2) {
    kernels::conv1d_forward_temporal_pair<T, 2><<<paired_blocks, kConvThreads, 0, stream>>>(
      static_cast<const T*>(e.input_temporal), static_cast<const T*>(b.weight),
      static_cast<const T*>(b.bias), static_cast<T*>(e.output_temporal),
      s.batch, s.in_length, s.in_channels, s.out_channels, s.kernel,
      s.stride, s.padding, s.dilation, s.groups, s.out_length);
  } else if (s.kernel == 3) {
    kernels::conv1d_forward_temporal_pair<T, 3><<<paired_blocks, kConvThreads, 0, stream>>>(
      static_cast<const T*>(e.input_temporal), static_cast<const T*>(b.weight),
      static_cast<const T*>(b.bias), static_cast<T*>(e.output_temporal),
      s.batch, s.in_length, s.in_channels, s.out_channels, s.kernel,
      s.stride, s.padding, s.dilation, s.groups, s.out_length);
  } else {
    kernels::conv1d_forward_temporal<T><<<blocks, kConvThreads, 0, stream>>>(
      static_cast<const T*>(e.input_temporal), static_cast<const T*>(b.weight),
      static_cast<const T*>(b.bias), static_cast<T*>(e.output_temporal),
      s.batch, s.in_length, s.in_channels, s.out_channels, s.kernel,
      s.stride, s.padding, s.dilation, s.groups, s.out_length);
  }
  apxinf::framework::check_cuda(cudaGetLastError());
  // Bias was accumulated before the dot product by the direct kernel.
  kernels::transpose_layout<T, false><<<layout_blocks(s.batch, s.out_channels, s.out_length), dim3(32, 8), 0, stream>>>(
      static_cast<const T*>(e.output_temporal), nullptr, static_cast<T*>(b.output),
      s.out_channels, s.out_length, s.batch);
  apxinf::framework::check_cuda(cudaGetLastError());
}
} // namespace

#ifdef APXINF_CONV1D_CUDNN
namespace {
void check_cudnn(cudnnStatus_t status) {
  if (status != CUDNN_STATUS_SUCCESS) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR, cudnnGetErrorString(status));
  }
}


void backend_set(cudnnBackendDescriptor_t descriptor,
                 cudnnBackendAttributeName_t name,
                 cudnnBackendAttributeType_t type, int64_t count,
                 const void* value) {
  check_cudnn(cudnnBackendSetAttribute(descriptor, name, type, count, value));
}

cudnnBackendDescriptor_t backend_tensor(CudnnBackendState& state,
    cudnnDataType_t dtype, int64_t id, const int64_t (&dims)[4],
    int64_t alignment) {
  const auto descriptor = state.create(CUDNN_BACKEND_TENSOR_DESCRIPTOR);
  const int64_t strides[] = {dims[1] * dims[2] * dims[3],
                            dims[2] * dims[3], dims[3], 1};
  const bool is_virtual = false;
  backend_set(descriptor, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &dtype);
  backend_set(descriptor, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, 4, dims);
  backend_set(descriptor, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, 4, strides);
  backend_set(descriptor, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &id);
  backend_set(descriptor, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment);
  backend_set(descriptor, CUDNN_ATTR_TENSOR_IS_VIRTUAL, CUDNN_TYPE_BOOLEAN, 1, &is_virtual);
  check_cudnn(cudnnBackendFinalize(descriptor));
  return descriptor;
}

bool prepare_cudnn_backend(apxinf_conv1d_execution& e, uint64_t workspace_limit,
                           bool channel_last) {
  const auto& s = e.spec;
  // The backend's depthwise engine preserves a different FP32 reduction tree
  // from the legacy convolution algorithms. Admit this geometry independently
  // of the caller; other convolutions retain the established legacy path.
  if (s.semantic != APXINF_CONV1D_SEMANTIC_CONV1D ||
      s.in_channels != s.groups || s.out_channels != s.in_channels) return false;
  // This backend describes its tensors in the NCHW scratch layout, so it cannot
  // serve a caller that binds channel-last buffers directly.
  if (channel_last) return false;
  auto state = std::make_unique<CudnnBackendState>();
  const auto dtype = s.dtype == APXINF_DTYPE_F32 ? CUDNN_DATA_FLOAT :
                     s.dtype == APXINF_DTYPE_F16 ? CUDNN_DATA_HALF : CUDNN_DATA_BFLOAT16;
  const int64_t x_dims[] = {s.batch, s.in_channels, 1, s.in_length};
  const int64_t y_dims[] = {s.batch, s.out_channels, 1, s.out_length};
  const int64_t w_dims[] = {s.out_channels, 1, 1, s.kernel};
  const auto x = backend_tensor(*state, dtype, 1, x_dims, 16);
  const auto w = backend_tensor(*state, dtype, 2, w_dims,
      std::min<int64_t>(16, s.weight_alignment));
  const auto y = backend_tensor(*state, dtype, 3, y_dims, 16);
  const auto convolution = state->create(CUDNN_BACKEND_CONVOLUTION_DESCRIPTOR);
  const auto compute = CUDNN_DATA_FLOAT;
  const auto mode = CUDNN_CROSS_CORRELATION;
  const int64_t spatial_dims = 2;
  const int64_t dilation[] = {1, s.dilation};
  const int64_t stride[] = {1, s.stride};
  const int64_t padding[] = {0, s.padding};
  backend_set(convolution, CUDNN_ATTR_CONVOLUTION_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &compute);
  backend_set(convolution, CUDNN_ATTR_CONVOLUTION_CONV_MODE, CUDNN_TYPE_CONVOLUTION_MODE, 1, &mode);
  backend_set(convolution, CUDNN_ATTR_CONVOLUTION_SPATIAL_DIMS, CUDNN_TYPE_INT64, 1, &spatial_dims);
  backend_set(convolution, CUDNN_ATTR_CONVOLUTION_DILATIONS, CUDNN_TYPE_INT64, 2, dilation);
  backend_set(convolution, CUDNN_ATTR_CONVOLUTION_FILTER_STRIDES, CUDNN_TYPE_INT64, 2, stride);
  backend_set(convolution, CUDNN_ATTR_CONVOLUTION_PRE_PADDINGS, CUDNN_TYPE_INT64, 2, padding);
  backend_set(convolution, CUDNN_ATTR_CONVOLUTION_POST_PADDINGS, CUDNN_TYPE_INT64, 2, padding);
  check_cudnn(cudnnBackendFinalize(convolution));
  const auto operation = state->create(CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR);
  const float alpha = 1.0f, beta = 0.0f;
  backend_set(operation, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_ALPHA, CUDNN_TYPE_FLOAT, 1, &alpha);
  backend_set(operation, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_BETA, CUDNN_TYPE_FLOAT, 1, &beta);
  backend_set(operation, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_X, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &x);
  backend_set(operation, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_W, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &w);
  backend_set(operation, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_Y, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &y);
  backend_set(operation, CUDNN_ATTR_OPERATION_CONVOLUTION_FORWARD_CONV_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &convolution);
  check_cudnn(cudnnBackendFinalize(operation));
  const auto graph = state->create(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR);
  backend_set(graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &e.handle);
  backend_set(graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &operation);
  check_cudnn(cudnnBackendFinalize(graph));
  const auto heuristics = state->create(CUDNN_BACKEND_ENGINEHEUR_DESCRIPTOR);
  const auto heuristic_mode = CUDNN_HEUR_MODE_INSTANT;
  backend_set(heuristics, CUDNN_ATTR_ENGINEHEUR_MODE, CUDNN_TYPE_HEUR_MODE, 1, &heuristic_mode);
  backend_set(heuristics, CUDNN_ATTR_ENGINEHEUR_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &graph);
  check_cudnn(cudnnBackendFinalize(heuristics));
  int64_t count = 0;
  check_cudnn(cudnnBackendGetAttribute(heuristics, CUDNN_ATTR_ENGINEHEUR_RESULTS,
      CUDNN_TYPE_BACKEND_DESCRIPTOR, 0, &count, nullptr));
  if (count <= 0 || count > 128) return false;
  std::vector<cudnnBackendDescriptor_t> configurations;
  for (int64_t i = 0; i < count; ++i) {
    configurations.push_back(state->create(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR));
  }
  check_cudnn(cudnnBackendGetAttribute(heuristics, CUDNN_ATTR_ENGINEHEUR_RESULTS,
      CUDNN_TYPE_BACKEND_DESCRIPTOR, count, &count, configurations.data()));
  for (const auto configuration : configurations) {
    auto engine = state->create(CUDNN_BACKEND_ENGINE_DESCRIPTOR);
    int64_t returned = 0;
    if (cudnnBackendGetAttribute(configuration, CUDNN_ATTR_ENGINECFG_ENGINE,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &returned, &engine) != CUDNN_STATUS_SUCCESS) continue;
    cudnnBackendNumericalNote_t notes[CUDNN_NUMERICAL_NOTE_TYPE_COUNT]{};
    int64_t note_count = 0;
    if (cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_NUMERICAL_NOTE,
        CUDNN_TYPE_NUMERICAL_NOTE, CUDNN_NUMERICAL_NOTE_TYPE_COUNT,
        &note_count, notes) != CUDNN_STATUS_SUCCESS) continue;
    bool rejected = false;
    for (int64_t i = 0; i < note_count; ++i) {
      rejected |= notes[i] == CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC ||
                  notes[i] == CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS ||
                  notes[i] == CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION ||
                  (s.dtype == APXINF_DTYPE_F32 && notes[i] == CUDNN_NUMERICAL_NOTE_TENSOR_CORE);
    }
    if (rejected) continue;
    cudnnBackendBehaviorNote_t behaviors[CUDNN_BEHAVIOR_NOTE_TYPE_COUNT]{};
    int64_t behavior_count = 0;
    if (cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE,
        CUDNN_TYPE_BEHAVIOR_NOTE, CUDNN_BEHAVIOR_NOTE_TYPE_COUNT,
        &behavior_count, behaviors) != CUDNN_STATUS_SUCCESS) continue;
    for (int64_t i = 0; i < behavior_count; ++i) {
      rejected |= behaviors[i] == CUDNN_BEHAVIOR_NOTE_REQUIRES_FILTER_INT8x32_REORDER ||
                  behaviors[i] == CUDNN_BEHAVIOR_NOTE_REQUIRES_BIAS_INT8x32_REORDER;
    }
    if (rejected) continue;
    const auto plan = state->create(CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR);
    backend_set(plan, CUDNN_ATTR_EXECUTION_PLAN_HANDLE, CUDNN_TYPE_HANDLE, 1, &e.handle);
    backend_set(plan, CUDNN_ATTR_EXECUTION_PLAN_ENGINE_CONFIG,
        CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &configuration);
    if (cudnnBackendFinalize(plan) != CUDNN_STATUS_SUCCESS) continue;
    int64_t bytes = 0, engine_index = 0;
    if (cudnnBackendGetAttribute(plan, CUDNN_ATTR_EXECUTION_PLAN_WORKSPACE_SIZE,
        CUDNN_TYPE_INT64, 1, &returned, &bytes) != CUDNN_STATUS_SUCCESS ||
        bytes < 0 || uint64_t(bytes) > workspace_limit) continue;
    if (cudnnBackendGetAttribute(engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX,
        CUDNN_TYPE_INT64, 1, &returned, &engine_index) != CUDNN_STATUS_SUCCESS) continue;
    state->plan = plan;
    e.workspace_bytes = size_t(bytes);
    e.summary = "cudnn.conv1d.graph.v2.engine=" + std::to_string(engine_index);
    e.backend = std::move(state);
    return true;
  }
  return false;
}

void bind_cudnn_backend(apxinf_conv1d_execution& e) {
  auto& state = *e.backend;
  state.pack = state.create(CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR);
  const int64_t ids[] = {1, 2, 3};
  void* pointers[] = {e.input_ncl, const_cast<void*>(e.bindings.weight), e.output_ncl};
  backend_set(state.pack, CUDNN_ATTR_VARIANT_PACK_UNIQUE_IDS, CUDNN_TYPE_INT64, 3, ids);
  backend_set(state.pack, CUDNN_ATTR_VARIANT_PACK_DATA_POINTERS, CUDNN_TYPE_VOID_PTR, 3, pointers);
  backend_set(state.pack, CUDNN_ATTR_VARIANT_PACK_WORKSPACE, CUDNN_TYPE_VOID_PTR, 1, &e.workspace);
  check_cudnn(cudnnBackendFinalize(state.pack));
}

void permute_filter(apxinf_conv1d_execution& e, uint64_t filter_elements) {
  const auto& s = e.spec;
  const bool transpose = s.semantic == APXINF_CONV1D_SEMANTIC_CONV_TRANSPOSE1D;
  const int64_t out_channels = transpose ? s.in_channels : s.out_channels;
  const int64_t in_channels = (transpose ? s.out_channels : s.in_channels) / s.groups;
  const auto stream = static_cast<cudaStream_t>(e.bindings.stream);
  const int blocks = blocks_for(int64_t(filter_elements));
  const void* source = e.bindings.weight;
  switch (s.dtype) {
    case APXINF_DTYPE_F32:
      kernels::permute_filter_krsc<float><<<blocks, kConvThreads, 0, stream>>>(
          static_cast<const float*>(source), static_cast<float*>(e.weight_krsc),
          out_channels, in_channels, s.kernel);
      break;
    case APXINF_DTYPE_F16:
      kernels::permute_filter_krsc<__half><<<blocks, kConvThreads, 0, stream>>>(
          static_cast<const __half*>(source), static_cast<__half*>(e.weight_krsc),
          out_channels, in_channels, s.kernel);
      break;
    default:
      kernels::permute_filter_krsc<__nv_bfloat16><<<blocks, kConvThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(source),
          static_cast<__nv_bfloat16*>(e.weight_krsc),
          out_channels, in_channels, s.kernel);
      break;
  }
  apxinf::framework::check_cuda(cudaGetLastError());
}

// Preserve the public NLC tensors and native PyTorch OIK/IOK filter layout.
// A coalesced transform permits the same NCHW Tensor-Core convolution algorithms
// as the reference instead of imposing a new persistent model weight format.
// Shared-memory transpose coalesces both reads and writes. Flatten tile and
// batch indices so large batches do not depend on CUDA's grid-y/z limits.
struct ConvCandidate {
  bool transpose;
  int algo;
  cudnnMathType_t math;
  size_t bytes;
};

// The SIMT entries the v7 list also returns are one to two orders of magnitude
// slower than the tensor-core engines on this hardware: the legacy direct
// engine costs 537 ms per call on the stage-4 16->16 geometry and 8.6 ms on the
// 128->128 k=11 dilation=5 one, against ~0.2 ms and ~0.6 ms. cuDNN reports those
// engines as CUDNN_DEFAULT_MATH, not as CUDNN_FMA_MATH -- the list here is
// (algo=1, math=1) for the tensor-op engine, (algo=1, math=0) and
// (algo=0, math=0) for the two SIMT ones -- so the test is "tensor-op math",
// not "not FMA". Whenever a tensor-op candidate exists the SIMT ones are
// dropped before choosing and before timing -- which is also the universe the
// plain NCHW path always used, so nothing that worked before is lost. A shape
// whose every admissible candidate is SIMT keeps them.
std::vector<ConvCandidate> prefer_tensor_op(std::vector<ConvCandidate> in) {
  const auto is_tensor_op = [](const ConvCandidate& c) {
    return c.math == CUDNN_TENSOR_OP_MATH || c.math == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION;
  };
  if (!std::any_of(in.begin(), in.end(), is_tensor_op)) return in;
  in.erase(std::remove_if(in.begin(), in.end(),
      [&](const ConvCandidate& c) { return !is_tensor_op(c); }), in.end());
  return in;
}

// The v7 ranking is an estimate. On long channel-last sequences it puts the
// legacy direct engine first (measured 537 ms per call on the stage-4 16->16
// geometry, against ~2 ms for the tensor-core engines), so an admissible set is
// timed on the real bindings and the fastest is kept. Prepare is not capture
// safe, the launch uses beta = 0 and no bias, and every execution overwrites the
// output buffer, so this probe does not change any observable result.
void measure_candidates(apxinf_conv1d_execution& e,
                        const std::vector<ConvCandidate>& candidates,
                        void* workspace) {
  const float alpha = 1.0f, beta = 0.0f;
  const auto stream = static_cast<cudaStream_t>(e.bindings.stream);
  const auto launch = [&](const ConvCandidate& candidate) {
    check_cudnn(cudnnSetConvolutionMathType(e.conv_desc, candidate.math));
    if (candidate.transpose) {
      check_cudnn(cudnnConvolutionBackwardData(e.handle, &alpha, e.filter_desc,
          e.weight_krsc, e.input_desc, e.bindings.input, e.conv_desc,
          static_cast<cudnnConvolutionBwdDataAlgo_t>(candidate.algo),
          workspace, candidate.bytes, &beta, e.output_desc, e.bindings.output));
    } else {
      check_cudnn(cudnnConvolutionForward(e.handle, &alpha, e.input_desc, e.bindings.input,
          e.filter_desc, e.weight_krsc, e.conv_desc,
          static_cast<cudnnConvolutionFwdAlgo_t>(candidate.algo),
          workspace, candidate.bytes, &beta, e.output_desc, e.bindings.output));
    }
  };
  cudaEvent_t begin{}, end{};
  apxinf::framework::check_cuda(cudaEventCreate(&begin));
  apxinf::framework::check_cuda(cudaEventCreate(&end));
  const ConvCandidate* best = nullptr;
  float best_ms = 0.0f;
  try {
    for (const auto& candidate : candidates) {
      // The v7 list advertises algorithms this device can compile a plan for but
      // not execute; an unusable candidate is dropped instead of failing prepare.
      try {
        launch(candidate);  // engine compilation and lazy workspace happen here
        apxinf::framework::check_cuda(cudaEventRecord(begin, stream));
        launch(candidate);
        apxinf::framework::check_cuda(cudaEventRecord(end, stream));
        apxinf::framework::check_cuda(cudaEventSynchronize(end));
      } catch (const Failure& error) {
        if (error.status != APXINF_STATUS_PROVIDER_ERROR) throw;
        apxinf::framework::check_cuda(cudaGetLastError());
        continue;
      }
      float elapsed = 0.0f;
      apxinf::framework::check_cuda(cudaEventElapsedTime(&elapsed, begin, end));
      if (best == nullptr || elapsed < best_ms) {
        best = &candidate;
        best_ms = elapsed;
      }
    }
  } catch (...) {
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    throw;
  }
  apxinf::framework::check_cuda(cudaEventDestroy(begin));
  apxinf::framework::check_cuda(cudaEventDestroy(end));
  if (best == nullptr) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "no admissible convolution algorithm executes on this device");
  }
  check_cudnn(cudnnSetConvolutionMathType(e.conv_desc, best->math));
  if (best->transpose) {
    e.backward_algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(best->algo);
  } else {
    e.forward_algo = static_cast<cudnnConvolutionFwdAlgo_t>(best->algo);
  }
  e.workspace_bytes = best->bytes;
  e.summary += ".measured";
}

bool prepare_cudnn(apxinf_conv1d_execution& e, uint64_t workspace_limit) {
  const auto& s = e.spec;
  // A single output channel is bandwidth-bound: layout conversion costs
  // more than the short dot products (vocoder conv_post measured on SM87).
  if (s.semantic == APXINF_CONV1D_SEMANTIC_CONV1D && s.out_channels == 1) return false;
  if (s.out_length > INT32_MAX || s.padding > INT32_MAX || s.stride > INT32_MAX ||
      s.dilation > INT32_MAX || s.groups > INT32_MAX) return false;
  check_cudnn(cudnnCreate(&e.handle));
  check_cudnn(cudnnSetStream(e.handle, static_cast<cudaStream_t>(e.bindings.stream)));
  check_cudnn(cudnnCreateTensorDescriptor(&e.input_desc));
  check_cudnn(cudnnCreateTensorDescriptor(&e.output_desc));
  check_cudnn(cudnnCreateFilterDescriptor(&e.filter_desc));
  check_cudnn(cudnnCreateConvolutionDescriptor(&e.conv_desc));
  const auto dtype = s.dtype == APXINF_DTYPE_F32 ? CUDNN_DATA_FLOAT :
                     s.dtype == APXINF_DTYPE_F16 ? CUDNN_DATA_HALF : CUDNN_DATA_BFLOAT16;
  // Channel-last descriptors over [N, C, 1, L] describe exactly the caller's
  // [batch, length, channels] buffers, so the provider reads and writes them in
  // place instead of the NCHW scratch copies. The layout is a preference, not a
  // requirement; see the single-candidate fallback below.
  const bool transpose = s.semantic == APXINF_CONV1D_SEMANTIC_CONV_TRANSPOSE1D;
  check_cudnn(cudnnSetConvolution2dDescriptor(e.conv_desc, 0, int(s.padding),
      1, int(s.stride), 1, int(s.dilation), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
  check_cudnn(cudnnSetConvolutionGroupCount(e.conv_desc, int(s.groups)));
  check_cudnn(cudnnSetConvolutionMathType(e.conv_desc,
      s.dtype == APXINF_DTYPE_F32 ? CUDNN_FMA_MATH : CUDNN_TENSOR_OP_MATH));
  bool channel_last = s.direct_channel_last != 0;
  bool found = false;
  int returned = 0;
  std::vector<ConvCandidate> candidates;
  for (;;) {
    const auto tensor_format = channel_last ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;
    // The math type steers the ranking of the enumeration query below, so it
    // must be the default one at the start of every attempt: the retry after a
    // single-candidate channel-last attempt would otherwise inherit that
    // candidate's math type. With FMA left over, the NCHW query ranks the
    // legacy FMA engines first and picks one 50x slower than the tensor-core
    // engine the untouched NCHW path gets (128-channel k=11 dilation=5: 8.56 ms
    // against 0.17 ms per call).
    check_cudnn(cudnnSetConvolutionMathType(e.conv_desc,
        s.dtype == APXINF_DTYPE_F32 ? CUDNN_FMA_MATH : CUDNN_TENSOR_OP_MATH));
    check_cudnn(cudnnSetTensor4dDescriptor(e.input_desc, tensor_format, dtype,
        int(s.batch), int(s.in_channels), 1, int(s.in_length)));
    check_cudnn(cudnnSetTensor4dDescriptor(e.output_desc, tensor_format, dtype,
        int(s.batch), int(s.out_channels), 1, int(s.out_length)));
    check_cudnn(cudnnSetFilter4dDescriptor(e.filter_desc, dtype, tensor_format,
        int(transpose ? s.in_channels : s.out_channels),
        int((transpose ? s.out_channels : s.in_channels) / s.groups), 1, int(s.kernel)));
    found = false;
    candidates.clear();
    try {
      found = prepare_cudnn_backend(e, workspace_limit, channel_last);
    } catch (const Failure& error) {
      if (error.status != APXINF_STATUS_PROVIDER_ERROR) throw;
    }
    // Whether the provider offered a real choice before the tensor-core
    // preference was applied; the channel-last fallback below keys on it.
    bool enough = false;
    if (!found && !transpose) {
      cudnnConvolutionFwdAlgoPerf_t perf[CUDNN_CONVOLUTION_FWD_ALGO_COUNT]{};
      const auto status = cudnnGetConvolutionForwardAlgorithm_v7(e.handle,
          e.input_desc, e.filter_desc, e.conv_desc, e.output_desc,
          CUDNN_CONVOLUTION_FWD_ALGO_COUNT, &returned, perf);
      if (status != CUDNN_STATUS_SUCCESS) return false;
      for (int i = 0; i < returned; ++i) {
        if (perf[i].status != CUDNN_STATUS_SUCCESS || perf[i].determinism != CUDNN_DETERMINISTIC ||
            perf[i].memory > workspace_limit) continue;
        size_t bytes = 0;
        if (cudnnGetConvolutionForwardWorkspaceSize(e.handle, e.input_desc, e.filter_desc,
            e.conv_desc, e.output_desc, perf[i].algo, &bytes) != CUDNN_STATUS_SUCCESS ||
            bytes > workspace_limit) continue;
        candidates.push_back({false, int(perf[i].algo), perf[i].mathType, bytes});
      }
      enough = candidates.size() > 1;
      candidates = prefer_tensor_op(std::move(candidates));
      if (!candidates.empty()) {
        e.forward_algo = static_cast<cudnnConvolutionFwdAlgo_t>(candidates.front().algo);
        e.workspace_bytes = candidates.front().bytes;
        check_cudnn(cudnnSetConvolutionMathType(e.conv_desc, candidates.front().math));
        e.summary = "cudnn.conv1d.v1.algo=" + std::to_string(candidates.front().algo);
        found = true;
      }
    } else if (!found) {
      cudnnConvolutionBwdDataAlgoPerf_t perf[CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT]{};
      const auto status = cudnnGetConvolutionBackwardDataAlgorithm_v7(e.handle,
          e.filter_desc, e.input_desc, e.conv_desc, e.output_desc,
          CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT, &returned, perf);
      if (status != CUDNN_STATUS_SUCCESS) return false;
      for (int i = 0; i < returned; ++i) {
        if (perf[i].status != CUDNN_STATUS_SUCCESS || perf[i].determinism != CUDNN_DETERMINISTIC ||
            perf[i].memory > workspace_limit) continue;
        size_t bytes = 0;
        if (cudnnGetConvolutionBackwardDataWorkspaceSize(e.handle, e.filter_desc, e.input_desc,
            e.conv_desc, e.output_desc, perf[i].algo, &bytes) != CUDNN_STATUS_SUCCESS ||
            bytes > workspace_limit) continue;
        candidates.push_back({true, int(perf[i].algo), perf[i].mathType, bytes});
      }
      enough = candidates.size() > 1;
      candidates = prefer_tensor_op(std::move(candidates));
      if (!candidates.empty()) {
        e.backward_algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(candidates.front().algo);
        e.workspace_bytes = candidates.front().bytes;
        check_cudnn(cudnnSetConvolutionMathType(e.conv_desc, candidates.front().math));
        e.summary = "cudnn.conv_transpose1d.v1.algo=" + std::to_string(candidates.front().algo);
        found = true;
      }
    }
    if (!found) return false;
    // A channel-last shape the provider serves with a single deterministic
    // engine leaves nothing to choose or measure -- and that one engine is the
    // legacy direct kernel, which the v7 estimate ranks first: 537 ms per call
    // on the 128-channel k=11 dilation=5 geometry, against ~0.3 ms for the
    // tensor-core engines. Such a shape keeps the NCHW path instead: two
    // transposes (~0.3 ms) over an engine set already tuned before this layout
    // existed. The NCHW enumeration is a different one, so retry without it.
    if (!channel_last || enough) break;
    channel_last = false;
    e.forward_algo = static_cast<cudnnConvolutionFwdAlgo_t>(0);
    e.backward_algo = static_cast<cudnnConvolutionBwdDataAlgo_t>(0);
    e.summary = "direct.conv1d.v1";
  }
  e.channel_last = channel_last;
  const size_t element_bytes = s.dtype == APXINF_DTYPE_F32 ? 4 : 2;
  const uint64_t input_bytes = uint64_t(s.batch) * s.in_length * s.in_channels * element_bytes;
  const uint64_t output_bytes = uint64_t(s.batch) * s.out_length * s.out_channels * element_bytes;
  if (input_bytes > SIZE_MAX || output_bytes > SIZE_MAX) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Conv1d transform size overflow");
  }
  if (channel_last) {
    // No layout scratch is needed; the one prepare-time transform left is the
    // filter permutation into the channel-last order.
    const uint64_t filter_elements = uint64_t(transpose ? s.in_channels : s.out_channels) *
        ((transpose ? s.out_channels : s.in_channels) / s.groups) * s.kernel;
    if (filter_elements > SIZE_MAX / element_bytes) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Conv1d filter size overflow");
    }
    apxinf::framework::check_cuda(cudaMalloc(&e.weight_krsc, filter_elements * element_bytes));
    permute_filter(e, filter_elements);
    // A single admissible candidate needs no probe; several are timed.
    if (candidates.size() > 1) {
      size_t widest = 0;
      for (const auto& candidate : candidates) widest = std::max(widest, candidate.bytes);
      void* probe = nullptr;
      if (widest) apxinf::framework::check_cuda(cudaMalloc(&probe, widest));
      try {
        measure_candidates(e, candidates, probe);
      } catch (...) {
        if (probe) cudaFree(probe);
        throw;
      }
      if (probe) apxinf::framework::check_cuda(cudaFree(probe));
    }
  } else {
    apxinf::framework::check_cuda(cudaMalloc(&e.input_ncl, input_bytes));
    apxinf::framework::check_cuda(cudaMalloc(&e.output_ncl, output_bytes));
  }
  if (e.workspace_bytes) apxinf::framework::check_cuda(cudaMalloc(&e.workspace, e.workspace_bytes));
  if (e.backend) {
    bind_cudnn_backend(e);
    // Some legal depthwise engines compile on their first execution. Materialize
    // that work on scratch tensors while prepare is outside capture; public
    // input/output tensors are not modified by this warmup.
    const auto stream = static_cast<cudaStream_t>(e.bindings.stream);
    apxinf::framework::check_cuda(cudaMemsetAsync(e.input_ncl, 0, input_bytes, stream));
    check_cudnn(cudnnBackendExecute(e.handle, e.backend->plan, e.backend->pack));
    apxinf::framework::check_cuda(cudaStreamSynchronize(stream));
  }
  if (channel_last) {
    e.summary += ".channel_last_direct";
  } else if (s.direct_channel_last) {
    e.summary += ".channel_last_fallback";
  }
  e.summary += ".workspace=" + std::to_string(e.workspace_bytes);
  e.use_cudnn = true;
  return true;
}

template<class T>
void enqueue_cudnn(apxinf_conv1d_execution& e) {
  const auto& s = e.spec;
  const auto& b = e.bindings;
  const auto stream = static_cast<cudaStream_t>(b.stream);
  const float alpha = 1.0f, beta = 0.0f;
  if (e.channel_last) {
    // The provider reads and writes the caller's buffers in place; the filter is
    // the prepare-time permutation, and the bias-free result is final.
    if (s.semantic == APXINF_CONV1D_SEMANTIC_CONV1D) {
      check_cudnn(cudnnConvolutionForward(e.handle, &alpha, e.input_desc, b.input,
          e.filter_desc, e.weight_krsc, e.conv_desc, e.forward_algo, e.workspace,
          e.workspace_bytes, &beta, e.output_desc, b.output));
    } else {
      check_cudnn(cudnnConvolutionBackwardData(e.handle, &alpha, e.filter_desc,
          e.weight_krsc, e.input_desc, b.input, e.conv_desc, e.backward_algo,
          e.workspace, e.workspace_bytes, &beta, e.output_desc, b.output));
    }
    return;
  }
  kernels::transpose_layout<T, false><<<layout_blocks(s.batch, s.in_length, s.in_channels), dim3(32, 8), 0, stream>>>(
      static_cast<const T*>(b.input), nullptr, static_cast<T*>(e.input_ncl),
      s.in_length, s.in_channels, s.batch);
  apxinf::framework::check_cuda(cudaGetLastError());
  if (e.backend) {
    check_cudnn(cudnnBackendExecute(e.handle, e.backend->plan, e.backend->pack));
  } else if (s.semantic == APXINF_CONV1D_SEMANTIC_CONV1D) {
    check_cudnn(cudnnConvolutionForward(e.handle, &alpha, e.input_desc, e.input_ncl,
        e.filter_desc, b.weight, e.conv_desc, e.forward_algo, e.workspace,
        e.workspace_bytes, &beta, e.output_desc, e.output_ncl));
  } else {
    check_cudnn(cudnnConvolutionBackwardData(e.handle, &alpha, e.filter_desc, b.weight,
        e.input_desc, e.input_ncl, e.conv_desc, e.backward_algo, e.workspace,
        e.workspace_bytes, &beta, e.output_desc, e.output_ncl));
  }
  kernels::transpose_layout<T, true><<<layout_blocks(s.batch, s.out_channels, s.out_length), dim3(32, 8), 0, stream>>>(
      static_cast<const T*>(e.output_ncl), static_cast<const T*>(b.bias),
      static_cast<T*>(b.output), s.out_channels, s.out_length, s.batch);
  apxinf::framework::check_cuda(cudaGetLastError());
}
} // namespace
#endif

extern "C" apxinf_status_t apxinf_conv1d_prepare(
    apxinf_runtime_t runtime, const apxinf_conv1d_spec_t* spec,
    const apxinf_conv1d_bindings_t* bindings, uint64_t workspace_limit,
    uint32_t ordered_accumulation,
    apxinf_conv1d_execution_t* execution) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime || !spec || !bindings || !execution) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Conv1d prepare argument");
    }
    *execution = nullptr;
    if (ordered_accumulation > 1) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid convolution accumulation constraint");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    cudaStreamCaptureStatus capture_status{};
    apxinf::framework::check_cuda(cudaStreamIsCapturing(
        static_cast<cudaStream_t>(bindings->stream), &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Conv1d prepare is not capture-safe");
    }
    auto result = std::make_unique<apxinf_conv1d_execution>();
    result->spec = *spec;
    result->bindings = *bindings;
    result->device = runtime->device;
    if (spec->direct_channel_last != 0 && ordered_accumulation != 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "direct channel-last layout has no ordered-accumulation path");
    }
    if (ordered_accumulation) prepare_temporal(*result, workspace_limit);
#ifdef APXINF_CONV1D_CUDNN
    if (!ordered_accumulation) {
      bool prepared = false;
      try {
        prepared = prepare_cudnn(*result, workspace_limit);
      } catch (const Failure& error) {
        if (error.status != APXINF_STATUS_PROVIDER_ERROR) throw;
      }
      if (!prepared) {
        // The direct layout has no baseline to fall back to: the channel-last
        // kernel applies the bias itself and expects the provider's geometry.
        if (spec->direct_channel_last != 0) {
          throw Failure(APXINF_STATUS_UNSUPPORTED,
                        "direct channel-last layout requires the convolution provider");
        }
        // Provider rejection leaves the validated baseline available. Destroy
        // partially constructed handles/layouts before installing the fallback.
        result = std::make_unique<apxinf_conv1d_execution>();
        result->spec = *spec;
        result->bindings = *bindings;
        result->device = runtime->device;
      }
    }
#else
    if (spec->direct_channel_last != 0) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "direct channel-last layout requires the convolution provider");
    }
#endif
    *execution = result.release();
  });
}

extern "C" apxinf_status_t apxinf_conv1d_enqueue(apxinf_conv1d_execution_t execution) {
  return apxinf::framework::abi_boundary([&] {
    if (!execution) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Conv1d execution");
    apxinf::framework::check_cuda(cudaSetDevice(execution->device));
    if (execution->use_temporal) {
      switch (execution->spec.dtype) {
        case APXINF_DTYPE_F32: enqueue_temporal<float>(*execution); break;
        case APXINF_DTYPE_F16: enqueue_temporal<__half>(*execution); break;
        case APXINF_DTYPE_BF16: enqueue_temporal<__nv_bfloat16>(*execution); break;
      }
      return;
    }
#ifdef APXINF_CONV1D_CUDNN
    if (execution->use_cudnn) {
      switch (execution->spec.dtype) {
        case APXINF_DTYPE_F32: enqueue_cudnn<float>(*execution); break;
        case APXINF_DTYPE_F16: enqueue_cudnn<__half>(*execution); break;
        case APXINF_DTYPE_BF16: enqueue_cudnn<__nv_bfloat16>(*execution); break;
      }
      return;
    }
#endif
    apxinf::framework::check_cuda(launch(execution->spec, execution->bindings));
  });
}

extern "C" void apxinf_conv1d_destroy(apxinf_conv1d_execution_t execution) {
  delete execution;
}
extern "C" const char* apxinf_conv1d_summary(apxinf_conv1d_execution_t execution) {
  return execution ? execution->summary.c_str() : "null";
}

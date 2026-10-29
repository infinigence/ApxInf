#include "../internal.h"
#include "../../../kernels/custom/gemm.cuh"

#ifdef APXINF_GEMM_CUTLASS
#include "../../../kernels/cutlass/ops/gemm/gemm_bf16_sm100.h"
#include "../../../kernels/cutlass/ops/gemm/gemm_e4m3_sm100.h"
#include "../../../kernels/cutlass/ops/gemm/gemm_nvfp4_sm100.h"
#endif
#ifdef APXINF_GEMM_CUTLASS_SM89
#include "../../../kernels/cutlass/ops/gemm/gemm_bf16_sm89.h"
#endif
#ifdef APXINF_GEMM_CUTLASS_SM87_W8A8
#include "../../../kernels/cutlass/ops/gemm/gemm_i8_bf16_sm80.h"
#endif

namespace apxinf::gemm {
namespace {

struct CutlassNvfp4State {
#ifdef APXINF_GEMM_CUTLASS
  apxinf::cuda_new::cutlass_ops::Nvfp4GemmExecution* nvfp4_execution = nullptr;
#endif

  ~CutlassNvfp4State() { release_resources(); }

  void release_resources() noexcept {
#ifdef APXINF_GEMM_CUTLASS
    if (nvfp4_execution != nullptr) {
      apxinf::cuda_new::cutlass_ops::nvfp4_gemm_destroy(nvfp4_execution);
    }
    nvfp4_execution = nullptr;
#endif
  }
};

CutlassNvfp4State& provider(Execution& state) {
  return *static_cast<CutlassNvfp4State*>(state.provider_state);
}

#ifdef APXINF_GEMM_CUTLASS_SM89
__global__ void pack_adjacent_bf16(const __nv_bfloat16* source,
                                   __nv_bfloat16* destination,
                                   int64_t rows,
                                   int64_t columns) {
  const int64_t half = columns / 2;
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < rows * columns;
       index += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = index / columns;
    const int64_t column = index % columns;
    const int64_t pair = column < half ? column : column - half;
    const int64_t lane = column < half ? 0 : 1;
    destination[row * columns + 2 * pair + lane] = source[index];
  }
}
#endif

#ifdef APXINF_GEMM_CUTLASS_SM87_W8A8
__global__ void transpose_w8a8_weight(const int8_t* source,
                                      int8_t* destination,
                                      int64_t rows,
                                      int64_t columns) {
  for (int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
       index < rows * columns;
       index += int64_t(gridDim.x) * blockDim.x) {
    const int64_t row = index / columns;
    const int64_t column = index % columns;
    destination[column * rows + row] = source[index];
  }
}
#endif

void check_cutlass_status(int status) {
  if (status != 0) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "CUTLASS launch status " + std::to_string(status));
  }
}

}  // namespace

size_t cutlass_fp8_resource_requirements(const Spec&) { return 0; }

// Packing is preparation-local. Executions only bind an already packed B.
size_t cutlass_w8a8_resource_requirements(const Spec&) { return 0; }
size_t cutlass_geglu_resource_requirements(const Spec&) { return 0; }
void prepare_cutlass_fp8_gemm(Execution&) {}
void prepare_cutlass_w8a8(Execution&) {}
void prepare_cutlass_geglu(Execution&) {}
void prepare_cutlass_geglu_sm89(Execution&) {}

void pack_weight(const Spec& spec, const apxinf_gemm_bindings_t& bindings,
                 void* destination, uint32_t layout) {
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int blocks = static_cast<int>(
      std::min<int64_t>((spec.k * spec.n + 255) / 256, 4096));
  switch (layout) {
    case 2:
      apxinf::cuda_new::custom::pack_gate_up<<<blocks, 256, 0, stream>>>(
          bindings.b, destination, spec.b_dtype, spec.k, spec.n);
      break;
#ifdef APXINF_GEMM_CUTLASS_SM89
    case 3:
      pack_adjacent_bf16<<<blocks, 256, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(bindings.b),
          static_cast<__nv_bfloat16*>(destination), spec.k, spec.n);
      break;
#endif
#ifdef APXINF_GEMM_CUTLASS_SM87_W8A8
    case 1:
      transpose_w8a8_weight<<<blocks, 256, 0, stream>>>(
          static_cast<const int8_t*>(bindings.b),
          static_cast<int8_t*>(destination), spec.k, spec.n);
      break;
#endif
    default:
      throw Failure(APXINF_STATUS_UNSUPPORTED, "unsupported weight packing");
  }
  check_cuda(cudaGetLastError());
}

void repack_weight_inplace(const Spec& spec,
                          const apxinf_gemm_bindings_t& bindings,
                          uint32_t target_layout, uint32_t& layout) {
  if (target_layout != 4) {
    const size_t bytes =
        static_cast<size_t>(spec.k * spec.n) * dtype_bytes(spec.b_dtype);
    TemporaryWeight temporary(bytes);
    pack_weight(spec, bindings, temporary.pointer, target_layout);
    const auto stream = static_cast<cudaStream_t>(bindings.stream);
    // Establish that packing succeeded before overwriting the source.
    check_cuda(cudaStreamSynchronize(stream));
    layout = APXINF_GEMM_LAYOUT_INVALID;  // Failed writeback must never be retried as canonical bytes.
    check_cuda(cudaMemcpyAsync(const_cast<void*>(bindings.b), temporary.pointer,
                              bytes, cudaMemcpyDeviceToDevice, stream));
    check_cuda(cudaStreamSynchronize(stream));
  }  // temporary is freed here, never owned by an Execution.
  layout = target_layout;
}

void destroy_cutlass(Execution& state) noexcept {
  delete static_cast<CutlassNvfp4State*>(state.provider_state);
  state.provider_state = nullptr;
}

cudaError_t launch_cutlass_fp8_gemm(Execution& state) {
  const auto& bindings = state.bindings;
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  using namespace apxinf::cuda_new::cutlass_ops;
  // supports_cutlass_fp8 admits exactly these two output dtypes.
  const auto entry = spec.output_dtype == APXINF_DTYPE_BF16 ? fp8_gemm_bf16
                                                            : fp8_gemm_f16;
  check_cutlass_status(entry(
      bindings.a, bindings.b, bindings.output, spec.m, spec.n, spec.k,
      bindings.alpha, state.configuration, stream));
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

cudaError_t launch_cutlass_w8a8(Execution& state) {
#ifdef APXINF_GEMM_CUTLASS_SM87_W8A8
  const auto& bindings = state.bindings;
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  return apxinf::cuda_new::cutlass_ops::w8a8_gemm_bf16(
      bindings.a, bindings.b, bindings.a_scales, bindings.b_scales,
      bindings.output, static_cast<int>(spec.m), static_cast<int>(spec.n),
      static_cast<int>(spec.k), stream);
#else
  (void)state;
  return cudaErrorNotSupported;
#endif
}

cudaError_t launch_cutlass_fp8_geglu(Execution& state) {
  const auto& bindings = state.bindings;
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  check_cutlass_status(
      apxinf::cuda_new::cutlass_ops::fp8_dual_geglu_detail::production_dual_geglu(
          bindings.a, bindings.b, bindings.output, spec.m, spec.n / 2, spec.k,
          spec.n, bindings.alpha, bindings.output_scale, stream));
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

cudaError_t launch_cutlass_bf16_geglu(Execution& state) {
  const auto& bindings = state.bindings;
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  check_cutlass_status(apxinf::cuda_new::cutlass_ops::bf16_dual_geglu_detail::
                           production_dual_geglu_bf16(
                               bindings.a, bindings.b, bindings.output, spec.m,
                               spec.n / 2, spec.k, spec.n, stream));
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

cudaError_t launch_cutlass_bf16_geglu_sm89(Execution& state) {
  const auto& bindings = state.bindings;
#ifdef APXINF_GEMM_CUTLASS_SM89
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  check_cutlass_status(
      apxinf::cuda_new::cutlass_ops::bf16_sm89_detail::interleaved_geglu(
          bindings.a, bindings.b, bindings.output, spec.m,
          spec.n / 2, spec.k, spec.n, 0, stream));
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}


size_t cutlass_nvfp4_resource_requirements(const Spec& spec) {
#ifdef APXINF_GEMM_CUTLASS
  // Report the largest requirement across tactics so the workspace budget can
  // prefilter this candidate before any tactic has been chosen.
  size_t worst = 0;
  const int tactics = apxinf::cuda_new::cutlass_ops::nvfp4_gemm_tactic_count();
  for (int tactic = 0; tactic < tactics; ++tactic) {
    worst = std::max(worst, apxinf::cuda_new::cutlass_ops::nvfp4_gemm_workspace_bytes(
                                static_cast<int>(spec.m), static_cast<int>(spec.n),
                                static_cast<int>(spec.k),
                                static_cast<int>(spec.sf_vec_size), tactic));
  }
  return worst;
#else
  (void)spec;
  return 0;
#endif
}

void prepare_cutlass_nvfp4(Execution& state) {
#ifdef APXINF_GEMM_CUTLASS
  const auto& bindings = state.bindings;
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  auto owned = std::make_unique<CutlassNvfp4State>();
  owned->nvfp4_execution = apxinf::cuda_new::cutlass_ops::nvfp4_gemm_prepare(
      bindings.a, bindings.a_block_scales, bindings.b, bindings.b_block_scales,
      bindings.output, static_cast<int>(spec.m), static_cast<int>(spec.n),
      static_cast<int>(spec.k), static_cast<int>(spec.sf_vec_size),
      bindings.alpha / bindings.output_scale, state.configuration, stream);
  if (owned->nvfp4_execution == nullptr) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "CUTLASS NVFP4 persistent prepare failed");
  }
  const size_t bytes =
      apxinf::cuda_new::cutlass_ops::nvfp4_gemm_execution_workspace_bytes(
          owned->nvfp4_execution);
  if (bytes > state.resource_limit) {
    owned->release_resources();
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "NVFP4 GEMM workspace exceeds the policy limit");
  }
  state.resource_bytes = bytes;
  state.provider_state = owned.release();
#else
  (void)state;
  throw Failure(APXINF_STATUS_PROVIDER_ERROR, "CUTLASS support is not built");
#endif
}

cudaError_t launch_cutlass_nvfp4(Execution& state) {
  const auto& bindings = state.bindings;
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  auto& resources = provider(state);
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  if (resources.nvfp4_execution == nullptr) return cudaErrorInvalidResourceHandle;
  check_cutlass_status(apxinf::cuda_new::cutlass_ops::nvfp4_gemm_launch(
      resources.nvfp4_execution));
  return cudaSuccess;

#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

}  // namespace apxinf::gemm

extern "C" apxinf_status_t apxinf_gemm_relayout_weight(
    const void* source, void* destination, int64_t k, int64_t n,
    uint32_t dtype, uint32_t from, uint32_t to, apxinf_cuda_stream_t handle) {
  return apxinf::gemm::abi_boundary([&] {
    if (source == nullptr || destination == nullptr || k <= 0 || n <= 0 ||
        k > INT64_MAX / n / 4 || from < 1 || from > 4 || to < 1 || to > 4 ||
        dtype > APXINF_DTYPE_I32 ||
        ((from == 2 || to == 2) && n % 512 != 0) ||
        ((from == 3 || to == 3) && n % 2 != 0)) {
      throw apxinf::gemm::Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid weight relayout");
    }
    const size_t bytes = size_t(k*n) * apxinf::gemm::dtype_bytes(dtype);
    const auto a = reinterpret_cast<uintptr_t>(source), b = reinterpret_cast<uintptr_t>(destination);
    if (a > UINTPTR_MAX-bytes || b > UINTPTR_MAX-bytes || (a < b+bytes && b < a+bytes)) {
      throw apxinf::gemm::Failure(APXINF_STATUS_INVALID_ARGUMENT, "weight relayout buffers overlap");
    }
    const auto stream = static_cast<cudaStream_t>(handle);
    cudaStreamCaptureStatus capture;
    apxinf::gemm::check_cuda(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
      throw apxinf::gemm::Failure(APXINF_STATUS_INVALID_ARGUMENT, "relayout weights before capture");
    }
    if (from == to) {
      apxinf::gemm::check_cuda(cudaMemcpyAsync(
          destination, source, bytes, cudaMemcpyDeviceToDevice, stream));
      apxinf::gemm::check_cuda(cudaStreamSynchronize(stream));
      return;
    }
    const int blocks = int(std::min<int64_t>((k*n+255)/256,4096));
    switch (apxinf::gemm::dtype_bytes(dtype)) {
      case 1: apxinf::cuda_new::custom::relayout_weight_bits<<<blocks,256,0,stream>>>(static_cast<const uint8_t*>(source),static_cast<uint8_t*>(destination),k,n,from,to); break;
      case 2: apxinf::cuda_new::custom::relayout_weight_bits<<<blocks,256,0,stream>>>(static_cast<const uint16_t*>(source),static_cast<uint16_t*>(destination),k,n,from,to); break;
      case 4: apxinf::cuda_new::custom::relayout_weight_bits<<<blocks,256,0,stream>>>(static_cast<const uint32_t*>(source),static_cast<uint32_t*>(destination),k,n,from,to); break;
    }
    apxinf::gemm::check_cuda(cudaGetLastError());
    apxinf::gemm::check_cuda(cudaStreamSynchronize(stream));
  });
}

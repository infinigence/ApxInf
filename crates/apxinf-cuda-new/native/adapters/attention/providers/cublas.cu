#include "../internal.h"
#include "../../cublas_resources.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cstdint>

namespace apxinf::attention::kernels {

// Head-major BF16 QK and P, with one warp per row. The lane-strided local
// sum followed by a butterfly reduction preserves eager FP32 softmax rounding
// before the explicit BF16 probability boundary. No intermediate is on host.
template <int Columns>
__global__ void attention_input_softmax(__nv_bfloat16* scores, const float* bias,
    int batch, int heads, int queries, int keys, bool broadcast_batch,
    bool causal, int query_start) {
  constexpr int width = Columns < 32 ? Columns : 32;
  constexpr int items = Columns / width;
  const int lane = threadIdx.x;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.y + threadIdx.y;
  if (row >= static_cast<int64_t>(batch) * heads * queries) return;
  const int query = row % queries;
  const int64_t bias_row = broadcast_batch ? row % (static_cast<int64_t>(heads) * queries) : row;
  float values[items];
  float maximum = -CUDART_INF_F;
#pragma unroll
  for (int i = 0; i < items; ++i) {
    const int key = lane + i * width;
    float value = -CUDART_INF_F;
    if (key < keys && (!causal || key <= query_start + query)) {
      value = __bfloat162float(__float2bfloat16(
          __bfloat162float(scores[row * keys + key]) + bias[bias_row * keys + key]));
    }
    values[i] = value;
    maximum = fmaxf(maximum, value);
  }
  const unsigned active = __activemask();
  for (int delta = width / 2; delta; delta /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(active, maximum, delta, width));
  float sum = 0.0F;
#pragma unroll
  for (int i = 0; i < items; ++i) {
    values[i] = maximum == -CUDART_INF_F ? 0.0F : expf(values[i] - maximum);
    sum += values[i];
  }
  for (int delta = width / 2; delta; delta /= 2)
    sum += __shfl_xor_sync(active, sum, delta, width);
#pragma unroll
  for (int i = 0; i < items; ++i) {
    const int key = lane + i * width;
    if (key < keys)
      scores[row * keys + key] = __float2bfloat16(sum == 0.0F ? 0.0F : values[i] / sum);
  }
}

inline cudaError_t launch_attention_input_softmax(void* scores, const float* bias,
    int batch, int heads, int queries, int keys, bool broadcast_batch,
    bool causal, int query_start, cudaStream_t stream) {
  const int64_t rows = static_cast<int64_t>(batch) * heads * queries;
#define APXINF_INPUT_SOFTMAX(C) \
  case C: { \
    constexpr int width = C < 32 ? C : 32; \
    constexpr int rows_per_block = 128 / width; \
    attention_input_softmax<C><<<static_cast<unsigned>((rows + rows_per_block - 1) / rows_per_block), dim3(width, rows_per_block), 0, stream>>>( \
        static_cast<__nv_bfloat16*>(scores), bias, batch, heads, queries, keys, broadcast_batch, causal, query_start); \
    break; \
  }
  int columns = 1;
  while (columns < keys) columns *= 2;
  switch (columns) {
    APXINF_INPUT_SOFTMAX(1)
    APXINF_INPUT_SOFTMAX(2)
    APXINF_INPUT_SOFTMAX(4)
    APXINF_INPUT_SOFTMAX(8)
    APXINF_INPUT_SOFTMAX(16)
    APXINF_INPUT_SOFTMAX(32)
    APXINF_INPUT_SOFTMAX(64)
    APXINF_INPUT_SOFTMAX(128)
    APXINF_INPUT_SOFTMAX(256)
    APXINF_INPUT_SOFTMAX(512)
    APXINF_INPUT_SOFTMAX(1024)
    default: return cudaErrorInvalidValue;
  }
#undef APXINF_INPUT_SOFTMAX
  return cudaGetLastError();
}


// Block reduction for the F32 provider. All threads consume shared scratch
// before another reduction can reuse it.
template <bool Maximum>
__device__ inline float f32_softmax_reduce(float value, float* scratch) {
  for (int delta = 16; delta; delta /= 2) {
    const float other = __shfl_down_sync(0xffffffff, value, delta);
    value = Maximum ? fmaxf(value, other) : value + other;
  }
  if (threadIdx.x % 32 == 0) scratch[threadIdx.x / 32] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float result = scratch[0];
    for (int warp = 1; warp < 4; ++warp)
      result = Maximum ? fmaxf(result, scratch[warp]) : result + scratch[warp];
    scratch[0] = result;
  }
  __syncthreads();
  const float result = scratch[0];
  __syncthreads();
  return result;
}

__global__ void attention_f32_softmax(float* scores, int queries, int keys,
                                     bool causal, int query_start) {
  __shared__ float scratch[4];
  const int64_t row = blockIdx.x;
  const int64_t valid = causal
      ? min(static_cast<int64_t>(keys), static_cast<int64_t>(query_start) + row % queries + 1)
      : keys;
  float* values = scores + row * keys;
  float maximum = -CUDART_INF_F;
  for (int key = threadIdx.x; key < valid; key += 128)
    maximum = fmaxf(maximum, values[key]);
  maximum = f32_softmax_reduce<true>(maximum, scratch);
  float sum = 0.0F;
  for (int key = threadIdx.x; key < keys; key += 128) {
    const float value = key < valid && maximum != -CUDART_INF_F
        ? expf(values[key] - maximum) : 0.0F;
    values[key] = value;
    sum += value;
  }
  sum = f32_softmax_reduce<false>(sum, scratch);
  for (int key = threadIdx.x; key < keys; key += 128)
    values[key] = sum == 0.0F ? 0.0F : values[key] / sum;
}

inline cudaError_t launch_attention_f32_softmax(float* scores, int batch,
    int heads, int queries, int keys, bool causal, int query_start,
    cudaStream_t stream) {
  const int64_t rows = static_cast<int64_t>(batch) * heads * queries;
  attention_f32_softmax<<<static_cast<unsigned>(rows), 128, 0, stream>>>(
      scores, queries, keys, causal, query_start);
  return cudaGetLastError();
}

}  // namespace apxinf::attention::kernels


namespace apxinf::attention {
namespace {
using apxinf::framework::check_cublas;
constexpr size_t kWorkspaceBytes = providers::CublasStreamResources::workspace_bytes;
struct CublasState {
  std::shared_ptr<providers::CublasStreamResources> shared;
  void* scores = nullptr;
  ~CublasState() {
    if (scores != nullptr) cudaFree(scores);
  }
};
}  // namespace

size_t cublas_resource_requirements(const Spec& spec, int) {
  const uint64_t elements = static_cast<uint64_t>(spec.batch) * spec.query_heads *
                            spec.query_tokens * spec.key_tokens;
  if (elements > (SIZE_MAX - kWorkspaceBytes) / dtype_bytes(spec.dtype))
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Attention cuBLAS scratch size overflow");
  return kWorkspaceBytes + static_cast<size_t>(elements) * dtype_bytes(spec.dtype);
}

void prepare_cublas(Execution& execution) {
  const size_t bytes = cublas_resource_requirements(execution.spec, 0);
  if (bytes > execution.resource_limit)
    throw Failure(APXINF_STATUS_UNSUPPORTED, "Attention cuBLAS scratch exceeds policy");
  auto state = std::make_unique<CublasState>();
  state->shared = providers::acquire_cublas_stream_resources(
      execution.device, static_cast<cudaStream_t>(execution.bindings.stream));
  check_cuda(cudaMalloc(&state->scores, bytes - kWorkspaceBytes));
  execution.resource_bytes = bytes;
  execution.provider_state = state.release();
}


cudaError_t launch_bf16_intermediate_cublas(Execution& execution) {
  const auto& s = execution.spec;
  const auto& b = execution.bindings;
  auto& state = *static_cast<CublasState*>(execution.provider_state);
  const auto stream = static_cast<cudaStream_t>(b.stream);
  // Handle, stream and workspace are immutable after preparation. Serialize
  // the whole attention sequence with other executions sharing this stream.
  std::lock_guard<std::mutex> lock(state.shared->enqueue_mutex);
  const float one = 1.0F, zero = 0.0F;
  const int h = s.query_heads, d = s.head_dim, q = s.query_tokens, k = s.key_tokens;
  const auto* query = static_cast<const __nv_bfloat16*>(b.query);
  const auto* key = static_cast<const __nv_bfloat16*>(b.key);
  const auto* value = static_cast<const __nv_bfloat16*>(b.value);
  auto* output = static_cast<__nv_bfloat16*>(b.output);
  // Canonical B,T,H,D tensors are interleaved column-major D-by-T matrices.
  // Each head's scores/probabilities are contiguous column-major K-by-Q.
  for (uint32_t batch = 0; batch < s.batch; ++batch) {
    check_cublas(cublasGemmStridedBatchedEx(state.shared->handle, CUBLAS_OP_T, CUBLAS_OP_N,
        k, q, d, &b.scale, key + static_cast<int64_t>(batch) * k * h * d,
        CUDA_R_16BF, h * d, d, query + static_cast<int64_t>(batch) * q * h * d,
        CUDA_R_16BF, h * d, d, &zero,
        static_cast<__nv_bfloat16*>(state.scores) + static_cast<int64_t>(batch) * h * q * k,
        CUDA_R_16BF, k, static_cast<int64_t>(q) * k, h,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  check_cuda(kernels::launch_attention_input_softmax(state.scores,
      static_cast<const float*>(b.bias), s.batch, h, q, k,
      s.bias_broadcast_batch != 0, s.mask == APXINF_ATTENTION_MASK_CAUSAL,
      s.query_start, stream));
  for (uint32_t batch = 0; batch < s.batch; ++batch) {
    check_cublas(cublasGemmStridedBatchedEx(state.shared->handle, CUBLAS_OP_N, CUBLAS_OP_N,
        d, q, k, &one, value + static_cast<int64_t>(batch) * k * h * d,
        CUDA_R_16BF, h * d, d,
        static_cast<__nv_bfloat16*>(state.scores) + static_cast<int64_t>(batch) * h * q * k,
        CUDA_R_16BF, k, static_cast<int64_t>(q) * k, &zero,
        output + static_cast<int64_t>(batch) * q * h * d,
        CUDA_R_16BF, h * d, d, h, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
  }
  return cudaGetLastError();
}

// Pedantic F32 GEMMs retain full F32 products (no TF32 conversion). Prepared
// handle/workspace are shared on one stream; score storage precedes capture.
cudaError_t launch_f32_cublas(Execution& execution) {
  const auto& s = execution.spec;
  const auto& b = execution.bindings;
  auto& state = *static_cast<CublasState*>(execution.provider_state);
  const auto stream = static_cast<cudaStream_t>(b.stream);
  std::lock_guard<std::mutex> lock(state.shared->enqueue_mutex);
  const float one = 1.0F, zero = 0.0F;
  const int h = s.query_heads, d = s.head_dim, q = s.query_tokens, k = s.key_tokens;
  auto* scores = static_cast<float*>(state.scores);
  for (uint32_t batch = 0; batch < s.batch; ++batch) {
    check_cublas(cublasSgemmStridedBatched(state.shared->handle,
        CUBLAS_OP_T, CUBLAS_OP_N, k, q, d, &b.scale,
        static_cast<const float*>(b.key) + static_cast<int64_t>(batch) * k * h * d,
        h * d, d,
        static_cast<const float*>(b.query) + static_cast<int64_t>(batch) * q * h * d,
        h * d, d, &zero, scores + static_cast<int64_t>(batch) * h * q * k,
        k, static_cast<int64_t>(q) * k, h));
  }
  check_cuda(kernels::launch_attention_f32_softmax(scores, s.batch, h, q, k,
      s.mask == APXINF_ATTENTION_MASK_CAUSAL, s.query_start, stream));
  for (uint32_t batch = 0; batch < s.batch; ++batch) {
    check_cublas(cublasSgemmStridedBatched(state.shared->handle,
        CUBLAS_OP_N, CUBLAS_OP_N, d, q, k, &one,
        static_cast<const float*>(b.value) + static_cast<int64_t>(batch) * k * h * d,
        h * d, d, scores + static_cast<int64_t>(batch) * h * q * k,
        k, static_cast<int64_t>(q) * k, &zero,
        static_cast<float*>(b.output) + static_cast<int64_t>(batch) * q * h * d,
        h * d, d, h));
  }
  return cudaGetLastError();
}

void destroy_cublas(Execution& execution) noexcept {
  delete static_cast<CublasState*>(execution.provider_state);
  execution.provider_state = nullptr;
}
}  // namespace apxinf::attention

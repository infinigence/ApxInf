#include "../include/apxinf_cuda/conv2d.h"
#include "../framework/runtime_internal.h"
#include "../kernels/custom/conv2d.cuh"

#include <cublasLt.h>

#include <climits>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

using apxinf::framework::Failure;
using apxinf::framework::check_cuda;
using apxinf::framework::check_cublas;
namespace kernels = apxinf::conv2d::kernels;

constexpr int kConvThreads = 256;
constexpr int kMaxBlocks = 65535;
// Upper bound handed to the cuBLASLt heuristic for algorithm scratch space.
constexpr size_t kMaxLtWorkspaceBytes = 32ull * 1024 * 1024;
// Upper bound on the im2col expansion; above this the materialized matrix
// costs more memory than the im2col + GEMM trade is worth.
constexpr uint64_t kMaxIm2colBytes = 512ull * 1024 * 1024;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

int64_t conv_output_size(int64_t in, int64_t kernel, int64_t stride, int64_t pad) {
  const int64_t numerator = in + 2 * pad - kernel;
  if (numerator < 0) return -1;
  return numerator / stride + 1;
}

uint64_t conv_element_bytes(const apxinf_conv2d_spec_t& spec) {
  return spec.dtype == APXINF_DTYPE_F32 ? 4 : 2;
}

void validate_spec(const apxinf_conv2d_spec_t& spec) {
  const bool supported_dtype = spec.dtype == APXINF_DTYPE_F32 ||
                               spec.dtype == APXINF_DTYPE_F16 ||
                               spec.dtype == APXINF_DTYPE_BF16;
  if (spec.version != APXINF_CONV2D_SPEC_VERSION || !supported_dtype ||
      spec.has_bias > 1 || spec.batch <= 0 || spec.in_h <= 0 ||
      spec.in_w <= 0 || spec.in_channels <= 0 || spec.out_channels <= 0 ||
      spec.kernel_h <= 0 || spec.kernel_w <= 0 || spec.stride_h <= 0 ||
      spec.stride_w <= 0 || spec.pad_h < 0 || spec.pad_w < 0 ||
      spec.out_h <= 0 || spec.out_w <= 0 || spec.batch > INT32_MAX ||
      spec.in_h > INT32_MAX || spec.in_w > INT32_MAX ||
      spec.in_channels > INT32_MAX || spec.out_channels > INT32_MAX ||
      spec.kernel_h > INT32_MAX || spec.kernel_w > INT32_MAX ||
      spec.stride_h > INT32_MAX || spec.stride_w > INT32_MAX ||
      spec.pad_h > INT32_MAX || spec.pad_w > INT32_MAX ||
      spec.out_h > INT32_MAX || spec.out_w > INT32_MAX ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.weight_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.output_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Conv2d Spec");
  }
  const int64_t element_bytes = spec.dtype == APXINF_DTYPE_F32 ? 4 : 2;
  auto checked_count = [element_bytes](int64_t a, int64_t b, int64_t c, int64_t d) {
    if (a > INT64_MAX / b || a * b > INT64_MAX / c ||
        a * b * c > INT64_MAX / d ||
        a * b * c * d > INT64_MAX / element_bytes) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Conv2d tensor size overflow");
    }
  };
  checked_count(spec.batch, spec.in_h, spec.in_w, spec.in_channels);
  checked_count(spec.batch, spec.out_h, spec.out_w, spec.out_channels);
  checked_count(spec.out_channels, spec.in_channels, spec.kernel_h, spec.kernel_w);

  const int64_t expected_h =
      conv_output_size(spec.in_h, spec.kernel_h, spec.stride_h, spec.pad_h);
  const int64_t expected_w =
      conv_output_size(spec.in_w, spec.kernel_w, spec.stride_w, spec.pad_w);
  if (expected_h <= 0 || expected_w <= 0 || expected_h != spec.out_h ||
      expected_w != spec.out_w) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Conv2d Spec out_h/out_w does not match the geometry");
  }
}

void validate_bindings(const apxinf_conv2d_spec_t& spec,
                       const apxinf_conv2d_bindings_t& bindings) {
  if (bindings.input == nullptr || bindings.weight == nullptr ||
      bindings.output == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Conv2d input, weight, or output");
  }
  const uint64_t element_bytes = spec.dtype == APXINF_DTYPE_F32 ? 4 : 2;
  const uintptr_t output_begin = reinterpret_cast<uintptr_t>(bindings.output);
  const uintptr_t output_bytes =
      uintptr_t(spec.batch * spec.out_h * spec.out_w * spec.out_channels) *
      element_bytes;
  auto reject_overlap = [&](const void* input, uint64_t count) {
    if (!input) return;
    const uintptr_t input_begin = reinterpret_cast<uintptr_t>(input);
    const uint64_t bytes = count * element_bytes;
    if (output_begin > UINTPTR_MAX - output_bytes || input_begin > UINTPTR_MAX - bytes ||
        (output_begin < input_begin + bytes && input_begin < output_begin + output_bytes)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Conv2d output overlaps an input");
    }
  };
  reject_overlap(bindings.input, spec.batch * spec.in_h * spec.in_w * spec.in_channels);
  reject_overlap(bindings.weight,
                 spec.out_channels * spec.in_channels * spec.kernel_h * spec.kernel_w);
  reject_overlap(bindings.bias, spec.out_channels);

  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Conv2d bias binding disagrees with Spec.has_bias");
  }
}

int blocks_for(int64_t count) {
  const int64_t blocks = (count + kConvThreads - 1) / kConvThreads;
  if (blocks < 1) return 1;
  return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

/* im2col for the cuBLASLt fast path. Expands the padded input into the
 * row-major matrix COL[rows, K], rows = B*OH*OW, K = Cin*KH*KW, with
 *   COL[((b*OH + oh)*OW + ow), ci*KH*KW + kh*KW + kw] =
 *       input[((b*IH + oh*stride_h - pad_h + kh)*IW + ow*stride_w - pad_w + kw)*Cin + ci]
 * and zero when the sampled coordinate falls outside the input (padding).
 * The k index order matches the innermost dims of weight [Cout, Cin, KH, KW]
 * so the following GEMM consumes the weight exactly as stored. One thread
 * per element with a grid-stride loop; values are copied without any type
 * conversion (im2col is pure data movement, the GEMM accumulates in FP32). */
template <class T>
__global__ void im2col_kernel(const T* __restrict__ input,
                              T* __restrict__ col, int64_t batch,
                              int64_t in_h, int64_t in_w, int64_t in_channels,
                              int64_t out_h, int64_t out_w, int64_t kernel_h,
                              int64_t kernel_w, int64_t stride_h,
                              int64_t stride_w, int64_t pad_h, int64_t pad_w) {
  const int64_t rows = batch * out_h * out_w;
  const int64_t cols = in_channels * kernel_h * kernel_w;
  const int64_t total = rows * cols;
  const int64_t step = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
       index < total; index += step) {
    const int64_t col_index = index % cols;
    int64_t row = index / cols;
    const int64_t ow = row % out_w;
    row /= out_w;
    const int64_t oh = row % out_h;
    const int64_t b = row / out_h;
    const int64_t kw = col_index % kernel_w;
    int64_t rem = col_index / kernel_w;
    const int64_t kh = rem % kernel_h;
    const int64_t ci = rem / kernel_h;
    const int64_t ih = oh * stride_h - pad_h + kh;
    const int64_t iw = ow * stride_w - pad_w + kw;
    if (ih < 0 || ih >= in_h || iw < 0 || iw >= in_w) {
      col[index] = kernels::from_float<T>(0.0f);
    } else {
      col[index] = input[((b * in_h + ih) * in_w + iw) * in_channels + ci];
    }
  }
}

template <class T>
cudaError_t launch_im2col_typed(const apxinf_conv2d_spec_t& spec,
                                const apxinf_conv2d_bindings_t& bindings,
                                void* im2col_buffer, cudaStream_t stream) {
  const auto* input = static_cast<const T*>(bindings.input);
  auto* col = static_cast<T*>(im2col_buffer);
  const int64_t rows = spec.batch * spec.out_h * spec.out_w;
  const int64_t cols = spec.in_channels * spec.kernel_h * spec.kernel_w;
  im2col_kernel<T><<<blocks_for(rows * cols), kConvThreads, 0, stream>>>(
      input, col, spec.batch, spec.in_h, spec.in_w, spec.in_channels,
      spec.out_h, spec.out_w, spec.kernel_h, spec.kernel_w, spec.stride_h,
      spec.stride_w, spec.pad_h, spec.pad_w);
  return cudaGetLastError();
}

cudaDataType_t conv_cuda_dtype(const apxinf_conv2d_spec_t& spec) {
  switch (spec.dtype) {
    case APXINF_DTYPE_F32:
      return CUDA_R_32F;
    case APXINF_DTYPE_F16:
      return CUDA_R_16F;
    case APXINF_DTYPE_BF16:
      return CUDA_R_16BF;
    default:
      return CUDA_R_32F;  // unreachable, gated by fast_geometry_eligible
  }
}

/* Geometry gate for the im2col + cuBLASLt path. The dtype must be one the
 * GEMM accepts, and the materialized im2col matrix must stay under
 * kMaxIm2colBytes. The check is written with division so a huge rows*K
 * product cannot overflow. */
bool fast_geometry_eligible(const apxinf_conv2d_spec_t& spec) {
  if (spec.dtype != APXINF_DTYPE_F32 && spec.dtype != APXINF_DTYPE_F16 &&
      spec.dtype != APXINF_DTYPE_BF16) {
    return false;
  }
  const int64_t rows = spec.batch * spec.out_h * spec.out_w;
  const int64_t cols = spec.in_channels * spec.kernel_h * spec.kernel_w;
  if (rows < 1 || cols < 1) return false;
  const uint64_t max_elements =
      (kMaxIm2colBytes - 1) / conv_element_bytes(spec);
  return static_cast<uint64_t>(rows) <= max_elements / static_cast<uint64_t>(cols);
}

/* GEMM mapping for the fast path (all cuBLASLt matrices are column-major):
 *
 *   The convolution output is row-major OUT[rows, Cout]. Reinterpreted as a
 *   column-major matrix it is D[Cout, rows] with ld = Cout, so the GEMM is
 *   m = Cout, n = rows, k = K.
 *
 *   A = the weight tensor, stored row-major [Cout, K]. Reinterpreted as a
 *   column-major matrix it is [K, Cout] with ld = K, and with opA = T the
 *   operated matrix op(A) = [Cout, K] is exactly the weight matrix as
 *   stored. This is why the fast path never transposes or repacks the
 *   weight: the row-major weight already is the transpose of the column-
 *   major operand cuBLASLt wants to consume.
 *
 *   B = the im2col buffer, written row-major [rows, K]. Reinterpreted as a
 *   column-major matrix it is [K, rows] with ld = K, used with opB = N.
 *
 *   C = D = the output binding (column-major [Cout, rows], ld = Cout) with
 *   beta = 0; C is never read.
 *
 * Pointer alignment is not re-checked here: valid_alignment already
 * guarantees power-of-two alignment up to 256 bytes for every binding, and
 * the layouts' leading dimensions are element counts fixed by the geometry. */
struct LtConvPlan {
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;  // weight, column-major [K, Cout]
  cublasLtMatrixLayout_t b_layout = nullptr;  // im2col, column-major [K, rows]
  cublasLtMatrixLayout_t c_layout = nullptr;  // output, column-major [Cout, rows]
  cublasLtMatrixLayout_t d_layout = nullptr;  // output, column-major [Cout, rows]
  cublasLtMatmulAlgo_t algo{};
  bool has_algo = false;
  bool with_bias = false;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  void* im2col_buffer = nullptr;
  size_t im2col_bytes = 0;

  LtConvPlan() = default;
  LtConvPlan(const LtConvPlan&) = delete;
  LtConvPlan& operator=(const LtConvPlan&) = delete;

  ~LtConvPlan() { release(); }

  // Idempotent: safe on a partially built plan and on a released one.
  void release() noexcept {
    if (workspace != nullptr) cudaFree(workspace);
    if (im2col_buffer != nullptr) cudaFree(im2col_buffer);
    workspace = nullptr;
    im2col_buffer = nullptr;
    if (a_layout != nullptr) cublasLtMatrixLayoutDestroy(a_layout);
    if (b_layout != nullptr) cublasLtMatrixLayoutDestroy(b_layout);
    if (c_layout != nullptr) cublasLtMatrixLayoutDestroy(c_layout);
    if (d_layout != nullptr) cublasLtMatrixLayoutDestroy(d_layout);
    a_layout = nullptr;
    b_layout = nullptr;
    c_layout = nullptr;
    d_layout = nullptr;
    if (matmul_desc != nullptr) cublasLtMatmulDescDestroy(matmul_desc);
    matmul_desc = nullptr;
    has_algo = false;
  }
};

/* Builds the cuBLASLt plan into *plan. Any failure (allocation, descriptor
 * creation, heuristic) releases the partially built plan and returns false
 * so the caller falls back to the naive kernel; nothing is thrown. */
bool try_build_lt_plan(const apxinf_conv2d_spec_t& spec,
                       const apxinf_conv2d_bindings_t& bindings,
                       cublasLtHandle_t handle, LtConvPlan* plan) noexcept {
  try {
    const int64_t rows = spec.batch * spec.out_h * spec.out_w;
    const int64_t gemm_k = spec.in_channels * spec.kernel_h * spec.kernel_w;
    const int64_t gemm_m = spec.out_channels;
    const cudaDataType_t data_type = conv_cuda_dtype(spec);

    plan->im2col_bytes = static_cast<size_t>(static_cast<uint64_t>(rows) *
                                             static_cast<uint64_t>(gemm_k) *
                                             conv_element_bytes(spec));
    check_cuda(cudaMalloc(&plan->im2col_buffer, plan->im2col_bytes));

    check_cublas(cublasLtMatmulDescCreate(&plan->matmul_desc,
                                          CUBLAS_COMPUTE_32F, CUDA_R_32F));
    const cublasOperation_t transa = CUBLAS_OP_T;
    check_cublas(cublasLtMatmulDescSetAttribute(
        plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa,
        sizeof(transa)));
    const cublasOperation_t transb = CUBLAS_OP_N;
    check_cublas(cublasLtMatmulDescSetAttribute(
        plan->matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb,
        sizeof(transb)));

    plan->with_bias = spec.has_bias != 0;
    if (plan->with_bias) {
      const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
      check_cublas(cublasLtMatmulDescSetAttribute(
          plan->matmul_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
          sizeof(epilogue)));
      const cudaDataType_t bias_type = data_type;
      check_cublas(cublasLtMatmulDescSetAttribute(
          plan->matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_type,
          sizeof(bias_type)));
      const void* bias_pointer = bindings.bias;
      check_cublas(cublasLtMatmulDescSetAttribute(
          plan->matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer,
          sizeof(bias_pointer)));
    }

    check_cublas(cublasLtMatrixLayoutCreate(&plan->a_layout, data_type,
                                            gemm_k, gemm_m, gemm_k));
    check_cublas(cublasLtMatrixLayoutCreate(&plan->b_layout, data_type,
                                            gemm_k, rows, gemm_k));
    check_cublas(cublasLtMatrixLayoutCreate(&plan->c_layout, data_type,
                                            gemm_m, rows, gemm_m));
    check_cublas(cublasLtMatrixLayoutCreate(&plan->d_layout, data_type,
                                            gemm_m, rows, gemm_m));

    cublasLtMatmulPreference_t preference = nullptr;
    check_cublas(cublasLtMatmulPreferenceCreate(&preference));
    const size_t workspace_limit = kMaxLtWorkspaceBytes;
    cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_limit, sizeof(workspace_limit));
    if (status != CUBLAS_STATUS_SUCCESS) {
      cublasLtMatmulPreferenceDestroy(preference);
      check_cublas(status);
    }
    cublasLtMatmulHeuristicResult_t candidates[8]{};
    int candidate_count = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        handle, plan->matmul_desc, plan->a_layout, plan->b_layout,
        plan->c_layout, plan->d_layout, preference, 8, candidates,
        &candidate_count);
    cublasLtMatmulPreferenceDestroy(preference);
    check_cublas(status);
    if (candidate_count < 1 || candidates[0].state != CUBLAS_STATUS_SUCCESS) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "Conv2d cuBLASLt heuristic is unavailable");
    }
    plan->algo = candidates[0].algo;
    plan->has_algo = true;
    plan->workspace_bytes = candidates[0].workspaceSize;
    if (plan->workspace_bytes > kMaxLtWorkspaceBytes) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "Conv2d cuBLASLt workspace policy exceeded");
    }
    if (plan->workspace_bytes != 0) {
      check_cuda(cudaMalloc(&plan->workspace, plan->workspace_bytes));
    }
    return true;
  } catch (...) {
    plan->release();
    return false;
  }
}

/* Runs one fast-path convolution: im2col into the plan's buffer, then the
 * cuBLASLt GEMM with an optional fused bias epilogue. Both input and output
 * data are re-read on every call; only buffer pointers and the plan are
 * baked in. The bias pointer is refreshed from the bindings before the GEMM
 * so the same helper serves the prepared and direct paths. */
void run_lt_conv(LtConvPlan& plan, cublasLtHandle_t handle,
                 const apxinf_conv2d_spec_t& spec,
                 const apxinf_conv2d_bindings_t& bindings) {
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  cudaError_t im2col_status = cudaErrorInvalidValue;
  switch (spec.dtype) {
    case APXINF_DTYPE_F32:
      im2col_status = launch_im2col_typed<float>(spec, bindings,
                                                 plan.im2col_buffer, stream);
      break;
    case APXINF_DTYPE_F16:
      im2col_status = launch_im2col_typed<__half>(spec, bindings,
                                                   plan.im2col_buffer, stream);
      break;
    case APXINF_DTYPE_BF16:
      im2col_status = launch_im2col_typed<__nv_bfloat16>(spec, bindings,
                                                          plan.im2col_buffer,
                                                          stream);
      break;
    default:
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "Conv2d fast path invoked with an unsupported dtype");
  }
  check_cuda(im2col_status);

  if (plan.with_bias) {
    const void* bias_pointer = bindings.bias;
    check_cublas(cublasLtMatmulDescSetAttribute(
        plan.matmul_desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer,
        sizeof(bias_pointer)));
  }
  const float alpha = 1.0F;
  const float beta = 0.0F;
  check_cublas(cublasLtMatmul(handle, plan.matmul_desc, &alpha, bindings.weight,
                              plan.a_layout, plan.im2col_buffer,
                              plan.b_layout, &beta, bindings.output,
                              plan.c_layout, bindings.output, plan.d_layout,
                              &plan.algo, plan.workspace,
                              plan.workspace_bytes, stream));
}

/* Naive direct-convolution launcher, used as the fallback whenever the fast
 * path is unavailable. */
template <class T>
cudaError_t launch_typed(const apxinf_conv2d_spec_t& spec,
                         const apxinf_conv2d_bindings_t& bindings) {
  const auto* input = static_cast<const T*>(bindings.input);
  const auto* weight = static_cast<const T*>(bindings.weight);
  const auto* bias = static_cast<const T*>(bindings.bias);
  auto* output = static_cast<T*>(bindings.output);
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int64_t total = spec.batch * spec.out_h * spec.out_w * spec.out_channels;
  kernels::conv2d_forward<T><<<blocks_for(total), kConvThreads, 0, stream>>>(
      input, weight, bias, output, spec.batch, spec.in_h, spec.in_w,
      spec.in_channels, spec.out_channels, spec.kernel_h, spec.kernel_w,
      spec.stride_h, spec.stride_w, spec.pad_h, spec.pad_w, spec.out_h,
      spec.out_w);
  return cudaGetLastError();
}

cudaError_t launch(const apxinf_conv2d_spec_t& spec,
                   const apxinf_conv2d_bindings_t& bindings) {
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

/* ---- Per-device plan cache for the direct (unprepared) launch path ----
 *
 * The direct launch entry point used to be allocation-free including under
 * stream capture. The first fast-path invocation for a new geometry now
 * allocates static scratch (im2col buffer + cuBLASLt workspace) and creates
 * descriptors, neither of which may happen while capturing. qwen3_asr does
 * not use stream capture, so the static buffers are warmed on a normal
 * stream by the first call; if a capture is already active and the plan is
 * not ready yet, we fall back to the naive kernel. Buffers are grow-only:
 * entries are never freed or shrunk, so once warmed every call is again
 * allocation-free. The cache itself is deliberately leaked at process exit
 * to avoid destroying cuBLASLt state during CUDA context teardown. */
bool same_lt_geometry(const apxinf_conv2d_spec_t& a,
                      const apxinf_conv2d_spec_t& b) {
  return a.dtype == b.dtype && a.has_bias == b.has_bias &&
         a.batch == b.batch && a.in_h == b.in_h && a.in_w == b.in_w &&
         a.in_channels == b.in_channels &&
         a.out_channels == b.out_channels &&
         a.kernel_h == b.kernel_h && a.kernel_w == b.kernel_w &&
         a.stride_h == b.stride_h && a.stride_w == b.stride_w &&
         a.pad_h == b.pad_h && a.pad_w == b.pad_w &&
         a.out_h == b.out_h && a.out_w == b.out_w;
}

struct LtCacheEntry {
  apxinf_conv2d_spec_t geometry{};
  LtConvPlan plan;
  bool usable = false;
};

struct LtDeviceCache {
  cublasLtHandle_t handle = nullptr;
  std::vector<std::unique_ptr<LtCacheEntry>> entries;
};

std::mutex& lt_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

std::map<int, LtDeviceCache>& lt_device_caches() {
  static auto* caches = new std::map<int, LtDeviceCache>();
  return *caches;
}

LtCacheEntry* find_lt_entry(LtDeviceCache& cache,
                            const apxinf_conv2d_spec_t& spec) {
  for (const auto& entry : cache.entries) {
    if (same_lt_geometry(entry->geometry, spec)) return entry.get();
  }
  return nullptr;
}

/* Attempts the fast path for the unprepared launch entry point. Returns
 * true when the convolution was enqueued via im2col + cuBLASLt; false when
 * the caller should run the naive kernel (gate rejected, capture with an
 * unready plan, or a build failure recorded earlier). Launch-time errors
 * inside the fast path propagate as exceptions. */
bool try_direct_fast(int device, const apxinf_conv2d_spec_t& spec,
                     const apxinf_conv2d_bindings_t& bindings) {
  if (!fast_geometry_eligible(spec)) return false;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  cudaStreamCaptureStatus capture_status{};
  // A failed capture query (e.g. the legacy stream during someone else's
  // capture) is treated conservatively as "not fast-path eligible".
  if (cudaStreamIsCapturing(stream, &capture_status) != cudaSuccess) {
    return false;
  }
  const bool capturing = capture_status != cudaStreamCaptureStatusNone;

  std::lock_guard<std::mutex> guard(lt_cache_mutex());
  LtDeviceCache& cache = lt_device_caches()[device];
  LtCacheEntry* entry = find_lt_entry(cache, spec);
  if (entry == nullptr) {
    if (capturing) return false;
    auto fresh = std::make_unique<LtCacheEntry>();
    fresh->geometry = spec;
    if (cache.handle == nullptr &&
        cublasLtCreate(&cache.handle) != CUBLAS_STATUS_SUCCESS) {
      cache.handle = nullptr;
      // Record the geometry as unusable so the failure is not retried on
      // every call.
      cache.entries.push_back(std::move(fresh));
      return false;
    }
    fresh->usable = try_build_lt_plan(spec, bindings, cache.handle,
                                      &fresh->plan);
    entry = fresh.get();
    cache.entries.push_back(std::move(fresh));
    if (!entry->usable) return false;
  } else if (!entry->usable) {
    return false;
  }
  run_lt_conv(entry->plan, cache.handle, spec, bindings);
  return true;
}

}  // namespace

extern "C" apxinf_status_t apxinf_conv2d_launch(
    apxinf_runtime_t runtime, const apxinf_conv2d_spec_t* spec,
    const apxinf_conv2d_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Conv2d argument");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    if (try_direct_fast(runtime->device, *spec, *bindings)) return;
    apxinf::framework::check_cuda(launch(*spec, *bindings));
  });
}

// A prepared execution snapshots the validated spec and bindings and, when
// the geometry qualifies, owns a cuBLASLt plan (descriptors, algo, static
// workspace and im2col buffer). Enqueue remains asynchronous; only the fast
// path's first prepare allocates. Enqueue is allocation-free in both modes,
// including under capture once the plan exists.
struct apxinf_conv2d_execution {
  apxinf_conv2d_spec_t spec{};
  apxinf_conv2d_bindings_t bindings{};
  int device = 0;
  std::string summary = "direct.conv2d.v1";
  bool fast = false;
  cublasLtHandle_t lt_handle = nullptr;
  LtConvPlan lt_plan{};

  ~apxinf_conv2d_execution() {
    lt_plan.release();
    if (lt_handle != nullptr) cublasLtDestroy(lt_handle);
  }
};

extern "C" apxinf_status_t apxinf_conv2d_prepare(
    apxinf_runtime_t runtime, const apxinf_conv2d_spec_t* spec,
    const apxinf_conv2d_bindings_t* bindings,
    apxinf_conv2d_execution_t* execution) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime || !spec || !bindings || !execution) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Conv2d prepare argument");
    }
    *execution = nullptr;
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    cudaStreamCaptureStatus capture_status{};
    apxinf::framework::check_cuda(cudaStreamIsCapturing(
        static_cast<cudaStream_t>(bindings->stream), &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "Conv2d prepare is not capture-safe");
    }
    auto result = std::make_unique<apxinf_conv2d_execution>();
    result->spec = *spec;
    result->bindings = *bindings;
    result->device = runtime->device;
    // Build the fast path best-effort: any failure (allocation, descriptor
    // creation, heuristic) silently degrades the execution to the naive
    // kernel instead of failing prepare.
    if (fast_geometry_eligible(*spec)) {
      cublasLtHandle_t handle = nullptr;
      if (cublasLtCreate(&handle) == CUBLAS_STATUS_SUCCESS) {
        result->lt_handle = handle;
        if (try_build_lt_plan(*spec, *bindings, handle, &result->lt_plan)) {
          result->fast = true;
          result->summary = "cublaslt.im2col.v1";
        }
      }
    }
    *execution = result.release();
  });
}

extern "C" apxinf_status_t apxinf_conv2d_enqueue(apxinf_conv2d_execution_t execution) {
  return apxinf::framework::abi_boundary([&] {
    if (!execution) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Conv2d execution");
    apxinf::framework::check_cuda(cudaSetDevice(execution->device));
    if (execution->fast) {
      run_lt_conv(execution->lt_plan, execution->lt_handle, execution->spec,
                  execution->bindings);
      return;
    }
    apxinf::framework::check_cuda(launch(execution->spec, execution->bindings));
  });
}

extern "C" void apxinf_conv2d_destroy(apxinf_conv2d_execution_t execution) {
  delete execution;
}
extern "C" const char* apxinf_conv2d_summary(apxinf_conv2d_execution_t execution) {
  return execution ? execution->summary.c_str() : "null";
}

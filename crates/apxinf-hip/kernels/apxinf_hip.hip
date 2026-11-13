// Device code and C ABI for apxinf-hip.
//
// Everything Rust calls goes through the `extern "C"` functions in this file,
// so the Rust side never depends on HIP struct layouts, C++ templates, or
// header macros. Every entry point returns an int status: 0 on success, a
// `hipError_t` value for runtime failures, `kBlasStatusBase + hipblasStatus_t`
// for hipBLAS failures, and `kInvalidArgument` for arguments this shim rejects.
//
// Numerics follow apxinf-cuda's kernels, which are the reference for this
// backend: BF16 and F32 storage, F32 arithmetic, one rounding at the final
// store. The BF16 conversion and the wave-size rules live in `common.h`.
//
// The operators `apxinf_hip::native` composes into pi05 live in the sibling
// translation units (`norm.hip`, `elementwise.hip`, `native_attention.hip`);
// this file keeps the runtime, hipBLAS and the `Backend` operators.

#include <hipblas/hipblas.h>

#if APXINF_HIP_HAS_BLASLT
#include <hipblaslt/hipblaslt.h>
#endif

#include <map>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <tuple>

#include "common.h"

namespace {

using namespace apxinf_hip;

// ── Elementwise ─────────────────────────────────────────────────────

template <typename T>
__global__ void silu_kernel(const T* x, T* y, int64_t n) {
  for (int64_t i = global_index(); i < n; i += grid_stride()) {
    const float v = Elem<T>::load(x, i);
    Elem<T>::store(y, i, v / (1.0f + expf(-v)));
  }
}

template <typename T>
__global__ void binary_kernel(const T* a, const T* b, T* y, int64_t n, int op) {
  for (int64_t i = global_index(); i < n; i += grid_stride()) {
    const float lhs = Elem<T>::load(a, i);
    const float rhs = Elem<T>::load(b, i);
    Elem<T>::store(y, i, op == 0 ? lhs + rhs : lhs * rhs);
  }
}

template <typename T>
__global__ void scale_kernel(const T* x, T* y, int64_t n, float factor) {
  for (int64_t i = global_index(); i < n; i += grid_stride()) {
    Elem<T>::store(y, i, Elem<T>::load(x, i) * factor);
  }
}

// ── RMSNorm ─────────────────────────────────────────────────────────

// One block per row. Matches apxinf-cuda: sum of squares and the scale are
// F32, the product is formed as (x * inverse_rms) * weight, and the result is
// rounded once. The normalized value is not rounded before the weight multiply.
template <typename T>
__global__ void rms_norm_kernel(const T* x, const T* weight, T* y, int64_t cols, float eps) {
  __shared__ float partial[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;

  float sum = 0.0f;
  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = Elem<T>::load(x, base + c);
    sum += v * v;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float inverse_rms = rsqrtf(partial[0] / static_cast<float>(cols) + eps);

  for (int64_t c = threadIdx.x; c < cols; c += blockDim.x) {
    const float v = Elem<T>::load(x, base + c);
    Elem<T>::store(y, base + c, v * inverse_rms * Elem<T>::load(weight, c));
  }
}

// Vectorized BF16 RMSNorm: uint4 loads with a contiguous 8-per-thread
// mapping (one slice = 256 threads x 8 = 2048 columns, S = ceil(cols/2048)),
// the row register-resident so the input is read once instead of twice and
// the loads are one instruction per 8 elements. The product expression
// (v * inverse_rms) * weight and the single rounding are unchanged; the
// reduction partitioning differs from the 256-wide tree, so inverse_rms can
// differ by 1 ulp (the norm.hip vec kernels' precedent). Gated on
// cols % 8 == 0, cols <= 8192 and 16-byte alignment; RmsNormLaunch routes
// everything else back to the templated scalar kernel.
template <int S>
__global__ void rms_norm_vec_kernel(const uint16_t* x, const uint16_t* weight, uint16_t* y,
                                    int64_t cols, float eps) {
  __shared__ float scratch[kReduceBlock];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * cols;
  float cache[S][8];
  float square_sum = 0.0f;
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int64_t v0 = static_cast<int64_t>(threadIdx.x) * 8 + s * 2048;
    if (v0 < cols) {
      const uint4 raw = *reinterpret_cast<const uint4*>(x + base + v0);
      const uint32_t w[4] = {raw.x, raw.y, raw.z, raw.w};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float v = bf16_to_f32(static_cast<uint16_t>(w[j >> 1] >> ((j & 1) * 16)));
        cache[s][j] = v;
        square_sum += v * v;
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) cache[s][j] = 0.0f;
    }
  }
  const float inverse_rms =
      rsqrtf(block_sum(square_sum, scratch) / static_cast<float>(cols) + eps);
#pragma unroll
  for (int s = 0; s < S; ++s) {
    const int64_t v0 = static_cast<int64_t>(threadIdx.x) * 8 + s * 2048;
    if (v0 < cols) {
      const uint4 raww = *reinterpret_cast<const uint4*>(weight + v0);
      const uint32_t ww[4] = {raww.x, raww.y, raww.z, raww.w};
      uint32_t r[4] = {0u, 0u, 0u, 0u};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float wv = bf16_to_f32(static_cast<uint16_t>(ww[j >> 1] >> ((j & 1) * 16)));
        r[j >> 1] |=
            static_cast<uint32_t>(f32_to_bf16(cache[s][j] * inverse_rms * wv)) << ((j & 1) * 16);
      }
      *reinterpret_cast<uint4*>(y + base + v0) = uint4{r[0], r[1], r[2], r[3]};
    }
  }
}

// ── RoPE (half-split) ───────────────────────────────────────────────

// Input [seq, heads, head_dim]. Pairs (p, p + head_dim/2), position
// pos_offset + s, frequency 1 / theta^(2p/head_dim) in F32 via powf — the
// `apply_batched` path apxinf-cuda's `Backend::rope` uses.
template <typename T>
__global__ void rope_half_kernel(const T* x, T* y, int seq, int heads, int head_dim, float theta,
                                 uint32_t pos_offset) {
  const int half = head_dim / 2;
  const int64_t total = static_cast<int64_t>(seq) * heads * half;
  for (int64_t idx = global_index(); idx < total; idx += grid_stride()) {
    const int p = static_cast<int>(idx % half);
    const int64_t sh = idx / half;
    const int h = static_cast<int>(sh % heads);
    const int s = static_cast<int>(sh / heads);

    const uint32_t pos = static_cast<uint32_t>(s) + pos_offset;
    const float freq = 1.0f / powf(theta, 2.0f * static_cast<float>(p) / static_cast<float>(head_dim));
    const float angle = static_cast<float>(pos) * freq;
    const float c = cosf(angle);
    const float sn = sinf(angle);

    const int64_t base = (static_cast<int64_t>(s) * heads + h) * head_dim;
    const float x0 = Elem<T>::load(x, base + p);
    const float x1 = Elem<T>::load(x, base + half + p);
    Elem<T>::store(y, base + p, x0 * c - x1 * sn);
    Elem<T>::store(y, base + half + p, x0 * sn + x1 * c);
  }
}

// ── Embedding ───────────────────────────────────────────────────────

// Row copy. Token ids were range-checked on the host before upload, so the
// kernel trusts them. `E` is the element's storage type; no conversion.
template <typename E>
__global__ void embedding_kernel(const E* table, const uint32_t* ids, E* y, int64_t n_ids,
                                 int64_t dim) {
  const int64_t total = n_ids * dim;
  for (int64_t idx = global_index(); idx < total; idx += grid_stride()) {
    const int64_t row = idx / dim;
    const int64_t col = idx % dim;
    y[idx] = table[static_cast<int64_t>(ids[row]) * dim + col];
  }
}

// ── Attention over the KV cache ─────────────────────────────────────

// One block per (query row i, query head h), single pass with an online
// softmax, so no score matrix is materialized and memory does not grow with
// kv_len. The cache is token-major: key j of KV head g starts at
// (j * n_kv_heads + g) * head_dim.
//
// Query row i sees keys [0, min(kv_len, kv_offset + i + 1)) — the causal rule
// apxinf-cuda applies through its kv_offset. Decode passes kv_offset = kv_len-1
// (all keys); prefill passes kv_len - q_len.
//
// Scores, probabilities and the PV accumulation all stay F32 and the output is
// rounded once. That is more precise than apxinf-cuda's materialized path,
// which rounds scores and probabilities to BF16; the difference is well inside
// BF16 output tolerance and is deliberate rather than an accident of porting.
//
// Dynamic shared memory: q and acc (head_dim floats each), then the tile's
// probabilities and a reduction buffer (blockDim floats each).
template <typename T>
__global__ void attention_kernel(const T* q, const T* k, const T* v, T* out, int n_heads,
                                 int n_kv_heads, int head_dim, int kv_len, int kv_offset,
                                 float scale) {
  extern __shared__ float smem[];
  float* q_s = smem;
  float* acc = q_s + head_dim;
  float* p_s = acc + head_dim;
  float* red = p_s + blockDim.x;

  const int i = blockIdx.x;
  const int h = blockIdx.y;
  const int tid = threadIdx.x;
  const int tile = blockDim.x;
  const int kv_head = h / (n_heads / n_kv_heads);
  const int64_t kv_stride = static_cast<int64_t>(n_kv_heads) * head_dim;
  const int64_t kv_head_base = static_cast<int64_t>(kv_head) * head_dim;
  const int64_t q_base = (static_cast<int64_t>(i) * n_heads + h) * head_dim;

  int visible = kv_offset + i + 1;
  if (visible > kv_len) visible = kv_len;

  for (int d = tid; d < head_dim; d += tile) {
    q_s[d] = Elem<T>::load(q, q_base + d);
    acc[d] = 0.0f;
  }
  __syncthreads();

  float running_max = -INFINITY;
  float running_sum = 0.0f;
  for (int t0 = 0; t0 < visible; t0 += tile) {
    const int j = t0 + tid;
    float score = -INFINITY;
    if (j < visible) {
      const int64_t key = static_cast<int64_t>(j) * kv_stride + kv_head_base;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) dot += q_s[d] * Elem<T>::load(k, key + d);
      score = dot * scale;
    }

    red[tid] = score;
    __syncthreads();
    for (int stride = tile / 2; stride > 0; stride >>= 1) {
      if (tid < stride) red[tid] = fmaxf(red[tid], red[tid + stride]);
      __syncthreads();
    }
    // At least one key in this tile is visible, so the new maximum is finite.
    const float tile_max = fmaxf(running_max, red[0]);
    __syncthreads();

    const float p = j < visible ? expf(score - tile_max) : 0.0f;
    p_s[tid] = p;
    red[tid] = p;
    __syncthreads();
    for (int stride = tile / 2; stride > 0; stride >>= 1) {
      if (tid < stride) red[tid] += red[tid + stride];
      __syncthreads();
    }
    // exp(-inf) is 0 on the first tile, which zeroes the empty accumulator.
    const float correction = expf(running_max - tile_max);
    running_sum = running_sum * correction + red[0];

    const int in_tile = min(tile, visible - t0);
    for (int d = tid; d < head_dim; d += tile) {
      float a = acc[d] * correction;
      for (int t = 0; t < in_tile; ++t) {
        a += p_s[t] * Elem<T>::load(v, static_cast<int64_t>(t0 + t) * kv_stride + kv_head_base + d);
      }
      acc[d] = a;
    }
    running_max = tile_max;
    __syncthreads();
  }

  const float inverse_sum = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
  for (int d = tid; d < head_dim; d += tile) {
    Elem<T>::store(out, q_base + d, acc[d] * inverse_sum);
  }
}

template <template <typename> class Launch, typename... Args>
int dispatch(int dtype, Args... args) {
  if (dtype == kDTypeF32) return Launch<float>::run(args...);
  if (dtype == kDTypeBF16) return Launch<uint16_t>::run(args...);
  return kInvalidArgument;
}

template <typename T>
struct SiluLaunch {
  static int run(const void* x, void* y, int64_t n, void* stream) {
    silu_kernel<T><<<grid_for(n, kElementwiseBlock), kElementwiseBlock, 0, as_stream(stream)>>>(
        static_cast<const T*>(x), static_cast<T*>(y), n);
    return last_launch_status();
  }
};

template <typename T>
struct BinaryLaunch {
  static int run(const void* a, const void* b, void* y, int64_t n, int op, void* stream) {
    binary_kernel<T><<<grid_for(n, kElementwiseBlock), kElementwiseBlock, 0, as_stream(stream)>>>(
        static_cast<const T*>(a), static_cast<const T*>(b), static_cast<T*>(y), n, op);
    return last_launch_status();
  }
};

template <typename T>
struct ScaleLaunch {
  static int run(const void* x, void* y, int64_t n, float factor, void* stream) {
    scale_kernel<T><<<grid_for(n, kElementwiseBlock), kElementwiseBlock, 0, as_stream(stream)>>>(
        static_cast<const T*>(x), static_cast<T*>(y), n, factor);
    return last_launch_status();
  }
};

template <typename T>
struct RmsNormLaunch {
  static int run(const void* x, const void* w, void* y, int64_t rows, int64_t cols, float eps,
                 void* stream) {
    // BF16 rows that fit the vec kernel's 4-slice bound take the vectorized
    // path; everything else (other dtypes, odd or wide columns, misaligned
    // tensors) keeps the templated scalar kernel.
    if constexpr (sizeof(T) == 2) {
      const bool aligned = [](std::initializer_list<const void*> ps) {
        for (const void* p : ps) {
          if (p == nullptr || (reinterpret_cast<uintptr_t>(p) & 0xf) != 0) return false;
        }
        return true;
      }({x, w, y});
      if (aligned && cols > 0 && cols <= 8192 && cols % 8 == 0) {
        const int slices = static_cast<int>((cols + 2047) / 2048);
        switch (slices) {
          case 1:
            rms_norm_vec_kernel<1><<<static_cast<unsigned>(rows), kReduceBlock, 0,
                                    as_stream(stream)>>>(static_cast<const uint16_t*>(x),
                                                         static_cast<const uint16_t*>(w),
                                                         static_cast<uint16_t*>(y), cols, eps);
            return last_launch_status();
          case 2:
            rms_norm_vec_kernel<2><<<static_cast<unsigned>(rows), kReduceBlock, 0,
                                    as_stream(stream)>>>(static_cast<const uint16_t*>(x),
                                                         static_cast<const uint16_t*>(w),
                                                         static_cast<uint16_t*>(y), cols, eps);
            return last_launch_status();
          case 3:
            rms_norm_vec_kernel<3><<<static_cast<unsigned>(rows), kReduceBlock, 0,
                                    as_stream(stream)>>>(static_cast<const uint16_t*>(x),
                                                         static_cast<const uint16_t*>(w),
                                                         static_cast<uint16_t*>(y), cols, eps);
            return last_launch_status();
          case 4:
            rms_norm_vec_kernel<4><<<static_cast<unsigned>(rows), kReduceBlock, 0,
                                    as_stream(stream)>>>(static_cast<const uint16_t*>(x),
                                                         static_cast<const uint16_t*>(w),
                                                         static_cast<uint16_t*>(y), cols, eps);
            return last_launch_status();
          default:
            break;
        }
      }
    }
    rms_norm_kernel<T><<<static_cast<unsigned>(rows), kReduceBlock, 0, as_stream(stream)>>>(
        static_cast<const T*>(x), static_cast<const T*>(w), static_cast<T*>(y), cols, eps);
    return last_launch_status();
  }
};

template <typename T>
struct RopeLaunch {
  static int run(const void* x, void* y, int seq, int heads, int head_dim, float theta,
                 uint32_t pos_offset, void* stream) {
    const int64_t total = static_cast<int64_t>(seq) * heads * (head_dim / 2);
    rope_half_kernel<T><<<grid_for(total, kElementwiseBlock), kElementwiseBlock, 0,
                          as_stream(stream)>>>(static_cast<const T*>(x), static_cast<T*>(y), seq,
                                               heads, head_dim, theta, pos_offset);
    return last_launch_status();
  }
};

template <typename T>
struct AttentionLaunch {
  static int run(const void* q, const void* k, const void* v, void* out, int q_len, int n_heads,
                 int n_kv_heads, int head_dim, int kv_len, int kv_offset, float scale,
                 void* stream) {
    const size_t shared =
        (2 * static_cast<size_t>(head_dim) + 2 * static_cast<size_t>(kAttentionBlock)) * sizeof(float);
    attention_kernel<T><<<dim3(q_len, n_heads), kAttentionBlock, shared, as_stream(stream)>>>(
        static_cast<const T*>(q), static_cast<const T*>(k), static_cast<const T*>(v),
        static_cast<T*>(out), n_heads, n_kv_heads, head_dim, kv_len, kv_offset, scale);
    return last_launch_status();
  }
};

}  // namespace

// ── Runtime ─────────────────────────────────────────────────────────

extern "C" int apxinf_hip_device_count(int* count) {
  return static_cast<int>(hipGetDeviceCount(count));
}

extern "C" int apxinf_hip_set_device(int device) {
  return static_cast<int>(hipSetDevice(device));
}

extern "C" int apxinf_hip_device_info(int device, char* arch, int arch_len, char* name,
                                      int name_len, int* warp_size, int* compute_units,
                                      uint64_t* total_memory, int* memory_pools) {
  hipDeviceProp_t prop;
  const hipError_t status = hipGetDeviceProperties(&prop, device);
  if (status != hipSuccess) return static_cast<int>(status);
  snprintf(arch, static_cast<size_t>(arch_len), "%s", prop.gcnArchName);
  snprintf(name, static_cast<size_t>(name_len), "%s", prop.name);
  *warp_size = prop.warpSize;
  *compute_units = prop.multiProcessorCount;
  *total_memory = static_cast<uint64_t>(prop.totalGlobalMem);
  *memory_pools = prop.memoryPoolsSupported;
  return 0;
}

extern "C" int apxinf_hip_stream_create(void** stream) {
  hipStream_t raw = nullptr;
  const hipError_t status = hipStreamCreate(&raw);
  *stream = raw;
  return static_cast<int>(status);
}

extern "C" int apxinf_hip_stream_destroy(void* stream) {
  return static_cast<int>(hipStreamDestroy(as_stream(stream)));
}

extern "C" int apxinf_hip_stream_synchronize(void* stream) {
  return static_cast<int>(hipStreamSynchronize(as_stream(stream)));
}

// Non-destructive stream probe: hipStreamQuery returns hipSuccess when the
// stream is idle and healthy, hipErrorNotReady while work is pending, and the
// first latched error when an earlier operation failed. Unlike the calls
// around it, this reports the error instead of clearing it, so a capture
// fallback can tell a recoverable failure from a poisoned stream.
extern "C" int apxinf_hip_stream_query(void* stream) {
  return static_cast<int>(hipStreamQuery(as_stream(stream)));
}

// Stream-ordered allocation when the device supports memory pools: a free is
// ordered after all work already queued on the stream, so dropping a tensor
// cannot release memory a pending kernel still reads. Without pools, hipFree
// synchronizes the device instead — slower, equally safe.
extern "C" int apxinf_hip_malloc(void** ptr, uint64_t bytes, int stream_ordered, void* stream) {
  *ptr = nullptr;
  if (bytes == 0) return 0;
  return static_cast<int>(stream_ordered ? hipMallocAsync(ptr, bytes, as_stream(stream))
                                         : hipMalloc(ptr, bytes));
}

extern "C" int apxinf_hip_free(void* ptr, int stream_ordered, void* stream) {
  if (ptr == nullptr) return 0;
  return static_cast<int>(stream_ordered ? hipFreeAsync(ptr, as_stream(stream)) : hipFree(ptr));
}

// Host-to-device copies synchronize before returning: the source is pageable
// host memory owned by the caller, and HIP does not promise an asynchronous
// copy from pageable memory has finished reading it when the call returns.
extern "C" int apxinf_hip_memcpy_htod(void* dst, const void* src, uint64_t bytes, void* stream) {
  if (bytes == 0) return 0;
  hipError_t status = hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, as_stream(stream));
  if (status != hipSuccess) return static_cast<int>(status);
  return static_cast<int>(hipStreamSynchronize(as_stream(stream)));
}

extern "C" int apxinf_hip_memcpy_dtoh(void* dst, const void* src, uint64_t bytes, void* stream) {
  if (bytes == 0) return 0;
  hipError_t status = hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToHost, as_stream(stream));
  if (status != hipSuccess) return static_cast<int>(status);
  return static_cast<int>(hipStreamSynchronize(as_stream(stream)));
}

extern "C" int apxinf_hip_memcpy_dtod(void* dst, const void* src, uint64_t bytes, void* stream) {
  if (bytes == 0) return 0;
  return static_cast<int>(
      hipMemcpyAsync(dst, src, bytes, hipMemcpyDeviceToDevice, as_stream(stream)));
}

extern "C" int apxinf_hip_error_string(int code, char* buf, int len) {
  const char* message;
  if (code == kNoCkInstance) {
    message = "no Composable Kernel attention instance matches these arguments";
  } else if (code == kCkException) {
    message = "Composable Kernel reported a launch failure (exception caught at the C boundary)";
  } else if (code >= kInvalidArgument) {
    message = "argument rejected by the apxinf-hip shim";
  } else if (code >= kBlasStatusBase) {
    message = hipblasStatusToString(static_cast<hipblasStatus_t>(code - kBlasStatusBase));
  } else {
    message = hipGetErrorString(static_cast<hipError_t>(code));
  }
  snprintf(buf, static_cast<size_t>(len), "%s", message);
  return 0;
}

// ── hipBLAS ─────────────────────────────────────────────────────────

extern "C" int apxinf_hip_blas_create(void** handle, void* stream) {
  hipblasHandle_t raw = nullptr;
  hipblasStatus_t status = hipblasCreate(&raw);
  *handle = raw;
  if (status != HIPBLAS_STATUS_SUCCESS) return kBlasStatusBase + static_cast<int>(status);
  status = hipblasSetStream(raw, as_stream(stream));
  return status == HIPBLAS_STATUS_SUCCESS ? 0 : kBlasStatusBase + static_cast<int>(status);
}

extern "C" int apxinf_hip_blas_destroy(void* handle) {
  const hipblasStatus_t status = hipblasDestroy(static_cast<hipblasHandle_t>(handle));
  return status == HIPBLAS_STATUS_SUCCESS ? 0 : kBlasStatusBase + static_cast<int>(status);
}

// Row-major C[m,n] = A[m,k] @ B[k,n] with F32 accumulation; C has A's dtype.
// hipBLAS is column-major, and a row-major matrix is its column-major
// transpose, so this computes C^T[n,m] = B^T[n,k] @ A^T[k,m]: B goes first and
// the m/n extents swap. No data is transposed.
extern "C" int apxinf_hip_gemm(void* handle, int dtype, int m, int n, int k, const void* a,
                               const void* b, void* c) {
  hipDataType type;
  if (dtype == kDTypeF32) {
    type = HIP_R_32F;
  } else if (dtype == kDTypeBF16) {
    type = HIP_R_16BF;
  } else {
    return kInvalidArgument;
  }
  const float alpha = 1.0f;
  const float beta = 0.0f;
  const hipblasStatus_t status =
      hipblasGemmEx(static_cast<hipblasHandle_t>(handle), HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k,
                    &alpha, b, type, n, a, type, k, &beta, c, type, n, HIPBLAS_COMPUTE_32F,
                    HIPBLAS_GEMM_DEFAULT);
  return status == HIPBLAS_STATUS_SUCCESS ? 0 : kBlasStatusBase + static_cast<int>(status);
}

// ── hipBLASLt ───────────────────────────────────────────────────────
//
// A second GEMM provider, selected per shape by `apxinf_hip::native::gemm`.
// Neither library wins everywhere on gfx1151: measured over GR00T's 26 GEMM
// shapes, hipBLASLt is 1.05x-3.71x faster on the 13 shapes below the roofline
// ridge point and 0.58x-0.90x *slower* on the 5 above it, so the choice has to
// be made per shape rather than globally. See
// `devlocal/gr00t-hip-port/reports/perf/gemm-provider-comparison.md`.
//
// Three things that measurement established and this code depends on:
//
//  1. A chosen algorithm cannot be persisted as a solution index. Recovering
//     one through `hipblaslt_ext::getAlgosFromIndex` returns the right kernel
//     by name but not its speed (7x slower on one shape). The serialized
//     `hipblasLtMatmulAlgo_t` -- 24 opaque bytes, documented in hipblaslt.h as
//     trivially serializable within one library version -- does round-trip.
//     Nothing here uses `hipblaslt-ext.hpp`: the C++ introspection API is not
//     needed for this, and its `getKernelNameFromAlgo` crashed the process in
//     2 of ~17 probe runs for reasons never established.
//
//  2. A deserialized algorithm must not be used before
//     `hipblasLtMatmulAlgoGetHeuristic` has run for that problem. The solution
//     map is populated lazily by that call, and a restored algorithm is only
//     an index into it; skipping the query segfaults inside rocBLASLt
//     ("Could not initialize Tensile host"). So `problem_for` always queries,
//     then overrides with the recorded tactic.
//
//  3. No workspace is needed. The fastest tactic for every one of those 26
//     shapes wants zero bytes, and offering none costs 0.00 ms, so the
//     preference is 0 and `hipblasLtMatmul` gets a null workspace. Tactics
//     must be *recorded* under the same preference, or the heuristic offers a
//     different candidate set.

#if APXINF_HIP_HAS_BLASLT

namespace {

// `(dtype, m, n, k, epilogue)`.
//
// The epilogue belongs in the key, not alongside it. It is set on the
// descriptor before the heuristic runs, so a problem with a fused bias and
// activation gets a different candidate set and a different winner than the
// same shape without one -- measured on GR00T, `41x1536x6144` is 1.04x slower
// on hipBLASLt as a plain GEMM and 1.09x *faster* with the bias and GELU
// folded in, because the separate kernel it replaces costs more than the
// slower tile does.
using LtKey = std::tuple<int, int, int, int, int>;

// Mirrors `apxinf_hip::native::Activation` for the subset hipBLASLt can fuse.
// Values are the C ABI's, not hipBLASLt's.
constexpr int kEpilogueNone = 0;
constexpr int kEpilogueBias = 1;
constexpr int kEpilogueBiasGeluTanh = 2;
constexpr int kEpilogueBiasRelu = 3;

// Descriptors for one problem, built once and kept for the handle's lifetime.
// Creating them per call costs more than the kernel saves: the dominant DiT
// shapes run in 26-100 microseconds.
struct LtProblem {
  hipblasLtMatmulDesc_t desc = nullptr;
  hipblasLtMatrixLayout_t b = nullptr, a = nullptr, c = nullptr;
  hipblasLtMatmulAlgo_t algo{};
  bool usable = false;
  // Set when a recipe names a tactic for this shape and it has not yet been
  // proven to run; cleared by `promote_tactic` either way.
  const hipblasLtMatmulAlgo_t* pending = nullptr;
  // Whether the descriptor holds a bias pointer that must be refreshed when
  // the caller passes a different one.
  bool has_bias = false;

  ~LtProblem() {
    if (c) hipblasLtMatrixLayoutDestroy(c);
    if (a) hipblasLtMatrixLayoutDestroy(a);
    if (b) hipblasLtMatrixLayoutDestroy(b);
    if (desc) hipblasLtMatmulDescDestroy(desc);
  }
  LtProblem() = default;
  LtProblem(const LtProblem&) = delete;
  LtProblem& operator=(const LtProblem&) = delete;
};

// The handle owns its caches, so nothing here is global: two contexts on two
// devices keep separate descriptors and separate tactic tables.
struct LtContext {
  hipblasLtHandle_t handle = nullptr;
  hipStream_t stream = nullptr;
  std::map<LtKey, LtProblem> problems;
  std::map<LtKey, hipblasLtMatmulAlgo_t> tactics;
};

hipDataType lt_dtype(int dtype) {
  return dtype == kDTypeBF16 ? HIP_R_16BF : HIP_R_32F;
}

// Build and cache the descriptors for one problem, choosing its algorithm.
//
// Layouts follow `apxinf_hip_gemm` exactly: hipBLASLt is column-major, a
// row-major matrix is its column-major transpose, so this states
// `C^T[n,m] = B^T[n,k] @ A^T[k,m]` and nothing is transposed in memory.
LtProblem* problem_for(LtContext* lt, int dtype, int m, int n, int k, int epilogue,
                       const void* bias) {
  auto key = LtKey{dtype, m, n, k, epilogue};
  auto found = lt->problems.find(key);
  if (found != lt->problems.end()) return &found->second;

  LtProblem& problem = lt->problems[key];
  const hipDataType type = lt_dtype(dtype);
  hipblasOperation_t op = HIPBLAS_OP_N;
  if (hipblasLtMatmulDescCreate(&problem.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F) !=
          HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(problem.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &op,
                                      sizeof(op)) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatmulDescSetAttribute(problem.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &op,
                                      sizeof(op)) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&problem.b, type, n, k, n) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&problem.a, type, k, m, k) != HIPBLAS_STATUS_SUCCESS ||
      hipblasLtMatrixLayoutCreate(&problem.c, type, n, m, n) != HIPBLAS_STATUS_SUCCESS) {
    // Transient, like a failed heuristic: do not cache the failure for the
    // handle's lifetime.
    lt->problems.erase(key);
    return nullptr;  // not usable; the caller falls back to hipBLAS
  }

  // The bias is one value per output feature. Row-major `C[m, n]` is
  // column-major `[n, m]`, so the output feature is the row index of the
  // column-major D -- which is exactly what HIPBLASLT_EPILOGUE_BIAS wants
  // ("length must match matrix D rows").
  //
  // hipBLASLt applies the bias and the activation to the F32 accumulator and
  // rounds once. The unfused path rounds the GEMM output to BF16, reads it
  // back, adds the bias and rounds again -- two extra roundings per site that
  // fusing removes, which moves this closer to the PyTorch reference rather
  // than away from it.
  if (epilogue != kEpilogueNone) {
    hipblasLtEpilogue_t mode = HIPBLASLT_EPILOGUE_BIAS;
    if (epilogue == kEpilogueBiasGeluTanh) mode = HIPBLASLT_EPILOGUE_GELU_BIAS;
    else if (epilogue == kEpilogueBiasRelu) mode = HIPBLASLT_EPILOGUE_RELU_BIAS;
    hipDataType bias_type = type;
    if (hipblasLtMatmulDescSetAttribute(problem.desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &mode,
                                        sizeof(mode)) != HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatmulDescSetAttribute(problem.desc, HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                        &bias_type, sizeof(bias_type)) !=
            HIPBLAS_STATUS_SUCCESS ||
        hipblasLtMatmulDescSetAttribute(problem.desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                        &bias, sizeof(bias)) != HIPBLAS_STATUS_SUCCESS) {
      lt->problems.erase(key);
      return nullptr;
    }
    problem.has_bias = true;
  }

  hipblasLtMatmulPreference_t preference = nullptr;
  if (hipblasLtMatmulPreferenceCreate(&preference) != HIPBLAS_STATUS_SUCCESS) {
    lt->problems.erase(key);
    return nullptr;
  }
  const uint64_t workspace = 0;
  hipblasLtMatmulPreferenceSetAttribute(preference,
                                        HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                        &workspace, sizeof(workspace));

  // Always queried, even when a tactic is recorded: see note 2 above.
  hipblasLtMatmulHeuristicResult_t heuristics[8];
  int found_count = 0;
  const hipblasStatus_t status = hipblasLtMatmulAlgoGetHeuristic(
      lt->handle, problem.desc, problem.b, problem.a, problem.c, problem.c, preference, 8,
      heuristics, &found_count);
  hipblasLtMatmulPreferenceDestroy(preference);
  // A non-success status or an empty list is a transient failure, not a
  // property of the problem: Tensile's host side initializes lazily and its
  // first query can fail. Caching `usable = false` here would pin the
  // fallback for the handle's lifetime, so erase the entry and let the next
  // call build it again.
  if (status != HIPBLAS_STATUS_SUCCESS || found_count == 0) {
    lt->problems.erase(key);
    return nullptr;
  }
  // `found_count` entries were written, but each one carries its own state:
  // a heuristic can report candidates whose introspection failed. Only a
  // candidate with `state == SUCCESS` has a valid `algo`; anything else is
  // not a usable algorithm no matter what the count said.
  int chosen = -1;
  for (int i = 0; i < found_count; ++i) {
    if (heuristics[i].state == HIPBLAS_STATUS_SUCCESS) {
      chosen = i;
      break;
    }
  }
  if (chosen < 0) {
    lt->problems.erase(key);
    return nullptr;
  }

  problem.algo = heuristics[chosen].algo;
  problem.usable = true;

  auto tactic = lt->tactics.find(key);
  if (tactic != lt->tactics.end()) problem.pending = &tactic->second;
  return &problem;
}

// Promote a recorded tactic, once, after proving it runs.
//
// Validation is a trial launch on the caller's own buffers rather than
// `hipblaslt_ext::matmulIsAlgoSupported`: that function mutates the
// algorithm's `max_workspace_bytes` to a value this code does not supply, and
// it lives in the C++ header this file deliberately avoids. The trial writes
// `c`, which the real launch immediately overwrites with the same product.
//
// A recipe naming a solution the current library does not have is the case
// this guards: it must degrade to the heuristic's pick, not to a wrong answer
// or a crash.
//
// Under graph capture there is no trial: the synchronize is illegal inside a
// capture, and a launch that failed would invalidate it. The tactic stays
// unpromoted and this call runs the heuristic's pick, captured like any other
// launch -- the graph then replays the heuristic's kernel for this shape
// until a capture-free process promotes the tactic. Slower, never wrong.
void promote_tactic(LtContext* lt, LtProblem* problem, const void* a, const void* b,
                    void* c) {
  const hipblasLtMatmulAlgo_t candidate = *problem->pending;
  problem->pending = nullptr;
  hipStreamCaptureStatus capture = hipStreamCaptureStatusNone;
  unsigned long long capture_id = 0;
  if (hipStreamGetCaptureInfo(lt->stream, &capture, &capture_id) != hipSuccess) return;
  if (capture != hipStreamCaptureStatusNone) return;
  const float alpha = 1.0f, beta = 0.0f;
  hipblasLtMatmulAlgo_t trial = candidate;
  if (hipblasLtMatmul(lt->handle, problem->desc, &alpha, b, problem->b, a, problem->a,
                      &beta, c, problem->c, c, problem->c, &trial, nullptr, 0,
                      lt->stream) != HIPBLAS_STATUS_SUCCESS) {
    return;
  }
  if (hipStreamSynchronize(lt->stream) != hipSuccess) return;
  problem->algo = candidate;
}

}  // namespace

extern "C" int apxinf_hip_blaslt_create(void** handle, void* stream) {
  *handle = nullptr;
  LtContext* lt = new LtContext();
  const hipblasStatus_t status = hipblasLtCreate(&lt->handle);
  if (status != HIPBLAS_STATUS_SUCCESS) {
    delete lt;
    return kBlasStatusBase + static_cast<int>(status);
  }
  lt->stream = as_stream(stream);
  *handle = lt;
  return 0;
}

extern "C" int apxinf_hip_blaslt_destroy(void* handle) {
  if (!handle) return 0;
  LtContext* lt = static_cast<LtContext*>(handle);
  lt->problems.clear();  // descriptors before the handle they belong to
  const hipblasStatus_t status = hipblasLtDestroy(lt->handle);
  delete lt;
  return status == HIPBLAS_STATUS_SUCCESS ? 0 : kBlasStatusBase + static_cast<int>(status);
}

// Record a measured tactic for one shape. `blob` is a serialized
// `hipblasLtMatmulAlgo_t`; `len` must match its size, which the caller reads
// from `apxinf_hip_blaslt_tactic_bytes` so the two sides cannot disagree.
extern "C" int apxinf_hip_blaslt_set_tactic(void* handle, int dtype, int m, int n, int k,
                                            int epilogue, const unsigned char* blob,
                                            int len) {
  if (!handle || !blob || len != static_cast<int>(sizeof(hipblasLtMatmulAlgo_t))) {
    return kInvalidArgument;
  }
  LtContext* lt = static_cast<LtContext*>(handle);
  hipblasLtMatmulAlgo_t algo{};
  memcpy(&algo, blob, sizeof(algo));
  lt->tactics[LtKey{dtype, m, n, k, epilogue}] = algo;
  // Drop any problem already built for this key so the tactic takes effect.
  lt->problems.erase(LtKey{dtype, m, n, k, epilogue});
  return 0;
}

extern "C" int apxinf_hip_blaslt_tactic_bytes() {
  return static_cast<int>(sizeof(hipblasLtMatmulAlgo_t));
}

// The two versions a recorded tactic is only valid under. hipBLASLt's is the
// one that matters -- its header states a recorded solution cannot be used
// across library versions -- and the ROCm runtime version goes alongside it so
// the directory name matches the CUDA side's `cuda<..>-cublas<..>` convention.
extern "C" int apxinf_hip_library_versions(int* rocm, int* blaslt) {
  *rocm = 0;
  *blaslt = 0;
  const hipError_t status = hipRuntimeGetVersion(rocm);
  if (status != hipSuccess) return static_cast<int>(status);
  // A real handle is required: `hipblasLtGetVersion(nullptr, ...)` returns
  // HIPBLAS_STATUS_NOT_INITIALIZED. This runs a handful of times per process,
  // so creating one here is cheaper than threading a handle through to every
  // caller that only wants to name a directory.
  hipblasLtHandle_t handle = nullptr;
  if (hipblasLtCreate(&handle) != HIPBLAS_STATUS_SUCCESS) return kInvalidArgument;
  const hipblasStatus_t version_status = hipblasLtGetVersion(handle, blaslt);
  hipblasLtDestroy(handle);
  return version_status == HIPBLAS_STATUS_SUCCESS ? 0 : kInvalidArgument;
}

// Row-major C[m,n] = A[m,k] @ B[k,n], as `apxinf_hip_gemm` but through
// hipBLASLt. Returns `kInvalidArgument` when the shape has no usable algorithm,
// which the Rust side treats as "fall back to hipBLAS" rather than an error.
extern "C" int apxinf_hip_gemm_lt(void* handle, int dtype, int m, int n, int k,
                                  const void* a, const void* b, void* c, int epilogue,
                                  const void* bias) {
  if (!handle) return kInvalidArgument;
  if (dtype != kDTypeF32 && dtype != kDTypeBF16) return kInvalidArgument;
  if (epilogue != kEpilogueNone && !bias) return kInvalidArgument;
  LtContext* lt = static_cast<LtContext*>(handle);
  LtProblem* problem = problem_for(lt, dtype, m, n, k, epilogue, bias);
  if (!problem || !problem->usable) return kInvalidArgument;
  // The descriptor caches the bias *pointer*, but one cached problem serves
  // every site with this shape -- all 32 DiT blocks share `41x1536x6144` and
  // each has its own bias. Refreshing it per call is a write into the
  // descriptor; the heuristic depends on the layouts and the epilogue mode,
  // not on where the bias lives, so the cached algorithm stays valid.
  if (problem->has_bias &&
      hipblasLtMatmulDescSetAttribute(problem->desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                      &bias, sizeof(bias)) != HIPBLAS_STATUS_SUCCESS) {
    return kInvalidArgument;
  }
  if (problem->pending) promote_tactic(lt, problem, a, b, c);
  const float alpha = 1.0f, beta = 0.0f;
  const hipblasStatus_t status =
      hipblasLtMatmul(lt->handle, problem->desc, &alpha, b, problem->b, a, problem->a,
                      &beta, c, problem->c, c, problem->c, &problem->algo, nullptr, 0,
                      lt->stream);
  return status == HIPBLAS_STATUS_SUCCESS ? 0 : kBlasStatusBase + static_cast<int>(status);
}

#else  // APXINF_HIP_HAS_BLASLT

// ROCm without hipBLASLt headers: the provider is simply unavailable, and
// `native::gemm` keeps using hipBLAS. Every stub keeps the full real
// signature -- argument count included -- so the Rust declarations in
// `ffi.rs` are checked against these too when a header-less build links.
extern "C" int apxinf_hip_blaslt_create(void** handle, void*) {
  *handle = nullptr;
  return kInvalidArgument;
}
extern "C" int apxinf_hip_blaslt_destroy(void*) { return 0; }
extern "C" int apxinf_hip_blaslt_set_tactic(void*, int, int, int, int, int,
                                            const unsigned char*, int) {
  return kInvalidArgument;
}
extern "C" int apxinf_hip_blaslt_tactic_bytes() { return 0; }
extern "C" int apxinf_hip_library_versions(int* rocm, int* blaslt) {
  *rocm = 0;
  *blaslt = 0;
  return kInvalidArgument;
}
extern "C" int apxinf_hip_gemm_lt(void*, int, int, int, int, const void*, const void*,
                                  void*, int, const void*) {
  return kInvalidArgument;
}

#endif  // APXINF_HIP_HAS_BLASLT

// ── HIP graphs ──────────────────────────────────────────────────────
//
// Capture the model's forward once and replay it, so the host issues one
// launch instead of 2131.
//
// What this buys, measured: the GPU is busy 79.4 ms of an 81.2 ms inference,
// so the gap a graph can close is the 1.8 ms of host-side launch time that is
// not hidden behind kernels. It does not shorten any kernel -- a 126 KB
// adaptive-layer-norm that takes 5.6 us on 41 rows is bound by wave dispatch
// across 20 CUs, and replaying it from a graph dispatches the same waves.
//
// What it requires of the captured region, and why the forward was changed to
// satisfy it: no device allocation (illegal during capture) and no copy from
// host memory that is rebuilt per call (the graph records the pointer, and
// replay would read freed memory). The DiT's cross-attention row indices used
// to be allocated and uploaded inside the forward; they are plan-invariant and
// now live in the prepared plan.
//
// Capture is thread-local to the stream and must see the exact sequence the
// replay will perform, which is what the plan cache already guarantees: the
// arena resets to the same base and the same `arena.tensor` calls follow, so
// every address the graph records is the address the replay uses.

extern "C" int apxinf_hip_graph_capture_begin(void* stream) {
  // `Relaxed` rather than `Global`: this stream is the only one issuing work,
  // and the relaxed mode permits queries that the global mode rejects.
  return static_cast<int>(
      hipStreamBeginCapture(as_stream(stream), hipStreamCaptureModeRelaxed));
}

// Ends capture and instantiates. On success `*exec` owns the executable graph
// and the caller must pass it to `apxinf_hip_graph_destroy`; on failure
// `*exec` is null and capture has been terminated either way.
//
// `*nodes` is the captured node count, read from the template graph while it
// still exists. It is the only way to tell a graph that captured the whole
// forward from one that captured nothing: an empty capture instantiates
// successfully and replays instantly, producing whatever was already in the
// output buffer.
extern "C" int apxinf_hip_graph_capture_end(void* stream, void** exec, int* nodes) {
  *exec = nullptr;
  *nodes = 0;
  hipGraph_t graph = nullptr;
  const hipError_t ended = hipStreamEndCapture(as_stream(stream), &graph);
  if (ended != hipSuccess) return static_cast<int>(ended);
  if (graph == nullptr) return kInvalidArgument;
  size_t count = 0;
  if (hipGraphGetNodes(graph, nullptr, &count) == hipSuccess) {
    *nodes = static_cast<int>(count);
  }
  hipGraphExec_t instantiated = nullptr;
  const hipError_t made = hipGraphInstantiate(&instantiated, graph, nullptr, nullptr, 0);
  // The template graph is not needed once instantiated; the executable holds
  // what it needs.
  hipGraphDestroy(graph);
  if (made != hipSuccess) return static_cast<int>(made);
  *exec = instantiated;
  return 0;
}

extern "C" int apxinf_hip_graph_launch(void* exec, void* stream) {
  if (!exec) return kInvalidArgument;
  return static_cast<int>(
      hipGraphLaunch(static_cast<hipGraphExec_t>(exec), as_stream(stream)));
}

extern "C" int apxinf_hip_graph_destroy(void* exec) {
  if (!exec) return 0;
  return static_cast<int>(hipGraphExecDestroy(static_cast<hipGraphExec_t>(exec)));
}

// ── CK attention stubs ──────────────────────────────────────────────
//
// `ck_attention.hip` defines these when the vendored Composable Kernel is
// present. Without it the symbols still have to exist, or the crate fails to
// link rather than falling back: `native::attention` is written to continue on
// apxinf's own kernel whenever CK reports that it cannot serve a call, and
// "not built in" is one of those cases.

#if !APXINF_HIP_HAS_CK

extern "C" int apxinf_hip_ck_attention_supported(int, int, int, int, int) { return 0; }
extern "C" int apxinf_hip_ck_attention(int, const void*, const void*, const void*, void*,
                                       int, int, int, int, int, int, long long, long long,
                                       long long, float, int, void*) {
  return kInvalidArgument;
}

#endif  // !APXINF_HIP_HAS_CK

// ── hipBLASLt autotune ──────────────────────────────────────────────
//
// Measure both providers for one shape and return the winner, so a recipe can
// be produced on the deploy machine instead of carried in the repository.
//
// The measurement is cache-cold, which is the whole difficulty. Timing one
// weight matrix back to back leaves it resident in the 32 MB MALL, and that
// is not the state a real inference sees: consecutive GEMMs in the model read
// different weights -- the DiT alone streams 576 MB per flow step -- so every
// one of them starts cold. The difference is not academic. Comparing this
// project's first (accidentally cache-warm) measurements against the
// cache-cold ones, 3 of 12 shapes change which provider wins:
//
//     41x1536x6144   warm 2.05x -> cold 1.05x   (warm would route it to Lt)
//     149x2048x2048  warm 0.87x -> cold 1.17x   (warm would leave Lt's gain)
//     149x6144x2048  warm 0.95x -> cold 1.24x   (same)
//
// So the tuner streams a ring of weight copies past the cache, exactly as the
// offline probe does, and the acceptance check for this code is that it
// reproduces the probe's routing.
//
// Inputs are read, never written: the tuner allocates its own output and its
// own ring, so a tuning pass cannot corrupt the request that triggered it.

#if APXINF_HIP_HAS_BLASLT

namespace {

// Spanning this much distinct weight memory between revisits evicts a buffer
// from a 32 MB MALL with margin. The buffer count is capped so a small weight
// matrix does not ask for thousands of allocations; at the cap the span is
// still about twice the cache.
constexpr size_t kTuneSpanBytes = 256u << 20;
constexpr int kTuneMaxRing = 256;
// Per candidate. Long enough to average out launch jitter, short enough that
// tuning GR00T's 26 shapes (31 keys with the epilogue variants) stays within
// a few seconds of load time.
constexpr int kTuneIterations = 40;
// A candidate whose answer drifts from hipBLAS by more than this is not a
// faster way to do the work, it is a different computation. Both accumulate
// in F32 and round once to BF16, so honest tile-order differences land near
// 1e-4; this is two orders of magnitude above that and far below a transposed
// or partially written result.
constexpr double kTuneMaxRelativeL2 = 2e-2;
// Fused candidates carry one further honest difference: the epilogue applies
// the bias and the activation to the F32 accumulator and rounds once, while
// the reference above rounds the GEMM, adds the bias and rounds again. That
// is a BF16 ulp, well inside the threshold, and in the direction of the
// higher-precision answer.

// sum((x-y)^2) and sum(y^2) in one pass, for the relative L2 above.
__global__ void squared_error_kernel(const uint16_t* x, const uint16_t* y, int64_t n,
                                     float* sums) {
  __shared__ float scratch[kReduceBlock];
  float error = 0.0f, reference = 0.0f;
  for (int64_t i = global_index(); i < n; i += grid_stride()) {
    const float a = bf16_to_f32(x[i]), b = bf16_to_f32(y[i]);
    error += (a - b) * (a - b);
    reference += b * b;
  }
  // `block_sum` leaves a barrier behind it, so the scratch buffer is safe to
  // use twice.
  error = block_sum(error, scratch);
  reference = block_sum(reference, scratch);
  if (threadIdx.x == 0) {
    atomicAdd(&sums[0], error);
    atomicAdd(&sums[1], reference);
  }
}

// The arithmetic `native::bias_activation` performs, used by the tuner for
// two things: to complete hipBLAS's arm (which still needs this kernel when
// the candidate it is compared against fuses it away), and to bring the
// reference up to what a fused candidate produces.
//
// The rounding differs from the fused path on purpose -- this rounds the GEMM
// output, adds the bias and rounds again, exactly as the model does today,
// while the epilogue rounds once. That difference is below the validation
// threshold and is a move toward the PyTorch reference, not away from it.
__global__ void bias_activation_for_tuning(const uint16_t* x, const uint16_t* bias,
                                           uint16_t* out, int64_t n, int cols, int gelu,
                                           int relu) {
  for (int64_t i = global_index(); i < n; i += grid_stride()) {
    float v = bf16_to_f32(x[i]) + bf16_to_f32(bias[i % cols]);
    if (gelu) {
      v = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    } else if (relu) {
      v = v > 0.0f ? v : 0.0f;
    }
    out[i] = f32_to_bf16(v);
  }
}

double elapsed_ms(hipEvent_t start, hipEvent_t stop) {
  float ms = 0.0f;
  hipEventElapsedTime(&ms, start, stop);
  return ms;
}

}  // namespace

// Returns 0 on success with `blas_ms`, `lt_ms` (negative when hipBLASLt served
// nothing) and the winner's serialized algorithm. The caller decides the
// provider; this only measures.
extern "C" int apxinf_hip_gemm_tune(void* lt_handle, void* blas_handle, int dtype, int m,
                                    int n, int k, const void* a, const void* b,
                                    int epilogue, const void* bias, double* blas_ms,
                                    double* lt_ms, unsigned char* blob, int blob_len,
                                    double* worst_relative_l2) {
  if (!lt_handle || !blas_handle || dtype != kDTypeBF16) return kInvalidArgument;
  if (!blob || blob_len != static_cast<int>(sizeof(hipblasLtMatmulAlgo_t))) {
    return kInvalidArgument;
  }
  if (epilogue != kEpilogueNone && !bias) return kInvalidArgument;
  LtContext* lt = static_cast<LtContext*>(lt_handle);
  LtProblem* problem = problem_for(lt, dtype, m, n, k, epilogue, bias);
  if (!problem || !problem->usable) return kInvalidArgument;

  *blas_ms = -1.0;
  *lt_ms = -1.0;
  *worst_relative_l2 = 0.0;

  const size_t weight_bytes = (size_t)k * n * 2;
  const size_t out_bytes = (size_t)m * n * 2;
  int ring = (int)((kTuneSpanBytes + weight_bytes - 1) / weight_bytes);
  ring = ring < 2 ? 2 : (ring > kTuneMaxRing ? kTuneMaxRing : ring);

  std::vector<void*> weights;
  void *out = nullptr, *reference = nullptr;
  float* sums = nullptr;
  int status = 0;
  auto release = [&]() {
    for (void* w : weights) hipFree(w);
    if (out) hipFree(out);
    if (reference) hipFree(reference);
    if (sums) hipFree(sums);
  };
  auto fail = [&](hipError_t error) {
    release();
    return kInvalidArgument + (int)error;
  };

  if (hipMalloc(&out, out_bytes) != hipSuccess) return fail(hipErrorOutOfMemory);
  if (hipMalloc(&reference, out_bytes) != hipSuccess) return fail(hipErrorOutOfMemory);
  if (hipMalloc(&sums, 2 * sizeof(float)) != hipSuccess) return fail(hipErrorOutOfMemory);
  for (int i = 0; i < ring; ++i) {
    void* copy = nullptr;
    // Running out of memory here is a reason to tune on a shorter ring, not to
    // fail the load: whatever was allocated still spans more than the cache.
    if (hipMalloc(&copy, weight_bytes) != hipSuccess) break;
    if (hipMemcpyAsync(copy, b, weight_bytes, hipMemcpyDeviceToDevice, lt->stream) !=
        hipSuccess) {
      hipFree(copy);
      break;
    }
    weights.push_back(copy);
  }
  if (weights.size() < 2) return fail(hipErrorOutOfMemory);
  ring = (int)weights.size();
  if (hipStreamSynchronize(lt->stream) != hipSuccess) return fail(hipErrorUnknown);

  hipEvent_t start = nullptr, stop = nullptr;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  int next = 0;
  auto rotate = [&]() {
    const void* w = weights[next];
    next = (next + 1) % ring;
    return w;
  };
  const float alpha = 1.0f, beta = 0.0f;

  // hipBLAS first: it is both a candidate and the reference every hipBLASLt
  // candidate is checked against.
  auto run_blas = [&](void* dst) {
    return hipblasGemmEx(static_cast<hipblasHandle_t>(blas_handle), HIPBLAS_OP_N,
                         HIPBLAS_OP_N, n, m, k, &alpha, rotate(), HIP_R_16BF, n, a,
                         HIP_R_16BF, k, &beta, dst, HIP_R_16BF, n, HIPBLAS_COMPUTE_32F,
                         HIPBLAS_GEMM_DEFAULT);
  };
  if (run_blas(reference) != HIPBLAS_STATUS_SUCCESS) {
    release();
    hipEventDestroy(start);
    hipEventDestroy(stop);
    return kInvalidArgument;
  }
  // A fused candidate emits the GEMM with the bias and activation applied, so
  // the reference has to carry them too. Without this every fused candidate
  // fails validation and the tuner silently routes everything to hipBLAS --
  // a performance bug that looks exactly like "fusion did not help".
  if (epilogue != kEpilogueNone) {
    bias_activation_for_tuning<<<grid_for((int64_t)m * n, kElementwiseBlock),
                                 kElementwiseBlock, 0, lt->stream>>>(
        static_cast<const uint16_t*>(reference), static_cast<const uint16_t*>(bias),
        static_cast<uint16_t*>(reference), (int64_t)m * n, n,
        epilogue == kEpilogueBiasGeluTanh, epilogue == kEpilogueBiasRelu);
  }
  hipStreamSynchronize(lt->stream);
  hipEventRecord(start, lt->stream);
  for (int i = 0; i < kTuneIterations; ++i) run_blas(out);
  hipEventRecord(stop, lt->stream);
  hipEventSynchronize(stop);
  *blas_ms = elapsed_ms(start, stop) / kTuneIterations;

  // Then every hipBLASLt candidate the heuristic offers for this problem.
  // With an epilogue, hipBLAS's arm is the GEMM *plus* the separate
  // bias/activation kernel it would still need, because that is what the
  // fused candidate replaces. Timing the bare GEMM against a fused one
  // compares different amounts of work and routes by the wrong answer.
  if (epilogue != kEpilogueNone) {
    const int64_t n_out = (int64_t)m * n;
    const int gelu = epilogue == kEpilogueBiasGeluTanh;
    const int relu = epilogue == kEpilogueBiasRelu;
    auto post = [&] {
      bias_activation_for_tuning<<<grid_for(n_out, kElementwiseBlock), kElementwiseBlock, 0,
                                   lt->stream>>>(
          static_cast<const uint16_t*>(out), static_cast<const uint16_t*>(bias),
          static_cast<uint16_t*>(out), n_out, n, gelu, relu);
    };
    post();
    hipStreamSynchronize(lt->stream);
    hipEventRecord(start, lt->stream);
    for (int i = 0; i < kTuneIterations; ++i) post();
    hipEventRecord(stop, lt->stream);
    hipEventSynchronize(stop);
    *blas_ms += elapsed_ms(start, stop) / kTuneIterations;
  }

  hipblasLtMatmulPreference_t preference = nullptr;
  hipblasLtMatmulPreferenceCreate(&preference);
  const uint64_t workspace = 0;
  hipblasLtMatmulPreferenceSetAttribute(
      preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace, sizeof(workspace));
  constexpr int kMaxCandidates = 16;
  hipblasLtMatmulHeuristicResult_t candidates[kMaxCandidates];
  int found = 0;
  hipblasLtMatmulAlgoGetHeuristic(lt->handle, problem->desc, problem->b, problem->a,
                                  problem->c, problem->c, preference, kMaxCandidates,
                                  candidates, &found);
  hipblasLtMatmulPreferenceDestroy(preference);

  const int64_t elements = (int64_t)m * n;
  for (int i = 0; i < found; ++i) {
    if (candidates[i].state != HIPBLAS_STATUS_SUCCESS) continue;
    auto run_lt = [&] {
      return hipblasLtMatmul(lt->handle, problem->desc, &alpha, rotate(), problem->b, a,
                             problem->a, &beta, out, problem->c, out, problem->c,
                             &candidates[i].algo, nullptr, 0, lt->stream);
    };
    if (run_lt() != HIPBLAS_STATUS_SUCCESS) continue;
    if (hipStreamSynchronize(lt->stream) != hipSuccess) continue;

    // Checked before it is timed: a candidate that computes the wrong thing
    // must never become the recorded winner, however fast it is.
    hipMemsetAsync(sums, 0, 2 * sizeof(float), lt->stream);
    squared_error_kernel<<<grid_for(elements, kReduceBlock), kReduceBlock, 0,
                           lt->stream>>>(static_cast<const uint16_t*>(out),
                                         static_cast<const uint16_t*>(reference),
                                         elements, sums);
    float host_sums[2] = {0.0f, 0.0f};
    hipMemcpyAsync(host_sums, sums, sizeof(host_sums), hipMemcpyDeviceToHost, lt->stream);
    if (hipStreamSynchronize(lt->stream) != hipSuccess) continue;
    const double relative =
        host_sums[1] > 0.0f ? sqrt((double)host_sums[0] / (double)host_sums[1]) : 0.0;
    if (relative > *worst_relative_l2) *worst_relative_l2 = relative;
    if (relative > kTuneMaxRelativeL2) continue;

    hipEventRecord(start, lt->stream);
    for (int j = 0; j < kTuneIterations; ++j) run_lt();
    hipEventRecord(stop, lt->stream);
    if (hipEventSynchronize(stop) != hipSuccess) continue;
    const double ms = elapsed_ms(start, stop) / kTuneIterations;
    if (*lt_ms < 0.0 || ms < *lt_ms) {
      *lt_ms = ms;
      memcpy(blob, &candidates[i].algo, sizeof(hipblasLtMatmulAlgo_t));
    }
  }

  hipEventDestroy(start);
  hipEventDestroy(stop);
  release();
  return status;
}

#else

extern "C" int apxinf_hip_gemm_tune(void*, void*, int, int, int, int, const void*,
                                    const void*, int, const void*, double*, double*,
                                    unsigned char*, int, double*) {
  return kInvalidArgument;
}

#endif  // APXINF_HIP_HAS_BLASLT

// ── Operators ───────────────────────────────────────────────────────

extern "C" int apxinf_hip_silu(int dtype, const void* x, void* y, int64_t n, void* stream) {
  if (n <= 0) return 0;
  return dispatch<SiluLaunch>(dtype, x, y, n, stream);
}

// op: 0 = add, 1 = multiply.
extern "C" int apxinf_hip_binary(int dtype, int op, const void* a, const void* b, void* y,
                                 int64_t n, void* stream) {
  if (op != 0 && op != 1) return kInvalidArgument;
  if (n <= 0) return 0;
  return dispatch<BinaryLaunch>(dtype, a, b, y, n, op, stream);
}

extern "C" int apxinf_hip_scale(int dtype, const void* x, void* y, int64_t n, float factor,
                                void* stream) {
  if (n <= 0) return 0;
  return dispatch<ScaleLaunch>(dtype, x, y, n, factor, stream);
}

extern "C" int apxinf_hip_rms_norm(int dtype, const void* x, const void* weight, void* y,
                                   int64_t rows, int64_t cols, float eps, void* stream) {
  if (rows <= 0) return 0;
  if (cols <= 0 || rows > 0x7fffffff) return kInvalidArgument;
  return dispatch<RmsNormLaunch>(dtype, x, weight, y, rows, cols, eps, stream);
}

extern "C" int apxinf_hip_rope(int dtype, const void* x, void* y, int seq, int heads,
                               int head_dim, float theta, uint32_t pos_offset, void* stream) {
  if (head_dim <= 0 || head_dim % 2 != 0) return kInvalidArgument;
  if (seq <= 0 || heads <= 0) return 0;
  return dispatch<RopeLaunch>(dtype, x, y, seq, heads, head_dim, theta, pos_offset, stream);
}

// element_bytes: 2 or 4. Embedding copies storage bits, so it is keyed on
// element width rather than dtype.
extern "C" int apxinf_hip_embedding(int element_bytes, const void* table, const void* ids,
                                    void* y, int64_t n_ids, int64_t dim, void* stream) {
  if (n_ids <= 0 || dim <= 0) return 0;
  const int64_t total = n_ids * dim;
  const auto* id_ptr = static_cast<const uint32_t*>(ids);
  if (element_bytes == 4) {
    embedding_kernel<uint32_t><<<grid_for(total, kElementwiseBlock), kElementwiseBlock, 0,
                                 as_stream(stream)>>>(static_cast<const uint32_t*>(table), id_ptr,
                                                      static_cast<uint32_t*>(y), n_ids, dim);
  } else if (element_bytes == 2) {
    embedding_kernel<uint16_t><<<grid_for(total, kElementwiseBlock), kElementwiseBlock, 0,
                                 as_stream(stream)>>>(static_cast<const uint16_t*>(table), id_ptr,
                                                      static_cast<uint16_t*>(y), n_ids, dim);
  } else {
    return kInvalidArgument;
  }
  return last_launch_status();
}

extern "C" int apxinf_hip_attention(int dtype, const void* q, const void* k, const void* v,
                                    void* out, int q_len, int n_heads, int n_kv_heads,
                                    int head_dim, int kv_len, int kv_offset, float scale,
                                    void* stream) {
  if (n_heads <= 0 || n_kv_heads <= 0 || n_heads % n_kv_heads != 0 || head_dim <= 0 ||
      kv_len <= 0 || kv_offset < 0) {
    return kInvalidArgument;
  }
  if (q_len <= 0) return 0;
  return dispatch<AttentionLaunch>(dtype, q, k, v, out, q_len, n_heads, n_kv_heads, head_dim,
                                   kv_len, kv_offset, scale, stream);
}

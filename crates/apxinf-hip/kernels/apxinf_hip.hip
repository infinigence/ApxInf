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
// store. BF16 conversion is done by hand rather than through <hip/hip_bf16.h>
// so the rounding is exactly IEEE round-to-nearest-even with quiet NaNs, the
// same as `half::bf16::from_f32` on the host.
//
// No kernel uses wave intrinsics or assumes a wave size. Reductions go through
// shared memory with power-of-two block sizes, so the same code is correct on
// wave32 (RDNA, e.g. gfx1151) and wave64 (CDNA) devices.

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>

namespace {

constexpr int kDTypeF32 = 0;
constexpr int kDTypeBF16 = 1;

constexpr int kBlasStatusBase = 100000;
constexpr int kInvalidArgument = 200000;

constexpr int kElementwiseBlock = 256;
constexpr int kReduceBlock = 256;
constexpr int kAttentionBlock = 128;
// Grid-stride loops cover any element count; capping the grid keeps launch
// dimensions far from the hardware limit without affecting correctness.
constexpr int64_t kMaxGrid = 1 << 20;

__device__ __forceinline__ float bf16_to_f32(uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float value) {
  uint32_t bits = __float_as_uint(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<uint16_t>((bits >> 16) | 0x40u);  // quiet NaN
  }
  bits += 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(bits >> 16);
}

template <typename T>
struct Elem;

template <>
struct Elem<float> {
  __device__ static float load(const float* p, int64_t i) { return p[i]; }
  __device__ static void store(float* p, int64_t i, float v) { p[i] = v; }
};

template <>
struct Elem<uint16_t> {
  __device__ static float load(const uint16_t* p, int64_t i) { return bf16_to_f32(p[i]); }
  __device__ static void store(uint16_t* p, int64_t i, float v) { p[i] = f32_to_bf16(v); }
};

__device__ __forceinline__ int64_t global_index() {
  return static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
}

__device__ __forceinline__ int64_t grid_stride() {
  return static_cast<int64_t>(gridDim.x) * blockDim.x;
}

unsigned grid_for(int64_t n, int block) {
  int64_t blocks = (n + block - 1) / block;
  if (blocks > kMaxGrid) blocks = kMaxGrid;
  return static_cast<unsigned>(blocks < 1 ? 1 : blocks);
}

int last_launch_status() { return static_cast<int>(hipGetLastError()); }

hipStream_t as_stream(void* stream) { return static_cast<hipStream_t>(stream); }

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
  if (code >= kInvalidArgument) {
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

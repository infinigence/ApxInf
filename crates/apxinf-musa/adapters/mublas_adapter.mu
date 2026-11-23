// Host-side BF16 GEMM for the static-inference operators.
//
// Why this is `mublasGemmEx` and not `mublasLtMatmul`
// ---------------------------------------------------
// It was mublasLt, with a plan per shape and the one algorithm this device
// offers. Profiling the model then showed every launch paired one-for-one with
// a `copyMatrixColMajorTiled` kernel -- 3927 of each in three passes -- and
// that copy cost 1180 ms of the 1956 ms the GEMMs took.
//
// The layouts were not the cause: they describe the row-major operands exactly,
// and re-deriving them by hand gives the same descriptors. This mublasLt build
// inserts the copy for this combination regardless of how the operands are
// declared -- describing the weight the only other way possible
// (`[K,N]` with `TRANS=T`) hangs inside `mublasLtMatmulAlgoGetHeuristic`. The
// same GEMM through `mublasGemmEx` launches no copy at all, and is 1.27x faster
// end to end on the most frequent shape, 2.05x under `mcu`.
//
// Both APIs compute the same identity with f32 accumulators and one rounding at
// the output, so the numbers do not move.
//
// Layout
// ------
// Same convention as the CUDA adapter (`crates/apxinf-cuda/adapters/
// cublaslt_adapter.mu`): row-major `D = A @ B` is computed through the
// column-major identity `D^T = B^T @ A^T`, so no operand is ever transposed in
// memory. For row-major `Act[M,K] @ W[K,N] -> Out[M,N]` the call is
// `gemm(OP_N, OP_N, N, M, K)` with
//
//   A = weight      { rows = N, cols = K, ld = N }
//   B = activation  { rows = K, cols = M, ld = K }
//   D = output      { rows = N, cols = M, ld = N }
//
// Checked against the identity: D_col[N,M](i,j) = sum_k W[k][i] * Act[j][k]
// = (Act @ W)[j][i] = Out[j][i]. The same shapes reproduce a double-precision
// CPU reference.
//
// Numerical contract
// ------------------
// `MUBLAS_COMPUTE_32F`: f32 accumulators, one rounding at the bf16 output. This
// is not a tuning choice. The eager bridge this replaced computed
// `_f32(activation) @ _f32(weight)` and rounded once on the way out
// (a torch bridge that is not part of this tree), and that is the contract the
// recorded golden tensors were produced under. A bf16-input algorithm with any
// other accumulator rounds per K-step and moves the numbers.
//
// Threading and lifetime
// ----------------------
// The handle is `thread_local`, as on CUDA: one host thread drives one context,
// and a handle is not usable concurrently from two threads. It lives as long as
// the stream it is keyed on and is destroyed by
// `apxinf_musa_mublas_release_stream`, before the driver stream goes; a captured
// graph that baked launches on that stream is torn down first.
//
// `mublasGemmEx` runs on the handle's stream rather than taking one, unlike the
// `mublasLtMatmul` call this replaced. `mublasSetStream` before every launch is
// what keeps the ordering the rest of the backend assumes; it is a handle
// assignment, not a synchronisation.

#include <mublas.h>
#include <musa_runtime.h>
#include <map>
#include <utility>

#include "adapter_common.h"

namespace {

// One handle per (device, stream), created on first use. Handles never migrate
// between streams: MUSA 5.1 rejects switching one away from an already
// destroyed capture stream, which is what made capture unusable before. The map
// is the whole state — a handle is looked up and used within the call, so
// nothing has to remember "the current one".
thread_local std::map<std::pair<int, musaStream_t>, mublasHandle_t> g_handles;

// Returns the handle bound to `stream`, or nullptr with `*status` set.
mublasHandle_t initialize(musaStream_t stream, int* status) {
  const int device = apxinf_adapters::stream_device(stream);
  if (device < 0) {
    // Status 1 — "handle not initialized, invalid or null". 5.1 declares both
    // spellings as the same enumerator; cublas, and therefore the CUDA port
    // this file follows, spells it `NOT_INITIALIZED`, but 4.3.5 declares only
    // `INVALID_HANDLE`. Using the alias every toolkit has keeps this
    // translation unit buildable on either.
    *status = MUBLAS_STATUS_INVALID_HANDLE;
    return nullptr;
  }
  const auto key = std::make_pair(device, stream);
  auto found = g_handles.find(key);
  if (found == g_handles.end()) {
    mublasHandle_t handle = nullptr;
    *status = mublasCreate(&handle);
    if (*status != MUBLAS_STATUS_SUCCESS) return nullptr;
    *status = mublasSetStream(handle, stream);
    if (*status != MUBLAS_STATUS_SUCCESS) {
      mublasDestroy(handle);
      return nullptr;
    }
    found = g_handles.emplace(key, handle).first;
  }
  *status = MUBLAS_STATUS_SUCCESS;
  return found->second;
}

}  // namespace

extern "C" void apxinf_musa_mublas_release_stream(musaStream_t stream) {
  for (auto it = g_handles.begin(); it != g_handles.end();) {
    if (it->first.second == stream) {
      mublasDestroy(it->second);
      it = g_handles.erase(it);
    } else ++it;
  }
}

// Argument order mirrors the CUDA adapter's `apxinf_static_bf16_gemm` so the
// two crates stay readable side by side. `m`, `n`, `k` are row-major: rows of
// the activation, columns of the weight, contraction dimension.
extern "C" int apxinf_musa_bf16_gemm(const void* activation, const void* weight,
                                     void* output, int m, int n, int k,
                                     float alpha, musaStream_t stream) {
  if (activation == nullptr || weight == nullptr || output == nullptr ||
      m <= 0 || n <= 0 || k <= 0) {
    return MUBLAS_STATUS_INVALID_VALUE;
  }
  int status = MUBLAS_STATUS_SUCCESS;
  mublasHandle_t g_handle = initialize(stream, &status);
  if (g_handle == nullptr) return status;
  // No `mublasSetStream` here: `initialize` only returns a handle already bound
  // to this stream, because that is the key it is stored under.

  // `alpha` and `beta` are host-stack locals read during the call: the default
  // pointer mode is host. Switching to device pointer mode would silently
  // break that.
  const float beta = 0.0f;

  // Rows not on a 256 boundary are padded, and on M=522 that padding is a third
  // of the last tile. `msplit_probe.mu` measures what recovering it is worth:
  // `m x k x n = 522 x 2048 x 32768` runs 11.52 ms as one call and 9.91 as
  // 512 + 10 (a full two tiles plus a narrow remainder), which is 27 ms a pass
  // over its 17 calls. But the same split *loses* on `m x k x n = 522 x 16384 x
  // 2048` (4.89 -> 5.87) and
  // on the 2048-column shapes, because the narrow call re-reads the whole
  // weight and its grid is only `N / 256` blocks -- 128 blocks read 134 MB in
  // ~3 ms, 8 blocks read 67 MB in ~8.6 ms (`weight_read_bw.mu` measures that
  // block-count curve: 8 blocks 7.8 GB/s, 32 blocks 26.8, 256 blocks 53.8).
  //
  // So the split is taken only where the remainder's grid is wide enough to
  // carry its own weight read. `kSplitMinColumns` is where the measured curve
  // crosses the threshold that makes the trade positive: at 128 blocks the
  // remainder pass costs about `weight / 45 GB/s` against a padding saving of
  // `pad * 2 * K * N / 8.55 TFLOPS`, and that is only a win for N this wide.
  constexpr int kTileRows = 256;
  constexpr int kSplitMinColumns = 8192;
  const int padded_rows = ((m + kTileRows - 1) / kTileRows) * kTileRows;
  if (padded_rows != m && n >= kSplitMinColumns && m > kTileRows) {
    const int whole_rows = (m / kTileRows) * kTileRows;
    int status = mublasGemmEx(g_handle, MUBLAS_OP_N, MUBLAS_OP_N, n, whole_rows, k,
                              &alpha, weight, MUSA_R_16BF, n, activation,
                              MUSA_R_16BF, k, &beta, output, MUSA_R_16BF, n,
                              MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
    if (status != MUBLAS_STATUS_SUCCESS) return status;
    // The remainder's rows start where the whole tiles ended, in both operands
    // and in the output. Row-major, so that is a row offset times the row
    // stride of each.
    const char* tail_activation =
        static_cast<const char*>(activation) + static_cast<size_t>(whole_rows) * k * 2;
    char* tail_output = static_cast<char*>(output) + static_cast<size_t>(whole_rows) * n * 2;
    return mublasGemmEx(g_handle, MUBLAS_OP_N, MUBLAS_OP_N, n, m - whole_rows, k,
                        &alpha, weight, MUSA_R_16BF, n, tail_activation,
                        MUSA_R_16BF, k, &beta, tail_output, MUSA_R_16BF, n,
                        MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
  }

  return mublasGemmEx(g_handle, MUBLAS_OP_N, MUBLAS_OP_N, n, m, k, &alpha,
                      weight, MUSA_R_16BF, n,        // lda
                      activation, MUSA_R_16BF, k,    // ldb
                      &beta, output, MUSA_R_16BF, n,  // ldc
                      MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
}

// A strided batch of column-major GEMMs, passed straight through to
// `mublasGemmStridedBatchedEx`.
//
// **The arguments are mublas's own**, not this file's row-major convention: `m`,
// `n`, `k` and the leading dimensions describe the column-major operands the
// vendor API takes. The row-major entry point above cannot express what the
// vision attention needs -- its operands are `[tokens, heads, 72]`, which is
// neither contiguous in the contraction nor uniformly strided across batches --
// so this one exists to let the caller say exactly what it means, transposes
// included.
//
// Strides are in elements. `trans_a`/`trans_b` are non-zero for `MUBLAS_OP_T`.
//
// Two entries, one accumulator type. C is f32 in both, and that is not a
// convenience: the library does not mix operand types within one product, so a
// wide C is what pins each product's operands to a single type -- f32 for the
// values product, whose other operand is the softmax output, and BF16 for the
// scores product, whose inputs arrive BF16 from the QKV split.
//
// **BF16 in with an f32 out is not in this library's documented type table.**
// The `mublasGemmStridedBatchedEx` supported-types list (`mublas_functions.h`,
// this toolkit's `internal/`) has `a = b = c` for bf16 and BF16 operands with
// an f32 C for f16 inputs, not this pairing. It is measured working
// (the port's vision-attention probe), and a rejection would come back as a
// nonzero status through `check_mublas`, so the loud failure mode is covered;
// the quiet one -- a toolkit that accepts the combination and misreads `C` --
// is the end-to-end cosine gate's to catch.
//
// Widening Q and K is what this port used to do. The permute entries in
// `custom_kernels.mu` carry the accounting of what narrowing them costs.
namespace {

int run_batched_colmajor(const void* a, const void* b, void* c, int m, int n,
                         int k, int trans_a, int trans_b, int lda, int ldb,
                         int ldc, long long stride_a, long long stride_b,
                         long long stride_c, int batch_count, float alpha,
                         musaDataType ab_type, musaStream_t stream) {
  if (a == nullptr || b == nullptr || c == nullptr || m <= 0 || n <= 0 ||
      k <= 0 || batch_count <= 0) {
    return MUBLAS_STATUS_INVALID_VALUE;
  }
  int status = MUBLAS_STATUS_SUCCESS;
  mublasHandle_t g_handle = initialize(stream, &status);
  if (g_handle == nullptr) return status;
  // No `mublasSetStream` here either; see `apxinf_musa_bf16_gemm`.

  const float beta = 0.0f;
  return mublasGemmStridedBatchedEx(
      g_handle, trans_a ? MUBLAS_OP_T : MUBLAS_OP_N,
      trans_b ? MUBLAS_OP_T : MUBLAS_OP_N, m, n, k, &alpha, a, ab_type, lda,
      stride_a, b, ab_type, ldb, stride_b, &beta, c, MUSA_R_32F, ldc, stride_c,
      batch_count, MUBLAS_COMPUTE_32F, MUBLAS_GEMM_DEFAULT);
}

}  // namespace

extern "C" int apxinf_musa_f32_gemm_batched_colmajor(
    const void* a, const void* b, void* c, int m, int n, int k, int trans_a,
    int trans_b, int lda, int ldb, int ldc, long long stride_a,
    long long stride_b, long long stride_c, int batch_count, float alpha,
    musaStream_t stream) {
  return run_batched_colmajor(a, b, c, m, n, k, trans_a, trans_b, lda, ldb, ldc,
                              stride_a, stride_b, stride_c, batch_count, alpha,
                              MUSA_R_32F, stream);
}

// The same batch with BF16 A/B, for the scores product: its operands come out
// of the QKV split already BF16, and the permutation that lays them out is a
// pure move rather than a widening copy when they stay that way.
extern "C" int apxinf_musa_bf16in_f32out_gemm_batched_colmajor(
    const void* a, const void* b, void* c, int m, int n, int k, int trans_a,
    int trans_b, int lda, int ldb, int ldc, long long stride_a,
    long long stride_b, long long stride_c, int batch_count, float alpha,
    musaStream_t stream) {
  return run_batched_colmajor(a, b, c, m, n, k, trans_a, trans_b, lda, ldb, ldc,
                              stride_a, stride_b, stride_c, batch_count, alpha,
                              MUSA_R_16BF, stream);
}

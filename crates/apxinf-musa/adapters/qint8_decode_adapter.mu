// MP22 W8A8 decode GEMM: gate/up KN (NN), other projections NK (NT).
// Derived from the BF16 GEMM template; provenance is in kernels/asm/README.md.
#include <musa.h>
#include <musa_runtime.h>
#include <climits>
#include <cstdint>
#include <map>
#include <utility>

#include "adapter_common.h"
#include "mubin_images.h"

namespace {
using namespace apxinf_adapters;
struct Image {
  const unsigned char* bytes;
  unsigned long size;
  const char* symbol;
};
const Image images[] = {
    {kQint8GemmNt, kQint8GemmNtSize,
     "apxinf_bf16s8gemm_nt_tc_128_32x128_epilogue"},
    {kQint8GemmNn, kQint8GemmNnSize,
     "apxinf_bf16s8gemm_nn_tc_128_32x128_epilogue"},
};
// Modules remain loaded while captured graphs can reference their functions.
thread_local std::map<std::pair<MUcontext, int>, MUfunction> functions;

MUfunction load(int nn, void* stream) {
  MUcontext context = nullptr;
  if (muCtxGetCurrent(&context) != MUSA_SUCCESS || !context) {
    set_error("INT8 decode GEMM: no current MUSA context");
    return nullptr;
  }
  const auto key = std::make_pair(context, nn);
  const auto found = functions.find(key);
  if (found != functions.end()) return found->second;
  if (capturing(stream)) {
    set_error("INT8 decode GEMM must be warmed before graph capture");
    return nullptr;
  }
  const auto& image = images[nn];
  if (!image.size) {
    set_error("INT8 decode GEMM image missing (requires MP22 assembler)");
    return nullptr;
  }
  MUmodule module = nullptr;
  MUresult status = muModuleLoadData(&module, image.bytes);
  if (status != MUSA_SUCCESS) {
    set_error("INT8 decode GEMM module load failed: %d", int(status));
    return nullptr;
  }
  MUfunction fn = nullptr;
  status = muModuleGetFunction(&fn, module, image.symbol);
  if (status != MUSA_SUCCESS) {
    muModuleUnload(module);
    set_error("INT8 decode GEMM symbol lookup failed: %d", int(status));
    return nullptr;
  }
  functions.emplace(key, fn);
  return fn;
}
}  // namespace

extern "C" int apxinf_musa_asm_qint8_decode(
    const void* a, const void* b, void* out, const void* a_scale,
    const void* b_scale, uint32_t a_scale_bytes, uint32_t b_scale_bytes,
    int m, int n, int k, int nn, void* stream) {
  if (!a || !b || !out || !a_scale || !b_scale || m <= 0 || m > 32 ||
      n <= 0 || n % 128 || k <= 0 || k % 64 || (nn != 0 && nn != 1) ||
      int64_t(m) * k > INT_MAX || int64_t(n) * k > INT_MAX ||
      int64_t(m) * n * 2 > INT_MAX ||
      a_scale_bytes < (uint64_t(m) + 512) * 2 ||
      b_scale_bytes < (uint64_t(n) + 512) * 2 ||
      a_scale_bytes > INT_MAX || b_scale_bytes > INT_MAX) {
    set_error("INT8 decode GEMM: invalid shape, layout or buffer size");
    return -1;
  }
  if (!select_device(stream_device(stream))) return -1;
  auto fn = load(nn, stream);
  if (!fn) return -1;
  int out_bytes = m * n * 2, a_bytes = m * k, b_bytes = n * k;
  int zero = 0, batch = 1, sa = m * k, sb = n * k, sc = m * n;
  int as_bytes = int(a_scale_bytes), bs_bytes = int(b_scale_bytes);
  int ldb = nn ? n : k;
  float alpha = 1.f, beta = 0.f, gamma = 0.f;
  void* nil = nullptr;
  void* args[] = {
      &out, &out_bytes, &zero, &a, &a_bytes, &zero, &b, &b_bytes, &zero,
      &nil, &zero, &zero, &nil, &zero, &zero,
      &a_scale, &as_bytes, &zero, &b_scale, &bs_bytes, &zero,
      &m, &n, &k, &k, &ldb, &n, &alpha, &beta, &gamma,
      &batch, &sa, &sb, &sc,
  };
  const auto status = muLaunchKernel(fn, n / 128, 1, 1, 128, 1, 1, 0,
                                    reinterpret_cast<MUstream>(stream), args, nullptr);
  if (status != MUSA_SUCCESS) {
    set_error("INT8 decode GEMM launch failed: %d", int(status));
    return -1;
  }
  return 0;
}

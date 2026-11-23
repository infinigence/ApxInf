#include "../include/apxinf_cuda/volume.h"
#include "../framework/runtime_internal.h"
#include "../kernels/custom/volume.cuh"
#include <algorithm>
#include <limits>
#include <memory>

namespace {
using apxinf::framework::Failure;
// Deterministic geometry/alignment dispatch; no workspace or persisted recipe.
struct Execution {
  apxinf_unfold3d_spec_t spec;
  apxinf_unfold3d_bindings_t bindings;
  int device;
  int64_t columns;
  bool vectorized;
};
int64_t multiply(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || (b && a > std::numeric_limits<int64_t>::max() / b))
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Unfold3d extent overflow");
  return a * b;
}
int64_t add(int64_t a, int64_t b) {
  if (a < 0 || b < 0 || a > std::numeric_limits<int64_t>::max() - b)
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Unfold3d extent overflow");
  return a + b;
}

template <int Channels>
bool launch_vectorized(const Execution& e) {
  const auto& s = e.spec;
  const auto* input = static_cast<const __nv_bfloat16*>(e.bindings.input);
  const auto* history = static_cast<const __nv_bfloat16*>(e.bindings.history);
  auto* output = static_cast<__nv_bfloat16*>(e.bindings.output);
  const int blocks = static_cast<int>(
      std::min<int64_t>(4096, (s.rows * e.columns / 8 + 255) / 256));
  auto stream = static_cast<cudaStream_t>(e.bindings.stream);
  if (s.kernel[0] == 3 && s.kernel[1] == 3 && s.kernel[2] == 3) {
    apxinf::volume::kernels::unfold3d_vectorized<Channels, 3, 3, 3>
        <<<blocks, 256, 0, stream>>>(s, input, history, output);
  } else if (s.kernel[0] == 1 && s.kernel[1] == 3 && s.kernel[2] == 3) {
    apxinf::volume::kernels::unfold3d_vectorized<Channels, 1, 3, 3>
        <<<blocks, 256, 0, stream>>>(s, input, history, output);
  } else if (s.kernel[0] == 3 && s.kernel[1] == 1 && s.kernel[2] == 1) {
    apxinf::volume::kernels::unfold3d_vectorized<Channels, 3, 1, 1>
        <<<blocks, 256, 0, stream>>>(s, input, history, output);
  } else if (s.kernel[0] == 1 && s.kernel[1] == 1 && s.kernel[2] == 1) {
    apxinf::volume::kernels::unfold3d_vectorized<Channels, 1, 1, 1>
        <<<blocks, 256, 0, stream>>>(s, input, history, output);
  } else {
    return false;
  }
  return true;
}

bool launch_vectorized(const Execution& e) {
  switch (e.spec.input[3]) {
    case 32: return launch_vectorized<32>(e);
    case 96: return launch_vectorized<96>(e);
    case 192: return launch_vectorized<192>(e);
    case 384: return launch_vectorized<384>(e);
    default: return false;
  }
}
}  // namespace

extern "C" apxinf_status_t apxinf_unfold3d_prepare(apxinf_runtime_t runtime,
    const apxinf_unfold3d_spec_t* spec, const apxinf_unfold3d_bindings_t* bindings,
    void** execution) {
  return apxinf::framework::abi_boundary([&] {
    if (!runtime || !spec || !bindings || !execution)
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Unfold3d argument");
    *execution = nullptr;
    const auto& s = *spec;
    if (s.version != APXINF_UNFOLD3D_SPEC_VERSION || s.reserved ||
        s.history_frames < 0 || s.row_start < 0 || s.rows <= 0 || s.input[3] <= 0 ||
        !bindings->input || !bindings->output ||
        ((s.history_frames != 0) != (bindings->history != nullptr)))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Unfold3d contract");
    int64_t columns = s.input[3], rows = 1, input_elements = s.input[3];
    for (int i = 0; i < 3; ++i) {
      if (s.input[i] <= 0 || s.kernel[i] <= 0 || s.stride[i] <= 0)
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Unfold3d dimension");
      int64_t padded = add(add(add(s.input[i], i == 0 ? s.history_frames : 0),
                                  s.padding_before[i]), s.padding_after[i]);
      if (padded < s.kernel[i] || s.output[i] != (padded - s.kernel[i]) / s.stride[i] + 1)
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Unfold3d output extent mismatch");
      columns = multiply(columns, s.kernel[i]);
      rows = multiply(rows, s.output[i]);
      input_elements = multiply(input_elements, i == 0 ? add(s.input[i], s.history_frames) : s.input[i]);
    }
    multiply(input_elements, 2);
    multiply(multiply(s.rows, columns), 2);
    if (add(s.row_start, s.rows) > rows)
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Unfold3d tile exceeds output");
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    cudaStreamCaptureStatus capture;
    apxinf::framework::check_cuda(cudaStreamIsCapturing(static_cast<cudaStream_t>(bindings->stream), &capture));
    if (capture != cudaStreamCaptureStatusNone)
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "Unfold3d preparation during capture");
    bool vectorized = input_elements <= INT32_MAX && rows <= INT32_MAX &&
        s.rows * columns <= INT32_MAX &&
        (reinterpret_cast<uintptr_t>(bindings->input) % 16) == 0 &&
        (reinterpret_cast<uintptr_t>(bindings->history) % 16) == 0 &&
        (reinterpret_cast<uintptr_t>(bindings->output) % 16) == 0;
    for (int i = 0; vectorized && i < 3; ++i) {
      // Bound pre-padding coordinates too: an otherwise tiny output can have
      // very large strides or padding. Such valid inputs use the scalar path.
      const int64_t last = add(multiply(s.output[i] - 1, s.stride[i]),
                               s.kernel[i] - 1);
      vectorized = s.stride[i] <= INT32_MAX &&
          s.padding_before[i] <= INT32_MAX && last <= INT32_MAX;
    }
    auto result = std::make_unique<Execution>(
        Execution{s, *bindings, runtime->device, columns, vectorized});
    *execution = result.release();
  });
}

extern "C" apxinf_status_t apxinf_unfold3d_enqueue(void* opaque) {
  return apxinf::framework::abi_boundary([&] {
    if (!opaque) throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Unfold3d execution");
    const auto& e = *static_cast<Execution*>(opaque);
    apxinf::framework::check_cuda(cudaSetDevice(e.device));
    if (!e.vectorized || !launch_vectorized(e)) {
      const int blocks = static_cast<int>(std::min<int64_t>(4096, (e.spec.rows * e.columns - 1) / 256 + 1));
      apxinf::volume::kernels::unfold3d<<<blocks, 256, 0, static_cast<cudaStream_t>(e.bindings.stream)>>>(
          e.spec, static_cast<const __nv_bfloat16*>(e.bindings.input),
          static_cast<const __nv_bfloat16*>(e.bindings.history),
          static_cast<__nv_bfloat16*>(e.bindings.output), e.columns);
    }
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}

extern "C" void apxinf_unfold3d_destroy(void* execution) {
  delete static_cast<Execution*>(execution);
}

// CPU primitive owned by the Python processor package; no model/runtime state.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <new>
#include <stdexcept>
#include <vector>

namespace {
struct Buffer {
  Py_buffer view{};
  ~Buffer() { if (view.obj) PyBuffer_Release(&view); }
};
struct Coefficient { Py_ssize_t low, high; float left, right; };
std::vector<Coefficient> coefficients(Py_ssize_t input, Py_ssize_t output) {
  std::vector<Coefficient> result(output);
  const float scale = float(input) / float(output);
  for (Py_ssize_t i = 0; i < output; ++i) {
    const float center = (float(i) + .5f) * scale;
    const auto low = std::clamp(Py_ssize_t(std::floor(center - .5f)), Py_ssize_t(0), input - 1);
    const auto high = std::min(low + 1, input - 1);
    const float left = std::max(0.f, 1.f - std::abs(float(low) + .5f - center));
    const float right = high == low ? 0.f : std::max(0.f, 1.f - std::abs(float(high) + .5f - center));
    const float total = left + right;
    result[i] = {low, high, left / total, right / total};
  }
  return result;
}
bool count_fits(Py_ssize_t rows, Py_ssize_t cols, Py_ssize_t element_bytes) {
  return rows > 0 && cols > 0 && cols <= PY_SSIZE_T_MAX / 3 / element_bytes &&
         rows <= PY_SSIZE_T_MAX / (cols * 3 * element_bytes);
}
PyObject* bilinear_upsample_rgb(PyObject*, PyObject* args) {
  PyObject* object;
  Py_ssize_t out_h, out_w;
  if (!PyArg_ParseTuple(args, "Onn", &object, &out_h, &out_w)) return nullptr;
  Buffer buffer;
  if (PyObject_GetBuffer(object, &buffer.view, PyBUF_STRIDES | PyBUF_FORMAT)) return nullptr;
  const auto& b = buffer.view;
  if (b.ndim != 3 || b.itemsize != 1 || !b.format || b.format[0] != 'B' || b.format[1] ||
      b.shape[0] <= 0 || b.shape[1] <= 0 || b.shape[2] != 3 ||
      b.strides[1] != 3 || b.strides[2] != 1 || b.suboffsets ||
      b.shape[1] > PY_SSIZE_T_MAX / 3 || b.strides[0] < b.shape[1] * 3) {
    PyErr_SetString(PyExc_ValueError, "expected nonempty HWC uint8 RGB with contiguous pixels and positive row stride");
    return nullptr;
  }
  const Py_ssize_t height = b.shape[0], width = b.shape[1];
  if (out_h < height || out_w < width) {
    PyErr_SetString(PyExc_ValueError, "bilinear upsampling requires output dimensions at least the input dimensions");
    return nullptr;
  }
  if (!count_fits(out_h, out_w, 1) || !count_fits(height, out_w, sizeof(float))) {
    PyErr_SetString(PyExc_OverflowError, "resize dimensions overflow storage");
    return nullptr;
  }
  try {
    const auto cx = coefficients(width, out_w), cy = coefficients(height, out_h);
    std::vector<float> horizontal(height * out_w * 3);
    PyObject* result = PyByteArray_FromStringAndSize(nullptr, out_h * out_w * 3);
    if (!result) return nullptr;
    auto* dst = reinterpret_cast<uint8_t*>(PyByteArray_AS_STRING(result));
    const auto* src = static_cast<const uint8_t*>(b.buf);
    // The first tap rounds to FP32; only the second tap uses FMA. Keep the
    // horizontal FP32 intermediate and round to nearest-even uint8 once.
    Py_BEGIN_ALLOW_THREADS
    for (Py_ssize_t y = 0; y < height; ++y) {
      for (Py_ssize_t x = 0; x < out_w; ++x) {
        const auto a = cx[x];
        for (int c = 0; c < 3; ++c) {
          const float first = float(src[y * b.strides[0] + a.low * 3 + c]) * a.left;
          horizontal[(y * out_w + x) * 3 + c] =
              std::fma(float(src[y * b.strides[0] + a.high * 3 + c]), a.right, first);
        }
      }
    }
    for (Py_ssize_t y = 0; y < out_h; ++y) {
      const auto a = cy[y];
      for (Py_ssize_t x = 0; x < out_w * 3; ++x) {
        const float first = horizontal[a.low * out_w * 3 + x] * a.left;
        const float value = std::fma(horizontal[a.high * out_w * 3 + x], a.right, first);
        dst[y * out_w * 3 + x] = uint8_t(std::clamp(std::nearbyint(value), 0.f, 255.f));
      }
    }
    Py_END_ALLOW_THREADS
    return result;
  } catch (const std::bad_alloc&) {
    return PyErr_NoMemory();
  } catch (const std::length_error&) {
    PyErr_SetString(PyExc_OverflowError, "resize storage exceeds container limits");
    return nullptr;
  }
}
PyMethodDef methods[] = {
    {"bilinear_upsample_rgb", bilinear_upsample_rgb, METH_VARARGS,
     "Resize HWC uint8 RGB by separable FP32 bilinear upsampling; return writable bytes."},
    {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_resize_native", nullptr, -1, methods};
}  // namespace
PyMODINIT_FUNC PyInit__resize_native() { return PyModule_Create(&module); }

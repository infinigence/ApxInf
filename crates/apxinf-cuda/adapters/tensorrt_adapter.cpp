#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

thread_local std::string last_error;

class Logger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, char const* message) noexcept override {
    if (severity <= Severity::kERROR) last_error = message ? message : "TensorRT error";
  }
};

Logger logger;

template <typename T>
struct TrtDelete {
  void operator()(T* value) const noexcept { delete value; }
};

struct Engine {
  std::unique_ptr<nvinfer1::IRuntime, TrtDelete<nvinfer1::IRuntime>> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine, TrtDelete<nvinfer1::ICudaEngine>> engine;
  std::unique_ptr<nvinfer1::IExecutionContext, TrtDelete<nvinfer1::IExecutionContext>> context;
};

void fail(char const* message) { last_error = message ? message : "TensorRT error"; }

}  // namespace

extern "C" {

char const* apxinf_trt_last_error() { return last_error.c_str(); }

void* apxinf_trt_load(char const* path) {
  last_error.clear();
  if (!path) {
    fail("TensorRT engine path is null");
    return nullptr;
  }
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    last_error = std::string("open TensorRT engine: ") + path;
    return nullptr;
  }
  auto const size = input.tellg();
  if (size <= 0) {
    fail("TensorRT engine is empty");
    return nullptr;
  }
  input.seekg(0, std::ios::beg);
  std::vector<char> bytes(static_cast<size_t>(size));
  if (!input.read(bytes.data(), size)) {
    fail("read TensorRT engine");
    return nullptr;
  }

  auto value = std::make_unique<Engine>();
  value->runtime.reset(nvinfer1::createInferRuntime(logger));
  if (!value->runtime) {
    fail("create TensorRT runtime");
    return nullptr;
  }
  value->engine.reset(value->runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
  if (!value->engine) {
    if (last_error.empty()) fail("deserialize TensorRT engine");
    return nullptr;
  }
  value->context.reset(value->engine->createExecutionContext());
  if (!value->context) {
    fail("create TensorRT execution context");
    return nullptr;
  }
  return value.release();
}

void apxinf_trt_destroy(void* opaque) { delete static_cast<Engine*>(opaque); }

int32_t apxinf_trt_num_io(void const* opaque) {
  auto const* value = static_cast<Engine const*>(opaque);
  return value ? value->engine->getNbIOTensors() : -1;
}

char const* apxinf_trt_tensor_name(void const* opaque, int32_t index) {
  auto const* value = static_cast<Engine const*>(opaque);
  if (!value || index < 0 || index >= value->engine->getNbIOTensors()) return nullptr;
  return value->engine->getIOTensorName(index);
}

int32_t apxinf_trt_tensor_mode(void const* opaque, char const* name) {
  auto const* value = static_cast<Engine const*>(opaque);
  if (!value || !name) return -1;
  return value->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT ? 0 : 1;
}

int32_t apxinf_trt_tensor_dtype(void const* opaque, char const* name) {
  auto const* value = static_cast<Engine const*>(opaque);
  if (!value || !name) return -1;
  // TensorRT appends data types without renumbering the stable ABI enum.
  // Returning the numeric value keeps this adapter source-compatible with
  // Jetson's older 10.3 headers (which do not name the newest FP4 member).
  return static_cast<int32_t>(value->engine->getTensorDataType(name));
}

int32_t apxinf_trt_tensor_shape(void const* opaque, char const* name, int64_t* dims,
                                int32_t capacity) {
  auto const* value = static_cast<Engine const*>(opaque);
  if (!value || !name) return -1;
  auto shape = value->context->getTensorShape(name);
  if (shape.nbDims < 0 || capacity < shape.nbDims) return -1;
  for (int32_t i = 0; i < shape.nbDims; ++i) dims[i] = shape.d[i];
  return shape.nbDims;
}

int32_t apxinf_trt_set_input_shape(void* opaque, char const* name, int64_t const* dims,
                                   int32_t rank) {
  last_error.clear();
  auto* value = static_cast<Engine*>(opaque);
  if (!value || !name || !dims || rank < 0 || rank > nvinfer1::Dims::MAX_DIMS) {
    fail("invalid TensorRT input shape arguments");
    return -1;
  }
  nvinfer1::Dims shape{};
  shape.nbDims = rank;
  for (int32_t i = 0; i < rank; ++i) shape.d[i] = dims[i];
  if (!value->context->setInputShape(name, shape)) {
    last_error = std::string("set TensorRT input shape: ") + name;
    return -1;
  }
  return 0;
}

int32_t apxinf_trt_set_address(void* opaque, char const* name, void* address) {
  last_error.clear();
  auto* value = static_cast<Engine*>(opaque);
  if (!value || !name || !address) {
    fail("invalid TensorRT tensor address arguments");
    return -1;
  }
  if (!value->context->setTensorAddress(name, address)) {
    last_error = std::string("set TensorRT tensor address: ") + name;
    return -1;
  }
  return 0;
}

int32_t apxinf_trt_enqueue(void* opaque, void* stream) {
  last_error.clear();
  auto* value = static_cast<Engine*>(opaque);
  if (!value || !stream) {
    fail("invalid TensorRT enqueue arguments");
    return -1;
  }
  if (!value->context->enqueueV3(static_cast<cudaStream_t>(stream))) {
    if (last_error.empty()) fail("TensorRT enqueueV3 failed");
    return -1;
  }
  return 0;
}

}  // extern "C"

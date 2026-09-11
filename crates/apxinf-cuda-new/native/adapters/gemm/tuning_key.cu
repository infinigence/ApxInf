#include "internal.h"

#include <iomanip>

namespace apxinf::gemm {
namespace {

void append_hex_string(std::ostringstream& output, const char* value) {
  const auto* byte = reinterpret_cast<const unsigned char*>(value);
  while (*byte != 0) {
    output << std::hex << std::setw(2) << std::setfill('0')
           << static_cast<unsigned int>(*byte++);
  }
  output << std::dec;
}

std::string compatibility_fingerprint(const cudaDeviceProp& properties) {
  std::ostringstream value;
  // UUID identifies a device instance, not an execution capability, and is
  // deliberately absent from persistent cache identities.
  value << "cc=" << properties.major << '.' << properties.minor;
  return value.str();
}

std::string performance_fingerprint(const cudaDeviceProp& properties) {
  std::ostringstream value;
  value << compatibility_fingerprint(properties) << "|name=";
  append_hex_string(value, properties.name);
  value << "|sms=" << properties.multiProcessorCount
        << "|threads-per-sm=" << properties.maxThreadsPerMultiProcessor
        << "|global-memory=" << properties.totalGlobalMem
        << "|memory-bus=" << properties.memoryBusWidth
        << "|l2=" << properties.l2CacheSize;
  return value.str();
}

std::string common_key(const Spec& spec,
                       const apxinf_gemm_policy_t& policy,
                       int runtime_version,
                       int driver_version,
                       const cudaDeviceProp& properties) {
  std::ostringstream key;
  key << "gemm-recipe-v6|" << APXINF_GEMM_BUILD_ID << '|'
      << properties.major * 10 + properties.minor << '|' << runtime_version << '|'
      << driver_version
      << '|' << cublasLtGetVersion() << '|'
      << compatibility_fingerprint(properties) << '|'
      << spec.version << '|' << static_cast<uint32_t>(spec.semantic) << '|'
      << spec.m << '|' << spec.n << '|' << spec.k << '|' << spec.a_dtype << '|'
      << spec.b_dtype << '|' << spec.accumulation_dtype << '|'
      << spec.output_dtype << '|' << spec.quantization << '|'
      << spec.b_is_immutable << '|'
      << spec.a_alignment << '|' << spec.b_alignment << '|'
      << spec.bias_alignment << '|' << spec.a_scales_alignment << '|'
      << spec.b_scales_alignment << '|' << spec.output_alignment;

  // Only the unit/non-unit predicates matter for selection. The scale values
  // themselves are execution bindings and must not fragment the cache.
  key << '|' << (spec.alpha_is_unit != 0 ? 1 : 0) << '|'
      << (spec.output_scale_is_unit != 0 ? 1 : 0) << '|'
      << policy.workspace_limit << '|' << policy.graph_safe << '|'
      << policy.deterministic;
  return key.str();
}

}  // namespace

TuningKeys tuning_keys(const Spec& spec,
                       const apxinf_gemm_policy_t& policy,
                       int device) {
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device));
  int runtime_version = 0;
  int driver_version = 0;
  check_cuda(cudaRuntimeGetVersion(&runtime_version));
  check_cuda(cudaDriverGetVersion(&driver_version));

  const std::string common =
      common_key(spec, policy, runtime_version, driver_version, properties);
  return {
      common + "|performance|" + performance_fingerprint(properties),
      common + "|compatible-hint",
  };
}

}  // namespace apxinf::gemm

// Private test hook. UUID is accepted so tests can prove that changing only
// UUID does not affect either fingerprint. It is not part of the public C ABI.
extern "C" size_t apxinf_gemm_test_hardware_fingerprint(
    const unsigned char* uuid,
    size_t uuid_size,
    int32_t multiprocessor_count,
    uint64_t total_global_memory,
    int32_t performance,
    char* output,
    size_t capacity) {
  (void)uuid;
  (void)uuid_size;
  cudaDeviceProp properties{};
  std::strncpy(properties.name, "Synthetic NVIDIA GPU",
               sizeof(properties.name) - 1);
  properties.major = 11;
  properties.minor = 0;
  properties.multiProcessorCount = multiprocessor_count;
  properties.maxThreadsPerMultiProcessor = 1536;
  properties.totalGlobalMem = total_global_memory;
  properties.memoryBusWidth = 256;
  properties.l2CacheSize = 16 * 1024 * 1024;
  const std::string value = performance != 0
                                ? apxinf::gemm::performance_fingerprint(properties)
                                : apxinf::gemm::compatibility_fingerprint(properties);
  if (output != nullptr && capacity != 0) {
    const size_t copied = std::min(value.size(), capacity - 1);
    std::memcpy(output, value.data(), copied);
    output[copied] = '\0';
  }
  return value.size();
}

// Copyright 2026 apxinf contributors.
// Static E4M3 helpers for the static inference Thor inference path.

#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

extern "C" cudaError_t apxinf_static_evict_l2(
    void* buffer, size_t bytes, uint32_t seed, cudaStream_t stream);

namespace {

constexpr size_t kWorkspaceBytes = 32ull << 20;

struct ShapeKey {
  int m;
  int n;
  int k;

  bool operator==(const ShapeKey& other) const {
    return m == other.m && n == other.n && k == other.k;
  }
};

struct ShapeHash {
  size_t operator()(const ShapeKey& key) const {
    return static_cast<size_t>(key.m) * 104729u +
           static_cast<size_t>(key.n) * 1039u +
           static_cast<size_t>(key.k) * 31u;
  }
};

struct GeluKey {
  ShapeKey shape;
  const void* bias;
  uint32_t output_scale_bits;

  bool operator==(const GeluKey& other) const {
    return shape == other.shape && bias == other.bias &&
           output_scale_bits == other.output_scale_bits;
  }
};

struct GeluHash {
  size_t operator()(const GeluKey& key) const {
    return ShapeHash{}(key.shape) ^
           (std::hash<const void*>{}(key.bias) << 1) ^
           static_cast<size_t>(key.output_scale_bits);
  }
};

struct ResidualKey {
  ShapeKey shape;
  const void* bias;

  bool operator==(const ResidualKey& other) const {
    return shape == other.shape && bias == other.bias;
  }
};

struct ResidualHash {
  size_t operator()(const ResidualKey& key) const {
    return ShapeHash{}(key.shape) ^ (std::hash<const void*>{}(key.bias) << 1);
  }
};

struct GemmPlan {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight = nullptr;
  cublasLtMatrixLayout_t activation = nullptr;
  cublasLtMatrixLayout_t output = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  bool has_algorithm = false;
};

struct Bf16GemmPlan {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight = nullptr;
  cublasLtMatrixLayout_t activation = nullptr;
  cublasLtMatrixLayout_t output = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  bool has_algorithm = false;
  bool autotuned = false;
  int returned_algorithms = 0;
  int best_rank = -1;
  float default_ms = -1.0f;
  float best_ms = -1.0f;
};

struct CustomAlgoConfig {
  int tile_id;
  int custom_option;
  int stages_id;
  int cluster_shape_id;
};

struct FusedGeluPlan {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight = nullptr;
  cublasLtMatrixLayout_t activation = nullptr;
  cublasLtMatrixLayout_t c = nullptr;
  cublasLtMatrixLayout_t d = nullptr;
  void* c_buffer = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  bool has_algorithm = false;
};

struct ResidualPlan {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight = nullptr;
  cublasLtMatrixLayout_t activation = nullptr;
  cublasLtMatrixLayout_t output = nullptr;
  half* zero_bias = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  bool has_algorithm = false;
};

// A CUDA context is driven from one host execution thread while its graph is
// prepared/captured. Thread-local state avoids cross-context global locking;
// prepare entry points below install every shape-dependent resource before
// capture, while execution entry points only look up an existing plan.
thread_local cublasLtHandle_t g_lt = nullptr;
thread_local void* g_workspace = nullptr;
thread_local std::unordered_map<ShapeKey, GemmPlan, ShapeHash> g_plans;
thread_local std::unordered_map<ShapeKey, Bf16GemmPlan, ShapeHash>
    g_bf16_plans;
thread_local std::unordered_map<ShapeKey, GemmPlan, ShapeHash>
    g_fp8_split_plans;
thread_local std::unordered_map<ShapeKey, GemmPlan, ShapeHash>
    g_bf16_split_plans;
thread_local std::unordered_map<GeluKey, FusedGeluPlan, GeluHash> g_gelu_plans;
thread_local std::unordered_map<ResidualKey, ResidualPlan, ResidualHash>
    g_residual_plans;
thread_local std::unordered_map<ResidualKey, ResidualPlan, ResidualHash>
    g_bias_plans;
thread_local std::unordered_map<ShapeKey, int, ShapeHash> g_cublaslt_ranks;
thread_local std::unordered_map<ShapeKey, int, ShapeHash> g_fp8_bias_ranks;
thread_local std::unordered_map<ShapeKey, int, ShapeHash> g_fp8_gelu_ranks;
thread_local std::unordered_map<ShapeKey, int, ShapeHash> g_fp8_residual_ranks;
thread_local std::unordered_map<ShapeKey, int, ShapeHash> g_bf16_ranks;
thread_local std::unordered_map<ShapeKey, CustomAlgoConfig, ShapeHash>
    g_fp8_custom_algorithms;
thread_local std::unordered_map<ShapeKey, CustomAlgoConfig, ShapeHash>
    g_fp8_bias_custom_algorithms;
thread_local std::unordered_map<ShapeKey, CustomAlgoConfig, ShapeHash>
    g_fp8_residual_custom_algorithms;
thread_local std::unordered_map<ShapeKey, CustomAlgoConfig, ShapeHash>
    g_bf16_custom_algorithms;
thread_local std::unordered_map<ShapeKey, CustomAlgoConfig, ShapeHash>
    g_fp8_split_custom_algorithms;
thread_local std::unordered_map<ShapeKey, CustomAlgoConfig, ShapeHash>
    g_bf16_split_custom_algorithms;
thread_local std::unordered_map<uint32_t, float*> g_device_scales;

void destroy_plan(GemmPlan* plan) {
  if (plan->operation != nullptr) cublasLtMatmulDescDestroy(plan->operation);
  if (plan->weight != nullptr) cublasLtMatrixLayoutDestroy(plan->weight);
  if (plan->activation != nullptr) cublasLtMatrixLayoutDestroy(plan->activation);
  if (plan->output != nullptr) cublasLtMatrixLayoutDestroy(plan->output);
  *plan = GemmPlan{};
}

void destroy_bf16_plan(Bf16GemmPlan* plan) {
  if (plan->operation != nullptr) cublasLtMatmulDescDestroy(plan->operation);
  if (plan->weight != nullptr) cublasLtMatrixLayoutDestroy(plan->weight);
  if (plan->activation != nullptr) cublasLtMatrixLayoutDestroy(plan->activation);
  if (plan->output != nullptr) cublasLtMatrixLayoutDestroy(plan->output);
  *plan = Bf16GemmPlan{};
}

void destroy_gelu_plan(FusedGeluPlan* plan) {
  if (plan->operation != nullptr) cublasLtMatmulDescDestroy(plan->operation);
  if (plan->weight != nullptr) cublasLtMatrixLayoutDestroy(plan->weight);
  if (plan->activation != nullptr) cublasLtMatrixLayoutDestroy(plan->activation);
  if (plan->c != nullptr) cublasLtMatrixLayoutDestroy(plan->c);
  if (plan->d != nullptr) cublasLtMatrixLayoutDestroy(plan->d);
  if (plan->c_buffer != nullptr) cudaFree(plan->c_buffer);
  *plan = FusedGeluPlan{};
}

void destroy_residual_plan(ResidualPlan* plan) {
  if (plan->operation != nullptr) cublasLtMatmulDescDestroy(plan->operation);
  if (plan->weight != nullptr) cublasLtMatrixLayoutDestroy(plan->weight);
  if (plan->activation != nullptr) cublasLtMatrixLayoutDestroy(plan->activation);
  if (plan->output != nullptr) cublasLtMatrixLayoutDestroy(plan->output);
  if (plan->zero_bias != nullptr) cudaFree(plan->zero_bias);
  *plan = ResidualPlan{};
}

void invalidate_fp8_shape_plans(const ShapeKey& key) {
  auto plan_it = g_plans.find(key);
  if (plan_it != g_plans.end()) {
    destroy_plan(&plan_it->second);
    g_plans.erase(plan_it);
  }
}

void invalidate_fp8_split_plans(const ShapeKey& key) {
  auto split_it = g_fp8_split_plans.find(key);
  if (split_it != g_fp8_split_plans.end()) {
    destroy_plan(&split_it->second);
    g_fp8_split_plans.erase(split_it);
  }
}

void invalidate_fp8_bias_plans(const ShapeKey& key) {
  for (auto it = g_bias_plans.begin(); it != g_bias_plans.end();) {
    if (it->first.shape == key) {
      destroy_residual_plan(&it->second);
      it = g_bias_plans.erase(it);
    } else {
      ++it;
    }
  }
}

void invalidate_fp8_residual_plans(const ShapeKey& key) {
  for (auto it = g_residual_plans.begin(); it != g_residual_plans.end();) {
    if (it->first.shape == key) {
      destroy_residual_plan(&it->second);
      it = g_residual_plans.erase(it);
    } else {
      ++it;
    }
  }
}

void invalidate_fp8_gelu_plans(const ShapeKey& key) {
  for (auto it = g_gelu_plans.begin(); it != g_gelu_plans.end();) {
    if (it->first.shape == key) {
      destroy_gelu_plan(&it->second);
      it = g_gelu_plans.erase(it);
    } else {
      ++it;
    }
  }
}

void invalidate_bf16_shape_plans(const ShapeKey& key) {
  auto plan_it = g_bf16_plans.find(key);
  if (plan_it != g_bf16_plans.end()) {
    destroy_bf16_plan(&plan_it->second);
    g_bf16_plans.erase(plan_it);
  }
  auto split_it = g_bf16_split_plans.find(key);
  if (split_it != g_bf16_split_plans.end()) {
    destroy_plan(&split_it->second);
    g_bf16_split_plans.erase(split_it);
  }
}

cublasStatus_t initialize() {
  if (g_lt != nullptr) return CUBLAS_STATUS_SUCCESS;
  cublasStatus_t status = cublasLtCreate(&g_lt);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  cudaError_t cuda_status = cudaMalloc(&g_workspace, kWorkspaceBytes);
  return cuda_status == cudaSuccess ? CUBLAS_STATUS_SUCCESS
                                    : CUBLAS_STATUS_ALLOC_FAILED;
}

cublasStatus_t configure_custom_algorithm(
    cublasLtMatmulDesc_t operation,
    cublasLtMatrixLayout_t weight,
    cublasLtMatrixLayout_t activation,
    cublasLtMatrixLayout_t output,
    cudaDataType_t weight_type,
    cudaDataType_t activation_type,
    cudaDataType_t output_type,
    const CustomAlgoConfig& config,
    cublasLtMatmulAlgo_t* algorithm) {
#if CUDART_VERSION < 13000
  (void)operation;
  (void)weight;
  (void)activation;
  (void)output;
  (void)weight_type;
  (void)activation_type;
  (void)output_type;
  (void)config;
  (void)algorithm;
  return CUBLAS_STATUS_NOT_SUPPORTED;
#else
  // ID 66 is the native CUDA 13 SM110 family found by exhaustive CustomFind.
  // CUDA 13 encodes the non-split path as one K partition.
  cublasStatus_t status = cublasLtMatmulAlgoInit(
      g_lt, CUBLAS_COMPUTE_32F, CUDA_R_32F,
      weight_type, activation_type, output_type, output_type,
      66, algorithm);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  cublasLtMatmulTile_t tile =
      static_cast<cublasLtMatmulTile_t>(config.tile_id);
  cublasLtMatmulStages_t stages =
      static_cast<cublasLtMatmulStages_t>(config.stages_id);
  int split_k = 1;
  int reduction_scheme = CUBLASLT_REDUCTION_SCHEME_NONE;
  int swizzle = 0;
  int custom_option = config.custom_option;
  uint16_t cluster_shape = static_cast<uint16_t>(config.cluster_shape_id);
  uint16_t inner_shape = 0;

  status = cublasLtMatmulAlgoConfigSetAttribute(
      algorithm, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile, sizeof(tile));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatmulAlgoConfigSetAttribute(
      algorithm, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k,
      sizeof(split_k));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatmulAlgoConfigSetAttribute(
      algorithm, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
      &reduction_scheme, sizeof(reduction_scheme));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatmulAlgoConfigSetAttribute(
      algorithm, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
      &swizzle, sizeof(swizzle));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatmulAlgoConfigSetAttribute(
      algorithm, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
      &custom_option, sizeof(custom_option));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatmulAlgoConfigSetAttribute(
      algorithm, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatmulAlgoConfigSetAttribute(
      algorithm, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
      &cluster_shape, sizeof(cluster_shape));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatmulAlgoConfigSetAttribute(
      algorithm, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
      &inner_shape, sizeof(inner_shape));
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  cublasLtMatmulHeuristicResult_t checked{};
  status = cublasLtMatmulAlgoCheck(
      g_lt, operation, weight, activation, output, output,
      algorithm, &checked);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  if (checked.state != CUBLAS_STATUS_SUCCESS ||
      checked.workspaceSize > kWorkspaceBytes) {
    return CUBLAS_STATUS_NOT_SUPPORTED;
  }
  return CUBLAS_STATUS_SUCCESS;
#endif
}

cublasStatus_t make_plan(const ShapeKey& key, GemmPlan* plan) {
  // Row-major D=A@B is computed through the column-major identity
  // D^T=B^T@A^T. Memory is already in the required column-major layout:
  // weight [K,N] -> [N,K], activation [M,K] -> [K,M].
  cublasStatus_t status = cublasLtMatmulDescCreate(
      &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  cublasOperation_t op = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  status = cublasLtMatrixLayoutCreate(
      &plan->weight, CUDA_R_8F_E4M3, key.n, key.k, key.n);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatrixLayoutCreate(
      &plan->activation, CUDA_R_8F_E4M3, key.k, key.m, key.k);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatrixLayoutCreate(
      &plan->output, CUDA_R_16F, key.n, key.m, key.n);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  auto custom_it = g_fp8_custom_algorithms.find(key);
  if (custom_it != g_fp8_custom_algorithms.end()) {
    status = configure_custom_algorithm(
        plan->operation, plan->weight, plan->activation, plan->output,
        CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16F,
        custom_it->second, &plan->algorithm);
    if (status == CUBLAS_STATUS_SUCCESS) plan->has_algorithm = true;
    return status;
  }

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  size_t workspace_bytes = kWorkspaceBytes;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes, sizeof(workspace_bytes));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    return status;
  }
  int requested = 1;
  auto rank_it = g_cublaslt_ranks.find(key);
  if (rank_it != g_cublaslt_ranks.end()) requested = rank_it->second + 1;
  std::vector<cublasLtMatmulHeuristicResult_t> results(requested);
  int returned = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      g_lt, plan->operation, plan->weight, plan->activation, plan->output,
      plan->output, preference, requested, results.data(), &returned);
  cublasLtMatmulPreferenceDestroy(preference);
  int rank = requested - 1;
  if (status == CUBLAS_STATUS_SUCCESS && returned > rank &&
      results[rank].state == CUBLAS_STATUS_SUCCESS) {
    plan->algorithm = results[rank].algo;
    plan->has_algorithm = true;
  } else if (status == CUBLAS_STATUS_SUCCESS) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
  }
  return status;
}

cublasStatus_t make_bf16_plan(const ShapeKey& key, Bf16GemmPlan* plan) {
  // Row-major D=A@B is represented as the column-major identity
  // D^T=B^T@A^T, matching the existing cuBLAS physical GEMM contract.
  cublasStatus_t status = cublasLtMatmulDescCreate(
      &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  cublasOperation_t op = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  status = cublasLtMatrixLayoutCreate(
      &plan->weight, CUDA_R_16BF, key.n, key.k, key.n);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatrixLayoutCreate(
      &plan->activation, CUDA_R_16BF, key.k, key.m, key.k);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatrixLayoutCreate(
      &plan->output, CUDA_R_16BF, key.n, key.m, key.n);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  auto custom_it = g_bf16_custom_algorithms.find(key);
  if (custom_it != g_bf16_custom_algorithms.end()) {
    status = configure_custom_algorithm(
        plan->operation, plan->weight, plan->activation, plan->output,
        CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
        custom_it->second, &plan->algorithm);
    if (status == CUBLAS_STATUS_SUCCESS) plan->has_algorithm = true;
    return status;
  }

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  size_t workspace_bytes = kWorkspaceBytes;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes, sizeof(workspace_bytes));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    return status;
  }
  int requested = 1;
  auto rank_it = g_bf16_ranks.find(key);
  if (rank_it != g_bf16_ranks.end()) requested = rank_it->second + 1;
  std::vector<cublasLtMatmulHeuristicResult_t> results(requested);
  int returned = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      g_lt, plan->operation, plan->weight, plan->activation, plan->output,
      plan->output, preference, requested, results.data(), &returned);
  cublasLtMatmulPreferenceDestroy(preference);
  int rank = requested - 1;
  if (status == CUBLAS_STATUS_SUCCESS && returned > rank &&
      results[rank].state == CUBLAS_STATUS_SUCCESS) {
    plan->algorithm = results[rank].algo;
    plan->has_algorithm = true;
  } else if (status == CUBLAS_STATUS_SUCCESS) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
  }
  return status;
}

cublasStatus_t make_split_plan(
    const ShapeKey& key, bool bf16, GemmPlan* plan) {
  // Keep the physical packed [K,N] weight and [M,N] output allocations. Each
  // plan covers N/2 logical rows with the original full-N leading dimension;
  // the second launch advances only the base pointer by N/2 elements.
  if (key.n <= 0 || (key.n & 1) != 0)
    return CUBLAS_STATUS_INVALID_VALUE;
  const int half_n = key.n / 2;
  const cudaDataType_t input_type = bf16 ? CUDA_R_16BF : CUDA_R_8F_E4M3;
  const cudaDataType_t output_type = bf16 ? CUDA_R_16BF : CUDA_R_16F;
  auto& configs = bf16 ? g_bf16_split_custom_algorithms
                       : g_fp8_split_custom_algorithms;
  auto config_it = configs.find(key);
  if (config_it == configs.end()) return CUBLAS_STATUS_NOT_INITIALIZED;

  cublasStatus_t status = cublasLtMatmulDescCreate(
      &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  cublasOperation_t op = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatrixLayoutCreate(
      &plan->weight, input_type, half_n, key.k, key.n);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatrixLayoutCreate(
      &plan->activation, input_type, key.k, key.m, key.k);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatrixLayoutCreate(
      &plan->output, output_type, half_n, key.m, key.n);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = configure_custom_algorithm(
      plan->operation, plan->weight, plan->activation, plan->output,
      input_type, input_type, output_type, config_it->second,
      &plan->algorithm);
  if (status == CUBLAS_STATUS_SUCCESS) plan->has_algorithm = true;
  return status;
}

cublasStatus_t make_gelu_plan(
    const ShapeKey& key, FusedGeluPlan* plan, int heuristic_rank = 0) {
  cublasStatus_t status = cublasLtMatmulDescCreate(
      &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  cublasOperation_t op = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  cudaDataType_t bias_type = CUDA_R_16F;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
      &bias_type, sizeof(bias_type));
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  // Row-major [M,K] @ [K,N] is represented as column-major
  // [N,K] @ [K,M]. CUDA 13 requires an FP16 C layout even when beta is
  // zero, while D may be quantized directly to E4M3.
  status = cublasLtMatrixLayoutCreate(
      &plan->weight, CUDA_R_8F_E4M3, key.n, key.k, key.n);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatrixLayoutCreate(
      &plan->activation, CUDA_R_8F_E4M3, key.k, key.m, key.k);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatrixLayoutCreate(
      &plan->c, CUDA_R_16F, key.n, key.m, key.n);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatrixLayoutCreate(
      &plan->d, CUDA_R_8F_E4M3, key.n, key.m, key.n);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  cudaError_t cuda_status = cudaMalloc(
      &plan->c_buffer,
      static_cast<size_t>(key.m) * static_cast<size_t>(key.n) * sizeof(half));
  if (cuda_status != cudaSuccess) return CUBLAS_STATUS_ALLOC_FAILED;

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  size_t workspace_bytes = kWorkspaceBytes;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes, sizeof(workspace_bytes));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    return status;
  }
  const int requested = heuristic_rank + 1;
  std::vector<cublasLtMatmulHeuristicResult_t> results(requested);
  int returned = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      g_lt, plan->operation, plan->weight, plan->activation, plan->c,
      plan->d, preference, requested, results.data(), &returned);
  cublasLtMatmulPreferenceDestroy(preference);
  if (status == CUBLAS_STATUS_SUCCESS && returned > heuristic_rank &&
      results[heuristic_rank].state == CUBLAS_STATUS_SUCCESS) {
    plan->algorithm = results[heuristic_rank].algo;
    plan->has_algorithm = true;
  } else if (status == CUBLAS_STATUS_SUCCESS) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
  }
  return status;
}

cublasStatus_t make_residual_plan(
    const ShapeKey& key, ResidualPlan* plan, int heuristic_rank = 0,
    const CustomAlgoConfig* custom = nullptr) {
  cublasStatus_t status = cublasLtMatmulDescCreate(
      &plan->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  cublasOperation_t op = CUBLAS_OP_N;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
      &epilogue, sizeof(epilogue));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  cudaDataType_t bias_type = CUDA_R_16F;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
      &bias_type, sizeof(bias_type));
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  status = cublasLtMatrixLayoutCreate(
      &plan->weight, CUDA_R_8F_E4M3, key.n, key.k, key.n);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatrixLayoutCreate(
      &plan->activation, CUDA_R_8F_E4M3, key.k, key.m, key.k);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  status = cublasLtMatrixLayoutCreate(
      &plan->output, CUDA_R_16F, key.n, key.m, key.n);
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  cudaError_t cuda_status = cudaMalloc(
      &plan->zero_bias, static_cast<size_t>(key.n) * sizeof(half));
  if (cuda_status != cudaSuccess) return CUBLAS_STATUS_ALLOC_FAILED;
  cuda_status = cudaMemset(
      plan->zero_bias, 0, static_cast<size_t>(key.n) * sizeof(half));
  if (cuda_status != cudaSuccess) return CUBLAS_STATUS_EXECUTION_FAILED;

  const void* bias_pointer = plan->zero_bias;
  status = cublasLtMatmulDescSetAttribute(
      plan->operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &bias_pointer, sizeof(bias_pointer));
  if (status != CUBLAS_STATUS_SUCCESS) return status;

  if (custom != nullptr) {
    status = configure_custom_algorithm(
        plan->operation, plan->weight, plan->activation, plan->output,
        CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16F,
        *custom, &plan->algorithm);
    if (status == CUBLAS_STATUS_SUCCESS) plan->has_algorithm = true;
    return status;
  }

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  size_t workspace_bytes = kWorkspaceBytes;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes, sizeof(workspace_bytes));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    return status;
  }
  int requested = heuristic_rank + 1;
  std::vector<cublasLtMatmulHeuristicResult_t> results(requested);
  int returned = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      g_lt, plan->operation, plan->weight, plan->activation, plan->output,
      plan->output, preference, requested, results.data(), &returned);
  cublasLtMatmulPreferenceDestroy(preference);
  if (status == CUBLAS_STATUS_SUCCESS && returned > heuristic_rank &&
      results[heuristic_rank].state == CUBLAS_STATUS_SUCCESS) {
    plan->algorithm = results[heuristic_rank].algo;
    plan->has_algorithm = true;
  } else if (status == CUBLAS_STATUS_SUCCESS) {
    status = CUBLAS_STATUS_NOT_SUPPORTED;
  }
  return status;
}

cudaError_t get_device_scale(float value, float** scale) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  auto it = g_device_scales.find(bits);
  if (it != g_device_scales.end()) {
    *scale = it->second;
    return cudaSuccess;
  }
  float* device_value = nullptr;
  cudaError_t status = cudaMalloc(&device_value, sizeof(float));
  if (status != cudaSuccess) return status;
  status = cudaMemcpy(device_value, &value, sizeof(float), cudaMemcpyHostToDevice);
  if (status != cudaSuccess) {
    cudaFree(device_value);
    return status;
  }
  g_device_scales.emplace(bits, device_value);
  *scale = device_value;
  return cudaSuccess;
}

cublasStatus_t prepare_gemm_plan(const ShapeKey& key) {
  cublasStatus_t status = initialize();
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  if (g_plans.find(key) != g_plans.end()) return CUBLAS_STATUS_SUCCESS;
  GemmPlan plan;
  status = make_plan(key, &plan);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_plan(&plan);
    return status;
  }
  g_plans.emplace(key, plan);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t prepare_bf16_gemm_plan(const ShapeKey& key) {
  cublasStatus_t status = initialize();
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  if (g_bf16_plans.find(key) != g_bf16_plans.end())
    return CUBLAS_STATUS_SUCCESS;
  Bf16GemmPlan plan;
  status = make_bf16_plan(key, &plan);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_bf16_plan(&plan);
    return status;
  }
  g_bf16_plans.emplace(key, plan);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t prepare_split_gemm_plan(const ShapeKey& key, bool bf16) {
  cublasStatus_t status = initialize();
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  auto& plans = bf16 ? g_bf16_split_plans : g_fp8_split_plans;
  if (plans.find(key) != plans.end()) return CUBLAS_STATUS_SUCCESS;
  GemmPlan plan;
  status = make_split_plan(key, bf16, &plan);
  if (status != CUBLAS_STATUS_SUCCESS) {
    destroy_plan(&plan);
    return status;
  }
  plans.emplace(key, plan);
  return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t prepare_gelu_plan(const GeluKey& key, float output_scale) {
  cublasStatus_t status = initialize();
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  auto it = g_gelu_plans.find(key);
  if (it == g_gelu_plans.end()) {
    FusedGeluPlan plan;
    int heuristic_rank = 0;
    auto rank_it = g_fp8_gelu_ranks.find(key.shape);
    if (rank_it != g_fp8_gelu_ranks.end()) heuristic_rank = rank_it->second;
    status = make_gelu_plan(key.shape, &plan, heuristic_rank);
    if (status != CUBLAS_STATUS_SUCCESS) {
      destroy_gelu_plan(&plan);
      return status;
    }
    it = g_gelu_plans.emplace(key, plan).first;
  }
  float* device_scale = nullptr;
  cudaError_t cuda_status = get_device_scale(1.0f / output_scale, &device_scale);
  if (cuda_status != cudaSuccess) return CUBLAS_STATUS_ALLOC_FAILED;
  status = cublasLtMatmulDescSetAttribute(
      it->second.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &key.bias, sizeof(key.bias));
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  return cublasLtMatmulDescSetAttribute(
      it->second.operation, CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
      &device_scale, sizeof(device_scale));
}

cublasStatus_t prepare_residual_plan(const ResidualKey& key) {
  cublasStatus_t status = initialize();
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  auto it = g_residual_plans.find(key);
  if (it == g_residual_plans.end()) {
    ResidualPlan plan;
    int heuristic_rank = 0;
    auto rank_it = g_fp8_residual_ranks.find(key.shape);
    if (rank_it != g_fp8_residual_ranks.end()) heuristic_rank = rank_it->second;
    const CustomAlgoConfig* custom = nullptr;
    auto custom_it = g_fp8_residual_custom_algorithms.find(key.shape);
    if (custom_it != g_fp8_residual_custom_algorithms.end()) {
      custom = &custom_it->second;
    }
    status = make_residual_plan(key.shape, &plan, heuristic_rank, custom);
    if (status != CUBLAS_STATUS_SUCCESS) {
      destroy_residual_plan(&plan);
      return status;
    }
    it = g_residual_plans.emplace(key, plan).first;
  }
  const void* bias_pointer =
      key.bias != nullptr ? key.bias : it->second.zero_bias;
  return cublasLtMatmulDescSetAttribute(
      it->second.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &bias_pointer, sizeof(bias_pointer));
}

cublasStatus_t prepare_bias_plan(const ResidualKey& key) {
  cublasStatus_t status = initialize();
  if (status != CUBLAS_STATUS_SUCCESS) return status;
  auto it = g_bias_plans.find(key);
  if (it == g_bias_plans.end()) {
    ResidualPlan plan;
    int heuristic_rank = 0;
    auto rank_it = g_fp8_bias_ranks.find(key.shape);
    if (rank_it != g_fp8_bias_ranks.end()) heuristic_rank = rank_it->second;
    const CustomAlgoConfig* custom = nullptr;
    auto custom_it = g_fp8_bias_custom_algorithms.find(key.shape);
    if (custom_it != g_fp8_bias_custom_algorithms.end()) {
      custom = &custom_it->second;
    }
    status = make_residual_plan(key.shape, &plan, heuristic_rank, custom);
    if (status != CUBLAS_STATUS_SUCCESS) {
      destroy_residual_plan(&plan);
      return status;
    }
    it = g_bias_plans.emplace(key, plan).first;
  }
  return cublasLtMatmulDescSetAttribute(
      it->second.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
      &key.bias, sizeof(key.bias));
}

}  // namespace

extern "C" int apxinf_static_prepare_bf16_gemm(
    int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  return static_cast<int>(prepare_bf16_gemm_plan(ShapeKey{m, n, k}));
}

extern "C" int apxinf_static_prepare_bf16_gemm_split(
    int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0 || (n & 1) != 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  return static_cast<int>(prepare_split_gemm_plan(ShapeKey{m, n, k}, true));
}

extern "C" int apxinf_static_set_cublaslt_bf16_gemm_heuristic(
    int m, int n, int k, int heuristic_rank) {
  if (m <= 0 || n <= 0 || k <= 0 || heuristic_rank < 0 ||
      heuristic_rank >= 64) return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  ShapeKey key{m, n, k};
  g_bf16_custom_algorithms.erase(key);
  g_bf16_split_custom_algorithms.erase(key);
  g_bf16_ranks[key] = heuristic_rank;
  invalidate_bf16_shape_plans(key);
  return static_cast<int>(CUBLAS_STATUS_SUCCESS);
}

extern "C" int apxinf_static_set_cublaslt_bf16_gemm_custom(
    int m, int n, int k, int tile_id, int custom_option,
    int stages_id, int cluster_shape_id) {
  if (m <= 0 || n <= 0 || k <= 0 || tile_id <= 0 || tile_id >= 1024 ||
      custom_option < 0 || custom_option >= 8 || stages_id <= 0 ||
      stages_id >= 64 || cluster_shape_id < 0 || cluster_shape_id >= 64) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  ShapeKey key{m, n, k};
  g_bf16_ranks.erase(key);
  g_bf16_split_custom_algorithms.erase(key);
  g_bf16_custom_algorithms[key] =
      CustomAlgoConfig{tile_id, custom_option, stages_id, cluster_shape_id};
  invalidate_bf16_shape_plans(key);
  cublasStatus_t status = prepare_bf16_gemm_plan(key);
  if (status != CUBLAS_STATUS_SUCCESS) {
    g_bf16_custom_algorithms.erase(key);
  }
  return static_cast<int>(status);
}

extern "C" int apxinf_static_set_cublaslt_bf16_gemm_split_custom(
    int m, int n, int k, int tile_id, int custom_option,
    int stages_id, int cluster_shape_id) {
  if (m <= 0 || n <= 0 || k <= 0 || (n & 1) != 0 ||
      tile_id <= 0 || tile_id >= 1024 || custom_option < 0 ||
      custom_option >= 8 || stages_id <= 0 || stages_id >= 64 ||
      cluster_shape_id < 0 || cluster_shape_id >= 64) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  ShapeKey key{m, n, k};
  g_bf16_ranks.erase(key);
  g_bf16_custom_algorithms.erase(key);
  g_bf16_split_custom_algorithms[key] =
      CustomAlgoConfig{tile_id, custom_option, stages_id, cluster_shape_id};
  invalidate_bf16_shape_plans(key);
  cublasStatus_t status = prepare_split_gemm_plan(key, true);
  if (status != CUBLAS_STATUS_SUCCESS) {
    g_bf16_split_custom_algorithms.erase(key);
  }
  return static_cast<int>(status);
}

extern "C" int apxinf_static_bf16_gemm(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || output == nullptr ||
      m <= 0 || n <= 0 || k <= 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  ShapeKey key{m, n, k};
  auto it = g_bf16_plans.find(key);
  if (it == g_bf16_plans.end())
    return static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED);

  const float beta = 0.0f;
  Bf16GemmPlan& plan = it->second;
  return static_cast<int>(cublasLtMatmul(
      g_lt, plan.operation, &alpha,
      weight, plan.weight, activation, plan.activation,
      &beta, output, plan.output, output, plan.output,
      plan.has_algorithm ? &plan.algorithm : nullptr,
      g_workspace, kWorkspaceBytes, stream));
}

extern "C" int apxinf_static_bf16_gemm_split(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || output == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || (n & 1) != 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  ShapeKey key{m, n, k};
  auto it = g_bf16_split_plans.find(key);
  if (it == g_bf16_split_plans.end())
    return static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED);
  const float beta = 0.0f;
  GemmPlan& plan = it->second;
  cublasStatus_t status = cublasLtMatmul(
      g_lt, plan.operation, &alpha,
      weight, plan.weight, activation, plan.activation,
      &beta, output, plan.output, output, plan.output,
      &plan.algorithm, g_workspace, kWorkspaceBytes, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  const size_t offset_bytes = static_cast<size_t>(n / 2) * sizeof(uint16_t);
  const auto* weight_second =
      static_cast<const uint8_t*>(weight) + offset_bytes;
  auto* output_second = static_cast<uint8_t*>(output) + offset_bytes;
  return static_cast<int>(cublasLtMatmul(
      g_lt, plan.operation, &alpha,
      weight_second, plan.weight, activation, plan.activation,
      &beta, output_second, plan.output, output_second, plan.output,
      &plan.algorithm, g_workspace, kWorkspaceBytes, stream));
}

extern "C" int apxinf_static_bf16_gemm_split_first(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || output == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || (n & 1) != 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  ShapeKey key{m, n, k};
  auto it = g_bf16_split_plans.find(key);
  if (it == g_bf16_split_plans.end())
    return static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED);
  const float beta = 0.0f;
  GemmPlan& plan = it->second;
  return static_cast<int>(cublasLtMatmul(
      g_lt, plan.operation, &alpha,
      weight, plan.weight, activation, plan.activation,
      &beta, output, plan.output, output, plan.output,
      &plan.algorithm, g_workspace, kWorkspaceBytes, stream));
}

extern "C" int apxinf_static_autotune_cublaslt_bf16_gemm(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, int max_algorithms,
    int warmup_iterations, int benchmark_iterations,
    int* did_tune, int* returned_algorithms, int* best_rank,
    float* default_ms, float* best_ms, cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || output == nullptr ||
      did_tune == nullptr || returned_algorithms == nullptr ||
      best_rank == nullptr || default_ms == nullptr || best_ms == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || max_algorithms <= 0 ||
      max_algorithms > 64 || warmup_iterations < 0 ||
      benchmark_iterations <= 0) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }

  ShapeKey key{m, n, k};
  cublasStatus_t status = prepare_bf16_gemm_plan(key);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  Bf16GemmPlan& plan = g_bf16_plans.find(key)->second;
  if (plan.autotuned) {
    *did_tune = 0;
    *returned_algorithms = plan.returned_algorithms;
    *best_rank = plan.best_rank;
    *default_ms = plan.default_ms;
    *best_ms = plan.best_ms;
    return static_cast<int>(CUBLAS_STATUS_SUCCESS);
  }

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  size_t workspace_bytes = kWorkspaceBytes;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes, sizeof(workspace_bytes));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    return static_cast<int>(status);
  }
  std::vector<cublasLtMatmulHeuristicResult_t> results(max_algorithms);
  int returned = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      g_lt, plan.operation, plan.weight, plan.activation, plan.output,
      plan.output, preference, max_algorithms, results.data(), &returned);
  cublasLtMatmulPreferenceDestroy(preference);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  if (returned == 0)
    return static_cast<int>(CUBLAS_STATUS_NOT_SUPPORTED);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (cudaEventCreate(&start) != cudaSuccess ||
      cudaEventCreate(&stop) != cudaSuccess) {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    return static_cast<int>(CUBLAS_STATUS_ALLOC_FAILED);
  }

  int device = 0;
  int l2_cache_bytes = 0;
  void* l2_eviction_buffer = nullptr;
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaDeviceGetAttribute(&l2_cache_bytes, cudaDevAttrL2CacheSize, device) !=
          cudaSuccess ||
      l2_cache_bytes <= 0) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(CUBLAS_STATUS_INTERNAL_ERROR);
  }
  const size_t l2_eviction_bytes =
      (static_cast<size_t>(l2_cache_bytes) * 4u + 255u) & ~size_t{255u};
  if (cudaMalloc(&l2_eviction_buffer, l2_eviction_bytes) != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(CUBLAS_STATUS_ALLOC_FAILED);
  }

  const float beta = 0.0f;
  int selected_rank = -1;
  float selected_ms = 1.0e30f;
  float rank_zero_ms = -1.0f;
  uint32_t eviction_seed = 0;
  for (int rank = 0; rank < returned; ++rank) {
    if (results[rank].state != CUBLAS_STATUS_SUCCESS) continue;
    bool valid = true;
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
      if (apxinf_static_evict_l2(l2_eviction_buffer, l2_eviction_bytes,
                                ++eviction_seed, stream) != cudaSuccess) {
        valid = false;
        break;
      }
      status = cublasLtMatmul(
          g_lt, plan.operation, &alpha,
          weight, plan.weight, activation, plan.activation,
          &beta, output, plan.output, output, plan.output,
          &results[rank].algo, g_workspace, kWorkspaceBytes, stream);
      if (status != CUBLAS_STATUS_SUCCESS) {
        valid = false;
        break;
      }
    }
    if (!valid || cudaStreamSynchronize(stream) != cudaSuccess) continue;
    float elapsed = 0.0f;
    for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
      if (apxinf_static_evict_l2(l2_eviction_buffer, l2_eviction_bytes,
                                ++eviction_seed, stream) != cudaSuccess ||
          cudaEventRecord(start, stream) != cudaSuccess) {
        valid = false;
        break;
      }
      status = cublasLtMatmul(
          g_lt, plan.operation, &alpha,
          weight, plan.weight, activation, plan.activation,
          &beta, output, plan.output, output, plan.output,
          &results[rank].algo, g_workspace, kWorkspaceBytes, stream);
      if (status != CUBLAS_STATUS_SUCCESS) {
        valid = false;
        break;
      }
      if (cudaEventRecord(stop, stream) != cudaSuccess ||
          cudaEventSynchronize(stop) != cudaSuccess) {
        valid = false;
        break;
      }
      float iteration_ms = 0.0f;
      if (cudaEventElapsedTime(&iteration_ms, start, stop) != cudaSuccess) {
        valid = false;
        break;
      }
      elapsed += iteration_ms;
    }
    if (!valid) continue;
    elapsed /= static_cast<float>(benchmark_iterations);
    if (rank == 0) rank_zero_ms = elapsed;
    if (elapsed < selected_ms) {
      selected_ms = elapsed;
      selected_rank = rank;
    }
  }
  cudaFree(l2_eviction_buffer);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (selected_rank < 0)
    return static_cast<int>(CUBLAS_STATUS_NOT_SUPPORTED);

  plan.algorithm = results[selected_rank].algo;
  plan.has_algorithm = true;
  plan.autotuned = true;
  plan.returned_algorithms = returned;
  plan.best_rank = selected_rank;
  plan.default_ms = rank_zero_ms;
  plan.best_ms = selected_ms;
  *did_tune = 1;
  *returned_algorithms = returned;
  *best_rank = selected_rank;
  *default_ms = rank_zero_ms;
  *best_ms = selected_ms;
  return static_cast<int>(CUBLAS_STATUS_SUCCESS);
}

extern "C" cudaError_t apxinf_static_native_fp8_supported(
    int device, int* supported) {
  if (device < 0 || supported == nullptr) return cudaErrorInvalidValue;
  int major = 0;
  int minor = 0;
  cudaError_t status = cudaDeviceGetAttribute(
      &major, cudaDevAttrComputeCapabilityMajor, device);
  if (status != cudaSuccess) return status;
  status = cudaDeviceGetAttribute(
      &minor, cudaDevAttrComputeCapabilityMinor, device);
  if (status != cudaSuccess) return status;

  // Ada SM89, Hopper SM90, Blackwell SM100/SM110, and later devices expose
  // native E4M3 Tensor Core GEMM. Ampere-family SM80/SM86/SM87 does not.
  *supported = (major > 8 || (major == 8 && minor >= 9)) ? 1 : 0;
  return cudaSuccess;
}

extern "C" int apxinf_static_prepare_fp8_gemm_f16(int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  return static_cast<int>(prepare_gemm_plan(ShapeKey{m, n, k}));
}

extern "C" int apxinf_static_prepare_fp8_gemm_split_f16(
    int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0 || (n & 1) != 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  return static_cast<int>(prepare_split_gemm_plan(ShapeKey{m, n, k}, false));
}

extern "C" int apxinf_static_prepare_fp8_gemm_bias_gelu_e4m3(
    const void* bias, int m, int n, int k, float output_scale) {
  if (bias == nullptr || m <= 0 || n <= 0 || k <= 0 ||
      !(output_scale > 0.0f)) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  uint32_t scale_bits = 0;
  std::memcpy(&scale_bits, &output_scale, sizeof(scale_bits));
  return static_cast<int>(prepare_gelu_plan(
      GeluKey{ShapeKey{m, n, k}, bias, scale_bits}, output_scale));
}

extern "C" int apxinf_static_prepare_fp8_gemm_bias_residual_f16(
    const void* bias, int m, int n, int k) {
  if (m <= 0 || n <= 0 || k <= 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  return static_cast<int>(
      prepare_residual_plan(ResidualKey{ShapeKey{m, n, k}, bias}));
}

extern "C" int apxinf_static_prepare_fp8_gemm_bias_f16(
    const void* bias, int m, int n, int k) {
  if (bias == nullptr || m <= 0 || n <= 0 || k <= 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  return static_cast<int>(
      prepare_bias_plan(ResidualKey{ShapeKey{m, n, k}, bias}));
}

extern "C" int apxinf_static_fp8_gemm_f16(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || output == nullptr ||
      m <= 0 || n <= 0 || k <= 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  ShapeKey key{m, n, k};
  auto it = g_plans.find(key);
  if (it == g_plans.end())
    return static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED);

  const float beta = 0.0f;
  GemmPlan& plan = it->second;
  cublasStatus_t status = cublasLtMatmul(
      g_lt, plan.operation, &alpha,
      weight, plan.weight, activation, plan.activation,
      &beta, output, plan.output, output, plan.output,
      plan.has_algorithm ? &plan.algorithm : nullptr,
      g_workspace, kWorkspaceBytes, stream);
  return static_cast<int>(status);
}

extern "C" int apxinf_static_fp8_gemm_split_f16(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || output == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || (n & 1) != 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  ShapeKey key{m, n, k};
  auto it = g_fp8_split_plans.find(key);
  if (it == g_fp8_split_plans.end())
    return static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED);
  const float beta = 0.0f;
  GemmPlan& plan = it->second;
  cublasStatus_t status = cublasLtMatmul(
      g_lt, plan.operation, &alpha,
      weight, plan.weight, activation, plan.activation,
      &beta, output, plan.output, output, plan.output,
      &plan.algorithm, g_workspace, kWorkspaceBytes, stream);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  const size_t weight_offset = static_cast<size_t>(n / 2);
  const size_t output_offset = weight_offset * sizeof(uint16_t);
  const auto* weight_second =
      static_cast<const uint8_t*>(weight) + weight_offset;
  auto* output_second = static_cast<uint8_t*>(output) + output_offset;
  return static_cast<int>(cublasLtMatmul(
      g_lt, plan.operation, &alpha,
      weight_second, plan.weight, activation, plan.activation,
      &beta, output_second, plan.output, output_second, plan.output,
      &plan.algorithm, g_workspace, kWorkspaceBytes, stream));
}

extern "C" int apxinf_static_fp8_gemm_split_first_f16(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || output == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || (n & 1) != 0)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  ShapeKey key{m, n, k};
  auto it = g_fp8_split_plans.find(key);
  if (it == g_fp8_split_plans.end())
    return static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED);
  const float beta = 0.0f;
  GemmPlan& plan = it->second;
  return static_cast<int>(cublasLtMatmul(
      g_lt, plan.operation, &alpha,
      weight, plan.weight, activation, plan.activation,
      &beta, output, plan.output, output, plan.output,
      &plan.algorithm, g_workspace, kWorkspaceBytes, stream));
}

extern "C" int apxinf_static_fp8_gemm_bias_gelu_e4m3(
    const void* activation, const void* weight, const void* bias, void* output,
    int m, int n, int k, float alpha, float output_scale,
    cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || m <= 0 || n <= 0 || k <= 0 ||
      !(output_scale > 0.0f)) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  uint32_t scale_bits = 0;
  std::memcpy(&scale_bits, &output_scale, sizeof(scale_bits));
  GeluKey key{ShapeKey{m, n, k}, bias, scale_bits};
  auto it = g_gelu_plans.find(key);
  if (it == g_gelu_plans.end())
    return static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED);
  FusedGeluPlan& plan = it->second;

  const float beta = 0.0f;
  cublasStatus_t status = cublasLtMatmul(
      g_lt, plan.operation, &alpha,
      weight, plan.weight, activation, plan.activation,
      &beta, plan.c_buffer, plan.c, output, plan.d,
      plan.has_algorithm ? &plan.algorithm : nullptr,
      g_workspace, kWorkspaceBytes, stream);
  return static_cast<int>(status);
}

extern "C" int apxinf_static_fp8_gemm_bias_residual_f16(
    const void* activation, const void* weight, const void* bias,
    const void* residual, void* output, int m, int n, int k, float alpha,
    cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || residual == nullptr ||
      output == nullptr || m <= 0 || n <= 0 || k <= 0) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  ResidualKey key{ShapeKey{m, n, k}, bias};
  auto it = g_residual_plans.find(key);
  if (it == g_residual_plans.end())
    return static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED);

  ResidualPlan& plan = it->second;
  const float beta = 1.0f;
  cublasStatus_t status = cublasLtMatmul(
      g_lt, plan.operation, &alpha,
      weight, plan.weight, activation, plan.activation,
      &beta, residual, plan.output, output, plan.output,
      plan.has_algorithm ? &plan.algorithm : nullptr,
      g_workspace, kWorkspaceBytes, stream);
  return static_cast<int>(status);
}

extern "C" int apxinf_static_fp8_gemm_bias_f16(
    const void* activation, const void* weight, const void* bias, void* output,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || m <= 0 || n <= 0 || k <= 0) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  ResidualKey key{ShapeKey{m, n, k}, bias};
  auto it = g_bias_plans.find(key);
  if (it == g_bias_plans.end())
    return static_cast<int>(CUBLAS_STATUS_NOT_INITIALIZED);

  ResidualPlan& plan = it->second;
  const float beta = 0.0f;
  return static_cast<int>(cublasLtMatmul(
      g_lt, plan.operation, &alpha,
      weight, plan.weight, activation, plan.activation,
      &beta, output, plan.output, output, plan.output,
      plan.has_algorithm ? &plan.algorithm : nullptr,
      g_workspace, kWorkspaceBytes, stream));
}

extern "C" int apxinf_static_set_cublaslt_gemm_heuristic(
    int m, int n, int k, int heuristic_rank) {
  if (m <= 0 || n <= 0 || k <= 0 || heuristic_rank < 0 ||
      heuristic_rank >= 64) return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  ShapeKey key{m, n, k};
  g_fp8_custom_algorithms.erase(key);
  g_cublaslt_ranks[key] = heuristic_rank;
  invalidate_fp8_shape_plans(key);
  return static_cast<int>(CUBLAS_STATUS_SUCCESS);
}

extern "C" int apxinf_static_set_cublaslt_fp8_fused_heuristic(
    int m, int n, int k, int epilogue, int heuristic_rank) {
  if (m <= 0 || n <= 0 || k <= 0 || heuristic_rank < 0 ||
      heuristic_rank >= 64) return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  ShapeKey key{m, n, k};
  switch (epilogue) {
    case 1:
      g_fp8_bias_custom_algorithms.erase(key);
      g_fp8_bias_ranks[key] = heuristic_rank;
      invalidate_fp8_bias_plans(key);
      break;
    case 2:
      g_fp8_gelu_ranks[key] = heuristic_rank;
      invalidate_fp8_gelu_plans(key);
      break;
    case 3:
      g_fp8_residual_custom_algorithms.erase(key);
      g_fp8_residual_ranks[key] = heuristic_rank;
      invalidate_fp8_residual_plans(key);
      break;
    default:
      return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  return static_cast<int>(CUBLAS_STATUS_SUCCESS);
}

extern "C" int apxinf_static_set_cublaslt_fp8_gemm_custom(
    int m, int n, int k, int tile_id, int custom_option,
    int stages_id, int cluster_shape_id) {
  if (m <= 0 || n <= 0 || k <= 0 || tile_id <= 0 || tile_id >= 1024 ||
      custom_option < 0 || custom_option >= 8 || stages_id <= 0 ||
      stages_id >= 64 || cluster_shape_id < 0 || cluster_shape_id >= 64) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  ShapeKey key{m, n, k};
  g_cublaslt_ranks.erase(key);
  g_fp8_custom_algorithms[key] =
      CustomAlgoConfig{tile_id, custom_option, stages_id, cluster_shape_id};
  invalidate_fp8_shape_plans(key);
  cublasStatus_t status = prepare_gemm_plan(key);
  if (status != CUBLAS_STATUS_SUCCESS) {
    g_fp8_custom_algorithms.erase(key);
  }
  return static_cast<int>(status);
}

extern "C" int apxinf_static_set_cublaslt_fp8_gemm_split_custom(
    int m, int n, int k, int tile_id, int custom_option,
    int stages_id, int cluster_shape_id) {
  if (m <= 0 || n <= 0 || k <= 0 || (n & 1) != 0 ||
      tile_id <= 0 || tile_id >= 1024 || custom_option < 0 ||
      custom_option >= 8 || stages_id <= 0 || stages_id >= 64 ||
      cluster_shape_id < 0 || cluster_shape_id >= 64) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  ShapeKey key{m, n, k};
  g_fp8_split_custom_algorithms[key] =
      CustomAlgoConfig{tile_id, custom_option, stages_id, cluster_shape_id};
  invalidate_fp8_split_plans(key);
  cublasStatus_t status = prepare_split_gemm_plan(key, false);
  if (status != CUBLAS_STATUS_SUCCESS) {
    g_fp8_split_custom_algorithms.erase(key);
  }
  return static_cast<int>(status);
}

extern "C" int apxinf_static_set_cublaslt_fp8_gemm_bias_custom(
    int m, int n, int k, int epilogue, int tile_id, int custom_option,
    int stages_id, int cluster_shape_id) {
  if (m <= 0 || n <= 0 || k <= 0 || tile_id <= 0 || tile_id >= 1024 ||
      custom_option < 0 || custom_option >= 8 || stages_id <= 0 ||
      stages_id >= 64 || cluster_shape_id < 0 || cluster_shape_id >= 64) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  cublasStatus_t status = initialize();
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  ShapeKey key{m, n, k};
  auto* ranks = epilogue == 1 ? &g_fp8_bias_ranks
                              : epilogue == 3 ? &g_fp8_residual_ranks : nullptr;
  auto* configs = epilogue == 1 ? &g_fp8_bias_custom_algorithms
                                : epilogue == 3 ? &g_fp8_residual_custom_algorithms : nullptr;
  if (ranks == nullptr || configs == nullptr)
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  ranks->erase(key);
  (*configs)[key] =
      CustomAlgoConfig{tile_id, custom_option, stages_id, cluster_shape_id};
  if (epilogue == 1) {
    invalidate_fp8_bias_plans(key);
  } else {
    invalidate_fp8_residual_plans(key);
  }

  // Validate the fully specified algorithm against the exact fused-bias
  // descriptor during startup. Unsupported configs fail closed before graph
  // capture; the real bias pointer is bound when its cached plan is prepared.
  ResidualPlan validation_plan;
  status = make_residual_plan(key, &validation_plan, 0, &configs->at(key));
  destroy_residual_plan(&validation_plan);
  if (status != CUBLAS_STATUS_SUCCESS) {
    configs->erase(key);
  }
  return static_cast<int>(status);
}

extern "C" int apxinf_static_autotune_cublaslt_fp8_gemm_f16(
    const void* activation, const void* weight, void* output,
    void* l2_eviction_buffer, size_t l2_eviction_bytes,
    int m, int n, int k, float alpha, int max_algorithms,
    int warmup_iterations, int benchmark_iterations,
    int* returned_algorithms, float* milliseconds, cudaStream_t stream) {
  if (activation == nullptr || weight == nullptr || output == nullptr ||
      l2_eviction_buffer == nullptr || l2_eviction_bytes == 0 ||
      returned_algorithms == nullptr || milliseconds == nullptr ||
      m <= 0 || n <= 0 || k <= 0 || max_algorithms <= 0 ||
      max_algorithms > 64 || warmup_iterations < 0 ||
      benchmark_iterations <= 0) {
    return static_cast<int>(CUBLAS_STATUS_INVALID_VALUE);
  }
  cublasStatus_t status = initialize();
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);

  ShapeKey key{m, n, k};
  auto it = g_plans.find(key);
  if (it == g_plans.end()) {
    GemmPlan plan;
    status = make_plan(key, &plan);
    if (status != CUBLAS_STATUS_SUCCESS) {
      destroy_plan(&plan);
      return static_cast<int>(status);
    }
    it = g_plans.emplace(key, plan).first;
  }
  GemmPlan& plan = it->second;

  cublasLtMatmulPreference_t preference = nullptr;
  status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  size_t workspace_bytes = kWorkspaceBytes;
  status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      &workspace_bytes, sizeof(workspace_bytes));
  if (status != CUBLAS_STATUS_SUCCESS) {
    cublasLtMatmulPreferenceDestroy(preference);
    return static_cast<int>(status);
  }
  std::vector<cublasLtMatmulHeuristicResult_t> results(max_algorithms);
  int returned = 0;
  status = cublasLtMatmulAlgoGetHeuristic(
      g_lt, plan.operation, plan.weight, plan.activation, plan.output,
      plan.output, preference, max_algorithms, results.data(), &returned);
  cublasLtMatmulPreferenceDestroy(preference);
  if (status != CUBLAS_STATUS_SUCCESS) return static_cast<int>(status);
  *returned_algorithms = returned;
  for (int i = 0; i < max_algorithms; ++i) milliseconds[i] = -1.0f;
  if (returned == 0) return static_cast<int>(CUBLAS_STATUS_NOT_SUPPORTED);

  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  if (cudaEventCreate(&start) != cudaSuccess || cudaEventCreate(&stop) != cudaSuccess) {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    return static_cast<int>(CUBLAS_STATUS_ALLOC_FAILED);
  }
  const float beta = 0.0f;
  int best_rank = -1;
  float best_ms = 1.0e30f;
  for (int rank = 0; rank < returned; ++rank) {
    if (results[rank].state != CUBLAS_STATUS_SUCCESS) continue;
    bool valid = true;
    for (int iteration = 0; iteration < warmup_iterations; ++iteration) {
      if (apxinf_static_evict_l2(l2_eviction_buffer, l2_eviction_bytes,
                                static_cast<uint32_t>(rank * 4099 + iteration),
                                stream) != cudaSuccess) {
        valid = false;
        break;
      }
      status = cublasLtMatmul(
          g_lt, plan.operation, &alpha,
          weight, plan.weight, activation, plan.activation,
          &beta, output, plan.output, output, plan.output,
          &results[rank].algo, g_workspace, kWorkspaceBytes, stream);
      if (status != CUBLAS_STATUS_SUCCESS) {
        valid = false;
        break;
      }
    }
    if (!valid || cudaStreamSynchronize(stream) != cudaSuccess) continue;
    float elapsed = 0.0f;
    for (int iteration = 0; iteration < benchmark_iterations; ++iteration) {
      if (apxinf_static_evict_l2(l2_eviction_buffer, l2_eviction_bytes,
                                static_cast<uint32_t>(rank * 65537 + iteration),
                                stream) != cudaSuccess ||
          cudaEventRecord(start, stream) != cudaSuccess) {
        valid = false;
        break;
      }
      status = cublasLtMatmul(
          g_lt, plan.operation, &alpha,
          weight, plan.weight, activation, plan.activation,
          &beta, output, plan.output, output, plan.output,
          &results[rank].algo, g_workspace, kWorkspaceBytes, stream);
      if (status != CUBLAS_STATUS_SUCCESS) {
        valid = false;
        break;
      }
      if (cudaEventRecord(stop, stream) != cudaSuccess ||
          cudaEventSynchronize(stop) != cudaSuccess) {
        valid = false;
        break;
      }
      float iteration_ms = 0.0f;
      if (cudaEventElapsedTime(&iteration_ms, start, stop) != cudaSuccess) {
        valid = false;
        break;
      }
      elapsed += iteration_ms;
    }
    if (!valid) continue;
    elapsed /= static_cast<float>(benchmark_iterations);
    milliseconds[rank] = elapsed;
    if (elapsed < best_ms) {
      best_ms = elapsed;
      best_rank = rank;
    }
  }
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  if (best_rank < 0) return static_cast<int>(CUBLAS_STATUS_NOT_SUPPORTED);
  plan.algorithm = results[best_rank].algo;
  plan.has_algorithm = true;
  g_cublaslt_ranks[key] = best_rank;
  return static_cast<int>(CUBLAS_STATUS_SUCCESS);
}

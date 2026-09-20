#pragma once

#include <stdint.h>

typedef struct {
  uint64_t workspace_limit;
  uint32_t online_tune;
  uint32_t allow_fallback;
  uint32_t graph_safe;
  uint32_t deterministic;
  const char* cache_dir;
} apxinf_tuning_policy_t;

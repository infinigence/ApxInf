#pragma once

#include "status.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define APXINF_DEVICE_INFO_VERSION 1u

typedef struct {
  uint32_t version;
  uint32_t compute_major;
  uint32_t compute_minor;
  uint32_t multiprocessor_count;
  char device_name[256];
} apxinf_device_info_t;

apxinf_status_t apxinf_runtime_create(int32_t device, apxinf_runtime_t* runtime);
apxinf_status_t apxinf_runtime_device_info(apxinf_runtime_t runtime,
                                           apxinf_device_info_t* info);
void apxinf_runtime_destroy(apxinf_runtime_t runtime);
const char* apxinf_last_error(void);

#ifdef __cplusplus
}
#endif

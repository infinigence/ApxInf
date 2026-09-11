#pragma once

#include "status.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_runtime_create(int32_t device, apxinf_runtime_t* runtime);
void apxinf_runtime_destroy(apxinf_runtime_t runtime);
const char* apxinf_last_error(void);

#ifdef __cplusplus
}
#endif

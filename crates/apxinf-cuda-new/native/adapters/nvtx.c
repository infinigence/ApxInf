#include <nvtx3/nvToolsExt.h>
int apxinf_nvtx_range_push(const char* name) { return nvtxRangePushA(name); }
int apxinf_nvtx_range_pop(void) { return nvtxRangePop(); }

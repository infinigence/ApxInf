// os_signpost_interval_begin/end are macros over a compile-time format
// builder, so Rust cannot call them directly. The format is a literal here and
// the caller's name is only ever an argument to it.
#include <dispatch/dispatch.h>
#include <os/signpost.h>
#include <stddef.h>
#include <stdint.h>

static os_log_t trace_log_handle;

static void create_log(void *context) {
    (void)context;
    trace_log_handle = os_log_create("com.apxinf", OS_LOG_CATEGORY_POINTS_OF_INTEREST);
}

static os_log_t trace_log(void) {
    static dispatch_once_t once;
    dispatch_once_f(&once, NULL, create_log);
    return trace_log_handle;
}

// Returns OS_SIGNPOST_ID_NULL when signposts are disabled; the end macro
// ignores that id, so the pair stays balanced either way.
uint64_t apxinf_signpost_begin(const char *name, int len) {
    os_log_t handle = trace_log();
    os_signpost_id_t id = os_signpost_id_generate(handle);
    os_signpost_interval_begin(handle, id, "range", "%{public}.*s", len, name);
    return id;
}

void apxinf_signpost_end(uint64_t id) {
    os_signpost_interval_end(trace_log(), id, "range");
}

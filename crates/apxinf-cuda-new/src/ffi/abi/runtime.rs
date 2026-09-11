use super::types::Runtime;

unsafe extern "C" {
    pub(crate) fn apxinf_runtime_create(device: i32, runtime: *mut Runtime) -> i32;
    pub(crate) fn apxinf_runtime_destroy(runtime: Runtime);
}

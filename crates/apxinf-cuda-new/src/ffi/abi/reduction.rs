use super::types::{CudaStream,Runtime};
use std::ffi::c_void;
#[repr(C)]
pub(crate) struct Spec {pub version:u32,pub reserved:u32,pub input_elements:i64,pub rows:i64,pub cols:i64}
#[repr(C)]
pub(crate) struct Bindings {pub input:*const f32,pub indices:*const u32,pub output:*mut f32,pub stream:CudaStream}
unsafe extern "C" {
 pub(crate) fn apxinf_indexed_mean_prepare(runtime:Runtime,spec:*const Spec,bindings:*const Bindings,execution:*mut *mut c_void)->i32;
 pub(crate) fn apxinf_indexed_mean_enqueue(execution:*mut c_void)->i32;
 pub(crate) fn apxinf_indexed_mean_destroy(execution:*mut c_void);
}

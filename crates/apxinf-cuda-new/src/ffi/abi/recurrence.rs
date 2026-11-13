use super::types::{CudaStream,Runtime};
use std::ffi::c_void;
#[repr(C)]
pub(crate) struct Spec {pub version:u32,pub reset:u32,pub count:i64,pub a:f64,pub b:f64,pub c:f64,pub d:f64}
#[repr(C)]
pub(crate) struct Bindings {pub input:*const f32,pub output:*mut f32,pub state:*mut f64,pub previous:*mut f64,pub stream:CudaStream}
unsafe extern "C" {
 pub(crate) fn apxinf_recurrence_prepare(runtime:Runtime,spec:*const Spec,bindings:*const Bindings,execution:*mut *mut c_void)->i32;
 pub(crate) fn apxinf_recurrence_enqueue(execution:*mut c_void)->i32;
 pub(crate) fn apxinf_recurrence_destroy(execution:*mut c_void);
}

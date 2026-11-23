use std::ffi::c_void;

pub(crate) use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_sampling_argmax(
        logits: *const c_void,
        index: *mut c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
}

mod fixed_launchers {
    use std::ffi::c_void;

    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        pub(crate) fn apxinf_sampling_suppress_logits_bf16(
            logits: *mut c_void,
            ids: *const u32,
            count: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;

use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_model_embedding_gather(
        table: *const c_void,
        ids: *const c_void,
        output: *mut c_void,
        tokens: i64,
        hidden: i64,
        vocab: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_model_argmax_bf16(
        logits: *const c_void,
        index: *mut c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
}

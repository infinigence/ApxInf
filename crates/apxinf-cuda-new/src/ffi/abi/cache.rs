use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_cache_append_bf16(
        new_data: *const c_void,
        cache: *mut c_void,
        n_kv_heads: i32,
        head_dim: i32,
        max_seq_len: i32,
        seq_len: i32,
        append_len: i32,
        stream: CudaStream,
    ) -> i32;
}

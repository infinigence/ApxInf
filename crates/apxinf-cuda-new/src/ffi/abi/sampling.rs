use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    /// Greedy selection. The stateful samplers in the same adapter use the
    /// `apxinf_sampling_*` prefix and are declared in `ffi/raw/sampling.rs`.
    pub(crate) fn apxinf_sampling_argmax(
        logits: *const c_void,
        index: *mut c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
}

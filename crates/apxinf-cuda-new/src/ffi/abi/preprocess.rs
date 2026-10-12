use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_preprocess_temporal_merged_patches_bf16(
        images: *const c_void,
        patches: *mut c_void,
        views: i32,
        image_size: i32,
        patch_size: i32,
        temporal_patch_size: i32,
        merge_size: i32,
        nhwc: i32,
        rescale_factor: f64,
        mean0: f32,
        mean1: f32,
        mean2: f32,
        std0: f32,
        std1: f32,
        std2: f32,
        stream: CudaStream,
    ) -> i32;
}

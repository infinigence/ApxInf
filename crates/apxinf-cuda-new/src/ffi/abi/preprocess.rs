use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_preprocess_temporal_merged_patches_bf16(
        images: *const c_void,
        patches: *mut c_void,
        views: i32,
        image_height: i32,
        image_width: i32,
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

mod fixed_launchers {
    use std::ffi::c_void;

    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        pub(crate) fn apxinf_preprocess_fourier_features_bf16(
            waypoints: *const c_void,
            freqs: *const c_void,
            out: *mut c_void,
            rows: i32,
            point_dim: i32,
            num_features: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_preprocess_concat7_cols_bf16(
            s0: *const c_void,
            s1: *const c_void,
            s2: *const c_void,
            s3: *const c_void,
            s4: *const c_void,
            s5: *const c_void,
            s6: *const c_void,
            dst: *mut c_void,
            rows: i32,
            cols: i32,
            broadcast_mask: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_preprocess_flow_update_f32(
            w: *mut c_void,
            endpoint: *const c_void,
            remaining: f32,
            step: f32,
            count: i64,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_preprocess_vision_position_bf16(
            table: *const c_void,
            embeddings: *mut c_void,
            position_ids: *mut c_void,
            source_side: i32,
            height: i32,
            width: i32,
            hidden: i32,
            merge_size: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_preprocess_rgb_u8_to_temporal2_merge2_rect_bf16(
            rgb: *const c_void,
            patches: *mut c_void,
            lut: *const c_void,
            grid_h: i32,
            grid_w: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_preprocess_pillow_bicubic_u8_axis(
            input: *const c_void,
            output: *mut c_void,
            input_offsets: *const c_void,
            output_offsets: *const c_void,
            bounds: *const c_void,
            weights: *const c_void,
            ksize: i32,
            in_w: i32,
            in_h: i32,
            out_w: i32,
            out_h: i32,
            batch: i32,
            horizontal: bool,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_preprocess_pixel_shuffle_4_bf16(
            input: *const c_void,
            output: *mut c_void,
            views: i32,
            tokens_per_view: i32,
            width: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_preprocess_pixel_shuffle_4_f16(
            input: *const c_void,
            output: *mut c_void,
            views: i32,
            tokens_per_view: i32,
            width: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;

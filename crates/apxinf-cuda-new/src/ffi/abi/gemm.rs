use std::ffi::{c_char, c_void};

pub(crate) use super::types::Policy;
use super::types::{CudaStream, Runtime};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub a_dtype: u32,
    pub b_dtype: u32,
    pub accumulation_dtype: u32,
    pub output_dtype: u32,
    pub quantization: u32,
    pub b_is_immutable: u32,
    pub a_alignment: u32,
    pub b_alignment: u32,
    pub bias_alignment: u32,
    pub a_scales_alignment: u32,
    pub b_scales_alignment: u32,
    pub output_alignment: u32,
    pub a_block_scales_alignment: u32,
    pub b_block_scales_alignment: u32,
    /// Elements per block scale along K; zero when the contract has none.
    pub sf_vec_size: u32,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    /// Structural predicates only: the scale values live in [`Bindings`].
    pub alpha_is_unit: u32,
    pub output_scale_is_unit: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub a: *const c_void,
    pub b: *const c_void,
    pub b_version: u64,
    pub b_is_immutable: u32,
    pub bias: *const c_void,
    pub a_scales: *const f32,
    pub b_scales: *const f32,
    pub a_block_scales: *const c_void,
    pub b_block_scales: *const c_void,
    pub output: *mut c_void,
    pub stream: CudaStream,
    pub alpha: f32,
    pub output_scale: f32,
}

pub(crate) type Execution = *mut c_void;

unsafe extern "C" {
    pub(crate) fn apxinf_gemm_prepare(
        runtime: Runtime,
        spec: *const Spec,
        policy: *const Policy,
        bindings: *const Bindings,
        execution: *mut Execution,
    ) -> i32;
    pub(crate) fn apxinf_gemm_enqueue(execution: Execution) -> i32;
    pub(crate) fn apxinf_gemm_destroy(execution: Execution);
    pub(crate) fn apxinf_gemm_nvfp4_scale_buffer_bytes(
        rows: i64,
        k: i64,
        sf_vec_size: u32,
    ) -> u64;
    pub(crate) fn apxinf_gemm_nvfp4_pack_block_scales(
        source_row_major: *const c_void,
        destination: *mut c_void,
        rows: i64,
        k: i64,
        sf_vec_size: u32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gemm_nvfp4_quantize_rms_norm(
        source_bf16: *const c_void,
        norm_weight: *const c_void,
        destination_packed: *mut c_void,
        destination_scales: *mut c_void,
        rows: i64,
        k: i64,
        sf_vec_size: u32,
        epsilon: f32,
        input_scale: f32,
        row_major_scales: i32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gemm_nvfp4_quantize_swiglu(
        source_bf16: *const c_void,
        destination_packed: *mut c_void,
        destination_scales: *mut c_void,
        rows: i64,
        k: i64,
        sf_vec_size: u32,
        input_scale: f32,
        row_major_scales: i32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gemm_nvfp4_quantize_activation(
        source_bf16: *const c_void,
        destination_packed: *mut c_void,
        destination_scales: *mut c_void,
        rows: i64,
        k: i64,
        sf_vec_size: u32,
        input_scale: f32,
        row_major_scales: i32,
        stream: CudaStream,
    ) -> i32;
    #[cfg(test)]
    pub(crate) fn apxinf_gemm_summary(execution: Execution) -> *const c_char;
    #[cfg(test)]
    pub(crate) fn apxinf_gemm_execution_weight_prepack_count(execution: Execution) -> u64;

    #[cfg(test)]
    pub(crate) fn apxinf_gemm_test_validate_candidates(
        runtime: Runtime,
        spec: *const Spec,
        policy: *const Policy,
        bindings: *const Bindings,
        expected_output: *const f32,
        expected_output_len: u64,
    ) -> i32;

    #[cfg(test)]
    pub(crate) fn apxinf_gemm_test_seed_recipe(
        spec: *const Spec,
        policy: *const Policy,
        device: i32,
        provider_id: u32,
        implementation_id: u32,
        implementation_version: u32,
        configuration: i32,
    ) -> i32;
}

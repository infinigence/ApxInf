//! Raw bindings for CUDA token sampling and normal generation.
//!
//! These live under `raw/` rather than `abi/` because they return
//! `cudaError_t`, the same convention as the CUDA runtime bindings beside them.
//! `abi/` is the `apxinf_status_t` surface — the greedy `apxinf_sampling_argmax`
//! is declared there for exactly that reason, even though it is the same
//! family and the same adapter file. The split tracks the error convention, so
//! moving these without converting them would make `abi/` inconsistent instead.

use std::ffi::c_void;

use super::cuda_runtime::{cudaError_t, cudaStream_t};

extern "C" {
    pub fn apxinf_sampling_workspace_sizes(
        vocab_size: u32,
        sort_bytes: *mut usize,
        scan_bytes: *mut usize,
    ) -> cudaError_t;

    #[allow(clippy::too_many_arguments)]
    pub fn apxinf_sampling_sample_token(
        logits: *const c_void,
        dtype: i32,
        vocab_size: u32,
        counts: *mut u32,
        repetition: f32,
        frequency: f32,
        presence: f32,
        selection: i32,
        temperature: f32,
        top_k: u32,
        top_p: f32,
        seed: u64,
        sequence: u64,
        draw: u64,
        return_logprob: u32,
        adjusted: *mut f32,
        token_ids: *mut u32,
        sorted_logits: *mut f32,
        sorted_tokens: *mut u32,
        weights: *mut f32,
        cdf: *mut f32,
        partial_values: *mut f32,
        partial_tokens: *mut u32,
        partial_count: u32,
        sort_workspace: *mut c_void,
        sort_workspace_bytes: usize,
        scan_workspace: *mut c_void,
        scan_workspace_bytes: usize,
        output: *mut c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;

    pub fn apxinf_sampling_fill_standard_normal(
        output: *mut c_void,
        dtype: i32,
        count: u64,
        seed: u64,
        sequence: u64,
        draw: u64,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

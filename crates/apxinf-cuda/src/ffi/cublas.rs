//! Raw cuBLAS bindings.

use std::ffi::c_void;

use super::cuda::cudaStream_t;

// ── cuBLAS types ────────────────────────────────────────────────────

pub type cublasHandle_t = *mut c_void;
pub type cublasStatus_t = i32;

pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;

#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
}

/// CUDA data type for cublasGemmEx.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum cudaDataType_t {
    CUDA_R_32F = 0,
    CUDA_R_16F = 2,
    CUDA_R_8I = 3,
    CUDA_R_32I = 10,
    CUDA_R_16BF = 14,
}

/// Compute type for cublasGemmEx.
#[repr(i32)]
#[derive(Debug, Clone, Copy)]
pub enum cublasComputeType_t {
    CUBLAS_COMPUTE_32F = 68,
    CUBLAS_COMPUTE_32I = 72,
}

extern "C" {
    pub fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t;
    pub fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t;
    pub fn cublasSetStream_v2(handle: cublasHandle_t, stream: cudaStream_t) -> cublasStatus_t;
    pub fn cublasGetVersion_v2(handle: cublasHandle_t, version: *mut i32) -> cublasStatus_t;

    /// Single-precision GEMM: C = alpha * op(A) * op(B) + beta * C
    pub fn cublasSgemm_v2(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const f32,
        a: *const c_void,
        lda: i32,
        b: *const c_void,
        ldb: i32,
        beta: *const f32,
        c: *mut c_void,
        ldc: i32,
    ) -> cublasStatus_t;

    /// Mixed-precision GEMM (used for bf16).
    pub fn cublasGemmEx(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        atype: cudaDataType_t,
        lda: i32,
        b: *const c_void,
        btype: cudaDataType_t,
        ldb: i32,
        beta: *const c_void,
        c: *mut c_void,
        ctype: cudaDataType_t,
        ldc: i32,
        computeType: cublasComputeType_t,
        algo: i32, // CUBLAS_GEMM_DEFAULT = -1
    ) -> cublasStatus_t;

    /// Strided batched GEMM: batch_count independent GEMMs with fixed strides.
    pub fn cublasGemmStridedBatchedEx(
        handle: cublasHandle_t,
        transa: cublasOperation_t,
        transb: cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: *const c_void,
        a: *const c_void,
        atype: cudaDataType_t,
        lda: i32,
        stridea: i64,
        b: *const c_void,
        btype: cudaDataType_t,
        ldb: i32,
        strideb: i64,
        beta: *const c_void,
        c: *mut c_void,
        ctype: cudaDataType_t,
        ldc: i32,
        stridec: i64,
        batchcount: i32,
        computetype: cublasComputeType_t,
        algo: i32, // CUBLAS_GEMM_DEFAULT = -1
    ) -> cublasStatus_t;
    pub fn apxinf_static_cublas_mqa_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        query_tokens: i32,
        key_tokens: i32,
        heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
    pub fn apxinf_static_cublas_mqa_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        query_tokens: i32,
        key_tokens: i32,
        heads: i32,
        head_dim: i32,
        stream: cudaStream_t,
    ) -> cublasStatus_t;
}

/// Check a cuBLAS call.
pub fn check_cublas(status: cublasStatus_t) -> std::result::Result<(), String> {
    if status == CUBLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(format!("cuBLAS error: status {status}"))
    }
}

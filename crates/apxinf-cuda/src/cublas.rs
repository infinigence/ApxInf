//! cuBLAS handle wrapper with GEMM operations.

use std::ffi::c_void;

use apxinf_core::DType;

use crate::buffer::CudaBuffer;
use crate::ffi;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CublasTranspose {
    None,
    Transpose,
}

impl CublasTranspose {
    fn raw(self) -> ffi::cublasOperation_t {
        match self {
            Self::None => ffi::cublasOperation_t::CUBLAS_OP_N,
            Self::Transpose => ffi::cublasOperation_t::CUBLAS_OP_T,
        }
    }
}

/// Owns a cuBLAS handle.
pub struct CublasHandle {
    handle: ffi::cublasHandle_t,
}

unsafe impl Send for CublasHandle {}
unsafe impl Sync for CublasHandle {}

impl CublasHandle {
    pub fn new() -> Result<Self, String> {
        let mut handle: ffi::cublasHandle_t = std::ptr::null_mut();
        unsafe {
            ffi::check_cublas(ffi::cublasCreate_v2(&mut handle))?;
        }
        Ok(Self { handle })
    }

    /// Set the CUDA stream for this handle.
    pub fn set_stream(&self, stream: &crate::CudaStream) -> Result<(), String> {
        unsafe { ffi::check_cublas(ffi::cublasSetStream_v2(self.handle, stream.handle())) }
    }

    /// Return the linked cuBLAS library version encoded as `CUBLAS_VERSION`.
    pub fn version(&self) -> Result<i32, String> {
        let mut version = 0;
        unsafe {
            ffi::check_cublas(ffi::cublasGetVersion_v2(self.handle, &mut version))?;
        }
        Ok(version)
    }

    /// Perform GEMM: C = alpha * A @ B + beta * C
    ///
    /// A: [m, k], B: [k, n], C: [m, n]
    /// All matrices are in row-major order (we tell cuBLAS they're column-major
    /// and swap A/B, which is the standard trick).
    pub fn gemm(
        &self,
        dtype: DType,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaBuffer, // [m, k] row-major
        b: &CudaBuffer, // [k, n] row-major
        beta: f32,
        c: &CudaBuffer, // [m, n] row-major
    ) -> Result<(), String> {
        // cuBLAS is column-major. For row-major C = A @ B, we compute
        // C^T = B^T @ A^T in column-major, which means:
        //   cublasSgemm(N, N, n, m, k, alpha, B, n, A, k, beta, C, n)
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;

        match dtype {
            DType::F32 => unsafe {
                ffi::check_cublas(ffi::cublasSgemm_v2(
                    self.handle,
                    ffi::cublasOperation_t::CUBLAS_OP_N,
                    ffi::cublasOperation_t::CUBLAS_OP_N,
                    n_i,
                    m_i,
                    k_i,
                    &alpha,
                    b.ptr(),
                    n_i,
                    a.ptr(),
                    k_i,
                    &beta,
                    c.ptr() as *mut c_void,
                    n_i,
                ))
            },
            DType::F16 | DType::BF16 => {
                let cuda_type = if dtype == DType::F16 {
                    ffi::cudaDataType_t::CUDA_R_16F
                } else {
                    ffi::cudaDataType_t::CUDA_R_16BF
                };
                let alpha_bytes = alpha.to_ne_bytes();
                let beta_bytes = beta.to_ne_bytes();
                unsafe {
                    ffi::check_cublas(ffi::cublasGemmEx(
                        self.handle,
                        ffi::cublasOperation_t::CUBLAS_OP_N,
                        ffi::cublasOperation_t::CUBLAS_OP_N,
                        n_i,
                        m_i,
                        k_i,
                        alpha_bytes.as_ptr() as *const c_void,
                        b.ptr(),
                        cuda_type,
                        n_i,
                        a.ptr(),
                        cuda_type,
                        k_i,
                        beta_bytes.as_ptr() as *const c_void,
                        c.ptr() as *mut c_void,
                        cuda_type,
                        n_i,
                        ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        -1, // CUBLAS_GEMM_DEFAULT
                    ))
                }
            }
            DType::F8E4M3 => Err("use kernels::gemm::fp8 for FP8 operands".into()),
        }
    }

    /// Strided batched GEMM: batch_count independent GEMMs with fixed strides.
    ///
    /// A: [batch_count, m, k] with element stride `stride_a`, B: [batch_count, k, n] with element stride `stride_b`,
    /// C: [batch_count, m, n] with stride `stride_c`.
    /// All matrices are row-major. Uses the same column-major swap trick as `gemm()`.
    pub fn batched_gemm(
        &self,
        dtype: DType,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaBuffer,
        stride_a: i64,
        b: &CudaBuffer,
        stride_b: i64,
        beta: f32,
        c: &CudaBuffer,
        stride_c: i64,
        batch_count: i32,
    ) -> Result<(), String> {
        // Same row-major trick: C = A @ B becomes C^T = B^T @ A^T in column-major
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;

        match dtype {
            DType::F32 => unsafe {
                ffi::check_cublas(ffi::cublasGemmStridedBatchedEx(
                    self.handle,
                    ffi::cublasOperation_t::CUBLAS_OP_N,
                    ffi::cublasOperation_t::CUBLAS_OP_N,
                    n_i, // swapped m
                    m_i, // swapped n
                    k_i,
                    &alpha as *const f32 as *const c_void,
                    b.ptr(), // swapped A
                    ffi::cudaDataType_t::CUDA_R_32F,
                    n_i,      // lda = swapped n
                    stride_b, // stride_a = stride_b (swapped)
                    a.ptr(),  // swapped B
                    ffi::cudaDataType_t::CUDA_R_32F,
                    k_i,      // ldb = swapped k
                    stride_a, // stride_b = stride_a (swapped)
                    &beta as *const f32 as *const c_void,
                    c.ptr() as *mut c_void,
                    ffi::cudaDataType_t::CUDA_R_32F,
                    n_i, // ldc = swapped n
                    stride_c,
                    batch_count,
                    ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    -1, // CUBLAS_GEMM_DEFAULT
                ))
            },
            DType::F16 | DType::BF16 => {
                let cuda_type = if dtype == DType::F16 {
                    ffi::cudaDataType_t::CUDA_R_16F
                } else {
                    ffi::cudaDataType_t::CUDA_R_16BF
                };
                let alpha_bytes = alpha.to_ne_bytes();
                let beta_bytes = beta.to_ne_bytes();
                unsafe {
                    ffi::check_cublas(ffi::cublasGemmStridedBatchedEx(
                        self.handle,
                        ffi::cublasOperation_t::CUBLAS_OP_N,
                        ffi::cublasOperation_t::CUBLAS_OP_N,
                        n_i,
                        m_i,
                        k_i,
                        alpha_bytes.as_ptr() as *const c_void,
                        b.ptr(),
                        cuda_type,
                        n_i,
                        stride_b,
                        a.ptr(),
                        cuda_type,
                        k_i,
                        stride_a,
                        beta_bytes.as_ptr() as *const c_void,
                        c.ptr() as *mut c_void,
                        cuda_type,
                        n_i,
                        stride_c,
                        batch_count,
                        ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        -1,
                    ))
                }
            }
            DType::F8E4M3 => Err("use kernels::gemm::fp8 for FP8 operands".into()),
        }
    }

    /// Strided batched GEMM with explicit transpose and row-stride contracts.
    /// Strides between batch matrices are expressed in elements; a zero
    /// stride broadcasts one matrix across every batch member.
    #[allow(clippy::too_many_arguments)]
    pub fn batched_gemm_ex(
        &self,
        dtype: DType,
        transa: CublasTranspose,
        transb: CublasTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaBuffer,
        lda: i32,
        stride_a: i64,
        b: &CudaBuffer,
        ldb: i32,
        stride_b: i64,
        beta: f32,
        c: &CudaBuffer,
        ldc: i32,
        stride_c: i64,
        batch_count: i32,
    ) -> Result<(), String> {
        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;
        let transa = transa.raw();
        let transb = transb.raw();

        match dtype {
            DType::F32 => unsafe {
                ffi::check_cublas(ffi::cublasGemmStridedBatchedEx(
                    self.handle,
                    transb,
                    transa,
                    n_i,
                    m_i,
                    k_i,
                    &alpha as *const f32 as *const c_void,
                    b.ptr(),
                    ffi::cudaDataType_t::CUDA_R_32F,
                    ldb,
                    stride_b,
                    a.ptr(),
                    ffi::cudaDataType_t::CUDA_R_32F,
                    lda,
                    stride_a,
                    &beta as *const f32 as *const c_void,
                    c.ptr() as *mut c_void,
                    ffi::cudaDataType_t::CUDA_R_32F,
                    ldc,
                    stride_c,
                    batch_count,
                    ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    -1,
                ))
            },
            DType::F16 | DType::BF16 => {
                let cuda_type = if dtype == DType::F16 {
                    ffi::cudaDataType_t::CUDA_R_16F
                } else {
                    ffi::cudaDataType_t::CUDA_R_16BF
                };
                let alpha_bytes = alpha.to_ne_bytes();
                let beta_bytes = beta.to_ne_bytes();
                unsafe {
                    ffi::check_cublas(ffi::cublasGemmStridedBatchedEx(
                        self.handle,
                        transb,
                        transa,
                        n_i,
                        m_i,
                        k_i,
                        alpha_bytes.as_ptr() as *const c_void,
                        b.ptr(),
                        cuda_type,
                        ldb,
                        stride_b,
                        a.ptr(),
                        cuda_type,
                        lda,
                        stride_a,
                        beta_bytes.as_ptr() as *const c_void,
                        c.ptr() as *mut c_void,
                        cuda_type,
                        ldc,
                        stride_c,
                        batch_count,
                        ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        -1,
                    ))
                }
            }
            DType::F8E4M3 => Err("use kernels::gemm::fp8 for FP8 operands".into()),
        }
    }

    /// Extended GEMM with transpose control, using cublasGemmEx.
    ///
    /// Computes C = alpha * op(A) @ op(B) + beta * C
    /// where op(X) is controlled by transa/transb.
    ///
    /// All matrices are row-major. The column-major swap is applied internally.
    /// A is [m, k] if transa=N, [k, m] if transa=T
    /// B is [k, n] if transb=N, [n, k] if transb=T
    /// C is [m, n]
    pub fn gemm_ex(
        &self,
        dtype: DType,
        transa: CublasTranspose,
        transb: CublasTranspose,
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        a: &CudaBuffer,
        lda: i32, // row stride of A in row-major
        b: &CudaBuffer,
        ldb: i32, // row stride of B in row-major
        beta: f32,
        c: &CudaBuffer,
        ldc: i32, // row stride of C in row-major (= n)
    ) -> Result<(), String> {
        // Row-major C = op(A) @ op(B) becomes column-major:
        //   C^T_col = op(B)_col^T @ op(A)_col^T
        // But using the identity M_row = (M_col)^T:
        //   C_col = op(B)_col @ op(A)_col
        //   (because (XY)^T = Y^T X^T and M_row = M_col^T)
        //
        // So in column-major we compute C_col = op_cm(A_cm) @ op_cm(B_cm)
        // with A_cm = B, B_cm = A, and the transposes swap:
        //   transa_cm = transb (B's row-major transpose becomes A_cm's column-major op)
        //   transb_cm = transa (A's row-major transpose becomes B_cm's column-major op)
        //
        // Dimensions: C_col has shape [n, m] so m_cm = n, n_cm = m, k_cm = k
        // Leading dimensions: lda_cm = ldb, ldb_cm = lda, ldc_cm = ldc

        let m_i = m as i32;
        let n_i = n as i32;
        let k_i = k as i32;

        let transa = transa.raw();
        let transb = transb.raw();
        match dtype {
            DType::F32 => unsafe {
                ffi::check_cublas(ffi::cublasGemmEx(
                    self.handle,
                    transb, // transa_cm = transb (swapped)
                    transa, // transb_cm = transa (swapped)
                    n_i,    // m_cm = n
                    m_i,    // n_cm = m
                    k_i,
                    &alpha as *const f32 as *const c_void,
                    b.ptr(), // A_cm = B (swapped)
                    ffi::cudaDataType_t::CUDA_R_32F,
                    ldb,     // lda_cm = ldb (swapped)
                    a.ptr(), // B_cm = A (swapped)
                    ffi::cudaDataType_t::CUDA_R_32F,
                    lda, // ldb_cm = lda (swapped)
                    &beta as *const f32 as *const c_void,
                    c.ptr() as *mut c_void,
                    ffi::cudaDataType_t::CUDA_R_32F,
                    ldc, // ldc_cm = ldc
                    ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    -1,
                ))
            },
            DType::F16 | DType::BF16 => {
                let cuda_type = if dtype == DType::F16 {
                    ffi::cudaDataType_t::CUDA_R_16F
                } else {
                    ffi::cudaDataType_t::CUDA_R_16BF
                };
                let alpha_bytes = alpha.to_ne_bytes();
                let beta_bytes = beta.to_ne_bytes();
                unsafe {
                    ffi::check_cublas(ffi::cublasGemmEx(
                        self.handle,
                        transb,
                        transa,
                        n_i,
                        m_i,
                        k_i,
                        alpha_bytes.as_ptr() as *const c_void,
                        b.ptr(),
                        cuda_type,
                        ldb,
                        a.ptr(),
                        cuda_type,
                        lda,
                        beta_bytes.as_ptr() as *const c_void,
                        c.ptr() as *mut c_void,
                        cuda_type,
                        ldc,
                        ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        -1,
                    ))
                }
            }
            DType::F8E4M3 => Err("use kernels::gemm::fp8 for FP8 operands".into()),
        }
    }

    pub fn raw(&self) -> ffi::cublasHandle_t {
        self.handle
    }

    /// Row-major W8A8 GEMM with INT32 accumulation.
    ///
    /// `activation` is physical `[m,k]` row-major. `weight_output_major` is
    /// physical `[n,k]` row-major (equivalently a `[k,n]` column-major view),
    /// matching the INT8 kernel contract. `output` is physical `[m,n]`
    /// row-major. Scaling and conversion to BF16 are left to the following
    /// stream-ordered kernel.
    pub fn gemm_int8_i32(
        &self,
        m: usize,
        n: usize,
        k: usize,
        activation: &CudaBuffer,
        weight_output_major: &CudaBuffer,
        output: &CudaBuffer,
    ) -> Result<(), String> {
        if activation.len() < m * k
            || weight_output_major.len() < n * k
            || output.len() < m * n * std::mem::size_of::<i32>()
        {
            return Err("INT8 GEMM buffer is smaller than its matrix shape".into());
        }
        let alpha = 1i32;
        let beta = 0i32;
        unsafe {
            ffi::check_cublas(ffi::cublasGemmEx(
                self.handle,
                ffi::cublasOperation_t::CUBLAS_OP_T,
                ffi::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                &alpha as *const i32 as *const c_void,
                weight_output_major.ptr(),
                ffi::cudaDataType_t::CUDA_R_8I,
                k as i32,
                activation.ptr(),
                ffi::cudaDataType_t::CUDA_R_8I,
                k as i32,
                &beta as *const i32 as *const c_void,
                output.ptr(),
                ffi::cudaDataType_t::CUDA_R_32I,
                n as i32,
                ffi::cublasComputeType_t::CUBLAS_COMPUTE_32I,
                99, // CUBLAS_GEMM_DEFAULT_TENSOR_OP
            ))
        }
    }
}

impl Drop for CublasHandle {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                let _ = ffi::cublasDestroy_v2(self.handle);
            }
        }
    }
}

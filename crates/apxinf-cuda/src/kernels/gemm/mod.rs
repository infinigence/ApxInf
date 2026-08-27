mod bf16;
mod fp8;
mod w8a8;

use apxinf_core::{DType, Device, Error, Result, Tensor};

use super::contracts::{checked_bytes, require_buffers, require_finite};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::cublas::CublasTranspose;
use crate::workspace::uninitialized_buffer;

pub use bf16::{
    autotune_cublaslt_bf16, gemm_bf16 as bf16, gemm_bf16_geglu_fused as bf16_geglu_fused,
    install_cublaslt_bf16_tactics, write_bf16, Bf16AutotuneResult, Bf16CublasLtTactic,
};
#[cfg(apxinf_cutlass_gemm)]
pub use fp8::autotune_cutlass_gemm_f16 as autotune_cutlass_fp8;
#[cfg(test)]
pub(crate) use fp8::prepare_cublaslt_fp8_gemm;
pub use fp8::{
    autotune_cublaslt_gemm_f16 as autotune_cublaslt_fp8, cold_l2_tuning_metadata, exact_fp8_tactic,
    gemm_fp8 as fp8, gemm_fp8_geglu_fused as fp8_geglu_fused, install_tuning_db,
    install_tuning_dbs, native_fp8_gemm_supported as native_fp8_supported, ColdL2TuningMetadata,
    CublasLtAlgorithmTiming, CutlassTacticTiming, Fp8WeightView,
};
#[cfg(test)]
pub(crate) use w8a8::gemm_w8a8_with_preference;
pub use w8a8::{gemm_w8a8 as w8a8, W8A8Layout, W8A8ScaleMode, W8A8WeightView};

pub fn matmul(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if activation.dtype() != weight.dtype() {
        return Err(Error::DTypeMismatch {
            expected: activation.dtype(),
            got: weight.dtype(),
        });
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device || weight.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: if activation.device() != expected_device {
                activation.device()
            } else {
                weight.device()
            },
        });
    }
    if activation.dtype() == DType::BF16 {
        return bf16::gemm_bf16(ctx, activation, weight);
    }
    let output_shape = activation.shape().matmul_shape(weight.shape())?;
    let m = activation.shape().dims()[activation.ndim() - 2];
    let k = activation.shape().dims()[activation.ndim() - 1];
    let n = weight.shape().dims()[weight.ndim() - 1];
    let output = uninitialized_buffer(
        ctx,
        output_shape.numel() * activation.dtype().size_in_bytes(),
    )?;
    let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    ctx.cublas()
        .gemm(
            activation.dtype(),
            m,
            n,
            k,
            1.0,
            &activation_buffer,
            &weight_buffer,
            0.0,
            &output,
        )
        .map_err(Error::Cuda)?;
    Ok(output.into_tensor(output_shape, activation.dtype()))
}

/// Row-major `A[M,K] @ B[K,N]` into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn write(
    ctx: &CudaContext,
    dtype: DType,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &CudaBuffer,
    b: &CudaBuffer,
    beta: f32,
    output: &CudaBuffer,
) -> Result<()> {
    require_finite("GEMM", &[alpha, beta])?;
    require_buffers(
        ctx,
        "GEMM",
        &[
            ("A", a, checked_bytes(dtype, &[m, k], "GEMM")?),
            ("B", b, checked_bytes(dtype, &[k, n], "GEMM")?),
            ("output", output, checked_bytes(dtype, &[m, n], "GEMM")?),
        ],
    )?;
    if dtype == DType::BF16 && beta == 0.0 {
        return bf16::write_bf16(ctx, a, b, output, m, n, k, alpha);
    }
    ctx.cublas()
        .gemm(dtype, m, n, k, alpha, a, b, beta, output)
        .map_err(apxinf_core::Error::Cuda)
}

/// GEMM with explicit transpose and row-stride contracts.
#[allow(clippy::too_many_arguments)]
pub fn write_ex(
    ctx: &CudaContext,
    dtype: DType,
    trans_a: CublasTranspose,
    trans_b: CublasTranspose,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a: &CudaBuffer,
    lda: i32,
    b: &CudaBuffer,
    ldb: i32,
    beta: f32,
    output: &CudaBuffer,
    ldc: i32,
) -> Result<()> {
    require_finite("GEMM_EX", &[alpha, beta])?;
    let (a_rows, a_cols) = match trans_a {
        CublasTranspose::None => (m, k),
        CublasTranspose::Transpose => (k, m),
    };
    let (b_rows, b_cols) = match trans_b {
        CublasTranspose::None => (k, n),
        CublasTranspose::Transpose => (n, k),
    };
    if lda <= 0
        || ldb <= 0
        || ldc <= 0
        || (lda as usize) < a_cols
        || (ldb as usize) < b_cols
        || (ldc as usize) < n
    {
        return Err(apxinf_core::Error::Other(format!(
            "GEMM_EX invalid row strides lda={lda}, ldb={ldb}, ldc={ldc}"
        )));
    }
    let strided_bytes = |rows: usize, stride: i32, cols: usize| -> Result<usize> {
        let elements = rows
            .saturating_sub(1)
            .checked_mul(stride as usize)
            .and_then(|offset| offset.checked_add(cols))
            .ok_or_else(|| apxinf_core::Error::Other("GEMM_EX buffer size overflow".into()))?;
        checked_bytes(dtype, &[elements], "GEMM_EX")
    };
    require_buffers(
        ctx,
        "GEMM_EX",
        &[
            ("A", a, strided_bytes(a_rows, lda, a_cols)?),
            ("B", b, strided_bytes(b_rows, ldb, b_cols)?),
            ("output", output, strided_bytes(m, ldc, n)?),
        ],
    )?;
    ctx.cublas()
        .gemm_ex(
            dtype, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, output, ldc,
        )
        .map_err(apxinf_core::Error::Cuda)
}

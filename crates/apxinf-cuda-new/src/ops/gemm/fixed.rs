//! Fixed and direct-launch GEMM operators.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::{ops, CudaContext};

fn output_for(ctx: &CudaContext, a: &Tensor, b: &Tensor, what: &str) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
        return Err(Error::Other(format!(
            "{what} shape mismatch: {a_dims:?} @ {b_dims:?}"
        )));
    }
    ctx.allocate_output(Shape::new(vec![a_dims[0], b_dims[1]]), DType::BF16)
}

fn bf16_alloc(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let mut output = output_for(ctx, activation, weight, "BF16 GEMM")?;
    ops::gemm(ctx, ops::GemmArgs::new(activation, weight, &mut output))?;
    Ok(output)
}

/// `matmul`: the legacy generic GEMM. BF16 routes to the tuned operator; any
/// other dtype (pi0fast's FP32 SigLIP patch projection) runs through raw
/// cuBLAS at the operand dtype, as the legacy kernel did — the tuned GEMM
/// operator is BF16-only and rounding the projection to BF16 changes the
/// autoregressive action tokens.
pub fn matmul(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    if activation.dtype() == DType::BF16 {
        return bf16_alloc(ctx, activation, weight);
    }
    if activation.dtype() != weight.dtype() {
        return Err(Error::DTypeMismatch {
            expected: activation.dtype(),
            got: weight.dtype(),
        });
    }
    let a_dims = activation.shape().dims();
    let w_dims = weight.shape().dims();
    if a_dims.len() != 2 || w_dims.len() != 2 || a_dims[1] != w_dims[0] {
        return Err(Error::Other(format!(
            "matmul expects [m,k] @ [k,n], got {a_dims:?} and {w_dims:?}"
        )));
    }
    let (m, k, n) = (a_dims[0], a_dims[1], w_dims[1]);
    let output = ctx.allocate_output(Shape::new(vec![m, n]), activation.dtype())?;
    let activation_buffer = crate::CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_buffer = crate::CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    let output_buffer = crate::CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
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
            &output_buffer,
        )
        .map_err(Error::Cuda)?;
    Ok(output)
}

// ── qwen_drive direct-launch GEMM helpers over raw cuBLAS ──────────────────
//
// These predate the tuned GEMM operator and keep their legacy arithmetic:
// FP32 accumulators held until the bias lands, one final BF16 rounding.

use crate::cublas::CublasTranspose;
use crate::ops::validation::{checked_bytes, require_buffers, require_finite};
use crate::CudaBuffer;

/// Raw strided GEMM into a caller-owned buffer: `output = alpha * op(a) @
/// op(b) + beta * output` with explicit leading dimensions.
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
        return Err(Error::Other(format!(
            "GEMM_EX invalid row strides lda={lda}, ldb={ldb}, ldc={ldc}"
        )));
    }
    let strided_bytes = |rows: usize, stride: i32, cols: usize| -> Result<usize> {
        let elements = rows
            .saturating_sub(1)
            .checked_mul(stride as usize)
            .and_then(|offset| offset.checked_add(cols))
            .ok_or_else(|| Error::Other("GEMM_EX buffer size overflow".into()))?;
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
        .map_err(Error::Cuda)
}

// ── Decode fast-path `write` surface (caller-owned buffers, graph-safe) ───

/// Row-major `A[M,K] @ B[K,N]` into caller-owned storage through raw cuBLAS.
/// No allocation, no tuned-operator session requirement — this is the same
/// direct `ctx.cublas().gemm(..)` dispatch the legacy decode graph used.
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
    ctx.cublas()
        .gemm(dtype, m, n, k, alpha, a, b, beta, output)
        .map_err(Error::Cuda)
}

//! BF16 gate/up projection with checkpoint-row-major weights.
use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use super::super::contracts::{checked_bytes, require_buffers};
use crate::{buffer::CudaBuffer, context::CudaContext, cublas::CublasTranspose};

/// Compute SwiGLU from input `[M,K]` and gate-then-up weight `[2N,K]`.
/// The first M rows are the logical result. An implementation may return
/// additional zero rows for a following projection; callers must slice the
/// projected result back to M before residual addition.
/// The fixed SM110 AOT route returns 3584 physical rows for M=3387 and fuses
/// activation into the GEMM epilogue. The generic route materializes a BF16
/// projection first; their rounding boundaries differ, so cross-route outputs
/// are not promised bit-identical.
pub fn bf16_swiglu_checkpoint(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    let x = input.shape().dims();
    let w = weight.shape().dims();
    if x.len() != 2
        || w.len() != 2
        || x.contains(&0)
        || w.contains(&0)
        || x[1] != w[1]
        || w[0] % 2 != 0
    {
        return Err(Error::Other(
            "BF16 SwiGLU expects input[M,K], weight[2N,K]".into(),
        ));
    }
    for tensor in [input, weight] {
        if tensor.dtype() != DType::BF16 || tensor.device() != Device::Cuda(ctx.device_id()) {
            return Err(Error::Other(
                "BF16 SwiGLU requires BF16 inputs on the context device".into(),
            ));
        }
    }
    let k = i32::try_from(x[1]).map_err(|_| Error::Other("SwiGLU K exceeds i32".into()))?;
    let n2 = i32::try_from(w[0]).map_err(|_| Error::Other("SwiGLU width exceeds i32".into()))?;
    i32::try_from(x[0]).map_err(|_| Error::Other("SwiGLU rows exceed i32".into()))?;
    let xp = CudaBuffer::from_tensor(input).map_err(Error::Cuda)?;
    let wp = CudaBuffer::from_tensor(weight).map_err(Error::Cuda)?;
    require_buffers(
        ctx,
        "BF16 SwiGLU",
        &[
            ("input", &xp, checked_bytes(DType::BF16, x, "SwiGLU input")?),
            (
                "weight",
                &wp,
                checked_bytes(DType::BF16, w, "SwiGLU weight")?,
            ),
        ],
    )?;

    #[cfg(apxinf_aot_sm110)]
    if ctx.caps().sm == 110
        && ctx.caps().multiprocessor_count == 20
        && x == [3387, 2560]
        && w == [18432, 2560]
    {
        if crate::workspace::may_prepare_native_resources() {
            let status = unsafe { crate::ffi::apxinf_quack_m256n256_init() };
            if status != 0 {
                return Err(Error::Other(format!(
                    "BF16 SwiGLU AOT prepare failed: {status}"
                )));
            }
        }
        let output = crate::workspace::output_buffer(
            ctx,
            checked_bytes(DType::BF16, &[3584, 9216], "SwiGLU padded output")?,
        )?;
        let status = unsafe {
            crate::ffi::apxinf_quack_m256n256_forward(
                xp.ptr(),
                wp.ptr(),
                output.ptr(),
                3387,
                20,
                ctx.stream().handle(),
            )
        };
        if status != 0 {
            return Err(Error::Other(format!(
                "BF16 SwiGLU AOT enqueue failed: {status}"
            )));
        }
        let prefix = 3387 * 9216 * 2;
        let tail = (3584 - 3387) * 9216 * 2;
        output
            .view(prefix, tail)
            .map_err(Error::Cuda)?
            .memset_async(0, tail, ctx.stream())
            .map_err(Error::Cuda)?;
        return Ok(output.into_tensor(Shape::new(vec![3584, 9216]), DType::BF16));
    }

    let projection = crate::workspace::output_buffer(
        ctx,
        checked_bytes(DType::BF16, &[x[0], w[0]], "SwiGLU projection")?,
    )?;
    ctx.cublas()
        .gemm_ex(
            DType::BF16,
            CublasTranspose::None,
            CublasTranspose::Transpose,
            x[0],
            w[0],
            x[1],
            1.0,
            &xp,
            k,
            &wp,
            k,
            0.0,
            &projection,
            n2,
        )
        .map_err(Error::Cuda)?;
    let projection = projection.into_tensor(Shape::new(vec![x[0], w[0]]), DType::BF16);
    super::super::activation::swiglu_bf16_rounded(ctx, &projection)
}

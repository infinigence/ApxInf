use apxinf_core::{DType, Error, Result};

use super::contracts::{check_cuda, checked_bytes, require_buffers, require_finite};
use crate::{ffi, CudaBuffer, CudaContext};

#[allow(clippy::too_many_arguments)]
pub fn residual_add_rmsnorm_pack8_bf16_into(
    ctx: &CudaContext,
    residual: &CudaBuffer,
    delta: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    columns: usize,
    rows: usize,
    eps: f32,
) -> Result<()> {
    require_finite("Qwen2.5-Omni pack8 residual RMSNorm", &[eps])?;
    let matrix = checked_bytes(
        DType::BF16,
        &[rows, columns],
        "Qwen2.5-Omni pack8 residual RMSNorm",
    )?;
    let weight_bytes = checked_bytes(
        DType::BF16,
        &[columns],
        "Qwen2.5-Omni pack8 residual RMSNorm",
    )?;
    require_buffers(
        ctx,
        "Qwen2.5-Omni pack8 residual RMSNorm",
        &[
            ("residual", residual, matrix),
            ("delta", delta, matrix),
            ("weight", weight, weight_bytes),
            ("output", output, matrix),
        ],
    )?;
    let alignment = residual.ptr() as usize
        | delta.ptr() as usize
        | weight.ptr() as usize
        | output.ptr() as usize;
    if ctx.caps().sm != 89
        || rows != 1
        || columns != 2048
        || eps.to_bits() != 1.0e-6f32.to_bits()
        || alignment & 15 != 0
    {
        return Err(Error::Other(
            "Qwen2.5-Omni pack8 residual RMSNorm contract mismatch".into(),
        ));
    }
    check_cuda(unsafe {
        ffi::apxinf_static_qwen25_omni_residual_rmsnorm_pack8_bf16(
            residual.ptr(),
            delta.ptr(),
            weight.ptr(),
            output.ptr(),
            rows as i32,
            columns as i32,
            eps,
            ctx.stream().handle(),
        )
    })
}

//! Family-private launchers for exact numerical contracts that have not yet
//! moved behind the native Spec dispatcher.
//!
//! These functions do not allocate outputs and are intentionally not part of
//! the public L3 API. They preserve the existing CUDA kernels, arguments, and
//! rounding order while `norm`, `fused_add_norm`, and `residual_add` remain the
//! only public entry points.

use apxinf_core::{Error, Result, Tensor};

use crate::ffi::abi::norm as abi;
use crate::ops::validation::{check_cuda, expect_bf16, gpu_ptr, matrix_shape};
use crate::CudaContext;

fn positive_epsilon(epsilon: f32, operation: &str) -> Result<()> {
    if epsilon.is_finite() && epsilon > 0.0 {
        Ok(())
    } else {
        Err(Error::Other(format!(
            "{operation} requires a positive finite epsilon"
        )))
    }
}

fn same_matrix(reference: &Tensor, tensor: &Tensor, operation: &str) -> Result<()> {
    if tensor.shape().dims() == reference.shape().dims() {
        Ok(())
    } else {
        Err(Error::Other(format!("{operation} matrix shape mismatch")))
    }
}

fn bf16_tensors(tensors: &[&Tensor], operation: &str) -> Result<()> {
    for tensor in tensors {
        expect_bf16(tensor, operation)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn weighted_adaptive(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    output: &Tensor,
    epsilon: f32,
) -> Result<()> {
    let (rows, cols) = matrix_shape(input, "weighted adaptive RMSNorm")?;
    if rows == 0
        || cols == 0
        || [weight, scale, shift]
            .iter()
            .any(|tensor| tensor.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "weighted adaptive RMSNorm shape mismatch".into(),
        ));
    }
    same_matrix(input, output, "weighted adaptive RMSNorm")?;
    positive_epsilon(epsilon, "weighted adaptive RMSNorm")?;
    bf16_tensors(
        &[input, weight, scale, shift, output],
        "weighted adaptive RMSNorm",
    )?;
    unsafe {
        check_cuda(abi::apxinf_norm_adaln_rms_norm_bf16(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            gpu_ptr(scale)?,
            gpu_ptr(shift)?,
            gpu_ptr(output)?,
            rows as i32,
            cols as i32,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}

pub(crate) fn unit_offset_residual(
    ctx: &CudaContext,
    input: &Tensor,
    residual: &Tensor,
    gate: &Tensor,
    hidden: &Tensor,
) -> Result<()> {
    let (rows, cols) = matrix_shape(input, "unit-offset gated residual")?;
    if rows == 0 || cols == 0 || gate.shape().dims() != [cols] {
        return Err(Error::Other(
            "unit-offset gated residual shape mismatch".into(),
        ));
    }
    same_matrix(input, residual, "unit-offset gated residual")?;
    same_matrix(input, hidden, "unit-offset gated residual")?;
    bf16_tensors(
        &[input, residual, gate, hidden],
        "unit-offset gated residual",
    )?;
    unsafe {
        check_cuda(abi::apxinf_norm_adaln_gate_residual_bf16(
            gpu_ptr(input)?,
            gpu_ptr(residual)?,
            gpu_ptr(gate)?,
            gpu_ptr(hidden)?,
            (rows * cols) as i64,
            cols as i32,
            ctx.stream().handle(),
        ))
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn unit_offset_residual_norm(
    ctx: &CudaContext,
    input: &Tensor,
    residual: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    hidden: &Tensor,
    normalized: &Tensor,
    epsilon: f32,
) -> Result<()> {
    let (rows, cols) = matrix_shape(input, "unit-offset gated residual RMSNorm")?;
    if rows == 0
        || cols == 0
        || cols > 8192
        || [gate, weight, scale, shift]
            .iter()
            .any(|tensor| tensor.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "unit-offset gated residual RMSNorm shape mismatch".into(),
        ));
    }
    for tensor in [residual, hidden, normalized] {
        same_matrix(input, tensor, "unit-offset gated residual RMSNorm")?;
    }
    positive_epsilon(epsilon, "unit-offset gated residual RMSNorm")?;
    bf16_tensors(
        &[
            input, residual, gate, weight, scale, shift, hidden, normalized,
        ],
        "unit-offset gated residual RMSNorm",
    )?;
    unsafe {
        check_cuda(abi::apxinf_norm_adaln_gate_residual_rms_bf16(
            gpu_ptr(input)?,
            gpu_ptr(residual)?,
            gpu_ptr(gate)?,
            gpu_ptr(weight)?,
            gpu_ptr(scale)?,
            gpu_ptr(shift)?,
            gpu_ptr(hidden)?,
            gpu_ptr(normalized)?,
            rows as i32,
            cols as i32,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}

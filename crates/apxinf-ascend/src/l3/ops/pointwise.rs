//! Pointwise operations — GeGLU, BiasActivation, EulerUpdate.
//!
//! Uses `aclnnGelu`, `aclnnMul`, `aclnnAdd`, `aclnnMuls`.

use apxinf_core::{DType, Error, Result, Shape, Storage, Tensor};

use crate::l3::context::AscendNewContext;

// ── Args structs ──────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointwiseSemantic {
    /// input [rows, 2*cols] → output [rows, cols] = A * Gelu(B)
    Geglu,
    /// output = activation(input + bias)
    BiasActivation,
    /// output = state + dt * velocity (Euler integration step)
    EulerUpdate,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointwiseActivation {
    None,
    Gelu,
    Silu,
}

pub struct PointwiseArgs<'a> {
    pub semantic: PointwiseSemantic,
    pub input: &'a Tensor,
    pub secondary: Option<&'a Tensor>,
    pub bias: Option<&'a Tensor>,
    pub out: &'a mut Tensor,
    pub activation: PointwiseActivation,
    pub dt: f32,
    pub output_scale: f32,
}

impl<'a> PointwiseArgs<'a> {
    pub fn new(semantic: PointwiseSemantic, input: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            semantic,
            input,
            secondary: None,
            bias: None,
            out,
            activation: PointwiseActivation::None,
            dt: 0.0,
            output_scale: 1.0,
        }
    }
}

// ── Implementation ────────────────────────────────────────────────────

pub fn pointwise(ctx: &AscendNewContext, args: PointwiseArgs<'_>) -> Result<()> {
    match args.semantic {
        PointwiseSemantic::Geglu => pointwise_geglu(ctx, args),
        PointwiseSemantic::BiasActivation => pointwise_bias_activation(ctx, args),
        PointwiseSemantic::EulerUpdate => pointwise_euler_update(ctx, args),
    }
}

/// GeGLU: input [rows, 2*cols] → out [rows, cols] = A * Gelu(B)
fn pointwise_geglu(ctx: &AscendNewContext, args: PointwiseArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.input.dtype();

    let input_dims = args.input.shape().dims();
    let rows = if input_dims.len() == 2 { input_dims[0] } else { 1 };
    let cols = input_dims[input_dims.len() - 1] / 2;

    let input_shape = Shape::new(vec![rows, cols * 2]);
    let out_shape = Shape::new(vec![rows, cols]);

    let input_ptr = device_ptr(args.input)?;
    let out_ptr = device_ptr(args.out)?;

    // Allocate throwaway out_gelu
    let out_gelu = ctx.allocate_output(out_shape.clone(), dtype)?;
    let out_gelu_ptr = device_ptr(&out_gelu)?;

    let acl_input = crate::acl_tensor::AclTensor::from_device(input_ptr, &input_shape, dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, &out_shape, dtype);
    let acl_out_gelu = crate::acl_tensor::AclTensor::from_device(out_gelu_ptr, &out_shape, dtype);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnGeGluGetWorkspaceSize(
                acl_input.raw(),
                1, // dim=1
                0, // approximate=0 (tanh)
                acl_out.raw(),
                acl_out_gelu.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnGeGlu(ws, ws_size, exec, s)
        },
        stream,
        "aclnnGeGlu (pointwise)",
    )
    .map_err(|e| Error::Other(format!("pointwise geglu: {e}")))
}

/// BiasActivation: out = activation(input + bias)
fn pointwise_bias_activation(ctx: &AscendNewContext, args: PointwiseArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.input.dtype();
    let shape = args.input.shape().clone();

    let input_ptr = device_ptr(args.input)?;
    let out_ptr = device_ptr(args.out)?;

    // Step 1: out = input + bias (or just use input directly)
    // For activation=None: use aclnnInplaceAdd on out (out += bias)
    // For activation=Gelu/Silu: need tmp buffer, then activation into out
    match args.activation {
        PointwiseActivation::None => {
            if let Some(bias) = args.bias {
                // out = input + bias → first copy input to out, then inplace add bias
                super::aclnn_d2d_copy(out_ptr, input_ptr, shape.numel(), dtype, stream)?;
                super::aclnn_inplace_add(
                    out_ptr,
                    device_ptr(bias)?,
                    &shape,
                    bias.shape(),
                    dtype,
                    stream,
                )?;
            } else {
                // No bias, no activation: just copy input to out
                super::aclnn_d2d_copy(out_ptr, input_ptr, shape.numel(), dtype, stream)?;
            }
            Ok(())
        }
        PointwiseActivation::Gelu | PointwiseActivation::Silu => {
            // Need a temp buffer for the pre-activation tensor
            let tmp = if let Some(bias) = args.bias {
                let tmp = ctx.allocate_output(shape.clone(), dtype)?;
                let acl_input = crate::acl_tensor::AclTensor::from_device(input_ptr, &shape, dtype);
                let acl_bias = crate::acl_tensor::AclTensor::from_device(device_ptr(bias)?, bias.shape(), dtype);
                let acl_tmp = crate::acl_tensor::AclTensor::from_device(device_ptr(&tmp)?, &shape, dtype);
                let acl_alpha = crate::acl_tensor::AclScalar::f32(1.0);

                crate::acl_tensor::run_op(
                    |ws, exec| unsafe {
                        crate::ffi::aclnn::aclnnAddGetWorkspaceSize(
                            acl_input.raw(),
                            acl_bias.raw(),
                            acl_alpha.raw(),
                            acl_tmp.raw(),
                            ws,
                            exec,
                        )
                    },
                    |ws, ws_size, exec, s| unsafe {
                        crate::ffi::aclnn::aclnnAdd(ws, ws_size, exec, s)
                    },
                    stream,
                    "aclnnAdd (bias_activation)",
                )
                .map_err(|e| Error::Other(format!("pointwise bias_activation add: {e}")))?;
                tmp
            } else {
                // No bias: copy input to tmp for activation
                let tmp = ctx.allocate_output(shape.clone(), dtype)?;
                super::aclnn_d2d_copy(device_ptr(&tmp)?, input_ptr, shape.numel(), dtype, stream)?;
                tmp
            };

            let tmp_ptr = device_ptr(&tmp)?;
            let acl_tmp = crate::acl_tensor::AclTensor::from_device(tmp_ptr, &shape, dtype);
            let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, &shape, dtype);

            match args.activation {
                PointwiseActivation::Gelu => {
                    crate::acl_tensor::run_op(
                        |ws, exec| unsafe {
                            crate::ffi::aclnn::aclnnGeluGetWorkspaceSize(
                                acl_tmp.raw(),
                                acl_out.raw(),
                                ws,
                                exec,
                            )
                        },
                        |ws, ws_size, exec, s| unsafe {
                            crate::ffi::aclnn::aclnnGelu(ws, ws_size, exec, s)
                        },
                        stream,
                        "aclnnGelu (bias_activation)",
                    )
                    .map_err(|e| Error::Other(format!("pointwise gelu: {e}")))
                }
                PointwiseActivation::Silu => {
                    crate::acl_tensor::run_op(
                        |ws, exec| unsafe {
                            crate::ffi::aclnn::aclnnSiluGetWorkspaceSize(
                                acl_tmp.raw(),
                                acl_out.raw(),
                                ws,
                                exec,
                            )
                        },
                        |ws, ws_size, exec, s| unsafe {
                            crate::ffi::aclnn::aclnnSilu(ws, ws_size, exec, s)
                        },
                        stream,
                        "aclnnSilu (bias_activation)",
                    )
                    .map_err(|e| Error::Other(format!("pointwise silu: {e}")))
                }
                _ => unreachable!(),
            }
        }
    }
}

/// EulerUpdate: out = state + dt * velocity
fn pointwise_euler_update(ctx: &AscendNewContext, args: PointwiseArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.input.dtype();
    let shape = args.input.shape().clone();

    let state_ptr = device_ptr(args.input)?;
    let out_ptr = device_ptr(args.out)?;

    // scaled_velocity = velocity * dt
    let velocity = args.secondary.ok_or_else(|| Error::Other("EulerUpdate requires secondary (velocity)".into()))?;
    let scaled = ctx.allocate_output(shape.clone(), dtype)?;
    let scaled_ptr = device_ptr(&scaled)?;

    let acl_vel = crate::acl_tensor::AclTensor::from_device(device_ptr(velocity)?, velocity.shape(), dtype);
    let acl_scaled = crate::acl_tensor::AclTensor::from_device(scaled_ptr, &shape, dtype);
    let acl_dt = crate::acl_tensor::AclScalar::f32(args.dt);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnMulsGetWorkspaceSize(
                acl_vel.raw(),
                acl_dt.raw(),
                acl_scaled.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnMuls(ws, ws_size, exec, s)
        },
        stream,
        "aclnnMuls (euler_update)",
    )
    .map_err(|e| Error::Other(format!("pointwise euler muls: {e}")))?;

    // out = state + scaled_velocity
    let acl_state = crate::acl_tensor::AclTensor::from_device(state_ptr, &shape, dtype);
    let acl_scaled = crate::acl_tensor::AclTensor::from_device(scaled_ptr, &shape, dtype);
    let acl_out = crate::acl_tensor::AclTensor::from_device(out_ptr, &shape, dtype);
    let acl_alpha = crate::acl_tensor::AclScalar::f32(1.0);

    crate::acl_tensor::run_op(
        |ws, exec| unsafe {
            crate::ffi::aclnn::aclnnAddGetWorkspaceSize(
                acl_state.raw(),
                acl_scaled.raw(),
                acl_alpha.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe {
            crate::ffi::aclnn::aclnnAdd(ws, ws_size, exec, s)
        },
        stream,
        "aclnnAdd (euler_update)",
    )
    .map_err(|e| Error::Other(format!("pointwise euler add: {e}")))
}

// ── Helpers ───────────────────────────────────────────────────────────

fn device_ptr(tensor: &Tensor) -> Result<usize> {
    match tensor.storage() {
        Storage::Gpu { handle, .. } => Ok(handle.ptr()),
        _ => Err(Error::Other(format!(
            "expected NPU tensor, got {:?}",
            tensor.device()
        ))),
    }
}

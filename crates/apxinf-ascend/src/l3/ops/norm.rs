//! Normalization operations —RMSNorm, LayerNorm, and fused residual+norm variants.
//!
//! Uses `aclnnRmsNorm`, `aclnnLayerNorm`, `aclnnAddRmsNorm`, and compositions
//! of `aclnnAdd` / `aclnnMul` for the adaptive variants.

use std::cell::RefCell;

use apxinf_core::{DType, Device, Error, Result, Shape, Storage, Tensor};

use crate::l3::context::AscendNewContext;

// ━ Args structs ━

pub struct RmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> RmsNormArgs<'a> {
    pub fn new(input: &'a Tensor, weight: &'a Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, weight, normalized, eps, output_scale: 1.0 }
    }
}

pub struct LayerNormArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub bias: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> LayerNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        weight: &'a Tensor,
        bias: &'a Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self { input, weight, bias, normalized, eps, output_scale: 1.0 }
    }
}

pub struct AdaptiveRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> AdaptiveRmsNormArgs<'a> {
    pub fn new(input: &'a Tensor, norm_style: &'a Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, norm_style, normalized, eps, output_scale: 1.0 }
    }
}

pub struct BiasResidualArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub hidden: &'a mut Tensor,
}

impl<'a> BiasResidualArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        hidden: &'a mut Tensor,
    ) -> Self {
        Self { input, bias, residual, hidden }
    }
}

pub struct BiasResidualRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub weight: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> BiasResidualRmsNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        weight: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self { input, bias, residual, weight, hidden, normalized, eps, output_scale: 1.0 }
    }
}

// ━ All-ones weight cache (RI-capture compatible) ━

struct OnesWeightCache {
    key: (usize, u8),
    tensor: Tensor,
}

thread_local! {
    static ONES_WEIGHT_CACHE: RefCell<Option<OnesWeightCache>> = const { RefCell::new(None) };
}

fn get_or_create_ones_weight(ctx: &AscendNewContext, cols: usize, dtype: DType) -> Result<Tensor> {
    let key = (cols, dtype as u8);
    let cached = ONES_WEIGHT_CACHE.with(|c| {
        c.borrow().as_ref().map_or(false, |e| e.key == key)
    });
    if cached {
        return ONES_WEIGHT_CACHE.with(|c| Ok(c.borrow().as_ref().unwrap().tensor.clone()));
    }

    let weight_shape = Shape::new(vec![cols]);
    let bytes: Vec<u8> = match dtype {
        DType::F32 => {
            let v: Vec<f32> = vec![1.0; cols];
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::BF16 => {
            let v: Vec<u16> = vec![0x3F80; cols];
            bytemuck::cast_slice(&v).to_vec()
        }
        DType::F16 => {
            let v: Vec<u16> = vec![0x3C00; cols];
            bytemuck::cast_slice(&v).to_vec()
        }
        _ => return Err(Error::Other("ones_weight: unsupported dtype".into())),
    };
    let cpu_weight = Tensor::from_raw(weight_shape, dtype, Device::Cpu, bytes)?;
    let tensor = crate::l3::transfers::to_npu(&cpu_weight, ctx.device_id())?;
    ONES_WEIGHT_CACHE.with(|c| {
        *c.borrow_mut() = Some(OnesWeightCache { key, tensor: tensor.clone() });
    });
    Ok(tensor)
}

pub struct BiasResidualLayerNormArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub weight: &'a Tensor,
    pub norm_bias: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> BiasResidualLayerNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        weight: &'a Tensor,
        norm_bias: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self { input, bias, residual, weight, norm_bias, hidden, normalized, eps, output_scale: 1.0 }
    }
}

pub struct AdaGateResidualArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub gate_style: &'a Tensor,
    pub hidden: &'a mut Tensor,
}

impl<'a> AdaGateResidualArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        residual: &'a Tensor,
        gate_style: &'a Tensor,
        hidden: &'a mut Tensor,
    ) -> Self {
        Self { input, residual, gate_style, hidden }
    }
}

pub struct AdaGateResidualRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub gate_style: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub output_scale: f32,
}

impl<'a> AdaGateResidualRmsNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        residual: &'a Tensor,
        norm_style: &'a Tensor,
        gate_style: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self { input, residual, norm_style, gate_style, hidden, normalized, eps, output_scale: 1.0 }
    }
}

pub struct BiasThenResidualArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub hidden: &'a mut Tensor,
}

impl<'a> BiasThenResidualArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        hidden: &'a mut Tensor,
    ) -> Self {
        Self { input, bias, residual, hidden }
    }
}

// ━ Implementations ━

/// RMS normalization via `aclnnRmsNorm`.
pub fn rms_norm(ctx: &AscendNewContext, args: RmsNormArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.input.dtype();
    let shape = args.input.shape().clone();

    // rstd output: same shape as input, but MUST be FP32 (not BF16)
    // aclnnRmsNorm rejects BF16 rstd with 161002 (PARAM_INVALID) on this CANN stack
    let rstd = ctx.allocate_output(shape.clone(), DType::F32)?;

    let acl_input = acl_tensor(args.input, dtype);
    let acl_weight = acl_tensor(args.weight, dtype);
    let acl_out = acl_tensor(args.normalized, dtype);
    let acl_rstd = acl_tensor(&rstd, DType::F32);

    run_op(
        |ws, exec| unsafe {
            ffi::aclnnRmsNormGetWorkspaceSize(
                acl_input.raw(),
                acl_weight.raw(),
                args.eps as f64,
                acl_out.raw(),
                acl_rstd.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { ffi::aclnnRmsNorm(ws, ws_size, exec, s) },
        stream,
        "aclnnRmsNorm",
    )
    .map_err(|e| Error::Other(format!("rms_norm: {e}")))
}

/// Layer normalization via `aclnnLayerNorm`.
pub fn layer_norm(ctx: &AscendNewContext, args: LayerNormArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.input.dtype();
    let shape = args.input.shape().clone();

    // mean and rstd outputs (not used, but required by the API)
    let norm_dims = shape.dims();
    let norm_shape_dim = norm_dims[norm_dims.len() - 1];
    let rows = shape.numel() / norm_shape_dim;
    let mean = ctx.allocate_output(Shape::new(vec![rows, 1]), DType::F32)?;
    let rstd = ctx.allocate_output(Shape::new(vec![rows, 1]), DType::F32)?;

    let acl_input = acl_tensor(args.input, dtype);
    let acl_weight = acl_tensor(args.weight, dtype);
    let acl_bias = acl_tensor(args.bias, dtype);
    let acl_out = acl_tensor(args.normalized, dtype);
    let acl_mean = acl_tensor(&mean, DType::F32);
    let acl_rstd = acl_tensor(&rstd, DType::F32);

    let norm_shape_arr: Vec<i64> = vec![norm_shape_dim as i64];
    let norm_shape = crate::acl_tensor::AclIntArray::new(&norm_shape_arr);

    run_op(
        |ws, exec| unsafe {
            ffi::aclnnLayerNormGetWorkspaceSize(
                acl_input.raw(),
                norm_shape.raw(),
                acl_weight.raw(),
                acl_bias.raw(),
                args.eps as f64,
                acl_out.raw(),
                acl_mean.raw(),
                acl_rstd.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { ffi::aclnnLayerNorm(ws, ws_size, exec, s) },
        stream,
        "aclnnLayerNorm",
    )
    .map_err(|e| Error::Other(format!("layer_norm: {e}")))
}

/// Adaptive RMS norm: split norm_style into scale + shift, then
/// `normalized = rms_norm(input) * scale + shift`.
pub fn adaptive_rms_norm(ctx: &AscendNewContext, args: AdaptiveRmsNormArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.input.dtype();
    let shape = args.input.shape().clone();

    // norm_style is [1, 2*cols] ->split into scale [cols] and shift [cols]
    let cols = shape.dims()[shape.dims().len() - 1];
    let norm_style_dims = args.norm_style.shape().dims();
    let style_cols = norm_style_dims[norm_style_dims.len() - 1];
    assert_eq!(style_cols, 2 * cols, "norm_style last dim must be 2*cols");

    // Step 1: RMS norm
    // Use a dummy weight of ones for the rms_norm call.
    // Cached on device —warmup pass computes + H2D, capture pass reuses
    // cached device pointer (no H2D during RI capture).
    let weight_shape = Shape::new(vec![cols]);
    let weight = get_or_create_ones_weight(ctx, cols, dtype)?;

    let rstd = ctx.allocate_output(shape.clone(), DType::F32)?;
    let acl_input = acl_tensor(args.input, dtype);
    let acl_weight = acl_tensor(&weight, dtype);
    let acl_out = acl_tensor(args.normalized, dtype);
    let acl_rstd = acl_tensor(&rstd, DType::F32);

    run_op(
        |ws, exec| unsafe {
            ffi::aclnnRmsNormGetWorkspaceSize(
                acl_input.raw(),
                acl_weight.raw(),
                args.eps as f64,
                acl_out.raw(),
                acl_rstd.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { ffi::aclnnRmsNorm(ws, ws_size, exec, s) },
        stream,
        "aclnnRmsNorm (adaptive)",
    )
    .map_err(|e| Error::Other(format!("adaptive_rms_norm rms: {e}")))?;

    // Step 2: Apply scale and shift from norm_style
    // scale = norm_style[:, :cols], shift = norm_style[:, cols:]
    // normalized = normalized * scale + shift
    // We use aclnnMul then aclnnAdd
    // For simplicity, extract scale/shift as sub-views of norm_style
    let scale_tensor = {
        let buf = crate::l3::buffer::AscendBuffer::from_tensor(args.norm_style)?;
        let half_bytes = cols * dtype.size_in_bytes();
        let scale_buf = buf.view(0, half_bytes)?;
        scale_buf.as_tensor(Shape::new(vec![cols]), dtype)?
    };
    let shift_tensor = {
        let buf = crate::l3::buffer::AscendBuffer::from_tensor(args.norm_style)?;
        let half_bytes = cols * dtype.size_in_bytes();
        let shift_buf = buf.view(half_bytes, half_bytes)?;
        shift_buf.as_tensor(Shape::new(vec![cols]), dtype)?
    };

    // CUDA + torch_npu contract: `normalized * (1 + scale) + shift`.
    // We compute it as `normalized + (normalized * scale) + shift` since `scale`
    // is a per-step modulation tensor that cannot be pre-baked into the weight
    // (unlike the non-adaptive `rms_bf16` path which uses a pre-baked `1 + w`).
    //
    // Step A: tmp = normalized * scale
    let tmp = ctx.allocate_output(shape.clone(), dtype)?;
    let acl_norm = acl_tensor(args.normalized, dtype);
    let acl_scale = acl_tensor(&scale_tensor, dtype);
    let acl_tmp = acl_tensor(&tmp, dtype);

    run_op(
        |ws, exec| unsafe {
            ffi::aclnnMulGetWorkspaceSize(acl_norm.raw(), acl_scale.raw(), acl_tmp.raw(), ws, exec)
        },
        |ws, ws_size, exec, s| unsafe { ffi::aclnnMul(ws, ws_size, exec, s) },
        stream,
        "aclnnMul (adaptive scale)",
    )
    .map_err(|e| Error::Other(format!("adaptive_rms_norm mul: {e}")))?;

    // Step B: scaled = normalized + tmp  (i.e. normalized * (1 + scale))
    let scaled = ctx.allocate_output(shape.clone(), dtype)?;
    let acl_norm = acl_tensor(args.normalized, dtype);
    let acl_tmp = acl_tensor(&tmp, dtype);
    let acl_scaled = acl_tensor(&scaled, dtype);
    let acl_alpha = crate::acl_tensor::AclScalar::f32(1.0);

    run_op(
        |ws, exec| unsafe {
            ffi::aclnnAddGetWorkspaceSize(
                acl_norm.raw(),
                acl_tmp.raw(),
                acl_alpha.raw(),
                acl_scaled.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { ffi::aclnnAdd(ws, ws_size, exec, s) },
        stream,
        "aclnnAdd (adaptive 1+scale)",
    )
    .map_err(|e| Error::Other(format!("adaptive_rms_norm 1+scale: {e}")))?;

    // Step C: normalized = scaled + shift
    let acl_scaled = acl_tensor(&scaled, dtype);
    let acl_shift = acl_tensor(&shift_tensor, dtype);
    let acl_out = acl_tensor(args.normalized, dtype);
    let acl_alpha = crate::acl_tensor::AclScalar::f32(1.0);

    run_op(
        |ws, exec| unsafe {
            ffi::aclnnAddGetWorkspaceSize(
                acl_scaled.raw(),
                acl_shift.raw(),
                acl_alpha.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { ffi::aclnnAdd(ws, ws_size, exec, s) },
        stream,
        "aclnnAdd (adaptive shift)",
    )
    .map_err(|e| Error::Other(format!("adaptive_rms_norm add: {e}")))
}

/// Bias + residual: `hidden = (input + bias) + residual`.
/// Uses `aclnnAdd` twice (or once if bias is None).
/// NOTE: aclnnAdd does not support in-place (self == out ->161002).
/// When bias is present, the second add (hidden + residual ->hidden) needs
/// a temp buffer because `hidden` is both input and output.
pub fn bias_residual(ctx: &AscendNewContext, args: BiasResidualArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.input.dtype();
    let shape = args.input.shape().clone();

    let acl_input = acl_tensor(args.input, dtype);
    let acl_out = acl_tensor(args.hidden, dtype);
    let acl_alpha = crate::acl_tensor::AclScalar::f32(1.0);

    if let Some(bias) = args.bias {
        // hidden = input + bias
        let acl_bias = acl_tensor(bias, dtype);
        run_op(
            |ws, exec| unsafe {
                ffi::aclnnAddGetWorkspaceSize(
                    acl_input.raw(),
                    acl_bias.raw(),
                    acl_alpha.raw(),
                    acl_out.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe { ffi::aclnnAdd(ws, ws_size, exec, s) },
            stream,
            "aclnnAdd (bias_residual bias)",
        )
        .map_err(|e| Error::Other(format!("bias_residual add bias: {e}")))?;

        // hidden += residual via aclnnInplaceAdd (avoids temp buffer + copy-back)
        super::aclnn_inplace_add(
            device_ptr(args.hidden)?,
            device_ptr(args.residual)?,
            &shape,
            args.residual.shape(),
            dtype,
            stream,
        )
        .map_err(|e| Error::Other(format!("bias_residual inplace add: {e}")))
    } else {
        // hidden = input + residual (input 鈮?hidden, safe)
        let acl_res = acl_tensor(args.residual, dtype);
        run_op(
            |ws, exec| unsafe {
                ffi::aclnnAddGetWorkspaceSize(
                    acl_input.raw(),
                    acl_res.raw(),
                    acl_alpha.raw(),
                    acl_out.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe { ffi::aclnnAdd(ws, ws_size, exec, s) },
            stream,
            "aclnnAdd (bias_residual)",
        )
        .map_err(|e| Error::Other(format!("bias_residual: {e}")))
    }
}

/// Fused bias + residual + RMS norm via `aclnnAddRmsNorm`.
///
/// `x = (input + bias) + residual`; `hidden = x`; `normalized = x / RMS(x) * weight`.
/// When bias is present, we add bias first, then use AddRmsNorm for the residual+norm.
pub fn bias_residual_rms_norm(ctx: &AscendNewContext, args: BiasResidualRmsNormArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.input.dtype();
    let shape = args.input.shape().clone();

    // rstd output: must be 2D [rows, 1] for aclnnAddRmsNorm
    let cols = shape.dims()[shape.dims().len() - 1];
    let rows = shape.numel() / cols;
    let rstd = ctx.allocate_output(Shape::new(vec![rows, 1]), DType::F32)?;

    if let Some(bias) = args.bias {
        // Step 1: input += bias in-place (avoids tmp buffer allocation + D2D write/read).
        // Safe because `args.input` is a fresh GEMM output (projection), not reused
        // after this call —verified at the bf16.rs:587-595 call site.
        super::aclnn_inplace_add(
            device_ptr(args.input)?,
            device_ptr(bias)?,
            &shape,
            bias.shape(),
            dtype,
            stream,
        )
        .map_err(|e| Error::Other(format!("bias_residual_rms_norm inplace bias: {e}")))?;

        // Step 2: AddRmsNorm(input, residual, weight) ->hidden, normalized, rstd
        let acl_input = acl_tensor(args.input, dtype);
        let acl_res = acl_tensor(args.residual, dtype);
        let acl_weight = acl_tensor(args.weight, dtype);
        let acl_hidden = acl_tensor(args.hidden, dtype);
        let acl_norm = acl_tensor(args.normalized, dtype);
        let acl_rstd = acl_tensor(&rstd, DType::F32);

        run_op(
            |ws, exec| unsafe {
                ffi::aclnnAddRmsNormGetWorkspaceSize(
                    acl_input.raw(),
                    acl_res.raw(),
                    acl_weight.raw(),
                    args.eps as f64,
                    acl_norm.raw(),
                    acl_rstd.raw(),
                    acl_hidden.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe { ffi::aclnnAddRmsNorm(ws, ws_size, exec, s) },
            stream,
            "aclnnAddRmsNorm",
        )
        .map_err(|e| Error::Other(format!("bias_residual_rms_norm: {e}")))
    } else {
        // No bias: AddRmsNorm(input, residual, weight) ->hidden, normalized, rstd
        let acl_input = acl_tensor(args.input, dtype);
        let acl_res = acl_tensor(args.residual, dtype);
        let acl_weight = acl_tensor(args.weight, dtype);
        let acl_hidden = acl_tensor(args.hidden, dtype);
        let acl_norm = acl_tensor(args.normalized, dtype);
        let acl_rstd = acl_tensor(&rstd, DType::F32);

        run_op(
            |ws, exec| unsafe {
                ffi::aclnnAddRmsNormGetWorkspaceSize(
                    acl_input.raw(),
                    acl_res.raw(),
                    acl_weight.raw(),
                    args.eps as f64,
                    acl_norm.raw(),
                    acl_rstd.raw(),
                    acl_hidden.raw(),
                    ws,
                    exec,
                )
            },
            |ws, ws_size, exec, s| unsafe { ffi::aclnnAddRmsNorm(ws, ws_size, exec, s) },
            stream,
            "aclnnAddRmsNorm",
        )
        .map_err(|e| Error::Other(format!("bias_residual_rms_norm: {e}")))
    }
}

/// Bias + residual + Layer norm.
pub fn bias_residual_layer_norm(ctx: &AscendNewContext, args: BiasResidualLayerNormArgs<'_>) -> Result<()> {
    // Step 1: hidden = (input + bias) + residual (reuse bias_residual)
    bias_residual(ctx, BiasResidualArgs::new(args.input, args.bias, args.residual, args.hidden))?;

    // Step 2: layer_norm(hidden, weight, norm_bias) ->normalized
    layer_norm(ctx, LayerNormArgs::new(args.hidden, args.weight, args.norm_bias, args.normalized, args.eps))
}

/// AdaGate residual: `hidden = input * gate_style[2*cols:3*cols] + residual`.
pub fn ada_gate_residual(ctx: &AscendNewContext, args: AdaGateResidualArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.input.dtype();
    let shape = args.input.shape().clone();
    let cols = shape.dims()[shape.dims().len() - 1];

    // gate_style is [1, 3*cols]: scale = [0:cols], shift = [cols:2*cols], gate = [2*cols:3*cols].
    // The gate is the THIRD segment (matches CUDA `gate_style[2*cols + col]`
    // and torch_npu `modulation.chunk(3)[2]`).
    let scale_tensor = {
        let buf = crate::l3::buffer::AscendBuffer::from_tensor(args.gate_style)?;
        let element_bytes = dtype.size_in_bytes();
        let gate_offset = 2 * cols * element_bytes;
        let gate_bytes = cols * element_bytes;
        let scale_buf = buf.view(gate_offset, gate_bytes)?; // gate part (third segment)
        scale_buf.as_tensor(Shape::new(vec![cols]), dtype)?
    };

    // tmp = input * scale (gate)
    let tmp = ctx.allocate_output(shape.clone(), dtype)?;
    let acl_input = acl_tensor(args.input, dtype);
    let acl_scale = acl_tensor(&scale_tensor, dtype);
    let acl_tmp = acl_tensor(&tmp, dtype);

    run_op(
        |ws, exec| unsafe {
            ffi::aclnnMulGetWorkspaceSize(acl_input.raw(), acl_scale.raw(), acl_tmp.raw(), ws, exec)
        },
        |ws, ws_size, exec, s| unsafe { ffi::aclnnMul(ws, ws_size, exec, s) },
        stream,
        "aclnnMul (ada_gate)",
    )
    .map_err(|e| Error::Other(format!("ada_gate_residual mul: {e}")))?;

    // hidden = tmp + residual
    let acl_tmp = acl_tensor(&tmp, dtype);
    let acl_res = acl_tensor(args.residual, dtype);
    let acl_out = acl_tensor(args.hidden, dtype);
    let acl_alpha = crate::acl_tensor::AclScalar::f32(1.0);

    run_op(
        |ws, exec| unsafe {
            ffi::aclnnAddGetWorkspaceSize(
                acl_tmp.raw(),
                acl_res.raw(),
                acl_alpha.raw(),
                acl_out.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { ffi::aclnnAdd(ws, ws_size, exec, s) },
        stream,
        "aclnnAdd (ada_gate)",
    )
    .map_err(|e| Error::Other(format!("ada_gate_residual add: {e}")))
}

/// AdaGate residual + RMS norm.
pub fn ada_gate_residual_rms_norm(
    ctx: &AscendNewContext,
    args: AdaGateResidualRmsNormArgs<'_>,
) -> Result<()> {
    // Step 1: ada_gate_residual ->hidden
    ada_gate_residual(ctx, AdaGateResidualArgs::new(args.input, args.residual, args.gate_style, args.hidden))?;

    // Step 2: adaptive_rms_norm(hidden, norm_style) ->normalized
    adaptive_rms_norm(ctx, AdaptiveRmsNormArgs::new(args.hidden, args.norm_style, args.normalized, args.eps))
}

/// Bias then residual: `hidden = input + bias; hidden = hidden + residual`.
pub fn bias_then_residual(ctx: &AscendNewContext, args: BiasThenResidualArgs<'_>) -> Result<()> {
    bias_residual(ctx, BiasResidualArgs::new(args.input, args.bias, args.residual, args.hidden))
}

// ━ Helpers ━

use crate::ffi::aclnn as ffi;
use crate::acl_tensor::{run_op, AclTensor, AclScalar, AclIntArray};

fn device_ptr(tensor: &Tensor) -> Result<usize> {
    match tensor.storage() {
        Storage::Gpu { handle, .. } => Ok(handle.ptr()),
        _ => Err(Error::Other(format!(
            "expected NPU tensor, got {:?}",
            tensor.device()
        ))),
    }
}

fn acl_tensor(tensor: &Tensor, dtype: DType) -> AclTensor {
    AclTensor::from_device(device_ptr(tensor).unwrap_or(0), tensor.shape(), dtype)
}

/// Zero-copy variant of [`inplace_add_rms_norm`].
///
/// Returns `(normalized, residual)` directly instead of writing into
/// pre-allocated buffers. The caller MUST NOT read `x1` or `x2` after this
/// call — both are overwritten in place by the operator:
///   * `x1` is overwritten with the normalized output `rms_norm(x1+x2, weight)`
///   * `x2` is overwritten with the residual sum `(x1 + x2)`
///
/// Because `Tensor::clone()` is a zero-copy Arc reference count, `x2` only
/// needs a clone to obtain an owned handle — no device-to-device copy occurs.
/// This eliminates the 2 `aclnnInplaceCopy` (d2d) ops per call that the
/// legacy [`inplace_add_rms_norm`] wrapper performs (~754 ops for PI0.5).
///
/// # Safety (logical)
///
/// The caller must ensure no other code reads the original `x1`/`x2` values
/// after this function returns. In the PI0.5 graph this is always true:
/// `inplace_add_rms_bf16` consumes `projection_with_bias` (owned) and
/// `residual` (whose last use is this call).
pub fn inplace_add_rms_norm_zero_copy(
    ctx: &AscendNewContext,
    mut x1: Tensor,
    x2: Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let stream = ctx.raw_stream();
    let dtype = x1.dtype();
    let shape = x1.shape().clone();

    let cols = shape.dims()[shape.dims().len() - 1];
    let rows = shape.numel() / cols;
    let rstd = ctx.allocate_output(Shape::new(vec![rows, 1]), DType::F32)?;

    // x2.clone() is zero-copy (Arc refcount on the GpuStorageHandle).
    // The operator will overwrite the shared device memory with the residual
    // sum — safe because the caller guarantees no further reads of x2's
    // original value.
    let x2_owned = x2.clone();

    let acl_x1 = acl_tensor(&x1, dtype);
    let acl_x2 = acl_tensor(&x2_owned, dtype);
    let acl_weight = acl_tensor(weight, dtype);
    let acl_rstd = acl_tensor(&rstd, DType::F32);

    run_op(
        |ws, exec| unsafe {
            ffi::aclnnInplaceAddRmsNormGetWorkspaceSize(
                acl_x1.raw(),
                acl_x2.raw(),
                acl_weight.raw(),
                eps as f64,
                acl_rstd.raw(),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { ffi::aclnnInplaceAddRmsNorm(ws, ws_size, exec, s) },
        stream,
        "aclnnInplaceAddRmsNorm",
    )
    .map_err(|e| Error::Other(format!("inplace_add_rms_norm_zero_copy: {e}")))?;

    // After the op: x1 holds normalized, x2_owned holds residual sum.
    // Return them directly — no d2d copy needed.
    // Drop acl_x1/acl_x2 (they only hold raw ptr views, not ownership).
    drop(acl_x1);
    drop(acl_x2);
    Ok((x1, x2_owned))
}

/// Args for fused add + layer norm.
pub struct AddLayerNormArgs<'a> {
    pub x1: &'a Tensor,
    pub x2: &'a Tensor,
    pub weight: &'a Tensor,
    pub bias: &'a Tensor,
    /// Optional bias added to the normalized output (may be NULL).
    pub norm_bias: Option<&'a Tensor>,
    pub normalized: &'a mut Tensor,
    /// Optional output for the residual sum `x1 + x2` (may be None → NULL).
    /// When provided, the caller can reuse the residual without a separate
    /// `aclnnAdd` op. Required for the vision encoder post-attention path.
    pub hidden: Option<&'a mut Tensor>,
    pub eps: f32,
}

/// Fused `x = x1 + x2; normalized = layer_norm(x, weight, bias)` via `aclnnAddLayerNorm`.
///
/// Replaces the 2-step `aclnnAdd + aclnnLayerNorm` used in the vision encoder
/// path. norm_bias (biasOptional) may be None (NULL pointer).
pub fn add_layer_norm(ctx: &AscendNewContext, mut args: AddLayerNormArgs<'_>) -> Result<()> {
    let stream = ctx.raw_stream();
    let dtype = args.x1.dtype();
    let shape = args.x1.shape().clone();

    let norm_dims = shape.dims();
    let norm_shape_dim = norm_dims[norm_dims.len() - 1];
    let rows = shape.numel() / norm_shape_dim;
    let mean = ctx.allocate_output(Shape::new(vec![rows, 1]), DType::F32)?;
    let rstd = ctx.allocate_output(Shape::new(vec![rows, 1]), DType::F32)?;

    let acl_x1 = acl_tensor(args.x1, dtype);
    let acl_x2 = acl_tensor(args.x2, dtype);
    let acl_weight = acl_tensor(args.weight, dtype);
    let acl_bias = acl_tensor(args.bias, dtype);
    let acl_norm_bias = args.norm_bias.map(|t| acl_tensor(t, dtype));
    let acl_out = acl_tensor(args.normalized, dtype);
    let acl_mean = acl_tensor(&mean, DType::F32);
    let acl_rstd = acl_tensor(&rstd, DType::F32);
    let acl_hidden = args.hidden.as_deref_mut().map(|t| acl_tensor(t, dtype));

    let norm_shape_arr: Vec<i64> = vec![norm_shape_dim as i64];
    let norm_shape = AclIntArray::new(&norm_shape_arr);

    run_op(
        |ws, exec| unsafe {
            ffi::aclnnAddLayerNormGetWorkspaceSize(
                acl_x1.raw(),
                acl_x2.raw(),
                acl_weight.raw(),
                acl_bias.raw(),
                acl_norm_bias.as_ref().map_or(std::ptr::null(), |t| t.raw()),
                args.eps as f64,
                false, // additional_output — not used, we want y/rstd/x only
                acl_out.raw(),
                acl_mean.raw(),
                acl_rstd.raw(),
                acl_hidden.as_ref().map_or(std::ptr::null(), |t| t.raw()),
                ws,
                exec,
            )
        },
        |ws, ws_size, exec, s| unsafe { ffi::aclnnAddLayerNorm(ws, ws_size, exec, s) },
        stream,
        "aclnnAddLayerNorm",
    )
    .map_err(|e| Error::Other(format!("add_layer_norm: {e}")))
}

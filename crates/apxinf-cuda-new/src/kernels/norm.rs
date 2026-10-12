//! Legacy `kernels::norm` names over cuda-new norm operators.

use apxinf_core::{DType, Result, Shape, Tensor};

use crate::{ops, CudaContext};

/// `rms_bf16`: RMS-normalize a `[rows, cols]` BF16 activation with a plain
/// `[cols]` scale.
///
/// Routes to `ops::norm::rms_norm` (the plain `x/rms * weight` contract the
/// legacy `kernels::norm::rms_bf16` implements). The `ops::mlp::rms_norm`
/// operator is a different, Gemma/Qwen3.5-convention norm that applies
/// `(1 + weight)` and must not be used here.
pub fn rms_bf16(ctx: &CudaContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let mut output =
        ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::rms_norm(ctx, ops::RmsNormArgs::new(input, weight, &mut output, eps))?;
    Ok(output)
}

/// `layer_bf16`: LayerNorm with weight and bias.
pub fn layer_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::layer_norm(ctx, ops::LayerNormArgs::new(input, weight, bias, &mut output, eps))?;
    Ok(output)
}

/// `rms_quantize_rows_bf16_e4m3`: RMS-normalize then rowwise-quantize to
/// E4M3, zero-padding to `output_cols`. Returns values plus per-row scales.
pub fn rms_quantize_rows_bf16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    output_cols: usize,
) -> Result<super::quantization::DynamicFp8Tensor> {
    use crate::ffi::abi::{quant_fused as abi, status};
    use crate::CudaBuffer;
    let dims = input.shape().dims();
    if dims.len() != 2 || input.dtype() != DType::BF16 {
        return Err(apxinf_core::Error::Other(
            "fused RMS quantization requires a rank-2 BF16 input".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if output_cols < cols {
        return Err(apxinf_core::Error::Other(
            "fused RMS quantization output narrower than input".into(),
        ));
    }
    let values = ctx.allocate_output(Shape::new(vec![rows, output_cols]), DType::F8E4M3)?;
    let scales = ctx.allocate_output(Shape::new(vec![rows]), DType::F32)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| apxinf_core::Error::Other(format!("{what} exceeds i32")))
    };
    let input_buffer = CudaBuffer::from_tensor(input).map_err(apxinf_core::Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight).map_err(apxinf_core::Error::Cuda)?;
    let values_buffer = CudaBuffer::from_tensor(&values).map_err(apxinf_core::Error::Cuda)?;
    let scales_buffer = CudaBuffer::from_tensor(&scales).map_err(apxinf_core::Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_quant_rms_norm_rows_bf16_e4m3(
            input_buffer.ptr(),
            weight_buffer.ptr(),
            values_buffer.ptr(),
            scales_buffer.ptr(),
            to_i32(rows, "rows")?,
            to_i32(cols, "cols")?,
            to_i32(output_cols, "output cols")?,
            eps,
            ctx.stream().handle(),
        ))?;
    }
    Ok(super::quantization::DynamicFp8Tensor { values, scales })
}

/// `adaptive_layer`: adaptive LayerNorm — `norm(x) * (1 + scale) + shift`,
/// where `modulation` packs `[scale; shift]` as a `[2*cols]` vector.
pub fn adaptive_layer(
    ctx: &CudaContext,
    input: &Tensor,
    modulation: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    use crate::ffi::abi::{gr00t as abi, status};
    use crate::CudaBuffer;
    let dims = input.shape().dims();
    let modulation_dims = modulation.shape().dims();
    if dims.len() != 2
        || input.dtype() != DType::BF16
        || modulation.dtype() != DType::BF16
        || modulation_dims != [dims[1] * 2]
        || dims[0] == 0
        || dims[1] == 0
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(apxinf_core::Error::Other(
            "adaptive LayerNorm expects aligned BF16 input/modulation and positive epsilon".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let to_u32 = |value: usize, what: &str| {
        u32::try_from(value).map_err(|_| apxinf_core::Error::Other(format!("{what} exceeds u32")))
    };
    let input_buffer = CudaBuffer::from_tensor(input).map_err(apxinf_core::Error::Cuda)?;
    let modulation_buffer = CudaBuffer::from_tensor(modulation).map_err(apxinf_core::Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(apxinf_core::Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_gr00t_adaptive_layer_norm_bf16(
            input_buffer.ptr(),
            modulation_buffer.ptr(),
            output_buffer.ptr(),
            to_u32(rows, "rows")?,
            to_u32(cols, "cols")?,
            eps,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}

/// `adaptive_layer_quant_bf16_e4m3`: adaptive LayerNorm plus a calibrated
/// E4M3 view of the rounded normalized output. Returns `(bf16, e4m3)`.
pub fn adaptive_layer_quant_bf16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    modulation: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<(Tensor, Tensor)> {
    use crate::ffi::abi::{gr00t as abi, status};
    use crate::CudaBuffer;
    let dims = input.shape().dims();
    let modulation_dims = modulation.shape().dims();
    if dims.len() != 2
        || input.dtype() != DType::BF16
        || modulation.dtype() != DType::BF16
        || modulation_dims != [dims[1] * 2]
        || dims[0] == 0
        || dims[1] == 0
        || !eps.is_finite()
        || eps <= 0.0
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(apxinf_core::Error::Other(
            "adaptive LayerNorm quantization expects aligned BF16 input/modulation and positive epsilon/scale".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let bf16 = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let fp8 = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::F8E4M3)?;
    let to_u32 = |value: usize, what: &str| {
        u32::try_from(value).map_err(|_| apxinf_core::Error::Other(format!("{what} exceeds u32")))
    };
    let input_buffer = CudaBuffer::from_tensor(input).map_err(apxinf_core::Error::Cuda)?;
    let modulation_buffer = CudaBuffer::from_tensor(modulation).map_err(apxinf_core::Error::Cuda)?;
    let bf16_buffer = CudaBuffer::from_tensor(&bf16).map_err(apxinf_core::Error::Cuda)?;
    let fp8_buffer = CudaBuffer::from_tensor(&fp8).map_err(apxinf_core::Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_gr00t_adaptive_layer_norm_quant_bf16_e4m3(
            input_buffer.ptr(),
            modulation_buffer.ptr(),
            bf16_buffer.ptr(),
            fp8_buffer.ptr(),
            to_u32(rows, "rows")?,
            to_u32(cols, "cols")?,
            eps,
            scale,
            ctx.stream().handle(),
        ))?;
    }
    Ok((bf16, fp8))
}

/// `rms_quant_bf16_e4m3`: RMS-normalize then apply a static E4M3 scale.
pub fn rms_quant_bf16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    use crate::ffi::abi::{gr00t as abi, status};
    use crate::CudaBuffer;
    let dims = input.shape().dims();
    if dims.len() != 2
        || input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || weight.shape().dims() != [dims[1]]
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(apxinf_core::Error::Other(
            "static inference BF16 RMSNorm quantization has incompatible dtype, shape, or scale"
                .into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::F8E4M3)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| apxinf_core::Error::Other(format!("{what} exceeds i32")))
    };
    let input_buffer = CudaBuffer::from_tensor(input).map_err(apxinf_core::Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight).map_err(apxinf_core::Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(apxinf_core::Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_gr00t_rms_norm_quant_bf16_e4m3(
            input_buffer.ptr(),
            weight_buffer.ptr(),
            output_buffer.ptr(),
            to_i32(rows, "rows")?,
            to_i32(cols, "cols")?,
            eps,
            scale,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}

/// `layer_quant_bf16_e4m3`: LayerNorm then apply a static E4M3 scale.
pub fn layer_quant_bf16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    use crate::ffi::abi::{gr00t as abi, status};
    use crate::CudaBuffer;
    let dims = input.shape().dims();
    if dims.len() != 2
        || input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || bias.dtype() != DType::BF16
        || weight.shape().dims() != [dims[1]]
        || bias.shape().dims() != [dims[1]]
        || !eps.is_finite()
        || eps <= 0.0
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(apxinf_core::Error::Other(
            "static inference LayerNorm expects BF16 [rows,cols] and BF16 [cols] affine tensors"
                .into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::F8E4M3)?;
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| apxinf_core::Error::Other(format!("{what} exceeds i32")))
    };
    let input_buffer = CudaBuffer::from_tensor(input).map_err(apxinf_core::Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight).map_err(apxinf_core::Error::Cuda)?;
    let bias_buffer = CudaBuffer::from_tensor(bias).map_err(apxinf_core::Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(apxinf_core::Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_gr00t_layer_norm_quant_bf16_e4m3(
            input_buffer.ptr(),
            weight_buffer.ptr(),
            bias_buffer.ptr(),
            output_buffer.ptr(),
            to_i32(rows, "rows")?,
            to_i32(cols, "cols")?,
            eps,
            scale,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}

// ── smolvla FP16 surface ────────────────────────────────────────────────────
//
// Ported verbatim from the legacy `kernels::norm` FP16 wrappers over the
// `apxinf_smolvla_*` launchers (native/adapters/smolvla/ops.cu).

/// `rms_f16`: RMS-normalize a `[rows, cols]` FP16 activation with a plain
/// `[cols]` scale (the Xavier FP16 path).
pub fn rms_f16(ctx: &CudaContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    use super::contracts::{f16_output, gpu_ptr, make_gpu_tensor, matrix_shape};
    use crate::ffi::abi::smolvla as abi;
    use apxinf_core::Error;
    let (rows, cols) = matrix_shape(input, "RMSNorm")?;
    if input.dtype() != DType::F16
        || weight.dtype() != DType::F16
        || weight.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "static inference FP16 RMSNorm shape mismatch".into(),
        ));
    }
    let output = f16_output(ctx, rows, cols)?;
    unsafe {
        super::contracts::check_cuda(abi::apxinf_smolvla_rms_norm_f16(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

/// `layer_f16`: LayerNorm with FP16 weight and bias.
pub fn layer_f16(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    use super::contracts::{f16_output, gpu_ptr, make_gpu_tensor, matrix_shape};
    use crate::ffi::abi::smolvla as abi;
    use apxinf_core::Error;
    let (rows, cols) = matrix_shape(input, "LayerNorm")?;
    if [input, weight, bias]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::F16)
        || weight.shape().dims() != [cols]
        || bias.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "static inference FP16 LayerNorm shape mismatch".into(),
        ));
    }
    let output = f16_output(ctx, rows, cols)?;
    unsafe {
        super::contracts::check_cuda(abi::apxinf_smolvla_layer_norm_f16(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

// ── Decode fast-path `_into` surface (caller-owned buffers, graph-safe) ───

/// RMS normalization into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn rms_into(
    ctx: &CudaContext,
    dtype: DType,
    input: &crate::CudaBuffer,
    weight: &crate::CudaBuffer,
    output: &crate::CudaBuffer,
    cols: usize,
    rows: usize,
    eps: f32,
) -> Result<()> {
    use super::contracts::{check_cuda, checked_bytes, require_buffers, require_finite};
    use crate::ffi::abi::llm_decode as abi;
    use apxinf_core::Error;
    require_finite("RMSNorm", &[eps])?;
    if eps <= 0.0 {
        return Err(Error::Other("RMSNorm epsilon must be positive".into()));
    }
    let matrix = checked_bytes(dtype, &[rows, cols], "RMSNorm")?;
    let weight_size = checked_bytes(dtype, &[cols], "RMSNorm")?;
    require_buffers(
        ctx,
        "RMSNorm",
        &[
            ("input", input, matrix),
            ("weight", weight, weight_size),
            ("output", output, matrix),
        ],
    )?;
    let status = unsafe {
        match dtype {
            DType::F32 => abi::apxinf_llmdec_rms_norm_f32(
                input.ptr(),
                weight.ptr(),
                output.ptr(),
                cols as u32,
                rows as u32,
                eps,
                ctx.stream().handle(),
            ),
            DType::BF16 => abi::apxinf_llmdec_rms_norm_bf16(
                input.ptr(),
                weight.ptr(),
                output.ptr(),
                cols as u32,
                rows as u32,
                eps,
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(Error::Other(format!(
                    "decode RMSNorm does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}

/// Fused residual add and BF16 RMS normalization into caller-owned storage.
/// Writes `residual + delta` back into `residual` and the normed result into
/// `output`, exactly like the legacy `rms_norm_add_bf16`.
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rms_bf16_into(
    ctx: &CudaContext,
    residual: &crate::CudaBuffer,
    delta: &crate::CudaBuffer,
    weight: &crate::CudaBuffer,
    output: &crate::CudaBuffer,
    cols: usize,
    rows: usize,
    eps: f32,
) -> Result<()> {
    use super::contracts::{check_cuda, checked_bytes, require_buffers, require_finite};
    use crate::ffi::abi::llm_decode as abi;
    require_finite("residual RMSNorm", &[eps])?;
    let matrix = checked_bytes(DType::BF16, &[rows, cols], "residual RMSNorm")?;
    let weight_size = checked_bytes(DType::BF16, &[cols], "residual RMSNorm")?;
    require_buffers(
        ctx,
        "residual RMSNorm",
        &[
            ("residual", residual, matrix),
            ("delta", delta, matrix),
            ("weight", weight, weight_size),
            ("output", output, matrix),
        ],
    )?;
    check_cuda(unsafe {
        abi::apxinf_llmdec_rms_norm_add_bf16(
            residual.ptr(),
            delta.ptr(),
            weight.ptr(),
            output.ptr(),
            cols as u32,
            rows as u32,
            eps,
            ctx.stream().handle(),
        )
    })
}

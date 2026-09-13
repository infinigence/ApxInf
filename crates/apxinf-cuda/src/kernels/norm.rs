//! Normalization operator contracts.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, check_cuda, checked_bytes, fp8_output, gpu_ptr, make_gpu_tensor, matrix_shape,
    matrix_tensor, require_buffers, require_finite, unsupported_dtype,
};
use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;

/// Inference BatchNorm + ReLU for contiguous NCHW or NCDHW BF16 input.
/// `invstd` contains FP32 1/sqrt(running_variance + epsilon), prepared at load.
pub fn batch_relu_bf16(
    ctx: &CudaContext,
    x: &Tensor,
    mean: &Tensor,
    invstd: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let d = x.shape().dims();
    if !matches!(d.len(), 4 | 5) || d.contains(&0) {
        return Err(Error::Other(
            "BatchNorm requires nonempty NCHW or NCDHW input".into(),
        ));
    }
    for t in [mean, invstd, weight, bias] {
        if t.shape().dims() != [d[1]] {
            return Err(Error::Other("BatchNorm channel count mismatch".into()));
        }
    }
    for (t, dtype) in [
        (x, DType::BF16),
        (mean, DType::BF16),
        (weight, DType::BF16),
        (bias, DType::BF16),
        (invstd, DType::F32),
    ] {
        if t.dtype() != dtype || t.device() != apxinf_core::Device::Cuda(ctx.device_id()) {
            return Err(Error::Other("BatchNorm dtype or device mismatch".into()));
        }
        checked_bytes(dtype, t.shape().dims(), "BatchNorm")?;
    }
    let bytes = checked_bytes(DType::BF16, d, "BatchNorm output")?;
    let spatial = d[2..]
        .iter()
        .try_fold(1usize, |a, &b| a.checked_mul(b))
        .ok_or_else(|| Error::Other("BatchNorm spatial overflow".into()))?;
    let int = |n: usize| {
        i32::try_from(n).map_err(|_| Error::Other("BatchNorm dimension overflow".into()))
    };
    let count = i64::try_from(bytes / 2)
        .map_err(|_| Error::Other("BatchNorm element count overflow".into()))?;
    let out = output_buffer(ctx, bytes)?;
    unsafe {
        check_cuda(ffi::apxinf_batch_norm_relu_bf16(
            gpu_ptr(x)?,
            gpu_ptr(mean)?,
            gpu_ptr(invstd)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            out.ptr(),
            int(d[1])?,
            int(spatial)?,
            count,
            ctx.stream().handle(),
        ))?;
    }
    Ok(make_gpu_tensor(
        x.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        out,
    ))
}

/// NCHW GroupNorm with BF16 mean/rstd/epsilon and FP32 affine arithmetic.
pub fn group_bf16_rounded(
    ctx: &CudaContext,
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    groups: usize,
    eps: f32,
) -> Result<Tensor> {
    let d = x.shape().dims();
    if d.len() != 4
        || groups == 0
        || d[1] % groups != 0
        || weight.shape().dims() != [d[1]]
        || bias.shape() != weight.shape()
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other(
            "invalid NCHW GroupNorm geometry or epsilon".into(),
        ));
    }
    for t in [x, weight, bias] {
        if t.dtype() != DType::BF16 || t.device() != apxinf_core::Device::Cuda(ctx.device_id()) {
            return Err(Error::Other(
                "GroupNorm requires BF16 tensors on the context device".into(),
            ));
        }
        checked_bytes(DType::BF16, t.shape().dims(), "GroupNorm")?;
    }
    let int = |v: usize| {
        i32::try_from(v).map_err(|_| Error::Other("GroupNorm dimension overflow".into()))
    };
    int(d[0]
        .checked_mul(groups)
        .ok_or_else(|| Error::Other("GroupNorm grid overflow".into()))?)?;
    let spatial = d[2]
        .checked_mul(d[3])
        .ok_or_else(|| Error::Other("GroupNorm spatial overflow".into()))?;
    let out = output_buffer(ctx, checked_bytes(DType::BF16, d, "GroupNorm output")?)?;
    unsafe {
        check_cuda(ffi::apxinf_group_norm_bf16_rounded(
            gpu_ptr(x)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            out.ptr(),
            int(d[0])?,
            int(d[1])?,
            int(spatial)?,
            int(groups)?,
            eps,
            ctx.stream().handle(),
        ))?;
    }
    Ok(make_gpu_tensor(
        x.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        out,
    ))
}

/// NCHW channel LayerNorm with BF16 arithmetic boundaries, including the affine step.
pub fn channel_layer_bf16_rounded(
    ctx: &CudaContext,
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let dims = x.shape().dims();
    if dims.len() != 4 || !eps.is_finite() || eps <= 0.0 {
        return Err(Error::Other("invalid channel LayerNorm geometry".into()));
    }
    let bytes = checked_bytes(DType::BF16, dims, "channel LayerNorm")?;
    if weight.shape().dims() != [dims[1]] || bias.shape() != weight.shape() {
        return Err(Error::Other(
            "channel LayerNorm affine width mismatch".into(),
        ));
    }
    for t in [x, weight, bias] {
        if t.dtype() != DType::BF16 || t.device() != apxinf_core::Device::Cuda(ctx.device_id()) {
            return Err(Error::Other(
                "channel LayerNorm requires BF16 tensors on the context device".into(),
            ));
        }
    }
    let int = |n: usize| {
        i32::try_from(n).map_err(|_| Error::Other("channel LayerNorm dimension overflow".into()))
    };
    let spatial = dims[2]
        .checked_mul(dims[3])
        .ok_or_else(|| Error::Other("channel LayerNorm spatial overflow".into()))?;
    int(dims[0]
        .checked_mul(spatial)
        .ok_or_else(|| Error::Other("channel LayerNorm grid overflow".into()))?)?;
    let (n, c, s) = (int(dims[0])?, int(dims[1])?, int(spatial)?);
    let out = output_buffer(ctx, bytes)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_channel_layer_norm_bf16_rounded(
            gpu_ptr(x)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            out.ptr(),
            n,
            c,
            s,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(out.into_tensor(x.shape().clone(), DType::BF16))
}

/// RMS normalization into caller-owned storage.
#[allow(clippy::too_many_arguments)]
pub fn rms_into(
    ctx: &CudaContext,
    dtype: DType,
    input: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    cols: usize,
    rows: usize,
    eps: f32,
) -> Result<()> {
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
            DType::F32 => ffi::apxinf_rms_norm_f32(
                input.ptr(),
                weight.ptr(),
                output.ptr(),
                cols as u32,
                rows as u32,
                eps,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_rms_norm_bf16(
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
#[allow(clippy::too_many_arguments)]
pub fn residual_add_rms_bf16_into(
    ctx: &CudaContext,
    residual: &CudaBuffer,
    delta: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    cols: usize,
    rows: usize,
    eps: f32,
) -> Result<()> {
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
        ffi::apxinf_rms_norm_add_bf16(
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

/// RMS normalization on CUDA. Dispatches on `input.dtype()`.
pub fn rms(ctx: &CudaContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let rows = if dims.len() == 1 { 1 } else { dims[0] };
    let cols = if dims.len() == 1 {
        dims[0]
    } else {
        dims[dims.len() - 1]
    };

    let out_bytes = input.size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match input.dtype() {
            DType::F32 => ffi::apxinf_rms_norm_f32(
                gpu_ptr(input)?,
                gpu_ptr(weight)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                eps,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_rms_norm_bf16(
                gpu_ptr(input)?,
                gpu_ptr(weight)?,
                out_buf.ptr(),
                cols as u32,
                rows as u32,
                eps,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("rms_norm", dtype),
        };
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        input.shape().clone(),
        input.dtype(),
        device_id,
        out_buf,
    ))
}

/// LayerNorm (bf16 only). `input` shape `[rows, cols]` (or `[cols]` for
/// rows=1). Weight + bias are `[cols]`. Vision blocks have both.
pub fn layer(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if input.dtype() != DType::BF16 {
        return Err(Error::Other("layer_norm: only BF16 supported".into()));
    }
    let device_id = ctx.device_id();
    let dims = input.shape().dims();
    let rows = if dims.len() == 1 { 1 } else { dims[0] };
    let cols = if dims.len() == 1 {
        dims[0]
    } else {
        dims[dims.len() - 1]
    };
    let out_buf = CudaBuffer::alloc_zeros(input.size_in_bytes(), device_id).map_err(Error::Cuda)?;
    unsafe {
        let res = ffi::apxinf_layer_norm_bf16(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            out_buf.ptr(),
            cols as u32,
            rows as u32,
            eps,
            ctx.stream().handle(),
        );
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::BF16,
        device_id,
        out_buf,
    ))
}
pub fn rms_bf16(ctx: &CudaContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "RMSNorm")?;
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || weight.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "static inference BF16 RMSNorm shape mismatch".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_rms_norm_bf16(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, cols, output))
}

/// Fuse BF16 RMSNorm with dynamic per-row E4M3 quantization.
pub fn rms_quantize_rows_bf16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    output_cols: usize,
) -> Result<super::quantization::DynamicFp8Tensor> {
    let (rows, cols) = matrix_shape(input, "dynamic RMSNorm quantization")?;
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || weight.shape().dims() != [cols]
        || output_cols < cols
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other(
            "dynamic RMSNorm quantization has incompatible input".into(),
        ));
    }
    let values = output_buffer(
        ctx,
        rows.checked_mul(output_cols)
            .ok_or_else(|| Error::Other("dynamic RMSNorm output size overflow".into()))?,
    )?;
    let scales = output_buffer(ctx, rows * DType::F32.size_in_bytes())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_dynamic_rms_norm_quantize_rows_bf16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            values.ptr(),
            scales.ptr(),
            rows as i32,
            cols as i32,
            output_cols as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(super::quantization::DynamicFp8Tensor {
        values: make_gpu_tensor(
            Shape::new(vec![rows, output_cols]),
            DType::F8E4M3,
            ctx.device_id(),
            values,
        ),
        scales: make_gpu_tensor(Shape::new(vec![rows]), DType::F32, ctx.device_id(), scales),
    })
}

pub fn layer_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "LayerNorm")?;
    if [input, weight, bias]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || weight.shape().dims() != [cols]
        || bias.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "static inference BF16 LayerNorm shape mismatch".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_layer_norm_bf16(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, cols, output))
}

pub fn adaptive_rms_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    style: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "AdaRMSNorm")?;
    if input.dtype() != DType::BF16
        || style.dtype() != DType::BF16
        || style.shape().dims() != [3 * cols]
    {
        return Err(Error::Other(
            "static inference BF16 AdaRMSNorm shape mismatch".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_ada_rms_norm_bf16(
            gpu_ptr(input)?,
            gpu_ptr(style)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, cols, output))
}
pub fn rms_quant_f16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "RMSNorm")?;
    if input.dtype() != DType::F16
        || weight.dtype() != DType::F16
        || weight.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "static inference RMSNorm expects FP16 [rows,cols] and FP16 [cols] scale".into(),
        ));
    }
    let output = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_rms_norm_quant_f16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}

pub fn rms_quant_bf16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "BF16 RMSNorm quantization")?;
    if input.dtype() != DType::BF16
        || weight.dtype() != DType::BF16
        || weight.shape().dims() != [cols]
        || !scale.is_finite()
        || scale <= 0.0
    {
        return Err(Error::Other(
            "static inference BF16 RMSNorm quantization has incompatible dtype, shape, or scale"
                .into(),
        ));
    }
    let output = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_rms_norm_quant_bf16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}

pub fn layer_quant_f16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "LayerNorm")?;
    if input.dtype() != DType::F16
        || weight.dtype() != DType::F16
        || bias.dtype() != DType::F16
        || weight.shape().dims() != [cols]
        || bias.shape().dims() != [cols]
    {
        return Err(Error::Other(
            "static inference LayerNorm expects FP16 [rows,cols] and FP16 [cols] affine tensors"
                .into(),
        ));
    }
    let output = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_layer_norm_quant_f16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            gpu_ptr(bias)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}

pub fn adaptive_rms_quant_f16_e4m3(
    ctx: &CudaContext,
    input: &Tensor,
    style: &Tensor,
    eps: f32,
    scale: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "AdaRMSNorm")?;
    if input.dtype() != DType::F16
        || style.dtype() != DType::F16
        || style.shape().dims() != [3 * cols]
    {
        return Err(Error::Other(
            "static inference AdaRMSNorm expects FP16 input and [3*cols] style".into(),
        ));
    }
    let output = fp8_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_ada_rms_norm_quant_f16_e4m3(
            gpu_ptr(input)?,
            gpu_ptr(style)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}

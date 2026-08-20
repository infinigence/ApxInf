use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::device_caps::CudaArchFamily;
use crate::ffi;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W8A8ScaleMode {
    DynamicRowPerOutputChannel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W8A8Layout {
    OutputMajor,
}

/// Borrowed W8A8 weight view prevents values/scales/layout mismatches.
#[derive(Clone, Copy)]
pub struct W8A8WeightView<'a> {
    pub values_i8: &'a CudaBuffer,
    pub scales_f32: &'a Tensor,
    pub input_dim: usize,
    pub output_dim: usize,
    pub scale_mode: W8A8ScaleMode,
    pub layout: W8A8Layout,
}

/// Dynamic-row-quantized W8A8 GEMM with BF16 output.
pub fn gemm_w8a8(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W8A8WeightView<'_>,
) -> Result<Tensor> {
    let prefer_cutlass = matches!(ctx.caps().arch_family, CudaArchFamily::Sm80);
    gemm_w8a8_with_preference(ctx, activation, weight, prefer_cutlass)
}

pub(crate) fn gemm_w8a8_with_preference(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W8A8WeightView<'_>,
    prefer_cutlass: bool,
) -> Result<Tensor> {
    if activation.dtype() != DType::BF16 || activation.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::Other(format!(
            "gemm_w8a8 expects a BF16 activation on CUDA {}, got {} on {}",
            ctx.device_id(),
            activation.dtype(),
            activation.device()
        )));
    }
    if weight.scale_mode != W8A8ScaleMode::DynamicRowPerOutputChannel
        || weight.layout != W8A8Layout::OutputMajor
    {
        return Err(Error::Other(
            "gemm_w8a8 received an unsupported scale mode or layout".into(),
        ));
    }
    let dims = activation.shape().dims();
    if dims.len() != 2 || dims[1] != weight.input_dim {
        return Err(Error::Other(format!(
            "gemm_w8a8 activation shape mismatch: expected [M,{}], got {dims:?}",
            weight.input_dim
        )));
    }
    if weight.values_i8.device() != ctx.device_id()
        || weight.values_i8.len() != weight.input_dim * weight.output_dim
        || weight.scales_f32.dtype() != DType::F32
        || weight.scales_f32.device() != Device::Cuda(ctx.device_id())
        || weight.scales_f32.shape().dims() != [weight.output_dim]
    {
        return Err(Error::Other(format!(
            "gemm_w8a8 weight contract mismatch: bytes {}, scales {} {:?}, expected [{},{}] on CUDA {}",
            weight.values_i8.len(),
            weight.scales_f32.dtype(),
            weight.scales_f32.shape().dims(),
            weight.output_dim,
            weight.input_dim,
            ctx.device_id()
        )));
    }

    let rows = dims[0];
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_scales = CudaBuffer::from_tensor(weight.scales_f32).map_err(Error::Cuda)?;
    let quantized = crate::workspace::output_buffer(ctx, rows * weight.input_dim)?;
    let row_scales = crate::workspace::output_buffer(ctx, rows * std::mem::size_of::<f32>())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_quantize_rows_bf16_int8(
            activation.ptr(),
            quantized.ptr(),
            row_scales.ptr(),
            rows as i32,
            weight.input_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }

    let output = crate::workspace::output_buffer(
        ctx,
        rows * weight.output_dim * DType::BF16.size_in_bytes(),
    )?;
    #[cfg(not(apxinf_cutlass_int8_sm80))]
    let _ = prefer_cutlass;
    #[cfg(apxinf_cutlass_int8_sm80)]
    if prefer_cutlass && weight.input_dim % 16 == 0 && weight.output_dim % 8 == 0 {
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_cutlass_int8_gemm_bf16(
                quantized.ptr(),
                weight.values_i8.ptr(),
                row_scales.ptr(),
                weight_scales.ptr(),
                output.ptr(),
                rows as i32,
                weight.output_dim as i32,
                weight.input_dim as i32,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        return Ok(output.into_tensor(Shape::new(vec![rows, weight.output_dim]), DType::BF16));
    }

    let accumulators = crate::workspace::output_buffer(
        ctx,
        rows * weight.output_dim * std::mem::size_of::<i32>(),
    )?;
    ctx.cublas()
        .gemm_int8_i32(
            rows,
            weight.output_dim,
            weight.input_dim,
            &quantized,
            weight.values_i8,
            &accumulators,
        )
        .map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_dequantize_int32_bf16(
            accumulators.ptr(),
            row_scales.ptr(),
            weight_scales.ptr(),
            output.ptr(),
            rows as i32,
            weight.output_dim as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output.into_tensor(Shape::new(vec![rows, weight.output_dim]), DType::BF16))
}

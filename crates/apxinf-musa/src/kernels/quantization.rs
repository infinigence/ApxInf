//! FP8 quantization operator contracts.
//!
//! MP31 implements static per-tensor BF16 to E4M3 conversion. Other precision
//! producers retain named unsupported errors instead of changing semantics.

use apxinf_core::{Result, Tensor};

use crate::context::MusaContext;
use crate::kernels::not_ported;

/// A per-row dynamically quantized FP8 tensor.
pub struct DynamicFp8Tensor {
    pub values: Tensor,
    pub scales: Tensor,
}

/// Quantize an FP16 tensor to E4M3 with a per-tensor scale.
pub fn quantize_f16_e4m3(_ctx: &MusaContext, _input: &Tensor, _scale: f32) -> Result<Tensor> {
    Err(not_ported("quantization::quantize_f16_e4m3"))
}

/// Quantize a BF16 tensor to E4M3 with a per-tensor scale.
pub fn quantize_bf16_e4m3(ctx: &MusaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = (ctx, input, scale);
        Err(not_ported("quantization::quantize_bf16_e4m3"))
    }
    #[cfg(apxinf_musa_available)]
    {
        use apxinf_core::{DType, Device, Error};
        use super::contracts::{checked_bytes, output_tensor, tensor_ptr};
        if crate::MUSA_ARCH != "mp_31" || (ctx.caps().compute_major, ctx.caps().compute_minor) != (3, 1)
            || input.dtype() != DType::BF16
            || input.device() != Device::Musa(ctx.device_id())
            || !scale.is_finite() || scale <= 0.0 || !scale.recip().is_finite()
        {
            return Err(Error::Other("FP8 conversion requires MP31 BF16 input and a finite positive scale".into()));
        }
        let count = checked_bytes(DType::BF16, input.shape().dims(), "FP8 conversion")? / 2;
        let output = output_tensor(ctx, input.shape().clone(), DType::F8E4M3,
            crate::workspace::output_buffer(ctx, count)?);
        if count > 0 {
            unsafe {
                crate::ffi::check_runtime(crate::ffi::custom::apxinf_musa_quantize_bf16_e4m3(
                    tensor_ptr(input)?, tensor_ptr(&output)?, count, scale, ctx.stream().handle(),
                )).map_err(crate::error::musa_error)?;
            }
        }
        Ok(output)
    }
}

pub fn concat_rows_quantize_bf16_e4m3(
    _ctx: &MusaContext,
    _first: &Tensor,
    _second: &Tensor,
    _scale: f32,
) -> Result<Tensor> {
    Err(not_ported("quantization::concat_rows_quantize_bf16_e4m3"))
}

/// Quantize a BF16 tensor to E4M3 one row at a time.
pub fn quantize_rows_bf16_e4m3(_ctx: &MusaContext, _input: &Tensor) -> Result<DynamicFp8Tensor> {
    Err(not_ported("quantization::quantize_rows_bf16_e4m3"))
}

/// Row quantization into a padded output width.
pub fn quantize_rows_bf16_e4m3_padded(
    _ctx: &MusaContext,
    _input: &Tensor,
    _output_cols: usize,
) -> Result<DynamicFp8Tensor> {
    Err(not_ported("quantization::quantize_rows_bf16_e4m3_padded"))
}

/// Take the first `output_cols` columns of a BF16 tensor.
pub fn slice_columns_bf16(
    _ctx: &MusaContext,
    _input: &Tensor,
    _output_cols: usize,
) -> Result<Tensor> {
    Err(not_ported("quantization::slice_columns_bf16"))
}

/// Cast FP16 to BF16.
pub fn cast_f16_bf16(_ctx: &MusaContext, _input: &Tensor) -> Result<Tensor> {
    Err(not_ported("quantization::cast_f16_bf16"))
}

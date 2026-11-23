//! KV-cache storage operator contracts.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, f16_output, matrix_shape, matrix_tensor, output_tensor, tensor_ptr,
};
use crate::context::MusaContext;
use crate::error::musa_error;
use crate::ffi;

/// Allocate a persistent K/V cache and copy the prefix into its first rows.
///
/// Native: this is one device-to-device copy on the context's stream. The
/// **tail keeps the arena's previous contents** — `output_buffer` does not
/// clear a reused view — and callers rely on that: `attention::mqa_bf16`
/// reads only the first `key_tokens` rows precisely because the rest is
/// unspecified.
pub fn reserve_prefix_bf16(
    ctx: &MusaContext,
    prefix: &Tensor,
    total_rows: usize,
) -> Result<Tensor> {
    let (prefix_rows, cols) = matrix_shape(prefix, "prefix KV cache")?;
    if prefix.dtype() != DType::BF16 || total_rows < prefix_rows {
        return Err(Error::Other(
            "static inference BF16 prefix KV cache has incompatible shape".into(),
        ));
    }
    let output = bf16_output(ctx, total_rows, cols)?;
    unsafe {
        ffi::check_runtime(ffi::musa::musaMemcpyAsync(
            output.ptr(),
            tensor_ptr(prefix)?,
            prefix.size_in_bytes(),
            ffi::musa::MUSA_MEMCPY_DEVICE_TO_DEVICE,
            ctx.stream().handle(),
        ))
        .map_err(musa_error)?;
    }
    // No wait. The copy is on the context's stream and every consumer of this
    // cache is a native operator on the same stream, so stream order is enough
    // -- and the blocking stream is additionally synchronised with the legacy
    // default stream (the port's stream-ordering probe), which is what would cover a
    // torch reader if one came back. A wait here would drain the device once per
    // layer for nothing: the reader that motivated it was a bridged op on
    // torch's stream, and the port has no bridged operators.
    Ok(matrix_tensor(ctx, total_rows, cols, output))
}

/// Allocate a persistent FP16 K/V cache and copy the prefix into its first rows.
pub fn reserve_prefix_f16(ctx: &MusaContext, prefix: &Tensor, total_rows: usize) -> Result<Tensor> {
    let (prefix_rows, cols) = matrix_shape(prefix, "prefix KV cache")?;
    if prefix.dtype() != DType::F16 || total_rows < prefix_rows {
        return Err(Error::Other(format!(
            "static inference prefix KV cache expected FP16 with total_rows >= {prefix_rows}, got {:?} and {total_rows}",
            prefix.shape().dims()
        )));
    }
    let output = f16_output(ctx, total_rows, cols)?;
    unsafe {
        ffi::check_runtime(ffi::musa::musaMemcpyAsync(
            output.ptr(),
            tensor_ptr(prefix)?,
            prefix.size_in_bytes(),
            ffi::musa::MUSA_MEMCPY_DEVICE_TO_DEVICE,
            ctx.stream().handle(),
        ))
        .map_err(musa_error)?;
    }
    // See `reserve_prefix_bf16`: no wait is needed now that no reader is bridged.
    Ok(output_tensor(
        ctx,
        Shape::new(vec![total_rows, cols]),
        DType::F16,
        output,
    ))
}

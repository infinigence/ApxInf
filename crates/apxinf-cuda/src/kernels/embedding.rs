//! Embedding and positional-embedding operator contracts.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use super::contracts::{
    bf16_output, check_cuda, checked_bytes, f16_output, gpu_ptr, make_gpu_tensor, matrix_shape,
    matrix_tensor, optional_ptr, require_address, require_buffers, require_finite,
    unsupported_dtype,
};
use crate::buffer::{CudaBuffer, CudaDeviceAddress};
use crate::context::CudaContext;
use crate::ffi;

pub fn lookup_into(
    ctx: &CudaContext,
    dtype: DType,
    table: &CudaBuffer,
    ids: CudaDeviceAddress,
    output: &CudaBuffer,
    embed_dim: usize,
    seq_len: usize,
) -> Result<()> {
    require_buffers(
        ctx,
        "decode embedding",
        &[
            (
                "table",
                table,
                checked_bytes(dtype, &[embed_dim], "decode embedding")?,
            ),
            (
                "output",
                output,
                checked_bytes(dtype, &[seq_len, embed_dim], "decode embedding")?,
            ),
        ],
    )?;
    require_address(ctx, "decode embedding", "ids", ids, seq_len * 4)?;
    let status = unsafe {
        match dtype {
            DType::F32 => ffi::apxinf_embedding_f32(
                table.ptr(),
                ids.ptr(),
                output.ptr(),
                embed_dim as u32,
                seq_len as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_embedding_bf16(
                table.ptr(),
                ids.ptr(),
                output.ptr(),
                embed_dim as u32,
                seq_len as u32,
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(apxinf_core::Error::Other(format!(
                    "decode embedding does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}

/// Embedding lookup on CUDA. Dispatches on table.dtype().
pub fn lookup(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &CudaBuffer,
    seq_len: usize,
) -> Result<Tensor> {
    let device_id = ctx.device_id();
    let embed_dim = table.shape().dims()[1];

    let out_shape = Shape::new(vec![seq_len, embed_dim]);
    let out_bytes = out_shape.numel() * table.dtype().size_in_bytes();
    let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;

    unsafe {
        let res = match table.dtype() {
            DType::F32 => ffi::apxinf_embedding_f32(
                gpu_ptr(table)?,
                ids.ptr(),
                out_buf.ptr(),
                embed_dim as u32,
                seq_len as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => ffi::apxinf_embedding_bf16(
                gpu_ptr(table)?,
                ids.ptr(),
                out_buf.ptr(),
                embed_dim as u32,
                seq_len as u32,
                ctx.stream().handle(),
            ),
            dtype => return unsupported_dtype("embedding", dtype),
        };
        ffi::check_cuda(res).map_err(Error::Cuda)?;
    }

    Ok(make_gpu_tensor(
        out_shape,
        table.dtype(),
        device_id,
        out_buf,
    ))
}
pub fn lookup_bf16(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &CudaBuffer,
    tokens: usize,
) -> Result<Tensor> {
    let dims = table.shape().dims();
    if table.dtype() != DType::BF16 || dims.len() != 2 || ids.len() < tokens * 4 || tokens == 0 {
        return Err(Error::Other(
            "static inference BF16 embedding expects [vocab,width] and device u32 ids".into(),
        ));
    }
    let output = bf16_output(ctx, tokens, dims[1])?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_embedding_bf16(
            gpu_ptr(table)?,
            ids.ptr(),
            output.ptr(),
            tokens as i32,
            dims[1] as i32,
            dims[0] as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, tokens, dims[1], output))
}

/// BF16 embedding lookup with a caller-owned scalar convention.
///
/// `scale` is applied in FP32 to each BF16 table element immediately before
/// the single BF16 output rounding. This is intentionally distinct from
/// [`lookup_bf16`], whose legacy static-inference contract derives
/// `sqrt(width)` inside the CUDA kernel.
pub fn lookup_scaled_bf16(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &CudaBuffer,
    tokens: usize,
    scale: f32,
) -> Result<Tensor> {
    require_finite("scaled embedding", &[scale])?;
    let dims = table.shape().dims();
    if table.dtype() != DType::BF16
        || table.device() != apxinf_core::Device::Cuda(ctx.device_id())
        || dims.len() != 2
        || dims[0] == 0
        || dims[1] == 0
        || ids.device() != ctx.device_id()
        || ids.len() < tokens.saturating_mul(std::mem::size_of::<u32>())
        || tokens == 0
    {
        return Err(Error::Other(
            "scaled BF16 embedding expects a CUDA [vocab,width] table and device u32 ids".into(),
        ));
    }
    let tokens_i32 = i32::try_from(tokens)
        .map_err(|_| Error::Other("scaled embedding token count exceeds i32".into()))?;
    let width_i32 = i32::try_from(dims[1])
        .map_err(|_| Error::Other("scaled embedding width exceeds i32".into()))?;
    let vocab_i32 = i32::try_from(dims[0])
        .map_err(|_| Error::Other("scaled embedding vocabulary exceeds i32".into()))?;
    let output = bf16_output(ctx, tokens, dims[1])?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_embedding_scaled_bf16(
            gpu_ptr(table)?,
            ids.ptr(),
            output.ptr(),
            tokens_i32,
            width_i32,
            vocab_i32,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, tokens, dims[1], output))
}

pub fn add_position_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(projection, "position embedding")?;
    let position_dims = position.shape().dims();
    let position_ok =
        position_dims == [tokens_per_view, cols] || position_dims == [1, tokens_per_view, cols];
    if projection.dtype() != DType::BF16
        || position.dtype() != DType::BF16
        || !position_ok
        || rows % tokens_per_view != 0
        || bias.is_some_and(|value| value.dtype() != DType::BF16 || value.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference BF16 position embedding shape mismatch".into(),
        ));
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_position_bf16(
            gpu_ptr(projection)?,
            optional_ptr(bias)?,
            gpu_ptr(position)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            tokens_per_view as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(matrix_tensor(ctx, rows, cols, output))
}
/// PaliGemma input embedding, including the required `sqrt(hidden_size)`
/// normalization from OpenPI's `Embedder.encode`.
pub fn lookup_f16(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &CudaBuffer,
    tokens: usize,
) -> Result<Tensor> {
    let dims = table.shape().dims();
    if table.dtype() != DType::F16 || dims.len() != 2 || ids.len() < tokens * 4 || tokens == 0 {
        return Err(Error::Other(
            "static inference embedding expects FP16 [vocab,width] and device u32 ids".into(),
        ));
    }
    let output = f16_output(ctx, tokens, dims[1])?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_embedding_f16(
            gpu_ptr(table)?,
            ids.ptr(),
            output.ptr(),
            tokens as i32,
            dims[1] as i32,
            dims[0] as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![tokens, dims[1]]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

pub fn add_position_f16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(projection, "position embedding")?;
    let position_dims = position.shape().dims();
    let position_ok =
        position_dims == [tokens_per_view, cols] || position_dims == [1, tokens_per_view, cols];
    if projection.dtype() != DType::F16
        || position.dtype() != DType::F16
        || !position_ok
        || rows % tokens_per_view != 0
        || bias.is_some_and(|x| x.dtype() != DType::F16 || x.shape().dims() != [cols])
    {
        return Err(Error::Other(
            "static inference position embedding has incompatible dtype or shape".into(),
        ));
    }
    let output = f16_output(ctx, rows, cols)?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_bias_position_f16(
            gpu_ptr(projection)?,
            bias.map(gpu_ptr)
                .transpose()?
                .unwrap_or(std::ptr::null_mut()),
            gpu_ptr(position)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            tokens_per_view as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::F16,
        ctx.device_id(),
        output,
    ))
}

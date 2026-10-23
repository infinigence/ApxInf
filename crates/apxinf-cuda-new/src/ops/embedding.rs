//! Embedding lookup and position operators.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::ffi::abi::{elementwise as abi, status};
use crate::{CudaBuffer, CudaContext};

/// `lookup`: gather `seq_len` rows of `table` by u32 token ids held in a
/// device buffer — the unscaled legacy semantic. Callers that need the
/// Gemma-style `sqrt(width)` scale apply it separately, exactly as they did
/// against the legacy backend, so the rounding sequence is unchanged.
pub fn lookup(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &CudaBuffer,
    seq_len: usize,
) -> Result<Tensor> {
    let dims = table.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other("embedding lookup expects a rank-2 table".into()));
    }
    if table.dtype() != DType::BF16 {
        return Err(Error::Other(
            "cuda-new embedding lookup currently supports BF16 tables".into(),
        ));
    }
    let cols = dims[1];
    let output = ctx.allocate_output(Shape::new(vec![seq_len, cols]), DType::BF16)?;
    let table_buffer = CudaBuffer::from_tensor(table).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    let rows = i64::try_from(seq_len).map_err(|_| Error::Other("lookup rows exceed i64".into()))?;
    let cols_i64 = i64::try_from(cols).map_err(|_| Error::Other("lookup cols exceed i64".into()))?;
    unsafe {
        status::check(abi::apxinf_elementwise_gather_rows_bf16(
            table_buffer.ptr(),
            ids.ptr(),
            output_buffer.ptr(),
            rows,
            cols_i64,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}

/// `add_position_f32_bf16`: F32 projection + learned position embedding
/// (+ optional bias), rounded once to BF16 — the vision patch-embedding
/// epilogue.
pub fn add_position_f32_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let dims = projection.shape().dims();
    if dims.len() != 2 || projection.dtype() != DType::F32 || position.dtype() != DType::F32 {
        return Err(Error::Other(
            "position embedding expects rank-2 F32 projection and position".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if tokens_per_view == 0 || rows % tokens_per_view != 0 {
        return Err(Error::Other(
            "position embedding rows must divide by tokens_per_view".into(),
        ));
    }
    let output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let projection_buffer = CudaBuffer::from_tensor(projection).map_err(Error::Cuda)?;
    let position_buffer = CudaBuffer::from_tensor(position).map_err(Error::Cuda)?;
    let bias_buffer = bias
        .map(CudaBuffer::from_tensor)
        .transpose()
        .map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    let count = i64::try_from(rows * cols)
        .map_err(|_| Error::Other("position embedding size exceeds i64".into()))?;
    let cols_i32 =
        i32::try_from(cols).map_err(|_| Error::Other("position embedding cols exceed i32".into()))?;
    let tokens_i32 = i32::try_from(tokens_per_view)
        .map_err(|_| Error::Other("tokens_per_view exceeds i32".into()))?;
    unsafe {
        status::check(abi::apxinf_elementwise_bias_position_f32_bf16(
            projection_buffer.ptr(),
            bias_buffer
                .as_ref()
                .map_or(std::ptr::null(), |buffer| buffer.ptr() as *const _),
            position_buffer.ptr(),
            output_buffer.ptr(),
            count,
            cols_i32,
            tokens_i32,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}

/// `sinusoidal_bf16`: sin/cos positional features from F32 positions.
pub fn sinusoidal_bf16(
    ctx: &CudaContext,
    positions: &Tensor,
    dim: usize,
    scale: f32,
    frequency_step: f32,
) -> Result<Tensor> {
    use crate::ffi::abi::hybrid as abi;
    use crate::ffi::raw::cuda_runtime as raw;
    let shape = positions.shape().dims();
    if shape.len() != 1
        || shape[0] == 0
        || dim == 0
        || dim % 2 != 0
        || !scale.is_finite()
        || !frequency_step.is_finite()
        || positions.dtype() != DType::F32
        || positions.device() != apxinf_core::Device::Cuda(ctx.device_id())
    {
        return Err(Error::Other(
            "invalid sinusoidal embedding shape, dtype, device or scale".into(),
        ));
    }
    let rows =
        i32::try_from(shape[0]).map_err(|_| Error::Other("embedding row count overflow".into()))?;
    let width = i32::try_from(dim).map_err(|_| Error::Other("embedding width overflow".into()))?;
    let output = ctx.allocate_output(Shape::new(vec![shape[0], dim]), DType::BF16)?;
    let positions_buffer = CudaBuffer::from_tensor(positions).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    unsafe {
        raw::check_cuda(abi::apxinf_hybrid_sinusoidal_embedding_bf16(
            positions_buffer.ptr(),
            output_buffer.ptr(),
            rows,
            width,
            scale,
            frequency_step,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(output)
}

// ── smolvla surface ─────────────────────────────────────────────────────────
//
// `lookup_bf16`/`lookup_f16` are the legacy sqrt(width)-scaled embedding
// lookups; cuda-new's Gather operator carries the same always-scaled kernel
// for both dtypes, so these route through it. `add_position_*` are the
// same-dtype bias-position gathers.

fn lookup_scaled(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &CudaBuffer,
    tokens: usize,
    dtype: DType,
) -> Result<Tensor> {
    use crate::ops;
    let dims = table.shape().dims();
    if table.dtype() != dtype || dims.len() != 2 || ids.len() < tokens * 4 || tokens == 0 {
        return Err(Error::Other(format!(
            "static inference {dtype} embedding expects [vocab,width] and device u32 ids"
        )));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![tokens, dims[1]]), dtype)?;
    let mut args = ops::GatherArgs::new(ops::GatherSemantic::EmbeddingLookup, table, &mut output);
    args.ids = Some(ids);
    args.vocab_size = dims[0];
    ops::gather(ctx, args)?;
    Ok(output)
}

/// `lookup_bf16`: sqrt(width)-scaled BF16 embedding lookup over device ids.
pub fn lookup_bf16(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &CudaBuffer,
    tokens: usize,
) -> Result<Tensor> {
    lookup_scaled(ctx, table, ids, tokens, DType::BF16)
}

/// `lookup_f16`: sqrt(width)-scaled FP16 embedding lookup over device ids.
pub fn lookup_f16(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &CudaBuffer,
    tokens: usize,
) -> Result<Tensor> {
    lookup_scaled(ctx, table, ids, tokens, DType::F16)
}

fn add_position_same_dtype(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
    dtype: DType,
) -> Result<Tensor> {
    use crate::ops;
    let dims = projection.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(
            "position embedding expects a rank-2 projection".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    let position_dims = position.shape().dims();
    let position_ok =
        position_dims == [tokens_per_view, cols] || position_dims == [1, tokens_per_view, cols];
    if projection.dtype() != dtype
        || position.dtype() != dtype
        || !position_ok
        || tokens_per_view == 0
        || rows % tokens_per_view != 0
        || bias.is_some_and(|value| value.dtype() != dtype || value.shape().dims() != [cols])
    {
        return Err(Error::Other(format!(
            "static inference {dtype} position embedding shape mismatch"
        )));
    }
    // The Gather contract wants a rank-2 position tensor; the legacy helper
    // also accepted a leading unit dim, so flatten it here.
    let position = if position_dims.len() == 3 {
        position.reshape(vec![tokens_per_view, cols])?
    } else {
        position.clone()
    };
    let mut output = ctx.allocate_output(Shape::new(vec![rows, cols]), dtype)?;
    let mut args = ops::GatherArgs::new(ops::GatherSemantic::BiasPosition, projection, &mut output);
    args.bias = bias;
    args.position = Some(&position);
    args.tokens_per_view = tokens_per_view;
    ops::gather(ctx, args)?;
    Ok(output)
}

/// `add_position_bf16`: BF16 projection + optional bias + learned per-view
/// position embedding.
pub fn add_position_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    add_position_same_dtype(ctx, projection, bias, position, tokens_per_view, DType::BF16)
}

/// `add_position_f16`: FP16 projection + optional bias + learned per-view
/// position embedding.
pub fn add_position_f16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    add_position_same_dtype(ctx, projection, bias, position, tokens_per_view, DType::F16)
}

// ── Decode fast-path `_into` surface (caller-owned buffers, graph-safe) ───

/// Allocation-free embedding lookup into caller-owned decode storage; token
/// ids are read from a device address (host-mapped on Tegra) at kernel time.
pub fn lookup_into(
    ctx: &CudaContext,
    dtype: DType,
    table: &CudaBuffer,
    ids: crate::CudaDeviceAddress,
    output: &CudaBuffer,
    embed_dim: usize,
    seq_len: usize,
) -> Result<()> {
    use super::validation::{check_cuda, checked_bytes, require_address, require_buffers};
    use crate::ffi::abi::llm_decode as llmdec;
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
            DType::F32 => llmdec::apxinf_llmdec_embedding_f32(
                table.ptr(),
                ids.ptr(),
                output.ptr(),
                embed_dim as u32,
                seq_len as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => llmdec::apxinf_llmdec_embedding_bf16(
                table.ptr(),
                ids.ptr(),
                output.ptr(),
                embed_dim as u32,
                seq_len as u32,
                ctx.stream().handle(),
            ),
            dtype => {
                return Err(Error::Other(format!(
                    "decode embedding does not support {dtype}"
                )))
            }
        }
    };
    check_cuda(status)
}

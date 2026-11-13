//! Embedding lookup and position operators.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::ffi::abi::{elementwise as abi, status};
use crate::{CudaBuffer, CudaContext};

/// F32 projection + learned position embedding
/// (+ optional bias), rounded once to BF16 — the vision patch-embedding
/// epilogue.
pub fn add_position_projected(
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
    let cols_i32 = i32::try_from(cols)
        .map_err(|_| Error::Other("position embedding cols exceed i32".into()))?;
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

/// Sin/cos positional features from F32 positions, written as BF16.
pub fn sinusoidal(
    ctx: &CudaContext,
    positions: &Tensor,
    dim: usize,
    scale: f32,
    frequency_step: f32,
) -> Result<Tensor> {
    use crate::ffi::abi::embedding as abi;
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
        raw::check_cuda(abi::apxinf_embedding_sinusoidal_embedding_bf16(
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
    use crate::ffi::abi::embedding as llmdec;
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
            DType::F32 => llmdec::apxinf_embedding_fixed_decode_lookup_f32(
                table.ptr(),
                ids.ptr(),
                output.ptr(),
                embed_dim as u32,
                seq_len as u32,
                ctx.stream().handle(),
            ),
            DType::BF16 => llmdec::apxinf_embedding_fixed_decode_lookup_bf16(
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

/// Reference sinusoidal table from scalar `times`, matching the speech
/// reference's `sin`-then-`cos` column layout. Unlike [`sinusoidal`], which
/// takes F32 positions, this keeps `times` in the storage dtype and rounds
/// `time * scale` to it before the frequency sweep — bit-identical to the
/// legacy kernel this replaces.
///
/// `max_period` derives the log frequency step exactly as the reference
/// does: `ln(max_period) / (width / 2 - 1)`.
pub fn sinusoidal_embedding_table(
    ctx: &CudaContext,
    times: &Tensor,
    output: &Tensor,
    scale: f32,
    max_period: f64,
) -> Result<()> {
    use crate::ops::validation::check_cuda;

    fn invalid(message: impl Into<String>) -> Error {
        Error::Other(message.into())
    }
    fn dtype_code(dtype: DType) -> Result<u32> {
        match dtype {
            DType::F32 => Ok(0),
            DType::F16 => Ok(1),
            DType::BF16 => Ok(2),
            _ => Err(invalid("sinusoidal table supports F32, F16, and BF16")),
        }
    }
    let dims = output.shape().dims().to_vec();
    if dims.len() != 2
        || dims[0] == 0
        || dims[1] < 4
        || dims[1] % 2 != 0
        || times.shape().dims() != [dims[0]]
        || !scale.is_finite()
        || !max_period.is_finite()
        || max_period <= 1.0
    {
        return Err(invalid(
            "invalid sinusoidal embedding geometry or frequency",
        ));
    }
    let dtype = times.dtype();
    if output.dtype() != dtype {
        return Err(invalid("sinusoidal table dtype mismatch"));
    }
    let code = dtype_code(dtype)?;
    let log_step = (max_period.ln() / (dims[1] / 2 - 1) as f64) as f32;
    let times_buffer = CudaBuffer::from_tensor(times).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    if times_buffer.len() < dims[0] * dtype.size_in_bytes()
        || output_buffer.len() < dims[0] * dims[1] * dtype.size_in_bytes()
    {
        return Err(invalid("sinusoidal table storage is too small"));
    }
    unsafe {
        check_cuda(crate::ffi::abi::embedding::apxinf_embedding_sinusoidal_table(
            times_buffer.ptr(),
            output_buffer.ptr(),
            dims[0] as i64,
            dims[1] as i64,
            scale,
            log_step,
            code,
            ctx.stream().handle(),
        ))?;
    }
    Ok(())
}

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use super::contracts::check_cuda;
use crate::workspace::uninitialized_buffer;
use crate::{ffi, CudaBuffer, CudaContext};

pub const HEADS: usize = 16;
pub const HEAD_DIM: usize = 80;
pub const HIDDEN: usize = HEADS * HEAD_DIM;
pub const INTERMEDIATE: usize = 3_420;

pub fn gate_up_bias_silu_mul_exact(
    ctx: &CudaContext,
    gate: &Tensor,
    gate_bias: &Tensor,
    up: &Tensor,
    up_bias: &Tensor,
) -> Result<Tensor> {
    let sequence = gate.shape().dims().first().copied().unwrap_or(0);
    let device = Device::Cuda(ctx.device_id());
    let matrix_shape = [sequence, INTERMEDIATE];
    if ctx.caps().sm != 89
        || sequence == 0
        || sequence > 65_535
        || gate.dtype() != DType::BF16
        || gate.device() != device
        || gate.shape().dims() != matrix_shape
        || up.dtype() != DType::BF16
        || up.device() != device
        || up.shape().dims() != matrix_shape
        || [gate_bias, up_bias].iter().any(|bias| {
            bias.dtype() != DType::BF16
                || bias.device() != device
                || bias.shape().dims() != [INTERMEDIATE]
        })
    {
        return Err(Error::Other(
            "Qwen2.5-Omni vision Gate/Up bias SiLU/multiply contract mismatch".into(),
        ));
    }
    let gate = CudaBuffer::from_tensor(gate).map_err(Error::Cuda)?;
    let gate_bias = CudaBuffer::from_tensor(gate_bias).map_err(Error::Cuda)?;
    let up = CudaBuffer::from_tensor(up).map_err(Error::Cuda)?;
    let up_bias = CudaBuffer::from_tensor(up_bias).map_err(Error::Cuda)?;
    let bytes = sequence
        .checked_mul(INTERMEDIATE)
        .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("vision Gate/Up output size overflow".into()))?;
    let output = uninitialized_buffer(ctx, bytes)?;
    check_cuda(unsafe {
        ffi::apxinf_static_qwen25_omni_vision_gate_up_bias_silu_mul_exact_bf16(
            gate.ptr(),
            gate_bias.ptr(),
            up.ptr(),
            up_bias.ptr(),
            output.ptr(),
            sequence as i32,
            INTERMEDIATE as i32,
            ctx.stream().handle(),
        )
    })?;
    Ok(output.into_tensor(Shape::new(vec![sequence, INTERMEDIATE]), DType::BF16))
}

pub fn packed_gate_up_bias_silu_mul_exact(
    ctx: &CudaContext,
    packed_gate_up: &Tensor,
    gate_bias: &Tensor,
    up_bias: &Tensor,
) -> Result<Tensor> {
    let sequence = packed_gate_up
        .shape()
        .dims()
        .first()
        .copied()
        .unwrap_or(0);
    let device = Device::Cuda(ctx.device_id());
    if ctx.caps().sm != 89
        || sequence == 0
        || sequence > 65_535
        || packed_gate_up.dtype() != DType::BF16
        || packed_gate_up.device() != device
        || packed_gate_up.shape().dims() != [sequence, 2 * INTERMEDIATE]
        || [gate_bias, up_bias].iter().any(|bias| {
            bias.dtype() != DType::BF16
                || bias.device() != device
                || bias.shape().dims() != [INTERMEDIATE]
        })
    {
        return Err(Error::Other(
            "Qwen2.5-Omni vision packed Gate/Up bias SiLU/multiply contract mismatch".into(),
        ));
    }
    let packed_gate_up = CudaBuffer::from_tensor(packed_gate_up).map_err(Error::Cuda)?;
    let gate_bias = CudaBuffer::from_tensor(gate_bias).map_err(Error::Cuda)?;
    let up_bias = CudaBuffer::from_tensor(up_bias).map_err(Error::Cuda)?;
    let bytes = sequence
        .checked_mul(INTERMEDIATE)
        .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("vision packed Gate/Up output size overflow".into()))?;
    let output = uninitialized_buffer(ctx, bytes)?;
    check_cuda(unsafe {
        ffi::apxinf_static_qwen25_omni_vision_packed_gate_up_bias_silu_mul_exact_bf16(
            packed_gate_up.ptr(),
            gate_bias.ptr(),
            up_bias.ptr(),
            output.ptr(),
            sequence as i32,
            INTERMEDIATE as i32,
            ctx.stream().handle(),
        )
    })?;
    Ok(output.into_tensor(Shape::new(vec![sequence, INTERMEDIATE]), DType::BF16))
}

pub fn bias_residual_exact(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: &Tensor,
    residual: &Tensor,
) -> Result<Tensor> {
    let sequence = projection.shape().dims().first().copied().unwrap_or(0);
    let device = Device::Cuda(ctx.device_id());
    let matrix_shape = [sequence, HIDDEN];
    if ctx.caps().sm != 89
        || sequence == 0
        || sequence > 65_535
        || projection.dtype() != DType::BF16
        || projection.device() != device
        || projection.shape().dims() != matrix_shape
        || residual.dtype() != DType::BF16
        || residual.device() != device
        || residual.shape().dims() != matrix_shape
        || bias.dtype() != DType::BF16
        || bias.device() != device
        || bias.shape().dims() != [HIDDEN]
    {
        return Err(Error::Other(
            "Qwen2.5-Omni vision exact bias/residual contract mismatch".into(),
        ));
    }
    let projection = CudaBuffer::from_tensor(projection).map_err(Error::Cuda)?;
    let bias = CudaBuffer::from_tensor(bias).map_err(Error::Cuda)?;
    let residual = CudaBuffer::from_tensor(residual).map_err(Error::Cuda)?;
    let bytes = sequence
        .checked_mul(HIDDEN)
        .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("vision exact bias/residual output size overflow".into()))?;
    let output = uninitialized_buffer(ctx, bytes)?;
    check_cuda(unsafe {
        ffi::apxinf_static_qwen25_omni_vision_bias_residual_exact_bf16(
            projection.ptr(),
            bias.ptr(),
            residual.ptr(),
            output.ptr(),
            sequence as i32,
            HIDDEN as i32,
            ctx.stream().handle(),
        )
    })?;
    Ok(output.into_tensor(Shape::new(vec![sequence, HIDDEN]), DType::BF16))
}

#[allow(clippy::too_many_arguments)]
pub fn qkv_bias_rope(
    ctx: &CudaContext,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    query_bias: &Tensor,
    key_bias: &Tensor,
    value_bias: &Tensor,
    theta: f32,
    positions: &CudaBuffer,
) -> Result<(Tensor, Tensor, Tensor)> {
    qkv_bias_rope_impl(
        ctx,
        query,
        key,
        value,
        query_bias,
        key_bias,
        value_bias,
        theta,
        positions,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn grouped_qkv_bias_rope(
    ctx: &CudaContext,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    query_bias: &Tensor,
    key_bias: &Tensor,
    value_bias: &Tensor,
    theta: f32,
    positions: &CudaBuffer,
    group_indices: &CudaBuffer,
) -> Result<(Tensor, Tensor, Tensor)> {
    qkv_bias_rope_impl(
        ctx,
        query,
        key,
        value,
        query_bias,
        key_bias,
        value_bias,
        theta,
        positions,
        Some(group_indices),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn packed_qkv_bias_rope(
    ctx: &CudaContext,
    packed_qkv: &Tensor,
    query_bias: &Tensor,
    key_bias: &Tensor,
    value_bias: &Tensor,
    theta: f32,
    positions: &CudaBuffer,
    group_indices: Option<&CudaBuffer>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let sequence = packed_qkv
        .shape()
        .dims()
        .first()
        .copied()
        .unwrap_or(0);
    let device = Device::Cuda(ctx.device_id());
    if ctx.caps().sm != 89
        || sequence == 0
        || sequence > 65_535
        || packed_qkv.dtype() != DType::BF16
        || packed_qkv.device() != device
        || packed_qkv.shape().dims() != [sequence, 3 * HIDDEN]
        || [query_bias, key_bias, value_bias].iter().any(|bias| {
            bias.dtype() != DType::BF16
                || bias.device() != device
                || bias.shape().dims() != [HIDDEN]
        })
        || theta.to_bits() != 10_000.0f32.to_bits()
        || positions.device() != ctx.device_id()
        || positions.len() != sequence * 2 * std::mem::size_of::<u32>()
        || group_indices.is_some_and(|indices| {
            indices.device() != ctx.device_id()
                || indices.len() != sequence * std::mem::size_of::<u32>()
        })
    {
        return Err(Error::Other(
            "Qwen2.5-Omni packed vision QKV bias/RoPE contract mismatch".into(),
        ));
    }

    let packed_qkv = CudaBuffer::from_tensor(packed_qkv).map_err(Error::Cuda)?;
    let query_bias = CudaBuffer::from_tensor(query_bias).map_err(Error::Cuda)?;
    let key_bias = CudaBuffer::from_tensor(key_bias).map_err(Error::Cuda)?;
    let value_bias = CudaBuffer::from_tensor(value_bias).map_err(Error::Cuda)?;
    let bytes = sequence
        .checked_mul(HIDDEN)
        .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("packed vision QKV output size overflow".into()))?;
    let query_output = uninitialized_buffer(ctx, bytes)?;
    let key_output = uninitialized_buffer(ctx, bytes)?;
    let value_output = uninitialized_buffer(ctx, bytes)?;
    check_cuda(unsafe {
        if let Some(indices) = group_indices {
            ffi::apxinf_static_qwen25_omni_vision_packed_grouped_qkv_bias_rope_bf16(
                packed_qkv.ptr(),
                query_bias.ptr(),
                key_bias.ptr(),
                value_bias.ptr(),
                query_output.ptr(),
                key_output.ptr(),
                value_output.ptr(),
                sequence as i32,
                HEADS as i32,
                HEAD_DIM as i32,
                theta,
                positions.ptr(),
                indices.ptr(),
                ctx.stream().handle(),
            )
        } else {
            ffi::apxinf_static_qwen25_omni_vision_packed_qkv_bias_rope_bf16(
                packed_qkv.ptr(),
                query_bias.ptr(),
                key_bias.ptr(),
                value_bias.ptr(),
                query_output.ptr(),
                key_output.ptr(),
                value_output.ptr(),
                sequence as i32,
                HEADS as i32,
                HEAD_DIM as i32,
                theta,
                positions.ptr(),
                ctx.stream().handle(),
            )
        }
    })?;
    let shape = Shape::new(vec![sequence, HEADS, HEAD_DIM]);
    Ok((
        query_output.into_tensor(shape.clone(), DType::BF16),
        key_output.into_tensor(shape.clone(), DType::BF16),
        value_output.into_tensor(shape, DType::BF16),
    ))
}

#[allow(clippy::too_many_arguments)]
fn qkv_bias_rope_impl(
    ctx: &CudaContext,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    query_bias: &Tensor,
    key_bias: &Tensor,
    value_bias: &Tensor,
    theta: f32,
    positions: &CudaBuffer,
    group_indices: Option<&CudaBuffer>,
) -> Result<(Tensor, Tensor, Tensor)> {
    let sequence = query.shape().dims().first().copied().unwrap_or(0);
    let device = Device::Cuda(ctx.device_id());
    let matrix_shape = [sequence, HIDDEN];
    if ctx.caps().sm != 89
        || sequence == 0
        || sequence > 65_535
        || query.dtype() != DType::BF16
        || query.device() != device
        || query.shape().dims() != matrix_shape
        || key.dtype() != DType::BF16
        || key.device() != device
        || key.shape().dims() != matrix_shape
        || value.dtype() != DType::BF16
        || value.device() != device
        || value.shape().dims() != matrix_shape
        || [query_bias, key_bias, value_bias].iter().any(|bias| {
            bias.dtype() != DType::BF16
                || bias.device() != device
                || bias.shape().dims() != [HIDDEN]
        })
        || theta.to_bits() != 10_000.0f32.to_bits()
        || positions.device() != ctx.device_id()
        || positions.len() != sequence * 2 * std::mem::size_of::<u32>()
        || group_indices.is_some_and(|indices| {
            indices.device() != ctx.device_id()
                || indices.len() != sequence * std::mem::size_of::<u32>()
        })
    {
        return Err(Error::Other(
            "Qwen2.5-Omni vision QKV bias/RoPE contract mismatch".into(),
        ));
    }

    let query = CudaBuffer::from_tensor(query).map_err(Error::Cuda)?;
    let key = CudaBuffer::from_tensor(key).map_err(Error::Cuda)?;
    let value = CudaBuffer::from_tensor(value).map_err(Error::Cuda)?;
    let query_bias = CudaBuffer::from_tensor(query_bias).map_err(Error::Cuda)?;
    let key_bias = CudaBuffer::from_tensor(key_bias).map_err(Error::Cuda)?;
    let value_bias = CudaBuffer::from_tensor(value_bias).map_err(Error::Cuda)?;
    let bytes = sequence
        .checked_mul(HIDDEN)
        .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| Error::Other("vision QKV output size overflow".into()))?;
    let query_output = uninitialized_buffer(ctx, bytes)?;
    let key_output = uninitialized_buffer(ctx, bytes)?;
    let value_output = uninitialized_buffer(ctx, bytes)?;
    check_cuda(unsafe {
        if let Some(indices) = group_indices {
            ffi::apxinf_static_qwen25_omni_vision_grouped_qkv_bias_rope_bf16(
                query.ptr(),
                key.ptr(),
                value.ptr(),
                query_bias.ptr(),
                key_bias.ptr(),
                value_bias.ptr(),
                query_output.ptr(),
                key_output.ptr(),
                value_output.ptr(),
                sequence as i32,
                HEADS as i32,
                HEAD_DIM as i32,
                theta,
                positions.ptr(),
                indices.ptr(),
                ctx.stream().handle(),
            )
        } else {
            ffi::apxinf_static_qwen25_omni_vision_qkv_bias_rope_bf16(
                query.ptr(),
                key.ptr(),
                value.ptr(),
                query_bias.ptr(),
                key_bias.ptr(),
                value_bias.ptr(),
                query_output.ptr(),
                key_output.ptr(),
                value_output.ptr(),
                sequence as i32,
                HEADS as i32,
                HEAD_DIM as i32,
                theta,
                positions.ptr(),
                ctx.stream().handle(),
            )
        }
    })?;
    let shape = Shape::new(vec![sequence, HEADS, HEAD_DIM]);
    Ok((
        query_output.into_tensor(shape.clone(), DType::BF16),
        key_output.into_tensor(shape.clone(), DType::BF16),
        value_output.into_tensor(shape, DType::BF16),
    ))
}

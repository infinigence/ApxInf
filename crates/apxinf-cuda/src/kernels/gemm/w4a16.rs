use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W4A16Layout {
    CompressedTensorsPackQuantized,
}

#[derive(Clone, Copy)]
pub struct W4A16WeightView<'a> {
    pub packed_i32: &'a Tensor,
    pub scales_bf16: &'a Tensor,
    pub zero_points_i32: &'a Tensor,
    pub input_dim: usize,
    pub output_dim: usize,
    pub group_size: usize,
    pub layout: W4A16Layout,
}

/// M=1 BF16 activation by compressed-tensors W4 group-32 asymmetric weight.
/// Output storage is caller-owned so repeated decode can reuse a stable address.
pub fn gemv_w4a16_write(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W4A16WeightView<'_>,
    output: &Tensor,
) -> Result<()> {
    gemv_w4a16_write_impl(ctx, activation, weight, output, true)
}

/// Explicit pre-staging baseline retained for operator A/B and rollback.
pub fn gemv_w4a16_write_direct(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W4A16WeightView<'_>,
    output: &Tensor,
) -> Result<()> {
    gemv_w4a16_write_impl(ctx, activation, weight, output, false)
}

/// BF16 MxK by compressed-tensors W4 NxK for a prefill tile of 1..=8 tokens.
/// The kernel reuses each streamed weight value across every token in the tile.
pub fn gemm_w4a16_m8_write(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W4A16WeightView<'_>,
    output: &Tensor,
) -> Result<()> {
    let expected_device = Device::Cuda(ctx.device_id());
    let activation_shape = activation.shape().dims();
    if activation.dtype() != DType::BF16
        || activation.device() != expected_device
        || activation_shape.len() != 2
        || activation_shape[0] == 0
        || activation_shape[0] > 8
        || activation_shape[1] != weight.input_dim
    {
        return Err(Error::Other(format!(
            "W4A16 M8 activation must be BF16 [1..=8,{}] on {}, got {} {:?} on {}",
            weight.input_dim,
            expected_device,
            activation.dtype(),
            activation_shape,
            activation.device()
        )));
    }
    let tokens = activation_shape[0];
    if output.dtype() != DType::BF16
        || output.device() != expected_device
        || output.shape().dims() != [tokens, weight.output_dim]
    {
        return Err(Error::Other(format!(
            "W4A16 M8 output must be BF16 [{tokens},{}] on {}",
            weight.output_dim, expected_device
        )));
    }
    validate_weight(ctx, weight)?;
    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let packed = CudaBuffer::from_tensor(weight.packed_i32).map_err(Error::Cuda)?;
    let scales = CudaBuffer::from_tensor(weight.scales_bf16).map_err(Error::Cuda)?;
    let zero_points = CudaBuffer::from_tensor(weight.zero_points_i32).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    let input_dim = i32::try_from(weight.input_dim)
        .map_err(|_| Error::Other("W4A16 M8 input dimension exceeds i32".into()))?;
    let output_dim = i32::try_from(weight.output_dim)
        .map_err(|_| Error::Other("W4A16 M8 output dimension exceeds i32".into()))?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_w4a16_gemm_m8_bf16(
            activation.ptr(),
            packed.ptr(),
            scales.ptr(),
            zero_points.ptr(),
            output.ptr(),
            tokens as i32,
            input_dim,
            output_dim,
            weight.group_size as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

fn gemv_w4a16_write_impl(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W4A16WeightView<'_>,
    output: &Tensor,
    staged: bool,
) -> Result<()> {
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.dtype() != DType::BF16
        || activation.device() != expected_device
        || activation.shape().dims() != [1, weight.input_dim]
    {
        return Err(Error::Other(format!(
            "W4A16 GEMV activation must be BF16 [1,{}] on {}, got {} {:?} on {}",
            weight.input_dim,
            expected_device,
            activation.dtype(),
            activation.shape().dims(),
            activation.device()
        )));
    }
    if output.dtype() != DType::BF16
        || output.device() != expected_device
        || output.shape().dims() != [1, weight.output_dim]
    {
        return Err(Error::Other(format!(
            "W4A16 GEMV output must be BF16 [1,{}] on {}",
            weight.output_dim, expected_device
        )));
    }
    validate_weight(ctx, weight)?;

    let activation = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let packed = CudaBuffer::from_tensor(weight.packed_i32).map_err(Error::Cuda)?;
    let scales = CudaBuffer::from_tensor(weight.scales_bf16).map_err(Error::Cuda)?;
    let zero_points = CudaBuffer::from_tensor(weight.zero_points_i32).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    let input_dim = i32::try_from(weight.input_dim)
        .map_err(|_| Error::Other("W4A16 input dimension exceeds i32".into()))?;
    let output_dim = i32::try_from(weight.output_dim)
        .map_err(|_| Error::Other("W4A16 output dimension exceeds i32".into()))?;
    let group_size = i32::try_from(weight.group_size)
        .map_err(|_| Error::Other("W4A16 group size exceeds i32".into()))?;
    let error = unsafe {
        if staged {
            ffi::apxinf_static_w4a16_gemv_bf16_staged(
                activation.ptr(),
                packed.ptr(),
                scales.ptr(),
                zero_points.ptr(),
                output.ptr(),
                input_dim,
                output_dim,
                group_size,
                ctx.stream().handle(),
            )
        } else {
            ffi::apxinf_static_w4a16_gemv_bf16(
                activation.ptr(),
                packed.ptr(),
                scales.ptr(),
                zero_points.ptr(),
                output.ptr(),
                input_dim,
                output_dim,
                group_size,
                ctx.stream().handle(),
            )
        }
    };
    ffi::check_cuda(error).map_err(Error::Cuda)
}

pub fn gemv_w4a16(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: W4A16WeightView<'_>,
) -> Result<Tensor> {
    let output = crate::transfers::to_cuda(
        &Tensor::zeros(vec![1, weight.output_dim], DType::BF16),
        ctx.device_id(),
    )?;
    gemv_w4a16_write(ctx, activation, weight, &output)?;
    Ok(output)
}

fn validate_weight(ctx: &CudaContext, weight: W4A16WeightView<'_>) -> Result<()> {
    if weight.layout != W4A16Layout::CompressedTensorsPackQuantized
        || weight.group_size != 32
        || weight.input_dim == 0
        || weight.output_dim == 0
        || weight.input_dim % 32 != 0
        || weight.output_dim % 8 != 0
    {
        return Err(Error::Other(format!(
            "unsupported W4A16 contract: layout={:?}, input={}, output={}, group={}",
            weight.layout, weight.input_dim, weight.output_dim, weight.group_size
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    let expected = [
        (
            "packed",
            weight.packed_i32,
            DType::I32,
            vec![weight.output_dim, weight.input_dim / 8],
        ),
        (
            "scales",
            weight.scales_bf16,
            DType::BF16,
            vec![weight.output_dim, weight.input_dim / weight.group_size],
        ),
        (
            "zero points",
            weight.zero_points_i32,
            DType::I32,
            vec![weight.output_dim / 8, weight.input_dim / weight.group_size],
        ),
    ];
    for (name, tensor, dtype, shape) in expected {
        if tensor.dtype() != dtype
            || tensor.device() != expected_device
            || tensor.shape().dims() != shape
        {
            return Err(Error::Other(format!(
                "W4A16 {name} must be {dtype} {shape:?} on {expected_device}, got {} {:?} on {}",
                tensor.dtype(),
                tensor.shape().dims(),
                tensor.device()
            )));
        }
    }
    Ok(())
}

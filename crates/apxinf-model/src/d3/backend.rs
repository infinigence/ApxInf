//! cuda-new-only allocation, views and transfers for d3.

use apxinf_core::{DType, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{CudaBuffer, CudaContext};

pub(crate) fn upload(ctx: &CudaContext, tensor: &Tensor) -> Result<Tensor> {
    let apxinf_core::Storage::Cpu(bytes) = tensor.storage() else {
        return Err(Error::Other("d3 upload expects a CPU tensor".into()));
    };
    let buffer = CudaBuffer::alloc(bytes.len().max(1), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(bytes).map_err(Error::Cuda)?;
    buffer
        .as_tensor(Shape::new(tensor.shape().dims().to_vec()), tensor.dtype())
        .map_err(Error::Cuda)
}

pub(crate) fn zeros(
    ctx: &CudaContext,
    shape: impl Into<Vec<usize>>,
    dtype: DType,
) -> Result<Tensor> {
    let shape = shape.into();
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    let buffer = CudaBuffer::alloc(bytes.max(1), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.zero().map_err(Error::Cuda)?;
    buffer
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

pub(crate) fn prefix(
    tensor: &Tensor,
    shape: impl Into<Vec<usize>>,
    dtype: DType,
) -> Result<Tensor> {
    let shape = shape.into();
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(0, bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

pub(crate) fn view_at(
    tensor: &Tensor,
    element_offset: usize,
    shape: impl Into<Vec<usize>>,
    dtype: DType,
) -> Result<Tensor> {
    let shape = shape.into();
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(element_offset * dtype.size_in_bytes(), bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

pub(crate) fn copy_i32(ctx: &CudaContext, tensor: &Tensor, values: &[i32]) -> Result<()> {
    let _ = ctx;
    let bytes = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect::<Vec<_>>();
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .copy_from_host(&bytes)
        .map_err(Error::Cuda)
}

pub(crate) fn read_f32(ctx: &CudaContext, tensor: &Tensor) -> Result<Vec<f32>> {
    ctx.synchronize().map_err(Error::Cuda)?;
    if tensor.dtype() != DType::F32 {
        return Err(Error::Other("d3 readback expects FP32".into()));
    }
    let mut bytes = vec![0u8; tensor.numel() * 4];
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .copy_to_host(&mut bytes)
        .map_err(Error::Cuda)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

pub(crate) fn read_bf16(ctx: &CudaContext, tensor: &Tensor) -> Result<Vec<f32>> {
    ctx.synchronize().map_err(Error::Cuda)?;
    if tensor.dtype() != DType::BF16 {
        return Err(Error::Other("d3 readback expects BF16".into()));
    }
    let mut bytes = vec![0u8; tensor.numel() * 2];
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .copy_to_host(&mut bytes)
        .map_err(Error::Cuda)?;
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| half::bf16::from_bits(u16::from_le_bytes(chunk.try_into().unwrap())).to_f32())
        .collect())
}

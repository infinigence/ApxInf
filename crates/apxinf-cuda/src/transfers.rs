//! Safe CUDA tensor transfer operations and private copy primitives.

use crate::ffi;
use crate::{CudaBuffer, CudaContext};
use apxinf_core::{Device, Error, Result, Storage, Tensor};

/// Transfer a CPU tensor to a CUDA device.
pub fn to_cuda(tensor: &Tensor, device_id: usize) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other("tensor is already on GPU".into()));
    }
    let bytes = tensor
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("expected CPU storage".into()))?;
    let buffer = CudaBuffer::alloc(bytes.len(), device_id).map_err(Error::Cuda)?;
    buffer.copy_from_host(bytes).map_err(Error::Cuda)?;
    Ok(buffer.into_tensor(tensor.shape().clone(), tensor.dtype()))
}

/// Copy a CPU tensor into shape-identical, stable-address CUDA storage.
pub fn copy_cpu_to_cuda(source: &Tensor, destination: &Tensor) -> Result<()> {
    if source.device() != Device::Cpu {
        return Err(Error::Other("copy source must be a CPU tensor".into()));
    }
    let device_id = match destination.device() {
        Device::Cuda(device_id) => device_id,
        device => return Err(Error::UnsupportedDevice(device)),
    };
    if source.shape() != destination.shape() || source.dtype() != destination.dtype() {
        return Err(Error::Other(format!(
            "fixed CUDA input mismatch: source {:?} {}, destination {:?} {}",
            source.shape().dims(),
            source.dtype(),
            destination.shape().dims(),
            destination.dtype()
        )));
    }
    let source = source
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("expected CPU storage".into()))?;
    let destination = destination
        .storage()
        .as_gpu()
        .ok_or_else(|| Error::Other("expected CUDA storage".into()))?;
    copy_host_to_device(device_id, source, destination.ptr).map_err(Error::Cuda)
}

/// Transfer a CUDA tensor back to CPU.
pub fn to_cpu(tensor: &Tensor) -> Result<Tensor> {
    let handle = match tensor.storage() {
        Storage::Gpu { handle, .. } => handle,
        _ => return Err(Error::Other("tensor is not on GPU".into())),
    };
    let device_id = match tensor.device() {
        Device::Cuda(device_id) => device_id,
        device => return Err(Error::UnsupportedDevice(device)),
    };
    let bytes = copy_device_to_host(device_id, handle.ptr, handle.len).map_err(Error::Cuda)?;
    Tensor::from_raw(tensor.shape().clone(), tensor.dtype(), Device::Cpu, bytes)
}

pub(crate) fn copy_host_to_device(
    device_id: usize,
    source: &[u8],
    destination: usize,
) -> std::result::Result<(), String> {
    unsafe {
        ffi::check_cuda(ffi::cudaSetDevice(device_id as i32))?;
        ffi::check_cuda(ffi::cudaMemcpy(
            destination as *mut std::ffi::c_void,
            source.as_ptr().cast(),
            source.len(),
            ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
        ))
    }
}

pub(crate) fn copy_device_to_host(
    device_id: usize,
    source: usize,
    len: usize,
) -> std::result::Result<Vec<u8>, String> {
    let mut destination = vec![0u8; len];
    unsafe {
        ffi::check_cuda(ffi::cudaSetDevice(device_id as i32))?;
        ffi::check_cuda(ffi::cudaDeviceSynchronize())?;
        ffi::check_cuda(ffi::cudaMemcpy(
            destination.as_mut_ptr().cast(),
            source as *const std::ffi::c_void,
            len,
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        ))?;
    }
    Ok(destination)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_tensor_2d_to_buffer(
    ctx: &CudaContext,
    source: &Tensor,
    destination: &CudaBuffer,
    destination_offset: usize,
    destination_pitch: usize,
    source_pitch: usize,
    width: usize,
    rows: usize,
) -> Result<()> {
    if rows == 0 || width == 0 {
        return Ok(());
    }
    if source.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::DeviceMismatch {
            expected: Device::Cuda(ctx.device_id()),
            got: source.device(),
        });
    }
    if destination.device() != ctx.device_id() {
        return Err(Error::Other(format!(
            "2D CUDA destination is on device {}, expected {}",
            destination.device(),
            ctx.device_id()
        )));
    }
    if width > source_pitch || width > destination_pitch {
        return Err(Error::Other(format!(
            "2D CUDA copy width {width} exceeds source/destination pitch {source_pitch}/{destination_pitch}"
        )));
    }
    let source = source
        .storage()
        .as_gpu()
        .ok_or_else(|| Error::Other("expected CUDA source tensor".into()))?;
    let source_required = rows
        .checked_mul(source_pitch)
        .ok_or_else(|| Error::Other("2D CUDA source size overflow".into()))?;
    if source_required > source.len {
        return Err(Error::Other(format!(
            "2D CUDA source requires {source_required} bytes, has {}",
            source.len
        )));
    }
    let destination_required = destination_offset
        .checked_add(
            rows.saturating_sub(1)
                .checked_mul(destination_pitch)
                .and_then(|offset| offset.checked_add(width))
                .ok_or_else(|| Error::Other("2D CUDA destination size overflow".into()))?,
        )
        .ok_or_else(|| Error::Other("2D CUDA destination offset overflow".into()))?;
    if destination_required > destination.len() {
        return Err(Error::Other(format!(
            "2D CUDA destination requires {destination_required} bytes, has {}",
            destination.len()
        )));
    }
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpy2DAsync(
            destination
                .ptr()
                .cast::<u8>()
                .add(destination_offset)
                .cast(),
            destination_pitch,
            source.ptr as *const std::ffi::c_void,
            source_pitch,
            width,
            rows,
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

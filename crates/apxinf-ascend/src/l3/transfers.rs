//! Host ↔ Device transfers for Ascend NPU.
//!
//! Mirrors `apxinf_cuda_new::transfers`.

use apxinf_core::{Device, DType, Error, Result, Shape, Storage, Tensor};
use crate::buffer::AscendBuffer as InnerBuffer;
use crate::ffi::acl::aclrtStream;
use crate::transfers;

/// Copy a CPU tensor to NPU device memory (synchronous).
pub fn to_npu(tensor: &Tensor, device_id: usize) -> Result<Tensor> {
    if tensor.device() == Device::Npu(device_id) {
        return Ok(tensor.clone());
    }
    let cpu_bytes = match tensor.storage() {
        Storage::Cpu(bytes) => bytes.as_slice(),
        _ => {
            return Err(Error::Other(format!(
                "to_npu: source must be CPU, got {:?}",
                tensor.device()
            )))
        }
    };

    let shape = tensor.shape().clone();
    let dtype = tensor.dtype();
    let num_bytes = shape.numel() * dtype.size_in_bytes();

    let buf = InnerBuffer::malloc(num_bytes, device_id)
        .map_err(|e| Error::Other(format!("to_npu alloc: {e}")))?;

    transfers::h2d(
        buf.raw_ptr(),
        cpu_bytes.as_ptr() as *const std::ffi::c_void,
        num_bytes,
    )
    .map_err(|e| Error::Other(format!("to_npu copy: {e}")))?;

    let handle = buf.to_storage_handle();
    Ok(unsafe {
        Tensor::from_raw_parts_unchecked(
            shape,
            dtype,
            Device::Npu(device_id),
            Storage::Gpu {
                device: Device::Npu(device_id),
                handle,
            },
        )
    })
}

/// Copy a CPU tensor to NPU device memory asynchronously on a stream.
///
/// The caller must `aclrtSynchronizeStream` before using the destination
/// buffer. This avoids per-copy host blocking — critical for weight upload
/// where hundreds of tensors are transferred sequentially.
pub fn to_npu_async(tensor: &Tensor, device_id: usize, stream: aclrtStream) -> Result<Tensor> {
    if tensor.device() == Device::Npu(device_id) {
        return Ok(tensor.clone());
    }
    let cpu_bytes = match tensor.storage() {
        Storage::Cpu(bytes) => bytes.as_slice(),
        _ => {
            return Err(Error::Other(format!(
                "to_npu_async: source must be CPU, got {:?}",
                tensor.device()
            )))
        }
    };

    let shape = tensor.shape().clone();
    let dtype = tensor.dtype();
    let num_bytes = shape.numel() * dtype.size_in_bytes();

    let buf = InnerBuffer::malloc(num_bytes, device_id)
        .map_err(|e| Error::Other(format!("to_npu_async alloc: {e}")))?;

    transfers::h2d_async(
        buf.raw_ptr(),
        cpu_bytes.as_ptr() as *const std::ffi::c_void,
        num_bytes,
        stream,
    )
    .map_err(|e| Error::Other(format!("to_npu_async copy: {e}")))?;

    let handle = buf.to_storage_handle();
    Ok(unsafe {
        Tensor::from_raw_parts_unchecked(
            shape,
            dtype,
            Device::Npu(device_id),
            Storage::Gpu {
                device: Device::Npu(device_id),
                handle,
            },
        )
    })
}

/// Copy into an existing shape-identical NPU buffer.
pub fn copy_cpu_to_npu(source: &Tensor, destination: &Tensor) -> Result<()> {
    if source.shape() != destination.shape() || source.dtype() != destination.dtype() {
        return Err(Error::Other(
            "copy_cpu_to_npu: shape or dtype mismatch".into(),
        ));
    }
    let cpu_bytes = match source.storage() {
        Storage::Cpu(bytes) => bytes.as_slice(),
        _ => return Err(Error::Other("copy_cpu_to_npu: source must be CPU".into())),
    };
    let dev_ptr = match destination.storage() {
        Storage::Gpu { handle, .. } => handle.ptr(),
        _ => return Err(Error::Other("copy_cpu_to_npu: dest must be NPU".into())),
    };
    let num_bytes = source.size_in_bytes();
    transfers::h2d(
        dev_ptr as *mut std::ffi::c_void,
        cpu_bytes.as_ptr() as *const std::ffi::c_void,
        num_bytes,
    )
    .map_err(|e| Error::Other(format!("copy_cpu_to_npu: {e}")))
}

/// Copy an NPU tensor to CPU.
pub fn to_cpu(tensor: &Tensor) -> Result<Tensor> {
    if tensor.device() == Device::Cpu {
        return Ok(tensor.clone());
    }
    let dev_ptr = match tensor.storage() {
        Storage::Gpu { handle, .. } => handle.ptr(),
        _ => return Err(Error::Other("to_cpu: not a device tensor".into())),
    };
    let num_bytes = tensor.size_in_bytes();
    let mut cpu_buf = vec![0u8; num_bytes];
    transfers::d2h(
        cpu_buf.as_mut_ptr() as *mut std::ffi::c_void,
        dev_ptr as *const std::ffi::c_void,
        num_bytes,
    )
    .map_err(|e| Error::Other(format!("to_cpu copy: {e}")))?;

    Ok(unsafe {
        Tensor::from_raw_parts_unchecked(
            tensor.shape().clone(),
            tensor.dtype(),
            Device::Cpu,
            Storage::cpu_from_bytes(cpu_buf),
        )
    })
}

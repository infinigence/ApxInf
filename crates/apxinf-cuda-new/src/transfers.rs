//! Safe CUDA tensor transfers and bounds-checked device copy primitives.

use apxinf_core::{Device, Error, Result, Storage, Tensor};

use crate::ffi;
use crate::CudaBuffer;

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
    copy_host_to_device(device_id, source, destination.ptr()).map_err(Error::Cuda)
}

/// Copy a shape-identical CUDA tensor into stable-address CUDA storage on the
/// context stream. Source and destination must be on the same device.
pub fn copy_cuda_to_cuda(
    ctx: &crate::CudaContext,
    source: &Tensor,
    destination: &Tensor,
) -> Result<()> {
    if source.device() != destination.device()
        || source.device() != Device::Cuda(ctx.device_id())
        || source.shape() != destination.shape()
        || source.dtype() != destination.dtype()
    {
        return Err(Error::Other(
            "fixed CUDA copy requires matching device, shape and dtype".into(),
        ));
    }
    let source = CudaBuffer::from_tensor(source).map_err(Error::Cuda)?;
    let destination = CudaBuffer::from_tensor(destination).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpyAsync(
            destination.ptr(),
            source.ptr(),
            source.len(),
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
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
    let bytes = copy_device_to_host(device_id, handle.ptr(), handle.len()).map_err(Error::Cuda)?;
    Tensor::from_raw(tensor.shape().clone(), tensor.dtype(), Device::Cpu, bytes)
}

pub(crate) fn copy_host_to_device(
    device_id: usize,
    source: &[u8],
    destination: usize,
) -> std::result::Result<(), String> {
    let device = i32::try_from(device_id)
        .map_err(|_| format!("CUDA device id {device_id} does not fit in i32"))?;
    unsafe {
        ffi::check_cuda(ffi::cudaSetDevice(device))?;
        ffi::check_cuda(ffi::cudaMemcpy(
            destination as *mut std::ffi::c_void,
            source.as_ptr().cast(),
            source.len(),
            ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
        ))?;
        // A pageable H2D copy may return after host staging, before device DMA.
        // Complete this upload before callers launch on a nonblocking stream.
        ffi::check_cuda(ffi::cudaStreamSynchronize(std::ptr::null_mut()))
    }
}

pub(crate) fn copy_device_to_host(
    device_id: usize,
    source: usize,
    len: usize,
) -> std::result::Result<Vec<u8>, String> {
    let device = i32::try_from(device_id)
        .map_err(|_| format!("CUDA device id {device_id} does not fit in i32"))?;
    let mut destination = vec![0u8; len];
    unsafe {
        ffi::check_cuda(ffi::cudaSetDevice(device))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::{pointwise, PointwiseArgs, PointwiseSemantic};
    use apxinf_core::{DType, Shape};

    #[test]
    fn pageable_upload_precedes_nonblocking_graph_replay() {
        let ctx = crate::CudaContext::new(0).unwrap();
        // The production three-camera BF16 activation size exercises pageable
        // DMA staging, which a scalar upload does not cover.
        let elements = 3 * 256 * 1152;
        let shape = Shape::new(vec![3 * 256, 1152]);
        let input_buffer = CudaBuffer::alloc(elements * 2, 0).unwrap();
        let output_buffer = CudaBuffer::alloc(elements * 2, 0).unwrap();
        let input = input_buffer.as_tensor(shape.clone(), DType::BF16).unwrap();
        let mut output = output_buffer.as_tensor(shape.clone(), DType::BF16).unwrap();
        let bias_buffer = CudaBuffer::alloc(1152 * 2, 0).unwrap();
        bias_buffer.zero_async(ctx.stream()).unwrap();
        let bias = bias_buffer
            .as_tensor(Shape::new(vec![1152]), DType::BF16)
            .unwrap();
        let graph = crate::capture(&ctx, || {
            let mut args =
                PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut output);
            args.bias = Some(&bias);
            pointwise(&ctx, args)
        })
        .unwrap();
        for tensor_copy in [false, true] {
            for iteration in 0..32 {
                let expected = if iteration % 2 == 0 {
                    0x3f80u16
                } else {
                    0xbf80u16
                };
                let bytes = expected.to_ne_bytes().repeat(elements);
                if tensor_copy {
                    let source =
                        Tensor::from_raw(shape.clone(), DType::BF16, Device::Cpu, bytes).unwrap();
                    copy_cpu_to_cuda(&source, &input).unwrap();
                } else {
                    input_buffer.copy_from_host(&bytes).unwrap();
                }
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                let mut actual = vec![0; elements * 2];
                output_buffer.copy_to_host(&mut actual).unwrap();
                if let Some((index, value)) = actual
                    .chunks_exact(2)
                    .enumerate()
                    .map(|(index, bytes)| (index, u16::from_ne_bytes([bytes[0], bytes[1]])))
                    .find(|(_, value)| *value != expected)
                {
                    panic!("upload/replay ordering: tensor_copy={tensor_copy}, iteration={iteration}, index={index}, expected={expected:#x}, actual={value:#x}");
                }
            }
        }
    }
}

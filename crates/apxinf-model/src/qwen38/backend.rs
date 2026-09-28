//! Device helpers: upload, zero-fill, readback and typed views.
//!
//! The model-family CUDA seam in the sense of
//! `doc/model-layer-architecture.md`: every other qwen38 module reaches
//! device memory through these helpers or through `apxinf_cuda_new::ops`.


use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{CudaBuffer, CudaContext};


pub(crate) fn cpu_bytes(tensor: &Tensor) -> &[u8] {
    match tensor.storage() {
        apxinf_core::Storage::Cpu(data) => data,
        _ => panic!("expected a CPU tensor"),
    }
}

pub(crate) fn upload(ctx: &CudaContext, bytes: &[u8], dims: Vec<usize>, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len().max(1), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

pub(crate) fn zeros(ctx: &CudaContext, dims: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = dims.iter().product::<usize>() * dtype.size_in_bytes();
    let buffer = CudaBuffer::alloc(bytes.max(1), ctx.device_id()).unwrap();
    buffer.copy_from_host(&vec![0u8; bytes.max(1)]).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

pub(crate) fn graph_tensor_bytes(tensor: &Tensor) -> Vec<u8> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
}

/// Report the magnitude range of a BF16 tensor, to decide whether an FP16
/// kernel could carry it.
///
/// FlashInfer's generic GDN prefill variant -- the one our 48 value heads and
/// group of 3 force us onto, neither being a power of two -- exists only with
/// FP16 I/O. FP16 saturates at 65504 and loses its last normal near 6.1e-5,
/// so adopting it hinges on where these activations actually sit. Measuring
/// beats assuming: the checkpoint is calibrated, and its magnitudes are not
/// obvious from the architecture.
pub(crate) fn prefix(tensor: &Tensor, dims: Vec<usize>, dtype: DType) -> Tensor {
    let span = dims.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .view(0, span)
        .unwrap()
        .as_tensor(Shape::new(dims), dtype)
        .unwrap()
}

pub(crate) fn view(tensor: &Tensor, dims: Vec<usize>, dtype: DType) -> Tensor {
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .as_tensor(Shape::new(dims), dtype)
        .unwrap()
}

pub(crate) fn zero_tensor(tensor: &Tensor) {
    // Device-side memset: reset_state clears ~427 MB (48 recurrent states +
    // conv windows + KV caches), and a host staging copy of that size costs
    // ~120 ms per generation on Thor.
    CudaBuffer::from_tensor(tensor).unwrap().zero().unwrap();
}

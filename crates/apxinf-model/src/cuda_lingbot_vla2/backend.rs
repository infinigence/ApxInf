//! cuda-new seam for the LingBot-VLA 2.0 family. Model code reaches the device only through these
//! safe operators and resources.

pub(crate) use crate::accelerator::downcast_cuda_new_arc as downcast_arc;
pub(crate) use apxinf_cuda_new::{
    ops, CudaBuffer as DeviceBuffer, CudaContext as Context, CudaNewBackend as RuntimeBackend,
};

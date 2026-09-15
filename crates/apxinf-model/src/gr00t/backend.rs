//! CUDA-facing seam for the GR00T runtime.

pub(crate) use crate::accelerator::cuda::{
    downcast_arc, kernels, transfers, DeviceBuffer, RuntimeBackend,
};
pub(crate) use apxinf_cuda::{tuning::TuningMode, CudaBackend};

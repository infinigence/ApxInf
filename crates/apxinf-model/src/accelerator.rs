//! Compile-time accelerator seam for model implementations.

use std::sync::Arc;

use apxinf_core::{Backend, Device, Error, Result};

pub(crate) fn create_backend(device: Device) -> Result<Arc<dyn Backend>> {
    match device {
        Device::Cpu => Ok(Arc::new(apxinf_core::CpuBackend)),
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => create_cuda_new_backend(device),
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => Err(Error::Other("CUDA support not compiled in".into())),
        #[cfg(feature = "hip")]
        Device::Hip(id) => hip::create(id),
        #[cfg(not(feature = "hip"))]
        Device::Hip(_) => Err(Error::Other("HIP support not compiled in".into())),
    }
}

/// AMD ROCm/HIP seam.
///
/// Deliberately thin compared to [`cuda`]: models reach this backend only
/// through `dyn Backend`, so there is no downcast helper. A family that needs
/// HIP-specific fused kernels would add one here, the same way PI0.5 does for
/// CUDA — but nothing needs that yet, and adding it before something does would
/// invite the coupling this backend was built to avoid.
#[cfg(feature = "hip")]
pub(crate) mod hip {
    use std::sync::Arc;

    use apxinf_core::{Backend, Result};

    pub(crate) fn create(device_id: usize) -> Result<Arc<dyn Backend>> {
        Ok(Arc::new(apxinf_hip::HipBackend::new(device_id)?))
    }
}

/// Create a backend on the `apxinf-cuda-new` runtime.
///
/// Models written entirely against `dyn Backend` — llama and qwen3-vl — take
/// this path. The returned backend satisfies the same portable trait as the
/// legacy one, so those families need no code change to switch runtimes: the
/// model composes unchanged and only the concrete backend differs.
#[cfg(feature = "cuda")]
pub(crate) fn create_cuda_new_backend(device: Device) -> Result<Arc<dyn Backend>> {
    let Device::Cuda(id) = device else {
        return Err(Error::Other(
            "the cuda-new backend requires a CUDA device".into(),
        ));
    };
    let ctx = Arc::new(apxinf_cuda_new::CudaContext::new(id).map_err(Error::Cuda)?);
    Ok(Arc::new(apxinf_cuda_new::CudaNewBackend::new(ctx)))
}

/// Recover the concrete cuda-new backend from an `Arc<dyn Backend>`.
///
/// Mirrors `cuda::downcast_arc` for families whose executors call the
/// cuda-new kernel seam directly (π0-FAST).
#[cfg(feature = "cuda")]
pub(crate) fn downcast_cuda_new_arc(
    backend: Arc<dyn Backend>,
) -> Option<Arc<apxinf_cuda_new::CudaNewBackend>> {
    backend
        .as_any()
        .downcast_ref::<apxinf_cuda_new::CudaNewBackend>()?;
    let raw = Arc::into_raw(backend);
    // SAFETY: the exact type was checked above. This keeps the same Arc
    // allocation and strong count while dropping only trait-object metadata.
    Some(unsafe { Arc::from_raw(raw as *const apxinf_cuda_new::CudaNewBackend) })
}

/// Borrow the concrete cuda-new backend from a `&dyn Backend`.
///
/// The llama / qwen3-vl decode graphs compose cuda-new kernels directly
/// (allocation-free decode workspace + CUDA Graph capture); this is their
/// intentional seam to the concrete backend.
#[cfg(feature = "cuda")]
pub(crate) fn downcast_cuda_new(
    backend: &dyn Backend,
) -> Option<&apxinf_cuda_new::CudaNewBackend> {
    backend
        .as_any()
        .downcast_ref::<apxinf_cuda_new::CudaNewBackend>()
}

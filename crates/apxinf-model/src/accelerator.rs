//! Compile-time accelerator seam for model implementations.

use std::sync::Arc;

use apxinf_core::{Backend, Device, Result};
#[cfg(not(feature = "cuda"))]
use apxinf_core::Error;

pub(crate) fn create_backend(device: Device) -> Result<Arc<dyn Backend>> {
    match device {
        Device::Cpu => Ok(Arc::new(apxinf_core::CpuBackend)),
        #[cfg(feature = "cuda")]
        Device::Cuda(id) => cuda::create(id),
        #[cfg(not(feature = "cuda"))]
        Device::Cuda(_) => Err(Error::Other("CUDA support not compiled in".into())),
    }
}

#[cfg(feature = "cuda")]
pub(crate) mod cuda {
    use std::sync::Arc;

    use apxinf_core::{Backend, Result};

    pub(crate) use apxinf_cuda::kernels;
    pub(crate) use apxinf_cuda::nvtx;
    pub(crate) use apxinf_cuda::transfers;
    pub(crate) use apxinf_cuda::tuning;
    pub(crate) use apxinf_cuda::{
        CublasTranspose, CudaBuffer as DeviceBuffer, CudaContext as Context,
        CudaDeviceAddress as DeviceAddress, CudaKVCache as KvCache,
        HostMappedBuffer as MappedBuffer,
    };
    pub(crate) type RuntimeBackend = apxinf_cuda::CudaBackend;

    /// Recover the concrete `CudaBackend` from a `&dyn Backend`.
    ///
    /// This downcast is intentional, not a design gap. PI0.5's hot path calls
    /// CUDA-specific fused kernels (e.g. `ada_gate_residual_rms_norm`,
    /// `qkv_rope`, `euler_update`) that the portable `apxinf_core::Backend`
    /// trait deliberately does not expose. The uniform `dyn Backend` only
    /// buys a single loading entry point (the registry returns `dyn`); the
    /// fused executor then recovers the concrete backend here on purpose.
    /// See doc/20260812-pi05-backend-decoupling/design.md.
    pub(crate) fn downcast(backend: &dyn Backend) -> Option<&RuntimeBackend> {
        backend.as_any().downcast_ref::<RuntimeBackend>()
    }

    pub(crate) fn downcast_arc(backend: Arc<dyn Backend>) -> Option<Arc<RuntimeBackend>> {
        downcast(&*backend)?;
        let raw = Arc::into_raw(backend);
        // SAFETY: the exact RuntimeBackend type was checked above. This keeps
        // the same Arc allocation and strong count while dropping only the
        // trait-object metadata.
        Some(unsafe { Arc::from_raw(raw as *const RuntimeBackend) })
    }

    pub(crate) fn create(device_id: usize) -> Result<Arc<dyn Backend>> {
        Ok(Arc::new(RuntimeBackend::new(device_id)?))
    }
}

use std::sync::Arc;
#[cfg(feature = "opaque-storage")]
use std::{any::Any, rc::Rc};

use crate::Device;

/// Raw data backing for a tensor.
///
/// CPU storage is a byte buffer. Pointer-backed GPU storage retains its owning
/// allocation. Optional opaque storage retains a lazy, thread-affine backend array.
#[derive(Debug, Clone)]
pub enum Storage {
    /// CPU-side contiguous byte buffer.
    Cpu(Vec<u8>),
    /// GPU storage owned by the CUDA backend.
    Gpu {
        device: Device,
        handle: GpuStorageHandle,
    },
    /// Thread-affine backend storage. This is an array owner, never a raw pointer.
    #[cfg(feature = "opaque-storage")]
    Opaque {
        device: Device,
        handle: OpaqueStorageHandle,
    },
}

/// Opaque handle to GPU memory. CUDA backends construct these and retain the
/// owning allocation so device memory stays live while the tensor is live.
///
/// The public fields are retained during the `apxinf-cuda-new` migration so
/// the existing CUDA backend keeps building unchanged. New backends should use
/// [`GpuStorageHandle::from_raw_parts`] and the accessor methods instead.
#[derive(Clone)]
pub struct GpuStorageHandle {
    /// Raw CUDA device pointer, cast to `usize`.
    pub ptr: usize,
    /// Total allocated bytes on device.
    pub len: usize,
    /// Retains the backend allocation owner until the last handle is dropped.
    pub _prevent_leak: Option<Arc<dyn std::any::Any + Send + Sync>>,
}

impl GpuStorageHandle {
    /// Construct a handle for a backend-owned GPU allocation.
    ///
    /// # Safety
    ///
    /// For every non-empty handle, `ptr..ptr + len` must be a valid device
    /// allocation on the declared [`Device`] for as long as `owner` is alive.
    /// The range must not wrap around the address space. Backends must retain
    /// an owner that releases the allocation only after the last handle drops.
    pub unsafe fn from_raw_parts(
        ptr: usize,
        len: usize,
        owner: Option<Arc<dyn std::any::Any + Send + Sync>>,
    ) -> Self {
        Self {
            ptr,
            len,
            _prevent_leak: owner,
        }
    }

    pub fn ptr(&self) -> usize {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn owner(&self) -> Option<&Arc<dyn std::any::Any + Send + Sync>> {
        self._prevent_leak.as_ref()
    }
}

impl std::fmt::Debug for GpuStorageHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuStorageHandle")
            .field("ptr", &format_args!("0x{:x}", self.ptr))
            .field("len", &self.len)
            .finish()
    }
}

impl Storage {
    /// Create a zeroed CPU buffer.
    pub fn cpu_zeros(num_bytes: usize) -> Self {
        Storage::Cpu(vec![0u8; num_bytes])
    }

    /// Create a CPU buffer from existing bytes.
    pub fn cpu_from_bytes(data: Vec<u8>) -> Self {
        Storage::Cpu(data)
    }

    /// Number of bytes in this storage.
    pub fn len(&self) -> usize {
        match self {
            Storage::Cpu(v) => v.len(),
            Storage::Gpu { handle, .. } => handle.len,
            #[cfg(feature = "opaque-storage")]
            Storage::Opaque { handle, .. } => handle.len(),
        }
    }

    /// Whether this storage is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get a reference to the CPU data, or `None` if on GPU.
    pub fn as_cpu(&self) -> Option<&[u8]> {
        match self {
            Storage::Cpu(v) => Some(v),
            Storage::Gpu { .. } => None,
            #[cfg(feature = "opaque-storage")]
            Storage::Opaque { .. } => None,
        }
    }

    /// Get a mutable reference to the CPU data, or `None` if on GPU.
    pub fn as_cpu_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Storage::Cpu(v) => Some(v),
            Storage::Gpu { .. } => None,
            #[cfg(feature = "opaque-storage")]
            Storage::Opaque { .. } => None,
        }
    }

    /// Get a reference to the GPU handle, or `None` if on CPU.
    pub fn as_gpu(&self) -> Option<&GpuStorageHandle> {
        match self {
            Storage::Cpu(_) => None,
            Storage::Gpu { handle, .. } => Some(handle),
            #[cfg(feature = "opaque-storage")]
            Storage::Opaque { .. } => None,
        }
    }
}

/// A retained backend-specific lazy array with a checked logical byte extent.
///
/// The owning backend must downcast and validate its actual dtype, device, and
/// element count before use. Core never dereferences this owner. It deliberately
/// uses Rc: enabling opaque storage makes Tensor thread-affine without imposing
/// unproven Send/Sync guarantees on a native runtime.
#[cfg(feature = "opaque-storage")]
#[derive(Clone)]
pub struct OpaqueStorageHandle {
    owner: Rc<dyn Any>,
    len: usize,
}

#[cfg(feature = "opaque-storage")]
impl OpaqueStorageHandle {
    pub(crate) fn new(owner: Rc<dyn Any>, len: usize) -> Self {
        Self { owner, len }
    }

    pub fn owner(&self) -> &Rc<dyn Any> {
        &self.owner
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[cfg(feature = "opaque-storage")]
impl std::fmt::Debug for OpaqueStorageHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpaqueStorageHandle")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

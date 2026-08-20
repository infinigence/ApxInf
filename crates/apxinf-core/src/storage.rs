use std::sync::Arc;

use crate::Device;

/// Raw data backing for a tensor.
///
/// CPU storage is a byte buffer. CUDA storage holds an opaque
/// handle that the backend crate interprets.
#[derive(Debug, Clone)]
pub enum Storage {
    /// CPU-side contiguous byte buffer.
    Cpu(Vec<u8>),
    /// GPU storage owned by the CUDA backend.
    Gpu {
        device: Device,
        handle: GpuStorageHandle,
    },
}

/// Opaque handle to GPU memory. `apxinf-cuda` constructs these and stores its
/// owning buffer in `_prevent_leak` so that
/// device memory is freed when all references are dropped.
#[derive(Clone)]
pub struct GpuStorageHandle {
    /// Raw CUDA device pointer, cast to usize.
    pub ptr: usize,
    /// Total allocated bytes on device.
    pub len: usize,
    /// Holds the owning backend buffer (for example, `Arc<CudaBuffer>`)
    /// so that Drop frees GPU memory automatically.
    pub _prevent_leak: Option<Arc<dyn std::any::Any + Send + Sync>>,
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
        }
    }

    /// Get a mutable reference to the CPU data, or `None` if on GPU.
    pub fn as_cpu_mut(&mut self) -> Option<&mut [u8]> {
        match self {
            Storage::Cpu(v) => Some(v),
            Storage::Gpu { .. } => None,
        }
    }

    /// Get a reference to the GPU handle, or `None` if on CPU.
    pub fn as_gpu(&self) -> Option<&GpuStorageHandle> {
        match self {
            Storage::Cpu(_) => None,
            Storage::Gpu { handle, .. } => Some(handle),
        }
    }
}

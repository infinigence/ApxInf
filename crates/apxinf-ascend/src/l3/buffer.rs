//! Device buffer abstraction for Ascend NPU.
//!
//! Mirrors `apxinf_cuda_new::CudaBuffer` — owns a contiguous device memory
//! region and supports view/as_tensor conversions.

use apxinf_core::{DType, Device, Error, Result, Shape, Storage, Tensor};
use crate::buffer::AscendBuffer as InnerBuffer;

/// An Ascend device buffer wrapping a contiguous allocation.
#[derive(Clone)]
pub struct AscendBuffer {
    inner: InnerBuffer,
    device_id: usize,
}

impl AscendBuffer {
    /// Allocate zero-initialized device memory of `bytes` bytes.
    pub fn alloc_zeros(bytes: usize, device_id: usize) -> Result<Self> {
        Self::alloc_zeros_impl(bytes, device_id)
    }

    /// Session-aware allocation: sub-allocate from pool if in session (no memset),
    /// otherwise fall back to `alloc_zeros`.
    pub fn alloc_zeros_session_aware(bytes: usize, device_id: usize) -> Result<Self> {
        if bytes == 0 {
            return Ok(Self::from_raw_ptr(0, 0, device_id));
        }
        if crate::is_in_session() {
            let pool_ptr = crate::current_workspace_pool();
            if !pool_ptr.is_null() {
                let pool = unsafe { &*pool_ptr };
                let (ptr, owned) = pool
                    .alloc(bytes as u64)
                    .map_err(|e| Error::Other(format!("pool alloc: {e}")))?;
                if !owned {
                    return Ok(Self::from_raw_ptr(ptr as usize, bytes, device_id));
                }
                // Fallback: individually allocated by pool (capacity
                // exceeded), but not zeroed. Skip memset in session — all
                // aclnn ops fully write their outputs, and aclrtMemset is
                // forbidden during RI capture (error 107030).
                crate::note_output_overflow(bytes);
                if crate::diag_mem_enabled() {
                    let (used, cap) = (pool.used(), pool.capacity());
                    eprintln!(
                        "[diag-mem] SESSION OUTPUT OVERFLOW bytes={bytes} ({:.1}MB) pool.used={used} cap={cap} ptr={:#x} mode={}",
                        bytes as f64 / (1024.0 * 1024.0),
                        ptr as usize,
                        if crate::capture_active() { "capture" } else { "eager" },
                    );
                }
                // Overflow output beyond pool capacity: register it with the
                // pool so it is released instead of leaked (A1). A bare
                // owner-on-drop is NOT safe in this async-session engine (the
                // buffer is still referenced by enqueued aclnn ops after the
                // tensor drops), so release is deferred:
                //  - eager/prepare (`pinned=false`): freed at the next
                //    `reset()`, after the runner synchronizes the stream;
                //  - RI capture (`pinned=true`): freed with the pool after the
                //    graph is destroyed, because replay references the address.
                pool.register_overflow(ptr, crate::capture_active());
                let inner = InnerBuffer::from_raw_ptr(ptr as usize, bytes, device_id);
                return Ok(Self { inner, device_id });
            }
        }
        Self::alloc_zeros_impl(bytes, device_id)
    }

    fn alloc_zeros_impl(bytes: usize, device_id: usize) -> Result<Self> {
        // Ensure the current thread has an ACL context bound.
        crate::context::restore_context();
        let inner = InnerBuffer::malloc(bytes, device_id)
            .map_err(|e| Error::Other(format!("AscendBuffer alloc: {e}")))?;
        let raw = inner.raw_ptr();
        if bytes > 0 {
            // Re-ensure context before memset (malloc may have been fine but
            // context could still be missing on this thread).
            crate::context::restore_context();
            let ret = unsafe {
                crate::ffi::acl::aclrtMemset(
                    raw as *mut std::ffi::c_void,
                    bytes,
                    0,
                    bytes,
                )
            };
            // Auto-recover from CONTEXT_NULL (107002).
            let ret = if ret == 107002 {
                crate::context::restore_context();
                unsafe {
                    crate::ffi::acl::aclrtMemset(
                        raw as *mut std::ffi::c_void,
                        bytes,
                        0,
                        bytes,
                    )
                }
            } else {
                ret
            };
            crate::ffi::acl::check(ret, "aclrtMemset zeros")
                .map_err(|e| Error::Other(e))?;
        }
        Ok(Self { inner, device_id })
    }

    /// Borrow the device memory from a tensor.
    pub fn from_tensor(tensor: &Tensor) -> Result<Self> {
        let ptr = match tensor.storage() {
            Storage::Gpu { handle, .. } => handle.ptr(),
            _ => return Err(Error::Other("from_tensor: not a device tensor".into())),
        };
        let device_id = match tensor.device() {
            Device::Npu(id) => id,
            _ => return Err(Error::Other("from_tensor: not on NPU".into())),
        };
        let inner = InnerBuffer::from_raw_ptr(ptr, tensor.size_in_bytes(), device_id);
        Ok(Self { inner, device_id })
    }

    /// Create a sub-view of this buffer at `offset` with `len` bytes.
    pub fn view(&self, offset: usize, len: usize) -> Result<Self> {
        let base = self.inner.ptr();
        if offset + len > self.inner.len() {
            return Err(Error::Other(format!(
                "view: offset({offset}) + len({len}) > buffer size({})",
                self.inner.len()
            )));
        }
        let inner = InnerBuffer::from_raw_ptr(base + offset, len, self.device_id);
        Ok(Self { inner, device_id: self.device_id })
    }

    /// Borrow this buffer as a tensor with the given shape and dtype.
    pub fn as_tensor(&self, shape: Shape, dtype: DType) -> Result<Tensor> {
        let handle = self.inner.to_storage_handle();
        Ok(unsafe {
            Tensor::from_raw_parts_unchecked(
                shape,
                dtype,
                Device::Npu(self.device_id),
                Storage::Gpu {
                    device: Device::Npu(self.device_id),
                    handle,
                },
            )
        })
    }

    /// Raw device pointer.
    pub fn raw_ptr(&self) -> usize {
        self.inner.ptr()
    }

    /// Buffer size in bytes.
    pub fn size(&self) -> usize {
        self.inner.len()
    }

    /// Device ID.
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Wrap an existing device pointer without ownership.
    pub fn from_raw_ptr(ptr: usize, size: usize, device_id: usize) -> Self {
        let inner = InnerBuffer::from_raw_ptr(ptr, size, device_id);
        Self { inner, device_id }
    }

    /// Copy host bytes into this device buffer.
    pub fn copy_from_host(&self, host_bytes: &[u8]) -> Result<()> {
        let len = host_bytes.len();
        if len > self.inner.len() {
            return Err(Error::Other(format!(
                "copy_from_host: {len} bytes > buffer size {}",
                self.inner.len()
            )));
        }
        if len > 0 {
            crate::transfers::h2d(
                self.inner.ptr() as *mut std::ffi::c_void,
                host_bytes.as_ptr() as *const std::ffi::c_void,
                len,
            )
            .map_err(|e| Error::Other(format!("copy_from_host: {e}")))?;
        }
        Ok(())
    }
}

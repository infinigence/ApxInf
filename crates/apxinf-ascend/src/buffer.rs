//! Device buffer allocation and lifecycle.
//!
//! Wraps `aclrtMalloc` / `aclrtFree` into an owned handle that also
//! constructs a `GpuStorageHandle` so device tensors can carry the
//! allocation through `apxinf_core::Storage::Gpu`.

use std::ffi::c_void;
use std::sync::Arc;

use apxinf_core::{Device, GpuStorageHandle};

use crate::ffi::acl::{self, aclrtMemMallocPolicy};

/// Owning handle to a contiguous block of NPU device memory.
///
/// The pointer is valid for the lifetime of this struct. When the last
/// `AscendBuffer` clone drops, the memory is freed via `aclrtFree`.
pub struct AscendBuffer {
    ptr: usize,
    len: usize,
    device: Device,
    /// Shared ownership so multiple tensors can reference the same allocation.
    _owner: Option<Arc<BufferOwner>>,
}

struct BufferOwner {
    ptr: *mut c_void,
}

impl Drop for BufferOwner {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { acl::aclrtFree(self.ptr) };
        }
    }
}

// SAFETY: The raw pointer points to device memory managed by ACL. It is safe
// to share across threads because ACL operations are serialized through streams.
unsafe impl Send for BufferOwner {}
unsafe impl Sync for BufferOwner {}

impl AscendBuffer {
    /// Allocate `len` bytes of NPU device memory.
    pub fn malloc(len: usize, device_id: usize) -> Result<Self, String> {
        // Preventive: ensure the current thread has an ACL context bound.
        crate::context::restore_context();

        let t0 = std::time::Instant::now();
        let mut dev_ptr: *mut c_void = std::ptr::null_mut();
        let ret = unsafe {
            acl::aclrtMalloc(
                &mut dev_ptr,
                len,
                aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST,
            )
        };
        // Auto-recover from CONTEXT_NULL (107002): restore the global context
        // and retry once.
        let ret = if ret == 107002 {
            crate::context::restore_context();
            unsafe {
                acl::aclrtMalloc(
                    &mut dev_ptr,
                    len,
                    aclrtMemMallocPolicy::ACL_MEM_MALLOC_HUGE_FIRST,
                )
            }
        } else {
            ret
        };
        acl::check(ret, "aclrtMalloc")?;
        crate::transfers::record_malloc_us(t0.elapsed().as_micros() as u64);

        let owner = Arc::new(BufferOwner { ptr: dev_ptr });

        Ok(Self {
            ptr: dev_ptr as usize,
            len,
            device: Device::Npu(device_id),
            _owner: Some(owner),
        })
    }

    /// Wrap an existing device pointer without taking ownership.
    /// The caller is responsible for freeing the memory.
    pub fn from_raw_ptr(ptr: usize, len: usize, device_id: usize) -> Self {
        Self {
            ptr,
            len,
            device: Device::Npu(device_id),
            _owner: None,
        }
    }

    pub fn ptr(&self) -> usize {
        self.ptr
    }

    pub fn raw_ptr(&self) -> *mut c_void {
        self.ptr as *mut c_void
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Construct a `GpuStorageHandle` so this buffer can be stored in a
    /// `apxinf_core::Tensor` via `Storage::Gpu`.
    ///
    /// The handle's `_prevent_leak` field holds an `Arc` to the `BufferOwner`,
    /// keeping the allocation alive as long as any handle clone exists.
    pub fn to_storage_handle(&self) -> GpuStorageHandle {
        let owner_arc: Arc<dyn std::any::Any + Send + Sync> = match &self._owner {
            Some(o) => Arc::clone(o) as Arc<dyn std::any::Any + Send + Sync>,
            None => Arc::new(()) as Arc<dyn std::any::Any + Send + Sync>,
        };
        // SAFETY: ptr..ptr+len is a valid ACL allocation for the owner's lifetime.
        unsafe { GpuStorageHandle::from_raw_parts(self.ptr, self.len, Some(owner_arc)) }
    }
}

impl Clone for AscendBuffer {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr,
            len: self.len,
            device: self.device,
            _owner: self._owner.clone(),
        }
    }
}

impl std::fmt::Debug for AscendBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AscendBuffer")
            .field("ptr", &format_args!("0x{:x}", self.ptr))
            .field("len", &self.len)
            .field("device", &self.device)
            .finish()
    }
}

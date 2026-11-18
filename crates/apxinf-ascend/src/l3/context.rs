//! Ascend L3 context — wraps `crate::AscendContext` and provides
//! the `allocate_output` method used by PI0.5's L3 operator contracts.

use apxinf_core::{DType, Error, Result, Shape, Storage, Tensor, Device};

use crate::l3::buffer::AscendBuffer;

/// L3 context for Ascend NPU, mirroring `apxinf_cuda_new::CudaContext`.
pub struct AscendNewContext {
    device_id: usize,
    inner: crate::AscendContext,
}

impl AscendNewContext {
    /// Create a new context on the given device.
    pub fn new(device_id: usize) -> Result<Self> {
        let inner = crate::AscendContext::new(device_id)
            .map_err(|e| Error::Other(format!("AscendNewContext init: {e}")))?;
        Ok(Self { device_id, inner })
    }

    /// Device ID.
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Synchronize the device.
    pub fn synchronize(&self) -> Result<()> {
        self.inner
            .synchronize_device()
            .map_err(|e| Error::Other(format!("synchronize: {e}")))
    }

    /// Allocate an output tensor with the given shape and dtype.
    ///
    /// During graph capture, the output tensor's device pointer is baked
    /// into the captured graph —  writes the new result to that
    /// exact address. Therefore the buffer MUST come from the session pool
    /// (deterministic address: pool is reset() at session start, so the
    /// same offset is reused every inference). An owned  would
    /// return a different address each call, causing 507035 on replay.
    ///
    /// Outside a session (eager mode), fall back to owned .
    pub fn allocate_output(&self, shape: Shape, dtype: DType) -> Result<Tensor> {
        self.ensure_context();
        let num_bytes = shape.numel() * dtype.size_in_bytes();
        let buf = AscendBuffer::alloc_zeros_session_aware(num_bytes, self.device_id)?;
        buf.as_tensor(shape, dtype)
    }

    /// Re-set the ACL current context for this thread.
    ///
    /// ACL context is thread-local: `aclrtSetCurrentContext` only affects the
    /// calling thread. If the calling thread never set a context (e.g. a
    /// thread-pool worker), `aclrtMalloc` returns 107002 (CONTEXT_NULL).
    /// This method ensures the context is bound before any ACL call.
    pub fn ensure_context(&self) {
        let ctx = self.inner.raw_context();
        unsafe {
            crate::ffi::acl::aclrtSetCurrentContext(ctx);
        }
    }

    /// Raw ACL stream pointer (for passing to aclnn operators).
    pub fn raw_stream(&self) -> crate::ffi::acl::aclrtStream {
        self.ensure_context();
        self.inner.raw_stream()
    }

    /// Reference to the underlying AscendContext.
    pub fn inner(&self) -> &crate::AscendContext {
        &self.inner
    }
}

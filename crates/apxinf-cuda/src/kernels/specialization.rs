//! Scoped opt-in to measured static-FP8 kernel specializations.
use crate::{ffi, CudaContext};
use std::{marker::PhantomData, rc::Rc};

/// Select SM110 specializations for one synchronous traversal. The policy is
/// thread-local, defaults off, and is restored on return, error, or panic.
/// Keep the guard on its creating thread; captured graphs retain their selected
/// kernels without requiring this host policy during replay.
pub struct StaticFp8KernelScope {
    previous: i32,
    _not_send: PhantomData<Rc<()>>,
}

impl StaticFp8KernelScope {
    pub fn new(ctx: &CudaContext, requested: bool) -> Self {
        let enabled = requested && ctx.caps().sm == 110;
        let previous = unsafe { ffi::apxinf_set_static_fp8_specializations(i32::from(enabled)) };
        Self {
            previous,
            _not_send: PhantomData,
        }
    }
}

impl Drop for StaticFp8KernelScope {
    fn drop(&mut self) {
        unsafe {
            ffi::apxinf_set_static_fp8_specializations(self.previous);
        }
    }
}

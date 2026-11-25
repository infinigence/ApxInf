use std::ptr::NonNull;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::Result;

use super::contracts::{normalize, Unfold3dArgs};
use crate::ffi::abi::{status, volume as abi};
use crate::ops::gemm::contracts::invalid;
use crate::{CudaBuffer, CudaContext, CudaStream};

/// One fixed baseline implementation, with zero device workspace and no
/// tuning. Preparation binds storage; enqueue is allocation-free and graph
/// safe. Captured graphs retain the execution and all input/output storage.
pub struct PreparedUnfold3d(Rc<Execution>);

struct Execution {
    execution: Option<NonNull<std::ffi::c_void>>,
    storage: Vec<CudaBuffer>,
    stream: Arc<CudaStream>,
}

impl PreparedUnfold3d {
    pub fn new(ctx: &CudaContext, args: Unfold3dArgs<'_>) -> Result<Self> {
        let normalized = normalize(ctx, args)?;
        if crate::workspace::is_measurement() {
            return Ok(Self(Rc::new(Execution {
                execution: None,
                storage: normalized.storage,
                stream: ctx.shared_stream(),
            })));
        }
        if !crate::workspace::may_prepare_native_resources() {
            return Err(invalid("Unfold3d must be prepared before capture"));
        }
        let mut execution = std::ptr::null_mut();
        unsafe {
            status::check(abi::apxinf_unfold3d_prepare(
                ctx.runtime(),
                &normalized.spec,
                &normalized.bindings,
                &mut execution,
            ))?;
        }
        Ok(Self(Rc::new(Execution {
            execution: Some(
                NonNull::new(execution).ok_or_else(|| invalid("null Unfold3d execution"))?,
            ),
            storage: normalized.storage,
            stream: ctx.shared_stream(),
        })))
    }

    pub fn enqueue(&self) -> Result<()> {
        if crate::workspace::is_measurement() {
            return Ok(());
        }
        let execution = &self.0;
        let _keep_alive = &execution.storage;
        crate::workspace::validate_capture_target(
            execution.stream.device(),
            execution.stream.handle() as usize,
        )?;
        crate::workspace::use_execution(execution)?;
        unsafe {
            status::check(abi::apxinf_unfold3d_enqueue(
                execution
                    .execution
                    .ok_or_else(|| invalid("measurement Unfold3d cannot execute"))?
                    .as_ptr(),
            ))
        }
    }
}

impl Drop for Execution {
    fn drop(&mut self) {
        unsafe {
            if let Some(execution) = self.execution {
                abi::apxinf_unfold3d_destroy(execution.as_ptr());
            }
        }
    }
}

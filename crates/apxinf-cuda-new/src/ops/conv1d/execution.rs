//! Prepared convolution resources owned by the execution session.
use super::contracts::Normalized;
use crate::ffi::abi::{conv1d as abi, status};
use crate::{CudaBuffer, CudaContext};
use apxinf_core::{Error, Result};
use std::{rc::Rc, sync::Arc};

const WORKSPACE_LIMIT: u64 = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExecutionKey {
    spec: abi::Spec,
    accumulation: super::contracts::Conv1dAccumulation,
    device: usize,
    input: usize,
    weight: usize,
    bias: usize,
    output: usize,
    stream: usize,
}

struct Execution {
    raw: abi::Execution,
    stream: Arc<crate::CudaStream>,
    _storage: Vec<CudaBuffer>,
}
impl Drop for Execution {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        unsafe { abi::apxinf_conv1d_destroy(self.raw) }
    }
}

pub(super) fn execute(ctx: &CudaContext, normalized: Normalized) -> Result<()> {
    let key = ExecutionKey {
        spec: normalized.spec,
        accumulation: normalized.accumulation,
        device: ctx.device_id(),
        input: normalized.bindings.input as usize,
        weight: normalized.bindings.weight as usize,
        bias: normalized.bindings.bias as usize,
        output: normalized.bindings.output as usize,
        stream: normalized.bindings.stream as usize,
    };
    let execution =
        if let Some(execution) = crate::workspace::lookup_execution::<_, Execution>(&key) {
            // The preparation traversal moved these bindings into the cached
            // execution, so their arena leases stay alive until the session
            // ends. A capture-time cache hit creates fresh leases for the same
            // addresses; release them at the same boundary or every later
            // arena allocation diverges from the prepared address sequence.
            if crate::workspace::is_capturing() {
                for buffer in &normalized.storage {
                    crate::workspace::retain_buffer(buffer);
                }
            }
            execution
        } else {
            if !crate::workspace::may_prepare_native_resources() {
                return Err(Error::Other(
                    "Conv1d execution cache miss; prepare the same bindings first".into(),
                ));
            }
            let mut raw = std::ptr::null_mut();
            unsafe {
                status::check(abi::apxinf_conv1d_prepare(
                    ctx.runtime(),
                    &normalized.spec,
                    &normalized.bindings,
                    WORKSPACE_LIMIT,
                    u32::from(
                        normalized.accumulation == super::contracts::Conv1dAccumulation::OrderedF32,
                    ),
                    &mut raw,
                ))?;
            }
            if raw.is_null() {
                return Err(Error::Other(
                    "Conv1d prepare returned null execution".into(),
                ));
            }
            let execution = Rc::new(Execution {
                raw,
                stream: ctx.shared_stream(),
                _storage: normalized.storage,
            });
            crate::workspace::store_execution(key, execution.clone());
            execution
        };
    crate::workspace::use_execution(&execution)?;
    crate::workspace::validate_capture_target(ctx.device_id(), execution.stream.handle() as usize)?;
    execution.stream.set_current_device().map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_conv1d_enqueue(execution.raw))?;
    }
    if crate::workspace::is_capturing() || crate::workspace::has_active_session() {
        Ok(())
    } else {
        ctx.synchronize().map_err(Error::Cuda)
    }
}

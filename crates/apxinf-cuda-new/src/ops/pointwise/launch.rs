use apxinf_core::Result;

use super::contracts::Normalized;
use crate::ffi::abi::{pointwise as abi, status};
use crate::CudaContext;

pub(crate) fn execute(ctx: &CudaContext, normalized: Normalized) -> Result<()> {
    // Direct kernels have no prepared execution object to own their bindings.
    // The graph must still retain every allocation after caller handles drop.
    let storage = std::rc::Rc::new(normalized.storage);
    crate::workspace::retain_resource(&storage);
    crate::workspace::validate_capture_target(
        ctx.device_id(),
        normalized.bindings.stream as usize,
    )?;
    unsafe {
        status::check(abi::apxinf_pointwise_launch(
            ctx.runtime(),
            &normalized.spec,
            &normalized.bindings,
        ))
    }
}

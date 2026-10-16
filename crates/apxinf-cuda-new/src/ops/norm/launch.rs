use apxinf_core::Result;

use super::contracts::Normalized;
use crate::ffi::abi::{norm as abi, status};
use crate::CudaContext;

pub(crate) fn execute(ctx: &CudaContext, normalized: Normalized) -> Result<()> {
    crate::workspace::validate_capture_target(
        ctx.device_id(),
        normalized.bindings.stream as usize,
    )?;
    if crate::workspace::is_capturing() {
        crate::workspace::retain_resource(&std::rc::Rc::new(normalized.storage));
    }
    unsafe {
        status::check(abi::apxinf_norm_launch(
            ctx.runtime(),
            &normalized.spec,
            &normalized.bindings,
        ))
    }
}

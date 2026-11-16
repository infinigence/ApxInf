use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{DType, Error, Result, Tensor};

use crate::ffi::abi::{status, volume as abi};
use crate::{CudaBuffer, CudaContext, CudaStream};

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct ExecutionKey {
    input: usize,
    output: usize,
    shape: [usize; 4],
    device: usize,
    stream: usize,
}
struct Execution {
    input: CudaBuffer,
    output: CudaBuffer,
    stream: Arc<CudaStream>,
}

/// Normalize NHWC RGB bytes as `bf16(f32(x) / 127.5 - 1)` and pack each
/// 2x2 cell into THWC channels ordered `(channel, x, y)`.
/// The output has shape `[frames, height/2, width/2, 12]`. Storage is retained
/// by the active execution session/Graph; enqueue performs no device allocation.
pub fn rgb_patch2x2(ctx: &CudaContext, input: &CudaBuffer, output: &Tensor) -> Result<()> {
    let invalid =
        || Error::Other("RGB patch2x2 requires device RGB bytes and BF16 [T,H,W,12] output".into());
    let dims = output.shape().dims();
    if dims.len() != 4
        || dims.contains(&0)
        || dims[3] != 12
        || output.dtype() != DType::BF16
        || input.device() != ctx.device_id()
    {
        return Err(invalid());
    }
    let height = dims[1].checked_mul(2).ok_or_else(invalid)?;
    let width = dims[2].checked_mul(2).ok_or_else(invalid)?;
    let count = dims[0]
        .checked_mul(height)
        .and_then(|v| v.checked_mul(width))
        .and_then(|v| v.checked_mul(3))
        .ok_or_else(invalid)?;
    if count > i64::MAX as usize / 2 || input.len() != count {
        return Err(invalid());
    }
    let out = CudaBuffer::from_tensor(output).map_err(Error::Cuda)?;
    if out.device() != ctx.device_id() || out.len() != count * 2 {
        return Err(invalid());
    }
    crate::workspace::validate_capture_target(ctx.device_id(), ctx.stream().handle() as usize)?;
    let key = ExecutionKey {
        input: input.ptr() as usize,
        output: out.ptr() as usize,
        shape: [dims[0], dims[1], dims[2], dims[3]],
        device: ctx.device_id(),
        stream: ctx.stream().handle() as usize,
    };
    let execution =
        if let Some(execution) = crate::workspace::lookup_execution::<_, Execution>(&key) {
            execution
        } else {
            if crate::workspace::has_active_session()
                && !crate::workspace::may_prepare_native_resources()
            {
                return Err(Error::Other(
                    "RGB patch2x2 session miss; prepare the same bindings first".into(),
                ));
            }
            let execution = Rc::new(Execution {
                input: input.retained_for_execution(),
                output: out.retained_for_execution(),
                stream: ctx.shared_stream(),
            });
            crate::workspace::store_execution(key, execution.clone());
            execution
        };
    crate::workspace::use_execution(&execution)?;
    unsafe {
        status::check(abi::apxinf_rgb_patch2x2(
            ctx.runtime(),
            execution.input.ptr(),
            execution.output.ptr(),
            dims[0] as i64,
            height as i64,
            width as i64,
            execution.stream.handle(),
        ))
    }
}

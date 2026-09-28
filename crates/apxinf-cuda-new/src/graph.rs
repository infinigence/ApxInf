//! Safe public CUDA Graph capture boundary.

use crate::context::CudaContext;
use crate::ffi;
use apxinf_core::{Error, Result};
use std::any::Any;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::Arc;

pub struct CapturedGraph {
    exec: ffi::cudaGraphExec_t,
    graph: ffi::cudaGraph_t,
    stream: Arc<crate::CudaStream>,
    _resources: Vec<Rc<dyn Any>>,
}

impl CapturedGraph {
    pub fn replay(&self) -> Result<()> {
        self.stream
            .with_current_device(|| unsafe {
                ffi::check_cuda(ffi::cudaGraphLaunch(self.exec, self.stream.handle()))
            })
            .map_err(Error::Cuda)
    }
}

impl apxinf_core::Graph for CapturedGraph {
    fn replay(&self) -> Result<()> {
        CapturedGraph::replay(self)
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        let _ = self.stream.with_current_device(|| unsafe {
            ffi::check_cuda(ffi::cudaGraphExecDestroy(self.exec))?;
            ffi::check_cuda(ffi::cudaGraphDestroy(self.graph))
        });
    }
}

pub(crate) fn begin(ctx: &CudaContext) -> std::result::Result<(), String> {
    let mode = ffi::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal;
    // Keep this device current until `end`; switching devices during capture
    // can invalidate a thread-local capture.
    ctx.stream().set_current_device()?;
    unsafe {
        ffi::check_cuda(ffi::cudaStreamBeginCapture(ctx.stream().handle(), mode))?;
    }
    crate::workspace::begin_capture_retention(ctx.device_id(), ctx.stream().handle() as usize);
    Ok(())
}

pub(crate) fn end(ctx: &CudaContext) -> std::result::Result<CapturedGraph, String> {
    let device_status = ctx.stream().set_current_device();
    let stream = ctx.stream().handle();
    let mut graph: ffi::cudaGraph_t = std::ptr::null_mut();
    let end_status = device_status
        .and_then(|_| unsafe { ffi::check_cuda(ffi::cudaStreamEndCapture(stream, &mut graph)) });
    let retained = crate::workspace::end_capture_retention();
    if end_status.is_err() {
        clear_capture_error();
    }
    end_status?;
    let mut exec: ffi::cudaGraphExec_t = std::ptr::null_mut();
    let status = unsafe {
        ffi::cudaGraphInstantiate(
            &mut exec,
            graph,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        )
    };
    if let Err(error) = ffi::check_cuda(status) {
        unsafe {
            let _ = ffi::cudaGraphDestroy(graph);
        }
        return Err(error);
    }
    Ok(CapturedGraph {
        exec,
        graph,
        stream: ctx.shared_stream(),
        _resources: retained,
    })
}

fn clear_capture_error() {
    unsafe {
        if matches!(ffi::cudaPeekAtLastError(), 900 | 901) {
            let _ = ffi::cudaGetLastError();
        }
    }
}

/// Capture asynchronous work submitted by `operation` into a CUDA Graph.
/// Prepared executions used by the closure are retained by the graph.
pub fn capture(ctx: &CudaContext, operation: impl FnOnce() -> Result<()>) -> Result<CapturedGraph> {
    if crate::workspace::is_capturing() {
        return Err(Error::Other(
            "nested CUDA Graph capture is not supported".into(),
        ));
    }
    begin(ctx).map_err(Error::Cuda)?;
    let operation_result = catch_unwind(AssertUnwindSafe(operation));
    let graph_result = end(ctx).map_err(Error::Cuda);
    match operation_result {
        Ok(Ok(())) => graph_result,
        Ok(Err(error)) => {
            drop(graph_result);
            Err(error)
        }
        Err(payload) => {
            drop(graph_result);
            resume_unwind(payload)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_eager_and_capture_recover(ctx: &CudaContext) {
        assert!(!crate::workspace::is_capturing());
        assert_eq!(unsafe { ffi::cudaPeekAtLastError() }, ffi::CUDA_SUCCESS);
        let buffer = crate::CudaBuffer::alloc(16, ctx.device_id()).unwrap();
        let fill = |value| unsafe {
            ffi::check_cuda(ffi::cudaMemsetAsync(
                buffer.ptr(),
                value,
                buffer.len(),
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        };
        fill(0x5a).unwrap();
        ctx.synchronize().unwrap();
        let mut actual = [0; 16];
        buffer.copy_to_host(&mut actual).unwrap();
        assert_eq!(actual, [0x5a; 16]);
        let graph = capture(ctx, || fill(0x2a)).unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        buffer.copy_to_host(&mut actual).unwrap();
        assert_eq!(actual, [0x2a; 16]);
    }

    #[test]
    fn failed_capture_preserves_operation_error_and_recovers() {
        let ctx = CudaContext::new(0).unwrap();
        let result = capture(&ctx, || ctx.synchronize().map_err(Error::Cuda));
        let error = result.err().expect("capture synchronization must fail");
        assert!(error.to_string().contains("CUDA error 900:"), "{error}");
        assert_eager_and_capture_recover(&ctx);
    }

    #[test]
    fn invalidated_capture_preserves_end_error_and_recovers() {
        let ctx = CudaContext::new(0).unwrap();
        let result = capture(&ctx, || {
            assert!(ctx.synchronize().is_err());
            Ok(())
        });
        let error = result.err().expect("invalidated capture must fail to end");
        assert!(error.to_string().contains("CUDA error 901:"), "{error}");
        assert_eager_and_capture_recover(&ctx);
    }

    #[test]
    fn invalidated_capture_preserves_panic_and_recovers() {
        let ctx = CudaContext::new(0).unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let _ = capture(&ctx, || {
                assert!(ctx.synchronize().is_err());
                panic!("capture recovery fixture");
            });
        }));
        let payload = result.err().expect("capture must resume the original panic");
        assert_eq!(
            payload.downcast_ref::<&str>(),
            Some(&"capture recovery fixture")
        );
        assert_eager_and_capture_recover(&ctx);
    }

    #[test]
    fn capture_cleanup_preserves_unrelated_cuda_errors() {
        let _ctx = CudaContext::new(0).unwrap();
        let error = unsafe { ffi::cudaSetDevice(-1) };
        assert_ne!(error, ffi::CUDA_SUCCESS);
        assert!(!matches!(error, 900 | 901));
        assert_eq!(unsafe { ffi::cudaPeekAtLastError() }, error);
        clear_capture_error();
        assert_eq!(unsafe { ffi::cudaGetLastError() }, error);
    }
}

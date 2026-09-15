//! Safe public CUDA Graph capture boundary.

use crate::context::CudaContext;
use crate::ffi;
use apxinf_core::{Error, Result};
use std::any::Any;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone, Copy)]
enum CaptureMode {
    ThreadLocal,
}

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

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        let _ = self.stream.with_current_device(|| unsafe {
            ffi::check_cuda(ffi::cudaGraphExecDestroy(self.exec))?;
            ffi::check_cuda(ffi::cudaGraphDestroy(self.graph))
        });
    }
}

fn begin(ctx: &CudaContext, mode: CaptureMode) -> std::result::Result<(), String> {
    let mode = match mode {
        CaptureMode::ThreadLocal => ffi::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
    };
    // Keep this device current until `end`; switching devices during capture
    // can invalidate a thread-local capture.
    ctx.stream().set_current_device()?;
    unsafe {
        ffi::check_cuda(ffi::cudaStreamBeginCapture(ctx.stream().handle(), mode))?;
    }
    crate::workspace::begin_capture_retention(ctx.device_id(), ctx.stream().handle() as usize);
    Ok(())
}

fn end(ctx: &CudaContext) -> std::result::Result<CapturedGraph, String> {
    let device_status = ctx.stream().set_current_device();
    let stream = ctx.stream().handle();
    let mut graph: ffi::cudaGraph_t = std::ptr::null_mut();
    let end_status = device_status
        .and_then(|_| unsafe { ffi::check_cuda(ffi::cudaStreamEndCapture(stream, &mut graph)) });
    let retained = crate::workspace::end_capture_retention();
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

/// Capture asynchronous work submitted by `operation` into a CUDA Graph.
/// Prepared executions used by the closure are retained by the graph.
pub fn capture(ctx: &CudaContext, operation: impl FnOnce() -> Result<()>) -> Result<CapturedGraph> {
    if crate::workspace::is_capturing() {
        return Err(Error::Other(
            "nested CUDA Graph capture is not supported".into(),
        ));
    }
    begin(ctx, CaptureMode::ThreadLocal).map_err(Error::Cuda)?;
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

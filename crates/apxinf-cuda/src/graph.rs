//! Private CUDA Graph ABI and handle lifetime boundary.

use crate::context::CudaContext;
use crate::ffi;

#[cfg(test)]
use crate::buffer::CudaBuffer;

#[derive(Clone, Copy)]
pub(crate) enum CaptureMode {
    ThreadLocal,
    Relaxed,
}

pub(crate) struct CapturedGraph {
    exec: ffi::cudaGraphExec_t,
    graph: ffi::cudaGraph_t,
    stream: ffi::cudaStream_t,
}

impl CapturedGraph {
    pub(crate) fn replay(&self) -> Result<(), String> {
        unsafe { ffi::check_cuda(ffi::cudaGraphLaunch(self.exec, self.stream)) }
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        unsafe {
            let _ = ffi::cudaGraphExecDestroy(self.exec);
            let _ = ffi::cudaGraphDestroy(self.graph);
        }
    }
}

pub(crate) fn begin(ctx: &CudaContext, mode: CaptureMode) -> Result<(), String> {
    let mode = match mode {
        CaptureMode::ThreadLocal => ffi::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
        CaptureMode::Relaxed => ffi::cudaStreamCaptureMode::cudaStreamCaptureModeRelaxed,
    };
    unsafe { ffi::check_cuda(ffi::cudaStreamBeginCapture(ctx.stream().handle(), mode)) }
}

pub(crate) fn end(ctx: &CudaContext) -> Result<CapturedGraph, String> {
    let stream = ctx.stream().handle();
    let mut graph: ffi::cudaGraph_t = std::ptr::null_mut();
    unsafe {
        let status = ffi::cudaStreamEndCapture(stream, &mut graph);
        if status != ffi::CUDA_SUCCESS {
            clear_capture_error();
            ffi::check_cuda(status)?;
        }
    }
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
        stream,
    })
}

/// Discard a failed or unwinding capture without instantiating its graph.
/// EndCapture releases the stream, but leaves CUDA's thread-local last error
/// populated. Native kernel launch checks must not inherit that capture error.
pub(crate) fn abort(ctx: &CudaContext) {
    let mut graph = std::ptr::null_mut();
    unsafe {
        let _ = ffi::cudaStreamEndCapture(ctx.stream().handle(), &mut graph);
        if !graph.is_null() {
            let _ = ffi::cudaGraphDestroy(graph);
        }
    }
    clear_capture_error();
}

fn clear_capture_error() {
    // Only consume Unsupported/Invalidated from the capture we just ended.
    // Leave unrelated asynchronous/device failures visible to the next check.
    unsafe {
        if matches!(ffi::cudaPeekAtLastError(), 900 | 901) {
            let _ = ffi::cudaGetLastError();
        }
    }
}

#[cfg(test)]
pub(crate) fn captured_memset(
    ctx: &CudaContext,
    buffer: &CudaBuffer,
    value: u8,
) -> Result<(), String> {
    unsafe {
        ffi::check_cuda(ffi::cudaMemsetAsync(
            buffer.ptr(),
            i32::from(value),
            buffer.len(),
            ctx.stream().handle(),
        ))
    }
}

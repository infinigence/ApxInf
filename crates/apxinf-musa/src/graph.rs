//! Native graph capture over the context's own stream.
//!
//! Capture records a traversal; it does not warm one up. The body it records
//! must already have run once on this same stream, through
//! [`crate::workspace::ExecutionSession::prepare`]. That is not an optimisation:
//! the vendor adapters register their per-stream handles and per-shape metadata
//! during that pass, and a shape first seen inside capture makes the library
//! allocate — which the runtime refuses with `900`, and which
//! [`crate::workspace::output_buffer`] avoids by taking every output from the
//! session arena instead of the driver.
//!
//! The stream is the context's own, so warm-up and capture cannot split: the
//! prepare pass runs on [`MusaContext::stream`] and so does the capture that
//! follows it.
//!
//! A capture that fails takes its stream with it: the runtime leaves it
//! `INVALIDATED`, where launches are refused and even device-wide
//! synchronisation fails. [`capture`] therefore hands the context a fresh
//! stream on the way out, so a caller that falls back to eager — which is what
//! `PreferGraph` asks for — has a stream to fall back onto.

use std::any::Any;
use std::ffi::c_void;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Error, Graph, Result};

use crate::context::MusaContext;
use crate::ffi;
use crate::workspace;

/// A captured traversal, with the prepared resources it used kept alive.
pub struct CapturedGraph {
    executable: *mut c_void,
    graph: *mut c_void,
    context: Arc<MusaContext>,
    _resources: Vec<Rc<dyn Any>>,
}

impl Graph for CapturedGraph {
    fn replay(&self) -> Result<()> {
        unsafe {
            ffi::check_runtime(ffi::musa::musaGraphLaunch(
                self.executable,
                self.context.stream().handle(),
            ))
        }
        .map_err(Error::Other)
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        // `musaGraphExecDestroy` does not wait for an in-flight replay, so the
        // device has to be idle before the executable goes. If that cannot be
        // established the executable is deliberately left alone: destroying it
        // under a pending launch frees storage the device is still reading,
        // while leaking one at teardown costs a few kilobytes.
        if let Err(error) = self.context.synchronize() {
            eprintln!(
                "MUSA graph: synchronize failed before teardown ({error}); leaving the \
                 executable allocated rather than freeing it under a possible replay"
            );
            return;
        }
        unsafe {
            let exec = ffi::musa::musaGraphExecDestroy(self.executable);
            let graph = ffi::musa::musaGraphDestroy(self.graph);
            if exec != ffi::musa::MUSA_SUCCESS {
                eprintln!(
                    "MUSA graph: executable destroy failed ({})",
                    ffi::runtime_error(exec)
                );
            }
            if graph != ffi::musa::MUSA_SUCCESS {
                eprintln!(
                    "MUSA graph: graph destroy failed ({})",
                    ffi::runtime_error(graph)
                );
            }
            // A failed destroy parks a sticky error that the next operator would
            // report as its own; drain it here, as every other runtime call in
            // this crate does.
            if exec != ffi::musa::MUSA_SUCCESS || graph != ffi::musa::MUSA_SUCCESS {
                let _ = ffi::musa::musaGetLastError();
            }
        }
    }
}

/// Capture the asynchronous work `operation` submits into a replayable graph.
///
/// `operation` runs exactly once. Callers that need a warm-up — every model
/// does, for the reason in the module comment — get it from the prepare pass.
pub fn capture(
    context: Arc<MusaContext>,
    operation: impl FnOnce() -> Result<()>,
) -> Result<CapturedGraph> {
    if workspace::is_capturing() {
        return Err(Error::Other(
            "nested MUSA graph capture is not supported".into(),
        ));
    }
    context.synchronize()?;
    let device = context.device_id();
    let stream = context.stream().handle();
    // A thread-local capture belongs to one device. Rebinding before opening it
    // keeps a context created on another thread from capturing into the wrong
    // one.
    unsafe { ffi::check_runtime(ffi::musa::musaSetDevice(device as i32)) }.map_err(Error::Other)?;
    unsafe { ffi::check_runtime(ffi::custom::apxinf_musa_capture_anchor(stream)) }
        .map_err(Error::Other)?;
    unsafe {
        ffi::check_runtime(ffi::musa::musaStreamBeginCapture(
            stream,
            ffi::musa::capture_mode::THREAD_LOCAL,
        ))
    }
    .map_err(Error::Other)?;
    workspace::begin_capture_retention();

    let operation_result = catch_unwind(AssertUnwindSafe(operation));
    let mut graph = std::ptr::null_mut();
    let closed = ffi::check_runtime(unsafe { ffi::musa::musaStreamEndCapture(stream, &mut graph) });
    let retained = workspace::end_capture_retention();
    // Leaving capture is where an invalidated capture is discovered, and the
    // stream does not survive it: it stays `INVALIDATED`, refusing launches and
    // failing `musaDeviceSynchronize` for as long as it exists. No status code
    // clears that, so the context gets a replacement stream here -- which is
    // what makes the eager fallback `PreferGraph` promises reachable at all.
    // Abandoning the record costs nothing: the synchronise this function opens
    // with ran, so none of the captured work ever executed, and doing it before
    // the unwind below leaves a panicking operator with a usable context too.
    let closed = match closed {
        Ok(()) => Ok(()),
        Err(capture) => Err(match context.replace_stream() {
            Ok(()) => format!("{capture}; abandoned that stream for a fresh one"),
            Err(replacement) => format!(
                "{capture}; and the replacement stream could not be created, so this context \
                 is left on the invalidated one: {replacement}"
            ),
        }),
    };

    let operator_result = match operation_result {
        Ok(result) => result,
        Err(payload) => {
            if !graph.is_null() {
                unsafe {
                    let _ = ffi::musa::musaGraphDestroy(graph);
                }
            }
            resume_unwind(payload)
        }
    };
    // A failure inside the region and a failure to leave it are different
    // diagnoses -- 901 is a capture the runtime invalidated, which is not the
    // operator's fault -- so both are reported rather than the second vanishing
    // behind the first.
    let failure = match (operator_result, closed) {
        (Ok(()), Ok(())) => None,
        (Ok(()), Err(capture)) => Some(format!("MUSA stream capture failed: {capture}")),
        (Err(operator), Ok(())) => Some(operator.to_string()),
        (Err(operator), Err(capture)) => Some(format!(
            "{operator} (and the stream then failed to leave capture: {capture})"
        )),
    };
    if let Some(failure) = failure {
        if !graph.is_null() {
            unsafe {
                let _ = ffi::musa::musaGraphDestroy(graph);
            }
        }
        return Err(Error::Other(failure));
    }
    let mut executable = std::ptr::null_mut();
    if let Err(error) = unsafe {
        ffi::check_runtime(ffi::musa::musaGraphInstantiate(&mut executable, graph, 0))
    } {
        unsafe {
            let _ = ffi::musa::musaGraphDestroy(graph);
        }
        return Err(Error::Other(error));
    }
    Ok(CapturedGraph {
        executable,
        graph,
        context,
        _resources: retained,
    })
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    use super::*;
    use crate::buffer::MusaBuffer;

    fn buffer(ctx: &MusaContext, value: f32) -> MusaBuffer {
        let buffer = MusaBuffer::alloc_on(ctx, 128).unwrap();
        let word = ((value.to_bits() >> 16) as u16).to_le_bytes();
        buffer.copy_from_host(&word.repeat(64)).unwrap();
        buffer
    }
    fn launch(ctx: &MusaContext, a: &MusaBuffer, b: &MusaBuffer, out: &MusaBuffer) -> Result<()> {
        unsafe {
            ffi::check_runtime(ffi::custom::apxinf_musa_euler_update_bf16(
                a.ptr(),
                b.ptr(),
                out.ptr(),
                64,
                1.,
                ctx.stream().handle(),
            ))
        }
        .map_err(Error::Other)
    }
    fn read(ctx: &MusaContext, out: &MusaBuffer) -> Vec<u8> {
        ctx.synchronize().unwrap();
        let mut data = vec![0; 128];
        out.copy_to_host(&mut data).unwrap();
        data
    }

    #[test]
    #[ignore = "requires a physical MUSA device"]
    fn physical_graph_replays_updated_inputs_and_restores_stream() {
        let ctx = Arc::new(MusaContext::new(0).unwrap());
        let original = ctx.stream().handle();
        let a = buffer(&ctx, 2.);
        let b = buffer(&ctx, 3.);
        let out = buffer(&ctx, 0.);
        // The traversal capture records must have run once on this stream; the
        // graph owns no host code to repeat it.
        launch(&ctx, &a, &b, &out).unwrap();
        let graph = capture(Arc::clone(&ctx), || launch(&ctx, &a, &b, &out)).unwrap();
        assert_eq!(ctx.stream().handle(), original);
        for value in [2f32, 7., -1.] {
            a.copy_from_host(&((value.to_bits() >> 16) as u16).to_le_bytes().repeat(64))
                .unwrap();
            graph.replay().unwrap();
            assert_eq!(
                read(&ctx, &out),
                (((value + 3.).to_bits() >> 16) as u16)
                    .to_le_bytes()
                    .repeat(64)
            );
        }
        drop(graph);
        launch(&ctx, &a, &b, &out).unwrap();
        assert_eq!(ctx.stream().handle(), original);
        assert_eq!(read(&ctx, &out), (0x4000u16).to_le_bytes().repeat(64));
    }

    #[test]
    #[ignore = "requires a physical MUSA device"]
    fn physical_failed_capture_recovers_eager_and_recaptures() {
        eprintln!("graph failure test: creating context");
        let ctx = Arc::new(MusaContext::new(0).unwrap());
        let original = ctx.stream().handle();
        let a = buffer(&ctx, 2.);
        let b = buffer(&ctx, 3.);
        let out = buffer(&ctx, 0.);
        eprintln!("graph failure test: buffers ready");
        let failed = capture(Arc::clone(&ctx), || {
            if workspace::is_capturing() {
                return Err(Error::Other(
                    "forced operator failure during capture".into(),
                ));
            }
            launch(&ctx, &a, &b, &out)
        });
        eprintln!("graph failure test: capture returned");
        assert!(failed.is_err());
        assert_eq!(ctx.stream().handle(), original);
        launch(&ctx, &a, &b, &out).unwrap();
        assert_eq!(read(&ctx, &out), (0x40a0u16).to_le_bytes().repeat(64));
        let graph = capture(Arc::clone(&ctx), || launch(&ctx, &a, &b, &out)).unwrap();
        graph.replay().unwrap();
        assert_eq!(read(&ctx, &out), (0x40a0u16).to_le_bytes().repeat(64));
    }

    /// An invalidated capture leaves a stream nothing runs on, so the context
    /// has to come back with a different one.
    ///
    /// This is the device half of `native_preparation_failure_falls_back_and_retained_plan_runs`:
    /// `PreferGraph` can only fall back to eager if eager still has a stream to
    /// run on, and a refused synchronise inside the region is the cheapest way
    /// to produce a real invalidation.
    #[test]
    #[ignore = "requires a physical MUSA device"]
    fn physical_invalidated_capture_replaces_the_stream() {
        let ctx = Arc::new(MusaContext::new(0).unwrap());
        let original = ctx.stream().handle();
        let a = buffer(&ctx, 2.);
        let b = buffer(&ctx, 3.);
        let out = buffer(&ctx, 0.);
        launch(&ctx, &a, &b, &out).unwrap();

        let failed = match capture(Arc::clone(&ctx), || {
            assert!(
                ctx.synchronize().is_err(),
                "a synchronise inside capture must be refused"
            );
            Ok(())
        }) {
            Ok(_) => panic!("an invalidated capture must fail"),
            Err(error) => error.to_string(),
        };
        assert!(
            failed.contains("abandoned that stream"),
            "the diagnosis must say what became of the stream: {failed}"
        );
        assert_ne!(
            ctx.stream().handle(),
            original,
            "the invalidated stream is still the context's"
        );

        // The replacement works: eager work runs on it, and a capture after it
        // succeeds rather than inheriting the invalidation.
        launch(&ctx, &a, &b, &out).unwrap();
        assert_eq!(read(&ctx, &out), (0x40a0u16).to_le_bytes().repeat(64));
        let graph = capture(Arc::clone(&ctx), || launch(&ctx, &a, &b, &out)).unwrap();
        graph.replay().unwrap();
        assert_eq!(read(&ctx, &out), (0x40a0u16).to_le_bytes().repeat(64));
    }
}

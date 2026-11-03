//! ACL context and device initialization.
//!
//! Wraps `aclInit` / `aclrtSetDevice` / `aclrtCreateContext` / `aclrtCreateStream`
//! into a single owned handle. The first `AscendContext::new` call initializes
//! ACL; the last `AscendContext` drop finalizes it. Process-wide init/finalize
//! is reference-counted via a static `AtomicUsize`.

use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::ffi::acl::{self, aclrtContext, aclrtStream};
use crate::ffi::aclnn;

static INIT_COUNT: AtomicUsize = AtomicUsize::new(0);

/// Global ACL context pointer for auto-recovery from ACL_ERROR_RT_CONTEXT_NULL.
/// Set by `AscendContext::new`, used by `AscendBuffer::malloc` as a fallback.
struct SendableContext(aclrtContext);
unsafe impl Send for SendableContext {}

static GLOBAL_CONTEXT: std::sync::Mutex<Option<SendableContext>> = std::sync::Mutex::new(None);

/// Re-set the current thread's ACL context from the global fallback.
/// Called when `aclrtMalloc` returns 107002 (CONTEXT_NULL).
pub fn restore_context() -> bool {
    let guard = GLOBAL_CONTEXT.lock().unwrap();
    if let Some(ref ctx) = *guard {
        unsafe { acl::aclrtSetCurrentContext(ctx.0) };
        true
    } else {
        false
    }
}

/// Owner of an ACL device context and its default stream.
///
/// Construction runs the full init sequence:
/// `aclInit` → `aclrtSetDevice` → `aclrtCreateContext` → `aclrtSetCurrentContext` → `aclrtCreateStream`.
///
/// Drop runs the reverse: `aclrtDestroyStream` → `aclrtDestroyContext` →
/// `aclrtResetDevice` → (if last context) `aclFinalize`.
pub struct AscendContext {
    device_id: i32,
    context: aclrtContext,
    stream: aclrtStream,
}

// SAFETY: ACL contexts and streams are thread-safe after creation. The
// Ascend context is not `Send` by default because the raw pointers are
// `*mut c_void`, but ACL documents that a context may be used from any
// thread after `aclrtSetCurrentContext`.
unsafe impl Send for AscendContext {}

impl AscendContext {
    /// Initialize ACL, select `device_id`, create a context and stream.
    pub fn new(device_id: usize) -> Result<Self, String> {
        // Process-wide init (refcounted).
        let prev = INIT_COUNT.fetch_add(1, Ordering::SeqCst);
        if prev == 0 {
            // ASCEND_OPP_PATH must be set before any aclnn operator can run.
            // If the caller hasn't set it, fall back to the standard CANN path.
            if std::env::var("ASCEND_OPP_PATH").is_err() {
                std::env::set_var(
                    "ASCEND_OPP_PATH",
                    "/usr/local/Ascend/cann-9.1.0/opp",
                );
            }

            let ret = unsafe { acl::aclInit(std::ptr::null()) };
            acl::check(ret, "aclInit")?;

            // aclnnInit loads the operator package; required before any
            // aclnnXxxGetWorkspaceSize call.
            let ret = unsafe { aclnn::aclnnInit(std::ptr::null()) };
            if let Err(e) = aclnn::check(ret, "aclnnInit") {
                unsafe { acl::aclFinalize() };
                INIT_COUNT.fetch_sub(1, Ordering::SeqCst);
                return Err(e);
            }
        }

        let device_id = device_id as i32;

        let ret = unsafe { acl::aclrtSetDevice(device_id) };
        if let Err(e) = acl::check(ret, "aclrtSetDevice") {
            maybe_finalize();
            return Err(e);
        }

        let mut context: aclrtContext = std::ptr::null_mut();
        let ret = unsafe { acl::aclrtCreateContext(&mut context, device_id) };
        if let Err(e) = acl::check(ret, "aclrtCreateContext") {
            unsafe { acl::aclrtResetDevice(device_id) };
            maybe_finalize();
            return Err(e);
        }

        let ret = unsafe { acl::aclrtSetCurrentContext(context) };
        if let Err(e) = acl::check(ret, "aclrtSetCurrentContext") {
            unsafe { acl::aclrtDestroyContext(context) };
            unsafe { acl::aclrtResetDevice(device_id) };
            maybe_finalize();
            return Err(e);
        }

        let mut stream: aclrtStream = std::ptr::null_mut();
        let ret = unsafe { acl::aclrtCreateStream(&mut stream) };
        if let Err(e) = acl::check(ret, "aclrtCreateStream") {
            unsafe { acl::aclrtDestroyContext(context) };
            unsafe { acl::aclrtResetDevice(device_id) };
            maybe_finalize();
            return Err(e);
        }

        // Save context globally for auto-recovery from CONTEXT_NULL errors.
        {
            let mut guard = GLOBAL_CONTEXT.lock().unwrap();
            *guard = Some(SendableContext(context));
        }

        Ok(Self {
            device_id,
            context,
            stream,
        })
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn raw_context(&self) -> aclrtContext {
        self.context
    }

    pub fn raw_stream(&self) -> aclrtStream {
        self.stream
    }

    /// Block until all ops on this context's stream complete.
    pub fn synchronize_stream(&self) -> Result<(), String> {
        let ret = unsafe { acl::aclrtSynchronizeStream(self.stream) };
        acl::check(ret, "aclrtSynchronizeStream")
    }

    /// Block until all ops on the current device complete.
    pub fn synchronize_device(&self) -> Result<(), String> {
        let ret = unsafe { acl::aclrtSynchronizeDevice() };
        acl::check(ret, "aclrtSynchronizeDevice")
    }
}

impl Drop for AscendContext {
    fn drop(&mut self) {
        unsafe {
            acl::aclrtDestroyStream(self.stream);
            acl::aclrtDestroyContext(self.context);
            acl::aclrtResetDevice(self.device_id);
        }
        maybe_finalize();
    }
}

fn maybe_finalize() {
    let prev = INIT_COUNT.fetch_sub(1, Ordering::SeqCst);
    if prev == 1 {
        unsafe { aclnn::aclnnFinalize() };
        unsafe { acl::aclFinalize() };
    }
}

/// Query the number of available NPU devices. Requires ACL to be initialized.
pub fn device_count() -> Result<u32, String> {
    let mut count: u32 = 0;
    let ret = unsafe { acl::aclrtGetDeviceCount(&mut count) };
    acl::check(ret, "aclrtGetDeviceCount")?;
    Ok(count)
}

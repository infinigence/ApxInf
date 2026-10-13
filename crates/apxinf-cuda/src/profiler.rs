//! Safe CUDA profiler capture boundaries for benchmark tooling.

/// Start collection for profilers configured with
/// `--capture-range=cudaProfilerApi`.
pub fn start() -> Result<(), String> {
    unsafe { crate::ffi::check_cuda(crate::ffi::cudaProfilerStart()) }
}

/// Stop collection for profilers configured with
/// `--capture-range=cudaProfilerApi`.
pub fn stop() -> Result<(), String> {
    unsafe { crate::ffi::check_cuda(crate::ffi::cudaProfilerStop()) }
}

/// A pair of CUDA events used to measure a region on the runtime stream.
///
/// The stop event is synchronized only when elapsed time is requested, so
/// benchmark instrumentation does not serialize the measured inference.
pub struct CudaEventTimer {
    start: crate::ffi::cudaEvent_t,
    stop: crate::ffi::cudaEvent_t,
}

impl CudaEventTimer {
    pub fn new() -> apxinf_core::Result<Self> {
        let mut timer = Self {
            start: std::ptr::null_mut(),
            stop: std::ptr::null_mut(),
        };
        unsafe {
            crate::ffi::check_cuda(crate::ffi::cudaEventCreate(&mut timer.start))
                .map_err(apxinf_core::Error::Cuda)?;
            if let Err(error) =
                crate::ffi::check_cuda(crate::ffi::cudaEventCreate(&mut timer.stop))
            {
                let _ = crate::ffi::cudaEventDestroy(timer.start);
                return Err(apxinf_core::Error::Cuda(error));
            }
        }
        Ok(timer)
    }

    pub fn start(&self, context: &crate::context::CudaContext) -> apxinf_core::Result<()> {
        unsafe {
            crate::ffi::check_cuda(crate::ffi::cudaEventRecord(
                self.start,
                context.stream().handle(),
            ))
            .map_err(apxinf_core::Error::Cuda)
        }
    }

    pub fn stop(&self, context: &crate::context::CudaContext) -> apxinf_core::Result<()> {
        unsafe {
            crate::ffi::check_cuda(crate::ffi::cudaEventRecord(
                self.stop,
                context.stream().handle(),
            ))
            .map_err(apxinf_core::Error::Cuda)
        }
    }

    pub fn elapsed_ms(&self) -> apxinf_core::Result<f64> {
        let mut milliseconds = 0.0f32;
        unsafe {
            crate::ffi::check_cuda(crate::ffi::cudaEventSynchronize(self.stop))
                .map_err(apxinf_core::Error::Cuda)?;
            crate::ffi::check_cuda(crate::ffi::cudaEventElapsedTime(
                &mut milliseconds,
                self.start,
                self.stop,
            ))
            .map_err(apxinf_core::Error::Cuda)?;
        }
        Ok(f64::from(milliseconds))
    }
}

impl Drop for CudaEventTimer {
    fn drop(&mut self) {
        unsafe {
            if !self.start.is_null() {
                let _ = crate::ffi::cudaEventDestroy(self.start);
            }
            if !self.stop.is_null() {
                let _ = crate::ffi::cudaEventDestroy(self.stop);
            }
        }
    }
}

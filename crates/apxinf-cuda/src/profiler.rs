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

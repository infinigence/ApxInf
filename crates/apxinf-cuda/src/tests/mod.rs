mod bf16;
mod fp8;
mod operators;
mod sampling;
mod w8a8;

/// Serializes tests that configure large dynamic shared memory on the GPU.
///
/// The FA2 launch path calls `cudaFuncSetAttribute` with
/// `cudaFuncAttributeMaxDynamicSharedMemorySize` for smem requests at or above
/// 48 KiB. Running several of those concurrently on one device makes that call
/// fail with `invalid argument`, which aborts the test process. Production is
/// unaffected: the graph replays these kernels serially. Tests that launch such
/// kernels must hold this guard.
pub(crate) static GPU_SMEM_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Acquires [`GPU_SMEM_GUARD`], ignoring poisoning so that one failing test does
/// not cascade into unrelated failures.
pub(crate) fn gpu_smem_guard() -> std::sync::MutexGuard<'static, ()> {
    GPU_SMEM_GUARD.lock().unwrap_or_else(|e| e.into_inner())
}

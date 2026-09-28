//! Both CUDA crates linked into one binary.
//!
//! `apxinf-cuda` (legacy) and `apxinf-cuda-new` used to share a crate name
//! and could only be built one at a time through a symlink swap. This test
//! is the coexistence contract: one process creates a context through each
//! crate and moves data through each crate's runtime. If their native
//! symbol sets ever collide again, this binary fails to link — the failure
//! mode is a build error, which is exactly where we want it.
//!
//! Both crates are optional dependencies, so this binary can only compile when
//! the `cuda` feature pulls them in — the same gate `qwen_drive_loading.rs` uses.
#![cfg(feature = "cuda")]

#[test]
#[ignore = "requires a CUDA device"]
fn both_cuda_crates_link_and_launch() {
    let host = [1.0f32, 2.0, 3.0, 4.0];
    let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();

    // Legacy crate: context + buffer round-trip through its runtime.
    let legacy_ctx = apxinf_cuda::CudaContext::new(0).expect("legacy context");
    let legacy_buf =
        apxinf_cuda::CudaBuffer::alloc(bytes.len(), legacy_ctx.device_id()).expect("legacy alloc");
    legacy_buf.copy_from_host(&bytes).expect("legacy h2d");
    let mut legacy_back = vec![0u8; bytes.len()];
    legacy_buf.copy_to_host(&mut legacy_back).expect("legacy d2h");
    assert_eq!(bytes, legacy_back, "legacy round-trip");

    // New crate: context + buffer round-trip through its (renamed) runtime.
    // Both extern "C" surfaces are live in this process at this point.
    let new_ctx = apxinf_cuda_new::CudaContext::new(0).expect("cuda-new context");
    let new_buf =
        apxinf_cuda_new::CudaBuffer::alloc(bytes.len(), new_ctx.device_id()).expect("new alloc");
    new_buf.copy_from_host(&bytes).expect("new h2d");
    let mut new_back = vec![0u8; bytes.len()];
    new_buf.copy_to_host(&mut new_back).expect("new d2h");
    assert_eq!(bytes, new_back, "cuda-new round-trip");

    legacy_ctx.synchronize().expect("legacy sync");
    new_ctx.synchronize().expect("cuda-new sync");
}

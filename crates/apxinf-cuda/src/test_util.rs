//! Test helpers for BF16 CUDA kernels.
//!
//! Every kernel gets a unit test that compares its BF16 output against a
//! reference fp32 computation, within a bf16-appropriate tolerance. This
//! module provides the plumbing.
//!
//! Design:
//! - `upload_fp32_as_bf16(data)` — take fp32 host data, downcast to bf16,
//!   upload to GPU, return a GPU tensor.
//! - `download_bf16_as_fp32(&t)` — download a bf16 GPU tensor, upcast to
//!   fp32 on host.
//! - `assert_bf16_close(actual_fp32, expected_fp32, rel_tol)` — compare
//!   with a relative tolerance appropriate for bf16 numerics (default ~1%
//!   relative, ~1e-3 absolute).

use half::bf16;

use apxinf_core::storage::GpuStorageHandle;
use apxinf_core::{DType, Device, Result, Shape, Storage, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;

/// Upload fp32 host data to CUDA as a bf16 device tensor.
pub fn upload_fp32_as_bf16(
    ctx: &CudaContext,
    data: &[f32],
    shape: impl Into<Shape>,
) -> Result<Tensor> {
    let shape = shape.into();
    assert_eq!(
        shape.numel(),
        data.len(),
        "upload_fp32_as_bf16: shape numel {} != data len {}",
        shape.numel(),
        data.len()
    );
    let bf: Vec<bf16> = data.iter().map(|&x| bf16::from_f32(x)).collect();
    let bytes: Vec<u8> = bf.iter().flat_map(|b| b.to_le_bytes()).collect();

    let device_id = ctx.device_id();
    let buf = CudaBuffer::alloc(bytes.len(), device_id).map_err(apxinf_core::Error::Cuda)?;
    buf.copy_from_host(&bytes)
        .map_err(apxinf_core::Error::Cuda)?;

    let handle = GpuStorageHandle {
        ptr: buf.ptr() as usize,
        len: buf.len(),
        _prevent_leak: Some(std::sync::Arc::new(buf)),
    };
    let device = Device::Cuda(device_id);
    Ok(Tensor::from_raw_parts(
        shape,
        DType::BF16,
        device,
        Storage::Gpu { device, handle },
    ))
}

/// Download the exact little-endian BF16 storage bytes from a GPU tensor.
pub fn download_bf16_bytes(tensor: &Tensor) -> Result<Vec<u8>> {
    assert_eq!(
        tensor.dtype(),
        DType::BF16,
        "download_bf16_bytes: expected BF16 tensor"
    );
    let handle = tensor
        .storage()
        .as_gpu()
        .ok_or_else(|| apxinf_core::Error::Other("expected GPU storage".into()))?;

    // Sync then copy the raw bytes back
    unsafe {
        crate::ffi::check_cuda(crate::ffi::cudaDeviceSynchronize())
            .map_err(apxinf_core::Error::Cuda)?;
    }
    let expected_bytes = tensor
        .numel()
        .checked_mul(DType::BF16.size_in_bytes())
        .expect("BF16 tensor byte size overflow");
    assert_eq!(
        handle.len, expected_bytes,
        "download_bf16_bytes: storage length does not match tensor shape"
    );
    let mut host_bytes = vec![0u8; expected_bytes];
    unsafe {
        crate::ffi::check_cuda(crate::ffi::cudaMemcpy(
            host_bytes.as_mut_ptr() as *mut std::ffi::c_void,
            handle.ptr as *const std::ffi::c_void,
            expected_bytes,
            crate::ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        ))
        .map_err(apxinf_core::Error::Cuda)?;
    }

    Ok(host_bytes)
}

/// Download a bf16 GPU tensor and upcast to fp32 on host.
pub fn download_bf16_as_fp32(tensor: &Tensor) -> Result<Vec<f32>> {
    let host_bytes = download_bf16_bytes(tensor)?;
    // Interpret bytes as bf16 and upcast.
    let numel = tensor.numel();
    let mut out = Vec::with_capacity(numel);
    for i in 0..numel {
        let lo = host_bytes[i * 2];
        let hi = host_bytes[i * 2 + 1];
        out.push(bf16::from_le_bytes([lo, hi]).to_f32());
    }
    Ok(out)
}

/// Assert `actual` matches `expected` within bf16-appropriate tolerance.
///
/// bf16 has an 8-bit mantissa (~2.3e-3 unit-in-last-place precision). We
/// use `abs_tol + rel_tol * |expected|` — a common combined tolerance:
/// - `abs_tol` catches small-magnitude entries where relative tolerance
///   is meaningless.
/// - `rel_tol` scales for large-magnitude entries.
///
/// Recommended defaults:
///   - Elementwise ops (silu, mul, add): `abs_tol=1e-3, rel_tol=1e-2`.
///   - Reductions (rms_norm, softmax): `abs_tol=1e-2, rel_tol=2e-2` —
///     accumulated bf16 error.
pub fn assert_bf16_close(actual: &[f32], expected: &[f32], abs_tol: f32, rel_tol: f32) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "assert_bf16_close: len mismatch {} != {}",
        actual.len(),
        expected.len()
    );

    let mut worst_abs = 0.0f32;
    let mut worst_idx = 0usize;
    let mut worst_actual = 0.0f32;
    let mut worst_expected = 0.0f32;
    let mut n_bad = 0usize;

    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            a.is_finite() && e.is_finite(),
            "assert_bf16_close: non-finite value at index {i}: actual={a}, expected={e}"
        );
        let abs_err = (a - e).abs();
        let tol = abs_tol + rel_tol * e.abs();
        if abs_err > tol {
            n_bad += 1;
            if abs_err > worst_abs {
                worst_abs = abs_err;
                worst_idx = i;
                worst_actual = a;
                worst_expected = e;
            }
        }
    }

    assert_eq!(
        n_bad,
        0,
        "assert_bf16_close: {n_bad}/{} elements outside tolerance \
         (abs={abs_tol}, rel={rel_tol}); \
         worst @ idx {worst_idx}: actual={worst_actual}, expected={worst_expected}, \
         abs_err={worst_abs}",
        actual.len(),
    );
}

/// Convenience: default tolerance for elementwise ops.
pub fn assert_bf16_close_elementwise(actual: &[f32], expected: &[f32]) {
    assert_bf16_close(actual, expected, 1e-3, 1e-2);
}

/// Convenience: default tolerance for reductions.
pub fn assert_bf16_close_reduction(actual: &[f32], expected: &[f32]) {
    assert_bf16_close(actual, expected, 1e-2, 2e-2);
}

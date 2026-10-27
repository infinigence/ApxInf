//! CPU math operations with optional BLAS acceleration.
//!
//! When a BLAS feature (`openblas`) is enabled at compile time,
//! matrix multiplication is dispatched to the vendor BLAS library. Otherwise a
//! naive Rust fallback is used.

/// Row-major general matrix multiply:  C = alpha * A * B + beta * C
///
/// - A: [M, K] row-major
/// - B: [K, N] row-major
/// - C: [M, N] row-major (output, must be pre-allocated)
///
/// This is the single-batch variant. Callers handle batching externally.
pub(crate) fn sgemm(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    assert!(a.len() >= m.checked_mul(k).expect("GEMM size overflow"));
    assert!(b.len() >= k.checked_mul(n).expect("GEMM size overflow"));
    assert!(c.len() >= m.checked_mul(n).expect("GEMM size overflow"));
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        c[..m * n].fill(0.0);
        return;
    }
    #[cfg(feature = "openblas")]
    {
        sgemm_openblas(m, k, n, a, b, c);
    }
    #[cfg(not(feature = "openblas"))]
    {
        sgemm_naive(m, k, n, a, b, c);
    }
}

// ── Naive fallback ──────────────────────────────────────────────────

#[cfg(not(feature = "openblas"))]
fn sgemm_naive(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

// ── OpenBLAS ────────────────────────────────────────────────────────

#[cfg(feature = "openblas")]
fn sgemm_openblas(m: usize, k: usize, n: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    use cblas::{sgemm as cblas_sgemm, Layout, Transpose};

    let m = i32::try_from(m).expect("GEMM M exceeds CBLAS integer range");
    let k = i32::try_from(k).expect("GEMM K exceeds CBLAS integer range");
    let n = i32::try_from(n).expect("GEMM N exceeds CBLAS integer range");
    // The checked lengths and row strides above cover every CBLAS access.
    unsafe {
        cblas_sgemm(
            Layout::RowMajor,
            Transpose::None,
            Transpose::None,
            m,
            n,
            k,
            1.0,
            a,
            k,
            b,
            n,
            0.0,
            c,
            n,
        );
    }
}

use apxinf_core::{DType, Device, Error, Result, Tensor};

pub(crate) fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    for tensor in [a, b] {
        if tensor.device() != Device::Cpu {
            return Err(Error::UnsupportedDevice(tensor.device()));
        }
    }
    if a.dtype() != b.dtype() {
        return Err(Error::DTypeMismatch {
            expected: b.dtype(),
            got: a.dtype(),
        });
    }
    let out_shape = a.shape().matmul_shape(b.shape())?;
    if a.shape().dims()[..a.ndim() - 2] != b.shape().dims()[..b.ndim() - 2] {
        return Err(Error::Other(
            "CPU matmul requires matching batch dimensions".into(),
        ));
    }

    let a_data = if a.dtype() == DType::F32 && a.numel() == 0 {
        std::borrow::Cow::Borrowed(&[][..])
    } else if a.dtype() == DType::F32 {
        std::borrow::Cow::Borrowed(a.as_f32()?)
    } else {
        std::borrow::Cow::Owned(a.to_f32_vec()?)
    };
    let b_data = if b.dtype() == DType::F32 && b.numel() == 0 {
        std::borrow::Cow::Borrowed(&[][..])
    } else if b.dtype() == DType::F32 {
        std::borrow::Cow::Borrowed(b.as_f32()?)
    } else {
        std::borrow::Cow::Owned(b.to_f32_vec()?)
    };

    let m = a.shape().dims()[a.ndim() - 2];
    let k = a.shape().dims()[a.ndim() - 1];
    let n = b.shape().dims()[b.ndim() - 1];
    let batch: usize = a.shape().dims()[..a.ndim() - 2].iter().product::<usize>();

    let mut out = vec![0.0f32; out_shape.numel()];
    for batch_idx in 0..batch {
        let a_off = batch_idx * m * k;
        let b_off = batch_idx * k * n;
        let o_off = batch_idx * m * n;
        crate::gemm::sgemm(
            m,
            k,
            n,
            &a_data[a_off..],
            &b_data[b_off..],
            &mut out[o_off..],
        );
    }

    Tensor::from_f32(out_shape, &out)
}

#[cfg(test)]
mod tests {
    use crate::CpuBackend;
    use apxinf_core::{Backend, Shape, Tensor};
    #[test]
    fn test_matmul_cpu() {
        // [2, 3] @ [3, 2] = [2, 2]
        let a = Tensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Tensor::from_f32(vec![3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let c = CpuBackend.matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &Shape::new(vec![2, 2]));
        let data = c.as_f32().unwrap();
        // Row 0: 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // Row 1: 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        assert_eq!(data, &[58.0, 64.0, 139.0, 154.0]);
    }
    #[test]
    fn matmul_batches_and_invalid_shapes() {
        let a = Tensor::from_f32(vec![2, 1, 2], &[1., 2., 3., 4.]).unwrap();
        let b = Tensor::from_f32(vec![2, 2, 1], &[5., 6., 7., 8.]).unwrap();
        assert_eq!(
            CpuBackend.matmul(&a, &b).unwrap().as_f32().unwrap(),
            &[17., 53.]
        );
        let b = Tensor::from_f32(vec![2, 1], &[5., 6.]).unwrap();
        assert!(CpuBackend.matmul(&a, &b).is_err());
        let a = Tensor::zeros(vec![2, 0], apxinf_core::DType::F32);
        let b = Tensor::zeros(vec![0, 3], apxinf_core::DType::F32);
        assert_eq!(
            CpuBackend.matmul(&a, &b).unwrap().as_f32().unwrap(),
            &[0.; 6]
        );
    }
}

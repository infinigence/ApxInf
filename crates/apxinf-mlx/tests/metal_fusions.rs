#![cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
use apxinf_core::Result;
use apxinf_mlx::fusions::{PackedResidualRmsNorm, QkNormRope};
use apxinf_mlx::{Array, MlxDType as D, Stream};

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn qk_norm_rope_respects_heads_rotation_and_rejects_bad_geometry() -> Result<()> {
    let s = Stream::metal(0)?;
    let kernel = QkNormRope::new(&s)?;
    let q = Array::from_f32(&s, &[1, 1, 16, 128], &vec![3.; 16 * 128])?.cast(D::BF16)?;
    let k = Array::from_f32(&s, &[1, 1, 8, 128], &vec![4.; 8 * 128])?.cast(D::BF16)?;
    let weights: Vec<f32> = (0..128).map(|i| if i < 64 { 0.5 } else { 2. }).collect();
    let qw = Array::from_f32(&s, &[128], &weights)?;
    let kw = Array::from_f32(&s, &[128], &vec![1.; 128])?;
    let cos = Array::zeros(&s, &[1, 1, 1, 128], D::BF16)?;
    let sin = Array::from_f32(&s, &[128], &vec![1.; 128])?.cast(D::BF16)?;
    let (qo, ko) = kernel.call(&q, &k, &qw, &kw, &cos, &sin)?;
    assert_eq!(qo.shape(), [1, 16, 1, 128]);
    assert_eq!(ko.shape(), [1, 8, 1, 128]);
    for row in qo.to_f32_vec()?.chunks_exact(128) {
        assert_eq!(&row[..64], &[-2.; 64]);
        assert_eq!(&row[64..], &[0.5; 64]);
    }
    for row in ko.to_f32_vec()?.chunks_exact(128) {
        assert_eq!(&row[..64], &[-1.; 64]);
        assert_eq!(&row[64..], &[1.; 64]);
    }
    assert!(kernel
        .call(&q.reshape(&[1, 16, 1, 128])?, &k, &qw, &kw, &cos, &sin)
        .is_err());
    assert!(kernel
        .call(&q, &k, &qw.cast(D::BF16)?, &kw, &cos, &sin)
        .is_err());
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn packed_residual_norm_returns_both_owned_bf16_consumers() -> Result<()> {
    let s = Stream::metal(0)?;
    let kernel = PackedResidualRmsNorm::new(&s, 1e-6)?;
    let x = Array::from_f32(&s, &[1, 1, 2048], &vec![1.; 2048])?.cast(D::BF16)?;
    let weight = Array::from_f32(&s, &[2048], &vec![2.; 2048])?.cast(D::BF16)?;
    let (sum, norm) = kernel.call(&x, &x, &weight)?;
    drop(kernel);
    assert_eq!(sum.shape(), [1, 1, 2048]);
    assert_eq!(norm.shape(), [1, 1, 2048]);
    assert_eq!(sum.to_f32_vec()?, vec![2.; 2048]);
    assert_eq!(norm.to_f32_vec()?, vec![2.; 2048]);
    assert_eq!(x.to_f32_vec()?, vec![1.; 2048]);
    assert!(PackedResidualRmsNorm::new(&s, f32::NAN).is_err());
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn packed_residual_norm_keeps_epsilon_per_kernel_instance() -> Result<()> {
    let s = Stream::metal(0)?;
    let x = Array::from_f32(&s, &[1, 1, 2048], &vec![0.5; 2048])?.cast(D::BF16)?;
    let weight = Array::from_f32(&s, &[2048], &vec![1.; 2048])?.cast(D::BF16)?;
    let small = PackedResidualRmsNorm::new(&s, 1e-6)?;
    let large = PackedResidualRmsNorm::new(&s, 3.)?;
    for (kernel, expected) in [(&small, 1.), (&large, 0.5), (&small, 1.)] {
        let (sum, norm) = kernel.call(&x, &x, &weight)?;
        assert_eq!(sum.to_f32_vec()?, vec![1.; 2048]);
        assert_eq!(norm.to_f32_vec()?, vec![expected; 2048]);
    }
    Ok(())
}

fn data(n: usize, period: usize, scale: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7 + 3) % period) as f32 * scale - (period as f32 * scale / 2.))
        .collect()
}

fn assert_within(actual: &[f32], expected: &[f32], relative: f32, absolute: f32) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= relative * e.abs() + absolute,
            "{i}: {a} vs {e}"
        );
    }
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn q4_transpose_decode_matches_mlx_dequantize_in_both_arithmetics() -> Result<()> {
    use apxinf_mlx::fusions::{Q4DecodeArithmetic, Q4TransposeDecode};
    let s = Stream::metal(0)?;
    let (rows, k) = (96, 128);
    let w = Array::from_f32(&s, &[rows, k], &data(rows * k, 29, 0.0625))?.cast(D::BF16)?;
    let [qw, scales, biases] = w.quantize(64, 4)?;
    for (arithmetic, reference) in [
        (
            Q4DecodeArithmetic::F16,
            qw.dequantize(&scales.cast(D::F16)?, &biases.cast(D::F16)?, 64, 4)?,
        ),
        (
            Q4DecodeArithmetic::Bf16,
            qw.dequantize(&scales, &biases, 64, 4)?.cast(D::F16)?,
        ),
    ] {
        let decode = Q4TransposeDecode::new(&s, arithmetic)?;
        let full = decode.call(&qw, &scales, &biases, 0, rows)?;
        assert_eq!((full.shape(), full.dtype()), (&[k, rows][..], D::F16));
        let expected = reference.transpose(&[1, 0])?.to_f32_vec()?;
        assert_eq!(full.to_f32_vec()?, expected, "{arithmetic:?}");
        let part = decode.call(&qw, &scales, &biases, 32, 64)?;
        let expected = reference
            .slice(&[32, 0], &[96, k])?
            .transpose(&[1, 0])?
            .to_f32_vec()?;
        assert_eq!(part.to_f32_vec()?, expected, "{arithmetic:?} row offset");
        assert!(decode.call(&qw, &scales, &biases, 0, 48).is_err());
        assert!(decode.call(&qw, &scales, &biases, 64, 64).is_err());
        assert!(decode
            .call(&qw, &scales.cast(D::F16)?, &biases, 0, 32)
            .is_err());
    }
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn padded_split_f16_gemm_matches_fp32_reference_rounded_to_bf16() -> Result<()> {
    use apxinf_mlx::fusions::{padded_rows, PadRowsF16, SplitF16Gemm};
    let s = Stream::metal(0)?;
    let (tokens, k, half) = (70, 64, 64);
    let padded = padded_rows(tokens);
    assert_eq!(padded, 128);
    let h = Array::from_f32(&s, &[1, tokens, k], &data(tokens * k, 13, 0.125))?.cast(D::BF16)?;
    let pad = PadRowsF16::new(&s)?;
    let x = pad.call(&h)?;
    assert_eq!((x.shape(), x.dtype()), (&[1, padded, k][..], D::F16));
    let xs = x.to_f32_vec()?;
    assert_eq!(&xs[..tokens * k], &h.to_f32_vec()?[..]);
    assert!(xs[tokens * k..].iter().all(|&v| v == 0.));

    let w = Array::from_f32(&s, &[k, 2 * half], &data(k * 2 * half, 11, 0.03125))?.cast(D::F16)?;
    let (w_lo, w_hi) = (
        w.slice(&[0, 0], &[k, half])?.contiguous()?,
        w.slice(&[0, half], &[k, 2 * half])?.contiguous()?,
    );
    let gemm = SplitF16Gemm::new(&s)?;
    let low = gemm.low(&x, tokens, &w_lo)?;
    let out = gemm.high(&w_hi, &low, &x, tokens)?;
    assert_eq!(
        (out.shape(), out.dtype()),
        (&[1, tokens, 2 * half][..], D::BF16)
    );
    let reference = h
        .cast(D::F32)?
        .matmul(&w.cast(D::F32)?)?
        .cast(D::BF16)?
        .to_f32_vec()?;
    assert_within(&out.to_f32_vec()?, &reference, 1. / 128., 1e-3);
    let low_values = low.to_f32_vec()?;
    let out_values = out.to_f32_vec()?;
    for t in 0..tokens {
        assert_eq!(
            &out_values[t * 2 * half..t * 2 * half + half],
            &low_values[t * half..(t + 1) * half]
        );
    }
    assert!(gemm.low(&x, 64, &w_lo).is_err());
    assert!(gemm.low(&h, tokens, &w_lo).is_err());
    assert!(gemm.high(&w_hi, &out, &x, tokens).is_err());
    assert!(pad.call(&h.cast(D::F16)?).is_err());
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn gated_delta_recurrence_matches_host_reference() -> Result<()> {
    use apxinf_mlx::fusions::GatedDeltaRecurrence;
    let s = Stream::metal(0)?;
    let (len, hk, dk, hv, dv) = (5, 2, 32, 4, 16);
    let bf16 = |shape: &[usize], values: Vec<f32>| -> Result<(Array, Vec<f32>)> {
        let a = Array::from_f32(&s, shape, &values)?.cast(D::BF16)?;
        let v = a.to_f32_vec()?;
        Ok((a, v))
    };
    let (q, qv) = bf16(&[1, len, hk, dk], data(len * hk * dk, 17, 0.02))?;
    let (k, kv) = bf16(&[1, len, hk, dk], data(len * hk * dk, 19, 0.02))?;
    let (v, vv) = bf16(&[1, len, hv, dv], data(len * hv * dv, 23, 0.05))?;
    let (beta, bv) = bf16(
        &[1, len, hv],
        data(len * hv, 7, 0.1).iter().map(|x| x + 0.5).collect(),
    )?;
    let gv: Vec<f32> = (0..len * hv).map(|i| 0.9 - 0.01 * (i % 5) as f32).collect();
    let g = Array::from_f32(&s, &[1, len, hv], &gv)?;
    let sv = data(hv * dv * dk, 31, 0.01);
    let state = Array::from_f32(&s, &[1, hv, dv, dk], &sv)?;
    let (y, next) = GatedDeltaRecurrence::new(&s)?.call(&q, &k, &v, &g, &beta, &state)?;

    let mut st = sv.clone();
    let mut yr = vec![0f32; len * hv * dv];
    for t in 0..len {
        for h in 0..hv {
            let kh = h / (hv / hk);
            let at = |x: &[f32], i| x[(t * hk + kh) * dk + i];
            for r in 0..dv {
                let row = &mut st[(h * dv + r) * dk..(h * dv + r + 1) * dk];
                row.iter_mut().for_each(|x| *x *= gv[t * hv + h]);
                let memory: f32 = (0..dk).map(|i| row[i] * at(&kv, i)).sum();
                let delta = (vv[(t * hv + h) * dv + r] - memory) * bv[t * hv + h];
                (0..dk).for_each(|i| row[i] += at(&kv, i) * delta);
                yr[(t * hv + h) * dv + r] = (0..dk).map(|i| row[i] * at(&qv, i)).sum();
            }
        }
    }
    let rounded = Array::from_f32(&s, &[yr.len()], &yr)?
        .cast(D::BF16)?
        .to_f32_vec()?;
    assert_within(&y.to_f32_vec()?, &rounded, 1. / 64., 1e-4);
    assert_within(&next.to_f32_vec()?, &st, 1e-4, 1e-5);
    let bad = GatedDeltaRecurrence::new(&s)?;
    assert!(bad
        .call(&q, &k, &v, &g.cast(D::BF16)?, &beta, &state)
        .is_err());
    assert!(bad
        .call(
            &q,
            &k,
            &v.reshape(&[1, len, hv * 2, dv / 2])?,
            &g,
            &beta,
            &state
        )
        .is_err());
    Ok(())
}

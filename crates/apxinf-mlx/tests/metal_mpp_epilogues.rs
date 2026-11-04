#![cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
//! Fused GEMM epilogues and LayerNorm modulation against composed MLX
//! operators. Layout-only epilogues must match bit for bit; arithmetic
//! epilogues round differently and are bounded. Needs MPP (Metal 4 tensors).
use apxinf_core::Result;
use apxinf_mlx::fusions::{
    GatedResidualLayerNormModulate, LayerNormModulate, MppBf16GemmNt, MppGemmDecodeQ4Rows,
    MppGemmGatedResidual, MppGemmHeadRmsRope, MppGemmHeadsWithPrefix, MppGemmSwigluPairs,
};
use apxinf_mlx::{Array, MlxDType as D, Stream};

fn normal(s: &Stream, shape: &[usize], seed: u64, scale: f32) -> Result<Array> {
    Array::random_normal(shape, D::F32, &Array::random_key(s, seed)?)?
        .mul(&Array::scalar(s, scale, D::F32)?)?
        .cast(D::BF16)
}

/// `[1, T, K] x [N, K]^T` in FP32, rounded to BF16 once.
fn dense(a: &Array, w: &Array) -> Result<Array> {
    a.cast(D::F32)?
        .matmul(&w.cast(D::F32)?.transpose(&[1, 0])?)?
        .cast(D::BF16)
}

fn assert_close(actual: &Array, expected: &Array, relative: f32, absolute: f32) -> Result<()> {
    assert_eq!(actual.shape(), expected.shape());
    let (a, e) = (actual.to_f32_vec()?, expected.to_f32_vec()?);
    for (i, (a, e)) in a.iter().zip(&e).enumerate() {
        assert!(
            (a - e).abs() <= relative * e.abs() + absolute,
            "{i}: {a} vs {e}"
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock and Metal 4 tensors"]
fn layout_epilogues_match_the_plain_gemm_bitwise() -> Result<()> {
    let s = Stream::metal(0)?;
    let (t, k, heads, text) = (128, 256, 2, 5);
    let n = heads * 128;
    let a = normal(&s, &[1, t, k], 1, 1.0)?;
    let w = normal(&s, &[n, k], 2, 0.05)?;
    let plain = MppBf16GemmNt::new(&s)?.call(&a, &w)?;

    let [q, scales, biases] = normal(&s, &[64, 4096], 3, 0.02)?.quantize(64, 4)?;
    let (out, decoded) = MppGemmDecodeQ4Rows::new(&s)?.call(&a, &w, &q, &scales, &biases)?;
    assert_eq!(out.to_bytes()?, plain.to_bytes()?);
    assert_eq!(
        decoded.to_bytes()?,
        q.dequantize(&scales, &biases, 64, 4)?.to_bytes()?
    );

    let prefix = normal(&s, &[1, heads, text, 128], 4, 1.0)?;
    let heads_out = MppGemmHeadsWithPrefix::new(&s)?.call(&a, &w, &prefix)?;
    let tokens = plain
        .reshape(&[1, t, heads, 128])?
        .transpose(&[0, 2, 1, 3])?;
    let expected = Array::concat(&[&prefix, &tokens], 2)?;
    assert_eq!(heads_out.to_bytes()?, expected.contiguous()?.to_bytes()?);
    assert!(MppGemmHeadsWithPrefix::new(&s)?
        .call(&a, &w, &normal(&s, &[1, 3, text, 128], 4, 1.0)?)
        .is_err());
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock and Metal 4 tensors"]
fn arithmetic_epilogues_match_composed_operators() -> Result<()> {
    let s = Stream::metal(0)?;
    let (t, k, n) = (128, 256, 128);
    let a = normal(&s, &[1, t, k], 5, 1.0)?;
    let w = normal(&s, &[n, k], 6, 0.05)?;
    let one = Array::scalar(&s, 1.0, D::BF16)?;

    // SwiGLU over 32-row gate/up pairs: compare with the de-interleaved halves.
    let pairs = w.reshape(&[n / 64, 2, 32, k])?;
    let half = |i: usize| -> Result<Array> { pairs.slice_axis(1, i, i + 1)?.reshape(&[n / 2, k]) };
    let (gate, up) = (
        dense(&a, &half(0)?)?.cast(D::F32)?,
        dense(&a, &half(1)?)?.cast(D::F32)?,
    );
    let expected = gate.mul(&gate.sigmoid()?)?.mul(&up)?.cast(D::BF16)?;
    let swiglu = MppGemmSwigluPairs::new(&s)?.call(&a, &w)?;
    assert_close(&swiglu, &expected, 2e-2, 2e-2)?;

    // Gated residual through broadcast modulation rows (strided reads).
    let res = normal(&s, &[1, t, n], 7, 1.0)?;
    let modulation = normal(&s, &[1, 1, 2 * n], 8, 1.0)?.broadcast_to(&[1, t, 2 * n])?;
    let gated = MppGemmGatedResidual::new(&s)?.call(&a, &w, &res, &modulation)?;
    let gemm = MppBf16GemmNt::new(&s)?.call(&a, &w)?;
    let expected = res.add(&modulation.slice_axis(2, n, 2 * n)?.tanh()?.mul(&gemm)?)?;
    assert_close(&gated, &expected, 1e-2, 1e-2)?;
    assert!(MppGemmGatedResidual::new(&s)?
        .call(&a, &w, &res, &modulation.slice_axis(2, 0, n)?)
        .is_err());

    // Head RMSNorm + interleaved RoPE, with and without the copied prefix.
    let (heads, text) = (n / 128, 3);
    let norm = normal(&s, &[128], 9, 0.1)?.add(&one)?;
    let angles: Vec<f32> = (0..t * 64)
        .map(|i| (i / 64) as f32 * 0.01 * ((i % 64) as f32 + 1.0))
        .collect();
    let table = |f: fn(f32) -> f32| {
        Array::from_f32(
            &s,
            &[t, 64],
            &angles.iter().map(|&x| f(x)).collect::<Vec<_>>(),
        )
    };
    let (cos, sin) = (table(f32::cos)?, table(f32::sin)?);
    let prefix = normal(&s, &[1, heads, text, 128], 10, 1.0)?;
    let kernel = MppGemmHeadRmsRope::new(&s)?;
    let q = kernel.call(&a, &w, &norm, &cos, &sin, &prefix, false)?;
    let keyed = kernel.call(&a, &w, &norm, &cos, &sin, &prefix, true)?;
    assert_eq!(q.shape(), [1, heads, t, 128]);
    assert_eq!(keyed.shape(), [1, heads, text + t, 128]);
    assert_eq!(
        keyed.slice_axis(2, 0, text)?.contiguous()?.to_bytes()?,
        prefix.to_bytes()?
    );
    assert_eq!(
        keyed
            .slice_axis(2, text, text + t)?
            .contiguous()?
            .to_bytes()?,
        q.to_bytes()?
    );
    let x = gemm.reshape(&[1, t, heads, 128])?.cast(D::F32)?;
    let rms = x
        .mul(&x)?
        .mean(&[-1], true)?
        .add(&Array::scalar(&s, 1e-6, D::F32)?)?
        .rsqrt()?;
    let normed = x
        .mul(&rms)?
        .mul(&norm.cast(D::F32)?)?
        .reshape(&[1, t, heads, 64, 2])?;
    let (even, odd) = (normed.slice_axis(4, 0, 1)?, normed.slice_axis(4, 1, 2)?);
    let (c, sn) = (
        cos.reshape(&[1, t, 1, 64, 1])?,
        sin.reshape(&[1, t, 1, 64, 1])?,
    );
    let rotated = Array::concat(
        &[
            &even.mul(&c)?.sub(&odd.mul(&sn)?)?,
            &even.mul(&sn)?.add(&odd.mul(&c)?)?,
        ],
        4,
    )?
    .reshape(&[1, t, heads, 128])?
    .transpose(&[0, 2, 1, 3])?
    .cast(D::BF16)?;
    assert_close(&q, &rotated, 2e-2, 2e-2)?;
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn layer_norm_modulation_matches_composed_operators() -> Result<()> {
    let s = Stream::metal(0)?;
    let t = 3;
    let one = Array::scalar(&s, 1.0, D::BF16)?;
    let x = normal(&s, &[1, t, 4096], 11, 2.0)?;
    let a = normal(&s, &[1, t, 4096], 12, 1.0)?;
    let m1 = normal(&s, &[1, 1, 8192], 13, 0.5)?.broadcast_to(&[1, t, 8192])?;
    let m2 = normal(&s, &[1, 1, 8192], 14, 0.5)?.broadcast_to(&[1, t, 8192])?;
    let modulate = |h: &Array, m: &Array| -> Result<Array> {
        h.fast_layer_norm(None, None, 1e-6)?
            .mul(&one.add(&m.slice_axis(2, 0, 4096)?)?)
    };
    let out = LayerNormModulate::new(&s)?.call(&x, &m1)?;
    assert_close(&out, &modulate(&x, &m1)?, 2e-2, 2e-2)?;

    let (x1, out) = GatedResidualLayerNormModulate::new(&s)?.call(&x, &a, &m1, &m2)?;
    let expected_x1 = x.add(&m1.slice_axis(2, 4096, 8192)?.tanh()?.mul(&a)?)?;
    assert_close(&x1, &expected_x1, 1e-2, 1e-2)?;
    assert_close(&out, &modulate(&x1, &m2)?, 2e-2, 2e-2)?;
    assert!(GatedResidualLayerNormModulate::new(&s)?
        .call(&x, &a, &m1.slice_axis(2, 0, 4096)?, &m2)
        .is_err());
    Ok(())
}

#![cfg(feature = "native")]
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

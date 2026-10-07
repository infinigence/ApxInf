#![cfg(feature = "native")]
//! Run only under the repository's shared Metal qualification lock. No test
//! here runs as part of default cargo test; all require explicit --ignored.
use apxinf_core::{
    contracts::{AttentionMask, AttentionOptions, AxisSlice},
    Backend, DType, Device, NextTokenLogits, PortableOps, Result, RngKey, SamplingBackend, Shape,
    Tensor, TokenSamplingInit, TokenSamplingParams, TokenSamplingSpec,
};
use apxinf_mlx::{Array, MlxBackend, MlxDType, Stream};
use std::rc::Rc;

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn opaque_storage_reconciles_reshape_and_rejects_foreign_metadata() -> Result<()> {
    let b = MlxBackend::new(0)?;
    let a = b.to_device(&Tensor::from_f32(vec![2, 3], &[1., 2., 3., 4., 5., 6.])?)?;
    let reshaped = a.reshape(vec![3, 2])?;
    assert_eq!(b.array(&reshaped)?.shape(), [3, 2]);
    assert_eq!(
        b.to_cpu(&b.permute(&reshaped, &[1, 0])?)?.to_f32_vec()?,
        [1., 3., 5., 2., 4., 6.]
    );
    let slice = b.slice_axis(
        &a,
        AxisSlice {
            axis: 1,
            start: 1,
            end: 3,
        },
    )?;
    assert_eq!(b.to_cpu(&slice)?.to_f32_vec()?, [2., 3., 5., 6.]);
    let foreign = Tensor::from_opaque_parts(
        Shape::new(vec![2, 3]),
        DType::F32,
        Device::Metal(0),
        24,
        Rc::new(123u32),
    )?;
    assert!(b.array(&foreign).is_err());
    let source = b.array(&a)?;
    let forged = Tensor::from_opaque_parts(
        Shape::new(vec![2, 3]),
        DType::BF16,
        Device::Metal(0),
        12,
        Rc::new(source),
    )?;
    assert!(b.array(&forged).is_err());
    let cpu = Array::from_f32(&Stream::cpu()?, &[2, 3], &[0.; 6])?;
    assert!(b.from_array(cpu).is_err());
    assert!(b.begin_capture().is_err());
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn portable_attention_obeys_grouping_causal_offsets_and_all_masked_rows() -> Result<()> {
    let b = MlxBackend::new(0)?;
    let q = b.to_device(&Tensor::from_f32(vec![1, 1, 2, 2], &[1., 0., 0., 1.])?)?;
    let k = b.to_device(&Tensor::from_f32(vec![1, 2, 1, 2], &[1., 0., 0., 1.])?)?;
    let v = b.to_device(&Tensor::from_f32(vec![1, 2, 1, 2], &[2., 4., 6., 8.])?)?;
    let out = b.attention(&q, &k, &v, &AttentionOptions::full(1.))?;
    let p = 1f32.exp() / (1f32.exp() + 1.);
    let expected = [
        2. * p + 6. * (1. - p),
        4. * p + 8. * (1. - p),
        2. * (1. - p) + 6. * p,
        4. * (1. - p) + 8. * p,
    ];
    for (x, y) in b.to_cpu(&out)?.to_f32_vec()?.iter().zip(expected) {
        assert!((x - y).abs() < 1e-5);
    }
    let mask = b.to_device(&Tensor::from_f32(
        vec![1, 1, 1, 2],
        &[f32::NEG_INFINITY; 2],
    )?)?;
    let masked = AttentionOptions {
        scale: 1.,
        mask: AttentionMask::Additive(&mask),
        scores: DType::F32,
        probabilities: DType::F32,
    };
    assert_eq!(
        b.to_cpu(&b.attention(&q, &k, &v, &masked)?)?.to_f32_vec()?,
        [0.; 4]
    );
    let causal = AttentionOptions {
        mask: AttentionMask::Causal {
            q_start: 0,
            k_start: 0,
        },
        ..AttentionOptions::full(1.)
    };
    assert_eq!(
        b.to_cpu(&b.attention(&q, &k, &v, &causal)?)?.to_f32_vec()?,
        [2., 4., 2., 4.]
    );
    Ok(())
}

#[test]
#[ignore = "requires shared Metal qualification lock"]
fn greedy_and_transactional_kv_respect_reset_and_capacity() -> Result<()> {
    let b = MlxBackend::new(0)?;
    let logits = b.to_device(&Tensor::from_f32(vec![1, 4], &[f32::NAN, 4., 4., 1.])?)?;
    let mut sampler = b.create_token_sampler(TokenSamplingSpec {
        vocab_size: 4,
        max_sequence_len: 3,
    })?;
    let params = TokenSamplingParams::greedy();
    sampler.begin(TokenSamplingInit {
        prompt_token_ids: &[0],
        params: &params,
        rng: RngKey::default(),
    })?;
    assert_eq!(
        sampler.sample(NextTokenLogits::last(&logits, 4)?)?.token_id,
        1
    );
    assert_eq!(
        sampler.sample(NextTokenLogits::last(&logits, 4)?)?.token_id,
        1
    );
    assert!(sampler.sample(NextTokenLogits::last(&logits, 4)?).is_err());
    let k = b.to_device(&Tensor::from_f32(vec![1, 1, 2], &[1., 0.])?)?;
    let v = b.to_device(&Tensor::from_f32(vec![1, 1, 2], &[3., 7.])?)?;
    let mut cache = b.create_kv_cache(2, 1, 2, 4);
    b.kv_append(&mut *cache, 0, &k, &v, 1)?;
    assert_eq!(cache.seq_len(), 0);
    b.kv_append(&mut *cache, 1, &k, &v, 1)?;
    let y = b.sdpa_decode(&k, &mut *cache, 0, 1, 1, 2, 1, 4)?;
    assert_eq!(b.to_cpu(&y)?.to_f32_vec()?, [3., 7.]);
    cache.advance(1);
    assert_eq!(cache.seq_len(), 1);
    cache.clear()?;
    assert_eq!(cache.seq_len(), 0);
    cache.advance(usize::MAX);
    assert!(b.kv_append(&mut *cache, 0, &k, &v, 1).is_err());
    cache.clear()?;
    b.kv_append(&mut *cache, 0, &k, &v, 1)?;
    b.kv_append(&mut *cache, 1, &k, &v, 1)?;
    cache.advance(1);
    assert_eq!(cache.seq_len(), 1);
    // Owned outputs are independent from clearing the request state.
    assert_eq!(b.to_cpu(&y)?.to_f32_vec()?, [3., 7.]);
    assert_eq!(b.array(&v)?.dtype(), MlxDType::F32);
    Ok(())
}

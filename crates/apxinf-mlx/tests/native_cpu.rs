#![cfg(feature = "native")]
use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, Compiled, MlxDType as D, Stream};
use std::{cell::Cell, rc::Rc};

#[test]
fn empty_compiled_inputs_and_stream_observations() -> Result<()> {
    let s = Stream::cpu()?;
    assert_eq!(s.counters(), Default::default());
    let a = Array::from_f32(&s, &[2], &[3., 4.])?;
    assert_eq!(s.counters().uploads_bytes, 8);
    let compiled = Compiled::new(&s, 1, move |args| {
        assert!(args.is_empty());
        Ok(vec![a.add(&a)?])
    })?;
    let first = compiled.call_and_eval(&[])?;
    let warm = s.counters();
    assert!(warm.trace_callbacks > 0);
    assert_eq!(warm.eval_calls, 1);
    assert_eq!(warm.downloads_bytes, 0);
    let second = compiled.call_and_eval(&[])?;
    assert_eq!(s.counters().trace_callbacks, warm.trace_callbacks);
    assert_eq!(s.counters().uploads_bytes, 8);
    assert_eq!(s.counters().eval_calls, 2);
    assert_eq!(second[0].to_f32_vec()?, [6., 8.]);
    assert_eq!(s.counters().downloads_bytes, 8);
    assert_eq!(s.clone().counters().eval_calls, 3);
    assert_eq!(Stream::cpu()?.counters(), Default::default());
    drop(compiled);
    assert_eq!(first[0].to_f32_vec()?, [6., 8.]);
    Ok(())
}

#[test]
fn disabled_compilation_is_rejected_in_child() -> Result<()> {
    const CHILD: &str = "APXINF_MLX_DISABLED_COMPILE_TEST_CHILD";
    if std::env::var_os(CHILD).is_some() {
        let s = Stream::cpu()?;
        let result = Compiled::new(&s, 1, |a| Ok(vec![a[0].clone()]));
        let message = result
            .err()
            .expect("disabled compiler must be rejected")
            .to_string();
        assert!(message.contains("MLX_DISABLE_COMPILE"), "{message}");
        return Ok(());
    }
    // Presence, not the string's truth value, is MLX's actual semantics.
    let output = std::process::Command::new(std::env::current_exe()?)
        .args([
            "--exact",
            "disabled_compilation_is_rejected_in_child",
            "--nocapture",
        ])
        .env(CHILD, "1")
        .env("MLX_DISABLE_COMPILE", "0")
        .output()?;
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    Ok(())
}

#[test]
fn preparation_verifies_replay_once_per_profile() -> Result<()> {
    let s = Stream::cpu()?;
    let compiled = Compiled::new(&s, 1, |a| Ok(vec![a[0].add(&a[0])?]))?;
    let input = Array::from_f32(&s, &[2], &[1., 2.])?;
    let output = compiled.prepare(&[input.clone()])?;
    let warm = s.counters();
    assert_eq!(warm.eval_calls, 2);
    assert!(warm.trace_callbacks > 0);
    let repeated = compiled.call_and_eval(&[input])?;
    assert_eq!(s.counters().trace_callbacks, warm.trace_callbacks);
    assert_eq!(output[0].to_f32_vec()?, [2., 4.]);
    assert_eq!(repeated[0].to_f32_vec()?, [2., 4.]);
    Ok(())
}

#[test]
fn upload_views_dtype_and_owned_lifetime() -> Result<()> {
    let s = Stream::cpu()?;
    let a = Array::from_f32(&s, &[2, 3], &[1., 2., 3., 4., 5., 6.])?;
    assert_eq!(
        a.transpose(&[1, 0])?.contiguous()?.to_f32_vec()?,
        [1., 4., 2., 5., 3., 6.]
    );
    assert_eq!(
        a.reshape(&[6])?.slice_axis(0, 1, 4)?.to_f32_vec()?,
        [2., 3., 4.]
    );
    let bf = a.cast(D::BF16)?;
    let raw = bf.to_bytes()?;
    assert_eq!(
        Array::from_bytes(&s, &[2, 3], D::BF16, &raw)?.to_f32_vec()?,
        [1., 2., 3., 4., 5., 6.]
    );
    let clone = a.clone();
    drop(a);
    drop(s);
    assert_eq!(clone.to_f32_vec()?, [1., 2., 3., 4., 5., 6.]);
    assert!(clone.reshape(&[5]).is_err());
    assert!(clone.transpose(&[0, 0]).is_err());
    assert!(clone.as_strided(&[2, 3], &[4, 1], 0).is_err());
    assert!(clone.as_strided(&[0], &[1], 7).is_err());
    Ok(())
}

#[test]
fn independent_explicit_state_and_trace_reuse() -> Result<()> {
    let s = Stream::cpu()?;
    let traces = Rc::new(Cell::new(0));
    let seen = traces.clone();
    let compiled = Compiled::new(&s, 2, move |a| {
        seen.set(seen.get() + 1);
        let y = a[0].matmul(&a[1])?;
        let next = a[2].add(&y)?;
        Ok(vec![y, next])
    })?;
    let weights = Array::from_f32(&s, &[2, 2], &[1., 2., 3., 4.])?;
    let state = Array::zeros(&s, &[1, 2], D::F32)?;
    let x = Array::from_f32(&s, &[1, 2], &[2., 3.])?;
    let first = compiled.call_and_eval(&[x, weights.clone(), state.clone()])?;
    assert_eq!(first[0].to_f32_vec()?, [11., 16.]);
    assert_eq!(first[1].to_f32_vec()?, [11., 16.]);
    let after_first = traces.get();
    let x2 = Array::from_f32(&s, &[1, 2], &[1., 1.])?;
    let second = compiled.call_and_eval(&[x2.clone(), weights.clone(), first[1].clone()])?;
    assert_eq!(second[1].to_f32_vec()?, [15., 22.]);
    assert_eq!(traces.get(), after_first);
    let fresh = compiled.call_and_eval(&[x2, weights, state.clone()])?;
    assert_eq!(fresh[1].to_f32_vec()?, [4., 6.]);
    assert_eq!(state.to_f32_vec()?, [0., 0.]);
    drop(compiled);
    assert_eq!(second[1].to_f32_vec()?, [15., 22.]);
    Ok(())
}

#[test]
fn repeated_compiled_instances_release_closures_and_keep_own_constants() -> Result<()> {
    let s = Stream::cpu()?;
    for value in 1..=32 {
        let marker = Rc::new(());
        let weak = Rc::downgrade(&marker);
        let constant = Array::from_f32(&s, &[1], &[value as f32])?;
        let compiled = Compiled::new(&s, 1, move |args| {
            let _keep_marker = &marker;
            Ok(vec![args[0].add(&constant)?])
        })?;
        let input = Array::from_f32(&s, &[1], &[1.])?;
        let output = compiled.call(&[input])?;
        // Even a lazy output owns what it needs after the callable and its
        // tracing closure are destroyed; no reused cache id changes constants.
        drop(compiled);
        assert!(weak.upgrade().is_none());
        assert_eq!(output[0].to_f32_vec()?, [value as f32 + 1.]);
    }
    Ok(())
}

#[test]
fn callback_errors_panic_and_stream_mismatch_are_contained() -> Result<()> {
    let s = Stream::cpu()?;
    let a = Array::from_f32(&s, &[1], &[1.])?;
    let bad = Compiled::new(&s, 1, |_| {
        Err(Error::Other("deliberate trace failure".into()))
    })?;
    assert!(bad
        .call(&[a.clone()])
        .unwrap_err()
        .to_string()
        .contains("deliberate trace failure"));
    let panics = Compiled::new(&s, 1, |_| panic!("deliberate callback panic"))?;
    assert!(panics
        .call(&[a.clone()])
        .unwrap_err()
        .to_string()
        .contains("panic in MLX"));
    let other = Stream::cpu()?;
    let b = Array::from_f32(&other, &[1], &[2.])?;
    assert!(a.add(&b).is_err());
    // A failed callback must not poison an independently prepared callable.
    let good = Compiled::new(&s, 1, |a| Ok(vec![a[0].add(&a[0])?]))?;
    assert_eq!(good.call_and_eval(&[a])?[0].to_f32_vec()?, [2.]);
    Ok(())
}

#[test]
fn dynamic_slice_update_preserves_old_state() -> Result<()> {
    let s = Stream::cpu()?;
    let compiled = Compiled::new(&s, 1, |a| {
        Ok(vec![a[0].slice_update(&a[1], &a[2], &[1])?])
    })?;
    let state = Array::zeros(&s, &[1, 4, 2], D::F32)?;
    let update = Array::from_f32(&s, &[1, 1, 2], &[7., 8.])?;
    let i1 = Array::from_i32(&s, &[1], &[1])?;
    let i2 = Array::from_i32(&s, &[1], &[2])?;
    let first = compiled.call_and_eval(&[state.clone(), update.clone(), i1])?;
    let second = compiled.call_and_eval(&[first[0].clone(), update, i2])?;
    assert_eq!(state.to_f32_vec()?, [0.; 8]);
    assert_eq!(first[0].to_f32_vec()?, [0., 0., 7., 8., 0., 0., 0., 0.]);
    assert_eq!(second[0].to_f32_vec()?, [0., 0., 7., 8., 7., 8., 0., 0.]);
    Ok(())
}

#[test]
fn quantized_matmul_and_array_math() -> Result<()> {
    let s = Stream::cpu()?;
    let data: Vec<f32> = (0..128).map(|i| ((i % 17) as f32 - 8.) / 8.).collect();
    let w = Array::from_f32(&s, &[2, 64], &data)?;
    let [qw, scales, biases] = w.quantize(64, 8)?;
    let dequantized = qw.dequantize(&scales, &biases, 64, 8)?;
    assert_eq!(dequantized.shape(), w.shape());
    for (actual, expected) in dequantized.to_f32_vec()?.iter().zip(&data) {
        assert!((actual - expected).abs() < 0.01);
    }
    let x = Array::from_f32(&s, &[1, 64], &vec![1.; 64])?;
    let expected = x.matmul(&w.transpose(&[1, 0])?)?.to_f32_vec()?;
    let actual = x
        .quantized_matmul(&qw, &scales, &biases, true, 64, 8)?
        .to_f32_vec()?;
    for (a, b) in actual.iter().zip(expected) {
        assert!((a - b).abs() < 0.2, "{a} {b}");
    }
    let pos = Array::arange(&s, 0., 4., 1., D::F32)?;
    let bool_mask = pos.less_equal(&Array::scalar(&s, 1., D::F32)?)?;
    assert_eq!(
        bool_mask
            .where_select(&pos, &Array::scalar(&s, -1., D::F32)?)?
            .to_f32_vec()?,
        [0., 1., -1., -1.]
    );
    assert_eq!(pos.argmax(-1)?.to_u32_scalar()?, 3);
    Ok(())
}

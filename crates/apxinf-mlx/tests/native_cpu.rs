#![cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]
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

#[test]
fn unsupported_metal_index_is_rejected_before_native_access() {
    for index in [1, usize::MAX] {
        assert!(matches!(
            Stream::metal(index),
            Err(Error::UnsupportedDevice(apxinf_core::Device::Metal(id))) if id == index
        ));
        assert!(matches!(
            apxinf_mlx::MlxBackend::new(index),
            Err(Error::UnsupportedDevice(apxinf_core::Device::Metal(id))) if id == index
        ));
    }
}

fn write_safetensors(path: &std::path::Path, tensors: &[(&str, &str, &[usize], &[u8])]) {
    let mut offset = 0;
    let entries = tensors
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            let entry = format!(
                r#""{name}":{{"dtype":"{dtype}","shape":{shape:?},"data_offsets":[{offset},{}]}}"#,
                offset + bytes.len()
            );
            offset += bytes.len();
            entry
        })
        .collect::<Vec<_>>();
    let header = format!("{{{}}}", entries.join(","));
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(header.as_bytes());
    for (_, _, _, bytes) in tensors {
        file.extend_from_slice(bytes);
    }
    std::fs::write(path, file).unwrap();
}

#[test]
fn safetensors_late_payload_failure_is_not_published() -> Result<()> {
    let stream = Stream::cpu()?;
    let file = std::env::temp_dir().join(format!(
        "apxinf-mlx-late-read-{}.safetensors",
        std::process::id()
    ));
    let payload = [1.0f32, 2.0, 3.0, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    write_safetensors(&file, &[("weight", "F32", &[4], &payload)]);
    let header_end = std::fs::metadata(&file)?.len() - payload.len() as u64;
    let before = stream.counters();
    let predicate_called = Cell::new(false);
    let result = apxinf_mlx::load_safetensors(&stream, &[&file], |name| {
        assert_eq!(name, "weight");
        assert!(!predicate_called.replace(true));
        std::fs::OpenOptions::new()
            .write(true)
            .open(&file)
            .unwrap()
            .set_len(header_end)
            .unwrap();
        true
    });
    std::fs::remove_file(&file)?;
    assert!(predicate_called.get());
    let failure = result
        .expect_err("a failed payload read must not publish checkpoint arrays")
        .to_string();
    assert!(failure.contains(file.file_name().unwrap().to_str().unwrap()));
    assert!(failure.contains("read"));
    assert_eq!(stream.counters().uploads_bytes, before.uploads_bytes);
    assert_eq!(stream.counters().eval_calls, before.eval_calls + 1);
    Ok(())
}

#[test]
fn affine_checkpoint_shards_load_once_and_reject_ambiguous_files() -> Result<()> {
    let s = Stream::cpu()?;
    let dir = std::env::temp_dir().join(format!("apxinf-mlx-safetensors-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let data: Vec<f32> = (0..512).map(|i| ((i % 23) as f32 - 11.) / 16.).collect();
    let w = Array::from_f32(&s, &[8, 64], &data)?.cast(D::BF16)?;
    let [qw, scales, biases] = w.quantize(64, 4)?;
    assert_eq!((qw.dtype(), qw.shape()), (D::U32, &[8, 8][..]));
    let (qw_b, scales_b, biases_b) = (qw.to_bytes()?, scales.to_bytes()?, biases.to_bytes()?);
    let norm = 0.5f32.to_le_bytes();
    let (first, second) = (dir.join("a.safetensors"), dir.join("b.safetensors"));
    write_safetensors(
        &first,
        &[
            ("l.weight", "U32", &[8, 8], &qw_b),
            ("l.scales", "BF16", &[8, 1], &scales_b),
        ],
    );
    write_safetensors(
        &second,
        &[
            ("l.biases", "BF16", &[8, 1], &biases_b),
            ("norm", "F32", &[1], &norm),
        ],
    );
    let before = s.counters();
    let loaded = apxinf_mlx::load_safetensors(&s, &[&first, &second], |_| true)?;
    let after = s.counters();
    assert_eq!(loaded.len(), 4);
    assert_eq!(after.eval_calls - before.eval_calls, 2);
    assert_eq!(
        after.uploads_bytes - before.uploads_bytes,
        qw_b.len() + scales_b.len() + biases_b.len() + norm.len()
    );
    assert_eq!(loaded["l.weight"].dtype(), D::U32);
    assert_eq!(loaded["l.weight"].to_bytes()?, qw_b);
    assert_eq!(loaded["norm"].to_f32_vec()?, [0.5]);
    let x = Array::from_f32(&s, &[2, 64], &data[..128])?.cast(D::BF16)?;
    let expected = x.quantized_matmul(&qw, &scales, &biases, true, 64, 4)?;
    let actual = x.quantized_matmul(
        &loaded["l.weight"],
        &loaded["l.scales"],
        &loaded["l.biases"],
        true,
        64,
        4,
    )?;
    assert_eq!(actual.to_bytes()?, expected.to_bytes()?);
    let before = s.counters();
    let norm_only = apxinf_mlx::load_safetensors(&s, &[&first, &second], |name| name == "norm")?;
    assert_eq!(norm_only.keys().collect::<Vec<_>>(), ["norm"]);
    assert_eq!(
        s.counters().uploads_bytes - before.uploads_bytes,
        norm.len()
    );

    let fail = |files: &[&std::path::Path]| {
        apxinf_mlx::load_safetensors(&s, files, |_| true)
            .unwrap_err()
            .to_string()
    };
    let duplicate = dir.join("duplicate.safetensors");
    write_safetensors(&duplicate, &[("l.scales", "BF16", &[8, 1], &scales_b)]);
    assert!(fail(&[&first, &duplicate]).contains("duplicate tensor `l.scales`"));
    let wide = dir.join("wide.safetensors");
    write_safetensors(&wide, &[("step", "I64", &[1], &7i64.to_le_bytes())]);
    assert!(fail(&[&wide]).contains("tensor `step`"));
    let truncated = dir.join("truncated.safetensors");
    let bytes = std::fs::read(&second).unwrap();
    std::fs::write(&truncated, &bytes[..bytes.len() - 2]).unwrap();
    assert!(fail(&[&truncated]).contains("truncated.safetensors"));
    assert!(fail(&[&dir.join("missing.safetensors")]).contains("missing.safetensors"));
    std::fs::remove_dir_all(&dir).unwrap();
    Ok(())
}

fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len());
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tolerance, "{a} vs {e}");
    }
}

#[test]
fn elementwise_reduction_and_normalization_ops_follow_mlx_semantics() -> Result<()> {
    let s = Stream::cpu()?;
    let x = Array::from_f32(&s, &[2, 3], &[-2., -0.5, 0., 0.5, 1., 3.])?;
    let y = Array::from_f32(&s, &[2, 3], &[1., -1., 0., 2., 0., -4.])?;
    close(
        &x.erf()?.to_f32_vec()?,
        &[
            -0.995_322_3,
            -0.520_499_9,
            0.,
            0.520_499_9,
            0.842_700_8,
            0.999_977_9,
        ],
        1e-6,
    );
    let zero = Array::scalar(&s, 0., D::F32)?;
    let softplus = x.logaddexp(&zero)?.to_f32_vec()?;
    let expected: Vec<f32> = [-2f32, -0.5, 0., 0.5, 1., 3.]
        .iter()
        .map(|v| v.exp().ln_1p())
        .collect();
    close(&softplus, &expected, 1e-6);
    assert_eq!(x.abs()?.to_f32_vec()?, [2., 0.5, 0., 0.5, 1., 3.]);
    assert_eq!(x.maximum(&y)?.to_f32_vec()?, [1., -0.5, 0., 2., 1., 3.]);
    assert_eq!(x.minimum(&y)?.to_f32_vec()?, [-2., -1., 0., 0.5, 0., -4.]);
    close(
        &y.norm(&[-1], false)?.to_f32_vec()?,
        &[2f32.sqrt(), 20f32.sqrt()],
        1e-6,
    );
    assert_eq!(y.norm(&[-1], true)?.shape(), [2, 1]);

    // addmm rounds bias + product once; softmax_with keeps the precise flag.
    let a = Array::from_f32(&s, &[1, 2], &[1., 2.])?;
    let b = Array::from_f32(&s, &[2, 2], &[1., 0., 0., 1.])?;
    let bias = Array::from_f32(&s, &[2], &[10., 20.])?;
    assert_eq!(bias.addmm(&a, &b, 1., 1.)?.to_f32_vec()?, [11., 22.]);
    assert_eq!(bias.addmm(&a, &b, 2., 0.5)?.to_f32_vec()?, [7., 14.]);
    for precise in [false, true] {
        close(
            &x.softmax_with(-1, precise)?.to_f32_vec()?,
            &x.softmax(-1)?.to_f32_vec()?,
            1e-6,
        );
    }

    // Unit-scale LayerNorm and unweighted RMSNorm over the last axis.
    let ln = x.fast_layer_norm(None, None, 1e-5)?.to_f32_vec()?;
    for row in ln.chunks_exact(3) {
        assert!(row.iter().sum::<f32>().abs() < 1e-5);
        assert!((row.iter().map(|v| v * v).sum::<f32>() / 3. - 1.).abs() < 1e-3);
    }
    let w = Array::from_f32(&s, &[3], &[2., 2., 2.])?;
    let b3 = Array::from_f32(&s, &[3], &[1., 1., 1.])?;
    let affine = x.fast_layer_norm(Some(&w), Some(&b3), 1e-5)?.to_f32_vec()?;
    close(
        &affine,
        &ln.iter().map(|v| 2. * v + 1.).collect::<Vec<_>>(),
        1e-5,
    );
    let weight_only = x.fast_layer_norm(Some(&w), None, 1e-5)?.to_f32_vec()?;
    close(
        &weight_only,
        &ln.iter().map(|v| 2. * v).collect::<Vec<_>>(),
        1e-5,
    );
    let rms = x.fast_rms_norm_unweighted(1e-6)?.to_f32_vec()?;
    close(
        &rms,
        &x.fast_rms_norm(&Array::from_f32(&s, &[3], &[1.; 3])?, 1e-6)?
            .to_f32_vec()?,
        0.,
    );
    assert!(x.fast_layer_norm(None, None, 0.).is_err());
    assert!(x.fast_rms_norm_unweighted(f32::NAN).is_err());
    Ok(())
}

#[test]
fn depthwise_causal_conv1d_matches_shifted_weighted_sums() -> Result<()> {
    let s = Stream::cpu()?;
    // Two channels, kernel 3, three leading zero rows: out[t] = sum_j w[c, j] x[t + j].
    let (len, channels, width) = (4usize, 2usize, 3usize);
    let x: Vec<f32> = (0..(len + width - 1) * channels)
        .map(|i| {
            if i < (width - 1) * channels {
                0.
            } else {
                i as f32
            }
        })
        .collect();
    let w = [1f32, 2., 3., -1., 0., 1.];
    let input = Array::from_f32(&s, &[1, len + width - 1, channels], &x)?;
    let weight = Array::from_f32(&s, &[channels, width, 1], &w)?;
    let out = input.conv1d(&weight, 1, 0, 1, channels)?;
    assert_eq!(out.shape(), [1, len, channels]);
    let mut expected = Vec::new();
    for t in 0..len {
        for c in 0..channels {
            expected.push(
                (0..width)
                    .map(|j| w[c * width + j] * x[(t + j) * channels + c])
                    .sum::<f32>(),
            );
        }
    }
    assert_eq!(out.to_f32_vec()?, expected);
    assert!(input.conv1d(&weight, 0, 0, 1, channels).is_err());
    assert!(input.conv1d(&weight, 1, 0, 1, 0).is_err());
    Ok(())
}

fn bits(a: &Array) -> Result<Vec<u32>> {
    Ok(a.to_f32_vec()?.iter().map(|v| v.to_bits()).collect())
}

#[test]
fn seeded_noise_matches_mlx_threefry() -> Result<()> {
    // Expected bits from MLX Python 0.31.2 and 0.32.3 on the CPU device:
    // mx.random.normal([2, 3], key=mx.random.key(42)).
    let s = Stream::cpu()?;
    let key = Array::random_key(&s, 42)?;
    assert_eq!((key.dtype(), key.shape()), (D::U32, &[2][..]));
    let x = Array::random_normal(&[2, 3], D::F32, &key)?;
    assert_eq!(
        bits(&x)?,
        [0x3f1cbd6d, 0x3f8fb0fc, 0x3f91941b, 0xbf500f3b, 0xbf63f190, 0x3e0142e6]
    );
    // A seed above 2^32 uses both key words.
    let wide = Array::random_normal(&[3], D::F32, &Array::random_key(&s, (1 << 40) + 7)?)?;
    assert_eq!(bits(&wide)?, [0xbf3e8d89, 0xbfb36082, 0x3d9b1cba]);
    // The same key gives the same samples; a different key does not.
    assert_eq!(
        bits(&Array::random_normal(&[2, 3], D::F32, &key)?)?,
        bits(&x)?
    );
    let other = Array::random_normal(&[2, 3], D::F32, &Array::random_key(&s, 43)?)?;
    assert_ne!(bits(&other)?, bits(&x)?);
    assert!(Array::random_normal(&[2], D::F32, &x).is_err());
    Ok(())
}

#[test]
fn image_operators_follow_mlx_semantics() -> Result<()> {
    // Expected bits from MLX Python 0.32.3 on the CPU device.
    let s = Stream::cpu()?;
    let input = Array::arange(&s, 0., 24., 1., D::F32)?
        .reshape(&[1, 3, 4, 2])?
        .div(&Array::scalar(&s, 8., D::F32)?)?
        .sub(&Array::scalar(&s, 1., D::F32)?)?;
    let weight = Array::arange(&s, 0., 54., 1., D::F32)?
        .reshape(&[3, 3, 3, 2])?
        .div(&Array::scalar(&s, 16., D::F32)?)?
        .sub(&Array::scalar(&s, 1., D::F32)?)?;
    let same = input.conv2d(&weight, [1, 1], [1, 1], [1, 1], 1)?;
    assert_eq!(same.shape(), [1, 3, 4, 3]);
    assert_eq!(
        &bits(&same)?[..6],
        [0x3fb00000, 0xbfb80000, 0xc0880000, 0x40020000, 0xbf000000, 0xc0420000]
    );
    let strided = input.conv2d(&weight, [2, 1], [1, 0], [1, 1], 1)?;
    assert_eq!(strided.shape(), [1, 2, 2, 3]);
    assert_eq!(
        bits(&strided)?,
        [
            0x40020000, 0xbf000000, 0xc0420000, 0x3f980000, 0x40020000, 0x40380000, 0xc0a00000,
            0x40bf0000, 0x41878000, 0xc0df0000, 0x40ec0000, 0x41adc000
        ]
    );
    assert!(input.conv2d(&weight, [0, 1], [1, 1], [1, 1], 1).is_err());

    let grid = Array::from_f32(&s, &[2, 2], &[1., 2., 3., 4.])?;
    assert_eq!(
        grid.repeat(2, 1)?.to_f32_vec()?,
        [1., 1., 2., 2., 3., 3., 4., 4.]
    );
    assert_eq!(grid.repeat(2, 0)?.shape(), [4, 2]);
    let t = Array::from_f32(&s, &[5], &[-3., -0.5, 0., 0.25, 2.])?;
    assert_eq!(
        bits(&t.tanh()?)?,
        [0xbf7ebbe9, 0xbeec9a9f, 0x0, 0x3e7acbf5, 0x3f76ca83]
    );
    assert_eq!(
        bits(&Array::from_f32(&s, &[3], &[0., 2., 9.])?.sqrt()?)?,
        [0x0, 0x3fb504f3, 0x40400000]
    );
    // mx.clip(x / 2 + 0.5, 0, 1) for BF16 x keeps BF16.
    let x = Array::from_f32(&s, &[3], &[-3., 0.25, 2.])?.cast(D::BF16)?;
    let scalar = |v| Array::scalar(&s, v, D::BF16);
    let image = x
        .div(&scalar(2.)?)?
        .add(&scalar(0.5)?)?
        .clip(&scalar(0.)?, &scalar(1.)?)?;
    assert_eq!(image.dtype(), D::BF16);
    assert_eq!(image.to_f32_vec()?, [0., 0.625, 1.]);
    Ok(())
}

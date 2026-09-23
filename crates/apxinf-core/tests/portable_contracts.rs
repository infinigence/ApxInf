use apxinf_core::{contracts::*, Backend, CpuBackend, DType, Device, Error, Tensor};
fn t(shape: &[usize]) -> Tensor {
    Tensor::zeros(shape.to_vec(), DType::F32)
}
#[test]
fn layout_contracts_cover_qkv_and_patch_merge() {
    assert_eq!(
        AxisSlice {
            axis: 1,
            start: 0,
            end: 1024
        }
        .output_shape(&[256, 3072])
        .unwrap(),
        vec![256, 1024]
    );
    assert_eq!(
        permuted_shape(&[2, 16, 16, 3, 14, 14], &[0, 1, 2, 4, 5, 3]).unwrap(),
        vec![2, 16, 16, 14, 14, 3]
    );
    assert_eq!(
        concatenated_shape(&[&[1, 512, 1, 256], &[1, 10, 1, 256]], 1).unwrap(),
        vec![1, 522, 1, 256]
    );
    broadcast_shape(&[256, 1152], &[2, 256, 1152]).unwrap();
    broadcast_shape(&[1, 1024], &[10, 1024]).unwrap();
    broadcast_shape(&[], &[10, 32]).unwrap();
}
#[test]
fn malformed_layouts_return_errors() {
    assert!(checked_elements(&[usize::MAX, 2]).is_err());
    assert!(checked_elements(&[1, 0]).is_err());
    for slice in [
        AxisSlice {
            axis: 2,
            start: 0,
            end: 1,
        },
        AxisSlice {
            axis: 1,
            start: 3,
            end: 2,
        },
        AxisSlice {
            axis: 1,
            start: 0,
            end: 4,
        },
    ] {
        assert!(slice.output_shape(&[2, 3]).is_err());
    }
    assert!(permuted_shape(&[2, 3], &[1, 1]).is_err());
    assert!(permuted_shape(&[2, 3], &[2, 0]).is_err());
    assert!(broadcast_shape(&[2, 3], &[3]).is_err());
    assert!(broadcast_shape(&[2], &[3]).is_err());
    assert!(concatenated_shape(&[], 0).is_err());
    assert!(concatenated_shape(&[&[2, 3], &[1, 3]], 1).is_err());
    assert!(concatenated_shape(&[&[usize::MAX], &[1]], 0).is_err());
}
#[test]
fn full_attention_covers_pi05_and_qwen_vision() {
    for (b, q_len, k_len, hq, hkv, d) in [
        (2, 256, 256, 16, 16, 72),
        (1, 10, 522, 8, 1, 256),
        (1, 16, 32, 16, 8, 128),
    ] {
        let q = t(&[b, q_len, hq, d]);
        let k = t(&[b, k_len, hkv, d]);
        assert_eq!(
            AttentionOptions::full(1. / (d as f32).sqrt())
                .validate(Device::Cpu, &q, &k, &k)
                .unwrap(),
            q.shape().dims()
        );
    }
}
#[test]
fn causal_offsets_and_mask_broadcast_are_explicit() {
    let q = t(&[1, 2, 4, 8]);
    let k = t(&[1, 7, 2, 8]);
    let mut o = AttentionOptions::full(0.5);
    o.mask = AttentionMask::Causal {
        q_start: 5,
        k_start: 0,
    };
    o.validate(Device::Cpu, &q, &k, &k).unwrap();
    o.mask = AttentionMask::Causal {
        q_start: usize::MAX,
        k_start: 0,
    };
    assert!(o.validate(Device::Cpu, &q, &k, &k).is_err());
    let mask = Tensor::from_f32(
        vec![1, 1, 1, 7],
        &[0., 0., 0., 0., 0., f32::NEG_INFINITY, f32::NEG_INFINITY],
    )
    .unwrap();
    o.mask = AttentionMask::Additive(&mask);
    o.validate(Device::Cpu, &q, &k, &k).unwrap();
    validate_mask_values(&mask).unwrap();
}
#[test]
fn attention_structure_validation_does_not_scan_mask_values() {
    let q = t(&[1, 2, 4, 8]);
    let k = t(&[1, 7, 2, 8]);
    for value in [f32::NAN, f32::INFINITY] {
        let mask = Tensor::from_f32(vec![1, 1, 1, 1], &[value]).unwrap();
        let mut o = AttentionOptions::full(0.5);
        o.mask = AttentionMask::Additive(&mask);
        assert_eq!(o.validate(Device::Cpu, &q, &k, &k).unwrap(), [1, 2, 4, 8]);
        assert!(validate_mask_values(&mask).is_err());
    }
    // Value validity does not replace rank, broadcast or dtype checks.
    for mask in [t(&[7]), t(&[1, 1, 1, 8])] {
        validate_mask_values(&mask).unwrap();
        let mut o = AttentionOptions::full(0.5);
        o.mask = AttentionMask::Additive(&mask);
        assert!(o.validate(Device::Cpu, &q, &k, &k).is_err());
    }
    let mask = Tensor::zeros(vec![1, 1, 1, 7], DType::BF16);
    let mut o = AttentionOptions::full(0.5);
    o.mask = AttentionMask::Additive(&mask);
    assert!(matches!(
        o.validate(Device::Cpu, &q, &k, &k),
        Err(Error::DTypeMismatch { .. })
    ));
}
#[test]
fn mask_values_are_checked_explicitly_at_creation_and_update() {
    let mut mask = Tensor::from_f32(
        vec![1, 1, 1, 5],
        &[f32::MIN, -1., 0., f32::MAX, f32::NEG_INFINITY],
    )
    .unwrap();
    validate_mask_values(&mask).unwrap();
    mask.as_f32_mut().unwrap()[0] = f32::NAN;
    assert!(validate_mask_values(&mask).is_err());
    mask.as_f32_mut().unwrap()[0] = f32::INFINITY;
    assert!(validate_mask_values(&mask).is_err());
    mask.as_f32_mut().unwrap()[0] = f32::NEG_INFINITY;
    validate_mask_values(&mask).unwrap();
    assert!(matches!(
        validate_mask_values(&Tensor::zeros(vec![1, 1, 1, 1], DType::BF16)),
        Err(Error::DTypeMismatch {
            expected: DType::F32,
            ..
        })
    ));
}
#[test]
fn mask_value_validation_rejects_device_storage_without_reading_it() {
    // Empty device storage needs no CUDA allocation/runtime. Even with no values
    // to scan, a device mask must not silently report successful value validation.
    let mask = Tensor::from_raw_parts(
        vec![1, 1, 1, 0].into(),
        DType::F32,
        Device::Cuda(0),
        apxinf_core::Storage::Gpu {
            device: Device::Cuda(0),
            handle: apxinf_core::storage::GpuStorageHandle {
                ptr: 0,
                len: 0,
                _prevent_leak: None,
            },
        },
    );
    assert!(matches!(
        validate_mask_values(&mask),
        Err(Error::UnsupportedDevice(Device::Cuda(0)))
    ));
}
#[test]
fn attention_rejects_mismatched_precision_shape_or_device() {
    let q = t(&[1, 2, 4, 8]);
    let k = t(&[1, 7, 2, 8]);
    let mut o = AttentionOptions::full(0.5);
    assert!(o.validate(Device::Cuda(0), &q, &k, &k).is_err());
    assert!(o
        .validate(Device::Cpu, &q, &t(&[1, 7, 3, 8]), &t(&[1, 7, 3, 8]))
        .is_err());
    assert!(o.validate(Device::Cpu, &q, &k, &t(&[1, 7, 2, 4])).is_err());
    o.scale = f32::NAN;
    assert!(o.validate(Device::Cpu, &q, &k, &k).is_err());
    o.scale = 0.5;
    o.probabilities = DType::F16;
    assert!(o.validate(Device::Cpu, &q, &k, &k).is_err());
    o.probabilities = DType::F32;
    assert!(o
        .validate(
            Device::Cpu,
            &q,
            &Tensor::zeros(vec![1, 7, 2, 8], DType::BF16),
            &k
        )
        .is_err());
}
#[test]
fn extensions_remain_object_safe_and_never_fake_execution() {
    let b: &dyn Backend = &CpuBackend;
    let x = t(&[2, 3]);
    assert!(matches!(
        b.cast(&x, DType::BF16),
        Err(Error::UnsupportedOp("cast"))
    ));
    assert!(matches!(
        b.broadcast_to(&x, &[4, 2, 3]),
        Err(Error::UnsupportedOp("broadcast_to"))
    ));
    assert!(matches!(
        b.slice_axis(
            &x,
            AxisSlice {
                axis: 1,
                start: 0,
                end: 2
            }
        ),
        Err(Error::UnsupportedOp("slice_axis"))
    ));
    assert!(matches!(
        b.concat_axis(&[&x, &x], 0),
        Err(Error::UnsupportedOp("concat_axis"))
    ));
    assert!(matches!(
        b.permute(&x, &[1, 0]),
        Err(Error::UnsupportedOp("permute"))
    ));
    let q = t(&[1, 2, 1, 4]);
    assert!(matches!(
        b.attention(&q, &q, &q, &AttentionOptions::full(0.5)),
        Err(Error::UnsupportedOp("attention"))
    ));
}

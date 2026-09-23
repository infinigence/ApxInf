use apxinf_core::{contracts::*, Backend, CpuBackend, DType, Device, Error, PortableOps, Tensor};
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

/// #3: validation runs before dispatch, so a contract violation is reported as a
/// contract error even when the backend has no implementation at all. Without the
/// sealed wrapper these would all surface as UnsupportedOp and hide the real bug.
#[test]
fn validated_entry_points_reject_bad_arguments_before_dispatch() {
    let b: &dyn Backend = &CpuBackend;
    let x = t(&[2, 3]);

    // Non-permutation axes: caught by contracts, not by the missing impl.
    assert!(matches!(
        b.permute(&x, &[0, 0]),
        Err(Error::Contract("axes must be a permutation of 0..rank"))
    ));
    // Out-of-bounds slice end.
    assert!(matches!(
        b.slice_axis(
            &x,
            AxisSlice {
                axis: 1,
                start: 0,
                end: 9
            }
        ),
        Err(Error::Contract(_))
    ));
    // Broadcast that would shrink an axis.
    assert!(matches!(
        b.broadcast_to(&x, &[3]),
        Err(Error::ShapeMismatch { .. })
    ));
    // Mixed dtypes into concat.
    assert!(matches!(
        b.concat_axis(&[&x, &Tensor::zeros(vec![2, 3], DType::BF16)], 0),
        Err(Error::DTypeMismatch { .. })
    ));
    // FP8 is not a portable arithmetic dtype.
    assert!(matches!(
        b.cast(&x, DType::F8E4M3),
        Err(Error::UnsupportedDType {
            got: DType::F8E4M3,
            ..
        })
    ));
    // Attention with Hq not a multiple of Hkv.
    let q = t(&[1, 2, 3, 8]);
    let k = t(&[1, 2, 2, 8]);
    assert!(matches!(
        b.attention(&q, &k, &k, &AttentionOptions::full(0.5)),
        Err(Error::Contract(_))
    ));
    // Device mismatch is caught before the impl is consulted.
    assert!(matches!(
        b.permute(&big_cuda_tensor(), &[1, 0]),
        Err(Error::DeviceMismatch { .. })
    ));
}

/// #3: layout ops are bit-preserving, so they must accept non-float dtypes that
/// `float_tensor` would reject. Proves the tensor_storage/float_tensor split.
#[test]
fn layout_ops_admit_non_float_dtypes_while_cast_does_not() {
    let b: &dyn Backend = &CpuBackend;
    let fp8 = Tensor::zeros(vec![2, 3], DType::F8E4M3);
    // Reaches the impl (UnsupportedOp) rather than failing dtype validation.
    assert!(matches!(
        b.permute(&fp8, &[1, 0]),
        Err(Error::UnsupportedOp("permute"))
    ));
    assert!(matches!(
        b.slice_axis(
            &fp8,
            AxisSlice {
                axis: 0,
                start: 0,
                end: 1
            }
        ),
        Err(Error::UnsupportedOp("slice_axis"))
    ));
    // Arithmetic still refuses it.
    assert!(matches!(
        b.cast(&fp8, DType::F32),
        Err(Error::UnsupportedDType { .. })
    ));
}

fn big_cuda_tensor() -> Tensor {
    Tensor::from_raw_parts(
        vec![2, 3].into(),
        DType::F32,
        Device::Cuda(0),
        apxinf_core::Storage::Gpu {
            device: Device::Cuda(0),
            handle: unsafe { apxinf_core::storage::GpuStorageHandle::from_raw_parts(0, 24, None) },
        },
    )
}

/// #4: every legacy default now reports UnsupportedOp with its own name, so a
/// portable path can identify the missing op without matching on message text.
#[test]
fn legacy_defaults_report_unsupported_op_by_name() {
    let b: &dyn Backend = &CpuBackend;
    let x = t(&[2, 3]);
    let names = [
        b.layer_norm(&x, &x, &x, 1e-5).unwrap_err(),
        b.gelu_tanh(&x).unwrap_err(),
        b.add_bias(&x, &x).unwrap_err(),
        b.concat_2d(&[&x]).unwrap_err(),
        b.vision_sdpa(&x, &x, &x, 2, 1, 3).unwrap_err(),
        b.rope_mrope(&x, 1, 3, 1e4, [1, 1, 1], &[0]).unwrap_err(),
        b.rope_vision_2d(&x, 1, 3, 1e4, &[0, 0]).unwrap_err(),
    ];
    assert_eq!(
        names
            .iter()
            .filter_map(|e| e.unsupported_op())
            .collect::<Vec<_>>(),
        vec![
            "layer_norm",
            "gelu_tanh",
            "add_bias",
            "concat_2d",
            "vision_sdpa",
            "rope_mrope",
            "rope_vision_2d"
        ]
    );
}

/// A backend whose `cast_impl` ignores the requested dtype and echoes the input.
/// Proves the public entry point rejects a wrong-dtype result even when the shape
/// is correct — reproduces the reviewer's cast(F32 -> BF16) case.
struct EchoCastBackend;
impl apxinf_core::SamplingBackend for EchoCastBackend {
    fn create_token_sampler(
        &self,
        _spec: apxinf_core::TokenSamplingSpec,
    ) -> Result<Box<dyn apxinf_core::TokenSampler>, Error> {
        unimplemented!()
    }
    fn create_normal_generator(
        &self,
        _output: Tensor,
    ) -> Result<Box<dyn apxinf_core::NormalGenerator>, Error> {
        unimplemented!()
    }
}
impl Backend for EchoCastBackend {
    // The one misbehaving hook: returns the input verbatim, ignoring `dtype`.
    fn cast_impl(&self, input: &Tensor, _dtype: DType) -> Result<Tensor, Error> {
        input.reshape(input.shape().dims().to_vec())
    }
    fn rms_norm(&self, _i: &Tensor, _w: &Tensor, _e: f32) -> Result<Tensor, Error> {
        unimplemented!()
    }
    fn silu(&self, _i: &Tensor) -> Result<Tensor, Error> {
        unimplemented!()
    }
    fn add(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor, Error> {
        unimplemented!()
    }
    fn mul(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor, Error> {
        unimplemented!()
    }
    fn scale(&self, _i: &Tensor, _f: f32) -> Result<Tensor, Error> {
        unimplemented!()
    }
    fn matmul(&self, _a: &Tensor, _b: &Tensor) -> Result<Tensor, Error> {
        unimplemented!()
    }
    fn rope(&self, _i: &Tensor, _h: usize, _d: usize, _t: f32, _p: u32) -> Result<Tensor, Error> {
        unimplemented!()
    }
    fn embedding(&self, _t: &Tensor, _ids: &[u32]) -> Result<Tensor, Error> {
        unimplemented!()
    }
    fn sdpa_decode(
        &self,
        _q: &Tensor,
        _kv: &mut dyn apxinf_core::KvCache,
        _l: usize,
        _h: usize,
        _kvh: usize,
        _d: usize,
        _kl: usize,
        _m: usize,
    ) -> Result<Tensor, Error> {
        unimplemented!()
    }
    fn sdpa_prefill(
        &self,
        _q: &Tensor,
        _kv: &mut dyn apxinf_core::KvCache,
        _l: usize,
        _h: usize,
        _kvh: usize,
        _d: usize,
        _kl: usize,
        _m: usize,
    ) -> Result<Tensor, Error> {
        unimplemented!()
    }
    fn create_kv_cache(
        &self,
        _n: usize,
        _kvh: usize,
        _d: usize,
        _m: usize,
    ) -> Box<dyn apxinf_core::KvCache> {
        unimplemented!()
    }
    fn kv_append(
        &self,
        _kv: &mut dyn apxinf_core::KvCache,
        _l: usize,
        _k: &Tensor,
        _v: &Tensor,
        _n: usize,
    ) -> Result<(), Error> {
        unimplemented!()
    }
    fn synchronize(&self) -> Result<(), Error> {
        Ok(())
    }
    fn begin_capture(&self) -> Result<(), Error> {
        unimplemented!()
    }
    fn end_capture(&self) -> Result<Box<dyn apxinf_core::Graph>, Error> {
        unimplemented!()
    }
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn to_device(&self, _t: &Tensor) -> Result<Tensor, Error> {
        unimplemented!()
    }
    fn to_cpu(&self, _t: &Tensor) -> Result<Tensor, Error> {
        unimplemented!()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Output validation rejects a result with the right shape but the wrong dtype.
/// Without the dtype check the public `cast` entry would return success.
#[test]
fn output_check_rejects_wrong_dtype_even_when_shape_matches() {
    let b: &dyn Backend = &EchoCastBackend;
    let x = t(&[2, 3]);
    // Shape is preserved, so a shape-only check would pass; dtype is still F32.
    assert!(matches!(
        b.cast(&x, DType::BF16),
        Err(Error::DTypeMismatch {
            expected: DType::BF16,
            got: DType::F32,
        })
    ));
    // Sanity: a same-dtype echo satisfies every output check.
    assert!(b.cast(&x, DType::F32).is_ok());
}

/// Output byte size is validated before dispatch: a broadcast whose element count
/// fits usize but whose byte extent (elements * 4) overflows is rejected with a
/// contract error rather than reaching the backend. The tiny input means the guard
/// fires before any large allocation is attempted.
#[test]
fn output_byte_overflow_is_rejected_before_dispatch() {
    let b: &dyn Backend = &CpuBackend;
    let huge = usize::MAX / 4 + 1; // element count is legal, * 4 bytes overflows
    assert!(matches!(
        b.broadcast_to(&t(&[1]), &[huge]),
        Err(Error::Contract("output byte size overflow"))
    ));
}

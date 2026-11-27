#![cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]

use apxinf_mlx::{Array, MetalKernel, MlxDType, QuantMode, Stream, TemplateArg};

#[test]
fn u8_view_and_basic_array_ops_match_semantics() {
    let s = Stream::cpu().unwrap();

    // U8 upload + view(U32) bitcast (little-endian).
    let raw = Array::from_u8(&s, &[2, 4], &[1, 2, 3, 4, 0x10, 0x20, 0x30, 0x40]).unwrap();
    assert_eq!(raw.dtype(), MlxDType::U8);
    let packed = raw.view(MlxDType::U32).unwrap();
    assert_eq!(packed.shape(), &[2, 1]);
    assert_eq!(packed.dtype(), MlxDType::U32);
    assert_eq!(packed.to_u32_vec().unwrap(), vec![0x04030201, 0x40302010]);

    // argsort + take_along_axis + floor_divide.
    let vals = Array::from_i32(&s, &[6], &[3, 1, 2, 1, 0, 2]).unwrap();
    let order = vals.argsort(0).unwrap();
    let sorted = vals.take_along_axis(&order, 0).unwrap();
    assert_eq!(sorted.to_i32_vec().unwrap(), vec![0, 1, 1, 2, 2, 3]);
    let inv = order.argsort(0).unwrap();
    let restored = sorted.take_along_axis(&inv, 0).unwrap();
    assert_eq!(restored.to_i32_vec().unwrap(), vec![3, 1, 2, 1, 0, 2]);
    let two = Array::from_u32(&s, &[], &[2]).unwrap();
    let q = order.floor_divide(&two).unwrap();
    assert_eq!(q.to_u32_vec().unwrap(), vec![2, 0, 1, 1, 2, 0]);

    // slice_strided (0::2 and 1::2 interleaved gate/up).
    let inter = Array::from_f32(&s, &[2, 6], &[0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]).unwrap();
    let even = inter.slice_strided(&[0, 0], &[2, 6], &[1, 2]).unwrap();
    let odd = inter.slice_strided(&[0, 1], &[2, 6], &[1, 2]).unwrap();
    assert_eq!(even.to_f32_vec().unwrap(), vec![0., 2., 4., 6., 8., 10.]);
    assert_eq!(odd.to_f32_vec().unwrap(), vec![1., 3., 5., 7., 9., 11.]);

    // slice_update_static + array_equal.
    let base = Array::zeros(&s, &[1, 4, 2], MlxDType::F32).unwrap();
    let patch = Array::from_f32(&s, &[1, 2, 2], &[1., 2., 3., 4.]).unwrap();
    let updated = base.slice_update_static(&patch, &[0, 1, 0], &[1, 3, 2]).unwrap();
    assert_eq!(updated.to_f32_vec().unwrap(), vec![0., 0., 1., 2., 3., 4., 0., 0.]);
    let expected = Array::from_f32(&s, &[1, 4, 2], &[0., 0., 1., 2., 3., 4., 0., 0.]).unwrap();
    assert!(updated.array_equal(&expected).unwrap().to_bool_scalar().unwrap());

    // full, greater_equal, less, logical_and, log, stop_gradient.
    let neg_one = Array::from_i32(&s, &[], &[-1]).unwrap();
    let filled = Array::full(&s, &[4], &neg_one, MlxDType::I32).unwrap();
    assert_eq!(filled.to_i32_vec().unwrap(), vec![-1, -1, -1, -1]);

    let pos = Array::from_i32(&s, &[4], &[-1, 0, 5, 10]).unwrap();
    let lo = Array::from_i32(&s, &[], &[0]).unwrap();
    let hi = Array::from_i32(&s, &[], &[8]).unwrap();
    let mask = pos.greater_equal(&lo).unwrap().logical_and(&pos.less(&hi).unwrap()).unwrap();
    assert_eq!(mask.to_bytes().unwrap(), vec![0, 1, 1, 0]);

    let e_in = Array::from_f32(&s, &[2], &[1.0, std::f32::consts::E]).unwrap();
    let logged = e_in.log().unwrap().stop_gradient().unwrap().to_f32_vec().unwrap();
    assert!(logged[0].abs() < 1e-6);
    assert!((logged[1] - 1.0).abs() < 1e-6);
}

#[test]
fn gather_qmm_and_quantized_matmul_support_affine_nvfp4_and_mxfp8() {
    let s = Stream::metal(0).unwrap();

    // 1. Affine 8-bit group 64: gather_qmm vs dequantize + matmul.
    let e = 4usize;
    let n = 64usize;
    let k = 128usize;
    let w_data: Vec<f32> = (0..(e * n * k))
        .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
        .collect();
    let w_bf16 = Array::from_f32(&s, &[e, n, k], &w_data)
        .unwrap()
        .cast(MlxDType::BF16)
        .unwrap();
    let [wq, sc, bi] = w_bf16.quantize(64, 8).unwrap();
    let deq = wq
        .dequantize_mode(&sc, Some(&bi), 64, 8, QuantMode::Affine, Some(MlxDType::BF16))
        .unwrap();

    let x_data: Vec<f32> = (0..(2 * k)).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let x = Array::from_f32(&s, &[2, 1, 1, k], &x_data)
        .unwrap()
        .cast(MlxDType::BF16)
        .unwrap();
    let rhs = Array::from_u32(&s, &[2, 2], &[3, 1, 0, 2]).unwrap();
    let lhs = Array::from_u32(&s, &[2, 1], &[0, 1]).unwrap();
    let out = x
        .gather_qmm(
            &wq,
            &sc,
            Some(&bi),
            Some(&lhs),
            Some(&rhs),
            true,
            64,
            8,
            QuantMode::Affine,
            None,
            false,
        )
        .unwrap();
    assert_eq!(out.shape(), &[2, 2, 1, n]);
    let got = out.to_f32_vec().unwrap();

    // Check against quantized_matmul_mode on expert 3 for row 0.
    let w3 = wq.slice_axis(0, 3, 4).unwrap().reshape(&[n, k / 4]).unwrap();
    let s3 = sc.slice_axis(0, 3, 4).unwrap().reshape(&[n, k / 64]).unwrap();
    let b3 = bi.slice_axis(0, 3, 4).unwrap().reshape(&[n, k / 64]).unwrap();
    let x0 = x.slice_axis(0, 0, 1).unwrap().reshape(&[1, k]).unwrap();
    let qmm_ref = x0
        .quantized_matmul_mode(&w3, &s3, Some(&b3), true, 64, 8, QuantMode::Affine)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let max_diff = got[..n]
        .iter()
        .zip(&qmm_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-2, "affine gather_qmm diff={max_diff}");
    assert_eq!(deq.shape(), &[e, n, k]);

    // 2. NVFP4 (group 16, bits 4, U8 scales, U32 packed weights via view).
    let nv_w_u8 = Array::from_u8(
        &s,
        &[e, n, k / 2],
        &(0..(e * n * k / 2)).map(|i| ((i * 37) % 256) as u8).collect::<Vec<_>>(),
    )
    .unwrap();
    let nv_w = nv_w_u8.view(MlxDType::U32).unwrap();
    let nv_sc = Array::from_u8(
        &s,
        &[e, n, k / 16],
        &vec![0x38u8; e * n * k / 16], // 1.0 in FP8 E4M3
    )
    .unwrap();
    let nv_out = x
        .gather_qmm(
            &nv_w,
            &nv_sc,
            None,
            Some(&lhs),
            Some(&rhs),
            true,
            16,
            4,
            QuantMode::Nvfp4,
            None,
            false,
        )
        .unwrap();
    let nv_deq = nv_w
        .dequantize_mode(&nv_sc, None, 16, 4, QuantMode::Nvfp4, Some(MlxDType::BF16))
        .unwrap();
    let nv_w3 = nv_deq.slice_axis(0, 3, 4).unwrap().reshape(&[n, k]).unwrap();
    let nv_ref = x0
        .cast(MlxDType::F32)
        .unwrap()
        .matmul(&nv_w3.cast(MlxDType::F32).unwrap().transpose(&[1, 0]).unwrap())
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let nv_got = nv_out.to_f32_vec().unwrap();
    let nv_diff = nv_got[..n]
        .iter()
        .zip(&nv_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(nv_diff < 0.25, "nvfp4 gather_qmm diff={nv_diff}");

    // 3. MXFP8 (group 32, bits 8, U8 scales, U32 packed weights via view).
    let mx_w_u8 = Array::from_u8(
        &s,
        &[e, n, k],
        &vec![0x38u8; e * n * k], // 1.0 in FP8 E4M3
    )
    .unwrap();
    let mx_w = mx_w_u8.view(MlxDType::U32).unwrap();
    let mx_sc = Array::from_u8(
        &s,
        &[e, n, k / 32],
        &vec![127u8; e * n * k / 32], // 2^0 = 1.0 in E8M0
    )
    .unwrap();
    let mx_out = x
        .gather_qmm(
            &mx_w,
            &mx_sc,
            None,
            Some(&lhs),
            Some(&rhs),
            true,
            32,
            8,
            QuantMode::Mxfp8,
            None,
            false,
        )
        .unwrap();
    let mx_deq = mx_w
        .dequantize_mode(&mx_sc, None, 32, 8, QuantMode::Mxfp8, Some(MlxDType::BF16))
        .unwrap();
    let mx_w3 = mx_deq.slice_axis(0, 3, 4).unwrap().reshape(&[n, k]).unwrap();
    let mx_ref = x0
        .cast(MlxDType::F32)
        .unwrap()
        .matmul(&mx_w3.cast(MlxDType::F32).unwrap().transpose(&[1, 0]).unwrap())
        .unwrap()
        .to_f32_vec()
        .unwrap();
    let mx_got = mx_out.to_f32_vec().unwrap();
    let mx_diff = mx_got[..n]
        .iter()
        .zip(&mx_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(mx_diff < 0.1, "mxfp8 gather_qmm diff={mx_diff}");
}

#[test]
fn sdpa_with_sinks_and_atomic_metal_kernel_work_on_metal() {
    let s = Stream::metal(0).unwrap();

    // sdpa_with_sinks: single query with huge sink dominates -> output ~ 0.
    let q = Array::from_f32(&s, &[1, 2, 1, 4], &[1., 0., 0., 0., 0., 1., 0., 0.])
        .unwrap()
        .cast(MlxDType::BF16)
        .unwrap();
    let k = Array::from_f32(&s, &[1, 2, 2, 4], &[1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
        .unwrap()
        .cast(MlxDType::BF16)
        .unwrap();
    let v = Array::from_f32(&s, &[1, 2, 2, 4], &[4., 4., 4., 4., 2., 2., 2., 2., 8., 8., 8., 8., 1., 1., 1., 1.])
        .unwrap()
        .cast(MlxDType::BF16)
        .unwrap();
    let sinks = Array::from_f32(&s, &[2], &[50.0, -50.0])
        .unwrap()
        .cast(MlxDType::BF16)
        .unwrap();
    let out = q
        .sdpa_with_sinks(&k, &v, 0.5, false, None, Some(&sinks))
        .unwrap()
        .to_f32_vec()
        .unwrap();
    // Head 0 has sink=50 -> output near 0.
    assert!(out[..4].iter().all(|&x| x.abs() < 1e-3), "head0={:?}", &out[..4]);
    // Head 1 has sink=-50 -> standard attention over [8, 1] with scores [0.5, 0.0].
    let w0 = 0.5f32.exp() / (0.5f32.exp() + 1.0);
    let exp1 = w0 * 8.0 + (1.0 - w0) * 1.0;
    assert!((out[4] - exp1).abs() < 0.1, "head1={} expected={exp1}", out[4]);

    // MetalKernel with atomic_outputs=true and init_value=0.0.
    let kern = MetalKernel::new_with_options(
        &s,
        "apxinf_test_atomic_sum",
        &["x"],
        &["acc"],
        "uint i = thread_position_in_grid.x; atomic_fetch_add_explicit(&acc[0], x[i], memory_order_relaxed);",
        "",
        true,
        true,
    )
    .unwrap();
    let inp = Array::from_u32(&s, &[16], &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]).unwrap();
    let res = unsafe {
        kern.call_with_init(
            &[inp],
            &[(&[1], MlxDType::U32)],
            [16, 1, 1],
            [16, 1, 1],
            &[("N", TemplateArg::Int(16))],
            Some(0.0),
        )
        .unwrap()
    };
    assert_eq!(res[0].to_u32_vec().unwrap(), vec![136]);
}

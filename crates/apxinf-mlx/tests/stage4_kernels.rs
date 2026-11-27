#![cfg(all(feature = "native", target_os = "macos", target_arch = "aarch64"))]

use apxinf_mlx::banded_attn::{is_nax_available, BandedAttnKernels};
use apxinf_mlx::exact_gemv::{ExactGemvKernels, VerifyHeadScreen};
use apxinf_mlx::loop_glue::{ConvFusedEpilogue, LoopGlueKernels};
use apxinf_mlx::moe_qmv::MoeKernels;
use apxinf_mlx::{Array, MlxDType, QuantMode, Stream};

fn det_f32(stream: &Stream, shape: &[usize], seed: u32, scale: f32) -> Array {
    let n: usize = shape.iter().product();
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(0x9e3779b9);
    let mut vals = Vec::with_capacity(n);
    for _ in 0..n {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        let u = ((s & 0xffff) as f32) / 65536.0 - 0.5;
        vals.push(u * scale);
    }
    Array::from_f32(stream, shape, &vals).unwrap()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let av = a.cast(MlxDType::F32).unwrap().to_f32_vec().unwrap();
    let bv = b.cast(MlxDType::F32).unwrap().to_f32_vec().unwrap();
    assert_eq!(av.len(), bv.len());
    av.iter()
        .zip(bv.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

#[test]
fn banded_attn_and_verify_exact_on_metal() {
    let stream = Stream::metal(0).unwrap();
    let k = BandedAttnKernels::new(stream.clone());
    assert!(k.xattn_xr_supported().unwrap());

    // 1. attn_res_mix: regs=true bit-equal to regs=false and matches reference
    let b0 = det_f32(&stream, &[1, 3, 2304], 1, 1.0);
    let b1 = det_f32(&stream, &[1, 3, 2304], 2, 1.0);
    let p = det_f32(&stream, &[1, 3, 2304], 3, 1.0);
    let q = det_f32(&stream, &[4, 2304], 4, 0.25);
    let r_base = k.attn_res_mix(&[b0.clone(), b1.clone()], &p, &q, 0.5, false).unwrap();
    let r_regs = k.attn_res_mix(&[b0, b1], &p, &q, 0.5, true).unwrap();
    assert!(r_base.array_equal(&r_regs).unwrap().to_bool_scalar().unwrap());

    // 2. flash_banded_attention & nax_full_attention vs masked_attention
    let (hq, hkv, d, l, s, r) = (24, 8, 128, 64, 128, 64);
    let qa = det_f32(&stream, &[1, hq, l, d], 10, 0.3).cast(MlxDType::BF16).unwrap();
    let ka = det_f32(&stream, &[1, hkv, s, d], 11, 0.3).cast(MlxDType::BF16).unwrap();
    let va = det_f32(&stream, &[1, hkv, s, d], 12, 0.3).cast(MlxDType::BF16).unwrap();
    let rela = det_f32(&stream, &[1, l, hq, r], 13, 0.2).cast(MlxDType::BF16).unwrap();
    let sinka = det_f32(&stream, &[1, l, hq], 14, 0.2).cast(MlxDType::BF16).unwrap();
    let scale = (d as f32).sqrt().recip();

    let ref_sw = k.masked_attention(&qa, &ka, &va, &rela, &sinka, scale, 64).unwrap();
    let fl_sw = k.flash_banded_attention(&qa, &ka, &va, &rela, &sinka, scale, 64).unwrap();
    assert!(max_abs_diff(&ref_sw, &fl_sw) < 2e-2, "flash sliding diff too large");

    let ref_fu = k.masked_attention(&qa, &ka, &va, &rela, &sinka, scale, 0).unwrap();
    if is_nax_available(&stream) {
        let nax_fu = k.nax_full_attention(&qa, &ka, &va, &rela, &sinka, scale).unwrap();
        assert!(max_abs_diff(&ref_fu, &nax_fu) < 2e-2, "nax full diff too large");
    }

    // 3. verify_attention_exact (rows_in_tg + xr + gate) vs per-row decode_attention
    let lv = 3usize;
    let base = 100usize;
    let sv = base + lv;
    let qv = det_f32(&stream, &[1, hq, lv, d], 20, 0.3).cast(MlxDType::BF16).unwrap();
    let kv = det_f32(&stream, &[1, hkv, sv, d], 21, 0.3).cast(MlxDType::BF16).unwrap();
    let vv = det_f32(&stream, &[1, hkv, sv, d], 22, 0.3).cast(MlxDType::BF16).unwrap();
    let relv = det_f32(&stream, &[1, lv, hq, r], 23, 0.2).cast(MlxDType::BF16).unwrap();
    let sinkv = det_f32(&stream, &[1, lv, hq], 24, 0.2).cast(MlxDType::BF16).unwrap();
    let gatev = det_f32(&stream, &[1, lv, hq * d], 25, 0.5).cast(MlxDType::BF16).unwrap();

    let ex_base = k
        .verify_attention_exact(&qv, &kv, &vv, &relv, &sinkv, scale, 0, base, 32, false, None, false)
        .unwrap();
    let ex_xr = k
        .verify_attention_exact(&qv, &kv, &vv, &relv, &sinkv, scale, 0, base, 32, true, None, true)
        .unwrap();
    assert!(ex_base.array_equal(&ex_xr).unwrap().to_bool_scalar().unwrap());

    let ex_gated = k
        .verify_attention_exact(&qv, &kv, &vv, &relv, &sinkv, scale, 0, base, 32, true, Some(&gatev), true)
        .unwrap();
    let ref_gated = ex_xr
        .transpose(&[0, 2, 1, 3])
        .unwrap()
        .reshape(&[1, lv, hq * d])
        .unwrap()
        .cast(MlxDType::F32)
        .unwrap()
        .mul(&gatev.cast(MlxDType::F32).unwrap().sigmoid().unwrap())
        .unwrap()
        .cast(MlxDType::BF16)
        .unwrap();
    assert!(ex_gated.array_equal(&ref_gated).unwrap().to_bool_scalar().unwrap());

    // Compare row 0 of verify_attention_exact against decode_attention at s = base + 1
    let dec0 = k
        .decode_attention(
            &qv.slice_axis(2, 0, 1).unwrap(),
            &kv.slice_axis(2, 0, base + 1).unwrap(),
            &vv.slice_axis(2, 0, base + 1).unwrap(),
            &relv.slice_axis(1, 0, 1).unwrap(),
            &sinkv.slice_axis(1, 0, 1).unwrap(),
            scale,
            0,
        )
        .unwrap();
    assert!(max_abs_diff(&ex_xr.slice_axis(2, 0, 1).unwrap(), &dec0) < 5e-3);
}

#[test]
fn moe_qmv_and_swiglu_kernels_on_metal() {
    let stream = Stream::metal(0).unwrap();
    let mk = MoeKernels::new(stream.clone());

    // 1. route_topk
    let logits = det_f32(&stream, &[3, 128], 30, 2.0);
    let bias = det_f32(&stream, &[128], 31, 0.5);
    let (inds, weights) = mk.route_topk(&logits, &bias, 8, 8.0).unwrap();
    assert_eq!(inds.shape(), &[3, 8]);
    assert_eq!(weights.shape(), &[3, 8]);
    let wsum = weights.sum(&[1], false).unwrap().to_f32_vec().unwrap();
    for s in wsum {
        assert!((s - 8.0).abs() < 1e-4, "route weight sum {s} != 8.0");
    }

    // 2. NVFP4 gather_qmv_dedup & gather_qmv_swiglu vs gather_qmm & moe_swiglu_scaled
    let (e, out_n, k_dim) = (16usize, 64usize, 256usize);
    let w_u8: Vec<u8> = (0..(e * out_n * (k_dim / 2))).map(|i| ((i * 37 + 13) & 0xff) as u8).collect();
    let w_u32 = Array::from_u8(&stream, &[e, out_n, k_dim / 2], &w_u8)
        .unwrap()
        .view(MlxDType::U32)
        .unwrap();
    let sc_u8: Vec<u8> = (0..(e * out_n * (k_dim / 16))).map(|i| 0x38 + ((i % 7) as u8)).collect();
    let sc = Array::from_u8(&stream, &[e, out_n, k_dim / 16], &sc_u8).unwrap();
    let x = det_f32(&stream, &[3, k_dim], 32, 0.5).cast(MlxDType::BF16).unwrap();
    let idx = Array::from_u32(&stream, &[3, 4], &[0, 3, 7, 11, 3, 5, 11, 15, 0, 2, 7, 9]).unwrap();
    let gs = det_f32(&stream, &[e], 33, 1.0).abs().unwrap();
    let c = Array::from_f32(&stream, &[1], &[1.0 / 2688.0]).unwrap();
    let gcap = Array::from_f32(&stream, &[1], &[4.0]).unwrap();
    let ucap = Array::from_f32(&stream, &[1], &[25.0]).unwrap();

    assert!(MoeKernels::dedup_qmv_supported(QuantMode::Nvfp4, &w_u32, 4, &x));
    let y_dedup = mk.gather_qmv_dedup(&x, &w_u32, &sc, &idx, 4, 16, false).unwrap();
    let y_ref = x
        .reshape(&[3, 1, 1, k_dim])
        .unwrap()
        .gather_qmm(&w_u32, &sc, None, None, Some(&idx), true, 16, 4, QuantMode::Nvfp4, None, false)
        .unwrap();
    assert!(y_dedup.array_equal(&y_ref).unwrap().to_bool_scalar().unwrap());

    let act_fused = mk
        .gather_qmv_swiglu(&x, &w_u32, &sc, &idx, 4, 16, Some(&gs), Some(&c), &gcap, &ucap)
        .unwrap();
    let act_unfused = mk
        .moe_swiglu_scaled(&y_dedup, &idx, Some(&gs), Some(&c), &gcap, &ucap)
        .unwrap();
    let act_v4 = mk
        .prefill_moe_swiglu_v4(&y_dedup, &idx, Some(&gs), Some(&c), &gcap, &ucap)
        .unwrap();
    assert!(act_fused.array_equal(&act_unfused).unwrap().to_bool_scalar().unwrap());
    assert!(act_v4.array_equal(&act_unfused).unwrap().to_bool_scalar().unwrap());

    // 3. moe_combine_rows & moe_combine_sorted
    let z = det_f32(&stream, &[3, 4, 1, 64], 34, 0.5).cast(MlxDType::BF16).unwrap();
    let rw = det_f32(&stream, &[3, 4], 35, 1.0).abs().unwrap();
    let comb_rows = mk.moe_combine_rows(&z, &rw, &idx, Some(&gs), Some(&c)).unwrap();
    let inv = Array::arange(&stream, 0.0, 12.0, 1.0, MlxDType::U32).unwrap();
    let s2 = gs.take(&idx.reshape(&[12]).unwrap(), 0).unwrap().mul(&c).unwrap().reshape(&[3, 4]).unwrap();
    let comb_sorted = mk.moe_combine_sorted(&z.reshape(&[12, 1, 64]).unwrap(), &inv, &rw, Some(&s2)).unwrap();
    assert!(comb_rows.array_equal(&comb_sorted).unwrap().to_bool_scalar().unwrap());
}

#[test]
fn exact_gemv_loopmix_and_screen_argmax_on_metal() {
    let stream = Stream::metal(0).unwrap();
    let gk = ExactGemvKernels::new(stream.clone());

    // 1. multirow_gemv & multirow_gemv_router16
    let w_bf16 = det_f32(&stream, &[128, 256], 40, 0.5).cast(MlxDType::BF16).unwrap();
    let w_f32 = w_bf16.cast(MlxDType::F32).unwrap();
    let x_bf16 = det_f32(&stream, &[3, 256], 41, 0.5).cast(MlxDType::BF16).unwrap();

    let mr_bf16 = gk.multirow_gemv(&w_bf16, &x_bf16, false).unwrap();
    let r0 = x_bf16.slice_axis(0, 0, 1).unwrap().matmul(&w_bf16.transpose(&[1, 0]).unwrap()).unwrap();
    assert!(mr_bf16.slice_axis(0, 0, 1).unwrap().array_equal(&r0).unwrap().to_bool_scalar().unwrap());

    let mr_f32 = gk.multirow_gemv(&w_f32, &x_bf16, true).unwrap();
    let mr_r16 = gk.multirow_gemv_router16(&w_bf16, &x_bf16).unwrap();
    assert!(mr_f32.array_equal(&mr_r16).unwrap().to_bool_scalar().unwrap());

    // 2. wide_gemv
    let w_wide = det_f32(&stream, &[256, 512], 42, 0.25).cast(MlxDType::BF16).unwrap();
    let x_wide = det_f32(&stream, &[9, 512], 43, 0.25).cast(MlxDType::BF16).unwrap();
    let y_wide = gk.wide_gemv(&w_wide, &x_wide).unwrap();
    let y_wide_ref = x_wide.matmul(&w_wide.transpose(&[1, 0]).unwrap()).unwrap();
    assert!(max_abs_diff(&y_wide, &y_wide_ref) < 2e-2);

    // 3. loopmix_down_pair_perm vs loopmix_down_pair & loopmix_gates_mix vs unfused
    let (n, r, k_dim, rank) = (2usize, 3usize, 384usize, 64usize);
    let z = det_f32(&stream, &[2 * n, r, k_dim], 44, 0.5);
    let w_down = det_f32(&stream, &[2 * n, k_dim, rank], 45, 0.25).cast(MlxDType::BF16).unwrap();
    assert!(ExactGemvKernels::loopmix_perm_supported(&w_down));
    let perm = ExactGemvKernels::loopmix_perm_weight(&w_down).unwrap();
    let d_base = gk.loopmix_down_pair(&z, &w_down).unwrap();
    let d_perm = gk.loopmix_down_pair_perm(&z, &w_down, &perm).unwrap();
    assert!(d_base.array_equal(&d_perm).unwrap().to_bool_scalar().unwrap());

    let w_gates = det_f32(&stream, &[n, rank, 2 * k_dim], 46, 0.25).cast(MlxDType::BF16).unwrap();
    let x_in = det_f32(&stream, &[n, 1, r, k_dim], 47, 0.5);
    let msg_in = det_f32(&stream, &[n, 1, r, k_dim], 48, 0.5);
    let gm_fused = gk.loopmix_gates_mix(&d_perm, &w_gates, &x_in, &msg_in).unwrap();
    let gates = gk.loopmix_gemv_t(&d_perm, &w_gates).unwrap().reshape(&[n, 1, r, 2 * k_dim]).unwrap();
    let gain = gates.slice_axis(3, 0, k_dim).unwrap().tanh().unwrap();
    let forget = gates.slice_axis(3, k_dim, 2 * k_dim).unwrap().tanh().unwrap();
    let one = Array::scalar(&stream, 1.0, MlxDType::F32).unwrap();
    let gm_ref = one.sub(&forget).unwrap().mul(&x_in).unwrap().add(&gain.mul(&msg_in).unwrap()).unwrap();
    assert!(gm_fused.array_equal(&gm_ref).unwrap().to_bool_scalar().unwrap());

    // 4. argmax_rows & screen_argmax
    let (v, vu, h) = (1024usize, 1000usize, 256usize);
    let head_w = det_f32(&stream, &[v, h], 49, 0.5).cast(MlxDType::BF16).unwrap();
    let h_in = det_f32(&stream, &[3, h], 50, 0.5).cast(MlxDType::BF16).unwrap();
    let [wq, sc, bi] = head_w.quantize(64, 8).unwrap();
    let deq = wq.dequantize(&sc, &bi, 64, 8).unwrap();
    let diff = head_w.cast(MlxDType::F32).unwrap().sub(&deq.cast(MlxDType::F32).unwrap()).unwrap().abs().unwrap();
    let err = diff
        .reshape(&[v, h / 64, 64])
        .unwrap()
        .max(2)
        .unwrap()
        .mul(&Array::scalar(&stream, 1.01, MlxDType::F32).unwrap())
        .unwrap()
        .cast(MlxDType::BF16)
        .unwrap();
    let max_w = head_w.cast(MlxDType::F32).unwrap().abs().unwrap().max(1).unwrap().max(0).unwrap().to_f32_vec().unwrap()[0];
    let ca = Array::from_f32(&stream, &[1], &[128.0 * (2.0f32).powi(-24) * max_w]).unwrap();
    // Identity softcap table (just bf16 -> f32) + monotone ptab
    let mut ttab_v = vec![0.0f32; 65536];
    let mut ptab_v = vec![0.0f32; 65536];
    for u in 0u32..65536 {
        let f = f32::from_bits(u << 16);
        ttab_v[u as usize] = f;
        ptab_v[u as usize] = if f.is_finite() { f - 1e-3 } else { f };
    }
    let ttab = Array::from_f32(&stream, &[65536], &ttab_v).unwrap();
    let ptab = Array::from_f32(&stream, &[65536], &ptab_v).unwrap();
    let mut bias_v = vec![0.0f32; v];
    for b in &mut bias_v[vu..] {
        *b = f32::NEG_INFINITY;
    }
    let bias = Array::from_f32(&stream, &[v], &bias_v).unwrap();
    let screen = VerifyHeadScreen { wq, sc, bi, err, ca, ttab: ttab.clone(), ptab, bias: bias.clone(), unpadded: vu };

    let raw = gk.multirow_gemv(&head_w, &h_in, false).unwrap();
    let ref_ids = gk.argmax_rows(&raw, Some(&ttab), Some(&bias)).unwrap().to_u32_vec().unwrap();
    let scr_ids = gk.screen_argmax(&head_w, &h_in, &screen).unwrap().to_i32_vec().unwrap();
    for (r_id, s_id) in ref_ids.iter().zip(scr_ids.iter()) {
        assert_eq!(*s_id, *r_id as i32);
    }
}

#[test]
fn loop_glue_kernels_on_metal() {
    let stream = Stream::metal(0).unwrap();
    let lk = LoopGlueKernels::new(stream.clone());

    // 1. l2_clip_fused (D=128 with tau, D=16)
    let x128 = det_f32(&stream, &[1, 3, 8, 128], 60, 2.0).cast(MlxDType::BF16).unwrap();
    let sc128 = det_f32(&stream, &[8, 128], 61, 1.0).abs().unwrap();
    let thr = Array::from_f32(&stream, &[1], &[1.0]).unwrap();
    let tau = Array::from_f32(&stream, &[3], &[1.05, 1.10, 1.15]).unwrap();
    let c128 = lk.l2_clip_fused(&x128, &sc128, &thr, Some(&tau)).unwrap();
    assert_eq!(c128.shape(), &[1, 3, 8, 128]);
    let x16 = det_f32(&stream, &[1, 3, 24, 16], 62, 2.0).cast(MlxDType::BF16).unwrap();
    let sc1 = Array::from_f32(&stream, &[1], &[1.0]).unwrap();
    let c16 = lk.l2_clip_fused(&x16, &sc1, &thr, None).unwrap();
    assert_eq!(c16.shape(), &[1, 3, 24, 16]);

    // 2. rms_norm_fused bit-equal to fast_rms_norm
    let xr = det_f32(&stream, &[1, 4, 2304], 63, 1.0);
    let wr = det_f32(&stream, &[2304], 64, 1.0).cast(MlxDType::BF16).unwrap();
    let r_fused = lk.rms_norm_fused(&xr, Some(&wr), 1e-6, MlxDType::BF16, None).unwrap();
    let r_ref = xr.fast_rms_norm(&wr.cast(MlxDType::F32).unwrap(), 1e-6).unwrap().cast(MlxDType::BF16).unwrap();
    assert!(r_fused.array_equal(&r_ref).unwrap().to_bool_scalar().unwrap());

    // 3. conv_fused vs prefill_static_conv & prefill_dynamic_conv
    let xc = det_f32(&stream, &[1, 3, 256], 65, 1.0).cast(MlxDType::BF16).unwrap();
    let s0 = det_f32(&stream, &[1, 1, 256], 66, 1.0);
    let s1 = det_f32(&stream, &[1, 1, 256], 67, 1.0);
    let s2 = det_f32(&stream, &[1, 1, 256], 68, 1.0);
    let st_arr = Array::concat(&[&s0, &s1, &s2], 1).unwrap();
    let sw = det_f32(&stream, &[256, 4], 69, 0.5).cast(MlxDType::BF16).unwrap();
    let (cy_f, cxp_f) = lk.conv_fused(&xc, &sw, &[s0.clone(), s1.clone(), s2.clone()], 0.7071, None, 1, ConvFusedEpilogue::None).unwrap();
    let (cy_p, cst_p) = lk.prefill_static_conv(&xc, &sw, &st_arr, 0.7071).unwrap();
    assert!(cy_f.array_equal(&cy_p).unwrap().to_bool_scalar().unwrap());
    assert!(cxp_f.slice_axis(1, 3, 6).unwrap().array_equal(&cst_p).unwrap().to_bool_scalar().unwrap());

    // Dynamic conv
    let dw = det_f32(&stream, &[4, 256], 70, 0.5);
    let dyn_taps = det_f32(&stream, &[1, 3, 8], 71, 0.5).cast(MlxDType::BF16).unwrap();
    let (dy_f, _) = lk.conv_fused(&xc, &dw, &[s0, s1, s2], 0.7071, Some(&dyn_taps), 2, ConvFusedEpilogue::None).unwrap();
    let (dy_p, _) = lk.prefill_dynamic_conv(&xc, &dyn_taps, &dw, &st_arr, 2, 0.7071).unwrap();
    assert!(dy_f.array_equal(&dy_p).unwrap().to_bool_scalar().unwrap());

    // 4. prefill_mix_head_opt vs prefill_mix_head, and prefill_mix_tail_rms vs prefill_mix_tail + rms_norm_fused
    let x_single = det_f32(&stream, &[1, 1, 16, 256], 72, 1.0);
    let x_bcast = x_single.broadcast_to(&[3, 1, 16, 256]).unwrap();
    let msg = det_f32(&stream, &[3, 1, 16, 256], 73, 1.0);
    let mh_base = lk.prefill_mix_head(&x_bcast, &msg, 1e-6).unwrap();
    let mh_opt = lk.prefill_mix_head_opt(&x_bcast, &msg, 1e-6, true).unwrap();
    assert!(mh_base.array_equal(&mh_opt).unwrap().to_bool_scalar().unwrap());

    let gates = det_f32(&stream, &[3, 16, 512], 74, 0.5);
    let nw0 = det_f32(&stream, &[256], 75, 1.0).cast(MlxDType::BF16).unwrap();
    let nw1 = det_f32(&stream, &[256], 76, 1.0).cast(MlxDType::BF16).unwrap();
    let nw2 = det_f32(&stream, &[256], 77, 1.0).cast(MlxDType::BF16).unwrap();
    let mt_rms = lk.prefill_mix_tail_rms(&gates, &x_bcast, &msg, &nw0, 1e-6, MlxDType::BF16, Some((&nw1, &nw2))).unwrap();
    let mt_only = lk.prefill_mix_tail(&gates, &x_bcast, &msg).unwrap();
    let r0 = lk.rms_norm_fused(&mt_only.slice_axis(0, 0, 1).unwrap(), Some(&nw0), 1e-6, MlxDType::BF16, None).unwrap();
    let r1 = lk.rms_norm_fused(&mt_only.slice_axis(0, 1, 2).unwrap(), Some(&nw1), 1e-6, MlxDType::BF16, None).unwrap();
    let r2 = lk.rms_norm_fused(&mt_only.slice_axis(0, 2, 3).unwrap(), Some(&nw2), 1e-6, MlxDType::BF16, None).unwrap();
    let mt_ref = Array::concat(&[&r0, &r1, &r2], 0).unwrap();
    assert!(mt_rms.array_equal(&mt_ref).unwrap().to_bool_scalar().unwrap());

    // 5. prefill_gate_out, prefill_residual_rms, prefill_residual_out
    let attn = det_f32(&stream, &[1, 4, 16, 64], 78, 0.5).cast(MlxDType::BF16).unwrap();
    let gate = det_f32(&stream, &[1, 16, 256], 79, 0.5).cast(MlxDType::BF16).unwrap();
    let go = lk.prefill_gate_out(&attn, &gate).unwrap();
    let go_ref = attn
        .transpose(&[0, 2, 1, 3])
        .unwrap()
        .reshape(&[1, 16, 256])
        .unwrap()
        .cast(MlxDType::F32)
        .unwrap()
        .mul(&gate.cast(MlxDType::F32).unwrap().sigmoid().unwrap())
        .unwrap()
        .cast(MlxDType::BF16)
        .unwrap();
    assert!(go.array_equal(&go_ref).unwrap().to_bool_scalar().unwrap());

    let h = det_f32(&stream, &[1, 16, 256], 80, 0.5).cast(MlxDType::BF16).unwrap();
    let a = det_f32(&stream, &[1, 16, 256], 81, 0.5).cast(MlxDType::BF16).unwrap();
    let (res, nrm) = lk.prefill_residual_rms(&h, &a, false, 1e-6).unwrap();
    let res_ref = h.cast(MlxDType::F32).unwrap().add(&a.cast(MlxDType::F32).unwrap()).unwrap();
    let nrm_ref = res_ref.fast_rms_norm_unweighted(1e-6).unwrap();
    assert!(res.array_equal(&res_ref).unwrap().to_bool_scalar().unwrap());
    assert!(nrm.array_equal(&nrm_ref).unwrap().to_bool_scalar().unwrap());

    let out_h = lk.prefill_residual_out(&res, &a, MlxDType::BF16).unwrap();
    let out_ref = res.add(&a.cast(MlxDType::F32).unwrap()).unwrap().cast(MlxDType::BF16).unwrap();
    assert!(out_h.array_equal(&out_ref).unwrap().to_bool_scalar().unwrap());
}

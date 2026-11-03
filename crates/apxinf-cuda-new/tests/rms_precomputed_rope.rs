use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::ops::rope::apply_precomputed;
use apxinf_cuda_new::{
    capture, norm, rms_norm_apply_precomputed, CudaBuffer, CudaContext, NormArgs,
};
use half::bf16;
fn tensor(d: &[usize], v: &[f32], dtype: DType, offset: usize) -> Tensor {
    let bytes: Vec<u8> = if dtype == DType::BF16 {
        v.iter()
            .flat_map(|&x| bf16::from_f32(x).to_bits().to_ne_bytes())
            .collect()
    } else {
        v.iter().flat_map(|x| x.to_ne_bytes()).collect()
    };
    let b = CudaBuffer::alloc(bytes.len() + offset, 0)
        .unwrap()
        .view(offset, bytes.len())
        .unwrap();
    b.copy_from_host(&bytes).unwrap();
    b.as_tensor(Shape::new(d.to_vec()), dtype).unwrap()
}
fn read(t: &Tensor) -> Vec<u16> {
    let mut v = vec![0; t.size_in_bytes()];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut v)
        .unwrap();
    v.chunks_exact(2)
        .map(|b| u16::from_ne_bytes(b.try_into().unwrap()))
        .collect()
}
fn round(x: f32) -> f32 {
    bf16::from_f32(x).to_f32()
}
fn unfused(ctx: &CudaContext, x: &Tensor, w: &Tensor, t: &Tensor, out: &Tensor) {
    let d = x.shape().dims();
    let rows = d[0] * d[1];
    norm(
        ctx,
        NormArgs::rms(
            &x.reshape(vec![rows, d[2]]).unwrap(),
            w,
            &mut out.reshape(vec![rows, d[2]]).unwrap(),
            1e-6,
        ),
    )
    .unwrap();
    apply_precomputed(ctx, out, t, out).unwrap();
}
#[test]
fn rms_rope_matches_unfused_cpu_alias_and_retained_graph() {
    let ctx = CudaContext::new(0).unwrap();
    let mut cases = vec![
        (1, 1, 2, 7),
        (3, 2, 64, 7),
        (7, 3, 128, 7),
        (3, 2, 130, 7),
        (3, 1, 192, 7),
        (1764, 16, 128, 7),
        (1764, 16, 128, 0),
        (1023, 1, 128, 0),
        (1024, 1, 128, 0),
    ];
    // Each pointer may independently disable the vector path. 1025 also
    // exercises the final subgroup after an exact 1024-row threshold.
    cases.extend((0..8).map(|mask| (1025, 1, 128, mask)));
    for (tokens, heads, dim, mask) in cases {
        let n = tokens * heads * dim;
        let shape = [tokens, heads, dim];
        let mut values: Vec<_> = (0..n)
            .map(|i| round((i as i32 * 37 % 1009 - 504) as f32 / 127.))
            .collect();
        if tokens == 7 || tokens == 1025 {
            for i in 0..dim {
                values[i] = if i % 2 == 0 { 0.0 } else { -0.0 };
            }
        }
        let weights: Vec<_> = (0..dim)
            .map(|i| round(0.5 + (i * 13 % 37) as f32 / 31.))
            .collect();
        let table: Vec<_> = (0..tokens * dim)
            .map(|i| {
                let c = i % dim;
                let angle = ((i / dim) * 17 + c % (dim / 2)) as f32 / 71.;
                if c < dim / 2 {
                    angle.cos()
                } else {
                    angle.sin()
                }
            })
            .collect();
        let x = tensor(
            &shape,
            &values,
            DType::BF16,
            if mask & 1 != 0 { 2 } else { 0 },
        );
        let w = tensor(
            &[dim],
            &weights,
            DType::BF16,
            if mask & 2 != 0 { 2 } else { 0 },
        );
        let t = tensor(&[tokens, 2, dim / 2], &table, DType::F32, 4);
        let out = tensor(
            &shape,
            &vec![0.; n],
            DType::BF16,
            if mask & 4 != 0 { 2 } else { 0 },
        );
        let expected = tensor(&shape, &vec![0.; n], DType::BF16, 0);
        unfused(&ctx, &x, &w, &t, &expected);
        rms_norm_apply_precomputed(&ctx, &x, &w, &t, 1e-6, &out).unwrap();
        ctx.synchronize().unwrap();
        let reference = read(&expected);
        assert_eq!(read(&out), reference, "{shape:?}");
        // Independent CPU formula checks the staged mathematical contract. The exact
        // comparison above additionally locks the native reduction's rounding order.
        for row in 0..tokens * heads {
            let base = row * dim;
            let inv = (values[base..base + dim]
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum::<f64>()
                / dim as f64
                + 1e-6)
                .sqrt()
                .recip() as f32;
            for c in 0..dim / 2 {
                let a = round(values[base + c] * inv * weights[c]);
                let b = round(values[base + c + dim / 2] * inv * weights[c + dim / 2]);
                let ti = row / heads * dim + c;
                let co = table[ti];
                let si = table[ti + dim / 2];
                for (col, v) in [
                    (c, round(a * co + b * (-si))),
                    (c + dim / 2, round(b * co + a * si)),
                ] {
                    let got = bf16::from_bits(reference[base + col]).to_f32();
                    assert!((got - v).abs() <= 0.04, "cpu {shape:?} {got} {v}");
                }
            }
        }
        let graph = capture(&ctx, || {
            rms_norm_apply_precomputed(&ctx, &x, &w, &t, 1e-6, &out)
        })
        .unwrap();
        // Change all bindings' contents, then remove their caller-owned Tensor leases.
        let new_values: Vec<_> = values.iter().map(|&x| -x).collect();
        let new_weight: Vec<_> = weights.iter().map(|&x| round(x + 0.125)).collect();
        let new_table: Vec<_> = table.iter().map(|&x| -x).collect();
        for (dst, src) in [
            (&x, tensor(&shape, &new_values, DType::BF16, 0)),
            (&w, tensor(&[dim], &new_weight, DType::BF16, 0)),
            (&t, tensor(&[tokens, 2, dim / 2], &new_table, DType::F32, 0)),
        ] {
            let mut bytes = vec![0; src.size_in_bytes()];
            CudaBuffer::from_tensor(&src)
                .unwrap()
                .copy_to_host(&mut bytes)
                .unwrap();
            CudaBuffer::from_tensor(dst)
                .unwrap()
                .copy_from_host(&bytes)
                .unwrap();
        }
        unfused(&ctx, &x, &w, &t, &expected);
        ctx.synchronize().unwrap();
        let reference = read(&expected);
        rms_norm_apply_precomputed(&ctx, &x, &w, &t, 1e-6, &x).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(read(&x), reference);
        let bytes: Vec<_> = new_values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_bits().to_ne_bytes())
            .collect();
        CudaBuffer::from_tensor(&x)
            .unwrap()
            .copy_from_host(&bytes)
            .unwrap();
        drop(x);
        drop(w);
        drop(t);
        for _ in 0..3 {
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(read(&out), reference);
        }
    }
}
#[test]
fn rms_rope_rejects_malformed_shapes_epsilon_and_overlap() {
    let ctx = CudaContext::new(0).unwrap();
    let x = tensor(&[1, 1, 128], &vec![1.; 128], DType::BF16, 0);
    let w = tensor(&[128], &vec![1.; 128], DType::BF16, 0);
    let t = tensor(&[1, 2, 64], &vec![0.5; 128], DType::F32, 0);
    let out = tensor(&[1, 1, 128], &vec![0.; 128], DType::BF16, 0);
    for eps in [0., -1., f32::NAN, f32::INFINITY] {
        assert!(rms_norm_apply_precomputed(&ctx, &x, &w, &t, eps, &out).is_err());
    }
    assert!(rms_norm_apply_precomputed(
        &ctx,
        &x,
        &w.reshape(vec![1, 128]).unwrap(),
        &t,
        1e-6,
        &out
    )
    .is_err());
    let alias = w.reshape(vec![1, 1, 128]).unwrap();
    assert!(rms_norm_apply_precomputed(&ctx, &x, &w, &t, 1e-6, &alias).is_err());
    let b = CudaBuffer::alloc(258, 0).unwrap();
    let a = b
        .view(0, 256)
        .unwrap()
        .as_tensor(Shape::new(vec![1, 1, 128]), DType::BF16)
        .unwrap();
    let o = b
        .view(2, 256)
        .unwrap()
        .as_tensor(a.shape().clone(), DType::BF16)
        .unwrap();
    assert!(rms_norm_apply_precomputed(&ctx, &a, &w, &t, 1e-6, &o).is_err());
    let o = CudaBuffer::from_tensor(&t)
        .unwrap()
        .view(0, 256)
        .unwrap()
        .as_tensor(x.shape().clone(), DType::BF16)
        .unwrap();
    assert!(rms_norm_apply_precomputed(&ctx, &x, &w, &t, 1e-6, &o).is_err());
}

#[test]
fn rms_rope_packed_special_values_match_unfused_bits() {
    let ctx = CudaContext::new(0).unwrap();
    let shape = [1025, 1, 128];
    let mut data = vec![0.25; 1025 * 128];
    for (row, bits) in [
        0u16, 0x8000, 1, 0x8001, 0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc0, 0xffc1,
    ]
    .into_iter()
    .enumerate()
    {
        data[row * 128] = bf16::from_bits(bits).to_f32();
    }
    let x = tensor(&shape, &data, DType::BF16, 0);
    let w = tensor(&[128], &vec![0.5; 128], DType::BF16, 0);
    let t = tensor(&[1025, 2, 64], &vec![0.75; 1025 * 128], DType::F32, 4);
    let out = tensor(&shape, &vec![0.; data.len()], DType::BF16, 0);
    let expected = tensor(&shape, &vec![0.; data.len()], DType::BF16, 0);
    unfused(&ctx, &x, &w, &t, &expected);
    rms_norm_apply_precomputed(&ctx, &x, &w, &t, 1e-6, &out).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(read(&out), read(&expected));
    rms_norm_apply_precomputed(&ctx, &x, &w, &t, 1e-6, &x).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(read(&x), read(&expected));
}

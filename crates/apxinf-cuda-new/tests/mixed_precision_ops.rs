//! Mixed-precision boundaries required by video diffusion, independently of a model.
use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::*;
use half::bf16;
fn f32_tensor(values: &[f32]) -> Tensor {
    let buffer = CudaBuffer::alloc(values.len() * 4, 0).unwrap();
    buffer
        .copy_from_host(
            &values
                .iter()
                .flat_map(|v| v.to_ne_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    buffer
        .as_tensor(Shape::new(vec![1, values.len()]), DType::F32)
        .unwrap()
}
fn read(t: &Tensor) -> Vec<f32> {
    let mut b = vec![0; t.size_in_bytes()];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut b)
        .unwrap();
    if t.dtype() == DType::F32 {
        b.chunks_exact(4)
            .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
            .collect()
    } else {
        b.chunks_exact(2)
            .map(|b| bf16::from_bits(u16::from_ne_bytes(b.try_into().unwrap())).to_f32())
            .collect()
    }
}
fn bf_tensor(shape: Vec<usize>, values: &[f32]) -> Tensor {
    let b = CudaBuffer::alloc(values.len() * 2, 0).unwrap();
    b.copy_from_host(
        &values
            .iter()
            .flat_map(|v| bf16::from_f32(*v).to_bits().to_ne_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    b.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}
#[test]
fn recurrence_preserves_f64_state_and_graph_ownership() {
    let ctx = CudaContext::new(0).unwrap();
    let initial = f32_tensor(&[16777216., 1., -3., 1e-10]);
    let values = f32_tensor(&[1., 0.25, -0.5, 1e-11]);
    let out = f32_tensor(&[0.; 4]);
    let state = F64RecurrenceState::new(&ctx, 4).unwrap();
    let reset = state.prepare_reset(&ctx, &initial, &out).unwrap();
    let coefficients = RecurrenceCoefficients {
        a: 1.,
        b: 1.,
        c: 0.25,
        d: 1.,
    };
    let update = state
        .prepare_update(&ctx, &values, &out, coefficients)
        .unwrap();
    let graph = capture(&ctx, || {
        reset.enqueue()?;
        for _ in 0..9 {
            update.enqueue()?;
        }
        Ok(())
    })
    .unwrap();
    drop(state);
    drop(reset);
    drop(update);
    drop(values);
    for _ in 0..3 {
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        let actual = read(&out);
        for (i, (&start, &value)) in [16777216f32, 1., -3., 1e-10]
            .iter()
            .zip([1f32, 0.25, -0.5, 1e-11].iter())
            .enumerate()
        {
            let mut x = start as f64;
            let mut previous = 0.;
            for _ in 0..9 {
                x = x + (value as f64 + 0.25 * previous);
                previous = value as f64;
            }
            assert_eq!(actual[i], x as f32);
        }
        assert!(actual[0] > 16777216.); // FP32 state would lose early +1 updates.
    }
    assert!(F64RecurrenceState::new(&ctx, 0).is_err());
}
#[test]
fn indexed_mean_noncontiguous_input_and_retained_graph() {
    let ctx = CudaContext::new(0).unwrap();
    let x: Vec<f32> = (0..33).map(|i| i as f32 - 15.).collect();
    let input = f32_tensor(&x);
    let out = f32_tensor(&[0.; 3]);
    let ids = [32, 0, 10, 31, 2, 15, 30, 4, 20, 29, 6, 25];
    let op = PreparedIndexedMean::new(&ctx, &input, &out, &ids).unwrap();
    let graph = capture(&ctx, || op.enqueue()).unwrap();
    drop(op);
    for _ in 0..3 {
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(read(&out), [15.5, -12., 2.5]);
    }
    assert!(PreparedIndexedMean::new(&ctx, &input, &out, &[33, 0, 1]).is_err());
    assert!(PreparedIndexedMean::new(&ctx, &input, &input, &[0; 33]).is_err());
}
#[test]
fn staged_rgb_preserves_every_byte_rounding_boundary() {
    let ctx = CudaContext::new(0).unwrap();
    let bytes: Vec<u8> = (0..16 * 16 * 3).map(|i| (i % 256) as u8).collect();
    let input = CudaBuffer::alloc(bytes.len(), 0).unwrap();
    input.copy_from_host(&bytes).unwrap();
    let mut out = bf_tensor(vec![256, 3], &vec![0.; 768]);
    let mut args = GatherArgs::rgb_to_patches(
        &input,
        &mut out,
        GatherPatchGeometry {
            views: 1,
            image_size: 16,
            patch_size: 1,
            nhwc: true,
        },
    );
    args.semantic = GatherSemantic::RgbToPatchesStaged;
    gather(&ctx, args).unwrap();
    ctx.synchronize().unwrap();
    let expected: Vec<_> = bytes
        .iter()
        .map(|&b| bf16::from_f32(bf16::from_f32(b as f32 / 127.5).to_f32() - 1.).to_f32())
        .collect();
    assert_eq!(read(&out), expected);
}
#[test]
fn channel_l2_uses_clamp_and_staged_bf16_arithmetic() {
    let ctx = CudaContext::new(0).unwrap();
    for width in [3, 96, 192, 384] {
        let values: Vec<f32> = (0..3 * width)
            .map(|i| if i < width { 0. } else { (i % 7) as f32 - 3. })
            .collect();
        let weights: Vec<f32> = (0..width).map(|i| 0.5 + (i % 3) as f32 * 0.25).collect();
        let input = bf_tensor(vec![3, width], &values);
        let weight = bf_tensor(vec![width], &weights);
        let mut out = bf_tensor(vec![3, width], &vec![0.; values.len()]);
        channel_l2_norm(
            &ctx,
            ChannelL2NormArgs {
                input: &input,
                weight: &weight,
                out: &mut out,
                eps: 1e-12,
            },
        )
        .unwrap();
        ctx.synchronize().unwrap();
        let actual = read(&out);
        for (row, chunk) in values.chunks_exact(width).enumerate() {
            let norm = bf16::from_f32(chunk.iter().map(|x| x * x).sum::<f32>().sqrt())
                .to_f32()
                .max(bf16::from_f32(1e-12).to_f32());
            for c in 0..width {
                let divided = bf16::from_f32(chunk[c] / norm).to_f32();
                let scaled = bf16::from_f32(divided * (width as f32).sqrt()).to_f32();
                assert_eq!(
                    actual[row * width + c],
                    bf16::from_f32(scaled * weights[c]).to_f32()
                );
            }
        }
    }
}
#[test]
fn fp32_multiply_and_euler_validate_shapes_and_partial_alias() {
    let ctx = CudaContext::new(0).unwrap();
    let a = f32_tensor(&[1., 2., 3., 4.]);
    let b = f32_tensor(&[2., 3., 4., 5.]);
    let mut out = f32_tensor(&[0.; 4]);
    let mut args = PointwiseArgs::new(PointwiseSemantic::Multiply, &a, &mut out);
    args.secondary = Some(&b);
    pointwise(&ctx, args).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(read(&out), [2., 6., 12., 20.]);
    let mut args = PointwiseArgs::new(PointwiseSemantic::EulerUpdate, &a, &mut out);
    args.secondary = Some(&b);
    args.dt = 0.25;
    pointwise(&ctx, args).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(read(&out), [1.5, 2.75, 4., 5.25]);
    let wrong = b.reshape(vec![2, 2]).unwrap();
    let mut args = PointwiseArgs::new(PointwiseSemantic::Multiply, &a, &mut out);
    args.secondary = Some(&wrong);
    assert!(pointwise(&ctx, args).is_err());
    let buffer = CudaBuffer::alloc(20, 0).unwrap();
    let a = buffer
        .view(0, 16)
        .unwrap()
        .as_tensor(Shape::new(vec![1, 4]), DType::F32)
        .unwrap();
    let mut output = buffer
        .view(4, 16)
        .unwrap()
        .as_tensor(a.shape().clone(), DType::F32)
        .unwrap();
    for semantic in [PointwiseSemantic::Multiply, PointwiseSemantic::EulerUpdate] {
        let mut args = PointwiseArgs::new(semantic, &a, &mut output);
        args.secondary = Some(&b);
        assert!(pointwise(&ctx, args).is_err());
        let mut args = PointwiseArgs::new(semantic, &a, &mut out);
        args.secondary = Some(&wrong);
        assert!(pointwise(&ctx, args).is_err());
        let wrong_input = a.reshape(vec![2, 2]).unwrap();
        let mut args = PointwiseArgs::new(semantic, &wrong_input, &mut out);
        args.secondary = Some(&wrong);
        assert!(pointwise(&ctx, args).is_err());
    }
}

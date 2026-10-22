use super::*;
use crate::{transfers, CudaContext};
use apxinf_core::{DType, Shape, Tensor};
#[path = "precision/torch_bicubic_fixtures.rs"]
mod golden;

fn host(shape: Vec<usize>, x: &[f32], dtype: DType) -> Tensor {
    match dtype {
        DType::F32 => Tensor::from_f32(shape, x),
        DType::F16 => Tensor::from_f16(
            shape,
            &x.iter()
                .copied()
                .map(half::f16::from_f32)
                .collect::<Vec<_>>(),
        ),
        DType::BF16 => Tensor::from_bf16(
            shape,
            &x.iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        ),
        _ => unreachable!(),
    }
    .unwrap()
}
#[test]
fn bicubic_matches_torch_edges_batches_channels_and_replays() {
    let ctx = CudaContext::new(0).unwrap();
    for (dtype, expected) in [
        (DType::F32, golden::F32),
        (DType::F16, golden::F16),
        (DType::BF16, golden::BF16),
    ] {
        let source = host(vec![2, 3, 4, 2], &golden::INPUT, dtype);
        let input = transfers::to_cuda(&source, 0).unwrap();
        let mut output = ctx
            .allocate_output(Shape::new(vec![2, 5, 7, 2]), dtype)
            .unwrap();
        resize_bicubic(&ctx, &input, &mut output).unwrap();
        ctx.synchronize().unwrap();
        let actual = transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap();
        let relative = (actual
            .iter()
            .zip(expected)
            .map(|(a, b)| f64::from(*a - b).powi(2))
            .sum::<f64>()
            / expected.iter().map(|b| f64::from(*b).powi(2)).sum::<f64>())
        .sqrt();
        assert!(
            relative < if dtype == DType::F32 { 1e-6 } else { 0.005 },
            "{dtype:?} relative L2={relative}"
        );
        let graph = crate::capture(&ctx, || resize_bicubic(&ctx, &input, &mut output)).unwrap();
        ctx.synchronize().unwrap();
        transfers::copy_cpu_to_cuda(
            &host(vec![2, 3, 4, 2], &golden::INPUT.map(|x| -x), dtype),
            &input,
        )
        .unwrap();
        drop(input);
        for _ in 0..3 {
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap(),
                actual.iter().map(|x| -x).collect::<Vec<_>>()
            );
        }
    }
}
#[test]
fn bicubic_rejects_alias_and_mismatched_channels() {
    let ctx = CudaContext::new(0).unwrap();
    let input =
        transfers::to_cuda(&host(vec![2, 3, 4, 2], &golden::INPUT, DType::BF16), 0).unwrap();
    let mut alias = input.clone();
    assert!(resize_bicubic(&ctx, &input, &mut alias).is_err());
    let mut wrong = ctx
        .allocate_output(Shape::new(vec![2, 5, 7, 3]), DType::BF16)
        .unwrap();
    assert!(resize_bicubic(&ctx, &input, &mut wrong).is_err());
}

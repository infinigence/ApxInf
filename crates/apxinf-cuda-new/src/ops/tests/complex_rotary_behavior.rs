use super::*;
use crate::{transfers, CudaContext};
use apxinf_core::{DType, Shape, Tensor};
#[path = "precision/torch_complex_rotary_fixtures.rs"]
mod golden;

fn upload(ctx: &CudaContext, values: &[f32], dtype: DType) -> Tensor {
    let x = match dtype {
        DType::BF16 => Tensor::from_bf16(
            vec![3, 2, 8],
            &values
                .iter()
                .copied()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        ),
        DType::F16 => Tensor::from_f16(
            vec![3, 2, 8],
            &values
                .iter()
                .copied()
                .map(half::f16::from_f32)
                .collect::<Vec<_>>(),
        ),
        _ => unreachable!(),
    }
    .unwrap();
    transfers::to_cuda(&x, ctx.device_id()).unwrap()
}
#[test]
fn complex_rotary_matches_torch_fp64_and_retains_graph_bindings() {
    let ctx = CudaContext::new(0).unwrap();
    for (dtype, expected) in [(DType::F16, golden::F16), (DType::BF16, golden::BF16)] {
        let table = ComplexRotaryTable::new(&ctx, 3, 8, &golden::COEFFICIENTS).unwrap();
        let input = upload(&ctx, &golden::VALUES, dtype);
        let mut output = ctx
            .allocate_output(Shape::new(vec![3, 2, 8]), dtype)
            .unwrap();
        complex_rotary(&ctx, &table, &input, &mut output).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(
            transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap(),
            expected
        );
        let graph =
            crate::capture(&ctx, || complex_rotary(&ctx, &table, &input, &mut output)).unwrap();
        ctx.synchronize().unwrap();
        let changed = upload(&ctx, &golden::VALUES.map(|v| -v), dtype);
        transfers::copy_cuda_to_cuda(&ctx, &changed, &input).unwrap();
        ctx.synchronize().unwrap();
        drop(table);
        drop(input);
        drop(changed);
        for _ in 0..3 {
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(
                transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap(),
                expected.map(|v| -v)
            );
        }
    }
}
#[test]
fn complex_rotary_validates_shape_coefficients_and_partial_aliases() {
    let ctx = CudaContext::new(0).unwrap();
    assert!(ComplexRotaryTable::new(&ctx, 3, 8, &[0.; 23]).is_err());
    assert!(ComplexRotaryTable::new(&ctx, 3, 8, &[f64::NAN; 24]).is_err());
    assert!(ComplexRotaryTable::new(&ctx, usize::MAX, 8, &[]).is_err());
    let table = ComplexRotaryTable::new(&ctx, 3, 8, &golden::COEFFICIENTS).unwrap();
    let input = upload(&ctx, &golden::VALUES, DType::BF16);
    let mut in_place = input.clone();
    complex_rotary(&ctx, &table, &input, &mut in_place).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(
        transfers::to_cpu(&input).unwrap().to_f32_vec().unwrap(),
        golden::BF16
    );
    let backing = crate::CudaBuffer::alloc(100, 0).unwrap();
    let input = backing
        .view(0, 96)
        .unwrap()
        .as_tensor(Shape::new(vec![3, 2, 8]), DType::BF16)
        .unwrap();
    let mut overlap = backing
        .view(2, 96)
        .unwrap()
        .as_tensor(Shape::new(vec![3, 2, 8]), DType::BF16)
        .unwrap();
    assert!(complex_rotary(&ctx, &table, &input, &mut overlap)
        .unwrap_err()
        .to_string()
        .contains("overlapping"));
    let mut wrong = ctx
        .allocate_output(Shape::new(vec![3, 1, 8]), DType::BF16)
        .unwrap();
    assert!(complex_rotary(&ctx, &table, &input, &mut wrong).is_err());
}

#[test]
fn complex_rotary_accepts_the_declared_head_grid_limit() {
    let ctx = CudaContext::new(0).unwrap();
    let table = ComplexRotaryTable::new(&ctx, 1, 2, &[0., 1.]).unwrap();
    for heads in [65535, 65536] {
        let values: Vec<_> = (0..heads).flat_map(|_| [1., 2.]).collect();
        let host = Tensor::from_bf16(
            vec![1, heads, 2],
            &values
                .into_iter()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let input = transfers::to_cuda(&host, 0).unwrap();
        let mut output = ctx
            .allocate_output(Shape::new(vec![1, heads, 2]), DType::BF16)
            .unwrap();
        let result = complex_rotary(&ctx, &table, &input, &mut output);
        if heads == 65536 {
            assert!(result.is_err());
        } else {
            result.unwrap();
            ctx.synchronize().unwrap();
            assert!(transfers::to_cpu(&output)
                .unwrap()
                .to_f32_vec()
                .unwrap()
                .chunks_exact(2)
                .all(|pair| pair == [-2., 1.]));
        }
    }
}

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

#[test]
fn complex_rotary_d128_matches_fp64_path_all_encodings_and_graph_rebind() {
    let ctx = CudaContext::new(0).unwrap();
    let (tokens, heads, dim) = (86, 6, 128);
    let count = tokens * heads * dim;
    // D=256 remains the separate FP64 path. Pairing adjacent D=128 heads
    // and repeating their coefficient table gives an independent dispatch
    // oracle with exactly the same scalar complex products.
    for dtype in [DType::F16, DType::BF16] {
        let host = match dtype {
            DType::F16 => Tensor::from_f16(
                vec![tokens, heads, dim],
                &(0..count)
                    .map(|i| half::f16::from_bits(i as u16))
                    .collect::<Vec<_>>(),
            ),
            DType::BF16 => Tensor::from_bf16(
                vec![tokens, heads, dim],
                &(0..count)
                    .map(|i| half::bf16::from_bits(i as u16))
                    .collect::<Vec<_>>(),
            ),
            _ => unreachable!(),
        }
        .unwrap();
        let bits = |tensor: &Tensor| {
            let host = transfers::to_cpu(tensor).unwrap();
            match dtype {
                DType::F16 => host
                    .as_f16()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                DType::BF16 => host
                    .as_bf16()
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                _ => unreachable!(),
            }
        };
        for edges in [false, true] {
            let input = transfers::to_cuda(&host, 0).unwrap();
            let oracle_input = input.reshape(vec![tokens, heads / 2, 2 * dim]).unwrap();
            let coefficients: Vec<_> = (0..tokens * dim)
                .map(|i| {
                    if edges {
                        // Signed zeros, double-rounding boundaries, F32
                        // under/overflow and cancellation. All are finite.
                        [
                            0.,
                            -0.,
                            1.,
                            -1.,
                            1.0039062501,
                            1.0004882812501,
                            f64::from_bits(1),
                            f32::from_bits(1) as f64,
                            1e-300,
                            1e300,
                            f64::MAX,
                        ][(i / 2 + 3 * (i % 2)) % 11]
                    } else {
                        let angle = (i / 2) as f64 * 0.013;
                        if i % 2 == 0 {
                            angle.cos()
                        } else {
                            angle.sin()
                        }
                    }
                })
                .collect();
            let oracle_coefficients: Vec<_> = coefficients
                .chunks_exact(dim)
                .flat_map(|row| row.iter().chain(row.iter()).copied())
                .collect();
            let table = ComplexRotaryTable::new(&ctx, tokens, dim, &coefficients).unwrap();
            let oracle_table =
                ComplexRotaryTable::new(&ctx, tokens, 2 * dim, &oracle_coefficients).unwrap();
            let mut output = ctx
                .allocate_output(Shape::new(vec![tokens, heads, dim]), dtype)
                .unwrap();
            let mut expected = ctx
                .allocate_output(Shape::new(vec![tokens, heads / 2, 2 * dim]), dtype)
                .unwrap();
            complex_rotary(&ctx, &oracle_table, &oracle_input, &mut expected).unwrap();
            complex_rotary(&ctx, &table, &input, &mut output).unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(bits(&output), bits(&expected), "{dtype:?} edges={edges}");
            let mut in_place = transfers::to_cuda(&host, 0).unwrap();
            let alias = in_place.clone();
            complex_rotary(&ctx, &table, &alias, &mut in_place).unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(bits(&in_place), bits(&expected), "{dtype:?} in-place");
            let graph =
                crate::capture(&ctx, || complex_rotary(&ctx, &table, &input, &mut output)).unwrap();
            // Rotate the complete encoding set and overwrite the retained
            // input, proving replay reads new inputs rather than captured data.
            let changed = match dtype {
                DType::F16 => Tensor::from_f16(
                    vec![tokens, heads, dim],
                    &(0..count)
                        .map(|i| half::f16::from_bits((i as u16).wrapping_add(97)))
                        .collect::<Vec<_>>(),
                ),
                DType::BF16 => Tensor::from_bf16(
                    vec![tokens, heads, dim],
                    &(0..count)
                        .map(|i| half::bf16::from_bits((i as u16).wrapping_add(97)))
                        .collect::<Vec<_>>(),
                ),
                _ => unreachable!(),
            }
            .unwrap();
            let changed = transfers::to_cuda(&changed, 0).unwrap();
            transfers::copy_cuda_to_cuda(&ctx, &changed, &input).unwrap();
            complex_rotary(&ctx, &oracle_table, &oracle_input, &mut expected).unwrap();
            ctx.synchronize().unwrap();
            let expected = bits(&expected);
            drop(table);
            drop(oracle_table);
            drop(input);
            drop(oracle_input);
            drop(changed);
            for _ in 0..3 {
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(bits(&output), expected, "{dtype:?} edges={edges} replay");
            }
        }
    }
}

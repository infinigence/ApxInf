//! Independent high-offset/outlier oracle and captured resource lifetime.
use super::*;
use crate::{capture, transfers, CudaContext};
use apxinf_core::{DType, Tensor};
use half::{bf16, f16};

#[test]
fn layer_norm_stable_moments_and_capture_lifetime() {
    let ctx = CudaContext::new(0).unwrap();
    for dtype in [DType::F16, DType::BF16] {
        let round = |x: f64| match dtype {
            DType::F16 => f16::from_f64(x).to_f32(),
            _ => bf16::from_f64(x).to_f32(),
        };
        let upload = |shape: Vec<usize>, values: &[f32]| {
            let host = match dtype {
                DType::F16 => Tensor::from_f16(
                    shape,
                    &values
                        .iter()
                        .copied()
                        .map(f16::from_f32)
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
                _ => Tensor::from_bf16(
                    shape,
                    &values
                        .iter()
                        .copied()
                        .map(bf16::from_f32)
                        .collect::<Vec<_>>(),
                )
                .unwrap(),
            };
            transfers::to_cuda(&host, 0).unwrap()
        };
        for cols in [4, 7, 1280, 5120] {
            let values: Vec<_> = (0..5 * cols)
                .map(|i| {
                    let c = i % cols;
                    round(match i / cols {
                        0 => 13.0,
                        1 => 1024.0 + (c % 7) as f64 * 8.0,
                        2 => {
                            if c == 0 {
                                8192.0
                            } else {
                                -0.25
                            }
                        }
                        3 => ((c * 37 % 101) as f64 - 50.0) / 8.0,
                        _ => ((c % 3) as f64 - 1.0) / 1024.0,
                    })
                })
                .collect();
            let weight: Vec<_> = (0..cols)
                .map(|c| round((c % 11 + 1) as f64 / 8.0))
                .collect();
            let bias: Vec<_> = (0..cols).map(|c| round((c % 5) as f64 / 16.0)).collect();
            let expected: Vec<_> = values
                .chunks(cols)
                .flat_map(|row| {
                    let mean = row.iter().map(|&x| f64::from(x)).sum::<f64>() / cols as f64;
                    let variance = row
                        .iter()
                        .map(|&x| (f64::from(x) - mean).powi(2))
                        .sum::<f64>()
                        / cols as f64;
                    row.iter()
                        .enumerate()
                        .map(|(c, &x)| {
                            round(
                                (f64::from(x) - mean) / (variance + 1e-6).sqrt()
                                    * f64::from(weight[c])
                                    + f64::from(bias[c]),
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect();
            let input = upload(vec![5, cols], &values);
            let w = upload(vec![cols], &weight);
            let b = upload(vec![cols], &bias);
            let mut output = upload(vec![5, cols], &vec![0.; values.len()]);
            layer_norm(&ctx, LayerNormArgs::new(&input, &w, &b, &mut output, 1e-6)).unwrap();
            ctx.synchronize().unwrap();
            let actual = transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap();
            let tolerance = if dtype == DType::BF16 { 0.008 } else { 0.001 };
            for (&a, &e) in actual.iter().zip(&expected) {
                assert!(
                    (a - e).abs() <= tolerance * (1. + e.abs()),
                    "{dtype:?}/{cols}: {a} != {e}"
                );
            }
            let graph = capture(&ctx, || {
                layer_norm(&ctx, LayerNormArgs::new(&input, &w, &b, &mut output, 1e-6))
            })
            .unwrap();
            drop(input);
            drop(w);
            drop(b);
            for _ in 0..3 {
                graph.replay().unwrap();
                ctx.synchronize().unwrap();
                assert_eq!(
                    transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap(),
                    actual
                );
            }
        }
    }
}

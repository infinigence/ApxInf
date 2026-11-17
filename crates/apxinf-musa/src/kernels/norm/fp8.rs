use crate::kernels::contracts::{require_matrix_bf16, require_positive_eps, require_vector_bf16};
use crate::MusaContext;
use apxinf_core::{DType, Error, Result, Tensor};

#[derive(Clone, Copy)]
pub enum Fp8Norm<'a> {
    Layer {
        weight: &'a Tensor,
        bias: &'a Tensor,
    },
    Rms {
        weight: &'a Tensor,
    },
    Adaptive {
        modulation: &'a Tensor,
    },
}

pub struct Fp8NormResidual<'a> {
    pub residual: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub round_bias: bool,
}

pub fn norm_quantized(
    ctx: &MusaContext,
    input: &Tensor,
    kind: Fp8Norm<'_>,
    residual: Option<Fp8NormResidual<'_>>,
    eps: f32,
    scale: f32,
) -> Result<(Option<Tensor>, Tensor)> {
    let (rows, cols) = require_matrix_bf16(ctx, input, "FP8 norm", "input")?;
    require_positive_eps(eps, "FP8 norm")?;
    if rows == 0
        || crate::MUSA_ARCH != "mp_31"
        || !scale.is_finite()
        || scale <= 0.0
        || !scale.recip().is_finite()
    {
        return Err(Error::Other(
            "FP8 norm requires MP31, nonempty input and a finite positive scale".into(),
        ));
    }
    let (mode, weight, bias) = match kind {
        Fp8Norm::Layer { weight, bias } => {
            require_vector_bf16(ctx, weight, cols, "FP8 norm", "weight")?;
            require_vector_bf16(ctx, bias, cols, "FP8 norm", "bias")?;
            (0, weight, Some(bias))
        }
        Fp8Norm::Rms { weight } => {
            require_vector_bf16(ctx, weight, cols, "FP8 norm", "weight")?;
            if cols != 2048 || residual.is_some() {
                return Err(Error::Other(
                    "FP8 RMSNorm requires width 2048 without residual".into(),
                ));
            }
            (1, weight, None)
        }
        Fp8Norm::Adaptive { modulation } => {
            require_vector_bf16(ctx, modulation, 2 * cols, "FP8 norm", "modulation")?;
            (2, modulation, None)
        }
    };
    if let Some(skip) = &residual {
        crate::kernels::contracts::require_matching_bf16(
            ctx,
            "FP8 norm",
            "residual",
            input,
            skip.residual,
        )?;
        if let Some(bias) = skip.bias {
            require_vector_bf16(ctx, bias, cols, "FP8 norm", "projection bias")?;
        }
        if !((cols == 1024 && !skip.round_bias && mode == 0)
            || (cols == 1536 && skip.round_bias && skip.bias.is_some() && mode != 1))
        {
            return Err(Error::Other(
                "unsupported FP8 residual norm contract".into(),
            ));
        }
    }
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = (weight, bias);
        Err(crate::kernels::not_ported("norm_quantized"))
    }
    #[cfg(apxinf_musa_available)]
    {
        use crate::kernels::contracts::{
            checked_bytes, checked_i32, output_tensor, output_tensor_dims, tensor_ptr,
        };
        if (ctx.caps().compute_major, ctx.caps().compute_minor) != (3, 1) {
            return Err(Error::Other("FP8 norm requires an MP31 device".into()));
        }
        let count = checked_bytes(DType::BF16, &[rows, cols], "FP8 norm")? / 2;
        let output = output_tensor(
            ctx,
            input.shape().clone(),
            DType::F8E4M3,
            crate::workspace::output_buffer(ctx, count)?,
        );
        let hidden = residual
            .as_ref()
            .map(|_| output_tensor_dims(ctx, &[rows, cols], DType::BF16))
            .transpose()?;
        let ptr = |value: Option<&Tensor>| {
            value
                .map(tensor_ptr)
                .transpose()
                .map(|value| value.unwrap_or(std::ptr::null_mut()))
        };
        unsafe {
            crate::kernels::check_native(
                crate::ffi::custom::apxinf_musa_norm_e4m3(
                    tensor_ptr(input)?,
                    tensor_ptr(weight)?,
                    ptr(bias)?,
                    ptr(residual.as_ref().map(|skip| skip.residual))?,
                    ptr(residual.as_ref().and_then(|skip| skip.bias))?,
                    ptr(hidden.as_ref())?,
                    tensor_ptr(&output)?,
                    checked_i32("FP8 norm", "rows", rows)?,
                    checked_i32("FP8 norm", "cols", cols)?,
                    mode,
                    i32::from(residual.as_ref().is_some_and(|skip| skip.round_bias)),
                    eps,
                    scale,
                    ctx.stream().handle(),
                ),
                "FP8 norm",
            )?;
        }
        Ok((hidden, output))
    }
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    use super::*;
    use crate::kernels::{fused, norm, quantization};
    use crate::{transfers, ExecutionSession};
    use apxinf_core::Graph;
    use std::sync::Arc;

    #[test]
    #[ignore = "requires idle MP31; APXINF_TEST_DEVICE selects the GPU"]
    fn fp8_norm_preserves_bf16_rounding_and_graph_outputs() {
        let device = std::env::var("APXINF_TEST_DEVICE")
            .unwrap_or_else(|_| "0".into())
            .parse()
            .unwrap();
        let ctx = Arc::new(MusaContext::new(device).unwrap());
        let tensor = |rows, cols, offset: usize| {
            let values: Vec<_> = (0..rows * cols)
                .map(|index| {
                    half::bf16::from_f32(((index * 37 + offset) % 509) as f32 / 127.0 - 2.0)
                })
                .collect();
            transfers::to_device(
                &Tensor::from_bf16(vec![rows, cols], &values).unwrap(),
                device,
            )
            .unwrap()
        };
        let mut cases = vec![
            (512, 1024, 0, false),
            (128, 4096, 0, false),
            (512, 1024, 0, true),
            (41, 1536, 0, true),
            (41, 1536, 2, true),
            (41, 1536, 2, false),
        ];
        for rows in [129, 143, 145, 146, 256] {
            cases.push((rows, 2048, 0, false));
            cases.push((rows, 2048, 1, false));
        }
        for (rows, cols, mode, with_residual) in cases {
            let input = tensor(rows, cols, 11);
            let weight = tensor(1, cols, 3).reshape(vec![cols]).unwrap();
            let bias = tensor(1, cols, 43).reshape(vec![cols]).unwrap();
            let modulation = tensor(1, 2 * cols, 7).reshape(vec![2 * cols]).unwrap();
            let residual = tensor(rows, cols, 73);
            let round_bias = cols == 1536;
            let kind = match mode {
                0 => Fp8Norm::Layer {
                    weight: &weight,
                    bias: &bias,
                },
                1 => Fp8Norm::Rms { weight: &weight },
                _ => Fp8Norm::Adaptive {
                    modulation: &modulation,
                },
            };
            let eps = 1e-6;
            let (expected_hidden, normalized) = if with_residual {
                let pair = if mode == 2 {
                    fused::bias_then_residual_adaptive_layer_bf16(
                        &ctx,
                        &input,
                        &bias,
                        &residual,
                        &modulation,
                        eps,
                    )
                } else if round_bias {
                    fused::bias_then_residual_layer_bf16(
                        &ctx, &input, &bias, &residual, &weight, &bias, eps,
                    )
                } else {
                    fused::bias_residual_layer_bf16(
                        &ctx,
                        &input,
                        Some(&bias),
                        &residual,
                        &weight,
                        &bias,
                        eps,
                    )
                }
                .unwrap();
                (Some(pair.hidden), pair.normalized)
            } else {
                let normalized = match mode {
                    0 => norm::layer_bf16(&ctx, &input, &weight, &bias, eps),
                    1 => norm::rms_bf16(&ctx, &input, &weight, eps),
                    _ => norm::adaptive_layer(&ctx, &input, &modulation, eps),
                }
                .unwrap();
                (None, normalized)
            };
            for scale in [0.0001, 0.0125, 0.25] {
                let expected = quantization::quantize_bf16_e4m3(&ctx, &normalized, scale).unwrap();
                let execute = || {
                    norm_quantized(
                        &ctx,
                        &input,
                        kind,
                        with_residual.then_some(Fp8NormResidual {
                            residual: &residual,
                            bias: Some(&bias),
                            round_bias,
                        }),
                        eps,
                        scale,
                    )
                };
                let session = ExecutionSession::with_capacity(8 << 20, device).unwrap();
                session.prepare(&execute).unwrap();
                let mut captured = None;
                let graph = crate::graph::capture(Arc::clone(&ctx), || {
                    captured = Some(session.run(&execute)?);
                    Ok(())
                })
                .unwrap();
                for _ in 0..2 {
                    graph.replay().unwrap();
                    ctx.synchronize().unwrap();
                    let (hidden, actual) = captured.as_ref().unwrap();
                    let actual = transfers::to_cpu(actual).unwrap();
                    let expected = transfers::to_cpu(&expected).unwrap();
                    let mismatches = actual
                        .as_f8_e4m3()
                        .unwrap()
                        .iter()
                        .zip(expected.as_f8_e4m3().unwrap())
                        .filter(|(actual, expected)| actual != expected)
                        .count();
                    assert_eq!(
                        mismatches, 0,
                        "{rows}x{cols} mode={mode} residual={with_residual} scale={scale}"
                    );
                    if let Some(expected) = &expected_hidden {
                        assert_eq!(
                            transfers::to_cpu(hidden.as_ref().unwrap())
                                .unwrap()
                                .as_bf16()
                                .unwrap(),
                            transfers::to_cpu(expected).unwrap().as_bf16().unwrap()
                        );
                    } else {
                        assert!(hidden.is_none());
                    }
                }
            }
            for scale in [0.0, -1.0, f32::NAN, f32::INFINITY] {
                assert!(norm_quantized(&ctx, &input, kind, None, eps, scale).is_err());
            }
        }
    }
}

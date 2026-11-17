use crate::MusaContext;
use apxinf_core::{Result, Tensor};
#[cfg(apxinf_musa_available)]
use apxinf_core::{DType, Device, Error};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8Layout {
    Nn,
    Nt,
}

pub fn gemm_fp8(
    ctx: &MusaContext,
    activation: &Tensor,
    weight: &Tensor,
    layout: Fp8Layout,
    scale: f32,
) -> Result<Tensor> {
    #[cfg(not(apxinf_musa_available))]
    {
        let _ = (ctx, activation, weight, layout, scale);
        Err(crate::kernels::not_ported("gemm_fp8"))
    }
    #[cfg(apxinf_musa_available)]
    {
        use crate::kernels::contracts::{checked_i32, output_tensor_dims, tensor_ptr};
        if crate::MUSA_ARCH != "mp_31"
            || (ctx.caps().compute_major, ctx.caps().compute_minor) != (3, 1)
            || !scale.is_finite()
            || scale <= 0.0
        {
            return Err(Error::Other(
                "FP8 GEMM requires MP31 and a finite positive scale".into(),
            ));
        }
        for tensor in [activation, weight] {
            if tensor.dtype() != DType::F8E4M3
                || tensor.device() != Device::Musa(ctx.device_id())
                || tensor.shape().dims().len() != 2
            {
                return Err(Error::Other(
                    "FP8 GEMM requires device-local E4M3 matrices".into(),
                ));
            }
        }
        let [rows, inner] = [activation.shape().dims()[0], activation.shape().dims()[1]];
        let dims = weight.shape().dims();
        let (weight_inner, cols) = match layout {
            Fp8Layout::Nn => (dims[0], dims[1]),
            Fp8Layout::Nt => (dims[1], dims[0]),
        };
        if inner != weight_inner || rows == 0 || inner == 0 || cols == 0 {
            return Err(Error::Other(
                "FP8 GEMM contraction mismatch or empty matrix".into(),
            ));
        }
        crate::workspace::note_shape(crate::workspace::OperationShape::Gemm {
            m: rows,
            n: cols,
            k: inner,
        })?;
        let output = output_tensor_dims(ctx, &[rows, cols], DType::BF16)?;
        let status = unsafe {
            crate::ffi::custom::apxinf_musa_mudnn_fp8_gemm(
                tensor_ptr(activation)?,
                tensor_ptr(weight)?,
                tensor_ptr(&output)?,
                checked_i32("fp8", "rows", rows)?,
                checked_i32("fp8", "cols", cols)?,
                checked_i32("fp8", "inner", inner)?,
                i32::from(layout == Fp8Layout::Nt),
                scale,
                ctx.stream().handle(),
            )
        };
        crate::ffi::custom::check_mudnn(status, || {
            format!("muDNN FP8 {rows}x{cols}x{inner} {layout:?}")
        })
        .map_err(crate::error::musa_error)?;
        Ok(output)
    }
}

#[cfg(all(test, apxinf_musa_available))]
mod tests {
    use super::*;
    use crate::kernels::quantization::quantize_bf16_e4m3;
    use crate::{transfers, ExecutionSession};
    use apxinf_core::Graph;
    use std::sync::Arc;

    #[test]
    #[ignore = "requires an idle MP31 GPU; APXINF_TEST_DEVICE selects it"]
    fn fp8_layouts_scales_and_graph_replay() {
        let device = std::env::var("APXINF_TEST_DEVICE")
            .unwrap_or_else(|_| "0".into())
            .parse()
            .unwrap();
        let ctx = Arc::new(MusaContext::new(device).unwrap());
        assert_eq!(crate::MUSA_ARCH, "mp_31");
        for rows in [1, 129, 145, 256] {
            let (cols, inner) = (128, 256);
            let activation_values: Vec<_> = (0..rows * inner)
                .map(|index| half::bf16::from_f32((index % 7) as f32 / 4.0 - 0.75))
                .collect();
            let weight_values: Vec<_> = (0..inner * cols)
                .map(|index| half::bf16::from_f32((index % 5) as f32 / 4.0 - 0.5))
                .collect();
            let activation = transfers::to_device(
                &Tensor::from_bf16(vec![rows, inner], &activation_values).unwrap(),
                device,
            )
            .unwrap();
            for layout in [Fp8Layout::Nn, Fp8Layout::Nt] {
                let (shape, packed) = if layout == Fp8Layout::Nn {
                    (vec![inner, cols], weight_values.clone())
                } else {
                    let mut packed = vec![half::bf16::ZERO; inner * cols];
                    for depth in 0..inner {
                        for column in 0..cols {
                            packed[column * inner + depth] = weight_values[depth * cols + column];
                        }
                    }
                    (vec![cols, inner], packed)
                };
                let weight =
                    transfers::to_device(&Tensor::from_bf16(shape, &packed).unwrap(), device)
                        .unwrap();
                let quantized_weight = quantize_bf16_e4m3(&ctx, &weight, 0.125).unwrap();
                let session = ExecutionSession::with_capacity(4 << 20, device).unwrap();
                for multiplier in [1.0, 0.5, 2.0] {
                    let execute = || {
                        let quantized = quantize_bf16_e4m3(&ctx, &activation, 0.125)?;
                        gemm_fp8(
                            &ctx,
                            &quantized,
                            &quantized_weight,
                            layout,
                            0.015625 * multiplier,
                        )
                    };
                    session.prepare(&execute).unwrap();
                    let mut output = None;
                    let graph = crate::graph::capture(Arc::clone(&ctx), || {
                        output = Some(session.run(&execute)?);
                        Ok(())
                    })
                    .unwrap();
                    graph.replay().unwrap();
                    ctx.synchronize().unwrap();
                    let values = transfers::to_cpu(output.as_ref().unwrap())
                        .unwrap()
                        .to_f32_vec()
                        .unwrap();
                    for row in [0, rows - 1] {
                        for column in [0, 37, cols - 1] {
                            let expected: f32 = (0..inner)
                                .map(|depth| {
                                    activation_values[row * inner + depth].to_f32()
                                        * weight_values[depth * cols + column].to_f32()
                                })
                                .sum();
                            assert_eq!(
                                values[row * cols + column],
                                half::bf16::from_f32(expected * multiplier).to_f32()
                            );
                        }
                    }
                }
                assert!(gemm_fp8(&ctx, &weight, &quantized_weight, layout, 1.0).is_err());
                assert!(quantize_bf16_e4m3(&ctx, &activation, 0.0).is_err());
                assert!(quantize_bf16_e4m3(&ctx, &activation, f32::NAN).is_err());
            }
        }
    }
}

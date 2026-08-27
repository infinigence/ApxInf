use apxinf_core::{Backend, Result, Tensor};
use half::bf16;

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::kernels::gemm::{gemm_w8a8_with_preference, W8A8Layout, W8A8ScaleMode, W8A8WeightView};
use crate::CudaBackend;

fn gemm_with_preference(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &CudaBuffer,
    scales: &Tensor,
    input_dim: usize,
    output_dim: usize,
    prefer_cutlass: bool,
) -> Result<Tensor> {
    gemm_w8a8_with_preference(
        ctx,
        activation,
        W8A8WeightView {
            values_i8: weight,
            scales_f32: scales,
            input_dim,
            output_dim,
            scale_mode: W8A8ScaleMode::DynamicRowPerOutputChannel,
            layout: W8A8Layout::OutputMajor,
        },
        prefer_cutlass,
    )
}

fn gemm(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &CudaBuffer,
    scales: &Tensor,
    input_dim: usize,
    output_dim: usize,
) -> Result<Tensor> {
    gemm_with_preference(
        ctx, activation, weight, scales, input_dim, output_dim, false,
    )
}

#[test]
fn w8a8_gemm_matches_small_reference() {
    let backend = CudaBackend::new(0).unwrap();
    let activation = Tensor::from_bf16(
        vec![2, 4],
        &[
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(4.0),
            bf16::from_f32(-1.0),
            bf16::from_f32(0.0),
            bf16::from_f32(1.0),
            bf16::from_f32(2.0),
        ],
    )
    .unwrap();
    let activation = backend.to_device(&activation).unwrap();
    // Two physical output-major rows: [1,0,-1,2] and [2,1,0,-1].
    let weight = CudaBuffer::alloc(8, 0).unwrap();
    weight
        .copy_from_host(&[1, 0, (-1i8) as u8, 2, 2, 1, 0, (-1i8) as u8])
        .unwrap();
    let scales = backend
        .to_device(&Tensor::from_f32(vec![2], &[1.0, 1.0]).unwrap())
        .unwrap();
    let output = gemm(backend.context(), &activation, &weight, &scales, 4, 2).unwrap();
    let actual = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
    let expected = [6.0f32, 0.0, 2.0, -4.0];
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= 0.05,
            "actual {actual}, expected {expected}"
        );
    }
}

#[cfg(apxinf_cutlass_int8_sm80)]
#[test]
fn fused_w8a8_gemm_applies_row_and_column_scales() {
    let backend = CudaBackend::new(0).unwrap();
    let mut activation = vec![bf16::from_f32(1.0); 32];
    activation[16..].fill(bf16::from_f32(-2.0));
    let activation = backend
        .to_device(&Tensor::from_bf16(vec![2, 16], &activation).unwrap())
        .unwrap();

    // Eight output channels containing the same integer weights but
    // distinct scales exercise CUTLASS's per-column epilogue indexing.
    let weight = CudaBuffer::alloc(8 * 16, 0).unwrap();
    weight.copy_from_host(&[1u8; 8 * 16]).unwrap();
    let scales = backend
        .to_device(
            &Tensor::from_f32(vec![8], &[0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]).unwrap(),
        )
        .unwrap();

    let output = gemm(backend.context(), &activation, &weight, &scales, 16, 8).unwrap();
    let actual = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
    let expected = [
        4.0f32, 8.0, 12.0, 16.0, 20.0, 24.0, 28.0, 32.0, -8.0, -16.0, -24.0, -32.0, -40.0, -48.0,
        -56.0, -64.0,
    ];
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(*actual, expected);
    }
}

#[cfg(apxinf_cutlass_int8_sm80)]
#[test]
fn fused_w8a8_matches_cublas_at_static_shape_classes() {
    let backend = CudaBackend::new(0).unwrap();
    for (rows, output_dim, input_dim) in [
        (130usize, 136usize, 144usize),
        (512, 1152, 4304),
        (544, 32_768, 2048),
        (544, 2048, 16_384),
        (10, 8192, 1024),
    ] {
        let activation_values = (0..rows * input_dim)
            .map(|index| {
                let row = index / input_dim;
                let col = index % input_dim;
                let value =
                    (((col * 17 + row * 3) % 29) as f32 - 14.0) * ((row % 5 + 1) as f32 / 37.0);
                bf16::from_f32(value)
            })
            .collect::<Vec<_>>();
        let activation = backend
            .to_device(&Tensor::from_bf16(vec![rows, input_dim], &activation_values).unwrap())
            .unwrap();
        let weight_values = (0..output_dim * input_dim)
            .map(|index| (((index * 13 + index / input_dim) % 15) as i8 - 7) as u8)
            .collect::<Vec<_>>();
        let weight = CudaBuffer::alloc(weight_values.len(), backend.device_id()).unwrap();
        weight.copy_from_host(&weight_values).unwrap();
        let scale_values = (0..output_dim)
            .map(|col| (col % 7 + 1) as f32 * 0.00137)
            .collect::<Vec<_>>();
        let scales = backend
            .to_device(&Tensor::from_f32(vec![output_dim], &scale_values).unwrap())
            .unwrap();

        let fused = gemm_with_preference(
            backend.context(),
            &activation,
            &weight,
            &scales,
            input_dim,
            output_dim,
            true,
        )
        .unwrap();
        let cublas = gemm_with_preference(
            backend.context(),
            &activation,
            &weight,
            &scales,
            input_dim,
            output_dim,
            false,
        )
        .unwrap();
        let fused = backend.to_cpu(&fused).unwrap().to_f32_vec().unwrap();
        let cublas = backend.to_cpu(&cublas).unwrap().to_f32_vec().unwrap();
        let max_abs = fused
            .iter()
            .zip(&cublas)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        let different = fused
            .iter()
            .zip(&cublas)
            .filter(|(lhs, rhs)| lhs != rhs)
            .count();
        println!(
                "W8A8 comparison [{rows},{output_dim},{input_dim}]: max_abs={max_abs}, different={different}/{}",
                fused.len()
            );
        assert!(max_abs <= 0.03125, "fused GEMM diverged from cuBLAS");
    }
}

use apxinf_core::{Backend, Result, Tensor};
use half::bf16;

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::kernels::gemm::{
    adaptive_layer_norm_quantize_w8a8_activation, gemm_quantized_w8a8, gemm_w8a8_with_preference,
    quantize_w8a8_activation, quantize_w8a8_silu_mul_activation, W8A8Layout, W8A8ScaleMode,
    W8A8WeightView,
};
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

#[cfg(apxinf_cutlass_int8_sm80)]
#[test]
fn fused_adaptive_layer_norm_w8a8_matches_decomposed_contract() {
    let backend = CudaBackend::new(0).unwrap();
    let (rows, cols, output_dim) = (5usize, 128usize, 16usize);
    let input_values = (0..rows * cols)
        .map(|index| bf16::from_f32(((index * 17 % 101) as f32 - 50.0) / 19.0))
        .collect::<Vec<_>>();
    let modulation_values = (0..2 * cols)
        .map(|index| bf16::from_f32(((index * 11 % 43) as f32 - 21.0) / 97.0))
        .collect::<Vec<_>>();
    let input = backend
        .to_device(&Tensor::from_bf16(vec![rows, cols], &input_values).unwrap())
        .unwrap();
    let modulation = backend
        .to_device(&Tensor::from_bf16(vec![2 * cols], &modulation_values).unwrap())
        .unwrap();
    let expected_normalized =
        crate::kernels::norm::adaptive_layer(backend.context(), &input, &modulation, 1.0e-6)
            .unwrap();
    let expected_quantized =
        quantize_w8a8_activation(backend.context(), &expected_normalized).unwrap();
    let (actual_normalized, actual_quantized) = adaptive_layer_norm_quantize_w8a8_activation(
        backend.context(),
        &input,
        &modulation,
        1.0e-6,
    )
    .unwrap();
    assert_eq!(
        backend
            .to_cpu(&actual_normalized)
            .unwrap()
            .to_f32_vec()
            .unwrap(),
        backend
            .to_cpu(&expected_normalized)
            .unwrap()
            .to_f32_vec()
            .unwrap()
    );

    let weight_values = (0..output_dim * cols)
        .map(|index| (((index * 13) % 31) as i8 - 15) as u8)
        .collect::<Vec<_>>();
    let weight = CudaBuffer::alloc(weight_values.len(), backend.device_id()).unwrap();
    weight.copy_from_host(&weight_values).unwrap();
    let scales = backend
        .to_device(&Tensor::from_f32(vec![output_dim], &vec![0.003; output_dim]).unwrap())
        .unwrap();
    let view = || W8A8WeightView {
        values_i8: &weight,
        scales_f32: &scales,
        input_dim: cols,
        output_dim,
        scale_mode: W8A8ScaleMode::DynamicRowPerOutputChannel,
        layout: W8A8Layout::OutputMajor,
    };
    let expected = gemm_quantized_w8a8(backend.context(), &expected_quantized, view()).unwrap();
    let actual = gemm_quantized_w8a8(backend.context(), &actual_quantized, view()).unwrap();
    assert_eq!(
        backend.to_cpu(&actual).unwrap().to_f32_vec().unwrap(),
        backend.to_cpu(&expected).unwrap().to_f32_vec().unwrap()
    );
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
fn fused_silu_mul_w8a8_matches_two_stage_contract() {
    let backend = CudaBackend::new(0).unwrap();
    let (rows, input_dim, output_dim) = (41usize, 256usize, 64usize);
    let gate_values = (0..rows * input_dim)
        .map(|index| bf16::from_f32(((index * 17 % 101) as f32 - 50.0) / 19.0))
        .collect::<Vec<_>>();
    let up_values = (0..rows * input_dim)
        .map(|index| bf16::from_f32(((index * 29 % 89) as f32 - 44.0) / 23.0))
        .collect::<Vec<_>>();
    let gate = backend
        .to_device(&Tensor::from_bf16(vec![rows, input_dim], &gate_values).unwrap())
        .unwrap();
    let up = backend
        .to_device(&Tensor::from_bf16(vec![rows, input_dim], &up_values).unwrap())
        .unwrap();
    let weight_values = (0..output_dim * input_dim)
        .map(|index| (((index * 13 + index / input_dim) % 31) as i8 - 15) as u8)
        .collect::<Vec<_>>();
    let weight = CudaBuffer::alloc(weight_values.len(), backend.device_id()).unwrap();
    weight.copy_from_host(&weight_values).unwrap();
    let scales = backend
        .to_device(&Tensor::from_f32(vec![output_dim], &vec![0.003f32; output_dim]).unwrap())
        .unwrap();
    let view = W8A8WeightView {
        values_i8: &weight,
        scales_f32: &scales,
        input_dim,
        output_dim,
        scale_mode: W8A8ScaleMode::DynamicRowPerOutputChannel,
        layout: W8A8Layout::OutputMajor,
    };

    let separated =
        crate::kernels::activation::silu_mul_bf16(backend.context(), &gate, &up).unwrap();
    let expected = gemm_with_preference(
        backend.context(),
        &separated,
        &weight,
        &scales,
        input_dim,
        output_dim,
        true,
    )
    .unwrap();
    let quantized = quantize_w8a8_silu_mul_activation(backend.context(), &gate, &up).unwrap();
    let actual = gemm_quantized_w8a8(backend.context(), &quantized, view).unwrap();
    let expected = backend.to_cpu(&expected).unwrap().to_f32_vec().unwrap();
    let actual = backend.to_cpu(&actual).unwrap().to_f32_vec().unwrap();
    assert_eq!(actual, expected);
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

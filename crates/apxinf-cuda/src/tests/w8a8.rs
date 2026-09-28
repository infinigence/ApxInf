use apxinf_core::{Backend, Result, Tensor};
use half::bf16;

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::kernels::gemm::{gemm_w8a8_with_preference, W8A8Layout, W8A8ScaleMode, W8A8WeightView};
use crate::CudaBackend;

fn bf16_bits(tensor: &Tensor) -> Vec<u16> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; tensor.size_in_bytes()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|value| u16::from_ne_bytes([value[0], value[1]]))
        .collect()
}

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

#[test]
fn default_w8a8_handles_dimensions_without_cutlass_alignment() {
    let backend = CudaBackend::new(0).unwrap();
    // PI0.5's patch projection has K=588. Also cover a non-aligned N.
    // Use the public resolver: the forced-preference test helper bypasses it.
    for (input_dim, output_dim) in [(588, 1152), (16, 10)] {
        let mut values = vec![bf16::from_f32(1.0); 2 * input_dim];
        values[input_dim..].fill(bf16::from_f32(-2.0));
        let activation = backend
            .to_device(&Tensor::from_bf16(vec![2, input_dim], &values).unwrap())
            .unwrap();
        let weight = CudaBuffer::alloc(input_dim * output_dim, backend.device_id()).unwrap();
        weight
            .copy_from_host(&vec![1u8; input_dim * output_dim])
            .unwrap();
        let scales = backend
            .to_device(&Tensor::from_f32(vec![output_dim], &vec![1.0; output_dim]).unwrap())
            .unwrap();
        let output = crate::kernels::gemm::w8a8(
            backend.context(),
            &activation,
            W8A8WeightView {
                values_i8: &weight,
                scales_f32: &scales,
                input_dim,
                output_dim,
                scale_mode: W8A8ScaleMode::DynamicRowPerOutputChannel,
                layout: W8A8Layout::OutputMajor,
            },
        )
        .unwrap();
        let actual = backend.to_cpu(&output).unwrap().to_f32_vec().unwrap();
        assert_eq!(actual[..output_dim], vec![input_dim as f32; output_dim]);
        assert_eq!(
            actual[output_dim..],
            vec![-2.0 * input_dim as f32; output_dim]
        );
    }
}

#[cfg(apxinf_cutlass_int8_sm80)]
#[test]
fn explicit_w8a8_producer_is_bitwise_exact_without_a_plain_gemm_plan() {
    use crate::kernels::gemm::{
        bias_gelu_quantize_w8a8_activation, quantize_w8a8_activation,
        try_gemm_quantized_w8a8_bias_gelu_quantized, try_gemm_quantized_w8a8_m41_n6144_k1536,
        w8a8_tuning_key_for_test,
    };
    use crate::tuning::{GemmTuningRecord, TacticBackend, TacticId, TacticStore, TuningSession};

    const M: usize = 41;
    const N: usize = 6144;
    const K: usize = 1536;
    let backend = CudaBackend::new(0).unwrap();
    if backend.context().caps().sm != 87 {
        return;
    }
    let activation_values = (0..M * K)
        .map(|index| bf16::from_f32(((index * 17 + index / K * 5) % 31) as f32 / 19.0 - 0.8))
        .collect::<Vec<_>>();
    let activation = backend
        .to_device(&Tensor::from_bf16(vec![M, K], &activation_values).unwrap())
        .unwrap();
    let quantized_input = quantize_w8a8_activation(backend.context(), &activation).unwrap();
    let weight_values = (0..N * K)
        .map(|index| (((index * 13 + index / K) % 15) as i8 - 7) as u8)
        .collect::<Vec<_>>();
    let weight = CudaBuffer::alloc(weight_values.len(), backend.context().device_id()).unwrap();
    weight.copy_from_host(&weight_values).unwrap();
    let scales = backend
        .to_device(
            &Tensor::from_f32(
                vec![N],
                &(0..N)
                    .map(|column| (column % 11 + 1) as f32 * 0.00031)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        )
        .unwrap();
    let bias = backend
        .to_device(
            &Tensor::from_bf16(
                vec![N],
                &(0..N)
                    .map(|column| bf16::from_f32((column % 17) as f32 * 0.003 - 0.02))
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
        )
        .unwrap();
    let view = || W8A8WeightView {
        values_i8: &weight,
        scales_f32: &scales,
        input_dim: K,
        output_dim: N,
        scale_mode: W8A8ScaleMode::DynamicRowPerOutputChannel,
        layout: W8A8Layout::OutputMajor,
    };
    let key = w8a8_tuning_key_for_test(backend.context(), M, N, K);
    let install = |tactic: TacticId| {
        let store = TacticStore::from_gemm_records([GemmTuningRecord {
            key: key.clone(),
            tactic,
            implementation_version: Some(tactic.backend.implementation_version()),
            milliseconds: Some(1.0),
        }])
        .unwrap();
        backend
            .context()
            .install_tuning(TuningSession::inference(store))
            .unwrap();
    };
    backend
        .context()
        .install_tuning(TuningSession::inference(TacticStore::default()))
        .unwrap();
    let projection =
        try_gemm_quantized_w8a8_m41_n6144_k1536(backend.context(), &quantized_input, view())
            .unwrap()
            .expect("explicit SM87 shape schedule must work with an empty tactic store");
    let reference_activated =
        crate::kernels::activation::bias_gelu_bf16(backend.context(), &projection, Some(&bias))
            .unwrap();
    let reference_quantized =
        bias_gelu_quantize_w8a8_activation(backend.context(), &projection, &bias).unwrap();
    let (candidate_activated, candidate_quantized) = try_gemm_quantized_w8a8_bias_gelu_quantized(
        backend.context(),
        &quantized_input,
        view(),
        &bias,
    )
    .unwrap()
    .expect("explicit SM87 producer fusion must work with an empty tactic store");
    backend.context().synchronize().unwrap();
    assert_eq!(
        bf16_bits(&candidate_activated),
        bf16_bits(&reference_activated)
    );
    assert_eq!(
        candidate_quantized.row_scale_bits().unwrap(),
        reference_quantized.row_scale_bits().unwrap()
    );
    assert_eq!(
        candidate_quantized.quantized_bytes().unwrap(),
        reference_quantized.quantized_bytes().unwrap()
    );

    // Persisted plain-GEMM choices cannot enable or disable an explicitly
    // selected producer. In particular, the old external tactic six remains
    // invalid for the generic provider even though the separate schedule exists.
    for tactic in [
        TacticId {
            backend: TacticBackend::Cutlass,
            value: 0,
        },
        TacticId {
            backend: TacticBackend::Vendor,
            value: 0,
        },
        TacticId {
            backend: TacticBackend::Cutlass,
            value: 5,
        },
        TacticId {
            backend: TacticBackend::Cutlass,
            value: 6,
        },
    ] {
        install(tactic);
        assert!(try_gemm_quantized_w8a8_bias_gelu_quantized(
            backend.context(),
            &quantized_input,
            view(),
            &bias,
        )
        .unwrap()
        .is_some());
        if tactic.backend == TacticBackend::Cutlass && tactic.value != 0 {
            let default = TacticId {
                backend: TacticBackend::Cutlass,
                value: 0,
            };
            assert_eq!(
                backend
                    .context()
                    .gemm_plans()
                    .resolve(backend.context(), &key, default)
                    .unwrap()
                    .tactic,
                default
            );
        }
    }
    backend.context().synchronize().unwrap();

    let narrow_scales = backend
        .to_device(&Tensor::from_f32(vec![8], &[1.0; 8]).unwrap())
        .unwrap();
    let narrow_weight = CudaBuffer::alloc(8 * K, backend.context().device_id()).unwrap();
    narrow_weight.copy_from_host(&vec![1u8; 8 * K]).unwrap();
    let narrow_bias = backend
        .to_device(&Tensor::from_bf16(vec![8], &[bf16::from_f32(0.0); 8]).unwrap())
        .unwrap();
    assert!(try_gemm_quantized_w8a8_bias_gelu_quantized(
        backend.context(),
        &quantized_input,
        W8A8WeightView {
            values_i8: &narrow_weight,
            scales_f32: &narrow_scales,
            input_dim: K,
            output_dim: 8,
            scale_mode: W8A8ScaleMode::DynamicRowPerOutputChannel,
            layout: W8A8Layout::OutputMajor,
        },
        &narrow_bias,
    )
    .unwrap()
    .is_none());
}

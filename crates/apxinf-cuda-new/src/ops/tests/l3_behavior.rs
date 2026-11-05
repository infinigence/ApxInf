//! Per-operator L3 semantic and conditional Graph behavior tests.
//!
//! Every new L3 operator must add a public semantic-contract test here. Add a
//! Graph replay test only when the operator has an independent execution path,
//! resource lifetime, binding rule, or capture behavior not already covered by
//! the shared framework tests.

use super::framework::{bytes_tensor, f16_tensor, f32_tensor, tensor, values, zeros_tensor};
use super::*;
use crate::CudaContext;
use apxinf_core::{DType, Shape};
use half::{bf16, f16};

fn f16_values(tensor: &apxinf_core::Tensor) -> Vec<f32> {
    let buffer = crate::CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|value| f16::from_bits(u16::from_ne_bytes([value[0], value[1]])).to_f32())
        .collect()
}

fn f32_values(tensor: &apxinf_core::Tensor) -> Vec<f32> {
    let buffer = crate::CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect()
}

fn i32_values(tensor: &apxinf_core::Tensor) -> Vec<i32> {
    let buffer = crate::CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|chunk| i32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn raw_bytes(tensor: &apxinf_core::Tensor) -> Vec<u8> {
    let buffer = crate::CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
}

fn rgb_buffer(device: usize, bytes: &[u8]) -> crate::CudaBuffer {
    let buffer = crate::CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer
}

#[test]
fn argmax_uses_one_semantic_with_an_optional_remap_binding() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let logits = tensor(0, vec![1, 4], &[1.0, 4.0, 3.0, 2.0]);
    let index = zeros_tensor(0, vec![1], DType::I32);
    argmax(&ctx, ArgmaxArgs::new(&logits, &index)).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(i32_values(&index), [1]);

    let remap_values = [10_i32, 20, 30, 40];
    let remap_bytes = remap_values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let remap = bytes_tensor(0, vec![4], DType::I32, &remap_bytes);
    let remapped_index = zeros_tensor(0, vec![1], DType::I32);
    let graph = crate::capture(&ctx, || {
        argmax(
            &ctx,
            ArgmaxArgs::new(&logits, &remapped_index).with_remap(&remap),
        )
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(i32_values(&remapped_index), [20]);
}

#[test]
fn preprocess_infers_output_dtype_and_preserves_layout_and_graph_semantics() {
    use apxinf_core::DType;
    use preprocess::{
        ImageLayout, PatchGeometry, PatchNormalization, PreprocessArgs, PreprocessSpec,
    };

    let ctx = CudaContext::new(0).unwrap();
    let logical = [0_u8, 64, 255, 32, 96, 224, 16, 128, 240, 48, 160, 208];
    let expected = logical
        .iter()
        .map(|value| f32::from(*value) * (2.0 / 255.0) - 1.0)
        .collect::<Vec<_>>();
    let geometry = PatchGeometry::Uniform {
        views: 1,
        image_height: 2,
        image_width: 2,
        patch_size: 1,
        temporal_patch_size: 1,
        merge_size: 1,
    };

    let nhwc = rgb_buffer(0, &logical);
    let f32_output = zeros_tensor(0, vec![4, 3], DType::F32);
    preprocess::preprocess(
        &ctx,
        PreprocessArgs {
            images: &nhwc,
            patches: &f32_output,
            lookup_table: None,
            spec: PreprocessSpec {
                layout: ImageLayout::Nhwc,
                geometry,
                normalization: PatchNormalization::UnitRange,
            },
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();
    for (actual, expected) in f32_values(&f32_output).iter().zip(&expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }

    let nchw_bytes = [0_u8, 32, 16, 48, 64, 96, 128, 160, 255, 224, 240, 208];
    let nchw = rgb_buffer(0, &nchw_bytes);
    let f16_output = zeros_tensor(0, vec![4, 3], DType::F16);
    preprocess::preprocess(
        &ctx,
        PreprocessArgs {
            images: &nchw,
            patches: &f16_output,
            lookup_table: None,
            spec: PreprocessSpec {
                layout: ImageLayout::Nchw,
                geometry,
                normalization: PatchNormalization::UnitRange,
            },
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();
    let expected_f16 = expected
        .iter()
        .map(|value| f16::from_f32(*value).to_f32())
        .collect::<Vec<_>>();
    assert_eq!(f16_values(&f16_output), expected_f16);

    let bf16_output = zeros_tensor(0, vec![4, 3], DType::BF16);
    let graph = crate::capture(&ctx, || {
        preprocess::preprocess(
            &ctx,
            PreprocessArgs {
                images: &nhwc,
                patches: &bf16_output,
                lookup_table: None,
                spec: PreprocessSpec {
                    layout: ImageLayout::Nhwc,
                    geometry,
                    normalization: PatchNormalization::UnitRange,
                },
            },
        )
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    let expected_bf16 = expected
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect::<Vec<_>>();
    assert_eq!(values(&bf16_output), expected_bf16);
}

#[test]
fn add_mul_and_scale_use_caller_owned_outputs_and_infer_dtype() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let left = tensor(0, vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let right = tensor(0, vec![2, 2], &[0.5, 1.0, 1.5, 2.0]);
    let sum = zeros_tensor(0, vec![2, 2], DType::BF16);
    let scaled = zeros_tensor(0, vec![2, 2], DType::BF16);
    let graph = crate::capture(&ctx, || {
        elementwise::add(&ctx, &left, &right, &sum)?;
        elementwise::scale(&ctx, &sum, &scaled, 2.0)
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&sum), [1.5, 3.0, 4.5, 6.0]);
    assert_eq!(values(&scaled), [3.0, 6.0, 9.0, 12.0]);

    let left = f16_tensor(0, vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let right = f16_tensor(0, vec![4], &[0.5, 1.0, 1.5, 2.0]);
    let sum = zeros_tensor(0, vec![4], DType::F16);
    let scaled = zeros_tensor(0, vec![4], DType::F16);
    elementwise::add(&ctx, &left, &right, &sum).unwrap();
    elementwise::scale(&ctx, &sum, &scaled, 0.5).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(f16_values(&sum), [1.5, 3.0, 4.5, 6.0]);
    assert_eq!(f16_values(&scaled), [0.75, 1.5, 2.25, 3.0]);

    let left = f32_tensor(0, vec![4], &[1.0, 2.0, 3.0, 4.0]);
    let right = f32_tensor(0, vec![4], &[0.5, 1.0, 1.5, 2.0]);
    let sum = zeros_tensor(0, vec![4], DType::F32);
    let product = zeros_tensor(0, vec![4], DType::F32);
    let graph = crate::capture(&ctx, || {
        elementwise::add(&ctx, &left, &right, &sum)?;
        elementwise::mul(&ctx, &sum, &right, &product)
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(f32_values(&sum), [1.5, 3.0, 4.5, 6.0]);
    assert_eq!(f32_values(&product), [0.75, 3.0, 6.75, 12.0]);
}

#[test]
fn norm_family_uses_three_formal_entries_and_replays_as_one_graph() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(0, vec![2, 4], &[1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -3.0]);
    let residual = tensor(0, vec![2, 4], &[0.5, -0.5, 1.0, -1.0, 2.0, 1.5, -0.5, 0.25]);
    let weight = tensor(0, vec![4], &[1.0, 1.0, 1.0, 1.0]);
    let normalized = zeros_tensor(0, vec![2, 4], DType::BF16);
    let hidden = zeros_tensor(0, vec![2, 4], DType::BF16);
    let fused_hidden = zeros_tensor(0, vec![2, 4], DType::BF16);
    let fused_normalized = zeros_tensor(0, vec![2, 4], DType::BF16);

    let graph = crate::capture(&ctx, || {
        norm::norm(&ctx, NormArgs::rms(&input, &weight, &normalized, 1e-6))?;
        residual_add(
            &ctx,
            ResidualAddArgs {
                input: &input,
                bias: None,
                residual: &residual,
                gate: None,
                hidden: &hidden,
                bias_order: BiasOrder::SingleRound,
            },
        )?;
        fused_add_norm(
            &ctx,
            FusedAddNormArgs {
                input: &input,
                bias: None,
                residual: &residual,
                gate: None,
                hidden: &fused_hidden,
                normalized: Some(&fused_normalized),
                kind: NormKind::Rms,
                affine: Affine::Weight {
                    weight: &weight,
                    bias: None,
                    offset: false,
                },
                quantized: None,
                numerics: NormNumerics::Flexible,
                bias_order: BiasOrder::SingleRound,
                eps: 1e-6,
            },
        )
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    let input_values = values(&input);
    let residual_values = values(&residual);
    let expected_hidden = input_values
        .iter()
        .zip(&residual_values)
        .map(|(input, residual)| bf16::from_f32(input + residual).to_f32())
        .collect::<Vec<_>>();
    assert_eq!(values(&hidden), expected_hidden);
    assert_eq!(values(&fused_hidden), expected_hidden);

    for (source, actual) in [
        (input_values.as_slice(), values(&normalized)),
        (expected_hidden.as_slice(), values(&fused_normalized)),
    ] {
        for (row, actual_row) in source.chunks_exact(4).zip(actual.chunks_exact(4)) {
            let inverse_rms = (row.iter().map(|value| value * value).sum::<f32>() / 4.0 + 1e-6)
                .sqrt()
                .recip();
            for (value, actual) in row.iter().zip(actual_row) {
                let expected = bf16::from_f32(value * inverse_rms).to_f32();
                assert!((actual - expected).abs() <= 0.016, "{actual} != {expected}");
            }
        }
    }
}

#[test]
fn stable_decode_norm_is_a_formal_graph_safe_candidate() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let f32_input = f32_tensor(0, vec![1, 2], &[3.0, 4.0]);
    let f32_weight = f32_tensor(0, vec![2], &[1.0, 1.0]);
    let f32_output = zeros_tensor(0, vec![1, 2], DType::F32);

    let residual = tensor(0, vec![1, 4], &[1.0, 2.0, 3.0, 4.0]);
    let delta = tensor(0, vec![1, 4], &[1.0, 1.0, 1.0, 1.0]);
    let weight = tensor(0, vec![4], &[1.0, 1.0, 1.0, 1.0]);
    let normalized = zeros_tensor(0, vec![1, 4], DType::BF16);

    let graph = crate::capture(&ctx, || {
        let mut plain = NormArgs::rms(&f32_input, &f32_weight, &f32_output, 1e-6);
        plain.numerics = NormNumerics::StableDecode;
        norm::norm(&ctx, plain)?;
        fused_add_norm(
            &ctx,
            FusedAddNormArgs {
                input: &delta,
                bias: None,
                residual: &residual,
                gate: None,
                hidden: &residual,
                normalized: Some(&normalized),
                kind: NormKind::Rms,
                affine: Affine::Weight {
                    weight: &weight,
                    bias: None,
                    offset: false,
                },
                quantized: None,
                numerics: NormNumerics::StableDecode,
                bias_order: BiasOrder::SingleRound,
                eps: 1e-6,
            },
        )
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    let inverse_rms = (12.5_f32 + 1e-6).sqrt().recip();
    for (actual, expected) in f32_values(&f32_output)
        .iter()
        .zip([3.0 * inverse_rms, 4.0 * inverse_rms])
    {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }

    let expected_hidden = [3.0_f32, 4.0, 5.0, 6.0];
    assert_eq!(values(&residual), expected_hidden);
    let inverse_rms = (expected_hidden
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        / 4.0
        + 1e-6)
        .sqrt()
        .recip();
    for (actual, value) in values(&normalized).iter().zip(expected_hidden) {
        let expected = bf16::from_f32(value * inverse_rms).to_f32();
        assert!((actual - expected).abs() <= 0.016, "{actual} != {expected}");
    }
}

#[test]
fn planning_norm_contracts_use_formal_entries_and_replay_in_graph() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(
        0,
        vec![2, 8],
        &[
            1.0, -2.0, 3.0, -4.0, 0.5, -0.75, 1.25, -1.5, -1.0, 2.5, -3.5, 4.5, -0.25, 0.625,
            -1.125, 1.75,
        ],
    );
    let residual = tensor(
        0,
        vec![2, 8],
        &[
            0.5, 1.0, -0.5, 0.25, -1.0, 2.0, -2.5, 3.0, 1.5, -0.25, 0.75, -1.25, 2.25, -3.0, 0.5,
            1.0,
        ],
    );
    let weight = tensor(
        0,
        vec![8],
        &[0.0, 0.125, -0.25, 0.5, -0.5, 0.25, -0.125, 0.375],
    );
    let separate_hidden = zeros_tensor(0, vec![2, 8], DType::BF16);
    let separate_normalized = zeros_tensor(0, vec![2, 8], DType::BF16);
    let fused_hidden = zeros_tensor(0, vec![2, 8], DType::BF16);
    let fused_normalized = zeros_tensor(0, vec![2, 8], DType::BF16);

    let projection = tensor(
        0,
        vec![2, 8],
        &[
            0.25, -0.5, 0.75, -1.0, 1.25, -1.5, 1.75, -2.0, 2.0, -1.75, 1.5, -1.25, 1.0, -0.75,
            0.5, -0.25,
        ],
    );
    let gate = tensor(0, vec![8], &[-0.5, -0.25, 0.0, 0.125, 0.25, 0.5, 0.75, 1.0]);
    let scale = tensor(
        0,
        vec![8],
        &[0.0, 0.25, -0.25, 0.5, -0.5, 0.125, -0.125, 0.75],
    );
    let shift = tensor(
        0,
        vec![8],
        &[0.5, -0.5, 0.25, -0.25, 0.125, -0.125, 0.75, -0.75],
    );
    let adaptive_hidden = zeros_tensor(0, vec![2, 8], DType::BF16);
    let adaptive_normalized = zeros_tensor(0, vec![2, 8], DType::BF16);
    let adaptive_fused_hidden = zeros_tensor(0, vec![2, 8], DType::BF16);
    let adaptive_fused_normalized = zeros_tensor(0, vec![2, 8], DType::BF16);

    let graph = crate::capture(&ctx, || {
        residual_add(
            &ctx,
            ResidualAddArgs {
                input: &input,
                bias: None,
                residual: &residual,
                gate: None,
                hidden: &separate_hidden,
                bias_order: BiasOrder::SingleRound,
            },
        )?;
        let mut plain = NormArgs::rms(&separate_hidden, &weight, &separate_normalized, 1e-6)
            .with_weight_offset();
        plain.numerics = NormNumerics::StablePlanning;
        norm::norm(&ctx, plain)?;
        fused_add_norm(
            &ctx,
            FusedAddNormArgs {
                input: &input,
                bias: None,
                residual: &residual,
                gate: None,
                hidden: &fused_hidden,
                normalized: Some(&fused_normalized),
                kind: NormKind::Rms,
                affine: Affine::Weight {
                    weight: &weight,
                    bias: None,
                    offset: true,
                },
                quantized: None,
                numerics: NormNumerics::StablePlanning,
                bias_order: BiasOrder::SingleRound,
                eps: 1e-6,
            },
        )?;

        residual_add(
            &ctx,
            ResidualAddArgs {
                input: &projection,
                bias: None,
                residual: &residual,
                gate: Some(ResidualGate::UnitOffset(&gate)),
                hidden: &adaptive_hidden,
                bias_order: BiasOrder::SingleRound,
            },
        )?;
        norm::norm(
            &ctx,
            NormArgs {
                input: &adaptive_hidden,
                output: Some(&adaptive_normalized),
                kind: NormKind::Rms,
                affine: Affine::WeightedAdaptive {
                    weight: &weight,
                    scale: &scale,
                    shift: &shift,
                },
                quantized: None,
                numerics: NormNumerics::Flexible,
                eps: 1e-6,
            },
        )?;
        fused_add_norm(
            &ctx,
            FusedAddNormArgs {
                input: &projection,
                bias: None,
                residual: &residual,
                gate: Some(ResidualGate::UnitOffset(&gate)),
                hidden: &adaptive_fused_hidden,
                normalized: Some(&adaptive_fused_normalized),
                kind: NormKind::Rms,
                affine: Affine::WeightedAdaptive {
                    weight: &weight,
                    scale: &scale,
                    shift: &shift,
                },
                quantized: None,
                numerics: NormNumerics::Flexible,
                bias_order: BiasOrder::SingleRound,
                eps: 1e-6,
            },
        )
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(raw_bytes(&separate_hidden), raw_bytes(&fused_hidden));
    assert_eq!(
        raw_bytes(&separate_normalized),
        raw_bytes(&fused_normalized)
    );
    assert_eq!(
        raw_bytes(&adaptive_hidden),
        raw_bytes(&adaptive_fused_hidden)
    );
    assert_eq!(
        raw_bytes(&adaptive_normalized),
        raw_bytes(&adaptive_fused_normalized)
    );
}

#[test]
fn bf16_static_e4m3_norm_preserves_legacy_bits_and_replays_as_one_graph() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(
        0,
        vec![2, 8],
        &[
            1.0, -2.0, 3.0, -4.0, 0.5, -0.75, 1.25, -1.5, -1.0, 2.5, -3.5, 4.5, -0.25, 0.625,
            -1.125, 1.75,
        ],
    );
    let residual = tensor(
        0,
        vec![2, 8],
        &[
            0.5, 0.25, -0.5, 1.0, -1.5, 2.0, 0.75, -0.25, 1.25, -0.5, 0.875, -1.0, 0.375, -0.75,
            1.5, -2.0,
        ],
    );
    let projection_bias = tensor(
        0,
        vec![8],
        &[0.25, -0.5, 0.75, -1.0, 0.125, -0.25, 0.5, -0.75],
    );
    let weight = tensor(
        0,
        vec![8],
        &[1.0, 0.75, 1.25, 0.5, 1.5, 0.875, 1.125, 0.625],
    );
    let bias = tensor(
        0,
        vec![8],
        &[0.0, 0.25, -0.25, 0.5, -0.5, 0.125, -0.125, 0.375],
    );
    let eps = 1e-5;
    let scale = 0.125;

    let rms = zeros_tensor(0, vec![2, 8], DType::F8E4M3);
    let layer = zeros_tensor(0, vec![2, 8], DType::F8E4M3);
    let fused_hidden = zeros_tensor(0, vec![2, 8], DType::BF16);
    let fused_normalized = zeros_tensor(0, vec![2, 8], DType::F8E4M3);
    let graph = crate::capture(&ctx, || {
        norm::norm(
            &ctx,
            NormArgs {
                input: &input,
                output: None,
                kind: NormKind::Rms,
                affine: Affine::Weight {
                    weight: &weight,
                    bias: None,
                    offset: false,
                },
                quantized: Some(QuantOut::StaticE4m3 {
                    output: &rms,
                    scale,
                }),
                numerics: NormNumerics::Flexible,
                eps,
            },
        )?;
        norm::norm(
            &ctx,
            NormArgs {
                input: &input,
                output: None,
                kind: NormKind::Layer,
                affine: Affine::Weight {
                    weight: &weight,
                    bias: Some(&bias),
                    offset: false,
                },
                quantized: Some(QuantOut::StaticE4m3 {
                    output: &layer,
                    scale,
                }),
                numerics: NormNumerics::Flexible,
                eps,
            },
        )?;
        fused_add_norm(
            &ctx,
            FusedAddNormArgs {
                input: &input,
                bias: Some(&projection_bias),
                residual: &residual,
                gate: None,
                hidden: &fused_hidden,
                normalized: None,
                kind: NormKind::Layer,
                affine: Affine::Weight {
                    weight: &weight,
                    bias: Some(&bias),
                    offset: false,
                },
                quantized: Some(QuantOut::StaticE4m3 {
                    output: &fused_normalized,
                    scale,
                }),
                numerics: NormNumerics::Flexible,
                bias_order: BiasOrder::SingleRound,
                eps,
            },
        )
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(
        raw_bytes(&rms),
        [71, 204, 86, 207, 68, 194, 75, 198, 198, 77, 215, 79, 186, 63, 201, 71]
    );
    assert_eq!(
        raw_bytes(&layer),
        [74, 196, 86, 197, 52, 176, 76, 30, 202, 79, 218, 83, 207, 60, 206, 76]
    );
    assert_eq!(
        raw_bytes(&fused_hidden),
        [
            224, 63, 16, 192, 80, 64, 128, 192, 96, 191, 128, 63, 32, 64, 32, 192, 0, 63, 192, 63,
            240, 191, 32, 64, 128, 62, 192, 190, 96, 63, 128, 191,
        ]
    );
    assert_eq!(
        raw_bytes(&fused_normalized),
        [76, 196, 84, 193, 207, 72, 81, 190, 58, 79, 217, 83, 201, 194, 68, 192]
    );
}

#[test]
fn adaptive_layer_norm_dual_output_preserves_legacy_bits_and_graph_replay() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(
        0,
        vec![2, 8],
        &[
            1.0, -2.0, 3.0, -4.0, 0.5, -0.75, 1.25, -1.5, -1.0, 2.5, -3.5, 4.5, -0.25, 0.625,
            -1.125, 1.75,
        ],
    );
    let modulation = tensor(
        0,
        vec![16],
        &[
            0.0, 0.125, -0.25, 0.5, -0.375, 0.25, 0.75, -0.5, 0.25, -0.5, 0.75, -1.0, 0.125, -0.25,
            0.5, -0.75,
        ],
    );
    let eps = 1e-5;
    let scale = 0.125;

    let plain = zeros_tensor(0, vec![2, 8], DType::BF16);
    let native = zeros_tensor(0, vec![2, 8], DType::BF16);
    let quantized = zeros_tensor(0, vec![2, 8], DType::F8E4M3);
    let graph = crate::capture(&ctx, || {
        norm::norm(
            &ctx,
            NormArgs {
                input: &input,
                output: Some(&plain),
                kind: NormKind::Layer,
                affine: Affine::Adaptive { style: &modulation },
                quantized: None,
                numerics: NormNumerics::Flexible,
                eps,
            },
        )?;
        norm::norm(
            &ctx,
            NormArgs {
                input: &input,
                output: Some(&native),
                kind: NormKind::Layer,
                affine: Affine::Adaptive { style: &modulation },
                quantized: Some(QuantOut::StaticE4m3 {
                    output: &quantized,
                    scale,
                }),
                numerics: NormNumerics::Flexible,
                eps,
            },
        )
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    let legacy_native_oracle = [
        99, 63, 182, 191, 251, 63, 108, 192, 190, 62, 4, 191, 234, 63, 133, 191, 189, 190, 0, 63,
        6, 191, 208, 63, 119, 189, 24, 190, 46, 191, 239, 190,
    ];
    assert_eq!(raw_bytes(&plain), legacy_native_oracle);
    assert_eq!(raw_bytes(&native), legacy_native_oracle);
    assert_eq!(
        raw_bytes(&quantized),
        [78, 211, 88, 223, 68, 200, 87, 208, 196, 72, 200, 85, 175, 186, 203, 199]
    );
}

#[test]
fn rowwise_quantized_norm_matches_legacy_entries_and_replays_as_one_graph() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(
        0,
        vec![2, 8],
        &[
            1.0, -2.0, 3.0, -4.0, 0.5, -0.75, 1.25, -1.5, -1.0, 2.5, -3.5, 4.5, -0.25, 0.625,
            -1.125, 1.75,
        ],
    );
    let residual = tensor(
        0,
        vec![2, 8],
        &[
            0.5, 0.25, -0.5, 1.0, -1.5, 2.0, 0.75, -0.25, 1.25, -0.5, 0.875, -1.0, 0.375, -0.75,
            1.5, -2.0,
        ],
    );
    let projection_bias = tensor(
        0,
        vec![8],
        &[0.25, -0.5, 0.75, -1.0, 0.125, -0.25, 0.5, -0.75],
    );
    let weight = tensor(
        0,
        vec![8],
        &[1.0, 0.75, 1.25, 0.5, 1.5, 0.875, 1.125, 0.625],
    );
    let bias = tensor(
        0,
        vec![8],
        &[0.0, 0.25, -0.25, 0.5, -0.5, 0.125, -0.125, 0.375],
    );
    let modulation = tensor(
        0,
        vec![16],
        &[
            0.0, 0.125, -0.25, 0.5, -0.375, 0.25, 0.75, -0.5, 0.25, -0.5, 0.75, -1.0, 0.125, -0.25,
            0.5, -0.75,
        ],
    );
    let eps = 1e-5;

    let rms_values = zeros_tensor(0, vec![2, 12], DType::F8E4M3);
    let rms_scales = zeros_tensor(0, vec![2], DType::F32);
    let fused_hidden = zeros_tensor(0, vec![2, 8], DType::BF16);
    let fused_values = zeros_tensor(0, vec![2, 12], DType::F8E4M3);
    let fused_scales = zeros_tensor(0, vec![2], DType::F32);
    let adaptive_native = zeros_tensor(0, vec![2, 8], DType::BF16);
    let adaptive_values = zeros_tensor(0, vec![2, 8], DType::I8);
    let adaptive_scales = zeros_tensor(0, vec![2], DType::F32);
    let layer_native = zeros_tensor(0, vec![2, 8], DType::BF16);
    let layer_values = zeros_tensor(0, vec![2, 8], DType::I8);
    let layer_scales = zeros_tensor(0, vec![2], DType::F32);

    let graph = crate::capture(&ctx, || {
        norm::norm(
            &ctx,
            NormArgs {
                input: &input,
                output: None,
                kind: NormKind::Rms,
                affine: Affine::Weight {
                    weight: &weight,
                    bias: None,
                    offset: false,
                },
                quantized: Some(QuantOut::RowwiseE4m3 {
                    values: &rms_values,
                    scales: &rms_scales,
                }),
                numerics: NormNumerics::Flexible,
                eps,
            },
        )?;
        fused_add_norm(
            &ctx,
            FusedAddNormArgs {
                input: &input,
                bias: Some(&projection_bias),
                residual: &residual,
                gate: None,
                hidden: &fused_hidden,
                normalized: None,
                kind: NormKind::Rms,
                affine: Affine::Weight {
                    weight: &weight,
                    bias: None,
                    offset: false,
                },
                quantized: Some(QuantOut::RowwiseE4m3 {
                    values: &fused_values,
                    scales: &fused_scales,
                }),
                numerics: NormNumerics::Flexible,
                bias_order: BiasOrder::SingleRound,
                eps,
            },
        )?;
        norm::norm(
            &ctx,
            NormArgs {
                input: &input,
                output: Some(&adaptive_native),
                kind: NormKind::Layer,
                affine: Affine::Adaptive { style: &modulation },
                quantized: Some(QuantOut::RowwiseI8 {
                    values: &adaptive_values,
                    scales: &adaptive_scales,
                }),
                numerics: NormNumerics::Flexible,
                eps,
            },
        )?;
        norm::norm(
            &ctx,
            NormArgs {
                input: &input,
                output: Some(&layer_native),
                kind: NormKind::Layer,
                affine: Affine::Weight {
                    weight: &weight,
                    bias: Some(&bias),
                    offset: false,
                },
                quantized: Some(QuantOut::RowwiseI8 {
                    values: &layer_values,
                    scales: &layer_scales,
                }),
                numerics: NormNumerics::Flexible,
                eps,
            },
        )
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(
        raw_bytes(&rms_values),
        [
            111, 243, 126, 247, 107, 234, 114, 238, 0, 0, 0, 0, 237, 116, 254, 118, 226, 102, 240,
            110, 0, 0, 0, 0,
        ]
    );
    assert_eq!(
        raw_bytes(&rms_scales),
        [122, 215, 131, 59, 202, 161, 135, 59]
    );
    assert_eq!(
        raw_bytes(&fused_hidden),
        [
            224, 63, 16, 192, 80, 64, 128, 192, 96, 191, 128, 63, 32, 64, 32, 192, 0, 63, 192, 63,
            240, 191, 32, 64, 128, 62, 192, 190, 96, 63, 128, 191,
        ]
    );
    assert_eq!(
        raw_bytes(&fused_values),
        [
            116, 244, 126, 246, 241, 108, 122, 243, 0, 0, 0, 0, 108, 117, 254, 119, 105, 232, 116,
            239, 0, 0, 0, 0,
        ]
    );
    assert_eq!(
        raw_bytes(&fused_scales),
        [15, 62, 112, 59, 35, 132, 128, 59]
    );
    assert_eq!(
        raw_bytes(&adaptive_native),
        [
            99, 63, 182, 191, 251, 63, 108, 192, 190, 62, 4, 191, 234, 63, 133, 191, 189, 190, 0,
            63, 6, 191, 208, 63, 119, 189, 24, 190, 46, 191, 239, 190,
        ]
    );
    assert_eq!(
        raw_bytes(&adaptive_values),
        [31, 207, 68, 129, 13, 238, 63, 220, 227, 39, 215, 127, 251, 244, 203, 220]
    );
    assert_eq!(
        raw_bytes(&adaptive_scales),
        [183, 219, 237, 60, 71, 163, 81, 60]
    );
    assert_eq!(
        raw_bytes(&layer_native),
        [
            35, 63, 187, 190, 226, 63, 203, 190, 190, 61, 122, 189, 59, 63, 104, 60, 31, 191, 107,
            63, 24, 192, 176, 63, 114, 191, 72, 62, 98, 191, 59, 63,
        ]
    );
    assert_eq!(
        raw_bytes(&layer_values),
        [46, 230, 127, 227, 7, 252, 53, 1, 223, 49, 129, 74, 205, 10, 209, 39]
    );
    assert_eq!(
        raw_bytes(&layer_scales),
        [143, 199, 99, 60, 101, 50, 153, 60]
    );
}

#[test]
fn decode_rope_uses_tensor_bindings_and_head_major_multihead_cache() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let position_bytes = 1_u32.to_ne_bytes();
    let position_tensor = bytes_tensor(0, vec![1], DType::I32, &position_bytes);
    let position = crate::CudaBuffer::from_tensor(&position_tensor)
        .unwrap()
        .address();

    let input = f32_tensor(0, vec![1, 4], &[1.0, 2.0, 3.0, 4.0]);
    let mut output = zeros_tensor(0, vec![1, 4], DType::F32);
    let graph = crate::capture(&ctx, || {
        decode_rotation(
            &ctx,
            DecodeRotationArgs {
                input: &input,
                output: &mut output,
                heads: 1,
                head_dim: 4,
                theta: 10_000.0,
                position: DecodeRotationPosition::OneDimensional(position),
            },
        )
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    let rotated = f32_values(&output);
    let expected = [
        1.0 * 1.0_f32.cos() - 3.0 * 1.0_f32.sin(),
        2.0 * 0.01_f32.cos() - 4.0 * 0.01_f32.sin(),
        1.0 * 1.0_f32.sin() + 3.0 * 1.0_f32.cos(),
        2.0 * 0.01_f32.sin() + 4.0 * 0.01_f32.cos(),
    ];
    for (actual, expected) in rotated.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }

    let query = tensor(
        0,
        vec![1, 1, 2, 4],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    );
    let key = tensor(
        0,
        vec![1, 1, 2, 4],
        &[9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    );
    let value = tensor(
        0,
        vec![1, 1, 2, 4],
        &[17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
    );
    let mut query_out = zeros_tensor(0, vec![1, 1, 2, 4], DType::BF16);
    let mut key_cache = zeros_tensor(0, vec![1, 2, 3, 4], DType::BF16);
    let mut value_cache = zeros_tensor(0, vec![1, 2, 3, 4], DType::BF16);
    let graph = crate::capture(&ctx, || {
        decode_rope(
            &ctx,
            DecodeRopeArgs::new(
                &query,
                &key,
                &value,
                &mut query_out,
                &mut key_cache,
                &mut value_cache,
                position,
                10_000.0,
            ),
        )
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(
        values(&value_cache),
        [
            0.0, 0.0, 0.0, 0.0, 17.0, 18.0, 19.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            21.0, 22.0, 23.0, 24.0, 0.0, 0.0, 0.0, 0.0,
        ]
    );
}

#[test]
fn vision_position_fuses_interpolation_merge_order_ids_and_graph_replay() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    // A 2x2 learned table whose interpolation onto 4x4 is exactly
    // [2 * row + column, 16 + 2 * row + column].
    let table = tensor(0, vec![4, 2], &[0.0, 16.0, 3.0, 19.0, 6.0, 22.0, 9.0, 25.0]);
    let mut embeddings = zeros_tensor(0, vec![16, 2], DType::BF16);
    let mut position_ids = zeros_tensor(0, vec![16, 2], DType::I32);
    let enqueue = |embeddings: &mut apxinf_core::Tensor, position_ids: &mut apxinf_core::Tensor| {
        vision_position(
            &ctx,
            VisionPositionArgs {
                table: &table,
                embeddings,
                position_ids,
                height: 4,
                width: 4,
                merge_size: 2,
            },
        )
    };
    enqueue(&mut embeddings, &mut position_ids).unwrap();
    ctx.synchronize().unwrap();

    let expected_ids = vec![
        0, 0, 0, 1, 1, 0, 1, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 0, 2, 1, 3, 0, 3, 1, 2, 2, 2, 3, 3, 2,
        3, 3,
    ];
    let expected_embeddings = expected_ids
        .chunks_exact(2)
        .flat_map(|position| {
            let value = (2 * position[0] + position[1]) as f32;
            [value, 16.0 + value]
        })
        .collect::<Vec<_>>();
    assert_eq!(i32_values(&position_ids), expected_ids);
    assert_eq!(values(&embeddings), expected_embeddings);

    let graph = crate::capture(&ctx, || enqueue(&mut embeddings, &mut position_ids)).unwrap();
    let embedding_buffer = crate::CudaBuffer::from_tensor(&embeddings).unwrap();
    let id_buffer = crate::CudaBuffer::from_tensor(&position_ids).unwrap();
    embedding_buffer
        .copy_from_host(&vec![0xff; embedding_buffer.len()])
        .unwrap();
    id_buffer
        .copy_from_host(&vec![0xff; id_buffer.len()])
        .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(i32_values(&position_ids), expected_ids);
    assert_eq!(values(&embeddings), expected_embeddings);
}

#[test]
fn vision_position_rejects_non_square_tables() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let table = tensor(0, vec![3, 2], &[0.0; 6]);
    let mut embeddings = zeros_tensor(0, vec![4, 2], DType::BF16);
    let mut position_ids = zeros_tensor(0, vec![4, 2], DType::I32);
    let error = vision_position(
        &ctx,
        VisionPositionArgs {
            table: &table,
            embeddings: &mut embeddings,
            position_ids: &mut position_ids,
            height: 2,
            width: 2,
            merge_size: 2,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("square table"), "{error}");
}

#[test]
fn pointwise_exact_gelu_runs_on_all_bf16_vector_paths() {
    let ctx = CudaContext::new(0).unwrap();
    for cols in [3, 2, 4] {
        let input = tensor(0, vec![1, cols], &vec![1.0; cols]);
        let bias = tensor(0, vec![cols], &vec![0.0; cols]);
        let mut output = tensor(0, vec![1, cols], &vec![0.0; cols]);
        let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut output);
        args.bias = Some(&bias);
        args.activation = PointwiseActivation::GeluExact;
        pointwise(&ctx, args).unwrap();
        ctx.synchronize().unwrap();
        for actual in values(&output) {
            assert!((actual - 0.83984375).abs() < 1e-6, "cols={cols}: {actual}");
        }
    }
}

#[test]
fn rowwise_swiglu_accepts_logical_bias_with_padded_projection() {
    let ctx = CudaContext::new(0).unwrap();
    let gate_up = tensor(0, vec![1, 12], &[1.0; 12]);
    // The GEMM projection is physically padded to 12 columns, but the
    // checkpoint bias describes only the logical gate/up pair (2 * 5).
    let bias = tensor(0, vec![10], &[0.0; 10]);
    let quantized =
        crate::ops::pointwise::swiglu_quantize_rows(&ctx, &gate_up, Some(&bias), 5, 8).unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(quantized.values.shape().dims(), [1, 8]);
    assert!(f32_values(&quantized.scales)[0].is_finite());
    let output = raw_bytes(&quantized.values);
    assert!(output[..5].iter().any(|value| *value != 0));
    assert_eq!(&output[5..], &[0, 0, 0]);
}

#[test]
fn quantization_rejects_packed_fp4_input() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let input = zeros_tensor(0, vec![1, 16], DType::E2M1Pair);
    for (semantic, dtype) in [
        (QuantizationSemantic::FixedScaleE4m3, DType::F8E4M3),
        (QuantizationSemantic::RowwiseE4m3, DType::F8E4M3),
        (QuantizationSemantic::CastF16ToBf16, DType::BF16),
        (QuantizationSemantic::CastBf16ToF32, DType::F32),
        (QuantizationSemantic::CastF32ToBf16, DType::BF16),
        (QuantizationSemantic::SliceColumnsBf16, DType::BF16),
        (QuantizationSemantic::RowwiseI8, DType::I8),
    ] {
        let mut output = zeros_tensor(0, vec![1, 16], dtype);
        let error =
            quantization(&ctx, QuantizationArgs::new(semantic, &input, &mut output)).unwrap_err();
        assert!(
            error.to_string().contains("requires F16 or BF16")
                || error.to_string().contains("dtype disagrees"),
            "{semantic:?}: {error}"
        );
    }
}

#[test]
fn quantization_casts_bf16_to_f32_exactly() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let source = [1.0, -2.5, 0.125, 65_536.0];
    let input = tensor(0, vec![1, source.len()], &source);
    let mut output = zeros_tensor(0, vec![1, source.len()], DType::F32);
    quantization(
        &ctx,
        QuantizationArgs::new(QuantizationSemantic::CastBf16ToF32, &input, &mut output),
    )
    .unwrap();
    ctx.synchronize().unwrap();
    let buffer = crate::CudaBuffer::from_tensor(&output).unwrap();
    let mut bytes = vec![0u8; source.len() * 4];
    buffer.copy_to_host(&mut bytes).unwrap();
    let actual = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(actual, source);
}

#[test]
fn quantization_casts_f32_to_bf16_with_rounding() {
    use apxinf_core::{DType, Shape};

    let ctx = CudaContext::new(0).unwrap();
    let source = [1.0f32, -2.5, 0.1, 65_536.0];
    let bytes = unsafe {
        std::slice::from_raw_parts(source.as_ptr().cast::<u8>(), std::mem::size_of_val(&source))
    };
    let buffer = crate::CudaBuffer::alloc(bytes.len(), 0).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    let input = buffer
        .as_tensor(Shape::new(vec![1, source.len()]), DType::F32)
        .unwrap();
    let mut output = zeros_tensor(0, vec![1, source.len()], DType::BF16);
    quantization(
        &ctx,
        QuantizationArgs::new(QuantizationSemantic::CastF32ToBf16, &input, &mut output),
    )
    .unwrap();
    ctx.synchronize().unwrap();
    let expected = source
        .iter()
        .map(|value| bf16::from_f32(*value).to_f32())
        .collect::<Vec<_>>();
    assert_eq!(values(&output), expected);
}

#[test]
fn silu_mul_rowwise_i8_dispatch_matches_legacy_and_replays_in_graph() {
    use crate::ffi::abi::quantization as abi;
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let (rows, cols) = if ctx.caps().sm == 87 {
        (156, 6144)
    } else {
        (5, 64)
    };
    let count = rows * cols;
    let gate_values = (0..count)
        .map(|index| ((index as f32) * 0.0031).sin() * 2.0)
        .collect::<Vec<_>>();
    let up_values = (0..count)
        .map(|index| ((index as f32) * 0.0023).cos() * 1.5)
        .collect::<Vec<_>>();
    let gate = tensor(0, vec![rows, cols], &gate_values);
    let up = tensor(0, vec![rows, cols], &up_values);
    let legacy_values = zeros_tensor(0, vec![rows, cols], DType::I8);
    let legacy_scales = zeros_tensor(0, vec![rows], DType::F32);
    let gate_buffer = crate::CudaBuffer::from_tensor(&gate).unwrap();
    let up_buffer = crate::CudaBuffer::from_tensor(&up).unwrap();
    let legacy_values_buffer = crate::CudaBuffer::from_tensor(&legacy_values).unwrap();
    let legacy_scales_buffer = crate::CudaBuffer::from_tensor(&legacy_scales).unwrap();
    unsafe {
        let status = if ctx.caps().sm == 87 {
            abi::apxinf_quantization_silu_mul_quantize_rows_bf16_int8_packed4(
                gate_buffer.ptr(),
                up_buffer.ptr(),
                legacy_values_buffer.ptr(),
                legacy_scales_buffer.ptr(),
                rows as i32,
                cols as i32,
                ctx.stream().handle(),
            )
        } else {
            abi::apxinf_quantization_silu_mul_quantize_rows_bf16_int8(
                gate_buffer.ptr(),
                up_buffer.ptr(),
                legacy_values_buffer.ptr(),
                legacy_scales_buffer.ptr(),
                rows as i32,
                cols as i32,
                ctx.stream().handle(),
            )
        };
        crate::ffi::raw::cuda_runtime::check_cuda(status).unwrap();
    }

    let mut output = zeros_tensor(0, vec![rows, cols], DType::I8);
    let mut scales = zeros_tensor(0, vec![rows], DType::F32);
    let graph = crate::capture(&ctx, || {
        let mut args =
            QuantizationArgs::new(QuantizationSemantic::SiluMulRowwiseI8, &gate, &mut output);
        args.secondary = Some(&up);
        args.scales = Some(&mut scales);
        quantization(&ctx, args)
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(raw_bytes(&output), raw_bytes(&legacy_values));
    assert_eq!(raw_bytes(&scales), raw_bytes(&legacy_scales));
}

#[test]
fn gdn_gated_norm_infers_sequence_dtype_and_optionally_quantizes_in_graph() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let (tokens, heads, head_dim) = (2, 2, 128);
    let dims = vec![tokens, heads, head_dim];
    let elements = tokens * heads * head_dim;
    let input_values = (0..elements)
        .map(|index| ((index as f32) * 0.017).sin())
        .collect::<Vec<_>>();
    let gate_values = (0..elements)
        .map(|index| ((index as f32) * 0.013).cos())
        .collect::<Vec<_>>();
    let weight_values = (0..head_dim)
        .map(|index| 0.75 + index as f32 / head_dim as f32)
        .collect::<Vec<_>>();
    let input = f16_tensor(0, dims.clone(), &input_values);
    let gate = tensor(0, dims.clone(), &gate_values);
    let weight = tensor(0, vec![head_dim], &weight_values);
    let reference = zeros_tensor(0, dims.clone(), DType::BF16);
    gdn_gated_norm(
        &ctx,
        GdnGatedNormArgs {
            input: &input,
            gate: GdnGate::Dense(&gate),
            weight: &weight,
            output: &reference,
            quantized: None,
            epsilon: 1e-6,
        },
    )
    .unwrap();

    let output = zeros_tensor(0, dims.clone(), DType::BF16);
    let quantized = zeros_tensor(0, dims, DType::F8E4M3);
    let graph = crate::capture(&ctx, || {
        gdn_gated_norm(
            &ctx,
            GdnGatedNormArgs {
                input: &input,
                gate: GdnGate::Dense(&gate),
                weight: &weight,
                output: &output,
                quantized: Some(GdnQuantizedOutput {
                    tensor: &quantized,
                    input_scale: 0.125,
                }),
                epsilon: 1e-6,
            },
        )
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(values(&output), values(&reference));
    let quantized_buffer = crate::CudaBuffer::from_tensor(&quantized).unwrap();
    let mut bytes = vec![0_u8; quantized_buffer.len()];
    quantized_buffer.copy_to_host(&mut bytes).unwrap();
    assert!(bytes.iter().any(|byte| *byte != 0));
}

#[test]
fn gdn_gated_norm_expresses_strided_gate_layout_in_one_entry() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let (tokens, heads, head_dim) = (2, 2, 8);
    let rows = tokens * heads;
    let input_values = (0..rows * head_dim)
        .map(|index| ((index as f32) * 0.17).sin())
        .collect::<Vec<_>>();
    let dense_gate_values = (0..rows * head_dim)
        .map(|index| ((index as f32) * 0.11).cos())
        .collect::<Vec<_>>();
    let weight_values = (0..head_dim)
        .map(|index| 0.75 + index as f32 / head_dim as f32)
        .collect::<Vec<_>>();
    let mut strided_gate_values = vec![0.0; tokens * heads * head_dim * 2];
    for token in 0..tokens {
        let source = token * heads * head_dim;
        let destination = token * heads * head_dim * 2 + heads * head_dim;
        strided_gate_values[destination..destination + heads * head_dim]
            .copy_from_slice(&dense_gate_values[source..source + heads * head_dim]);
    }

    let input = tensor(0, vec![rows, head_dim], &input_values);
    let dense_gate = tensor(0, vec![rows, head_dim], &dense_gate_values);
    let strided_gate = tensor(0, vec![tokens, heads * head_dim * 2], &strided_gate_values);
    let weight = tensor(0, vec![head_dim], &weight_values);
    let dense = zeros_tensor(0, vec![rows, head_dim], DType::BF16);
    let strided = zeros_tensor(0, vec![rows, head_dim], DType::BF16);

    gdn_gated_norm(
        &ctx,
        GdnGatedNormArgs {
            input: &input,
            gate: GdnGate::Dense(&dense_gate),
            weight: &weight,
            output: &dense,
            quantized: None,
            epsilon: 1e-6,
        },
    )
    .unwrap();
    let graph = crate::capture(&ctx, || {
        gdn_gated_norm(
            &ctx,
            GdnGatedNormArgs {
                input: &input,
                gate: GdnGate::Strided {
                    tensor: &strided_gate,
                    column_offset: heads * head_dim,
                    heads,
                },
                weight: &weight,
                output: &strided,
                quantized: None,
                epsilon: 1e-6,
            },
        )
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    for (dense, strided) in values(&dense).iter().zip(values(&strided)) {
        assert!((dense - strided).abs() <= 0.016, "{dense} != {strided}");
    }
}

fn u32_buffer(device: usize, values: &[u32]) -> crate::CudaBuffer {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer = crate::CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
}

#[test]
fn standalone_rope_semantics_share_the_family_dispatch_and_replay() {
    let ctx = CudaContext::new(0).unwrap();
    let input_values = [0.25, -0.5, 0.75, -1.0, 0.125, 0.375, -0.625, 0.875];
    let input = tensor(0, vec![2, 1, 4], &input_values);

    let eager = rope_apply_batched(&ctx, &input, 1, 4, 10_000.0, 0).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(&raw_bytes(&eager)[..8], &raw_bytes(&input)[..8]);

    let mrope =
        rope_apply_mrope(&ctx, &input, 1, 4, 10_000.0, [0, 1, 1], &[0, 0, 0, 0, 0, 0]).unwrap();
    let vision = rope_apply_vision_2d(&ctx, &input, 1, 4, 10_000.0, &[0, 0, 0, 0]).unwrap();
    let f16_input = f16_tensor(0, vec![1, 1, 4], &[0.25, -0.5, 0.75, -1.0]);
    let f16_output = rope::apply_half_split(&ctx, &f16_input, 1, 4, 10_000.0, 0).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(raw_bytes(&mrope), raw_bytes(&input));
    assert_eq!(raw_bytes(&vision), raw_bytes(&input));
    assert_eq!(raw_bytes(&f16_output), raw_bytes(&f16_input));

    let session = ExecutionSession::with_capacity(4096, 0).unwrap();
    prepare_with_session(&session, || {
        rope_apply_batched(&ctx, &input, 1, 4, 10_000.0, 0).map(|_| ())
    })
    .unwrap();
    let mut captured = None;
    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            captured = Some(rope_apply_batched(&ctx, &input, 1, 4, 10_000.0, 0)?);
            Ok(())
        })
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(raw_bytes(captured.as_ref().unwrap()), raw_bytes(&eager));
}

#[test]
fn rope_half_split_f16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let input_values = [0.25, -0.5, 0.75, -1.0, 0.125, 0.375, -0.625, 0.875];
    let input = f16_tensor(0, vec![2, 1, 4], &input_values);
    let output = rope::apply_half_split(&ctx, &input, 1, 4, 10_000.0, 3).unwrap();
    ctx.synchronize().unwrap();

    let mut expected = Vec::with_capacity(input_values.len());
    for (token, values) in input_values.chunks_exact(4).enumerate() {
        let position = (token + 3) as f32;
        let mut rotated = [0.0; 4];
        for index in 0..2 {
            let angle = position / 10_000.0_f32.powf((2 * index) as f32 / 4.0);
            let (sin, cos) = angle.sin_cos();
            rotated[index] = values[index] * cos - values[index + 2] * sin;
            rotated[index + 2] = values[index] * sin + values[index + 2] * cos;
        }
        expected.extend(rotated);
    }

    for (actual, expected) in f16_values(&output).iter().zip(expected) {
        assert!((actual - expected).abs() <= 2e-3, "{actual} != {expected}");
    }
}

#[test]
fn gather_replace_rows_preserves_unmapped_rows() {
    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(0, vec![3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let replacement = tensor(0, vec![1, 2], &[7.0, 8.0]);
    let row_map = u32_buffer(0, &[u32::MAX, 0, u32::MAX]);
    let mut output = zeros_tensor(0, vec![3, 2], apxinf_core::DType::BF16);
    let mut args = GatherArgs::new(GatherSemantic::ReplaceRows, &input, &mut output);
    args.replacement = Some(&replacement);
    args.row_map = Some(&row_map);
    gather(&ctx, args).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&output), vec![1.0, 2.0, 7.0, 8.0, 5.0, 6.0]);
}

#[test]
fn multimodal_rope_rejects_unimplemented_section_layouts() {
    let ctx = CudaContext::new(0).unwrap();
    let mut data = zeros_tensor(0, vec![1, 1, 256], apxinf_core::DType::BF16);
    let positions = zeros_tensor(0, vec![1, 3], apxinf_core::DType::I32);
    let error = multimodal_rope(
        &ctx,
        MultimodalRopeArgs {
            data: &mut data,
            positions: &positions,
            rotary_dim: 64,
            sections: [10, 11, 11],
            theta: 1_000_000.0,
        },
    )
    .unwrap_err();
    assert!(error.to_string().contains("validated d3-mini"), "{error}");
}

#[test]
fn vision_rope_position_and_table_bindings_share_one_graph_safe_family_entry() {
    let ctx = CudaContext::new(0).unwrap();
    let (tokens, heads, head_dim) = (2, 2, 4);
    let width = 3 * heads * head_dim;
    let qkv_values = (0..tokens * width)
        .map(|index| ((index % 17) as f32 - 8.0) / 8.0)
        .collect::<Vec<_>>();
    let bias_values = (0..width)
        .map(|index| ((index % 7) as f32 - 3.0) / 32.0)
        .collect::<Vec<_>>();
    let qkv = tensor(0, vec![tokens, width], &qkv_values);
    let bias = tensor(0, vec![width], &bias_values);
    let positions = u32_buffer(0, &[1, 3, 2, 4]);
    let theta = 10_000.0;
    let table =
        rope::prepare_vision_2d_rotation_table(&ctx, tokens, head_dim, theta, &positions).unwrap();

    let output_shape = Shape::new(vec![tokens, heads, head_dim]);
    let mut direct_q = ctx
        .allocate_output(output_shape.clone(), DType::BF16)
        .unwrap();
    let mut direct_k = ctx
        .allocate_output(output_shape.clone(), DType::BF16)
        .unwrap();
    let mut direct_v = ctx
        .allocate_output(output_shape.clone(), DType::BF16)
        .unwrap();
    rope::split_qkv_vision(
        &ctx,
        rope::VisionQkvRopeArgs {
            qkv: &qkv,
            bias: &bias,
            q: &mut direct_q,
            k: &mut direct_k,
            v: &mut direct_v,
            heads,
            head_dim,
            rotation: rope::VisionRotation::Positions {
                ids: &positions,
                theta,
            },
        },
    )
    .unwrap();
    let mut table_q = ctx
        .allocate_output(output_shape.clone(), DType::BF16)
        .unwrap();
    let mut table_k = ctx
        .allocate_output(output_shape.clone(), DType::BF16)
        .unwrap();
    let mut table_v = ctx
        .allocate_output(output_shape.clone(), DType::BF16)
        .unwrap();
    rope::split_qkv_vision(
        &ctx,
        rope::VisionQkvRopeArgs {
            qkv: &qkv,
            bias: &bias,
            q: &mut table_q,
            k: &mut table_k,
            v: &mut table_v,
            heads,
            head_dim,
            rotation: rope::VisionRotation::Table(&table),
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(raw_bytes(&direct_q), raw_bytes(&table_q));
    assert_eq!(raw_bytes(&direct_k), raw_bytes(&table_k));
    assert_eq!(raw_bytes(&direct_v), raw_bytes(&table_v));

    let session = ExecutionSession::with_capacity(4096, 0).unwrap();
    prepare_with_session(&session, || {
        let mut query = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
        let mut key = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
        let mut value = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
        rope::split_qkv_vision(
            &ctx,
            rope::VisionQkvRopeArgs {
                qkv: &qkv,
                bias: &bias,
                q: &mut query,
                k: &mut key,
                v: &mut value,
                heads,
                head_dim,
                rotation: rope::VisionRotation::Table(&table),
            },
        )
    })
    .unwrap();
    let mut captured_output = None;
    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut query = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
            let mut key = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
            let mut value = ctx.allocate_output(output_shape.clone(), DType::BF16)?;
            rope::split_qkv_vision(
                &ctx,
                rope::VisionQkvRopeArgs {
                    qkv: &qkv,
                    bias: &bias,
                    q: &mut query,
                    k: &mut key,
                    v: &mut value,
                    heads,
                    head_dim,
                    rotation: rope::VisionRotation::Table(&table),
                },
            )?;
            captured_output = Some((query, key, value));
            Ok(())
        })
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    let (captured_q, captured_k, captured_v) = captured_output.unwrap();
    assert_eq!(raw_bytes(&direct_q), raw_bytes(&captured_q));
    assert_eq!(raw_bytes(&direct_k), raw_bytes(&captured_k));
    assert_eq!(raw_bytes(&direct_v), raw_bytes(&captured_v));
}

#[test]
fn rounded_axial_rope_preserves_legacy_rounding_and_graph_replay() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let (tokens, heads, head_dim) = (2, 1, 4);
    let width = 3 * heads * head_dim;
    let qkv_values = (0..tokens * width)
        .map(|index| ((index % 13) as f32 - 6.0) / 7.0)
        .collect::<Vec<_>>();
    let bias_values = (0..width)
        .map(|index| ((index % 5) as f32 - 2.0) / 31.0)
        .collect::<Vec<_>>();
    let qkv = tensor(0, vec![tokens, width], &qkv_values);
    let bias = tensor(0, vec![width], &bias_values);
    let position_values = [1_i32, 3, 2, 4];
    let position_bytes = position_values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let positions = bytes_tensor(0, vec![tokens, 2], DType::I32, &position_bytes);

    let mut eager_q = zeros_tensor(0, vec![tokens, heads, head_dim], DType::BF16);
    let mut eager_k = zeros_tensor(0, vec![tokens, heads, head_dim], DType::BF16);
    let mut eager_v = zeros_tensor(0, vec![tokens, heads, head_dim], DType::BF16);
    split_qkv_axial_rope(
        &ctx,
        AxialRopeArgs {
            qkv: &qkv,
            bias: Some(&bias),
            positions: &positions,
            query: &mut eager_q,
            key: &mut eager_k,
            value: &mut eager_v,
            theta: 10_000.0,
            numerics: AxialRopeNumerics::RoundedInput,
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();

    // V is not rotated. This checks the legacy BF16 add-and-round point
    // independently from graph determinism below.
    let projection = heads * head_dim;
    let expected_v = (0..tokens)
        .flat_map(|token| {
            let qkv_values = &qkv_values;
            let bias_values = &bias_values;
            (0..projection).map(move |col| {
                let input =
                    bf16::from_f32(qkv_values[token * width + 2 * projection + col]).to_f32();
                let bias = bf16::from_f32(bias_values[2 * projection + col]).to_f32();
                bf16::from_f32(input + bias).to_f32()
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(values(&eager_v), expected_v);

    let mut captured_q = zeros_tensor(0, vec![tokens, heads, head_dim], DType::BF16);
    let mut captured_k = zeros_tensor(0, vec![tokens, heads, head_dim], DType::BF16);
    let mut captured_v = zeros_tensor(0, vec![tokens, heads, head_dim], DType::BF16);
    let graph = crate::capture(&ctx, || {
        split_qkv_axial_rope(
            &ctx,
            AxialRopeArgs {
                qkv: &qkv,
                bias: Some(&bias),
                positions: &positions,
                query: &mut captured_q,
                key: &mut captured_k,
                value: &mut captured_v,
                theta: 10_000.0,
                numerics: AxialRopeNumerics::RoundedInput,
            },
        )
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(raw_bytes(&eager_q), raw_bytes(&captured_q));
    assert_eq!(raw_bytes(&eager_k), raw_bytes(&captured_k));
    assert_eq!(raw_bytes(&eager_v), raw_bytes(&captured_v));
}

#[test]
fn mrope_cache_split_honors_offset_and_replays_bit_identically() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let (tokens, q_heads, kv_heads, head_dim, capacity) = (2, 2, 1, 4, 4);
    let q_width = q_heads * head_dim;
    let kv_width = kv_heads * head_dim;
    let width = q_width + 2 * kv_width;
    let qkv_values = (0..tokens * width)
        .map(|index| ((index % 19) as f32 - 9.0) / 9.0)
        .collect::<Vec<_>>();
    let bias_values = (0..width)
        .map(|index| ((index % 7) as f32 - 3.0) / 37.0)
        .collect::<Vec<_>>();
    let qkv = tensor(0, vec![tokens, width], &qkv_values);
    let bias = tensor(0, vec![width], &bias_values);
    let position_values = [1_i32, 3, 5, 2, 4, 6];
    let position_bytes = position_values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let positions = bytes_tensor(0, vec![tokens, 3], DType::I32, &position_bytes);
    let cache_values = vec![-7.0; capacity * kv_width];

    let run = |query: &mut apxinf_core::Tensor,
               key_cache: &mut apxinf_core::Tensor,
               value_cache: &mut apxinf_core::Tensor| {
        split_qkv_mrope_cache(
            &ctx,
            MropeCacheArgs {
                qkv: &qkv,
                bias: Some(&bias),
                positions: &positions,
                query,
                key_cache,
                value_cache,
                theta: 10_000.0,
                sections: [1, 1, 0],
                cache_offset: 1,
            },
        )
    };

    let mut eager_q = zeros_tensor(0, vec![tokens, q_heads, head_dim], DType::BF16);
    let mut eager_k = tensor(0, vec![capacity, kv_heads, head_dim], &cache_values);
    let mut eager_v = tensor(0, vec![capacity, kv_heads, head_dim], &cache_values);
    run(&mut eager_q, &mut eager_k, &mut eager_v).unwrap();
    ctx.synchronize().unwrap();

    let key_values = values(&eager_k);
    let value_values = values(&eager_v);
    assert_eq!(&key_values[..kv_width], &cache_values[..kv_width]);
    assert_eq!(&key_values[3 * kv_width..], &cache_values[3 * kv_width..]);
    assert_eq!(&value_values[..kv_width], &cache_values[..kv_width]);
    assert_eq!(&value_values[3 * kv_width..], &cache_values[3 * kv_width..]);
    let expected_v = (0..tokens)
        .flat_map(|token| {
            let qkv_values = &qkv_values;
            let bias_values = &bias_values;
            (0..kv_width).map(move |col| {
                let input =
                    bf16::from_f32(qkv_values[token * width + q_width + kv_width + col]).to_f32();
                let bias = bf16::from_f32(bias_values[q_width + kv_width + col]).to_f32();
                bf16::from_f32(input + bias).to_f32()
            })
        })
        .collect::<Vec<_>>();
    assert_eq!(&value_values[kv_width..3 * kv_width], expected_v);

    let mut captured_q = zeros_tensor(0, vec![tokens, q_heads, head_dim], DType::BF16);
    let mut captured_k = tensor(0, vec![capacity, kv_heads, head_dim], &cache_values);
    let mut captured_v = tensor(0, vec![capacity, kv_heads, head_dim], &cache_values);
    let graph = crate::capture(&ctx, || {
        run(&mut captured_q, &mut captured_k, &mut captured_v)
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(raw_bytes(&eager_q), raw_bytes(&captured_q));
    assert_eq!(raw_bytes(&eager_k), raw_bytes(&captured_k));
    assert_eq!(raw_bytes(&eager_v), raw_bytes(&captured_v));
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0; batch * query_tokens * query_heads * head_dim];
    for batch_index in 0..batch {
        for query_token in 0..query_tokens {
            for query_head in 0..query_heads {
                let kv_head = query_head / (query_heads / kv_heads);
                let valid_keys = if causal {
                    (query_token + key_tokens - query_tokens + 1).min(key_tokens)
                } else {
                    key_tokens
                };
                let mut scores = Vec::with_capacity(valid_keys);
                for key_token in 0..valid_keys {
                    let mut score = 0.0;
                    for dimension in 0..head_dim {
                        let q = (((batch_index * query_tokens + query_token) * query_heads
                            + query_head)
                            * head_dim)
                            + dimension;
                        let k = (((batch_index * key_tokens + key_token) * kv_heads + kv_head)
                            * head_dim)
                            + dimension;
                        score += query[q] * key[k];
                    }
                    scores.push(score * scale);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = scores.iter().map(|score| (score - maximum).exp()).sum();
                for dimension in 0..head_dim {
                    let result = scores
                        .iter()
                        .enumerate()
                        .map(|(key_token, score)| {
                            let v = (((batch_index * key_tokens + key_token) * kv_heads + kv_head)
                                * head_dim)
                                + dimension;
                            ((*score - maximum).exp() / denominator) * value[v]
                        })
                        .sum();
                    let output_index = (((batch_index * query_tokens + query_token) * query_heads
                        + query_head)
                        * head_dim)
                        + dimension;
                    output[output_index] = result;
                }
            }
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
pub(super) fn kv_cache_attention_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    query_tokens: usize,
    valid_key_tokens: usize,
    key_capacity: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query_start: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0; batch * query_tokens * query_heads * head_dim];
    for batch_index in 0..batch {
        for query_token in 0..query_tokens {
            for query_head in 0..query_heads {
                let kv_head = query_head / (query_heads / kv_heads);
                let keys = if causal {
                    (query_start + query_token + 1).min(valid_key_tokens)
                } else {
                    valid_key_tokens
                };
                let mut scores = Vec::with_capacity(keys);
                for key_token in 0..keys {
                    let mut score = 0.0;
                    for dimension in 0..head_dim {
                        let q = (((batch_index * query_tokens + query_token) * query_heads
                            + query_head)
                            * head_dim)
                            + dimension;
                        let k = (((batch_index * key_capacity + key_token) * kv_heads + kv_head)
                            * head_dim)
                            + dimension;
                        score += query[q] * key[k];
                    }
                    scores.push(score * scale);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = scores.iter().map(|score| (score - maximum).exp()).sum();
                for dimension in 0..head_dim {
                    let result = scores
                        .iter()
                        .enumerate()
                        .map(|(key_token, score)| {
                            let v = (((batch_index * key_capacity + key_token) * kv_heads
                                + kv_head)
                                * head_dim)
                                + dimension;
                            ((*score - maximum).exp() / denominator) * value[v]
                        })
                        .sum();
                    let output_index = (((batch_index * query_tokens + query_token) * query_heads
                        + query_head)
                        * head_dim)
                        + dimension;
                    output[output_index] = result;
                }
            }
        }
    }
    output
}

#[test]
fn gemm_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm(&ctx, args).unwrap();
    assert!(values(&out).iter().all(|&value| value == 3.0));
}

#[test]
fn gemm_geglu_is_a_separate_semantic_domain() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (3, 5, 1024);
    let a_values = vec![0.25; m * k];
    let b_values: Vec<_> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) / 16.0).collect();
    let a = tensor(0, vec![m, k], &a_values);
    let b = tensor(0, vec![k, n], &b_values);
    let mut out = tensor(0, vec![m, n / 2], &vec![0.0; m * n / 2]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_geglu(&ctx, GemmGegluArgs { gemm: args }).unwrap();
    let actual = values(&out);
    assert_eq!(actual.len(), m * n / 2);
    for row in 0..m {
        for column in 0..n / 2 {
            let dot = |target| {
                bf16::from_f32(
                    (0..k)
                        .map(|inner| a_values[row * k + inner] * b_values[inner * n + target])
                        .sum(),
                )
                .to_f32()
            };
            let gate = dot(column);
            let expected = 0.5
                * gate
                * (1.0 + (0.79788456 * (gate + 0.044715 * gate.powi(3))).tanh())
                * dot(column + n / 2);
            assert!((actual[row * n / 2 + column] - expected).abs() < 0.01);
        }
    }
}

#[test]
fn gemm_swiglu_keeps_pair_layout_and_provider_padding_private() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![1, 1], &[1.0]);
    let mut weights = vec![1.0; 512];
    weights[256..].fill(2.0);
    let b = tensor(0, vec![1, 512], &weights);
    let mut out = tensor(0, vec![1, 256], &[0.0; 256]);
    let session = crate::ExecutionSession::with_capacity(4096, 0).unwrap();
    let enqueue = || {
        let mut gemm = GemmArgs::new(&a, &b, &mut out);
        gemm.weight_layout = GemmWeightLayout::GateUpPairs { pair_width: 256 };
        gemm.policy.online_tune = false;
        gemm_swiglu(&ctx, GemmSwiGluArgs::new(gemm))
    };
    prepare_with_session(&session, enqueue).unwrap();
    ctx.synchronize().unwrap();

    let rounded_silu = bf16::from_f32(1.0 / (1.0 + (-1.0f32).exp())).to_f32();
    let expected = bf16::from_f32(rounded_silu * 2.0).to_f32();
    let expected_output = values(&out);
    assert!(expected_output
        .iter()
        .all(|actual| (*actual - expected).abs() < 1e-3));

    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut gemm = GemmArgs::new(&a, &b, &mut out);
            gemm.weight_layout = GemmWeightLayout::GateUpPairs { pair_width: 256 };
            gemm.policy.online_tune = false;
            gemm_swiglu(&ctx, GemmSwiGluArgs::new(gemm))
        })
    })
    .unwrap();
    let sentinel = std::iter::repeat(bf16::from_f32(-123.0).to_bits())
        .take(256)
        .flat_map(u16::to_ne_bytes)
        .collect::<Vec<_>>();
    crate::CudaBuffer::from_tensor(&out)
        .unwrap()
        .copy_from_host(&sentinel)
        .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), expected_output);
}

#[test]
fn gemm_bias_gelu_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let bias = tensor(0, vec![4], &[0.5; 4]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_bias_gelu(
        &ctx,
        GemmBiasGeluArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();
    let output = values(&out);
    let x: f32 = 3.5;
    let expected = 0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x.powi(3))).tanh());
    assert!(output.iter().all(|value| (*value - expected).abs() < 0.02));
}

#[test]
fn gemm_bias_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let bias = tensor(0, vec![4], &[0.5, -0.5, 1.0, -1.0]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_bias(
        &ctx,
        GemmBiasArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();
    let expected = [3.5, 2.5, 4.0, 2.0, 3.5, 2.5, 4.0, 2.0];
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.01);
    }
}

#[test]
fn gemm_bias_rounds_once_after_bias_for_bf16_output() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![1, 1], &[1.0078125]);
    let b = tensor(0, vec![1, 1], &[1.0078125]);
    let bias = tensor(0, vec![1], &[0.00390625]);
    let mut out = zeros_tensor(0, vec![1, 1], DType::BF16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_bias(
        &ctx,
        GemmBiasArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();

    // round_bf16(a*b + bias) is 1.0234375. Rounding a*b to BF16 first
    // lands on a tie after adding bias and incorrectly returns 1.015625.
    assert_eq!(values(&out), [1.0234375]);
}

#[test]
fn gemm_bias_accepts_output_major_checkpoint_weights_and_replays() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(0, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let weight = tensor(
        0,
        vec![4, 3],
        &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
    );
    let bias = tensor(0, vec![4], &[0.5, -0.5, 1.0, -1.0]);
    let mut out = zeros_tensor(0, vec![2, 4], DType::BF16);
    let session = crate::ExecutionSession::with_capacity(4096, 0).unwrap();
    let enqueue = || {
        let mut gemm = GemmArgs::new(&input, &weight, &mut out);
        gemm.weight_layout = GemmWeightLayout::OutputMajor;
        gemm.policy.online_tune = false;
        gemm_bias(&ctx, GemmBiasArgs { gemm, bias: &bias })
    };
    prepare_with_session(&session, enqueue).unwrap();
    ctx.synchronize().unwrap();
    let expected = [1.5, 1.5, 4.0, 5.0, 4.5, 4.5, 7.0, 14.0];
    assert_eq!(values(&out), expected);

    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut gemm = GemmArgs::new(&input, &weight, &mut out);
            gemm.weight_layout = GemmWeightLayout::OutputMajor;
            gemm.policy.online_tune = false;
            gemm_bias(&ctx, GemmBiasArgs { gemm, bias: &bias })
        })
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), expected);
}

#[test]
fn output_major_gemm_bias_matches_the_legacy_cublas_rounding_path() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (7, 64, 48);
    let input_values = (0..m * k)
        .map(|index| ((index * 17 % 41) as f32 - 20.0) / 32.0)
        .collect::<Vec<_>>();
    let weight_values = (0..n * k)
        .map(|index| ((index * 29 % 53) as f32 - 26.0) / 64.0)
        .collect::<Vec<_>>();
    let bias_values = (0..n)
        .map(|index| ((index * 7 % 19) as f32 - 9.0) / 128.0)
        .collect::<Vec<_>>();
    let input = tensor(0, vec![m, k], &input_values);
    let weight = tensor(0, vec![n, k], &weight_values);
    let bias = tensor(0, vec![n], &bias_values);

    let rounded_bias = values(&bias);
    let accumulator_values = (0..m)
        .flat_map(|_| rounded_bias.iter().copied())
        .collect::<Vec<_>>();
    let accumulator = f32_tensor(0, vec![m, n], &accumulator_values);
    let input_buffer = crate::CudaBuffer::from_tensor(&input).unwrap();
    let weight_buffer = crate::CudaBuffer::from_tensor(&weight).unwrap();
    let accumulator_buffer = crate::CudaBuffer::from_tensor(&accumulator).unwrap();
    ctx.cublas()
        .gemm_bf16_f32_ex(
            crate::cublas::CublasTranspose::None,
            crate::cublas::CublasTranspose::Transpose,
            m,
            n,
            k,
            1.0,
            &input_buffer,
            k as i32,
            &weight_buffer,
            k as i32,
            1.0,
            &accumulator_buffer,
            n as i32,
        )
        .unwrap();
    let baseline = zeros_tensor(0, vec![m, n], DType::BF16);
    quantization::cast(&ctx, &accumulator, &baseline).unwrap();

    let mut actual = zeros_tensor(0, vec![m, n], DType::BF16);
    let mut gemm = GemmArgs::new(&input, &weight, &mut actual);
    gemm.weight_layout = GemmWeightLayout::OutputMajor;
    gemm.policy.online_tune = false;
    gemm_bias(&ctx, GemmBiasArgs { gemm, bias: &bias }).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&actual), values(&baseline));
}

#[test]
fn legacy_cublas_numerics_are_selected_through_gemm_policy() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (7, 64, 48);
    let input_values = (0..m * k)
        .map(|index| ((index * 17 % 41) as f32 - 20.0) / 32.0)
        .collect::<Vec<_>>();
    let weight_values = (0..k * n)
        .map(|index| ((index * 29 % 53) as f32 - 26.0) / 64.0)
        .collect::<Vec<_>>();
    let input = tensor(0, vec![m, k], &input_values);
    let weight = tensor(0, vec![k, n], &weight_values);
    let baseline = zeros_tensor(0, vec![m, n], DType::BF16);
    let input_buffer = crate::CudaBuffer::from_tensor(&input).unwrap();
    let weight_buffer = crate::CudaBuffer::from_tensor(&weight).unwrap();
    let baseline_buffer = crate::CudaBuffer::from_tensor(&baseline).unwrap();
    ctx.cublas()
        .gemm(
            DType::BF16,
            m,
            n,
            k,
            1.0,
            &input_buffer,
            &weight_buffer,
            0.0,
            &baseline_buffer,
        )
        .unwrap();

    let mut actual = zeros_tensor(0, vec![m, n], DType::BF16);
    let session = crate::ExecutionSession::with_capacity(4096, 0).unwrap();
    let enqueue = || {
        let mut gemm = GemmArgs::new(&input, &weight, &mut actual);
        gemm.policy.numerics = GemmNumerics::LegacyCublas;
        gemm.policy.online_tune = false;
        crate::ops::gemm(&ctx, gemm)
    };
    prepare_with_session(&session, enqueue).unwrap();
    ctx.synchronize().unwrap();
    let expected = values(&baseline);
    assert_eq!(values(&actual), expected);

    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut gemm = GemmArgs::new(&input, &weight, &mut actual);
            gemm.policy.numerics = GemmNumerics::LegacyCublas;
            gemm.policy.online_tune = false;
            crate::ops::gemm(&ctx, gemm)
        })
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&actual), expected);
}

#[test]
fn packed_qkv_attention_uses_static_metadata_and_replays_numeric_output() {
    let ctx = CudaContext::new(0).unwrap();
    if ctx.caps().sm < 100 {
        return;
    }
    let (batch, tokens, heads, head_dim) = (2, 256, 16, 72);
    let elements = batch * tokens * heads * head_dim;
    let mut packed_values = Vec::with_capacity(3 * elements);
    let mut expected = Vec::with_capacity(elements);
    for batch_index in 0..batch {
        for token in 0..tokens {
            for plane in 0..3 {
                for head in 0..heads {
                    for dimension in 0..head_dim {
                        packed_values.push(match plane {
                            0 => 0.25,
                            1 => 0.5,
                            _ => (batch_index * 8 + token % 8 + head + dimension % 4) as f32 / 32.0,
                        });
                    }
                }
            }
            for head in 0..heads {
                for dimension in 0..head_dim {
                    expected.push(((batch_index * 8 + head + dimension % 4) as f32 + 3.5) / 32.0);
                }
            }
        }
    }
    let qkv = f16_tensor(0, vec![batch, tokens, 3, heads, head_dim], &packed_values);
    let mut out = f16_tensor(
        0,
        vec![batch, tokens, heads, head_dim],
        &vec![0.0; elements],
    );
    let normalized =
        super::attention_contracts::normalize(&ctx, AttentionArgs::from_packed_qkv(&qkv, &mut out))
            .unwrap();
    assert_eq!(normalized.spec.semantic, 0);
    assert_eq!(normalized.spec.qkv_layout, 1);
    assert!(normalized.bindings.decode_meta.is_null());
    assert_eq!(normalized.spec.dynamic_decode, 0);
    assert!(normalized.resources.is_empty());
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();
    prepare_with_session(&session, || {
        let mut args = AttentionArgs::from_packed_qkv(&qkv, &mut out);
        args.policy.online_tune = false;
        args.policy.cache_dir = Some(String::new());
        attention(&ctx, args)
    })
    .unwrap();
    let eager = crate::transfers::to_cpu(&out)
        .unwrap()
        .to_f32_vec()
        .unwrap();
    for (actual, reference) in eager.iter().zip(&expected) {
        assert!(
            (actual - reference).abs() < 0.002,
            "{actual} != {reference}"
        );
    }
    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut args = AttentionArgs::from_packed_qkv(&qkv, &mut out);
            args.policy.online_tune = false;
            args.policy.cache_dir = Some(String::new());
            attention(&ctx, args)
        })
    })
    .unwrap();
    let sentinel: Vec<_> = (0..elements)
        .flat_map(|_| half::f16::from_f32(-123.0).to_bits().to_ne_bytes())
        .collect();
    let output = crate::CudaBuffer::from_tensor(&out).unwrap();
    for _ in 0..2 {
        output.copy_from_host(&sentinel).unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        let replayed = crate::transfers::to_cpu(&out)
            .unwrap()
            .to_f32_vec()
            .unwrap();
        assert_eq!(replayed, eager);
    }
}

#[test]
fn attention_supports_mha_gqa_mqa_and_causal_masking() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, query_tokens, key_tokens, query_heads, head_dim) = (1, 2, 3, 4, 2);
    let query_values: Vec<_> = (0..batch * query_tokens * query_heads * head_dim)
        .map(|index| (index as f32 - 7.0) / 8.0)
        .collect();
    let scale = 0.5;

    for (kv_heads, causal) in [(4, false), (2, false), (1, false), (2, true)] {
        let key_values: Vec<_> = (0..batch * key_tokens * kv_heads * head_dim)
            .map(|index| ((index * 3 % 11) as f32 - 5.0) / 8.0)
            .collect();
        let value_values: Vec<_> = (0..batch * key_tokens * kv_heads * head_dim)
            .map(|index| ((index * 5 % 13) as f32 - 6.0) / 8.0)
            .collect();
        let query = tensor(
            0,
            vec![batch, query_tokens, query_heads, head_dim],
            &query_values,
        );
        let key = tensor(0, vec![batch, key_tokens, kv_heads, head_dim], &key_values);
        let value = tensor(
            0,
            vec![batch, key_tokens, kv_heads, head_dim],
            &value_values,
        );
        let mut out = tensor(
            0,
            vec![batch, query_tokens, query_heads, head_dim],
            &vec![0.0; query_values.len()],
        );
        let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
        args.scale = scale;
        args.mask = if causal {
            AttentionMask::Causal
        } else {
            AttentionMask::None
        };
        args.policy.online_tune = false;
        attention(&ctx, args).unwrap();

        let expected = attention_reference(
            &query_values,
            &key_values,
            &value_values,
            batch,
            query_tokens,
            key_tokens,
            query_heads,
            kv_heads,
            head_dim,
            scale,
            causal,
        );
        for (index, (actual, expected)) in values(&out).iter().zip(expected).enumerate() {
            assert!(
                (*actual - expected).abs() < 0.02,
                "attention element {index} for kv_heads={kv_heads}, causal={causal} is {actual}, expected {expected}"
            );
        }
    }
}

#[test]
fn stable_f16_gqa_is_a_graph_safe_dense_attention_candidate() {
    use apxinf_core::DType;

    let ctx = CudaContext::new(0).unwrap();
    let (batch, query_tokens, key_tokens, query_heads, kv_heads, head_dim) = (1, 2, 3, 4, 2, 4);
    let query_values = (0..batch * query_tokens * query_heads * head_dim)
        .map(|index| (index as f32 - 8.0) / 16.0)
        .collect::<Vec<_>>();
    let key_values = (0..batch * key_tokens * kv_heads * head_dim)
        .map(|index| (index as f32 - 5.0) / 12.0)
        .collect::<Vec<_>>();
    let value_values = (0..batch * key_tokens * kv_heads * head_dim)
        .map(|index| (index as f32 - 7.0) / 10.0)
        .collect::<Vec<_>>();
    let query = f16_tensor(
        0,
        vec![batch, query_tokens, query_heads, head_dim],
        &query_values,
    );
    let key = f16_tensor(0, vec![batch, key_tokens, kv_heads, head_dim], &key_values);
    let value = f16_tensor(
        0,
        vec![batch, key_tokens, kv_heads, head_dim],
        &value_values,
    );
    let mut output = zeros_tensor(
        0,
        vec![batch, query_tokens, query_heads, head_dim],
        DType::F16,
    );

    let normalized = {
        let mut args = AttentionArgs::new(&query, &key, &value, &mut output).causal();
        args.numerics = AttentionNumerics::Stable;
        args.policy.online_tune = false;
        args.policy.cache_dir = Some(String::new());
        super::attention_contracts::normalize(&ctx, args).unwrap()
    };
    let execution = super::attention_execution::prepare(&ctx, normalized).unwrap();
    assert!(
        execution.summary().starts_with("cublas-f16-gqa-stable "),
        "unexpected stable F16 provider: {}",
        execution.summary()
    );

    let session = ExecutionSession::with_capacity(4096, 0).unwrap();
    prepare_with_session(&session, || {
        let mut args = AttentionArgs::new(&query, &key, &value, &mut output).causal();
        args.numerics = AttentionNumerics::Stable;
        args.policy.online_tune = false;
        args.policy.cache_dir = Some(String::new());
        attention(&ctx, args)
    })
    .unwrap();
    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut args = AttentionArgs::new(&query, &key, &value, &mut output).causal();
            args.numerics = AttentionNumerics::Stable;
            args.policy.online_tune = false;
            args.policy.cache_dir = Some(String::new());
            attention(&ctx, args)
        })
    })
    .unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    let scale = 1.0 / (head_dim as f32).sqrt();
    let expected = attention_reference(
        &query_values,
        &key_values,
        &value_values,
        batch,
        query_tokens,
        key_tokens,
        query_heads,
        kv_heads,
        head_dim,
        scale,
        true,
    );
    for (actual, expected) in f16_values(&output).iter().zip(expected) {
        assert!((actual - expected).abs() < 0.01, "{actual} != {expected}");
    }
}

#[test]
fn attention_prepares_before_capture_and_replays_from_cache() {
    let ctx = CudaContext::new(0).unwrap();
    let query = tensor(0, vec![1, 2, 2, 2], &[0.25; 8]);
    let key = tensor(0, vec![1, 3, 1, 2], &[0.5; 6]);
    let value = tensor(0, vec![1, 3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut out = tensor(0, vec![1, 2, 2, 2], &[0.0; 8]);
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    prepare_with_session(&session, || {
        let mut prepare = AttentionArgs::new(&query, &key, &value, &mut out);
        prepare.policy.online_tune = false;
        attention(&ctx, prepare)
    })
    .unwrap();
    let expected = values(&out);

    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut captured = AttentionArgs::new(&query, &key, &value, &mut out);
            captured.policy.online_tune = false;
            attention(&ctx, captured)
        })
    })
    .unwrap();
    let output = crate::CudaBuffer::from_tensor(&out).unwrap();
    let sentinel: Vec<_> = (0..8)
        .flat_map(|_| bf16::from_f32(-123.0).to_bits().to_ne_bytes())
        .collect();
    output.copy_from_host(&sentinel).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), expected);
}

#[test]
fn kv_cache_attention_uses_trailing_causal_positions() {
    let ctx = CudaContext::new(0).unwrap();
    let query_values = vec![0.5, -0.25, 0.25, 0.75];
    let key_values = vec![0.25, 0.5, -0.5, 0.25, 0.75, -0.25];
    let value_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let query = tensor(0, vec![1, 2, 1, 2], &query_values);
    let key = tensor(0, vec![1, 3, 1, 2], &key_values);
    let value = tensor(0, vec![1, 3, 1, 2], &value_values);
    let mut out = tensor(0, vec![1, 2, 1, 2], &[0.0; 4]);
    let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out);
    args.policy.online_tune = false;
    kv_cache_attention(&ctx, args).unwrap();

    let expected = attention_reference(
        &query_values,
        &key_values,
        &value_values,
        1,
        2,
        3,
        1,
        1,
        2,
        1.0 / 2.0_f32.sqrt(),
        true,
    );
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.02);
    }
}

#[test]
fn kv_cache_attention_honors_capacity_valid_length_and_query_position() {
    let ctx = CudaContext::new(0).unwrap();
    let query_values = vec![0.5, -0.25, 0.25, 0.75, -0.5, 0.5, 0.75, 0.25];
    let key_values = vec![
        0.25, 0.5, -0.5, 0.25, 0.75, -0.25, 0.5, 0.5, 99.0, 99.0, -0.25, 0.75, 0.5, -0.5, 0.25,
        0.25, -0.75, 0.5, 88.0, 88.0,
    ];
    let value_values = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 999.0, 999.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7,
        1.8, 888.0, 888.0,
    ];
    let query = tensor(0, vec![2, 2, 1, 2], &query_values);
    let key = tensor(0, vec![2, 5, 1, 2], &key_values);
    let value = tensor(0, vec![2, 5, 1, 2], &value_values);
    let mut out = tensor(0, vec![2, 2, 1, 2], &[0.0; 8]);
    let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out);
    args.valid_key_tokens = 4;
    args.query_start = 1;
    args.policy.online_tune = false;
    kv_cache_attention(&ctx, args).unwrap();

    let expected = kv_cache_attention_reference(
        &query_values,
        &key_values,
        &value_values,
        2,
        2,
        4,
        5,
        1,
        1,
        2,
        1,
        1.0 / 2.0_f32.sqrt(),
        true,
    );
    for (index, (actual, expected)) in values(&out).iter().zip(expected).enumerate() {
        assert!(
            (*actual - expected).abs() < 0.02,
            "KV-cache capacity result {index} is {actual}, expected {expected}"
        );
    }
}

#[test]
fn kv_cache_graph_replays_all_positions_with_one_capacity_spec() {
    let ctx = CudaContext::new(0).unwrap();
    let head_dim = 64;
    let key_capacity = 64;
    let query_values: Vec<_> = (0..head_dim)
        .map(|index| ((index * 3 % 17) as f32 - 8.0) / 16.0)
        .collect();
    let key_values: Vec<_> = (0..key_capacity * head_dim)
        .map(|index| ((index * 5 % 19) as f32 - 9.0) / 16.0)
        .collect();
    let value_values: Vec<_> = (0..key_capacity * head_dim)
        .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
        .collect();
    let query = tensor(0, vec![1, 1, 1, head_dim], &query_values);
    let key = tensor(0, vec![1, key_capacity, 1, head_dim], &key_values);
    let value = tensor(0, vec![1, key_capacity, 1, head_dim], &value_values);
    let mut out = tensor(0, vec![1, 1, 1, head_dim], &vec![0.0; head_dim]);
    let decode_meta = KvCacheDecodeMeta::new(0, key_capacity).unwrap();
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    prepare_with_session(&session, || {
        let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out)
            .with_decode_meta(&decode_meta);
        args.valid_key_tokens = 1;
        args.query_start = 0;
        args.policy.online_tune = false;
        kv_cache_attention(&ctx, args)
    })
    .unwrap();

    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out)
                .with_decode_meta(&decode_meta);
            args.valid_key_tokens = 1;
            args.query_start = 0;
            args.policy.online_tune = false;
            kv_cache_attention(&ctx, args)
        })
    })
    .unwrap();

    // This crosses every old power-of-two bucket boundary, including 7/8/9,
    // and reaches valid_len == key_capacity without preparing or capturing
    // another execution.
    for valid_key_tokens in 1..=key_capacity {
        // Updating after synchronization is required because replay is async.
        decode_meta
            .update(valid_key_tokens, valid_key_tokens - 1)
            .unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        let expected = kv_cache_attention_reference(
            &query_values,
            &key_values,
            &value_values,
            1,
            1,
            valid_key_tokens,
            key_capacity,
            1,
            1,
            head_dim,
            valid_key_tokens - 1,
            1.0 / (head_dim as f32).sqrt(),
            true,
        );
        let actual = values(&out);
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (*actual - expected).abs() < 0.03 + 0.02 * expected.abs(),
                "step-{valid_key_tokens} graph output {index} is {actual}, expected {expected}"
            );
        }
    }
}

#[test]
fn kv_cache_decode_meta_enforces_capacity() {
    let decode_meta = KvCacheDecodeMeta::new(0, 9).unwrap();
    decode_meta.update(9, 8).unwrap();
    assert!(decode_meta.update(10, 9).is_err());
}

#[test]
fn kv_cache_decode_meta_rejects_non_decode_queries() {
    let ctx = CudaContext::new(0).unwrap();
    let key = tensor(0, vec![1, 4, 1, 2], &[0.25; 8]);
    let value = tensor(0, vec![1, 4, 1, 2], &[0.5; 8]);
    let decode_meta = KvCacheDecodeMeta::new(0, 4).unwrap();

    let query = tensor(0, vec![1, 2, 1, 2], &[0.75; 4]);
    let mut out = tensor(0, vec![1, 2, 1, 2], &[0.0; 4]);
    let args =
        KvCacheAttentionArgs::new(&query, &key, &value, &mut out).with_decode_meta(&decode_meta);
    let error = kv_cache_attention(&ctx, args).unwrap_err();
    assert!(error.to_string().contains("causal single-token"), "{error}");

    let query = tensor(0, vec![1, 1, 1, 2], &[0.75; 2]);
    let mut out = tensor(0, vec![1, 1, 1, 2], &[0.0; 2]);
    let args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out)
        .with_decode_meta(&decode_meta)
        .non_causal();
    let error = kv_cache_attention(&ctx, args).unwrap_err();
    assert!(error.to_string().contains("causal single-token"), "{error}");

    let query = tensor(0, vec![2, 1, 1, 2], &[0.75; 4]);
    let key = tensor(0, vec![2, 4, 1, 2], &[0.25; 16]);
    let value = tensor(0, vec![2, 4, 1, 2], &[0.5; 16]);
    let mut out = tensor(0, vec![2, 1, 1, 2], &[0.0; 4]);
    let args =
        KvCacheAttentionArgs::new(&query, &key, &value, &mut out).with_decode_meta(&decode_meta);
    let error = kv_cache_attention(&ctx, args).unwrap_err();
    assert!(error.to_string().contains("batch-1"), "{error}");
}

#[test]
fn allocation_free_fa2_decode_matches_attention_contract() {
    let ctx = CudaContext::new(0).unwrap();
    let query_values: Vec<_> = (0..24 * 256)
        .map(|index| ((index * 7 % 31) as f32 - 15.0) / 32.0)
        .collect();
    for key_tokens in [128, 2048, 2176] {
        let cache_values = 4 * key_tokens * 256;
        let key_values: Vec<_> = (0..cache_values)
            .map(|index| ((index * 5 % 29) as f32 - 14.0) / 32.0)
            .collect();
        let value_values: Vec<_> = (0..cache_values)
            .map(|index| ((index * 11 % 37) as f32 - 18.0) / 32.0)
            .collect();
        let query = tensor(0, vec![1, 1, 24, 256], &query_values);
        let key = tensor(0, vec![1, key_tokens, 4, 256], &key_values);
        let value = tensor(0, vec![1, key_tokens, 4, 256], &value_values);
        let mut reference = tensor(0, vec![1, 1, 24, 256], &vec![0.0; 24 * 256]);
        let direct = tensor(0, vec![1, 1, 24, 256], &vec![0.0; 24 * 256]);
        let workspace = zeros_tensor(
            0,
            vec![decode_attention_workspace_bytes().div_ceil(4)],
            apxinf_core::DType::F32,
        );

        let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut reference);
        args.valid_key_tokens = key_tokens;
        args.query_start = key_tokens - 1;
        args.policy.online_tune = true;
        args.policy.allow_fallback = false;
        kv_cache_attention(&ctx, args).unwrap();
        decode_attention(
            &ctx,
            &query,
            &key,
            &value,
            &direct,
            &workspace,
            key_tokens,
            1.0 / 256.0f32.sqrt(),
        )
        .unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(
            values(&direct),
            values(&reference),
            "key_tokens={key_tokens}"
        );
    }
}

#[test]
fn kv_cache_attention_non_causal_uses_only_valid_cache_rows() {
    let ctx = CudaContext::new(0).unwrap();
    let query_values = vec![0.5, -0.25];
    let key_values = vec![0.25, 0.5, -0.5, 0.25, 0.75, -0.25, 99.0, 99.0];
    let value_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 999.0, 999.0];
    let query = tensor(0, vec![1, 1, 1, 2], &query_values);
    let key = tensor(0, vec![1, 4, 1, 2], &key_values);
    let value = tensor(0, vec![1, 4, 1, 2], &value_values);
    let mut out = tensor(0, vec![1, 1, 1, 2], &[0.0; 2]);
    let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out).non_causal();
    args.valid_key_tokens = 3;
    args.policy.online_tune = false;
    kv_cache_attention(&ctx, args).unwrap();

    let expected = kv_cache_attention_reference(
        &query_values,
        &key_values,
        &value_values,
        1,
        1,
        3,
        4,
        1,
        1,
        2,
        0,
        1.0 / 2.0_f32.sqrt(),
        false,
    );
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.02);
    }
}

#[test]
fn segmented_attention_keeps_packed_segments_isolated() {
    let ctx = CudaContext::new(0).unwrap();
    let query_values = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let key_values = vec![1.0, 0.0, 0.0, 1.0, 1.0, -1.0];
    let value_values = vec![2.0, 4.0, 6.0, 8.0, 100.0, 200.0];
    let query = tensor(0, vec![3, 1, 2], &query_values);
    let key = tensor(0, vec![3, 1, 2], &key_values);
    let value = tensor(0, vec![3, 1, 2], &value_values);
    let mut out = tensor(0, vec![3, 1, 2], &[0.0; 6]);
    let host_offsets = [0, 2, 3];
    let offsets = u32_buffer(0, &host_offsets);
    let mut args =
        SegmentedAttentionArgs::new(&query, &key, &value, &mut out, &offsets, &host_offsets);
    args.policy.online_tune = false;
    segmented_attention(&ctx, args).unwrap();

    let mut expected = attention_reference(
        &query_values[..4],
        &key_values[..4],
        &value_values[..4],
        1,
        2,
        2,
        1,
        1,
        2,
        1.0 / 2.0_f32.sqrt(),
        false,
    );
    expected.extend_from_slice(&value_values[4..]);
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.02);
    }
}

#[test]
fn segmented_attention_prepares_before_capture_and_replays_from_cache() {
    let ctx = CudaContext::new(0).unwrap();
    let query = tensor(0, vec![3, 1, 2], &[0.25; 6]);
    let key = tensor(0, vec![3, 1, 2], &[0.5; 6]);
    let value = tensor(0, vec![3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut out = tensor(0, vec![3, 1, 2], &[0.0; 6]);
    let host_offsets = [0, 2, 3];
    let offsets = u32_buffer(0, &host_offsets);
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    prepare_with_session(&session, || {
        let mut args =
            SegmentedAttentionArgs::new(&query, &key, &value, &mut out, &offsets, &host_offsets);
        args.policy.online_tune = false;
        segmented_attention(&ctx, args)
    })
    .unwrap();
    with_session(&session, || {
        let mut args =
            SegmentedAttentionArgs::new(&query, &key, &value, &mut out, &offsets, &host_offsets);
        args.policy.online_tune = false;
        segmented_attention(&ctx, args)
    })
    .unwrap();
    ctx.synchronize().unwrap();
    let expected = values(&out);

    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut args = SegmentedAttentionArgs::new(
                &query,
                &key,
                &value,
                &mut out,
                &offsets,
                &host_offsets,
            );
            args.policy.online_tune = false;
            segmented_attention(&ctx, args)
        })
    })
    .unwrap();
    let output = crate::CudaBuffer::from_tensor(&out).unwrap();
    let sentinel: Vec<_> = (0..6)
        .flat_map(|_| bf16::from_f32(-123.0).to_bits().to_ne_bytes())
        .collect();
    output.copy_from_host(&sentinel).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), expected);
}

#[test]
fn attention_semantics_use_distinct_native_registries() {
    let ctx = CudaContext::new(0).unwrap();
    let query = tensor(0, vec![1, 1, 1, 2], &[0.5, -0.25]);
    let key = tensor(0, vec![1, 2, 1, 2], &[0.25, 0.5, -0.5, 0.25]);
    let value = tensor(0, vec![1, 2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);

    let mut dense_out = tensor(0, vec![1, 1, 1, 2], &[0.0; 2]);
    let mut dense_args = AttentionArgs::new(&query, &key, &value, &mut dense_out);
    dense_args.policy.online_tune = false;
    let dense = super::attention_execution::prepare(
        &ctx,
        super::attention_contracts::normalize(&ctx, dense_args).unwrap(),
    )
    .unwrap();

    let mut cache_out = tensor(0, vec![1, 1, 1, 2], &[0.0; 2]);
    let mut cache_args = KvCacheAttentionArgs::new(&query, &key, &value, &mut cache_out);
    cache_args.valid_key_tokens = 1;
    cache_args.query_start = 0;
    cache_args.policy.online_tune = false;
    let cache = super::attention_execution::prepare(
        &ctx,
        super::normalize_kv_cache_attention(&ctx, cache_args).unwrap(),
    )
    .unwrap();

    let segmented_query = tensor(0, vec![2, 1, 2], &[0.5, -0.25, 0.25, 0.75]);
    let segmented_key = tensor(0, vec![2, 1, 2], &[0.25, 0.5, -0.5, 0.25]);
    let segmented_value = tensor(0, vec![2, 1, 2], &[1.0, 2.0, 3.0, 4.0]);
    let mut segmented_out = tensor(0, vec![2, 1, 2], &[0.0; 4]);
    let host_offsets = [0, 2];
    let offsets = u32_buffer(0, &host_offsets);
    let mut segmented_args = SegmentedAttentionArgs::new(
        &segmented_query,
        &segmented_key,
        &segmented_value,
        &mut segmented_out,
        &offsets,
        &host_offsets,
    );
    segmented_args.policy.online_tune = false;
    let segmented = super::attention_execution::prepare(
        &ctx,
        super::normalize_segmented_attention(&ctx, segmented_args).unwrap(),
    )
    .unwrap();

    let dense_summary = dense.summary();
    let cache_summary = cache.summary();
    let segmented_summary = segmented.summary();
    assert!(dense_summary.starts_with("custom-attention-fallback "));
    assert!(cache_summary.starts_with("custom-kv-cache-attention "));
    assert!(
        segmented_summary.starts_with("custom-segmented-attention ")
            || segmented_summary.starts_with("flash-attention-2-segmented ")
            || segmented_summary.starts_with("specialized-segmented-vision-sm110 "),
        "unexpected segmented Attention candidate: {segmented_summary}"
    );
    assert_ne!(dense_summary, cache_summary);
    assert_ne!(dense_summary, segmented_summary);
    assert_ne!(cache_summary, segmented_summary);
}

//! Full-attention primitives against host references.
//!
//! Scope, same caveat as the GDN tests: these confirm the kernels compute the
//! stated transforms with the model's real geometry. The rotary pairing
//! convention is also checked against Qwen3.5's interleaved T/H/W frequency
//! selection.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};

const HEAD_DIM: usize = 256;
const HEADS: usize = 24;
const KV_HEADS: usize = 4;

fn upload_bf16(ctx: &CudaContext, values: &[f32], dims: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), DType::BF16).unwrap()
}

fn upload_i32(ctx: &CudaContext, values: &[i32], dims: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), DType::I32).unwrap()
}

fn read_bf16(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|v| half::bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32())
        .collect()
}

#[test]
fn rotary_dim_follows_the_partial_factor() {
    // head_dim 256 with partial_rotary_factor 0.25 rotates 64 elements and
    // leaves 192 alone. Getting this wrong rotates the whole head, which still
    // runs.
    assert_eq!(ops::rotary_dim(HEAD_DIM, 0.25), 64);
    assert_eq!(ops::rotary_dim(128, 0.25), 32);
    // Odd results round down: the rotation pairs elements.
    assert_eq!(ops::rotary_dim(100, 0.25), 24);
}

#[test]
fn partial_rope_rotates_only_the_rotary_prefix() {
    let ctx = CudaContext::new(0).unwrap();
    let tokens = 5usize;
    let width = ops::rotary_dim(HEAD_DIM, 0.25);
    let theta = 1.0e7f32;

    let values: Vec<f32> = (0..tokens * KV_HEADS * HEAD_DIM)
        .map(|index| ((index % 23) as f32 - 11.0) / 7.0)
        .collect();
    let data = upload_bf16(&ctx, &values, vec![tokens, KV_HEADS, HEAD_DIM]);
    let positions: Vec<i32> = (0..tokens as i32).map(|p| p * 3 + 1).collect();
    let device_positions = upload_i32(&ctx, &positions, vec![tokens]);

    ops::partial_rope(&ctx, &data, &device_positions, width, theta).unwrap();
    ctx.synchronize().unwrap();
    let produced = read_bf16(&data);

    let half = width / 2;
    let mut worst_rotated = 0.0f64;
    for token in 0..tokens {
        for head in 0..KV_HEADS {
            let base = (token * KV_HEADS + head) * HEAD_DIM;
            for index in 0..half {
                let frequency = (theta as f64).powf(-2.0 * index as f64 / width as f64);
                let angle = positions[token] as f64 * frequency;
                let low = half::bf16::from_f32(values[base + index]).to_f32() as f64;
                let high = half::bf16::from_f32(values[base + half + index]).to_f32() as f64;
                let expect_low = low * angle.cos() - high * angle.sin();
                let expect_high = high * angle.cos() + low * angle.sin();
                for (got, want) in [
                    (produced[base + index] as f64, expect_low),
                    (produced[base + half + index] as f64, expect_high),
                ] {
                    worst_rotated = worst_rotated.max((got - want).abs() / want.abs().max(1e-2));
                }
            }
            // Everything past the rotary prefix must be untouched, which is
            // the part a full-width RoPE would silently break.
            for index in width..HEAD_DIM {
                let untouched = half::bf16::from_f32(values[base + index]).to_f32();
                assert_eq!(
                    produced[base + index],
                    untouched,
                    "element {index} past the rotary width was modified"
                );
            }
        }
    }
    println!("rotated prefix worst relative: {worst_rotated:.5}");
    assert!(worst_rotated < 2e-2, "rotation drifted by {worst_rotated}");
}

#[test]
fn partial_mrope_selects_qwen35_interleaved_axes() {
    let ctx = CudaContext::new(0).unwrap();
    let tokens = 2usize;
    let width = 64usize;
    let theta = 1.0e7f32;
    let sections = [11usize, 11, 10];
    let values: Vec<f32> = (0..tokens * HEAD_DIM)
        .map(|index| ((index % 29) as f32 - 14.0) / 9.0)
        .collect();
    let mut data = upload_bf16(&ctx, &values, vec![tokens, 1, HEAD_DIM]);
    let positions = [2, 17, 41, 3, 19, 43];
    let device_positions = upload_i32(&ctx, &positions, vec![tokens, 3]);

    ops::multimodal_rope(
        &ctx,
        ops::MultimodalRopeArgs {
            data: &mut data,
            positions: &device_positions,
            rotary_dim: width,
            sections,
            theta,
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();
    let produced = read_bf16(&data);

    let half = width / 2;
    for token in 0..tokens {
        for pair in 0..half {
            // Transformers Qwen3.5 starts with T and overwrites every third
            // frequency with H/W within each configured section.
            let axis = if pair % 3 == 1 && pair < sections[1] * 3 {
                1
            } else if pair % 3 == 2 && pair < sections[2] * 3 {
                2
            } else {
                0
            };
            let angle = positions[token * 3 + axis] as f64
                * (theta as f64).powf(-2.0 * pair as f64 / width as f64);
            let base = token * HEAD_DIM;
            let low = half::bf16::from_f32(values[base + pair]).to_f32() as f64;
            let high = half::bf16::from_f32(values[base + half + pair]).to_f32() as f64;
            let expected = [
                low * angle.cos() - high * angle.sin(),
                high * angle.cos() + low * angle.sin(),
            ];
            for (got, want) in [
                (produced[base + pair] as f64, expected[0]),
                (produced[base + half + pair] as f64, expected[1]),
            ] {
                assert!(
                    (got - want).abs() / want.abs().max(1e-2) < 2e-2,
                    "token {token} pair {pair}: got {got}, expected {want}"
                );
            }
        }
    }
}

#[test]
fn partial_mrope_is_bitwise_text_compatible() {
    let ctx = CudaContext::new(0).unwrap();
    let tokens = 4usize;
    let width = 64usize;
    let theta = 1.0e7f32;
    let values: Vec<f32> = (0..tokens * KV_HEADS * HEAD_DIM)
        .map(|index| ((index % 31) as f32 - 15.0) / 8.0)
        .collect();
    let legacy = upload_bf16(&ctx, &values, vec![tokens, KV_HEADS, HEAD_DIM]);
    let mut multimodal = upload_bf16(&ctx, &values, vec![tokens, KV_HEADS, HEAD_DIM]);
    let positions: Vec<i32> = (0..tokens as i32).map(|position| position * 7).collect();
    let triples: Vec<i32> = positions
        .iter()
        .flat_map(|&position| [position, position, position])
        .collect();
    let legacy_positions = upload_i32(&ctx, &positions, vec![tokens]);
    let multimodal_positions = upload_i32(&ctx, &triples, vec![tokens, 3]);

    ops::partial_rope(&ctx, &legacy, &legacy_positions, width, theta).unwrap();
    ops::multimodal_rope(
        &ctx,
        ops::MultimodalRopeArgs {
            data: &mut multimodal,
            positions: &multimodal_positions,
            rotary_dim: width,
            sections: [11, 11, 10],
            theta,
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(read_bf16(&legacy), read_bf16(&multimodal));
}

#[test]
fn vision_qkv_rope_uses_height_then_width_axes() {
    let ctx = CudaContext::new(0).unwrap();
    let (tokens, heads, head_dim) = (2usize, 2usize, 8usize);
    let projection = heads * head_dim;
    let theta = 10_000.0f32;
    let qkv_values: Vec<f32> = (0..tokens * 3 * projection)
        .map(|index| ((index % 37) as f32 - 18.0) / 11.0)
        .collect();
    let bias_values: Vec<f32> = (0..3 * projection)
        .map(|index| ((index % 9) as f32 - 4.0) / 17.0)
        .collect();
    let positions = [3, 19, 11, 5];
    let qkv = upload_bf16(&ctx, &qkv_values, vec![tokens, 3 * projection]);
    let bias = upload_bf16(&ctx, &bias_values, vec![3 * projection]);
    let positions_dev = upload_i32(&ctx, &positions, vec![tokens, 2]);
    let mut query = upload_bf16(
        &ctx,
        &vec![0.0; tokens * projection],
        vec![tokens, heads, head_dim],
    );
    let mut key = upload_bf16(
        &ctx,
        &vec![0.0; tokens * projection],
        vec![tokens, heads, head_dim],
    );
    let mut value = upload_bf16(
        &ctx,
        &vec![0.0; tokens * projection],
        vec![tokens, heads, head_dim],
    );

    ops::split_qkv_axial_rope(
        &ctx,
        ops::AxialRopeArgs {
            qkv: &qkv,
            bias: Some(&bias),
            positions: &positions_dev,
            query: &mut query,
            key: &mut key,
            value: &mut value,
            theta,
            numerics: ops::AxialRopeNumerics::Flexible,
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();
    let produced = [read_bf16(&query), read_bf16(&key), read_bf16(&value)];

    let half = head_dim / 2;
    let quarter = half / 2;
    for token in 0..tokens {
        for projection_index in 0..3 {
            for head in 0..heads {
                let source_base = token * 3 * projection + projection_index * projection;
                let bias_base = projection_index * projection;
                let destination_base = (token * heads + head) * head_dim;
                if projection_index == 2 {
                    for element in 0..head_dim {
                        let source = half::bf16::from_f32(
                            half::bf16::from_f32(
                                qkv_values[source_base + head * head_dim + element],
                            )
                            .to_f32()
                                + half::bf16::from_f32(
                                    bias_values[bias_base + head * head_dim + element],
                                )
                                .to_f32(),
                        )
                        .to_f32();
                        assert_eq!(produced[2][destination_base + element], source);
                    }
                    continue;
                }
                for pair in 0..half {
                    let axis = usize::from(pair >= quarter);
                    let axis_pair = pair % quarter;
                    let angle = positions[token * 2 + axis] as f64
                        * (theta as f64).powf(-2.0 * axis_pair as f64 / half as f64);
                    let round_with_bias = |element: usize| {
                        half::bf16::from_f32(
                            half::bf16::from_f32(
                                qkv_values[source_base + head * head_dim + element],
                            )
                            .to_f32()
                                + half::bf16::from_f32(
                                    bias_values[bias_base + head * head_dim + element],
                                )
                                .to_f32(),
                        )
                        .to_f32() as f64
                    };
                    let first = round_with_bias(pair);
                    let second = round_with_bias(half + pair);
                    let expected = [
                        first * angle.cos() - second * angle.sin(),
                        first * angle.sin() + second * angle.cos(),
                    ];
                    for (element, expected) in [(pair, expected[0]), (half + pair, expected[1])] {
                        let got = produced[projection_index][destination_base + element] as f64;
                        assert!(
                            (got - expected).abs() / expected.abs().max(1e-2) < 2e-2,
                            "projection {projection_index}, token {token}, head {head}, element {element}: got {got}, expected {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn query_and_gate_split_follows_the_projection_layout() {
    let ctx = CudaContext::new(0).unwrap();
    let tokens = 3usize;
    // q_proj is [12288, 5120] = 2 * (24 heads * 256), query and gate
    // interleaved per head rather than as two contiguous halves.
    let fused_values: Vec<f32> = (0..tokens * HEADS * 2 * HEAD_DIM)
        .map(|index| index as f32 / 1000.0)
        .collect();
    let fused = upload_bf16(&ctx, &fused_values, vec![tokens, HEADS, 2 * HEAD_DIM]);
    let query = upload_bf16(
        &ctx,
        &vec![0.0; tokens * HEADS * HEAD_DIM],
        vec![tokens, HEADS, HEAD_DIM],
    );
    let gate = upload_bf16(
        &ctx,
        &vec![0.0; tokens * HEADS * HEAD_DIM],
        vec![tokens, HEADS, HEAD_DIM],
    );

    ops::split_query_and_gate(&ctx, &fused, &query, &gate).unwrap();
    ctx.synchronize().unwrap();

    let produced_query = read_bf16(&query);
    let produced_gate = read_bf16(&gate);
    for token in 0..tokens {
        for head in 0..HEADS {
            for element in 0..HEAD_DIM {
                let slot = (token * HEADS + head) * 2 * HEAD_DIM + element;
                let flat = (token * HEADS + head) * HEAD_DIM + element;
                assert_eq!(
                    produced_query[flat],
                    half::bf16::from_f32(fused_values[slot]).to_f32()
                );
                assert_eq!(
                    produced_gate[flat],
                    half::bf16::from_f32(fused_values[slot + HEAD_DIM]).to_f32()
                );
            }
        }
    }
}

/// The full-attention output gate is `x * sigmoid(z)`.
///
/// `config.json` advertises `output_gate_type: "swish"`, which would make it
/// `x * silu(z) = x * z * sigmoid(z)`. Nothing reads that key: the reference
/// `Qwen3_5Attention.forward` does `attn_output * torch.sigmoid(gate)`
/// (modeling_qwen3_5.py:818), and the code is the artifact that defines the
/// model. This test pins the sigmoid form and, to keep a future swap from
/// passing quietly, checks that the silu alternative is far enough away that
/// the first assertion could not have accepted it.
///
/// The GDN output gate really is silu -- see `gdn_gated_norm`, which this does
/// not cover.
#[test]
fn attention_gate_is_sigmoid_not_silu() {
    let ctx = CudaContext::new(0).unwrap();
    let count = 4096usize;
    let data_values: Vec<f32> = (0..count).map(|i| ((i % 17) as f32 - 8.0) / 5.0).collect();
    let gate_values: Vec<f32> = (0..count).map(|i| ((i % 13) as f32 - 6.0) / 3.0).collect();
    let data = upload_bf16(&ctx, &data_values, vec![count]);
    let gate = upload_bf16(&ctx, &gate_values, vec![count]);

    ops::apply_output_gate(&ctx, ops::OutputGateArgs::new(&data, &gate)).unwrap();
    ctx.synchronize().unwrap();
    let produced = read_bf16(&data);

    let mut worst = 0.0f64;
    let mut worst_against_silu = 0.0f64;
    for index in 0..count {
        let x = half::bf16::from_f32(data_values[index]).to_f32() as f64;
        let z = half::bf16::from_f32(gate_values[index]).to_f32() as f64;
        let sigmoid = 1.0 / (1.0 + (-z).exp());
        let expected = x * sigmoid;
        let got = produced[index] as f64;
        worst = worst.max((got - expected).abs() / expected.abs().max(1e-2));
        // How far the silu-gated alternative sits from what we got. If the two
        // were ever swapped, `worst` above would catch it only because this
        // margin is large -- so assert the margin too, and the pair of checks
        // stays a real discriminator rather than a tautology.
        let silu_variant = x * z * sigmoid;
        worst_against_silu =
            worst_against_silu.max((got - silu_variant).abs() / got.abs().max(1e-2));
    }
    println!("attention gate worst relative: {worst:.5}");
    println!("  margin against the silu form: {worst_against_silu:.5}");
    assert!(worst < 2e-2, "sigmoid gate drifted by {worst}");
    assert!(
        worst_against_silu > 0.5,
        "sanity: silu and sigmoid gates must differ enough for the check above \
         to discriminate, but the worst gap was only {worst_against_silu}"
    );
}

#[test]
fn rounded_output_gate_accepts_dense_and_interleaved_sources() {
    let ctx = CudaContext::new(0).unwrap();
    let (rows, heads, head_dim) = (3usize, 2usize, 8usize);
    let width = heads * head_dim;
    let projection_width = 2 * width + 7;
    let data_values: Vec<f32> = (0..rows * width)
        .map(|index| ((index % 19) as f32 - 9.0) / 7.0)
        .collect();
    let gate_values: Vec<f32> = (0..rows * width)
        .map(|index| ((index % 11) as f32 - 5.0) / 4.0)
        .collect();
    let mut projection_values = vec![0.0f32; rows * projection_width];
    for row in 0..rows {
        for head in 0..heads {
            for element in 0..head_dim {
                let gate_index = (row * heads + head) * head_dim + element;
                let projection_index =
                    row * projection_width + head * 2 * head_dim + head_dim + element;
                projection_values[projection_index] = gate_values[gate_index];
            }
        }
    }

    let dense_data = upload_bf16(&ctx, &data_values, vec![rows, width]);
    let interleaved_data = upload_bf16(&ctx, &data_values, vec![rows, width]);
    let dense_gate = upload_bf16(&ctx, &gate_values, vec![rows, width]);
    let projection = upload_bf16(&ctx, &projection_values, vec![rows, projection_width]);

    ops::apply_output_gate(
        &ctx,
        ops::OutputGateArgs::new(&dense_data, &dense_gate)
            .with_numerics(ops::OutputGateNumerics::RoundedSigmoid),
    )
    .unwrap();
    ops::apply_output_gate(
        &ctx,
        ops::OutputGateArgs::from_projection(&interleaved_data, &projection, heads, head_dim),
    )
    .unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(read_bf16(&dense_data), read_bf16(&interleaved_data));
}

#[test]
fn attention_gate_family_is_graph_safe() {
    let ctx = CudaContext::new(0).unwrap();
    let (tokens, heads, head_dim) = (2usize, 2usize, 4usize);
    let elements = tokens * heads * head_dim;
    let fused_values = (0..2 * elements)
        .map(|index| ((index % 17) as f32 - 8.0) / 7.0)
        .collect::<Vec<_>>();
    let data_values = (0..elements)
        .map(|index| ((index % 13) as f32 - 6.0) / 5.0)
        .collect::<Vec<_>>();
    let gate_values = (0..elements)
        .map(|index| ((index % 11) as f32 - 5.0) / 4.0)
        .collect::<Vec<_>>();
    let fused = upload_bf16(&ctx, &fused_values, vec![tokens, heads, 2 * head_dim]);
    let dense_gate = upload_bf16(&ctx, &gate_values, vec![tokens, heads, head_dim]);
    let eager_query = upload_bf16(&ctx, &vec![0.0; elements], vec![tokens, heads, head_dim]);
    let eager_split_gate = upload_bf16(&ctx, &vec![0.0; elements], vec![tokens, heads, head_dim]);
    let eager_data = upload_bf16(&ctx, &data_values, vec![tokens, heads, head_dim]);
    ops::split_query_and_gate(&ctx, &fused, &eager_query, &eager_split_gate).unwrap();
    ops::apply_output_gate(
        &ctx,
        ops::OutputGateArgs::new(&eager_data, &dense_gate)
            .with_numerics(ops::OutputGateNumerics::RoundedSigmoid),
    )
    .unwrap();

    let graph_query = upload_bf16(&ctx, &vec![0.0; elements], vec![tokens, heads, head_dim]);
    let graph_split_gate = upload_bf16(&ctx, &vec![0.0; elements], vec![tokens, heads, head_dim]);
    let graph_data = upload_bf16(&ctx, &data_values, vec![tokens, heads, head_dim]);
    let graph = apxinf_cuda_new::capture(&ctx, || {
        ops::split_query_and_gate(&ctx, &fused, &graph_query, &graph_split_gate)?;
        ops::apply_output_gate(
            &ctx,
            ops::OutputGateArgs::new(&graph_data, &dense_gate)
                .with_numerics(ops::OutputGateNumerics::RoundedSigmoid),
        )
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    assert_eq!(read_bf16(&graph_query), read_bf16(&eager_query));
    assert_eq!(read_bf16(&graph_split_gate), read_bf16(&eager_split_gate));
    assert_eq!(read_bf16(&graph_data), read_bf16(&eager_data));
}

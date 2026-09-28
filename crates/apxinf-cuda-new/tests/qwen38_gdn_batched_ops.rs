//! The batched GDN operators against the single-token path they replace.
//!
//! Prefill runs one kernel over the whole prompt where decode runs one kernel
//! per token. These tests pin the two together on synthetic data, so a
//! divergence shows up here rather than as a drifting generation 64 layers
//! later.
//!
//! The conv test is the one that matters most: it also covers the window
//! hand-off, which is invisible to a shape check and silently corrupts the
//! first decode step after a prompt.
//!
//! ```text
//! bash crates/apxinf-cuda-new/test-new.sh \
//!   test -p apxinf-cuda --test qwen38_gdn_batched_ops -- --ignored --nocapture
//! ```

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};

fn upload_bf16(ctx: &CudaContext, values: &[f32], dims: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), DType::BF16).unwrap()
}

fn upload_f32(ctx: &CudaContext, values: &[f32], dims: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), DType::F32).unwrap()
}

fn zeros(ctx: &CudaContext, dims: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = dims.iter().product::<usize>() * dtype.size_in_bytes();
    let buffer = CudaBuffer::alloc(bytes, ctx.device_id()).unwrap();
    buffer.copy_from_host(&vec![0u8; bytes]).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn read_bf16(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|raw| f32::from_bits((u16::from_le_bytes([raw[0], raw[1]]) as u32) << 16))
        .collect()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|raw| f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
        .collect()
}

/// A tensor slice sharing another tensor's storage, offset by `elements`.
/// Matches how the driver carves per-token views out of a cache.
fn slice_of(tensor: &Tensor, elements: usize, dims: Vec<usize>, dtype: DType) -> Tensor {
    let element = dtype.size_in_bytes();
    let span = dims.iter().product::<usize>() * element;
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .view(elements * element, span)
        .unwrap()
        .as_tensor(Shape::new(dims), dtype)
        .unwrap()
}

#[test]
#[ignore = "requires a GPU"]
fn fused_conv_prepare_matches_separate_and_continues_decode() {
    let ctx = CudaContext::new(0).unwrap();
    let read = |tensor: &Tensor| {
        let buffer = CudaBuffer::from_tensor(tensor).unwrap();
        let mut bytes = vec![0u8; buffer.len()];
        buffer.copy_to_host(&mut bytes).unwrap();
        assert!(!bytes.is_empty());
        bytes
    };
    for (k_heads, v_heads) in [(1usize, 1usize), (16, 48), (16, 64)] {
        let channels = (2 * k_heads + v_heads) * 128;
        for tokens in [1usize, 3, 4, 63, 64, 65, 129, 2048] {
            let values: Vec<f32> = (0..tokens * channels)
                .map(|index| (index as f32 * 0.023).sin() * 3.5).collect();
            let weights: Vec<f32> = (0..channels * 4)
                .map(|index| (index as f32 * 0.043).cos() * 0.31).collect();
            let input = upload_bf16(&ctx, &values, vec![tokens, channels]);
            let weight = upload_bf16(&ctx, &weights, vec![channels, 4]);
            let convolved = zeros(&ctx, vec![tokens, channels], DType::BF16);
            let decay = upload_f32(&ctx, &vec![-0.125; tokens * v_heads], vec![tokens, v_heads]);
            let reference = [
                zeros(&ctx, vec![tokens, k_heads, 128], DType::F16),
                zeros(&ctx, vec![tokens, k_heads, 128], DType::F16),
                zeros(&ctx, vec![tokens, v_heads, 128], DType::F16),
                zeros(&ctx, vec![tokens, v_heads], DType::F32),
                zeros(&ctx, vec![channels, 4], DType::F32),
            ];
            let candidate: Vec<Tensor> = reference.iter().enumerate().map(|(index, tensor)| {
                let output = zeros(&ctx, tensor.shape().dims().to_vec(), tensor.dtype());
                let buffer = CudaBuffer::from_tensor(&output).unwrap();
                buffer.copy_from_host(&vec![0xaa + index as u8; buffer.len()]).unwrap();
                output
            }).collect();
            ops::gdn_causal_conv_forward(&ctx, &input, &weight, &convolved,
                Some(&reference[4]), tokens, channels, 4).unwrap();
            ops::gdn_prepare_flashinfer(&ctx, &convolved, &reference[0], &reference[1],
                &reference[2], &decay, &reference[3], tokens, channels, k_heads, v_heads, 128, 1e-6).unwrap();
            ops::gdn_conv_prepare_flashinfer(&ctx, &input, &weight, &candidate[4],
                &candidate[0], &candidate[1], &candidate[2], &decay, &candidate[3],
                tokens, k_heads, v_heads, 1e-6).unwrap();
            ctx.synchronize().unwrap();
            for (index, (expected, actual)) in reference.iter().zip(&candidate).enumerate() {
                assert_eq!(read(expected), read(actual), "tokens={tokens}, heads={k_heads}/{v_heads}, output={index}");
                if index < 3 {
                    assert!(read(actual).chunks_exact(2).all(|bytes|
                        half::f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).is_finite()));
                } else {
                    assert!(read_f32(actual).iter().all(|value| value.is_finite()));
                }
            }
            let next = upload_bf16(&ctx, &values[..channels], vec![channels]);
            let original_out = zeros(&ctx, vec![channels], DType::BF16);
            let fused_out = zeros(&ctx, vec![channels], DType::BF16);
            ops::gdn_causal_conv_step(&ctx, &reference[4], &next, &weight, &original_out).unwrap();
            ops::gdn_causal_conv_step(&ctx, &candidate[4], &next, &weight, &fused_out).unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(read(&original_out), read(&fused_out));
            assert_eq!(read(&reference[4]), read(&candidate[4]));
        }
    }
}

fn worst_relative(produced: &[f32], expected: &[f32]) -> f32 {
    assert_eq!(produced.len(), expected.len());
    let mut worst = 0.0f32;
    for (got, want) in produced.iter().zip(expected.iter()) {
        // The magnitudes here are order 1, so a small absolute floor keeps
        // near-zero outputs from dominating a relative measure.
        worst = worst.max((got - want).abs() / want.abs().max(1e-2));
    }
    worst
}

#[test]
#[ignore = "requires a GPU"]
fn fused_fp16_gated_norm_matches_explicit_bf16_conversion() {
    let ctx = CudaContext::new(0).unwrap();
    let heads = 48;
    let head_dim = 128;
    let weight = upload_bf16(&ctx, &vec![1.0; head_dim], vec![head_dim]);
    for tokens in [1, 3, 63, 64, 65, 129] {
        let dims = vec![tokens, heads, head_dim];
        let count = tokens * heads * head_dim;
        let bytes: Vec<u8> = (0..count)
            .flat_map(|index| {
                let raw = (index % 65536) as u16;
                let finite = if raw & 0x7c00 == 0x7c00 { 0 } else { raw };
                finite.to_le_bytes()
            })
            .collect();
        let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
        buffer.copy_from_host(&bytes).unwrap();
        let input = buffer.as_tensor(Shape::new(dims.clone()), DType::F16).unwrap();
        let gate_values: Vec<f32> = (0..count)
            .map(|index| ((index * 7919 % 65536) as f32 - 32768.0) / 4096.0)
            .collect();
        let gate = upload_bf16(&ctx, &gate_values, dims.clone());
        let wide = zeros(&ctx, dims.clone(), DType::BF16);
        let expected = zeros(&ctx, dims.clone(), DType::BF16);
        let actual = zeros(&ctx, dims, DType::BF16);
        CudaBuffer::from_tensor(&expected).unwrap().copy_from_host(&vec![0xaa; count * 2]).unwrap();
        CudaBuffer::from_tensor(&actual).unwrap().copy_from_host(&vec![0x55; count * 2]).unwrap();
        ops::convert_f16_to_bf16(&ctx, &input, &wide, count).unwrap();
        ops::gdn_gated_norm_seq(&ctx, &wide, &gate, &weight, &expected, tokens, heads, head_dim, 1e-6).unwrap();
        ops::gdn_gated_norm_seq(&ctx, &input, &gate, &weight, &actual, tokens, heads, head_dim, 1e-6).unwrap();
        ctx.synchronize().unwrap();
        let produced = read_bf16(&actual);
        assert_eq!(produced.len(), count);
        assert!(produced.iter().all(|value| value.is_finite()));
        assert_eq!(produced, read_bf16(&expected), "tokens={tokens}");
        let wrong_type = zeros(&ctx, vec![tokens, heads, head_dim], DType::F32);
        assert!(ops::gdn_gated_norm_seq(&ctx, &wrong_type, &gate, &weight, &actual, tokens, heads, head_dim, 1e-6).is_err());
    }
}

#[test]
#[ignore = "requires a GPU; run with --test-threads=1"]
fn preserved_warp_prepare_matches_four_warp_reduction() {
    let ctx = CudaContext::new(0).unwrap();
    let key_heads = 16;
    let value_heads = 48;
    let dim = 128;
    let row_width = (2 * key_heads + value_heads) * dim;
    let previous = std::env::var_os("APXINF_GDN_PREPARE_PARALLEL");
    for tokens in [1, 3, 63, 64, 65, 129] {
        let values: Vec<f32> = (0..tokens * row_width)
            .map(|index| ((index * 7919 + 37) % 65536) as f32 / 4096.0 - 8.0)
            .collect();
        let fused = upload_bf16(&ctx, &values, vec![tokens, row_width]);
        let decay = upload_f32(&ctx, &vec![-0.5; tokens * value_heads], vec![tokens, value_heads]);
        let make_outputs = || [
            zeros(&ctx, vec![tokens, key_heads, dim], DType::F16),
            zeros(&ctx, vec![tokens, key_heads, dim], DType::F16),
            zeros(&ctx, vec![tokens, value_heads, dim], DType::F16),
            zeros(&ctx, vec![tokens, value_heads], DType::F32),
        ];
        let baseline = make_outputs();
        let candidate = make_outputs();
        for (variant, outputs) in [("0", &baseline), ("5", &candidate)] {
            std::env::set_var("APXINF_GDN_PREPARE_PARALLEL", variant);
            for output in outputs {
                let storage = CudaBuffer::from_tensor(output).unwrap();
                storage.copy_from_host(&vec![if variant == "0" { 0xaa } else { 0x55 }; storage.len()]).unwrap();
            }
            ops::gdn_prepare_flashinfer(&ctx, &fused, &outputs[0], &outputs[1], &outputs[2],
                &decay, &outputs[3], tokens, row_width, key_heads, value_heads, dim, 1e-6).unwrap();
        }
        ctx.synchronize().unwrap();
        for (expected, actual) in baseline.iter().zip(candidate.iter()) {
            let first = CudaBuffer::from_tensor(expected).unwrap();
            let second = CudaBuffer::from_tensor(actual).unwrap();
            assert!(first.len() > 0);
            assert_eq!(first.len(), second.len());
            let mut expected_bytes = vec![0; first.len()];
            let mut actual_bytes = vec![0; second.len()];
            first.copy_to_host(&mut expected_bytes).unwrap();
            second.copy_to_host(&mut actual_bytes).unwrap();
            assert_eq!(expected_bytes, actual_bytes, "tokens={tokens}");
        }
    }
    match previous {
        Some(value) => std::env::set_var("APXINF_GDN_PREPARE_PARALLEL", value),
        None => std::env::remove_var("APXINF_GDN_PREPARE_PARALLEL"),
    }
}

const CHANNELS: usize = 10240;
const CONV_WIDTH: usize = 4;

fn conv_inputs(tokens: usize) -> (Vec<f32>, Vec<f32>) {
    let input: Vec<f32> = (0..tokens * CHANNELS)
        .map(|index| {
            let token = (index / CHANNELS) as f32;
            let channel = (index % CHANNELS) as f32;
            ((channel * 0.013).sin() + 0.5 * (token * 0.29).cos()) * 1.2
        })
        .collect();
    let weight: Vec<f32> = (0..CHANNELS * CONV_WIDTH)
        .map(|index| ((index as f32) * 0.0007).sin() * 0.8)
        .collect();
    (input, weight)
}

/// The batched conv must reproduce the single-token conv token for token, and
/// leave the same window behind.
#[test]
#[ignore = "requires a GPU"]
fn batched_conv_matches_the_single_token_path() {
    let ctx = CudaContext::new(0).unwrap();

    // 65 deliberately straddles a chunk boundary and is not a multiple of any
    // natural tiling, so an off-by-one at the tail shows up.
    for &tokens in &[1usize, 2, 4, 5, 63, 64, 65, 129, 2049] {
        let (input, weight) = conv_inputs(tokens);
        let input_device = upload_bf16(&ctx, &input, vec![tokens, CHANNELS]);
        let weight_device = upload_bf16(&ctx, &weight, vec![CHANNELS, CONV_WIDTH]);

        // Batched: one launch over the whole prompt.
        let batched_out = zeros(&ctx, vec![tokens, CHANNELS], DType::BF16);
        let batched_window = zeros(&ctx, vec![CHANNELS, CONV_WIDTH], DType::F32);
        ops::gdn_causal_conv_forward(
            &ctx,
            &input_device,
            &weight_device,
            &batched_out,
            Some(&batched_window),
            tokens,
            CHANNELS,
            CONV_WIDTH,
        )
        .unwrap();

        // Reference: the single-token kernel, one call per token, carrying its
        // own window exactly as decode does.
        let stepped_out = zeros(&ctx, vec![tokens, CHANNELS], DType::BF16);
        let stepped_window = zeros(&ctx, vec![CHANNELS, CONV_WIDTH], DType::F32);
        for token in 0..tokens {
            let token_in = slice_of(
                &input_device,
                token * CHANNELS,
                vec![CHANNELS],
                DType::BF16,
            );
            let token_out = slice_of(
                &stepped_out,
                token * CHANNELS,
                vec![CHANNELS],
                DType::BF16,
            );
            ops::gdn_causal_conv_step(
                &ctx,
                &stepped_window,
                &token_in,
                &weight_device,
                &token_out,
            )
            .unwrap();
        }
        ctx.synchronize().unwrap();

        let worst = worst_relative(&read_bf16(&batched_out), &read_bf16(&stepped_out));
        assert_eq!(read_bf16(&batched_out), read_bf16(&stepped_out), "conv bit equality, tokens={tokens}");
        println!("tokens={tokens:3}  conv output worst relative: {worst:.6}");
        // Both paths accumulate in f32 and round once to BF16, so they should
        // agree to the last bit; the tolerance only allows for the two loops
        // summing in a different order.
        assert!(
            worst < 1e-2,
            "batched conv diverged from the stepped conv at tokens={tokens}: {worst}"
        );

        // The window hand-off. A prefill that produces correct outputs but a
        // wrong window still breaks the very next decode step.
        let window_worst =
            worst_relative(&read_f32(&batched_window), &read_f32(&stepped_window));
        assert_eq!(read_f32(&batched_window), read_f32(&stepped_window), "conv state equality, tokens={tokens}");
        println!("tokens={tokens:3}  conv window  worst relative: {window_worst:.6}");
        assert!(
            window_worst < 1e-3,
            "batched conv left a different window at tokens={tokens}: {window_worst}"
        );
    }
}

/// A prompt followed by a decode step must equal running every token through
/// the single-token path. This is the property prefill actually depends on.
#[test]
#[ignore = "requires a GPU"]
fn decode_continues_correctly_after_a_batched_conv() {
    let ctx = CudaContext::new(0).unwrap();
    let prompt = 37usize;
    let total = prompt + 1;

    let (input, weight) = conv_inputs(total);
    let input_device = upload_bf16(&ctx, &input, vec![total, CHANNELS]);
    let weight_device = upload_bf16(&ctx, &weight, vec![CHANNELS, CONV_WIDTH]);

    // Prefill the first `prompt` tokens in one launch, then step the last one.
    // The wrapper checks shapes exactly, so the prompt span is passed as a
    // view rather than as a row count into a longer buffer.
    let prefill_out = zeros(&ctx, vec![total, CHANNELS], DType::BF16);
    let prefill_window = zeros(&ctx, vec![CHANNELS, CONV_WIDTH], DType::F32);
    let prompt_in = slice_of(&input_device, 0, vec![prompt, CHANNELS], DType::BF16);
    let prompt_out = slice_of(&prefill_out, 0, vec![prompt, CHANNELS], DType::BF16);
    ops::gdn_causal_conv_forward(
        &ctx,
        &prompt_in,
        &weight_device,
        &prompt_out,
        Some(&prefill_window),
        prompt,
        CHANNELS,
        CONV_WIDTH,
    )
    .unwrap();
    let last_in = slice_of(&input_device, prompt * CHANNELS, vec![CHANNELS], DType::BF16);
    let last_out = slice_of(&prefill_out, prompt * CHANNELS, vec![CHANNELS], DType::BF16);
    ops::gdn_causal_conv_step(&ctx, &prefill_window, &last_in, &weight_device, &last_out)
        .unwrap();

    // Reference: every token through the single-token path.
    let stepped_out = zeros(&ctx, vec![total, CHANNELS], DType::BF16);
    let stepped_window = zeros(&ctx, vec![CHANNELS, CONV_WIDTH], DType::F32);
    for token in 0..total {
        let token_in = slice_of(&input_device, token * CHANNELS, vec![CHANNELS], DType::BF16);
        let token_out = slice_of(&stepped_out, token * CHANNELS, vec![CHANNELS], DType::BF16);
        ops::gdn_causal_conv_step(&ctx, &stepped_window, &token_in, &weight_device, &token_out)
            .unwrap();
    }
    ctx.synchronize().unwrap();

    let produced = read_bf16(&prefill_out);
    let expected = read_bf16(&stepped_out);
    // Check the decoded token specifically: it is the one that depends on the
    // window the batched kernel wrote.
    let worst_decoded = worst_relative(
        &produced[prompt * CHANNELS..],
        &expected[prompt * CHANNELS..],
    );
    println!("decode-after-prefill worst relative: {worst_decoded:.6}");
    assert!(
        worst_decoded < 1e-2,
        "the decode step after a batched prefill diverged: {worst_decoded}"
    );
}

const HEADS: usize = 48;
const HEAD_DIM: usize = 128;

/// Sequence-axis decay/beta against the per-token kernel.
#[test]
#[ignore = "requires a GPU"]
fn sequence_decay_and_beta_matches_the_per_token_kernel() {
    let ctx = CudaContext::new(0).unwrap();
    let tokens = 65usize;

    let a: Vec<f32> = (0..tokens * HEADS)
        .map(|index| ((index as f32) * 0.011).sin() * 2.0)
        .collect();
    let b: Vec<f32> = (0..tokens * HEADS)
        .map(|index| ((index as f32) * 0.017).cos() * 1.5)
        .collect();
    let a_log: Vec<f32> = (0..HEADS).map(|head| (head as f32) * 0.01 - 0.4).collect();
    let dt_bias: Vec<f32> = (0..HEADS).map(|head| (head as f32) * 0.02 - 0.3).collect();

    let a_device = upload_bf16(&ctx, &a, vec![tokens, HEADS]);
    let b_device = upload_bf16(&ctx, &b, vec![tokens, HEADS]);
    let a_log_device = upload_bf16(&ctx, &a_log, vec![HEADS]);
    let dt_bias_device = upload_bf16(&ctx, &dt_bias, vec![HEADS]);

    let decay = zeros(&ctx, vec![tokens, HEADS], DType::F32);
    let beta = zeros(&ctx, vec![tokens, HEADS], DType::F32);
    ops::gdn_decay_and_beta_seq(
        &ctx,
        &a_device,
        &b_device,
        &a_log_device,
        &dt_bias_device,
        &decay,
        &beta,
        tokens,
        HEADS,
    )
    .unwrap();

    let stepped_decay = zeros(&ctx, vec![tokens, HEADS], DType::F32);
    let stepped_beta = zeros(&ctx, vec![tokens, HEADS], DType::F32);
    for token in 0..tokens {
        ops::gdn_decay_and_beta(
            &ctx,
            &slice_of(&a_device, token * HEADS, vec![HEADS], DType::BF16),
            &slice_of(&b_device, token * HEADS, vec![HEADS], DType::BF16),
            &a_log_device,
            &dt_bias_device,
            &slice_of(&stepped_decay, token * HEADS, vec![HEADS], DType::F32),
            &slice_of(&stepped_beta, token * HEADS, vec![HEADS], DType::F32),
        )
        .unwrap();
    }
    ctx.synchronize().unwrap();

    let decay_worst = worst_relative(&read_f32(&decay), &read_f32(&stepped_decay));
    let beta_worst = worst_relative(&read_f32(&beta), &read_f32(&stepped_beta));
    println!("decay worst relative: {decay_worst:.6}  beta worst relative: {beta_worst:.6}");
    // Identical arithmetic in both kernels, so this should be exact.
    assert!(decay_worst < 1e-6, "sequence decay diverged: {decay_worst}");
    assert!(beta_worst < 1e-6, "sequence beta diverged: {beta_worst}");
}

/// Sequence-axis gated norm against the per-token kernel.
#[test]
#[ignore = "requires a GPU"]
fn sequence_gated_norm_matches_the_per_token_kernel() {
    let ctx = CudaContext::new(0).unwrap();
    let tokens = 65usize;
    let span = tokens * HEADS * HEAD_DIM;

    let input: Vec<f32> = (0..span)
        .map(|index| ((index as f32) * 0.0031).sin() * 1.7)
        .collect();
    let gate: Vec<f32> = (0..span)
        .map(|index| ((index as f32) * 0.0047).cos() * 1.1)
        .collect();
    let weight: Vec<f32> = (0..HEAD_DIM)
        .map(|index| 1.0 + (index as f32) * 0.002)
        .collect();

    let input_device = upload_bf16(&ctx, &input, vec![tokens, HEADS, HEAD_DIM]);
    let gate_device = upload_bf16(&ctx, &gate, vec![tokens, HEADS, HEAD_DIM]);
    let weight_device = upload_bf16(&ctx, &weight, vec![HEAD_DIM]);

    let output = zeros(&ctx, vec![tokens, HEADS, HEAD_DIM], DType::BF16);
    ops::gdn_gated_norm_seq(
        &ctx,
        &input_device,
        &gate_device,
        &weight_device,
        &output,
        tokens,
        HEADS,
        HEAD_DIM,
        1e-6,
    )
    .unwrap();

    let stepped = zeros(&ctx, vec![tokens, HEADS, HEAD_DIM], DType::BF16);
    let head_span = HEADS * HEAD_DIM;
    for token in 0..tokens {
        ops::gdn_gated_norm(
            &ctx,
            &slice_of(&input_device, token * head_span, vec![HEADS, HEAD_DIM], DType::BF16),
            &slice_of(&gate_device, token * head_span, vec![HEADS, HEAD_DIM], DType::BF16),
            &weight_device,
            &slice_of(&stepped, token * head_span, vec![HEADS, HEAD_DIM], DType::BF16),
            1e-6,
        )
        .unwrap();
    }
    ctx.synchronize().unwrap();

    let worst = worst_relative(&read_bf16(&output), &read_bf16(&stepped));
    assert_eq!(read_bf16(&output), read_bf16(&stepped), "norm bit equality");
    println!("gated norm worst relative: {worst:.6}");
    assert!(worst < 1e-3, "sequence gated norm diverged: {worst}");
}

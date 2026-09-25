//! Gated DeltaNet kernels against a host reference.
//!
//! Scope note, because it matters: the reference here implements the same
//! recurrence the kernel does, so these tests establish that the kernels
//! compute the stated rule correctly -- the head sharing, the state layout,
//! the decay, the rank-1 update, the readout. They do **not** establish that
//! the stated rule is Qwen3.5's. That requires a layer-wise comparison against
//! a reference engine running this checkpoint, which is still open.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{ops, CudaBuffer, CudaContext};

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

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
        .collect()
}

fn read_bf16(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|value| half::bf16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32())
        .collect()
}

/// The same recurrence the kernel implements, in f64 on the host.
#[allow(clippy::too_many_arguments)]
fn reference_step(
    state: &mut [f64],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: &[f32],
    beta: &[f32],
    v_heads: usize,
    k_heads: usize,
    v_dim: usize,
    k_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; v_heads * v_dim];
    let group = v_heads / k_heads;
    for head in 0..v_heads {
        let k_head = head / group;
        let head_decay = (decay[head] as f64).exp();
        let head_beta = beta[head] as f64;
        let base = head * v_dim * k_dim;

        for v_index in 0..v_dim {
            let row = base + v_index * k_dim;
            let mut predicted = 0.0f64;
            for index in 0..k_dim {
                state[row + index] *= head_decay;
                predicted += state[row + index] * k[k_head * k_dim + index] as f64;
            }
            let delta = (v[head * v_dim + v_index] as f64 - predicted) * head_beta;
            let mut out = 0.0f64;
            for index in 0..k_dim {
                state[row + index] += delta * k[k_head * k_dim + index] as f64;
                out += state[row + index] * q[k_head * k_dim + index] as f64;
            }
            // The reference divides the query by sqrt(k_dim) before the
            // recurrence. q only ever reaches the readout, so scaling the
            // finished reduction is equivalent.
            output[head * v_dim + v_index] = (out / (k_dim as f64).sqrt()) as f32;
        }
    }
    output
}

#[test]
fn recurrent_step_matches_the_reference_over_many_tokens() {
    let ctx = CudaContext::new(0).unwrap();
    // The model's real head geometry: 48 value heads over 16 key heads, so
    // each key head serves three value heads. Getting that sharing wrong is
    // the most likely structural mistake, and it only shows up when the head
    // counts differ.
    let (v_heads, k_heads, v_dim, k_dim) = (48usize, 16usize, 128usize, 128usize);

    let state_len = ops::gdn_state_elements(v_heads, v_dim, k_dim);
    assert_eq!(state_len, 48 * 128 * 128);
    let device_state = upload_f32(
        &ctx,
        &vec![0.0f32; state_len],
        vec![v_heads, v_dim, k_dim],
    );
    let mut host_state = vec![0.0f64; state_len];

    let output = upload_bf16(&ctx, &vec![0.0f32; v_heads * v_dim], vec![v_heads, v_dim]);

    let mut worst = 0.0f64;
    // Several tokens, so an error in how the state carries forward compounds
    // instead of cancelling.
    for token in 0..6usize {
        let q: Vec<f32> = (0..k_heads * k_dim)
            .map(|index| (((index + token * 13) % 17) as f32 - 8.0) / 11.0)
            .collect();
        let k: Vec<f32> = (0..k_heads * k_dim)
            .map(|index| (((index * 3 + token * 7) % 19) as f32 - 9.0) / 13.0)
            .collect();
        let v: Vec<f32> = (0..v_heads * v_dim)
            .map(|index| (((index * 5 + token * 11) % 23) as f32 - 11.0) / 17.0)
            .collect();
        let decay: Vec<f32> = (0..v_heads)
            .map(|head| -0.05 - 0.01 * (head % 7) as f32)
            .collect();
        let beta: Vec<f32> = (0..v_heads)
            .map(|head| 0.3 + 0.05 * (head % 5) as f32)
            .collect();

        let device_q = upload_bf16(&ctx, &q, vec![k_heads, k_dim]);
        let device_k = upload_bf16(&ctx, &k, vec![k_heads, k_dim]);
        let device_v = upload_bf16(&ctx, &v, vec![v_heads, v_dim]);
        let device_decay = upload_f32(&ctx, &decay, vec![v_heads]);
        let device_beta = upload_f32(&ctx, &beta, vec![v_heads]);

        ops::gdn_recurrent_step(
            &ctx,
            &device_state,
            &device_q,
            &device_k,
            &device_v,
            &device_decay,
            &device_beta,
            &output,
            k_heads,
        )
        .unwrap();
        ctx.synchronize().unwrap();

        // The reference consumes BF16-rounded inputs so the comparison isolates
        // the recurrence rather than re-measuring input rounding.
        let q_rounded: Vec<f32> = q.iter().map(|x| half::bf16::from_f32(*x).to_f32()).collect();
        let k_rounded: Vec<f32> = k.iter().map(|x| half::bf16::from_f32(*x).to_f32()).collect();
        let v_rounded: Vec<f32> = v.iter().map(|x| half::bf16::from_f32(*x).to_f32()).collect();
        let expected = reference_step(
            &mut host_state,
            &q_rounded,
            &k_rounded,
            &v_rounded,
            &decay,
            &beta,
            v_heads,
            k_heads,
            v_dim,
            k_dim,
        );

        let produced = read_bf16(&output);
        for (got, want) in produced.iter().zip(expected.iter()) {
            let denominator = (want.abs() as f64).max(1e-2);
            worst = worst.max((*got as f64 - *want as f64).abs() / denominator);
        }
        println!("token {token}: worst relative {worst:.5}");
    }

    // The kernel accumulates in f32 and writes BF16; the reference is f64.
    // 2% bounds that over six compounding steps.
    assert!(worst < 2e-2, "recurrent step drifted by {worst}");

    // The state itself must match too: an output that happens to agree while
    // the state has diverged would break on the next token.
    let device_final = read_f32(&device_state);
    let mut state_worst = 0.0f64;
    for (got, want) in device_final.iter().zip(host_state.iter()) {
        let denominator = want.abs().max(1e-2);
        state_worst = state_worst.max((*got as f64 - want).abs() / denominator);
    }
    println!("final state worst relative: {state_worst:.5}");
    assert!(state_worst < 2e-2, "state drifted by {state_worst}");
}

#[test]
fn causal_conv_window_advances_like_a_shift_register() {
    let ctx = CudaContext::new(0).unwrap();
    let (channels, width) = (64usize, 4usize);
    let window = upload_f32(&ctx, &vec![0.0f32; channels * width], vec![channels, width]);
    let weight_values: Vec<f32> = (0..channels * width)
        .map(|index| ((index % 5) as f32 - 2.0) / 4.0)
        .collect();
    let weight = upload_bf16(&ctx, &weight_values, vec![channels, width]);
    let output = upload_bf16(&ctx, &vec![0.0f32; channels], vec![channels]);

    let mut history = vec![vec![0.0f32; width]; channels];
    for token in 0..5usize {
        let input_values: Vec<f32> = (0..channels)
            .map(|channel| ((channel + token * 3) % 11) as f32 / 7.0 - 0.5)
            .collect();
        let input = upload_bf16(&ctx, &input_values, vec![channels]);
        ops::gdn_causal_conv_step(&ctx, &window, &input, &weight, &output).unwrap();
        ctx.synchronize().unwrap();

        let produced = read_bf16(&output);
        for channel in 0..channels {
            history[channel].rotate_left(1);
            history[channel][width - 1] = half::bf16::from_f32(input_values[channel]).to_f32();
            let mut accumulator = 0.0f32;
            for index in 0..width {
                accumulator += history[channel][index]
                    * half::bf16::from_f32(weight_values[channel * width + index]).to_f32();
            }
            let expected = accumulator / (1.0 + (-accumulator).exp());
            let error = (produced[channel] - expected).abs() / expected.abs().max(1e-2);
            assert!(
                error < 2e-2,
                "token {token} channel {channel}: {} vs {expected}",
                produced[channel]
            );
        }
    }
}

#[test]
fn l2_normalization_makes_each_head_unit_norm() {
    let ctx = CudaContext::new(0).unwrap();
    let (heads, head_dim) = (16usize, 128usize);
    let values: Vec<f32> = (0..heads * head_dim)
        .map(|index| ((index % 13) as f32 - 6.0) * (1.0 + (index / head_dim) as f32))
        .collect();
    let data = upload_bf16(&ctx, &values, vec![heads, head_dim]);
    ops::gdn_l2_normalize_heads(&ctx, &data, 1e-6).unwrap();
    ctx.synchronize().unwrap();

    let produced = read_bf16(&data);
    for head in 0..heads {
        let norm: f32 = produced[head * head_dim..(head + 1) * head_dim]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        assert!(
            (norm - 1.0).abs() < 2e-2,
            "head {head} has norm {norm}, expected 1"
        );
    }
}

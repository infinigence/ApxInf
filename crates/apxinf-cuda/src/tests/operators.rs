use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::kernels::activation::{gelu_tanh, silu};
use crate::kernels::attention::{
    causal_mask, gqa_bf16, softmax, softmax_causal, vision, AttentionMask,
};
use crate::kernels::cache::append;
use crate::kernels::elementwise::{
    add, add_bias, add_into, euler_two_stage_bf16, euler_update_bf16, mul, mul_style_gate_bf16,
    mul_style_gate_bf16_into, scale, scale_into,
};
use crate::kernels::embedding::{lookup, lookup_bf16, lookup_scaled_bf16};
use crate::kernels::fused::adaptive_gate_residual_bf16;
use crate::kernels::norm::{layer, rms, rms_affine_bf16, rms_bf16};
use crate::kernels::rope::{
    apply, apply_batched, apply_mrope, apply_precomputed_bf16, apply_vision_2d, rope_tables_bf16,
    sinusoidal_time_embedding_bf16,
};
use half::bf16;
use sha2::{Digest, Sha256};

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn gpu_ptr(tensor: &Tensor) -> Result<*mut std::ffi::c_void> {
    Ok(CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?.ptr())
}

fn make_gpu_tensor(shape: Shape, dtype: DType, _device: usize, buffer: CudaBuffer) -> Tensor {
    buffer.into_tensor(shape, dtype)
}

fn upload_u32(ctx: &CudaContext, values: &[u32]) -> CudaBuffer {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
}
use crate::test_util::{
    assert_bf16_close_elementwise, assert_bf16_close_reduction, download_bf16_as_fp32,
    download_bf16_bytes, upload_fp32_as_bf16,
};

fn silu_ref(x: f32) -> f32 {
    x / (1.0f32 + (-x).exp())
}

#[test]
fn silu_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    // A mix of magnitudes and signs so we exercise the tails of exp/sigmoid.
    let input: Vec<f32> = (-32..32).map(|i| (i as f32) * 0.25).collect();
    let expected: Vec<f32> = input.iter().map(|&x| silu_ref(x)).collect();

    let bf_in = upload_fp32_as_bf16(&ctx, &input, vec![input.len()]).unwrap();
    let bf_out = silu(&ctx, &bf_in).unwrap();
    let actual = download_bf16_as_fp32(&bf_out).unwrap();

    assert_bf16_close_elementwise(&actual, &expected);
}

// ── Elementwise: add ──────────────────────────────────────────────

#[test]
fn add_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let n = 128;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 6.4).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32) * -0.05 + 3.2).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    let ta = upload_fp32_as_bf16(&ctx, &a, vec![n]).unwrap();
    let tb = upload_fp32_as_bf16(&ctx, &b, vec![n]).unwrap();
    let out = add(&ctx, &ta, &tb).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── Elementwise: mul ──────────────────────────────────────────────

#[test]
fn mul_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let n = 64;
    let a: Vec<f32> = (0..n).map(|i| (i as f32) * 0.25 - 8.0).collect();
    let b: Vec<f32> = (0..n).map(|i| (i as f32) * 0.125).collect();
    let expected: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x * y).collect();

    let ta = upload_fp32_as_bf16(&ctx, &a, vec![n]).unwrap();
    let tb = upload_fp32_as_bf16(&ctx, &b, vec![n]).unwrap();
    let out = mul(&ctx, &ta, &tb).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── Elementwise: scale ────────────────────────────────────────────

#[test]
fn scale_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let n = 100;
    let input: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - 5.0).collect();
    let factor = 0.25f32;
    let expected: Vec<f32> = input.iter().map(|x| x * factor).collect();

    let t = upload_fp32_as_bf16(&ctx, &input, vec![n]).unwrap();
    let out = scale(&ctx, &t, factor).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

#[test]
fn caller_owned_scale_preserves_two_stage_bf16_euler_rounding() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let state_value = bf16::from_f32(0.72265625);
    let velocity_value = bf16::from_f32(8.1875);
    let dt = -0.1f32;
    let product = bf16::from_f32(velocity_value.to_f32() * dt);
    let expected = bf16::from_f32(state_value.to_f32() + product.to_f32()).to_f32();

    let state = upload_fp32_as_bf16(&ctx, &[state_value.to_f32()], vec![1]).unwrap();
    let velocity = upload_fp32_as_bf16(&ctx, &[velocity_value.to_f32()], vec![1]).unwrap();
    let velocity_buffer = CudaBuffer::from_tensor(&velocity).unwrap();
    let exact = euler_two_stage_bf16(&ctx, &state, &velocity, dt).unwrap();
    let exact_value = download_bf16_as_fp32(&exact).unwrap()[0];
    assert_eq!(exact_value, expected);

    let fused = euler_update_bf16(&ctx, &state, &velocity, dt).unwrap();
    let fused_value = download_bf16_as_fp32(&fused).unwrap()[0];
    assert_ne!(
        fused_value, exact_value,
        "conviction fixture must distinguish one-round fused Euler"
    );
    assert!(euler_two_stage_bf16(&ctx, &state, &velocity, f32::NAN).is_err());
    let bad_velocity = velocity.reshape(vec![1, 1]).unwrap();
    assert!(euler_two_stage_bf16(&ctx, &state, &bad_velocity, dt).is_err());

    let scaled = CudaBuffer::alloc_zeros(2, ctx.device_id()).unwrap();
    assert!(scale_into(&ctx, DType::BF16, &velocity_buffer, &scaled, 2, dt,).is_err());
    assert!(scale_into(&ctx, DType::BF16, &velocity_buffer, &scaled, 1, f32::NAN,).is_err());
}

#[test]
fn caller_owned_style_gate_preserves_bf16_product_before_residual() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let residual_value = bf16::from_f32(0.486328125);
    let projection_value = bf16::from_f32(1.9296875);
    let gate_value = bf16::from_f32(-2.9375);
    let expected_product = bf16::from_f32(projection_value.to_f32() * gate_value.to_f32());
    let expected = bf16::from_f32(residual_value.to_f32() + expected_product.to_f32()).to_f32();

    let residual = upload_fp32_as_bf16(&ctx, &[residual_value.to_f32()], vec![1, 1]).unwrap();
    let projection = upload_fp32_as_bf16(&ctx, &[projection_value.to_f32()], vec![1, 1]).unwrap();
    let style = upload_fp32_as_bf16(&ctx, &[0.0, 0.0, gate_value.to_f32()], vec![3]).unwrap();
    let residual_buffer = CudaBuffer::from_tensor(&residual).unwrap();
    let projection_buffer = CudaBuffer::from_tensor(&projection).unwrap();
    let product = mul_style_gate_bf16(&ctx, &projection, &style).unwrap();
    assert_eq!(
        download_bf16_as_fp32(&product).unwrap()[0],
        expected_product.to_f32()
    );
    let product_buffer = CudaBuffer::from_tensor(&product).unwrap();
    let output_buffer = CudaBuffer::alloc_zeros(2, ctx.device_id()).unwrap();
    add_into(
        &ctx,
        DType::BF16,
        &residual_buffer,
        &product_buffer,
        &output_buffer,
        1,
    )
    .unwrap();
    let exact = make_gpu_tensor(Shape::new(vec![1, 1]), DType::BF16, 0, output_buffer);
    let exact_value = download_bf16_as_fp32(&exact).unwrap()[0];
    assert_eq!(exact_value, expected);

    let fused = adaptive_gate_residual_bf16(&ctx, &projection, &residual, &style).unwrap();
    let fused_value = download_bf16_as_fp32(&fused).unwrap()[0];
    assert_ne!(
        fused_value, exact_value,
        "conviction fixture must distinguish one-round fused gate residual"
    );

    let short_style = CudaBuffer::alloc_zeros(4, ctx.device_id()).unwrap();
    assert!(mul_style_gate_bf16_into(
        &ctx,
        &projection_buffer,
        &short_style,
        &product_buffer,
        1,
        1,
    )
    .is_err());
    let bad_style = style.reshape(vec![1, 3]).unwrap();
    assert!(mul_style_gate_bf16(&ctx, &projection, &bad_style).is_err());
}

#[test]
fn explicit_scaled_embedding_preserves_bf16_scale_contract() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let width = 2560usize;
    let table_value = bf16::from_f32(-0.99609375);
    let table =
        upload_fp32_as_bf16(&ctx, &vec![table_value.to_f32(); width], vec![1, width]).unwrap();
    let ids = upload_u32(&ctx, &[0]);
    let bf16_scale = bf16::from_f32((width as f32).sqrt()).to_f32();
    assert_eq!(bf16_scale, 50.5);

    let exact = lookup_scaled_bf16(&ctx, &table, &ids, 1, bf16_scale).unwrap();
    let exact_value = download_bf16_as_fp32(&exact).unwrap()[0];
    let expected = bf16::from_f32(table_value.to_f32() * bf16_scale).to_f32();
    assert_eq!(exact_value, expected);

    let legacy = lookup_bf16(&ctx, &table, &ids, 1).unwrap();
    let legacy_value = download_bf16_as_fp32(&legacy).unwrap()[0];
    assert_ne!(
        legacy_value, exact_value,
        "conviction fixture must distinguish FP32 sqrt(width) from BF16 scale"
    );
    assert!(lookup_scaled_bf16(&ctx, &table, &ids, 1, f32::NAN).is_err());
}

#[test]
fn affine_rmsnorm_adds_raw_weight_offset_in_fp32() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let input = upload_fp32_as_bf16(&ctx, &[-4.0, -4.0], vec![1, 2]).unwrap();
    let raw_value = bf16::from_f32(-0.056640625);
    let raw_weight =
        upload_fp32_as_bf16(&ctx, &[raw_value.to_f32(), raw_value.to_f32()], vec![2]).unwrap();
    let exact = rms_affine_bf16(&ctx, &input, &raw_weight, 1e-6, 1.0).unwrap();
    let exact_value = download_bf16_as_fp32(&exact).unwrap()[0];
    assert_eq!(exact_value, -0.94140625);

    let early_scale = bf16::from_f32(1.0 + raw_value.to_f32()).to_f32();
    let rounded_weight = upload_fp32_as_bf16(&ctx, &[early_scale, early_scale], vec![2]).unwrap();
    let legacy = rms_bf16(&ctx, &input, &rounded_weight, 1e-6).unwrap();
    let legacy_value = download_bf16_as_fp32(&legacy).unwrap()[0];
    assert_ne!(
        legacy_value, exact_value,
        "conviction fixture must distinguish early BF16 affine rounding"
    );
    assert!(rms_affine_bf16(&ctx, &input, &raw_weight, 1e-6, f32::NAN).is_err());
}

#[test]
fn precomputed_rope_preserves_unfused_bf16_boundaries() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let input_values = [0.5f32, -0.75, 1.25, -1.5];
    let cosine_values = [0.5f32, 0.25, 0.5, 0.25];
    let sine_values = [0.75f32, -0.5, 0.75, -0.5];
    let input = upload_fp32_as_bf16(&ctx, &input_values, vec![1, 1, 4]).unwrap();
    let cosine = upload_fp32_as_bf16(&ctx, &cosine_values, vec![1, 4]).unwrap();
    let sine = upload_fp32_as_bf16(&ctx, &sine_values, vec![1, 4]).unwrap();
    let output = apply_precomputed_bf16(&ctx, &input, &cosine, &sine).unwrap();
    let actual = download_bf16_as_fp32(&output).unwrap();

    let input_bf16 = input_values.map(bf16::from_f32);
    let cosine_bf16 = cosine_values.map(bf16::from_f32);
    let sine_bf16 = sine_values.map(bf16::from_f32);
    let mut expected = [0.0f32; 4];
    for pair in 0..2 {
        let first_cos = bf16::from_f32(input_bf16[pair].to_f32() * cosine_bf16[pair].to_f32());
        let first_sin = bf16::from_f32(-input_bf16[2 + pair].to_f32() * sine_bf16[pair].to_f32());
        let second_cos =
            bf16::from_f32(input_bf16[2 + pair].to_f32() * cosine_bf16[2 + pair].to_f32());
        let second_sin = bf16::from_f32(input_bf16[pair].to_f32() * sine_bf16[2 + pair].to_f32());
        expected[pair] = bf16::from_f32(first_cos.to_f32() + first_sin.to_f32()).to_f32();
        expected[2 + pair] = bf16::from_f32(second_cos.to_f32() + second_sin.to_f32()).to_f32();
    }
    assert_eq!(actual, expected);

    let bad_cosine = cosine.reshape(vec![2, 2]).unwrap();
    assert!(apply_precomputed_bf16(&ctx, &input, &bad_cosine, &sine).is_err());
}

// ── Reduction: rms_norm ───────────────────────────────────────────

#[test]
fn gpu_sinusoidal_time_embedding_matches_declared_fp32_formula() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let dimension = 1024usize;
    let min_period = 4e-3f32;
    let max_period = 4.0f32;
    let mut current = 1.0f32;
    let times = (0..10)
        .map(|_| {
            let value = bf16::from_f32(current).to_f32();
            current += -0.1f32;
            value
        })
        .collect::<Vec<_>>();
    let time_tensor = upload_fp32_as_bf16(&ctx, &times, vec![times.len()]).unwrap();
    let output =
        sinusoidal_time_embedding_bf16(&ctx, &time_tensor, dimension, min_period, max_period)
            .unwrap();
    assert_eq!(output.shape().dims(), [10, 1024]);

    let half = dimension / 2;
    let mut expected = vec![0.0f32; times.len() * dimension];
    for (step, &time) in times.iter().enumerate() {
        for frequency in 0..half {
            let fraction_step = 1.0f32 / (half - 1) as f32;
            let fraction = if frequency < half / 2 {
                fraction_step * frequency as f32
            } else {
                1.0f32 - fraction_step * (half - frequency - 1) as f32
            };
            let period = min_period * (max_period / min_period).powf(fraction);
            let angle = (1.0f32 / period * (2.0f32 * std::f32::consts::PI)) * time;
            expected[step * dimension + frequency] = bf16::from_f32(angle.sin()).to_f32();
            expected[step * dimension + half + frequency] = bf16::from_f32(angle.cos()).to_f32();
        }
    }
    let actual = download_bf16_as_fp32(&output).unwrap();
    assert_bf16_close_elementwise(&actual, &expected);
    // OpenDM e41 / torch 2.11+cu130 reference receipt over raw BF16 bytes:
    // c7ca95ac394388a96d4485f223a3c1671f1f98a0dcd32dadf8cae87666e538ba.
    // NVCC 12.3/SM89 differs at 12/10,240 transcendental rounding points, so
    // the all-element tolerance plus exact endpoints below are the portable
    // correctness gate; unlike RoPE, this receipt is deliberately not asserted.
    for (index, expected) in [
        (0usize, -6.198883056640625e-05f32),
        (1, -0.78125),
        (255, -0.251953125),
        (256, -0.80078125),
        (511, 1.0),
        (512, 1.0),
        (513, -0.62109375),
        (767, 0.96875),
        (768, 0.6015625),
        (1023, -4.377216100692749e-08),
        (1024, -0.6328125),
        (5119, 0.5859375),
        (5120, -3.0994415283203125e-05),
        (10239, 0.98828125),
    ] {
        assert_eq!(actual[index], expected, "time embedding index {index}");
    }

    let zero = upload_fp32_as_bf16(&ctx, &[0.0], vec![1]).unwrap();
    let endpoints = sinusoidal_time_embedding_bf16(&ctx, &zero, 8, min_period, max_period).unwrap();
    assert_eq!(
        download_bf16_as_fp32(&endpoints).unwrap(),
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    );
    assert!(sinusoidal_time_embedding_bf16(&ctx, &zero, 7, min_period, max_period).is_err());
}

#[test]
fn gpu_rope_tables_cover_default_and_linear_scaled_positions() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let positions = [0u32, 1, 563, 564];
    let positions_buffer = upload_u32(&ctx, &positions);
    let head_dim = 256usize;

    for (theta, factor) in [(10_000.0f32, 1.0f32), (1_000_000.0f32, 8.0f32)] {
        let tables = rope_tables_bf16(
            &ctx,
            &positions_buffer,
            positions.len(),
            head_dim,
            theta,
            factor,
        )
        .unwrap();
        assert_eq!(tables.cosine.shape().dims(), [positions.len(), head_dim]);
        assert_eq!(tables.sine.shape().dims(), [positions.len(), head_dim]);
        let cosine = download_bf16_as_fp32(&tables.cosine).unwrap();
        let sine = download_bf16_as_fp32(&tables.sine).unwrap();
        let half = head_dim / 2;
        let mut expected_cosine = vec![0.0f32; positions.len() * head_dim];
        let mut expected_sine = vec![0.0f32; positions.len() * head_dim];
        for (token, &position) in positions.iter().enumerate() {
            for pair in 0..half {
                let inverse_frequency =
                    1.0f32 / theta.powf(2.0f32 * pair as f32 / head_dim as f32) / factor;
                let angle = position as f32 * inverse_frequency;
                let cos = bf16::from_f32(angle.cos()).to_f32();
                let sin = bf16::from_f32(angle.sin()).to_f32();
                expected_cosine[token * head_dim + pair] = cos;
                expected_cosine[token * head_dim + half + pair] = cos;
                expected_sine[token * head_dim + pair] = sin;
                expected_sine[token * head_dim + half + pair] = sin;
            }
        }
        assert_bf16_close_elementwise(&cosine, &expected_cosine);
        assert_bf16_close_elementwise(&sine, &expected_sine);
        for token in 0..positions.len() {
            assert_eq!(
                &cosine[token * head_dim..token * head_dim + half],
                &cosine[token * head_dim + half..(token + 1) * head_dim]
            );
            assert_eq!(
                &sine[token * head_dim..token * head_dim + half],
                &sine[token * head_dim + half..(token + 1) * head_dim]
            );
        }
    }

    let all_positions = (0u32..564).collect::<Vec<_>>();
    let all_positions_buffer = upload_u32(&ctx, &all_positions);
    // Raw-BF16 hashes captured from OpenDM e41 / torch 2.11+cu130 and
    // independently reproduced by this kernel with NVCC 12.3 on SM89.
    for (theta, factor, cosine_hash, sine_hash, selected) in [
        (
            10_000.0f32,
            1.0f32,
            "0349ff6171990255b0e8fdf5ab9e3a7c2067a3e31f2ead11308a2c0c83d7fa34",
            "e1559a88b9a5e5fa2de4ac146aa743ebdb8823eb609d78f8a5dab34c4381a81c",
            [
                (256usize, 0.5390625f32, 0.83984375f32),
                (144128, -0.79296875, -0.609375),
                (144129, -0.7421875, 0.66796875),
                (144255, 1.0, 0.060546875),
            ],
        ),
        (
            1_000_000.0f32,
            8.0f32,
            "380764d83226facc7422bfc32beb572e18102e7fc1adc1feaa99a75dd4f20fb8",
            "68b6a6d212a5a0d9a03b6a9ccccb3845ef22f167081b4669b6d1a25cf3ff8be3",
            [
                (256usize, 0.9921875f32, 0.12451171875f32),
                (144128, 0.306640625, 0.953125),
                (144129, 0.94140625, 0.3359375),
                (144255, 1.0, 7.82012939453125e-05),
            ],
        ),
    ] {
        let tables = rope_tables_bf16(
            &ctx,
            &all_positions_buffer,
            all_positions.len(),
            head_dim,
            theta,
            factor,
        )
        .unwrap();
        let cosine = download_bf16_as_fp32(&tables.cosine).unwrap();
        let sine = download_bf16_as_fp32(&tables.sine).unwrap();
        for (index, expected_cosine, expected_sine) in selected {
            assert_eq!(cosine[index], expected_cosine, "cosine index {index}");
            assert_eq!(sine[index], expected_sine, "sine index {index}");
        }
        assert_eq!(
            sha256_hex(&download_bf16_bytes(&tables.cosine).unwrap()),
            cosine_hash,
            "official S564 cosine BF16 oracle hash"
        );
        assert_eq!(
            sha256_hex(&download_bf16_bytes(&tables.sine).unwrap()),
            sine_hash,
            "official S564 sine BF16 oracle hash"
        );
    }

    assert!(rope_tables_bf16(
        &ctx,
        &positions_buffer,
        positions.len(),
        head_dim,
        10_000.0,
        0.0,
    )
    .is_err());
    let short_positions = upload_u32(&ctx, &positions[..3]);
    assert!(rope_tables_bf16(
        &ctx,
        &short_positions,
        positions.len(),
        head_dim,
        10_000.0,
        1.0,
    )
    .is_err());
}

#[test]
fn rms_norm_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (2usize, 64usize);
    let input: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 17) as f32 - 8.0) * 0.1)
        .collect();
    let weight: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32) * 0.01).collect();
    let eps = 1e-5f32;

    // Reference computation
    let mut expected = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let off = r * cols;
        let row = &input[off..off + cols];
        let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv_rms = (mean_sq + eps).sqrt().recip();
        for i in 0..cols {
            expected[off + i] = row[i] * inv_rms * weight[i];
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let t_w = upload_fp32_as_bf16(&ctx, &weight, vec![cols]).unwrap();
    let out = rms(&ctx, &t_in, &t_w, eps).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── Reduction: softmax ────────────────────────────────────────────

#[test]
fn softmax_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (3usize, 32usize);
    let input: Vec<f32> = (0..rows * cols)
        .map(|i| ((i % 11) as f32 - 5.0) * 0.5)
        .collect();

    let mut expected = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let off = r * cols;
        let row = &input[off..off + cols];
        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row.iter().map(|x| (x - max_v).exp()).sum();
        for i in 0..cols {
            expected[off + i] = (row[i] - max_v).exp() / sum;
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let out = softmax(&ctx, &t_in).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── RoPE (batched, half-split) ────────────────────────────────────

#[test]
fn rope_batched_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (2usize, 2usize, 8usize);
    let theta = 10000.0f32;
    let pos_offset = 3u32;

    let input: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| ((i as f32) * 0.1).sin() * 2.0)
        .collect();

    // fp32 reference (half-split): pair (i, i + head_dim/2)
    let mut expected = vec![0.0f32; input.len()];
    let half = head_dim / 2;
    for s in 0..seq_len {
        let pos = pos_offset as usize + s;
        for h in 0..n_heads {
            let base = s * n_heads * head_dim + h * head_dim;
            for pair in 0..half {
                let freq = 1.0f32 / theta.powf(2.0 * pair as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let c = angle.cos();
                let sn = angle.sin();
                let x0 = input[base + pair];
                let x1 = input[base + half + pair];
                expected[base + pair] = x0 * c - x1 * sn;
                expected[base + half + pair] = x0 * sn + x1 * c;
            }
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![seq_len, n_heads, head_dim]).unwrap();
    let out = apply_batched(&ctx, &t_in, n_heads, head_dim, theta, pos_offset).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── RoPE (interleaved pairs) ──────────────────────────────────────

#[test]
fn rope_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (1usize, 2usize, 8usize);
    let theta = 10000.0f32;
    let pos_offset = 5u32;

    let input: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| ((i as f32) * 0.2).cos())
        .collect();

    // fp32 reference for the interleaved (2i, 2i+1) variant
    let mut expected = vec![0.0f32; input.len()];
    for s in 0..seq_len {
        let pos = pos_offset as usize + s;
        for h in 0..n_heads {
            let base = s * n_heads * head_dim + h * head_dim;
            for pair in 0..head_dim / 2 {
                let freq = 1.0f32 / theta.powf(2.0 * pair as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let c = angle.cos();
                let sn = angle.sin();
                let x0 = input[base + 2 * pair];
                let x1 = input[base + 2 * pair + 1];
                expected[base + 2 * pair] = x0 * c - x1 * sn;
                expected[base + 2 * pair + 1] = x0 * sn + x1 * c;
            }
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![seq_len, n_heads, head_dim]).unwrap();
    let out = apply(&ctx, &t_in, n_heads, head_dim, theta, pos_offset).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── Embedding lookup ──────────────────────────────────────────────

#[test]
fn embedding_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (vocab, embed_dim) = (16usize, 8usize);
    let seq = [3u32, 0u32, 15u32];
    let table: Vec<f32> = (0..vocab * embed_dim)
        .map(|i| (i as f32) * 0.01 - 1.0)
        .collect();

    let mut expected = Vec::with_capacity(seq.len() * embed_dim);
    for &tid in &seq {
        let off = tid as usize * embed_dim;
        expected.extend_from_slice(&table[off..off + embed_dim]);
    }

    // Upload table as bf16 and ids as raw u32 buffer.
    let t_table = upload_fp32_as_bf16(&ctx, &table, vec![vocab, embed_dim]).unwrap();
    let ids_bytes: Vec<u8> = seq.iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let ids_buf = crate::buffer::CudaBuffer::alloc(ids_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    ids_buf
        .copy_from_host(&ids_bytes)
        .map_err(Error::Cuda)
        .unwrap();

    let out = lookup(&ctx, &t_table, &ids_buf, seq.len()).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── Causal mask ───────────────────────────────────────────────────

#[test]
fn causal_mask_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (4usize, 6usize);
    let kv_offset = 0u32;
    let input: Vec<f32> = (0..rows * cols).map(|i| i as f32 * 0.1).collect();
    // Expected: below the diagonal + kv_offset stays, above becomes -inf.
    let mut expected = input.clone();
    for r in 0..rows {
        for c in 0..cols {
            if c > r + kv_offset as usize {
                expected[r * cols + c] = f32::NEG_INFINITY;
            }
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let out = causal_mask(&ctx, &t_in, kv_offset).unwrap();
    let got = download_bf16_as_fp32(&out).unwrap();
    // Special-case -inf comparison (any tolerance fails for infinities).
    for i in 0..rows * cols {
        if expected[i].is_infinite() {
            assert!(
                got[i].is_infinite() && got[i].is_sign_negative(),
                "expected -inf at {i}, got {}",
                got[i]
            );
        } else {
            assert!(
                (got[i] - expected[i]).abs() <= 1e-3 + 1e-2 * expected[i].abs(),
                "idx {i}: got {}, expected {}",
                got[i],
                expected[i]
            );
        }
    }
}

// ── Attention softmax (fused causal + softmax) ────────────────────

#[test]
fn attention_softmax_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, kv_len) = (2usize, 3usize, 5usize);
    let rows = seq_len * n_heads;
    let cols = kv_len;
    let kv_offset = 0u32;
    let input: Vec<f32> = (0..rows * cols)
        .map(|i| ((i as f32) % 7.0) * 0.3 - 1.0)
        .collect();

    // Reference: for each row, seq_pos = row / n_heads; valid_cols = min(seq_pos + kv_offset + 1, cols).
    let mut expected = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let seq_pos = r / n_heads;
        let valid = (seq_pos + kv_offset as usize + 1).min(cols);
        let row = &input[r * cols..r * cols + cols];
        let max_v = row[..valid]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = row[..valid].iter().map(|x| (x - max_v).exp()).sum();
        for c in 0..cols {
            if c < valid {
                expected[r * cols + c] = (row[c] - max_v).exp() / sum;
            } else {
                expected[r * cols + c] = 0.0;
            }
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let out = softmax_causal(&ctx, &t_in, kv_offset, n_heads as u32).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── KV cache append ───────────────────────────────────────────────

#[test]
fn kv_cache_append_bf16_writes_correct_slot() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_kv_heads, head_dim, max_seq_len) = (2usize, 4usize, 16usize);
    let seq_len = 3usize; // current cache position (append starts here)
    let append_len = 2usize;

    // Fresh zero cache, one layer.
    let cache_bytes = n_kv_heads * max_seq_len * head_dim * 2;
    let cache_buf = crate::buffer::CudaBuffer::alloc_zeros(cache_bytes, 0)
        .map_err(Error::Cuda)
        .unwrap();

    // New data layout: [append_len, n_kv_heads, head_dim]
    let new_data: Vec<f32> = (0..append_len * n_kv_heads * head_dim)
        .map(|i| (i as f32) + 1.0)
        .collect();
    let new_t =
        upload_fp32_as_bf16(&ctx, &new_data, vec![append_len, n_kv_heads, head_dim]).unwrap();

    append(
        &ctx,
        &cache_buf,
        &new_t,
        n_kv_heads,
        head_dim,
        max_seq_len,
        seq_len,
        append_len,
    )
    .unwrap();

    // Read the cache back and validate the written slot.
    unsafe {
        crate::ffi::check_cuda(crate::ffi::cudaDeviceSynchronize()).unwrap();
    }
    let mut cache_host = vec![0u8; cache_bytes];
    unsafe {
        crate::ffi::check_cuda(crate::ffi::cudaMemcpy(
            cache_host.as_mut_ptr() as *mut std::ffi::c_void,
            cache_buf.ptr() as *const std::ffi::c_void,
            cache_bytes,
            crate::ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        ))
        .unwrap();
    }

    // Interpret as bf16 → fp32 host slice.
    let cache_bf: Vec<half::bf16> = cache_host
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
        .collect();
    // For each (s, h, d), cache[h * max_seq_len * head_dim + (seq_len+s)*head_dim + d]
    // should equal new_data[s*n_kv_heads*head_dim + h*head_dim + d].
    for s in 0..append_len {
        for h in 0..n_kv_heads {
            for d in 0..head_dim {
                let cache_idx = h * max_seq_len * head_dim + (seq_len + s) * head_dim + d;
                let src_idx = s * n_kv_heads * head_dim + h * head_dim + d;
                let got = cache_bf[cache_idx].to_f32();
                let want = new_data[src_idx];
                assert!(
                    (got - want).abs() < 1e-2,
                    "cache[{cache_idx}] got {got}, want {want}"
                );
            }
        }
    }
}

// ── Decode-pos kernel variants (rope_decode, attn_softmax_decode, kv_cache_append_decode) ──

#[test]
fn rope_decode_bf16_matches_rope_bf16() {
    // The decode kernel reads pos from a device buffer, seq_len=1 implicitly.
    // Correctness: match the batched form at seq_len=1.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_heads, head_dim) = (2usize, 8usize);
    let theta = 10000.0f32;
    let pos = 4u32;

    let input: Vec<f32> = (0..n_heads * head_dim).map(|i| (i as f32) * 0.1).collect();

    let t_ref = upload_fp32_as_bf16(&ctx, &input, vec![1, n_heads, head_dim]).unwrap();
    let expected_out = apply_batched(&ctx, &t_ref, n_heads, head_dim, theta, pos).unwrap();
    let expected = download_bf16_as_fp32(&expected_out).unwrap();

    // Run decode kernel directly through FFI.
    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![n_heads, head_dim]).unwrap();
    let out_bytes = t_in.size_in_bytes();
    let out_buf = crate::buffer::CudaBuffer::alloc_zeros(out_bytes, 0)
        .map_err(Error::Cuda)
        .unwrap();
    let pos_bytes = pos.to_ne_bytes();
    let pos_buf = crate::buffer::CudaBuffer::alloc(4, 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();

    unsafe {
        crate::ffi::check_cuda(crate::ffi::apxinf_rope_decode_bf16(
            gpu_ptr(&t_in).unwrap(),
            out_buf.ptr(),
            head_dim as u32,
            n_heads as u32,
            theta,
            pos_buf.ptr(),
            ctx.stream().handle(),
        ))
        .unwrap();
        crate::ffi::check_cuda(crate::ffi::cudaStreamSynchronize(ctx.stream().handle())).unwrap();
    }

    let out_tensor = make_gpu_tensor(Shape::new(vec![n_heads, head_dim]), DType::BF16, 0, out_buf);
    let actual = download_bf16_as_fp32(&out_tensor).unwrap();
    assert_bf16_close_elementwise(&actual, &expected);
}

#[test]
fn attention_softmax_decode_bf16_matches_full() {
    // Decode variant is a special case of attention_softmax with rows=n_heads.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_heads, kv_len) = (3usize, 6usize);
    let pos = 4u32; // valid_cols = pos + 1 = 5
    let input: Vec<f32> = (0..n_heads * kv_len)
        .map(|i| ((i as f32) % 5.0) * 0.4 - 1.0)
        .collect();

    // Reference: attention_softmax with rows=n_heads, kv_offset=pos, n_heads=n_heads.
    let t_ref = upload_fp32_as_bf16(&ctx, &input, vec![n_heads, kv_len]).unwrap();
    let expected_out = softmax_causal(&ctx, &t_ref, pos, n_heads as u32).unwrap();
    let expected = download_bf16_as_fp32(&expected_out).unwrap();

    // Run decode kernel directly.
    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![n_heads, kv_len]).unwrap();
    let out_bytes = t_in.size_in_bytes();
    let out_buf = crate::buffer::CudaBuffer::alloc_zeros(out_bytes, 0)
        .map_err(Error::Cuda)
        .unwrap();
    let pos_bytes = pos.to_ne_bytes();
    let pos_buf = crate::buffer::CudaBuffer::alloc(4, 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();

    unsafe {
        crate::ffi::check_cuda(crate::ffi::apxinf_attention_softmax_decode_bf16(
            gpu_ptr(&t_in).unwrap(),
            out_buf.ptr(),
            kv_len as u32,
            n_heads as u32,
            pos_buf.ptr(),
            ctx.stream().handle(),
        ))
        .unwrap();
        crate::ffi::check_cuda(crate::ffi::cudaStreamSynchronize(ctx.stream().handle())).unwrap();
    }

    let out_tensor = make_gpu_tensor(Shape::new(vec![n_heads, kv_len]), DType::BF16, 0, out_buf);
    let actual = download_bf16_as_fp32(&out_tensor).unwrap();
    assert_bf16_close_reduction(&actual, &expected);
}

#[test]
fn kv_cache_append_decode_bf16_writes_correct_slot() {
    // Decode variant: 1 row of new data, position from device buffer.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_kv_heads, head_dim, max_seq_len) = (2usize, 4usize, 16usize);
    let pos = 5u32;

    let cache_bytes = n_kv_heads * max_seq_len * head_dim * 2;
    let cache_buf = crate::buffer::CudaBuffer::alloc_zeros(cache_bytes, 0)
        .map_err(Error::Cuda)
        .unwrap();

    // new_data shape: [n_kv_heads, head_dim] (no leading append_len)
    let new_data: Vec<f32> = (0..n_kv_heads * head_dim)
        .map(|i| (i as f32) + 1.0)
        .collect();
    let new_t = upload_fp32_as_bf16(&ctx, &new_data, vec![n_kv_heads, head_dim]).unwrap();

    let pos_bytes = pos.to_ne_bytes();
    let pos_buf = crate::buffer::CudaBuffer::alloc(4, 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();

    unsafe {
        crate::ffi::check_cuda(crate::ffi::apxinf_kv_cache_append_decode_bf16(
            cache_buf.ptr(),
            gpu_ptr(&new_t).unwrap(),
            n_kv_heads as u32,
            head_dim as u32,
            max_seq_len as u32,
            pos_buf.ptr(),
            ctx.stream().handle(),
        ))
        .unwrap();
        crate::ffi::check_cuda(crate::ffi::cudaDeviceSynchronize()).unwrap();
    }

    let mut cache_host = vec![0u8; cache_bytes];
    unsafe {
        crate::ffi::check_cuda(crate::ffi::cudaMemcpy(
            cache_host.as_mut_ptr() as *mut std::ffi::c_void,
            cache_buf.ptr() as *const std::ffi::c_void,
            cache_bytes,
            crate::ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        ))
        .unwrap();
    }

    let cache_bf: Vec<half::bf16> = cache_host
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
        .collect();
    for h in 0..n_kv_heads {
        for d in 0..head_dim {
            let cache_idx = h * max_seq_len * head_dim + (pos as usize) * head_dim + d;
            let src_idx = h * head_dim + d;
            let got = cache_bf[cache_idx].to_f32();
            let want = new_data[src_idx];
            assert!(
                (got - want).abs() < 1e-2,
                "cache[{cache_idx}] got {got}, want {want}"
            );
        }
    }
}

// ── mRoPE (Qwen3-VL) ──────────────────────────────────────────────

/// Reference implementation mirroring HF `apply_interleaved_mrope`
/// (rotate_half + axis-per-pair lookup). Used as the ground truth in
/// unit tests below.
fn mrope_reference(
    input: &[f32],
    seq_len: usize,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    sections: [usize; 3],
    pos_ids: &[[u32; 3]],
) -> Vec<f32> {
    assert_eq!(pos_ids.len(), seq_len);
    let half = head_dim / 2;
    let mut out = vec![0.0f32; input.len()];
    let (sec_h, sec_w) = (sections[1], sections[2]);
    for s in 0..seq_len {
        for h in 0..n_heads {
            let base = s * n_heads * head_dim + h * head_dim;
            for pair in 0..half {
                let axis = if pair % 3 == 1 && pair < sec_h * 3 {
                    1
                } else if pair % 3 == 2 && pair < sec_w * 3 {
                    2
                } else {
                    0
                };
                let pos = pos_ids[s][axis];
                let freq = 1.0f32 / theta.powf(2.0 * pair as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let c = angle.cos();
                let sn = angle.sin();
                let x0 = input[base + pair];
                let x1 = input[base + half + pair];
                out[base + pair] = x0 * c - x1 * sn;
                out[base + half + pair] = x0 * sn + x1 * c;
            }
        }
    }
    out
}

#[test]
fn rope_mrope_bf16_matches_reference_text_only() {
    // With pos_ids = (i, i, i) for every token, mRoPE degenerates to
    // 1-D RoPE with rotate_half. Verifies the axis dispatch is a no-op
    // when all axes are equal, which is the text-only case.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (3usize, 2usize, 128usize);
    let theta = 5_000_000.0f32;
    let sections = [24usize, 20, 20];

    let input: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| ((i as f32) * 0.02).sin())
        .collect();

    let pos_ids: Vec<[u32; 3]> = (0..seq_len)
        .map(|i| [i as u32, i as u32, i as u32])
        .collect();
    let expected = mrope_reference(
        &input, seq_len, n_heads, head_dim, theta, sections, &pos_ids,
    );

    // Upload input and pos_ids buffer to device.
    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![seq_len, n_heads, head_dim]).unwrap();
    let pos_bytes: Vec<u8> = pos_ids
        .iter()
        .flat_map(|t| t.iter().flat_map(|&v| v.to_ne_bytes()))
        .collect();
    let pos_buf = crate::buffer::CudaBuffer::alloc(pos_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();

    let out = apply_mrope(&ctx, &t_in, n_heads, head_dim, theta, sections, &pos_buf).unwrap();
    let actual = download_bf16_as_fp32(&out).unwrap();
    assert_bf16_close_elementwise(&actual, &expected);
}

#[test]
fn rope_mrope_bf16_matches_reference_distinct_axes() {
    // Distinct (t, h, w) per token — exercises the axis dispatch. The
    // T section (24 pairs; the leftover) is exercised by the tail
    // pair_idx >= 60 which always falls through to T regardless of
    // pair_idx % 3.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (2usize, 4usize, 128usize);
    let theta = 5_000_000.0f32;
    let sections = [24usize, 20, 20];
    let pos_ids: Vec<[u32; 3]> = vec![[7, 3, 11], [8, 4, 12]];

    let input: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| (((i as f32) * 0.03).cos() - 0.1) * 0.5)
        .collect();

    let expected = mrope_reference(
        &input, seq_len, n_heads, head_dim, theta, sections, &pos_ids,
    );

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![seq_len, n_heads, head_dim]).unwrap();
    let pos_bytes: Vec<u8> = pos_ids
        .iter()
        .flat_map(|t| t.iter().flat_map(|&v| v.to_ne_bytes()))
        .collect();
    let pos_buf = crate::buffer::CudaBuffer::alloc(pos_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();

    let out = apply_mrope(&ctx, &t_in, n_heads, head_dim, theta, sections, &pos_buf).unwrap();
    let actual = download_bf16_as_fp32(&out).unwrap();
    assert_bf16_close_elementwise(&actual, &expected);
}

#[test]
fn rope_mrope_decode_bf16_matches_batched_seq1() {
    // Decode kernel: seq_len=1 implicitly, pos_ids buffer is [3] u32.
    // Must match rope_mrope at seq_len=1.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (n_heads, head_dim) = (4usize, 128usize);
    let theta = 5_000_000.0f32;
    let sections = [24usize, 20, 20];
    let pos_ids = [[9u32, 5, 13]];

    let input: Vec<f32> = (0..n_heads * head_dim).map(|i| (i as f32) * 0.01).collect();

    // Reference via batched path (seq_len=1).
    let t_ref = upload_fp32_as_bf16(&ctx, &input, vec![1, n_heads, head_dim]).unwrap();
    let pos_bytes: Vec<u8> = pos_ids
        .iter()
        .flat_map(|t| t.iter().flat_map(|&v| v.to_ne_bytes()))
        .collect();
    let pos_buf_batched = crate::buffer::CudaBuffer::alloc(pos_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf_batched
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();
    let expected_out = apply_mrope(
        &ctx,
        &t_ref,
        n_heads,
        head_dim,
        theta,
        sections,
        &pos_buf_batched,
    )
    .unwrap();
    let expected = download_bf16_as_fp32(&expected_out).unwrap();

    // Decode kernel direct-FFI, [3] pos buffer.
    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![n_heads, head_dim]).unwrap();
    let pos_bytes3: Vec<u8> = pos_ids[0].iter().flat_map(|&v| v.to_ne_bytes()).collect();
    let pos_buf_dec = crate::buffer::CudaBuffer::alloc(pos_bytes3.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf_dec
        .copy_from_host(&pos_bytes3)
        .map_err(Error::Cuda)
        .unwrap();
    let out_buf = crate::buffer::CudaBuffer::alloc_zeros(t_in.size_in_bytes(), 0)
        .map_err(Error::Cuda)
        .unwrap();

    unsafe {
        crate::ffi::check_cuda(crate::ffi::apxinf_rope_mrope_decode_bf16(
            gpu_ptr(&t_in).unwrap(),
            out_buf.ptr(),
            head_dim as u32,
            n_heads as u32,
            theta,
            pos_buf_dec.ptr(),
            sections[1] as u32,
            sections[2] as u32,
            ctx.stream().handle(),
        ))
        .unwrap();
        crate::ffi::check_cuda(crate::ffi::cudaStreamSynchronize(ctx.stream().handle())).unwrap();
    }

    let out_tensor = make_gpu_tensor(Shape::new(vec![n_heads, head_dim]), DType::BF16, 0, out_buf);
    let actual = download_bf16_as_fp32(&out_tensor).unwrap();
    assert_bf16_close_elementwise(&actual, &expected);
}

// ── LayerNorm / GELU-tanh / add-bias (Qwen3-VL vision) ────────────

#[test]
fn layer_norm_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (3usize, 32usize);
    let eps = 1e-6f32;

    let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.05 - 0.7).collect();
    let weight: Vec<f32> = (0..cols).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let bias: Vec<f32> = (0..cols).map(|i| -0.1 + (i as f32) * 0.003).collect();

    // Reference computed in fp32 (as the kernel does internally).
    let mut expected = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let off = r * cols;
        let mean = input[off..off + cols].iter().sum::<f32>() / cols as f32;
        let var = input[off..off + cols]
            .iter()
            .map(|v| (v - mean).powi(2))
            .sum::<f32>()
            / cols as f32;
        let inv = (var + eps).sqrt().recip();
        for c in 0..cols {
            expected[off + c] = weight[c] * (input[off + c] - mean) * inv + bias[c];
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let t_w = upload_fp32_as_bf16(&ctx, &weight, vec![cols]).unwrap();
    let t_b = upload_fp32_as_bf16(&ctx, &bias, vec![cols]).unwrap();
    let out = layer(&ctx, &t_in, &t_w, &t_b, eps).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

#[test]
fn gelu_tanh_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let input: Vec<f32> = (0..65).map(|i| -4.0 + (i as f32) * 0.125).collect();

    let beta = (2.0f32 / std::f32::consts::PI).sqrt();
    let alpha = 0.044715f32;
    let expected: Vec<f32> = input
        .iter()
        .map(|&x| 0.5 * x * (1.0 + (beta * (x + alpha * x * x * x)).tanh()))
        .collect();

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![input.len()]).unwrap();
    let out = gelu_tanh(&ctx, &t_in).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

#[test]
fn add_bias_bf16_matches_fp32_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (rows, cols) = (5usize, 16usize);
    let input: Vec<f32> = (0..rows * cols).map(|i| (i as f32) * 0.1 - 2.0).collect();
    let bias: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.05 - 0.4).collect();
    let mut expected = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            expected[r * cols + c] = input[r * cols + c] + bias[c];
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![rows, cols]).unwrap();
    let t_b = upload_fp32_as_bf16(&ctx, &bias, vec![cols]).unwrap();
    let out = add_bias(&ctx, &t_in, &t_b).unwrap();
    assert_bf16_close_elementwise(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── Vision 2D-RoPE ───────────────────────────────────────────────

#[test]
fn rope_vision_2d_bf16_matches_reference() {
    // HF vision RoPE: head_dim=64, 16 freq pairs per axis (h then w).
    // pair p < 16 uses h coord; pair p >= 16 uses w coord.
    // inv_freq[i] = 1/theta^(2i/32) for i in [0,16).  rotate_half.
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq_len, n_heads, head_dim) = (2usize, 4usize, 64usize);
    let theta = 10000.0f32;
    let pos_ids: Vec<[u32; 2]> = vec![[3u32, 7], [5, 11]];

    let input: Vec<f32> = (0..seq_len * n_heads * head_dim)
        .map(|i| ((i as f32) * 0.03).sin())
        .collect();

    // Reference.
    let half = head_dim / 2; // 32
    let mut expected = vec![0.0f32; input.len()];
    for s in 0..seq_len {
        for h in 0..n_heads {
            let base = s * n_heads * head_dim + h * head_dim;
            for p in 0..half {
                let axis = if p < half / 2 { 0 } else { 1 };
                let pair_in_axis = if p < half / 2 { p } else { p - half / 2 };
                let pos = pos_ids[s][axis];
                let freq = 1.0f32 / theta.powf(2.0 * pair_in_axis as f32 / half as f32);
                let angle = pos as f32 * freq;
                let c = angle.cos();
                let sn = angle.sin();
                let x0 = input[base + p];
                let x1 = input[base + half + p];
                expected[base + p] = x0 * c - x1 * sn;
                expected[base + half + p] = x0 * sn + x1 * c;
            }
        }
    }

    let t_in = upload_fp32_as_bf16(&ctx, &input, vec![seq_len, n_heads, head_dim]).unwrap();
    let pos_bytes: Vec<u8> = pos_ids
        .iter()
        .flat_map(|t| t.iter().flat_map(|&v| v.to_ne_bytes()))
        .collect();
    let pos_buf = crate::buffer::CudaBuffer::alloc(pos_bytes.len(), 0)
        .map_err(Error::Cuda)
        .unwrap();
    pos_buf
        .copy_from_host(&pos_bytes)
        .map_err(Error::Cuda)
        .unwrap();
    let out = apply_vision_2d(&ctx, &t_in, n_heads, head_dim, theta, &pos_buf).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── Vision SDPA (non-causal full attention) ──────────────────────

#[cfg(apxinf_fa2_sm80)]
fn gqa_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0f32; query_tokens * query_heads * head_dim];
    let ratio = query_heads / kv_heads;
    let scale = (head_dim as f32).sqrt().recip();
    for query in 0..query_tokens {
        let causal_limit = key_tokens - query_tokens + query;
        for head in 0..query_heads {
            let kv_head = head / ratio;
            let mut scores = vec![f32::NEG_INFINITY; key_tokens];
            let mut maximum = f32::NEG_INFINITY;
            for key in 0..key_tokens {
                if causal && key > causal_limit {
                    continue;
                }
                let mut score = 0.0f32;
                for dim in 0..head_dim {
                    score += q[(query * query_heads + head) * head_dim + dim]
                        * k[(key * kv_heads + kv_head) * head_dim + dim];
                }
                score *= scale;
                scores[key] = score;
                maximum = maximum.max(score);
            }
            let mut denominator = 0.0f32;
            for score in &mut scores {
                if score.is_finite() {
                    *score = (*score - maximum).exp();
                    denominator += *score;
                } else {
                    *score = 0.0;
                }
            }
            for dim in 0..head_dim {
                let mut value = 0.0f32;
                for key in 0..key_tokens {
                    value += (scores[key] / denominator)
                        * v[(key * kv_heads + kv_head) * head_dim + dim];
                }
                output[(query * query_heads + head) * head_dim + dim] = value;
            }
        }
    }
    output
}

#[cfg(apxinf_fa2_sm80)]
#[test]
fn gqa_bf16_matches_reference_for_dm05_head_geometry() {
    let _guard = crate::tests::gpu_smem_guard();
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (query_tokens, key_tokens, query_heads, kv_heads, head_dim) =
        (10usize, 574usize, 8usize, 4usize, 256usize);
    let q = (0..query_tokens * query_heads * head_dim)
        .map(|index| bf16::from_f32(((index * 17 % 97) as f32 - 48.0) / 256.0).to_f32())
        .collect::<Vec<_>>();
    let k = (0..key_tokens * kv_heads * head_dim)
        .map(|index| bf16::from_f32(((index * 13 % 89) as f32 - 44.0) / 256.0).to_f32())
        .collect::<Vec<_>>();
    let v = (0..key_tokens * kv_heads * head_dim)
        .map(|index| bf16::from_f32(((index * 11 % 83) as f32 - 41.0) / 128.0).to_f32())
        .collect::<Vec<_>>();
    let expected = gqa_reference(
        &q,
        &k,
        &v,
        query_tokens,
        key_tokens,
        query_heads,
        kv_heads,
        head_dim,
        false,
    );
    let q = upload_fp32_as_bf16(&ctx, &q, vec![query_tokens, query_heads, head_dim]).unwrap();
    let k = upload_fp32_as_bf16(&ctx, &k, vec![key_tokens, kv_heads, head_dim]).unwrap();
    let v = upload_fp32_as_bf16(&ctx, &v, vec![key_tokens, kv_heads, head_dim]).unwrap();
    let output = gqa_bf16(&ctx, &q, &k, &v, AttentionMask::None).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&output).unwrap(), &expected);
}

#[cfg(apxinf_fa2_sm80)]
#[test]
fn regular_noncausal_gqa_matches_reference_above_splitkv_query_limit() {
    let _guard = crate::tests::gpu_smem_guard();
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (query_tokens, key_tokens, query_heads, kv_heads, head_dim) =
        (65usize, 67usize, 8usize, 4usize, 256usize);
    let q = (0..query_tokens * query_heads * head_dim)
        .map(|index| bf16::from_f32(((index * 7 % 61) as f32 - 30.0) / 256.0).to_f32())
        .collect::<Vec<_>>();
    let k = (0..key_tokens * kv_heads * head_dim)
        .map(|index| bf16::from_f32(((index * 11 % 67) as f32 - 33.0) / 256.0).to_f32())
        .collect::<Vec<_>>();
    let v = (0..key_tokens * kv_heads * head_dim)
        .map(|index| bf16::from_f32(((index * 13 % 71) as f32 - 35.0) / 128.0).to_f32())
        .collect::<Vec<_>>();
    let expected = gqa_reference(
        &q,
        &k,
        &v,
        query_tokens,
        key_tokens,
        query_heads,
        kv_heads,
        head_dim,
        false,
    );
    let q = upload_fp32_as_bf16(&ctx, &q, vec![query_tokens, query_heads, head_dim]).unwrap();
    let k = upload_fp32_as_bf16(&ctx, &k, vec![key_tokens, kv_heads, head_dim]).unwrap();
    let v = upload_fp32_as_bf16(&ctx, &v, vec![key_tokens, kv_heads, head_dim]).unwrap();
    let output = gqa_bf16(&ctx, &q, &k, &v, AttentionMask::None).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&output).unwrap(), &expected);
}

#[cfg(apxinf_fa2_sm80)]
#[test]
fn causal_gqa_uses_bottom_right_alignment() {
    let _guard = crate::tests::gpu_smem_guard();
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (query_tokens, key_tokens, query_heads, kv_heads, head_dim) =
        (3usize, 7usize, 8usize, 4usize, 256usize);
    let q = (0..query_tokens * query_heads * head_dim)
        .map(|index| bf16::from_f32(((index * 7 % 43) as f32 - 21.0) / 128.0).to_f32())
        .collect::<Vec<_>>();
    let k = (0..key_tokens * kv_heads * head_dim)
        .map(|index| bf16::from_f32(((index * 5 % 37) as f32 - 18.0) / 128.0).to_f32())
        .collect::<Vec<_>>();
    let v = (0..key_tokens * kv_heads * head_dim)
        .map(|index| bf16::from_f32(((index * 3 % 31) as f32 - 15.0) / 64.0).to_f32())
        .collect::<Vec<_>>();
    let expected = gqa_reference(
        &q,
        &k,
        &v,
        query_tokens,
        key_tokens,
        query_heads,
        kv_heads,
        head_dim,
        true,
    );
    let q = upload_fp32_as_bf16(&ctx, &q, vec![query_tokens, query_heads, head_dim]).unwrap();
    let k = upload_fp32_as_bf16(&ctx, &k, vec![key_tokens, kv_heads, head_dim]).unwrap();
    let v = upload_fp32_as_bf16(&ctx, &v, vec![key_tokens, kv_heads, head_dim]).unwrap();
    let output = gqa_bf16(&ctx, &q, &k, &v, AttentionMask::Causal).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&output).unwrap(), &expected);
}

#[test]
fn vision_sdpa_bf16_matches_reference() {
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let (seq, n_heads, head_dim) = (6usize, 2usize, 64usize);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    let q: Vec<f32> = (0..seq * n_heads * head_dim)
        .map(|i| (i as f32 * 0.01 - 0.3).sin())
        .collect();
    let k: Vec<f32> = (0..seq * n_heads * head_dim)
        .map(|i| (i as f32 * 0.013).cos())
        .collect();
    let v: Vec<f32> = (0..seq * n_heads * head_dim)
        .map(|i| (i as f32 * 0.007).tanh())
        .collect();

    // Reference: non-causal, per head.
    let mut expected = vec![0.0f32; seq * n_heads * head_dim];
    for h in 0..n_heads {
        for qi in 0..seq {
            // scores[ki] = (Q[qi,h] · K[ki,h]) * scale
            let mut scores = vec![0.0f32; seq];
            let mut mx = f32::NEG_INFINITY;
            for ki in 0..seq {
                let mut s = 0.0;
                for d in 0..head_dim {
                    s += q[qi * n_heads * head_dim + h * head_dim + d]
                        * k[ki * n_heads * head_dim + h * head_dim + d];
                }
                s *= scale;
                scores[ki] = s;
                if s > mx {
                    mx = s;
                }
            }
            let mut sum = 0.0;
            for ki in 0..seq {
                scores[ki] = (scores[ki] - mx).exp();
                sum += scores[ki];
            }
            for ki in 0..seq {
                scores[ki] /= sum;
            }
            for d in 0..head_dim {
                let mut acc = 0.0;
                for ki in 0..seq {
                    acc += scores[ki] * v[ki * n_heads * head_dim + h * head_dim + d];
                }
                expected[qi * n_heads * head_dim + h * head_dim + d] = acc;
            }
        }
    }

    let t_q = upload_fp32_as_bf16(&ctx, &q, vec![seq, n_heads, head_dim]).unwrap();
    let t_k = upload_fp32_as_bf16(&ctx, &k, vec![seq, n_heads, head_dim]).unwrap();
    let t_v = upload_fp32_as_bf16(&ctx, &v, vec![seq, n_heads, head_dim]).unwrap();
    let out = vision(&ctx, &t_q, &t_k, &t_v, seq, n_heads, head_dim).unwrap();
    assert_bf16_close_reduction(&download_bf16_as_fp32(&out).unwrap(), &expected);
}

// ── concat_2d (fused weight packing) ─────────────────────────────

#[test]
fn concat_2d_bf16_packs_qkv_correctly() {
    // Simulates the fused-QKV weight packing: concat(wq, wk, wv)
    // along the output axis. wq=[hidden,hidden], wk=wv=[hidden,kv_proj].
    use crate::backend::CudaBackend;
    use apxinf_core::Backend;

    let be = CudaBackend::new(0).expect("CUDA device required");
    let hidden = 64;
    let kv_proj = 32;
    let rows = hidden;

    let wq: Vec<f32> = (0..rows * hidden).map(|i| (i as f32) * 0.01).collect();
    let wk: Vec<f32> = (0..rows * kv_proj)
        .map(|i| (i as f32) * 0.02 - 1.0)
        .collect();
    let wv: Vec<f32> = (0..rows * kv_proj)
        .map(|i| (i as f32) * 0.03 + 0.5)
        .collect();

    let t_wq = upload_fp32_as_bf16(be.context(), &wq, vec![rows, hidden]).unwrap();
    let t_wk = upload_fp32_as_bf16(be.context(), &wk, vec![rows, kv_proj]).unwrap();
    let t_wv = upload_fp32_as_bf16(be.context(), &wv, vec![rows, kv_proj]).unwrap();

    let packed = be.concat_2d(&[&t_wq, &t_wk, &t_wv]).expect("concat_2d");
    let out = download_bf16_as_fp32(&packed).unwrap();
    let total_cols = hidden + 2 * kv_proj;
    assert_eq!(packed.shape().dims(), &[rows, total_cols]);

    // Build expected = wq | wk | wv concatenated row-by-row.
    let mut expected = vec![0.0f32; rows * total_cols];
    for r in 0..rows {
        for c in 0..hidden {
            expected[r * total_cols + c] = wq[r * hidden + c];
        }
        for c in 0..kv_proj {
            expected[r * total_cols + hidden + c] = wk[r * kv_proj + c];
        }
        for c in 0..kv_proj {
            expected[r * total_cols + hidden + kv_proj + c] = wv[r * kv_proj + c];
        }
    }
    assert_bf16_close_elementwise(&out, &expected);
}

#[test]
fn concat_2d_bf16_packs_gate_up_correctly() {
    // Simulates the fused Gate/Up weight packing.
    use crate::backend::CudaBackend;
    use apxinf_core::Backend;

    let be = CudaBackend::new(0).expect("CUDA device required");
    let hidden = 64;
    let inter = 128;
    let rows = hidden;

    let w_gate: Vec<f32> = (0..rows * inter).map(|i| (i as f32) * 0.01).collect();
    let w_up: Vec<f32> = (0..rows * inter).map(|i| (i as f32) * 0.02 - 0.5).collect();

    let t_gate = upload_fp32_as_bf16(be.context(), &w_gate, vec![rows, inter]).unwrap();
    let t_up = upload_fp32_as_bf16(be.context(), &w_up, vec![rows, inter]).unwrap();

    let packed = be.concat_2d(&[&t_gate, &t_up]).expect("concat_2d");
    let out = download_bf16_as_fp32(&packed).unwrap();
    let total_cols = 2 * inter;
    assert_eq!(packed.shape().dims(), &[rows, total_cols]);

    let mut expected = vec![0.0f32; rows * total_cols];
    for r in 0..rows {
        for c in 0..inter {
            expected[r * total_cols + c] = w_gate[r * inter + c];
        }
        for c in 0..inter {
            expected[r * total_cols + inter + c] = w_up[r * inter + c];
        }
    }
    assert_bf16_close_elementwise(&out, &expected);
}

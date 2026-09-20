//! Torch-golden precision tests for every public L3 operator.
//!
//! Every new L3 operator must add `<op>_all_candidates_match_torch` here,
//! extend `generate_torch_l3_fixtures.py`, and regenerate
//! `torch_l3_fixtures.rs`. Tests compare final L3 outputs only; they must not
//! depend on or prescribe a candidate's internal implementation.

use super::framework::{
    bf16_bits_tensor, bytes_tensor, f16_tensor, f32_tensor, scales, tensor, zeros_tensor,
};
use super::l3_behavior::{attention_reference, kv_cache_attention_reference};
use super::*;
use crate::CudaContext;
use apxinf_core::{DType, Tensor};
use std::time::Instant;

#[path = "torch_l3_fixtures.rs"]
mod torch_fixture;

fn u32_buffer(device: usize, values: &[u32]) -> crate::CudaBuffer {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer = crate::CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
}

fn validate_all_candidates<'a>(
    ctx: &CudaContext,
    args: GemmArgs<'a>,
    semantic: super::contracts::Semantic,
    bias: Option<&'a Tensor>,
    expected: &'a [f32],
) -> apxinf_core::Result<()> {
    let normalized = super::contracts::normalize(ctx, args, semantic, bias)?;
    super::execution::validate_candidates(ctx, &normalized, expected)
}

fn configure_torch_case(args: &mut GemmArgs<'_>, alpha: f32, output_scale: f32) {
    args.alpha = alpha;
    args.output_scale = output_scale;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
}

#[test]
fn attention_all_candidates_match_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, query_tokens, key_tokens, query_heads, kv_heads, head_dim) = (1, 3, 4, 4, 2, 8);
    let query_values: Vec<_> = (0..batch * query_tokens * query_heads * head_dim)
        .map(|index| ((index * 3 % 17) as f32 - 8.0) / 16.0)
        .collect();
    let key_values: Vec<_> = (0..batch * key_tokens * kv_heads * head_dim)
        .map(|index| ((index * 5 % 19) as f32 - 9.0) / 16.0)
        .collect();
    let value_values: Vec<_> = (0..batch * key_tokens * kv_heads * head_dim)
        .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
        .collect();
    let scale = 1.0 / (head_dim as f32).sqrt();
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
    let mut out = zeros_tensor(
        0,
        vec![batch, query_tokens, query_heads, head_dim],
        DType::BF16,
    );
    let mut args = AttentionArgs::new(&query, &key, &value, &mut out).causal();
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    let normalized = super::attention_contracts::normalize(&ctx, args).unwrap();
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
    super::attention_execution::validate_candidates(&ctx, &normalized, &expected).unwrap();
}

#[test]
fn kv_cache_attention_all_candidates_match_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, query_tokens, valid_key_tokens, key_capacity, heads, head_dim) = (2, 2, 4, 6, 2, 8);
    let query_values: Vec<_> = (0..batch * query_tokens * heads * head_dim)
        .map(|index| ((index * 3 % 17) as f32 - 8.0) / 16.0)
        .collect();
    let key_values: Vec<_> = (0..batch * key_capacity * heads * head_dim)
        .map(|index| ((index * 5 % 19) as f32 - 9.0) / 16.0)
        .collect();
    let value_values: Vec<_> = (0..batch * key_capacity * heads * head_dim)
        .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
        .collect();
    let query = tensor(0, vec![batch, query_tokens, heads, head_dim], &query_values);
    let key = tensor(0, vec![batch, key_capacity, heads, head_dim], &key_values);
    let value = tensor(0, vec![batch, key_capacity, heads, head_dim], &value_values);
    let mut out = zeros_tensor(0, vec![batch, query_tokens, heads, head_dim], DType::BF16);
    let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out);
    args.valid_key_tokens = valid_key_tokens;
    args.query_start = 1;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    let normalized = super::normalize_kv_cache_attention(&ctx, args).unwrap();
    let expected = kv_cache_attention_reference(
        &query_values,
        &key_values,
        &value_values,
        batch,
        query_tokens,
        valid_key_tokens,
        key_capacity,
        heads,
        heads,
        head_dim,
        1,
        1.0 / (head_dim as f32).sqrt(),
        true,
    );
    super::attention_execution::validate_candidates(&ctx, &normalized, &expected).unwrap();
}

#[test]
fn segmented_attention_all_candidates_match_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (tokens, heads, head_dim) = (4, 2, 8);
    let query_values: Vec<_> = (0..tokens * heads * head_dim)
        .map(|index| ((index * 3 % 17) as f32 - 8.0) / 16.0)
        .collect();
    let key_values: Vec<_> = (0..tokens * heads * head_dim)
        .map(|index| ((index * 5 % 19) as f32 - 9.0) / 16.0)
        .collect();
    let value_values: Vec<_> = (0..tokens * heads * head_dim)
        .map(|index| ((index * 7 % 23) as f32 - 11.0) / 16.0)
        .collect();
    let query = tensor(0, vec![tokens, heads, head_dim], &query_values);
    let key = tensor(0, vec![tokens, heads, head_dim], &key_values);
    let value = tensor(0, vec![tokens, heads, head_dim], &value_values);
    let mut out = zeros_tensor(0, vec![tokens, heads, head_dim], DType::BF16);
    let host_offsets = [0, 2, 2, 4];
    let offsets = u32_buffer(0, &host_offsets);
    let mut args =
        SegmentedAttentionArgs::new(&query, &key, &value, &mut out, &offsets, &host_offsets);
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    let normalized = super::normalize_segmented_attention(&ctx, args).unwrap();
    let elements_per_segment = 2 * heads * head_dim;
    let mut expected = attention_reference(
        &query_values[..elements_per_segment],
        &key_values[..elements_per_segment],
        &value_values[..elements_per_segment],
        1,
        2,
        2,
        heads,
        heads,
        head_dim,
        1.0 / (head_dim as f32).sqrt(),
        false,
    );
    expected.extend(attention_reference(
        &query_values[elements_per_segment..],
        &key_values[elements_per_segment..],
        &value_values[elements_per_segment..],
        1,
        2,
        2,
        heads,
        heads,
        head_dim,
        1.0 / (head_dim as f32).sqrt(),
        false,
    ));
    super::attention_execution::validate_candidates(&ctx, &normalized, &expected).unwrap();
}

#[test]
fn attention_sm100_fmha_candidate_matches_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, tokens, heads, head_dim) = (1, 256, 16, 72);
    let elements = batch * tokens * heads * head_dim;
    let query = tensor(
        0,
        vec![batch, tokens, heads, head_dim],
        &vec![0.0; elements],
    );
    let key = tensor(
        0,
        vec![batch, tokens, heads, head_dim],
        &vec![0.0; elements],
    );
    let value = tensor(
        0,
        vec![batch, tokens, heads, head_dim],
        &vec![1.0; elements],
    );
    let mut out = zeros_tensor(0, vec![batch, tokens, heads, head_dim], DType::BF16);
    let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    let normalized = super::attention_contracts::normalize(&ctx, args).unwrap();
    super::attention_execution::validate_candidates(&ctx, &normalized, &vec![1.0; elements])
        .unwrap();
}

#[test]
fn attention_f16_vision_uses_fa2_and_matches_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, tokens, heads, head_dim) = (1, 256, 16, 72);
    let elements = batch * tokens * heads * head_dim;
    let shape = vec![batch, tokens, heads, head_dim];
    let query = f16_tensor(0, shape.clone(), &vec![0.0; elements]);
    let key = f16_tensor(0, shape.clone(), &vec![0.0; elements]);
    let value = f16_tensor(0, shape.clone(), &vec![1.0; elements]);
    let mut out = zeros_tensor(0, shape, DType::F16);
    let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    let normalized = super::attention_contracts::normalize(&ctx, args).unwrap();
    super::attention_execution::validate_candidates(&ctx, &normalized, &vec![1.0; elements])
        .unwrap();
    let execution = super::attention_execution::prepare(&ctx, normalized).unwrap();
    assert!(
        execution.summary().starts_with("flash-attention-2 "),
        "unexpected F16 vision provider: {}",
        execution.summary()
    );
}

#[test]
fn attention_f16_language_mqa_uses_fa2() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, tokens, query_heads, kv_heads, head_dim) = (1, 522, 8, 1, 256);
    let query_shape = vec![batch, tokens, query_heads, head_dim];
    let kv_shape = vec![batch, tokens, kv_heads, head_dim];
    let query = zeros_tensor(0, query_shape.clone(), DType::F16);
    let key = zeros_tensor(0, kv_shape.clone(), DType::F16);
    let value = zeros_tensor(0, kv_shape, DType::F16);
    let mut out = zeros_tensor(0, query_shape, DType::F16);
    let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
    args.policy.allow_fallback = false;
    let normalized = super::attention_contracts::normalize(&ctx, args).unwrap();
    let execution = super::attention_execution::prepare(&ctx, normalized).unwrap();
    assert!(
        execution.summary().starts_with("flash-attention-2 "),
        "unexpected F16 language MQA provider: {}",
        execution.summary()
    );
}

#[test]
fn attention_f16_action_cache_uses_fa2() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, query_tokens, key_tokens, query_heads, kv_heads, head_dim) =
        (1, 10, 532, 8, 1, 256);
    let query_shape = vec![batch, query_tokens, query_heads, head_dim];
    let cache_shape = vec![batch, key_tokens, kv_heads, head_dim];
    let query = zeros_tensor(0, query_shape.clone(), DType::F16);
    let key = zeros_tensor(0, cache_shape.clone(), DType::F16);
    let value = zeros_tensor(0, cache_shape, DType::F16);
    let mut out = zeros_tensor(0, query_shape, DType::F16);
    let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out).non_causal();
    args.policy.allow_fallback = false;
    let normalized = super::normalize_kv_cache_attention(&ctx, args).unwrap();
    let execution = super::attention_execution::prepare(&ctx, normalized).unwrap();
    assert!(
        execution.summary().starts_with("flash-attention-2-kv-cache "),
        "unexpected F16 action KV-cache provider: {}",
        execution.summary()
    );
}

#[test]
#[ignore = "Thor-only performance characterization"]
fn profile_pi05_f16_attention_shapes() {
    fn measure(
        ctx: &CudaContext,
        execution: &super::attention_execution::Execution,
        iterations: usize,
    ) -> f64 {
        for _ in 0..20 {
            execution.enqueue_for_test().unwrap();
        }
        ctx.synchronize().unwrap();
        let start = Instant::now();
        for _ in 0..iterations {
            execution.enqueue_for_test().unwrap();
        }
        ctx.synchronize().unwrap();
        start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64
    }

    let ctx = CudaContext::new(0).unwrap();

    let vision_shape = vec![2, 256, 16, 72];
    let vision_q = zeros_tensor(0, vision_shape.clone(), DType::F16);
    let vision_k = zeros_tensor(0, vision_shape.clone(), DType::F16);
    let vision_v = zeros_tensor(0, vision_shape.clone(), DType::F16);
    let mut vision_out = zeros_tensor(0, vision_shape, DType::F16);
    let vision = super::attention_execution::prepare(
        &ctx,
        super::attention_contracts::normalize(
            &ctx,
            AttentionArgs::new(&vision_q, &vision_k, &vision_v, &mut vision_out),
        )
        .unwrap(),
    )
    .unwrap();

    let language_q_shape = vec![1, 522, 8, 256];
    let language_kv_shape = vec![1, 522, 1, 256];
    let language_q = zeros_tensor(0, language_q_shape.clone(), DType::F16);
    let language_k = zeros_tensor(0, language_kv_shape.clone(), DType::F16);
    let language_v = zeros_tensor(0, language_kv_shape, DType::F16);
    let mut language_out = zeros_tensor(0, language_q_shape, DType::F8E4M3);
    let mut language_args =
        AttentionArgs::new(&language_q, &language_k, &language_v, &mut language_out);
    language_args.output_scale = 1.0 / 16.0;
    let language = super::attention_execution::prepare(
        &ctx,
        super::attention_contracts::normalize(&ctx, language_args).unwrap(),
    )
    .unwrap();

    let action_q_shape = vec![1, 10, 8, 256];
    let action_kv_shape = vec![1, 532, 1, 256];
    let action_q = zeros_tensor(0, action_q_shape.clone(), DType::F16);
    let action_k = zeros_tensor(0, action_kv_shape.clone(), DType::F16);
    let action_v = zeros_tensor(0, action_kv_shape, DType::F16);
    let mut action_out = zeros_tensor(0, action_q_shape, DType::F16);
    let action_args =
        KvCacheAttentionArgs::new(&action_q, &action_k, &action_v, &mut action_out).non_causal();
    let action = super::attention_execution::prepare(
        &ctx,
        super::normalize_kv_cache_attention(&ctx, action_args).unwrap(),
    )
    .unwrap();

    let vision_ms = measure(&ctx, &vision, 200);
    let language_ms = measure(&ctx, &language, 200);
    let action_ms = measure(&ctx, &action, 500);
    let weighted_ms = 27.0 * vision_ms + 17.0 * language_ms + 180.0 * action_ms;
    eprintln!(
        "PI0.5 Attention profile: vision={vision_ms:.6}ms ({}) language={language_ms:.6}ms ({}) action={action_ms:.6}ms ({}) weighted={weighted_ms:.3}ms",
        vision.summary(),
        language.summary(),
        action.summary()
    );
}

#[test]
fn attention_f16_mqa_522_uses_direct_e4m3_fa2() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, tokens, query_heads, kv_heads, head_dim) = (1, 522, 8, 1, 256);
    let query_shape = vec![batch, tokens, query_heads, head_dim];
    let kv_shape = vec![batch, tokens, kv_heads, head_dim];
    let query = f16_tensor(
        0,
        query_shape.clone(),
        &vec![0.0; batch * tokens * query_heads * head_dim],
    );
    let key = f16_tensor(
        0,
        kv_shape.clone(),
        &vec![0.0; batch * tokens * kv_heads * head_dim],
    );
    let value = f16_tensor(
        0,
        kv_shape,
        &vec![1.0; batch * tokens * kv_heads * head_dim],
    );
    let mut out = zeros_tensor(0, query_shape, DType::F8E4M3);
    let output_scale = 1.0 / 16.0;
    let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
    args.output_scale = output_scale;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;
    let normalized = super::attention_contracts::normalize(&ctx, args).unwrap();
    let expected = vec![1.0; batch * tokens * query_heads * head_dim];
    super::attention_execution::validate_candidates(&ctx, &normalized, &expected).unwrap();
    let execution = super::attention_execution::prepare(&ctx, normalized).unwrap();
    assert!(
        execution
            .summary()
            .starts_with("flash-attention-2-f16-e4m3-522 "),
        "unexpected direct-E4M3 provider: {}",
        execution.summary()
    );
    drop(execution);
    let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
    args.output_scale = output_scale;
    args.policy.allow_fallback = false;
    super::attention(&ctx, args).unwrap();

    let output = crate::CudaBuffer::from_tensor(&out).unwrap();
    let mut bytes = vec![0; output.len()];
    output.copy_to_host(&mut bytes).unwrap();
    // E4M3 encodes 1.0 / output_scale = 16.0 as 0x58.
    assert!(bytes.iter().all(|&value| value == 0x58));
}

#[test]
fn torch_validation_requires_the_l3_output_shape() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let args = GemmArgs::new(&a, &b, &mut out);
    let error = validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        &[1.0; 7],
    )
    .unwrap_err();
    assert!(error
        .to_string()
        .contains("Torch validation output does not match"));
}

#[test]
fn gemm_all_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        f::BF16_GEMM,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    configure_torch_case(&mut args, f::FP8_UNIT_ALPHA, f::FP8_UNIT_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        f::FP8_UNIT_GEMM,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::fp8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        f::FP8_SCALED_GEMM,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::I8, f::W8A8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::I8, f::W8A8_B);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::BF16);
    let mut args = GemmArgs::w8a8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::W8A8_ALPHA, f::W8A8_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        None,
        f::W8A8_GEMM,
    )
    .unwrap();
}

#[test]
fn gemm_bias_all_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        Some(&bias),
        f::BF16_GEMM_BIAS,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let bias = f32_tensor(0, vec![f::N], f::FP8_BIAS);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::fp8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        Some(&bias),
        f::FP8_SCALED_GEMM_BIAS,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::I8, f::W8A8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::I8, f::W8A8_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::BF16);
    let mut args = GemmArgs::w8a8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::W8A8_ALPHA, f::W8A8_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        Some(&bias),
        f::W8A8_GEMM_BIAS,
    )
    .unwrap();
}

#[test]
fn gemm_bias_gelu_all_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBiasGelu,
        Some(&bias),
        f::BF16_GEMM_BIAS_GELU,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let bias = f32_tensor(0, vec![f::N], f::FP8_BIAS);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::fp8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmBiasGelu,
        Some(&bias),
        f::FP8_SCALED_GEMM_BIAS_GELU,
    )
    .unwrap();
}

#[test]
fn gemm_geglu_all_candidates_match_torch() {
    use torch_fixture as f;

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, 2 * f::N], f::BF16_GEGLU_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmGeglu,
        None,
        f::BF16_GEMM_GEGLU,
    )
    .unwrap();

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, 2 * f::N], DType::F8E4M3, f::FP8_GEGLU_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    configure_torch_case(&mut args, f::FP8_UNIT_ALPHA, f::FP8_UNIT_OUTPUT_SCALE);
    validate_all_candidates(
        &ctx,
        args,
        super::contracts::Semantic::GemmGeglu,
        None,
        f::FP8_UNIT_GEMM_GEGLU,
    )
    .unwrap();
}

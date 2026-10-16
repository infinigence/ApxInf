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

#[path = "torch_split_geglu_fixtures.rs"]
mod split_fixture;

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
        execution
            .summary()
            .starts_with("flash-attention-2-split-kv ")
            || execution
                .summary()
                .starts_with("flash-attention-2-kv-cache "),
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

#[test]
fn gemm_geglu_three_view_all_candidates_match_torch() {
    use split_fixture as fixture;

    let ctx = CudaContext::new(0).unwrap();
    // T10 (768 patches + 10) and T21 (768 patches + 21) three-view rows.
    for rows in [778usize, 789] {
        let (inner, width) = (2048, 16384);
        let mut activation = vec![0u8; rows * inner];
        let mut weight = vec![0u8; inner * width * 2];
        for row in 0..rows {
            activation[row * inner + row % 8] = 0x38;
        }
        for channel in 0..8 {
            for column in 0..width {
                weight[channel * width * 2 + column] = fixture::SPLIT_GEGLU_VALUES[channel];
                weight[channel * width * 2 + width + column] =
                    fixture::SPLIT_GEGLU_VALUES[column % 8];
            }
        }
        let activation = bytes_tensor(0, vec![rows, inner], DType::F8E4M3, &activation);
        let weight = bytes_tensor(0, vec![inner, width * 2], DType::F8E4M3, &weight);
        let mut output = zeros_tensor(0, vec![rows, width], DType::F8E4M3);
        for (alpha, scale, golden) in [
            (1.0, 1.0, fixture::SPLIT_GEGLU_UNIT),
            (0.75, 1.25, fixture::SPLIT_GEGLU_SCALED),
        ] {
            let expected: Vec<_> = (0..rows * width)
                .map(|index| golden[(index / width % 8) * 8 + index % 8])
                .collect();
            let mut args = GemmArgs::new(&activation, &weight, &mut output)
                .with_immutable_weight(WeightVersion::new(1));
            args.quantization = GemmQuantization::Fp8UnitScale;
            configure_torch_case(&mut args, alpha, scale);
            args.policy.workspace_limit = 1024 * 1024 * 1024;
            validate_all_candidates(
                &ctx,
                args,
                super::contracts::Semantic::GemmGeglu,
                None,
                &expected,
            )
            .unwrap();
        }
    }
}

#[path = "adaptive_norm_torch_fixture.rs"]
mod adaptive_norm_fixture;

#[test]
fn mixed_precision_rms_norm_matches_torch() {
    let ctx = CudaContext::new(0).unwrap();
    for case in adaptive_norm_fixture::CASES {
        let shape = vec![2, case.width];
        let input = bf16_bits_tensor(0, shape.clone(), case.input);
        let residual = bf16_bits_tensor(0, shape.clone(), case.residual);
        let style_values = [case.scale, case.shift].concat();
        let style = f32_tensor(0, vec![2 * case.width], &style_values);
        let gate_values = [vec![0.0; 2 * case.width], case.gate.to_vec()].concat();
        let gate = f32_tensor(0, vec![3 * case.width], &gate_values);
        let weight_values: Vec<_> = case.scale.iter().map(|value| 1.0 + value).collect();
        let weight = f32_tensor(0, vec![case.width], &weight_values);
        let mut output = zeros_tensor(0, shape.clone(), DType::BF16);
        let mut hidden = zeros_tensor(0, shape, DType::BF16);
        let check = |tensor: &Tensor, expected: &[u16]| {
            let buffer = crate::CudaBuffer::from_tensor(tensor).unwrap();
            let mut bytes = vec![0u8; buffer.len()];
            buffer.copy_to_host(&mut bytes).unwrap();
            let actual: Vec<_> = bytes
                .chunks_exact(2)
                .map(|word| u16::from_ne_bytes([word[0], word[1]]))
                .collect();
            assert_eq!(actual, expected, "width={}", case.width);
        };
        rms_norm(&ctx, RmsNormArgs::new(&input, &weight, &mut output, 1e-6)).unwrap();
        ctx.synchronize().unwrap();
        check(&output, case.rms);
        let bias = bf16_bits_tensor(0, vec![case.width], case.bias);
        for (bias, expected_hidden, expected_norm) in [
            (None, case.residual_hidden, case.residual_rms),
            (Some(&bias), case.bias_hidden, case.bias_rms),
        ] {
            bias_residual_rms_norm(
                &ctx,
                BiasResidualRmsNormArgs::new(
                    &input,
                    bias,
                    &residual,
                    &weight,
                    &mut hidden,
                    &mut output,
                    1e-6,
                ),
            )
            .unwrap();
            ctx.synchronize().unwrap();
            check(&hidden, expected_hidden);
            check(&output, expected_norm);
        }
        adaptive_rms_norm(
            &ctx,
            AdaptiveRmsNormArgs::new(&input, &style, &mut output, 1e-6),
        )
        .unwrap();
        ctx.synchronize().unwrap();
        check(&output, case.adaptive);
        let graph = crate::capture(&ctx, || {
            ada_gate_residual_rms_norm(
                &ctx,
                AdaGateResidualRmsNormArgs::new(
                    &input,
                    &residual,
                    &style,
                    &gate,
                    &mut hidden,
                    &mut output,
                    1e-6,
                ),
            )
        })
        .unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        check(&hidden, case.hidden);
        check(&output, case.gated);
    }
}

#[path = "staged_rope_torch_fixture.rs"]
mod staged_rope_fixture;

#[test]
fn staged_bf16_rope_matches_torch() {
    let ctx = CudaContext::new(0).unwrap();
    for case in staged_rope_fixture::CASES {
        let input = bf16_bits_tensor(0, vec![2, 3 * case.dim], case.input);
        let bias =
            (!case.bias.is_empty()).then(|| bf16_bits_tensor(0, vec![3 * case.dim], case.bias));
        let mut q = zeros_tensor(0, vec![2, 1, case.dim], DType::BF16);
        let mut k = zeros_tensor(0, vec![4, 1, case.dim], DType::BF16);
        let mut v = zeros_tensor(0, vec![4, 1, case.dim], DType::BF16);
        let mut launch = || {
            rope(
                &ctx,
                RopeArgs {
                    semantic: RopeSemantic::SplitQkvRopeStagedBf16,
                    qkv: &input,
                    bias: bias.as_ref(),
                    q: &mut q,
                    k: &mut k,
                    v: &mut v,
                    q_heads: 1,
                    kv_heads: 1,
                    head_dim: case.dim,
                    theta: 10_000.0,
                    position_offset: case.position,
                    kv_output_offset: 1,
                },
            )
        };
        launch().unwrap();
        ctx.synchronize().unwrap();
        let graph = crate::capture(&ctx, launch).unwrap();
        let check = |q: &Tensor, k: &Tensor, v: &Tensor, scale: f32| {
            for (tensor, words, padded) in
                [(q, case.q, false), (k, case.k, true), (v, case.v, true)]
            {
                let mut expected = vec![0.0; if padded { case.dim } else { 0 }];
                expected.extend(
                    words
                        .iter()
                        .map(|word| half::bf16::from_bits(*word).to_f32() * scale),
                );
                if padded {
                    expected.extend(vec![0.0; case.dim]);
                }
                assert_eq!(
                    super::framework::values(tensor),
                    expected,
                    "dim={} position={}",
                    case.dim,
                    case.position
                );
            }
        };
        check(&q, &k, &v, 1.0);
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        check(&q, &k, &v, 1.0);
        if bias.is_none() {
            let doubled: Vec<_> = case
                .input
                .iter()
                .flat_map(|word| {
                    half::bf16::from_f32(half::bf16::from_bits(*word).to_f32() * 2.0)
                        .to_bits()
                        .to_ne_bytes()
                })
                .collect();
            crate::CudaBuffer::from_tensor(&input)
                .unwrap()
                .copy_from_host(&doubled)
                .unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            check(&q, &k, &v, 2.0);
        }
    }
}

#[path = "staged_geglu_torch_fixture.rs"]
mod staged_geglu_fixture;

#[test]
fn staged_bf16_geglu_matches_torch() {
    let ctx = CudaContext::new(0).unwrap();
    for (rows, cols) in [
        (2, 7),
        (2, 6),
        (2, 8),
        (50, 8192),
        (264, 16384),
        (712, 16384),
        (968, 16384),
    ] {
        let mut input = Vec::with_capacity(rows * cols * 2);
        let mut expected = Vec::with_capacity(rows * cols);
        for row in 0..rows {
            for pattern in [staged_geglu_fixture::GATE, staged_geglu_fixture::UP] {
                input.extend((0..cols).map(|col| pattern[(row * cols + col) % pattern.len()]));
            }
            expected.extend((0..cols).map(|col| {
                half::bf16::from_bits(staged_geglu_fixture::OUTPUT[(row * cols + col) % 16])
                    .to_f32()
            }));
        }
        let input = bf16_bits_tensor(0, vec![rows, cols * 2], &input);
        let mut output = zeros_tensor(0, vec![rows, cols], DType::BF16);
        let mut launch = || {
            pointwise(
                &ctx,
                PointwiseArgs::new(PointwiseSemantic::GegluStagedBf16, &input, &mut output),
            )
        };
        launch().unwrap();
        ctx.synchronize().unwrap();
        let graph = crate::capture(&ctx, launch).unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(
            super::framework::values(&output),
            expected,
            "rows={rows} cols={cols}"
        );
        crate::CudaBuffer::from_tensor(&input)
            .unwrap()
            .copy_from_host(&vec![0; rows * cols * 4])
            .unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert!(super::framework::values(&output)
            .iter()
            .all(|value| *value == 0.0));
    }
}

#[path = "f32_frontend_torch_fixture.rs"]
mod f32_frontend_fixture;

#[test]
fn f32_pointwise_matches_torch() {
    let ctx = CudaContext::new(0).unwrap();
    let input = f32_tensor(0, vec![2, 7], f32_frontend_fixture::INPUT);
    let bias = f32_tensor(0, vec![7], f32_frontend_fixture::BIAS);
    for (activation, expected) in [
        (PointwiseActivation::None, f32_frontend_fixture::NONE),
        (PointwiseActivation::Gelu, f32_frontend_fixture::GELU),
        (PointwiseActivation::GeluExact, f32_frontend_fixture::EXACT),
        (PointwiseActivation::Silu, f32_frontend_fixture::SILU),
    ] {
        let mut output = zeros_tensor(0, vec![2, 7], DType::F32);
        let graph = crate::capture(&ctx, || {
            let mut args =
                PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut output);
            args.bias = Some(&bias);
            args.activation = activation;
            pointwise(&ctx, args)
        })
        .unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        for (actual, expected) in super::framework::f32_values(&output).iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 2e-6,
                "{activation:?}: {actual} != {expected}"
            );
        }
    }
    let state = f32_tensor(0, vec![1, 2], &[1.0, -1.0]);
    let velocity = f32_tensor(0, vec![1, 2], &[-10.0, 10.0]);
    let mut output = zeros_tensor(0, vec![1, 2], DType::F32);
    let graph = crate::capture(&ctx, || {
        let mut args = PointwiseArgs::new(PointwiseSemantic::EulerUpdate, &state, &mut output);
        args.secondary = Some(&velocity);
        args.dt = 0.1;
        pointwise(&ctx, args)
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(
        super::framework::f32_values(&output),
        [0.0, 0.0],
        "Euler must not fuse multiply and add"
    );
}

#[test]
fn f32_gather_preserves_patch_and_bias_boundaries() {
    let ctx = CudaContext::new(0).unwrap();
    for views in 1..=3 {
        let mut expected = Vec::new();
        let image_nchw: Vec<u8> = (0..views * 3 * 16 * 16).map(|i| (i % 256) as u8).collect();
        for view in 0..views {
            for py in 0..8 {
                for px in 0..8 {
                    for channel in 0..3 {
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let pixel = image_nchw
                                    [(view * 3 + channel) * 256 + (py * 2 + dy) * 16 + px * 2 + dx];
                                expected.push((pixel as f32 / 255.0) * 2.0 - 1.0);
                            }
                        }
                    }
                }
            }
        }
        for nhwc in [false, true] {
            let mut image = image_nchw.clone();
            if nhwc {
                for view in 0..views {
                    for pixel in 0..256 {
                        for channel in 0..3 {
                            image[(view * 256 + pixel) * 3 + channel] =
                                image_nchw[(view * 3 + channel) * 256 + pixel];
                        }
                    }
                }
            }
            let images = crate::CudaBuffer::alloc(image.len(), 0).unwrap();
            images.copy_from_host(&image).unwrap();
            let mut output = zeros_tensor(0, vec![views * 64, 12], DType::F32);
            let graph = crate::capture(&ctx, || {
                gather(
                    &ctx,
                    GatherArgs::rgb_to_patches(
                        &images,
                        &mut output,
                        GatherPatchGeometry {
                            views,
                            image_size: 16,
                            patch_size: 2,
                            nhwc,
                        },
                    ),
                )
            })
            .unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert_eq!(super::framework::f32_values(&output), expected);
            images.copy_from_host(&vec![0; image.len()]).unwrap();
            graph.replay().unwrap();
            ctx.synchronize().unwrap();
            assert!(super::framework::f32_values(&output)
                .iter()
                .all(|value| *value == -1.0));
        }
    }
    let input = f32_tensor(0, vec![2, 2], &[1e8, -1e8, 1e8, -1e8]);
    let owner = std::sync::Arc::downgrade(input.storage().as_gpu().unwrap().owner().unwrap());
    let bias = f32_tensor(0, vec![2], &[-1e8, 1e8]);
    let position = f32_tensor(0, vec![1, 2], &[1.0, -1.0]);
    let mut output = zeros_tensor(0, vec![2, 2], DType::F32);
    let graph = crate::capture(&ctx, || {
        let mut args = GatherArgs::new(GatherSemantic::BiasPosition, &input, &mut output);
        args.bias = Some(&bias);
        args.position = Some(&position);
        args.tokens_per_view = 1;
        gather(&ctx, args)
    })
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(super::framework::f32_values(&output), [1.0, -1.0, 1.0, -1.0]);
    drop(input);
    assert!(owner.upgrade().is_some());
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(super::framework::f32_values(&output), [1.0, -1.0, 1.0, -1.0]);
    drop(graph);
    assert!(owner.upgrade().is_none());
}

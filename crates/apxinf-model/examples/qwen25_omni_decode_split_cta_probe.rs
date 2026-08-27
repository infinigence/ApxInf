use std::time::Instant;

use apxinf_core::{Backend, DType, Error, Tensor};
use apxinf_cuda::kernels::qwen25_omni_attention::{
    grouped2_split_cta_write, grouped4_split_cta_write, split_cta_write,
    SplitCtaWorkspace, HEAD_DIM, KV_HEADS, QUERY_HEADS, WIDTH,
};
use apxinf_cuda::{CudaBackend, CudaBuffer, CudaKVCache};
use half::bf16;

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(std::env::var(name)
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(default))
}

fn deterministic_bf16(count: usize, multiplier: u32, offset: f32) -> Vec<bf16> {
    (0..count)
        .map(|index| {
            let bits = (index as u32).wrapping_mul(multiplier).wrapping_add(1);
            bf16::from_f32((bits & 0xffff) as f32 / 16_384.0 - offset)
        })
        .collect()
}

fn error_metrics(candidate: &[f32], reference: &[f32]) -> serde_json::Value {
    let mut max_abs = 0.0f64;
    let mut mean_abs = 0.0f64;
    let mut dot = 0.0f64;
    let mut candidate_norm = 0.0f64;
    let mut reference_norm = 0.0f64;
    for (&candidate, &reference) in candidate.iter().zip(reference) {
        let candidate = candidate as f64;
        let reference = reference as f64;
        let error = (candidate - reference).abs();
        max_abs = max_abs.max(error);
        mean_abs += error;
        dot += candidate * reference;
        candidate_norm += candidate * candidate;
        reference_norm += reference * reference;
    }
    let count = candidate.len() as f64;
    serde_json::json!({
        "max_abs": max_abs,
        "mean_abs": mean_abs / count,
        "cosine": dot / (candidate_norm.sqrt() * reference_norm.sqrt()),
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let kv_len = env_usize("APXINF_QWEN25_DECODE_PROBE_KV", 32_767)?;
    let max_seq_len = env_usize("APXINF_QWEN25_DECODE_PROBE_MAX_KV", 32_768)?;
    let warmups = env_usize("APXINF_QWEN25_DECODE_PROBE_WARMUPS", 10)?;
    let iterations = env_usize("APXINF_QWEN25_DECODE_PROBE_ITERATIONS", 100)?;
    if kv_len <= 11_264 || kv_len > max_seq_len || warmups == 0 || iterations == 0 {
        return Err("invalid Qwen2.5-Omni split-CTA probe contract".into());
    }

    let backend = CudaBackend::new(0)?;
    let context = backend.context();
    let query_values = deterministic_bf16(QUERY_HEADS * HEAD_DIM, 22_695_477, 2.0);
    let key_values = deterministic_bf16(kv_len * KV_HEADS * HEAD_DIM, 15_485_863, 1.5);
    let value_values = deterministic_bf16(kv_len * KV_HEADS * HEAD_DIM, 11_351_524, 1.25);
    let query = backend.to_device(&Tensor::from_bf16(
        vec![1, QUERY_HEADS, HEAD_DIM],
        &query_values,
    )?)?;
    let key = backend.to_device(&Tensor::from_bf16(
        vec![kv_len, KV_HEADS, HEAD_DIM],
        &key_values,
    )?)?;
    let value = backend.to_device(&Tensor::from_bf16(
        vec![kv_len, KV_HEADS, HEAD_DIM],
        &value_values,
    )?)?;
    let mut cache = CudaKVCache::new(0, 1, KV_HEADS, HEAD_DIM, max_seq_len)?;
    cache.append(context, 0, &key, &value, kv_len)?;
    let baseline = backend.sdpa_decode(
        &query,
        &mut cache,
        0,
        QUERY_HEADS,
        KV_HEADS,
        HEAD_DIM,
        kv_len,
        max_seq_len,
    )?;
    context.synchronize().map_err(Error::Cuda)?;
    let baseline_values = backend.to_cpu(&baseline)?.to_f32_vec()?;

    for _ in 0..warmups {
        let _ = backend.sdpa_decode(
            &query,
            &mut cache,
            0,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
            kv_len,
            max_seq_len,
        )?;
    }
    context.synchronize().map_err(Error::Cuda)?;
    let started = Instant::now();
    for _ in 0..iterations {
        let _ = backend.sdpa_decode(
            &query,
            &mut cache,
            0,
            QUERY_HEADS,
            KV_HEADS,
            HEAD_DIM,
            kv_len,
            max_seq_len,
        )?;
    }
    context.synchronize().map_err(Error::Cuda)?;
    let baseline_ms = started.elapsed().as_secs_f64() * 1_000.0 / iterations as f64;

    let output = backend.to_device(&Tensor::zeros(vec![1, WIDTH], DType::BF16))?;
    let position =
        CudaBuffer::alloc(std::mem::size_of::<u32>(), 0).map_err(Error::Cuda)?;
    position
        .copy_from_host(&u32::try_from(kv_len - 1)?.to_ne_bytes())
        .map_err(Error::Cuda)?;
    let workspace = SplitCtaWorkspace::new(context)?;
    let mut records = Vec::new();
    for split_count in [40usize] {
        for _ in 0..warmups {
            split_cta_write(
                context,
                &query,
                cache.k_buffer(0),
                cache.v_buffer(0),
                &output,
                &workspace,
                split_count,
                kv_len,
                max_seq_len,
                (HEAD_DIM as f32).sqrt().recip(),
                position.address(),
            )?;
        }
        context.synchronize().map_err(Error::Cuda)?;
        let started = Instant::now();
        for _ in 0..iterations {
            split_cta_write(
                context,
                &query,
                cache.k_buffer(0),
                cache.v_buffer(0),
                &output,
                &workspace,
                split_count,
                kv_len,
                max_seq_len,
                (HEAD_DIM as f32).sqrt().recip(),
                position.address(),
            )?;
        }
        context.synchronize().map_err(Error::Cuda)?;
        let milliseconds = started.elapsed().as_secs_f64() * 1_000.0 / iterations as f64;
        let candidate = backend.to_cpu(&output)?.to_f32_vec()?;
        records.push(serde_json::json!({
            "implementation": "split_cta",
            "query_heads_per_cta": 1,
            "split_count": split_count,
            "milliseconds": milliseconds,
            "speedup_over_baseline": baseline_ms / milliseconds,
            "correctness": error_metrics(&candidate, &baseline_values),
        }));
    }
    for split_count in [48usize] {
        for _ in 0..warmups {
            grouped2_split_cta_write(
                context,
                &query,
                cache.k_buffer(0),
                cache.v_buffer(0),
                &output,
                &workspace,
                split_count,
                kv_len,
                max_seq_len,
                (HEAD_DIM as f32).sqrt().recip(),
                position.address(),
            )?;
        }
        context.synchronize().map_err(Error::Cuda)?;
        let started = Instant::now();
        for _ in 0..iterations {
            grouped2_split_cta_write(
                context,
                &query,
                cache.k_buffer(0),
                cache.v_buffer(0),
                &output,
                &workspace,
                split_count,
                kv_len,
                max_seq_len,
                (HEAD_DIM as f32).sqrt().recip(),
                position.address(),
            )?;
        }
        context.synchronize().map_err(Error::Cuda)?;
        let milliseconds = started.elapsed().as_secs_f64() * 1_000.0 / iterations as f64;
        let candidate = backend.to_cpu(&output)?.to_f32_vec()?;
        records.push(serde_json::json!({
            "implementation": "grouped2_split_cta",
            "query_heads_per_cta": 2,
            "split_count": split_count,
            "milliseconds": milliseconds,
            "speedup_over_baseline": baseline_ms / milliseconds,
            "correctness": error_metrics(&candidate, &baseline_values),
        }));
    }
    for split_count in [64usize] {
        for _ in 0..warmups {
            grouped4_split_cta_write(
                context,
                &query,
                cache.k_buffer(0),
                cache.v_buffer(0),
                &output,
                &workspace,
                split_count,
                kv_len,
                max_seq_len,
                (HEAD_DIM as f32).sqrt().recip(),
                position.address(),
            )?;
        }
        context.synchronize().map_err(Error::Cuda)?;
        let started = Instant::now();
        for _ in 0..iterations {
            grouped4_split_cta_write(
                context,
                &query,
                cache.k_buffer(0),
                cache.v_buffer(0),
                &output,
                &workspace,
                split_count,
                kv_len,
                max_seq_len,
                (HEAD_DIM as f32).sqrt().recip(),
                position.address(),
            )?;
        }
        context.synchronize().map_err(Error::Cuda)?;
        let milliseconds = started.elapsed().as_secs_f64() * 1_000.0 / iterations as f64;
        let candidate = backend.to_cpu(&output)?.to_f32_vec()?;
        records.push(serde_json::json!({
            "implementation": "grouped4_split_cta",
            "query_heads_per_cta": 4,
            "split_count": split_count,
            "milliseconds": milliseconds,
            "speedup_over_baseline": baseline_ms / milliseconds,
            "correctness": error_metrics(&candidate, &baseline_values),
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "apxinf.qwen25_omni.decode_split_cta_probe.v1",
            "gpu": context.caps().device_name,
            "sm": context.caps().sm,
            "contract": {
                "kv_len": kv_len,
                "max_seq_len": max_seq_len,
                "query_heads": QUERY_HEADS,
                "kv_heads": KV_HEADS,
                "head_dim": HEAD_DIM,
                "dtype": "bf16",
                "warmups": warmups,
                "iterations": iterations,
            },
            "baseline_ms": baseline_ms,
            "records": records,
        }))?
    );
    Ok(())
}

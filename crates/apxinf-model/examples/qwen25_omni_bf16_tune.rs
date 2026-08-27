use apxinf_core::{Backend, DType, Tensor};
use apxinf_cuda::{
    kernels::gemm::{autotune_cublaslt_bf16, install_cublaslt_bf16_tactics, Bf16CublasLtTactic},
    CudaBackend,
};
use half::bf16;

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(std::env::var(name)
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(default))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m = env_usize("APXINF_QWEN25_OMNI_TUNE_M", 4096)?;
    let max_algorithms = env_usize("APXINF_QWEN25_OMNI_TUNE_MAX_ALGORITHMS", 8)?;
    let warmup_iterations = env_usize("APXINF_QWEN25_OMNI_TUNE_WARMUP", 1)?;
    let benchmark_iterations = env_usize("APXINF_QWEN25_OMNI_TUNE_ITERATIONS", 3)?;
    let kv_len = env_usize("APXINF_QWEN25_OMNI_TUNE_KV_LEN", 0)?;
    if m == 0 || max_algorithms == 0 || max_algorithms > 64 || benchmark_iterations == 0 {
        return Err("invalid Qwen2.5-Omni BF16 tuning budget".into());
    }

    let mut shapes = vec![
        ("q_o", m, 2048usize, 2048usize),
        ("k_v", m, 256usize, 2048usize),
        ("packed_qkv", m, 2560usize, 2048usize),
        ("gate_up", m, 11008usize, 2048usize),
        ("packed_gate_up", m, 22016usize, 2048usize),
        ("down", m, 2048usize, 11008usize),
    ];
    if kv_len > 0 {
        shapes.push(("flat_score", m * 8, kv_len, 128));
        shapes.push(("flat_value", m * 8, 128, kv_len));
    }
    let backend = CudaBackend::new(0)?;
    let mut records = Vec::with_capacity(shapes.len());
    for (name, m, n, k) in shapes {
        eprintln!("tuning {name}: M={m} N={n} K={k}");
        let activation = backend.to_device(&Tensor::zeros(vec![m, k], DType::BF16))?;
        let weight = backend.to_device(&Tensor::zeros(vec![k, n], DType::BF16))?;
        let timing = autotune_cublaslt_bf16(
            backend.context(),
            &activation,
            &weight,
            i32::try_from(max_algorithms)?,
            warmup_iterations,
            benchmark_iterations,
        )?;
        records.push(serde_json::json!({
            "name": name,
            "m": m,
            "n": n,
            "k": k,
            "vendor_ms": timing.vendor_ms,
            "cublaslt_default_ms": timing.cublaslt_default_ms,
            "cublaslt_best_ms": timing.cublaslt_best_ms,
            "cublaslt_best_rank": timing.heuristic_rank,
            "cublaslt_returned_algorithms": timing.returned_algorithms,
            "best_speedup_over_vendor": timing.vendor_ms / timing.cublaslt_best_ms,
        }));
    }

    let activation_values = (0..2048)
        .map(|index| bf16::from_f32(((index * 17 % 31) as f32 - 15.0) / 128.0))
        .collect::<Vec<_>>();
    let weight_values = (0..2048 * 22016)
        .map(|index| bf16::from_f32(((index * 13 % 29) as f32 - 14.0) / 128.0))
        .collect::<Vec<_>>();
    let activation = backend.to_device(&Tensor::from_bf16(vec![1, 2048], &activation_values)?)?;
    let weight = backend.to_device(&Tensor::from_bf16(vec![2048, 22016], &weight_values)?)?;
    let reference = backend
        .to_cpu(&backend.matmul(&activation, &weight)?)?
        .to_f32_vec()?;
    install_cublaslt_bf16_tactics(
        backend.context(),
        &[Bf16CublasLtTactic {
            m: 1,
            n: 22016,
            k: 2048,
            heuristic_rank: 1,
            milliseconds: 0.10788639634847641,
        }],
    )?;
    let candidate = backend
        .to_cpu(&backend.matmul(&activation, &weight)?)?
        .to_f32_vec()?;
    let mut max_abs = 0.0f32;
    let mut square_error = 0.0f64;
    let mut dot = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut candidate_norm = 0.0f64;
    for (&expected, &actual) in reference.iter().zip(&candidate) {
        let error = (expected - actual).abs();
        max_abs = max_abs.max(error);
        square_error += f64::from(error * error);
        dot += f64::from(expected * actual);
        reference_norm += f64::from(expected * expected);
        candidate_norm += f64::from(actual * actual);
    }
    let rmse = (square_error / reference.len() as f64).sqrt();
    let cosine = dot / (reference_norm.sqrt() * candidate_norm.sqrt());

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "apxinf.qwen25_omni.bf16_tuning_probe.v1",
            "gpu": backend.context().caps().device_name.as_str(),
            "sm": backend.context().caps().sm,
            "cuda": backend.context().library_versions().cuda.as_str(),
            "cublas": backend.context().library_versions().cublas.as_str(),
            "contract": {
                "model": "Qwen/Qwen2.5-Omni-3B",
                "prompt_rows": m,
                "cache_state": "cold_l2",
                "max_algorithms": max_algorithms,
                "warmup_iterations": warmup_iterations,
                "benchmark_iterations": benchmark_iterations,
            },
            "records": records,
            "packed_gate_up_validation": {
                "shape": [1, 2048, 22016],
                "heuristic_rank": 1,
                "max_abs": max_abs,
                "rmse": rmse,
                "cosine": cosine,
            },
        }))?
    );
    Ok(())
}

use std::collections::BTreeMap;

use apxinf_core::{Backend, DType, Tensor};
use apxinf_cuda::{
    kernels::gemm::autotune_cublaslt_bf16,
    tuning::{KERNEL_BUILD_ID, TUNING_SCHEMA_V1},
    CudaBackend,
};
use apxinf_model::pi05::{Pi05Config, Pi05ExecutionSchedule};

fn parse_csv(name: &str, default: &[usize]) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let Some(raw) = std::env::var_os(name) else {
        return Ok(default.to_vec());
    };
    raw.to_string_lossy()
        .split(',')
        .map(|value| Ok(value.trim().parse()?))
        .collect()
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn std::error::Error>> {
    Ok(std::env::var(name)
        .ok()
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(default))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let views = parse_csv("APXINF_PI05_TUNE_VIEWS", &[2, 3])?;
    let token_counts = parse_csv("APXINF_PI05_TUNE_TOKENS", &[10, 21, 50, 200])?;
    let max_algorithms = env_usize("APXINF_PI05_TUNE_MAX_ALGORITHMS", 64)?;
    let warmup_iterations = env_usize("APXINF_PI05_TUNE_WARMUP", 10)?;
    let benchmark_iterations = env_usize("APXINF_PI05_TUNE_ITERATIONS", 30)?;
    if views.iter().any(|views| !matches!(views, 2 | 3)) {
        return Err("APXINF_PI05_TUNE_VIEWS only accepts 2 and 3".into());
    }
    if token_counts
        .iter()
        .any(|tokens| *tokens == 0 || *tokens > 200)
    {
        return Err("APXINF_PI05_TUNE_TOKENS must be in 1..=200".into());
    }

    let mut shapes: BTreeMap<(usize, usize, usize), Vec<String>> = BTreeMap::new();
    for &num_views in &views {
        let mut config = Pi05Config::thor_two_view();
        config.num_views = num_views;
        for &token_count in &token_counts {
            let schedule = Pi05ExecutionSchedule::for_token_count(&config, token_count)?;
            for shape in schedule.gemms {
                shapes
                    .entry((shape.m, shape.n, shape.k))
                    .or_default()
                    .push(format!("{}v/t{token_count}/{}", num_views, shape.name));
            }
        }
    }

    let backend = CudaBackend::new(0)?;
    let mut records = Vec::with_capacity(shapes.len());
    for ((m, n, k), profiles) in shapes {
        eprintln!("tuning BF16 M={m} N={n} K={k}: {}", profiles.join(","));
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
        let (backend_name, tactic_id, milliseconds) = if timing.cublaslt_best_ms < timing.vendor_ms
        {
            ("cublaslt", timing.heuristic_rank, timing.cublaslt_best_ms)
        } else {
            ("vendor", 0, timing.vendor_ms)
        };
        eprintln!(
            "  winner={backend_name}:{tactic_id} vendor={:.6}ms lt_default={:.6}ms lt_best={:.6}ms",
            timing.vendor_ms, timing.cublaslt_default_ms, timing.cublaslt_best_ms
        );
        records.push(serde_json::json!({
            "key": {
                "op": "bf16",
                "m": m,
                "n": n,
                "k": k,
                "activation_dtype": "bf16",
                "weight_dtype": "bf16",
                "output_dtype": "bf16",
                "layout": "row_major",
                "scale_mode": "none",
                "epilogue": "none",
                "workspace_limit": usize::MAX,
            },
            "tactic": {
                "backend": backend_name,
                "id": tactic_id,
            },
            "milliseconds": milliseconds,
            "profiles": profiles,
            "candidates": {
                "vendor_ms": timing.vendor_ms,
                "cublaslt_default_ms": timing.cublaslt_default_ms,
                "cublaslt_best_ms": timing.cublaslt_best_ms,
                "cublaslt_best_rank": timing.heuristic_rank,
                "cublaslt_returned_algorithms": timing.returned_algorithms,
            },
        }));
    }

    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": TUNING_SCHEMA_V1,
            "kernel_build_id": KERNEL_BUILD_ID,
            "device_name": backend.context().caps().device_name.as_str(),
            "sm": backend.context().caps().sm,
            "cuda_version": backend.context().library_versions().cuda.as_str(),
            "cublas_version": backend.context().library_versions().cublas.as_str(),
            "tuning_policy": {
                "cache_state": "cold_l2",
                "warmup_iterations": warmup_iterations,
                "benchmark_iterations": benchmark_iterations,
                "max_cublaslt_algorithms": max_algorithms,
                "views": views,
                "token_counts": token_counts,
            },
            "records": records,
        }))?
    );
    Ok(())
}

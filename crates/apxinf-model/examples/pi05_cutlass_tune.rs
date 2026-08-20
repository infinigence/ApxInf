use std::collections::BTreeMap;

use apxinf_core::{Backend, Tensor};
use apxinf_cuda::{
    kernels::gemm::{autotune_cublaslt_fp8, autotune_cutlass_fp8, cold_l2_tuning_metadata},
    tuning::{KERNEL_BUILD_ID, TUNING_SCHEMA_V1},
    CudaBackend,
};
use apxinf_model::pi05::{Pi05Config, Pi05ExecutionSchedule};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let token_counts = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "10,21,50,200".into())
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    let views = std::env::args()
        .nth(2)
        .unwrap_or_else(|| "2,3".into())
        .split(',')
        .map(str::parse::<usize>)
        .collect::<Result<Vec<_>, _>>()?;
    if views.is_empty() || views.iter().any(|views| *views == 0) {
        return Err("views must be a non-empty comma-separated list of positive integers".into());
    }
    if token_counts.is_empty()
        || token_counts
            .iter()
            .any(|tokens| *tokens == 0 || *tokens > 200)
    {
        return Err("token counts must be a non-empty comma-separated list in 1..=200".into());
    }
    let warmup = std::env::args()
        .nth(3)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(5usize);
    let iterations = std::env::args()
        .nth(4)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(25usize);
    if iterations == 0 {
        return Err("benchmark iterations must be non-zero".into());
    }
    let backend = CudaBackend::new(0)?;
    let cold_l2 = cold_l2_tuning_metadata(backend.context())?;
    let mut shapes = BTreeMap::<(usize, usize, usize), Vec<serde_json::Value>>::new();
    let mut prefixes = BTreeMap::new();
    for &num_views in &views {
        let config = match num_views {
            2 => Pi05Config::thor_two_view(),
            3 => Pi05Config::thor_three_view(),
            _ => {
                let mut config = Pi05Config::thor_two_view();
                config.num_views = num_views;
                config
            }
        };
        for &token_count in &token_counts {
            let schedule = Pi05ExecutionSchedule::for_token_count(&config, token_count)?;
            prefixes.insert(
                format!("{num_views}v_t{token_count}"),
                schedule.prefix_tokens,
            );
            for shape in &schedule.gemms {
                shapes
                    .entry((shape.m, shape.n, shape.k))
                    .or_default()
                    .push(serde_json::json!({
                        "views": num_views,
                        "token_count": token_count,
                        "name": shape.name,
                        "stage": format!("{:?}", shape.stage).to_lowercase(),
                        "repetitions": shape.repetitions,
                    }));
            }
        }
    }

    let mut output = serde_json::Map::new();
    for ((m, n, k), workloads) in shapes {
        eprintln!("cold-L2 exact tune M={m} N={n} K={k}");
        let activation = backend.to_device(&Tensor::from_f8_e4m3(vec![m, k], &vec![0; m * k])?)?;
        let weight = backend.to_device(&Tensor::from_f8_e4m3(vec![k, n], &vec![0; k * n])?)?;
        let timings = if n >= 1024 && n % 16 == 0 && k % 16 == 0 {
            autotune_cutlass_fp8(
                backend.context(),
                &activation,
                &weight,
                1.0,
                1.0,
                warmup,
                iterations,
            )?
        } else {
            Vec::new()
        };
        let best_cutlass = timings
            .iter()
            .min_by(|a, b| a.milliseconds.total_cmp(&b.milliseconds))
            .copied();
        let cublaslt_timings = autotune_cublaslt_fp8(
            backend.context(),
            &activation,
            &weight,
            1.0,
            1.0,
            32,
            warmup,
            iterations,
        )?;
        let best_cublaslt = cublaslt_timings
            .iter()
            .min_by(|a, b| a.milliseconds.total_cmp(&b.milliseconds));
        let (backend_name, tactic, milliseconds) = match (best_cutlass, best_cublaslt) {
            (Some(cutlass), Some(cublaslt)) if cublaslt.milliseconds < cutlass.milliseconds => {
                ("cublaslt", cublaslt.heuristic_rank, cublaslt.milliseconds)
            }
            (Some(cutlass), _) => ("cutlass", cutlass.tactic, cutlass.milliseconds),
            (None, Some(cublaslt)) => ("cublaslt", cublaslt.heuristic_rank, cublaslt.milliseconds),
            (None, None) => return Err(format!("no tactic accepted [{m},{n},{k}]").into()),
        };
        output.insert(
            format!("fp8_f16_m{m}_n{n}_k{k}"),
            serde_json::json!({
                "backend": backend_name,
                "tactic": tactic,
                "milliseconds": milliseconds,
                "workloads": workloads,
                "cutlass_candidates": timings.iter().map(|x| serde_json::json!({
                    "tactic": x.tactic,
                    "milliseconds": x.milliseconds,
                })).collect::<Vec<_>>(),
                "cublaslt_candidates": cublaslt_timings.iter().map(|x| serde_json::json!({
                    "heuristic_rank": x.heuristic_rank,
                    "milliseconds": x.milliseconds,
                })).collect::<Vec<_>>(),
            }),
        );
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
            "generator": {
                "name": "pi05_cutlass_tune",
                "crate_version": env!("CARGO_PKG_VERSION"),
                "method": "cold_l2_exact_shape",
            },
            "measurement": {
                "warmup_iterations": warmup,
                "benchmark_iterations": iterations,
                "l2_cache_bytes": cold_l2.l2_cache_bytes,
                "eviction_buffer_bytes": cold_l2.eviction_buffer_bytes,
                "eviction_policy": "read_write_full_4x_l2_before_every_launch",
                "timing": "CUDA events around GEMM only; eviction excluded",
                "selection": "minimum candidate mean; exact physical M/N/K",
            },
            "token_counts": token_counts,
            "views": views,
            "prefix_tokens_by_view": prefixes,
            "tactics": output
        }))?
    );
    Ok(())
}

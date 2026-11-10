//! GR00T N1.7 on HIP against fixture inputs: eager latency and a stage split.
//!
//! Inference goes through the same path `AutoModel` and Python use --
//! `hip_gr00t::load_runner`, a prepared plan, `run` -- so what is measured is
//! what a policy gets.
//!
//! The timing boundary is `doc/gr00t-n1.7.md`'s: it starts at the official
//! processor's host tensors and ends after the action reaches the host. That
//! excludes the simulator and raw-observation preprocessing, so the number is
//! comparable to the published Thor/Orin figures.
//!
//! ```text
//! hip_gr00t_bench <checkpoint> --fixture <dir> [--warmup N] [--iterations N]
//!     [--stages] [--output <json>]
//! ```
//!
//! `--stages` additionally runs vision, backbone, adapter and the flow loop
//! one at a time, synchronizing after each, so the total can be attributed.
//! Those runs are separate from the latency samples: the synchronization
//! needed to time a stage also removes the overlap a real run gets.

#![cfg(feature = "hip")]

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_hip::native::HipArena;
use apxinf_hip::HipContext;
use apxinf_model::hip_gr00t::action_head::{self, DitConditioning};
use apxinf_model::hip_gr00t::{backbone, load_runner, vision};
use apxinf_model::vla::{
    Observation, VisionObservation, VlaMetadata, VlaRequest, VlaRuntime,
};
use apxinf_model::LoadOptions;

type AnyResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

struct Fixture {
    pixel_values: Tensor,
    grid: Vec<[u32; 3]>,
    token_ids: Vec<u32>,
    attention_mask: Vec<u8>,
    state: Tensor,
    noise: Tensor,
    embodiment_id: usize,
    name: String,
}

fn read_bf16(root: &Path, name: &str, dims: Vec<usize>) -> AnyResult<Tensor> {
    let bytes = std::fs::read(root.join(name))?;
    let expected = dims.iter().product::<usize>() * 2;
    if bytes.len() != expected {
        return Err(format!("{name} is {} bytes, expected {expected}", bytes.len()).into());
    }
    let values: Vec<half::bf16> = bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    Ok(Tensor::from_bf16(dims, &values)?)
}

fn load_fixture(root: &Path) -> AnyResult<Fixture> {
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json"))?)?;
    if manifest["schema"].as_str() != Some("apxinf.gr00t-n1.7.preprocessed-fixture.v1") {
        return Err(format!("unexpected fixture schema {:?}", manifest["schema"]).into());
    }
    let grid: Vec<[u32; 3]> = std::fs::read(root.join("image_grid_thw.bin"))?
        .chunks_exact(12)
        .map(|c| {
            let v = |i: usize| u32::from_le_bytes([c[i], c[i + 1], c[i + 2], c[i + 3]]);
            [v(0), v(4), v(8)]
        })
        .collect();
    let patches: usize = grid
        .iter()
        .map(|&[t, h, w]| t as usize * h as usize * w as usize)
        .sum();
    Ok(Fixture {
        pixel_values: read_bf16(root, "pixel_values.bin", vec![patches, 1536])?,
        grid,
        token_ids: std::fs::read(root.join("token_ids.bin"))?
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        attention_mask: std::fs::read(root.join("attention_mask.bin"))?,
        state: read_bf16(root, "state.bin", vec![1, 132])?,
        noise: read_bf16(root, "noise.bin", vec![40, 132])?,
        embodiment_id: manifest["embodiment_id"].as_u64().ok_or("embodiment_id")? as usize,
        name: manifest["fixture"].as_str().unwrap_or("?").to_owned(),
    })
}

fn percentile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let index = ((sorted.len() - 1) as f64 * fraction).round() as usize;
    sorted[index]
}

fn main() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let usage = "usage: hip_gr00t_bench <checkpoint> --fixture <dir> \
                 [--warmup N] [--iterations N] [--stages] [--output <json>]";
    let root = PathBuf::from(args.get(1).filter(|a| !a.starts_with("--")).ok_or(usage)?);
    let fixture_dir = PathBuf::from(flag("--fixture").ok_or(usage)?);
    let warmup: usize = flag("--warmup").map_or(Ok(10), |v| v.parse())?;
    let iterations: usize = flag("--iterations").map_or(Ok(50), |v| v.parse())?;

    let fixture = load_fixture(&fixture_dir)?;
    let ctx = Arc::new(HipContext::new(0)?);
    let load_started = Instant::now();
    let runner = load_runner(&root, Arc::clone(&ctx), &LoadOptions::default())?;
    let load_ms = load_started.elapsed().as_secs_f64() * 1000.0;

    let observation = Observation {
        vision: VisionObservation::Patches(fixture.pixel_values.clone()),
        token_ids: fixture.token_ids.clone(),
        state: Some(fixture.state.clone()),
        action_mask: None,
    };
    let metadata = VlaMetadata {
        attention_mask: Some(&fixture.attention_mask),
        image_grid_thw: Some(&fixture.grid),
        embodiment_id: Some(fixture.embodiment_id),
        planning: None,
    };
    let request = VlaRequest::provided_with_metadata(&observation, &fixture.noise, metadata);

    // Warm-up builds the runner's cached plan, so plan construction stays out
    // of the samples: a policy prepares once and runs many times. A second
    // plan from `prepare_plan` would only double the arena and report a peak
    // of zero, having never run.
    for _ in 0..warmup {
        std::hint::black_box(runner.infer_host_f32(&request)?);
    }

    apxinf_hip::native::reset_dispatch_counts();
    let mut samples = Vec::with_capacity(iterations);
    let mut last = Vec::new();
    for _ in 0..iterations {
        let started = Instant::now();
        last = runner.infer_host_f32(&request)?;
        samples.push(started.elapsed().as_secs_f64() * 1000.0);
        std::hint::black_box(&last);
    }
    let mut sorted = samples.clone();
    sorted.sort_by(f64::total_cmp);
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;

    println!("fixture {} ({} views)", fixture.name, fixture.grid.len());
    println!("load {load_ms:.0} ms, execution {}", runner.execution_mode());
    println!(
        "latency over {iterations} runs: P50 {:.3} ms  P95 {:.3} ms  mean {mean:.3} ms  \
         min {:.3}  max {:.3}",
        percentile(&sorted, 0.50),
        percentile(&sorted, 0.95),
        sorted[0],
        sorted[sorted.len() - 1],
    );
    // Which provider each GEMM actually went to.
    //
    // `native::gemm` falls back to hipBLAS silently whenever hipBLASLt cannot
    // serve a shape, which is right for correctness and wrong for a
    // measurement: an A/B arm that never left the default path looks exactly
    // like one that found no speedup. Both counts are zero on the default
    // provider, which touches no atomics.
    // Whether the forward actually replayed from a graph. Capture being off,
    // or having failed, leaves the eager path in use and the right answer
    // returned, so a latency number alone cannot tell the two apart.
    if let Some(prepared) = runner.prepared_for_report() {
        let (replays, nodes) = prepared.graph_usage();
        if nodes > 0 {
            println!("HIP graph: {nodes} nodes, {replays} of the runs replayed from it");
        } else if matches!(
            std::env::var("APXINF_HIP_GRAPH").unwrap_or_default().trim(),
            "0" | "false" | "off"
        ) {
            println!("HIP graph: disabled by APXINF_HIP_GRAPH; every run was eager");
        } else {
            println!(
                "HIP graph: NOT captured, though capture is on. The forward is not \
                 device-resident; see the apxinf-hip message above."
            );
        }
    }

    let (via_hipblas, via_lt) = apxinf_hip::native::dispatch_counts();
    if via_hipblas + via_lt > 0 {
        println!(
            "GEMM provider: {via_lt} through hipBLASLt, {via_hipblas} through hipBLAS \
             ({:.0}% routed)",
            100.0 * via_lt as f64 / (via_lt + via_hipblas) as f64
        );
    }

    let arena_bound = runner.model().arena_bytes(
        fixture.pixel_values.shape().dims()[0],
        fixture.token_ids.len(),
    );
    println!("arena bound {:.0} MiB", arena_bound as f64 / (1 << 20) as f64);

    // One clean inference with the recorder armed, so the shape census counts
    // a single request rather than warmup + samples.
    apxinf_hip::native::reset_recorded_shapes();
    std::hint::black_box(runner.infer_host_f32(&request)?);
    let shapes = apxinf_hip::native::recorded_shapes();
    if !shapes.is_empty() {
        let mut rows: Vec<_> = shapes
            .iter()
            .map(|&((m, k, n), calls)| {
                // Weight traffic is what a bandwidth-bound stage pays, and it
                // is what orders the tuning work.
                let weight_bytes = (k * n * 2) as u64 * calls;
                let flops = 2u64 * m as u64 * k as u64 * n as u64 * calls;
                ((m, k, n), calls, weight_bytes, flops)
            })
            .collect();
        rows.sort_by_key(|r| std::cmp::Reverse(r.2));
        let total_bytes: u64 = rows.iter().map(|r| r.2).sum();
        let total_calls: u64 = rows.iter().map(|r| r.1).sum();
        println!(
            "\nGEMM shapes in one inference ({} distinct, {total_calls} calls, \
             {:.2} GiB of weights):",
            rows.len(),
            total_bytes as f64 / (1u64 << 30) as f64
        );
        println!("  {:>6} {:>6} {:>6}  {:>6}  {:>9}  {:>6}", "M", "K", "N", "calls", "weight MiB", "share");
        for ((m, k, n), calls, bytes, _) in &rows {
            println!(
                "  {m:>6} {k:>6} {n:>6}  {calls:>6}  {:>9.1}  {:>5.1}%",
                *bytes as f64 / (1u64 << 20) as f64,
                100.0 * *bytes as f64 / total_bytes as f64
            );
        }
    }

    let mut stages = serde_json::Map::new();
    if args.iter().any(|a| a == "--stages") {
        println!("\nstage split (each synchronized, so the sum exceeds the total above):");
        let model = runner.model();
        let config = model.config();
        let backbone_config = model.backbone_config();
        let patches = fixture.pixel_values.shape().dims()[0];
        let (vision_plan, backbone_plan) = model.prepare_shape(
            patches,
            &fixture.grid,
            &fixture.token_ids,
            &fixture.attention_mask,
        )?;
        let embodiment = model.embodiment(fixture.embodiment_id)?;
        let constants = HipArena::new(&ctx, 64 << 20)?;
        let conditioning = DitConditioning::new(&ctx, config, model.weights(), &constants)?;
        let arena = HipArena::new(&ctx, model.arena_bytes(patches, fixture.token_ids.len()))?;
        let pixel_values = ctx.to_device(&fixture.pixel_values)?;
        let state = ctx.to_device(&fixture.state)?;
        let noise = ctx.to_device(&fixture.noise)?;
        let groups = apxinf_model::hip_gr00t::geometry::build_backbone_token_groups(
            &fixture.token_ids,
            &fixture.attention_mask,
            backbone_config.image_token_id,
        )?;

        // One repeat runs the whole chain and synchronizes between stages. Timing
        // a single stage in isolation would need either an arena reset (which
        // discards the input the next stage reads) or no reset at all (which
        // overflows after a few repeats).
        let repeats = 10usize;
        let mut totals = [0.0f64; 4];
        ctx.synchronize()?;
        for _ in 0..repeats {
            arena.reset();
            let mut mark = Instant::now();
            let mut lap = |totals: &mut [f64; 4], index: usize, mark: &mut Instant| -> AnyResult<()> {
                ctx.synchronize()?;
                totals[index] += mark.elapsed().as_secs_f64() * 1000.0;
                *mark = Instant::now();
                Ok(())
            };

            let vision_output = vision::forward(
                &ctx, backbone_config, model.weights(), &arena, &pixel_values, &vision_plan,
            )?;
            lap(&mut totals, 0, &mut mark)?;

            let hidden = backbone::forward(
                &ctx, backbone_config, model.weights(), &arena, &vision_output, &backbone_plan,
            )?;
            lap(&mut totals, 1, &mut mark)?;

            let adapted =
                backbone::forward_adapter(&ctx, config, model.weights(), &arena, &hidden)?;
            lap(&mut totals, 2, &mut mark)?;

            let split = action_head::split_backbone(
                &ctx, &arena, &adapted,
                &action_head::DitRowIndices::new(&ctx, &groups.non_image, &groups.image)?,
            )?;
            action_head::forward(
                &ctx, config, model.weights(), &embodiment, &conditioning, &arena, &split,
                &state, &noise,
            )?;
            lap(&mut totals, 3, &mut mark)?;
        }
        let names = ["vision tower", "text backbone", "VLLN + VL blocks", "DiT (4 flow steps)"];
        let sum: f64 = totals.iter().sum::<f64>() / repeats as f64;
        for (name, total) in names.iter().zip(&totals) {
            let milliseconds = total / repeats as f64;
            println!("  {name:<22} {milliseconds:8.3} ms   {:4.1}%", 100.0 * milliseconds / sum);
            stages.insert((*name).to_owned(), serde_json::json!(milliseconds));
        }
        println!("  {:<22} {sum:8.3} ms", "sum");
    }

    if let Some(path) = flag("--output") {
        let report = serde_json::json!({
            "schema": "apxinf.gr00t-n1.7.hip-benchmark.v1",
            "fixture": fixture.name,
            "views": fixture.grid.len(),
            "checkpoint": root,
            "precision": "bf16",
            "execution": runner.execution_mode(),
            "timing_boundary":
                "official-processor tensors through synchronized model core and action D2H",
            "warmup": warmup,
            "iterations": iterations,
            "load_ms": load_ms,
            "latency_ms": {
                "p50": percentile(&sorted, 0.50),
                "p95": percentile(&sorted, 0.95),
                "mean": mean,
                "min": sorted[0],
                "max": sorted[sorted.len() - 1],
            },
            "stage_ms": stages,
            "gemm_shapes": shapes.iter().map(|&((m, k, n), calls)| {
                serde_json::json!({"m": m, "k": k, "n": n, "calls": calls})
            }).collect::<Vec<_>>(),
            "arena_bound": arena_bound,
            "gemm_dispatches": {"hipblas": via_hipblas, "hipblaslt": via_lt},
            "gemm_provider": std::env::var("APXINF_HIP_GEMM_PROVIDER").unwrap_or_default(),
            "gemm_tactics": std::env::var("APXINF_HIP_GEMM_TACTICS").unwrap_or_default(),
            "output": {"shape": runner.contract().action_shape, "values": last},
        });
        std::fs::write(&path, serde_json::to_string_pretty(&report)? + "\n")?;
        println!("\nwrote {path}");
    }
    Ok(())
}

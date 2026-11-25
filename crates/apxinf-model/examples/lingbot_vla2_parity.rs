//! LingBot-VLA 2.0 through the public entry (`AutoModel` -> `LoadedModel::Vla` -> prepare/run), compared with a
//! reference run of the upstream PyTorch model.
//!
//! The reference inputs and outputs are raw little-endian files in `<golden-bin-dir>`, one set per tag:
//!
//! | file | content |
//! |---|---|
//! | `<tag>_patches.f32` | `[768, 1536]` f32, the three 256x256 views as Qwen3-VL patch rows |
//! | `<tag>_tokens.u32` | prompt token ids (chat-wrapped, unpadded) |
//! | `<tag>_state.f32` | `[55]` f32, normalised state |
//! | `<tag>_noise.f32` | `[50, 55]` f32, the flow-matching noise |
//! | `<tag>_actions_bf16ref.f32` | `[50, 55]` f32, the upstream BF16 result for those inputs |
//! | `<tag>_actions_fakequant.f32` | optional, the same arithmetic as a CPU INT8 simulation |
//!
//! cargo run --release -p apxinf-model --features cuda --example lingbot_vla2_parity -- \
//!     <checkpoint-dir> <golden-bin-dir> <tag> [eager|graph] [autotune]
//!
//! `graph` (the default) requires CUDA Graph capture; on an 8 GB card it also needs `autotune`.

use std::path::{Path, PathBuf};
use std::time::Instant;

use apxinf_core::{DType, Device, Tensor};
use apxinf_model::{
    AutoModel, ExecutionPolicy, LoadOptions, Observation, VisionObservation, VlaRequest,
};

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn read_u32(path: &Path) -> Vec<u32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

struct Stats {
    cosine: f64,
    rel_l2: f64,
    mean_abs: f64,
    max_abs: f64,
}

fn compare(actual: &[f32], reference: &[f32]) -> Stats {
    let (mut dd, mut rr, mut aa, mut ar, mut sum_abs, mut max_abs) =
        (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &r) in actual.iter().zip(reference) {
        let (a, r) = (a as f64, r as f64);
        dd += (a - r).powi(2);
        rr += r * r;
        aa += a * a;
        ar += a * r;
        sum_abs += (a - r).abs();
        max_abs = max_abs.max((a - r).abs());
    }
    Stats {
        cosine: ar / (aa.sqrt() * rr.sqrt()).max(1e-30),
        rel_l2: (dd / rr.max(1e-30)).sqrt(),
        mean_abs: sum_abs / actual.len() as f64,
        max_abs,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        return Err(format!(
            "usage: {} <checkpoint-dir> <golden-bin-dir> <tag> [eager|graph] [autotune]",
            args[0]
        )
        .into());
    }
    let checkpoint = PathBuf::from(&args[1]);
    let golden = PathBuf::from(&args[2]);
    let tag = &args[3];
    let policy = match args.get(4).map(String::as_str) {
        Some("eager") => ExecutionPolicy::Eager,
        _ => ExecutionPolicy::RequireGraph,
    };

    let patches = read_f32(&golden.join(format!("{tag}_patches.f32")));
    let tokens = read_u32(&golden.join(format!("{tag}_tokens.u32")));
    let state = read_f32(&golden.join(format!("{tag}_state.f32")));
    let noise = read_f32(&golden.join(format!("{tag}_noise.f32")));
    let reference = read_f32(&golden.join(format!("{tag}_actions_bf16ref.f32")));
    let fake_quant = golden
        .join(format!("{tag}_actions_fakequant.f32"))
        .is_file()
        .then(|| read_f32(&golden.join(format!("{tag}_actions_fakequant.f32"))));

    // `--autotune` (5th argument) lets the GEMM registry pick CUTLASS W8A8 instead of the default cuBLAS fallback.
    let autotune = args.get(5).map(String::as_str) == Some("autotune");
    let options = LoadOptions {
        model_name: Some("lingbot_vla2".into()),
        autotune,
        ..LoadOptions::default()
    };
    let t0 = Instant::now();
    let model = AutoModel::load_model(Device::Cuda(0), &checkpoint, &options)?;
    println!("load: {:.1}s", t0.elapsed().as_secs_f64());

    let observation = Observation {
        vision: VisionObservation::Patches(Tensor::from_f32(vec![768, 1536], &patches)?),
        token_ids: tokens.clone(),
        state: Some(Tensor::from_f32(vec![1, 55], &state)?),
        action_mask: None,
    };
    let noise_tensor = Tensor::from_f32(vec![50, 55], &noise)?;
    let grids = [[1u32, 16, 16]; 3];
    let mut request = VlaRequest::provided(&observation, &noise_tensor);
    request.metadata.image_grid_thw = Some(&grids);

    let t0 = Instant::now();
    let prepared = model.prepare_for(&request, policy)?;
    println!(
        "prepare ({policy:?}): {:.1}s status={:?}",
        t0.elapsed().as_secs_f64(),
        prepared.status()
    );

    let first = prepared.run(&request)?;
    let host = apxinf_cuda_new::transfers::to_cpu(first.tensor())?.to_f32_vec()?;
    // [50, 64] padded -> [50, 55]
    let mut actions = Vec::with_capacity(50 * 55);
    for row in 0..50 {
        actions.extend_from_slice(&host[row * 64..row * 64 + 55]);
    }
    assert!(actions.iter().all(|v| v.is_finite()), "non-finite action");
    assert!(
        host.chunks_exact(64)
            .all(|row| row[55..].iter().all(|v| *v == 0.0 || v.abs() < 1e-2)),
        "padding columns are not ~0"
    );

    let s = compare(&actions, &reference);
    println!(
        "vs BF16 reference : cos {:.5}  relL2 {:.4}  mean|d| {:.4}  max|d| {:.3}",
        s.cosine, s.rel_l2, s.mean_abs, s.max_abs
    );
    if let Some(fq) = &fake_quant {
        let s = compare(&actions, fq);
        println!(
            "vs INT8 simulation: cos {:.5}  relL2 {:.4}  mean|d| {:.4}  max|d| {:.3}",
            s.cosine, s.rel_l2, s.mean_abs, s.max_abs
        );
    }

    // steady-state latency of the prepared plan, host->device inputs included, result synchronised
    let mut times = Vec::new();
    for _ in 0..12 {
        let t = Instant::now();
        let out = prepared.run(&request)?;
        drop(out);
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!(
        "run latency ms: min {:.1}  median {:.1}  max {:.1}",
        times[0],
        times[times.len() / 2],
        times[times.len() - 1]
    );
    let _ = DType::F32;
    Ok(())
}

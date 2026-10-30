//! Dump π0.5's intermediate tensors from the CUDA BF16 implementation, as the
//! reference a new backend is compared against stage by stage.
//!
//! Runs one eager inference through the public stage methods of `Bf16Model`
//! on fixture inputs and writes each stage as an F32 `.npy` file:
//!
//! | file | shape |
//! |---|---|
//! | `patches` | `[views * 256, 588]` |
//! | `vision` | `[views * 256, 2048]`, after the multimodal projector |
//! | `prefix` | `[prefix, 2048]`, vision tokens then scaled token embeddings |
//! | `k_LL`, `v_LL` | `[prefix, 256]`, language layer `LL`'s RoPE'd K and V |
//! | `state_step00` | `[H, 32]`, after the first Euler step |
//! | `actions` | `[H, 32]`, after the last |
//!
//! ```text
//! pi05_stage_dump <checkpoint> --out <dir> [--views N] [--action-horizon N]
//!     [--token-count T] [--images-u8 <raw>] [--token-ids-u32le <raw>] [--noise-bf16-u16le <raw>]
//! ```
//!
//! `pi05_hip_bench --dump-stages` writes the same files from HIP;
//! `scripts/compare_pi05_stages.py` compares two dump directories.

#[path = "support/pi05_fixtures.rs"]
mod fixtures;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use apxinf_core::Tensor;
use apxinf_cuda_new::{transfers, CudaBuffer, CudaContext};
use apxinf_model::pi05::{
    build_bf16_model, upload_time_embeddings_bf16, Bf16Weights, Pi05Config, Pi05Weights,
};
use fixtures::{write_npy, AnyResult, Fixture, Geometry};

fn host(tensor: &Tensor) -> AnyResult<Vec<f32>> {
    Ok(transfers::to_cpu(tensor)?.to_f32_vec()?)
}

fn main() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let usage = "usage: pi05_stage_dump <checkpoint> --out <dir> [--views N] [--action-horizon N] \
                 [--token-count T] [--images-u8 <raw>] [--token-ids-u32le <raw>] [--noise-bf16-u16le <raw>]";
    let checkpoint = args.get(1).filter(|a| !a.starts_with("--")).ok_or(usage)?;
    let out = PathBuf::from(flag("--out").ok_or(usage)?);
    let token_count: usize = flag("--token-count").map_or(Ok(10), |v| v.parse())?;

    let source = Path::new(checkpoint);
    let root = if source.is_dir() {
        source
    } else {
        source.parent().unwrap_or(Path::new("."))
    };
    let mut config = match root.join("config.json") {
        path if path.is_file() => Pi05Config::from_json_file(&path)?,
        _ => Pi05Config::default(),
    };
    if let Some(views) = flag("--views") {
        config.num_views = views.parse()?;
    }
    if let Some(horizon) = flag("--action-horizon") {
        config.action_horizon = horizon.parse()?;
    }
    config.validate()?;
    let config = Arc::new(config);

    // `build_bf16_model` takes the context by `Arc`, as `pi05_bench` builds it.
    #[allow(clippy::arc_with_non_send_sync)]
    let context = Arc::new(CudaContext::new(0).map_err(std::io::Error::other)?);
    eprintln!("loading π0.5 checkpoint...");
    let host_weights = Pi05Weights::from_safetensors(&config, source)?;
    let weights = Arc::new(Bf16Weights::from_host(&host_weights, &context)?);
    let model = build_bf16_model(Arc::clone(&context), Arc::clone(&config), weights)?;

    let geometry = Geometry {
        views: config.num_views,
        image_size: config.image_size,
        patch_size: config.patch_size,
        action_horizon: config.action_horizon,
        action_dim: config.action_dim,
    };
    let fixture = Fixture::load(
        geometry,
        token_count,
        flag("--images-u8").as_deref(),
        flag("--token-ids-u32le").as_deref(),
        flag("--noise-bf16-u16le").as_deref(),
    )?;
    let patch_shape = vec![
        config.num_views * config.patches_per_view(),
        3 * config.patch_size.pow(2),
    ];
    let patches = transfers::to_cuda(
        &Tensor::from_bf16(patch_shape.clone(), &fixture.patches(geometry))?,
        0,
    )?;
    let noise_shape = vec![config.action_horizon, config.action_dim];
    let noise: Vec<half::bf16> = fixture
        .noise
        .iter()
        .map(|&v| half::bf16::from_f32(v))
        .collect();
    let noise = transfers::to_cuda(&Tensor::from_bf16(noise_shape.clone(), &noise)?, 0)?;
    let id_bytes: Vec<u8> = fixture
        .token_ids
        .iter()
        .flat_map(|id| id.to_le_bytes())
        .collect();
    let token_ids = CudaBuffer::alloc_zeros(id_bytes.len(), context.device_id())
        .map_err(std::io::Error::other)?;
    token_ids
        .copy_from_host(&id_bytes)
        .map_err(std::io::Error::other)?;
    let time_embeddings = upload_time_embeddings_bf16(&config, &context)?;

    let vision = model.encode_vision(&patches)?;
    let prefix = model.embed_prefix(&vision, &token_ids, token_count)?;
    let cache = model.prefix_forward(&prefix)?;
    let dt = -config.flow_start_time / config.num_flow_steps as f32;
    let mut state = noise;
    for (step, embedding) in time_embeddings.iter().enumerate() {
        state = model.denoise_step(&state, embedding, &cache, dt)?;
        if step == 0 {
            write_npy(&out, "state_step00", &noise_shape, &host(&state)?)?;
        }
    }

    write_npy(&out, "patches", &patch_shape, &host(&patches)?)?;
    write_npy(&out, "vision", vision.shape().dims(), &host(&vision)?)?;
    write_npy(&out, "prefix", prefix.shape().dims(), &host(&prefix)?)?;
    let kv_width = config.language.num_kv_heads * config.language.head_dim;
    for (layer, (key, value)) in cache.keys.iter().zip(&cache.values).enumerate() {
        let rows = cache.tokens * kv_width;
        write_npy(
            &out,
            &format!("k_{layer:02}"),
            &[cache.tokens, kv_width],
            &host(key)?[..rows],
        )?;
        write_npy(
            &out,
            &format!("v_{layer:02}"),
            &[cache.tokens, kv_width],
            &host(value)?[..rows],
        )?;
    }
    write_npy(&out, "actions", &noise_shape, &host(&state)?)?;
    eprintln!("wrote CUDA BF16 stages to {}", out.display());
    Ok(())
}

//! Dump the HIP port's GR00T backbone stages as F32 `.npy` files, for
//! stage-by-stage comparison against a reference.
//!
//! Runs one eager inference through the public stage methods of
//! [`ModelVariant`] on fixture inputs and writes each stage as an F32 `.npy`:
//!
//! | file | shape | stage |
//! |---|---|---|
//! | `pos_embeds` | `[patches, hidden]` | interpolated position table |
//! | `vision_primary` | `[patches/merge², 2048]` | vision tower, primary merger |
//! | `vision_deepstack_{i}` | `[patches/merge², 2048]` | vision tower, deepstack |
//! | `backbone_features_raw` | `[tokens, 2048]` | Qwen3-VL text stack |
//! | `backbone_features_post_adapter` | `[tokens, 2048]` | VLLN + VL blocks |
//! | `state_features` | `[1, 1536]` | state encoder output |
//! | `actions` | `[horizon, action_dim]` | final action chunk |
//!
//! ```text
//! hip_gr00t_stage_dump <checkpoint> --fixture <dir> --out <dir> [--embodiment N]
//! ```
//!
//! `<fixture>` is an `apxinf.gr00t-n1.7.preprocessed-fixture.v1` directory.
//! `devlocal/gr00t-hip-port/scripts/compare_stages.py` pairs the dump with a
//! reference `.npz`.

#![cfg(feature = "hip")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use apxinf_core::Tensor;
use apxinf_hip::native::HipArena;
use apxinf_hip::HipContext;
use apxinf_model::hip_gr00t::config::Gr00tConfig;
use apxinf_model::hip_gr00t::load_runner;
use apxinf_model::hip_gr00t::model::token_groups::build_backbone_token_groups;
use apxinf_model::hip_gr00t::weights::host::Gr00tWeights;
use apxinf_model::LoadOptions;

type AnyResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

fn write_npy(dir: &Path, name: &str, shape: &[usize], values: &[f32]) -> AnyResult<()> {
    assert_eq!(
        shape.iter().product::<usize>(),
        values.len(),
        "{name}: shape and data disagree"
    );
    let dims = shape.iter().map(|d| format!("{d},")).collect::<String>();
    let mut header = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}), }}");
    let padding = (64 - (10 + header.len() + 1) % 64) % 64;
    header.push_str(&" ".repeat(padding));
    header.push('\n');
    let mut bytes = b"\x93NUMPY\x01\x00".to_vec();
    bytes.extend((header.len() as u16).to_le_bytes());
    bytes.extend(header.as_bytes());
    bytes.extend(values.iter().flat_map(|v| v.to_le_bytes()));
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join(format!("{name}.npy")), bytes)?;
    Ok(())
}

fn dump(ctx: &HipContext, dir: &Path, name: &str, tensor: &Tensor) -> AnyResult<()> {
    let host = ctx.to_cpu(tensor)?.to_f32_vec()?;
    write_npy(dir, name, tensor.shape().dims(), &host)
        .map_err(|e| apxinf_core::Error::Other(format!("{name}: {e}")))?;
    Ok(())
}

fn main() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let usage = "usage: hip_gr00t_stage_dump <checkpoint> --fixture <dir> --out <dir> \
                 [--embodiment N]";
    let root = PathBuf::from(args.get(1).filter(|a| !a.starts_with("--")).ok_or(usage)?);
    let fixture_dir = PathBuf::from(flag("--fixture").ok_or(usage)?);
    let out = PathBuf::from(flag("--out").ok_or(usage)?);

    let config = Gr00tConfig::from_json_file(&root.join("config.json"))?;
    // Loaded for validation only: the runner below performs the real load.
    let (backbone_config, _weights) =
        Gr00tWeights::from_safetensors(&config, &root.join("assets/cosmos"), &root)?;

    // Grid first: it fixes how many patch rows `pixel_values` must hold.
    let grid_bytes = std::fs::read(fixture_dir.join("image_grid_thw.bin"))?;
    if grid_bytes.len() % 12 != 0 || grid_bytes.is_empty() {
        return Err(format!(
            "image_grid_thw.bin is {} bytes, not 12 per grid",
            grid_bytes.len()
        )
        .into());
    }
    let grid: Vec<[u32; 3]> = grid_bytes
        .chunks_exact(12)
        .map(|c| {
            let v = |i: usize| u32::from_le_bytes([c[i], c[i + 1], c[i + 2], c[i + 3]]);
            [v(0), v(4), v(8)]
        })
        .collect();

    let patch_width = backbone_config.vision.in_channels
        * backbone_config.vision.temporal_patch_size
        * backbone_config.vision.patch_size
        * backbone_config.vision.patch_size;
    let pixel_bytes = std::fs::read(fixture_dir.join("pixel_values.bin"))?;
    let row_bytes = patch_width * 2;
    if pixel_bytes.len() % row_bytes != 0 {
        return Err(format!(
            "pixel_values.bin is {} bytes, not a whole number of {patch_width}-wide BF16 rows",
            pixel_bytes.len()
        )
        .into());
    }
    let patches = pixel_bytes.len() / row_bytes;
    let token_bytes = std::fs::read(fixture_dir.join("token_ids.bin"))?;
    if token_bytes.len() % 4 != 0 {
        return Err(format!("token_ids.bin is {} bytes, not u32", token_bytes.len()).into());
    }
    let token_ids: Vec<u32> = token_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let attention_mask = std::fs::read(fixture_dir.join("attention_mask.bin"))?;
    let read_bf16 = |name: &str, dims: Vec<usize>| -> AnyResult<Tensor> {
        let bytes = std::fs::read(fixture_dir.join(name))?;
        let expected = dims.iter().product::<usize>() * 2;
        if bytes.len() != expected {
            return Err(format!("{name} is {} bytes, expected {expected}", bytes.len()).into());
        }
        let values: Vec<half::bf16> = bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
            .collect();
        Ok(Tensor::from_bf16(dims, &values)?)
    };
    let state_host = read_bf16("state.bin", vec![1, config.state_input_dim()?])?;
    let noise_host = read_bf16("noise.bin", vec![config.action_horizon, config.max_action_dim])?;
    // The fixture names the embodiment it was captured for; using any other
    // one would silently select the wrong adapters.
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_dir.join("manifest.json"))?)?;
    let embodiment_id = flag("--embodiment")
        .map(|value| value.parse::<usize>())
        .transpose()?
        .or_else(|| manifest["embodiment_id"].as_u64().map(|v| v as usize))
        .ok_or("fixture manifest has no embodiment_id; pass --embodiment")?;
    let pixel_host: Vec<half::bf16> = pixel_bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])))
        .collect();
    let pixel_host = Tensor::from_bf16(vec![patches, patch_width], &pixel_host)?;

    let ctx = Arc::new(HipContext::new(0)?);
    let runner = load_runner(&root, Arc::clone(&ctx), &LoadOptions::default())?;
    let model = runner.model();

    // The runner validates the same request `infer` would; reusing its
    // preparation keeps the dump on the public path.
    let pixel_values = ctx.to_device(&pixel_host)?;
    let state = ctx.to_device(&state_host)?;
    let noise = ctx.to_device(&noise_host)?;

    let blocks = model.prepare_shape(patches, &grid, &token_ids, &attention_mask)?;
    let embodiment = model.embodiment(embodiment_id)?;
    let arena = HipArena::new(
        &ctx,
        model
            .workspace_requirements(patches, token_ids.len())?
            .arena_bytes,
    )?;

    // Host-side preparation, so a position-table mismatch is distinguishable
    // from a block mismatch.
    let groups = build_backbone_token_groups(
        &token_ids,
        &attention_mask,
        backbone_config.image_token_id,
    )?;
    println!(
        "grid {grid:?} -> {patches} patches, {} non-image / {} image rows, {} tokens",
        groups.non_image.len(),
        groups.image.len(),
        token_ids.len(),
    );

    let mut written = Vec::new();
    let mut stage = |name: &str, tensor: &Tensor| -> AnyResult<()> {
        dump(&ctx, &out, name, tensor)?;
        written.push(name.to_string());
        Ok(())
    };

    let vision_output = model.vision_stage(&arena, &pixel_values, &blocks)?;
    stage("vision_primary", &vision_output.primary)?;
    for (index, deepstack) in vision_output.deepstack.iter().enumerate() {
        stage(&format!("vision_deepstack_{index}"), deepstack)?;
    }

    let hidden = model.backbone_stage(&arena, &vision_output, &blocks)?;
    // The reference's own names, so the comparison needs no alias table.
    stage("backbone_features_raw", &hidden)?;
    let adapted = model.adapter_stage(&arena, &hidden)?;
    stage("backbone_features_post_adapter", &adapted)?;

    let actions = model.action_stage(&arena, &adapted, &state, &noise, &blocks, &embodiment)?;
    stage("actions", &actions)?;

    println!(
        "wrote {} stages to {} (arena peak {:.0} MiB)",
        written.len(),
        out.display(),
        arena.peak() as f64 / (1 << 20) as f64,
    );
    Ok(())
}

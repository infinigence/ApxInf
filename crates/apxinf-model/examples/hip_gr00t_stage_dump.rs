//! Dump the HIP port's GR00T backbone stages as F32 `.npy` files, for
//! stage-by-stage comparison against a reference.
//!
//! ```text
//! hip_gr00t_stage_dump <checkpoint> --fixture <dir> --out <dir>
//! ```
//!
//! `<fixture>` is a `apxinf.gr00t-n1.7.preprocessed-fixture.v1` directory.
//! Stage names match the two references this port compares against:
//! `backbone_features_raw` / `backbone_features_post_adapter` are the NVIDIA
//! GPU reference's own names, and the vision-tower stages match
//! `devlocal/gr00t-hip-port/fixtures/_work/step3_vision_stage_reference.py`.
//! `devlocal/gr00t-hip-port/scripts/compare_vision_stages.py` pairs them up.

#![cfg(feature = "hip")]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use apxinf_core::{Result, Tensor};
use apxinf_hip::native::HipArena;
use apxinf_hip::HipContext;
use apxinf_model::hip_gr00t::model::blocks::action_head::{self, DitConditioning};
use apxinf_model::hip_gr00t::model::blocks::backbone;
use apxinf_model::hip_gr00t::weights::Gr00tWeights;
use apxinf_model::hip_gr00t::config::Gr00tConfig;
use apxinf_model::hip_gr00t::model::stages::StageSink;
use apxinf_model::hip_gr00t::model::blocks::vision;
use apxinf_model::hip_gr00t::weights::HipGr00tWeights;

type AnyResult<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Writes each stage straight to disk, so a 131-tensor dump never sits in RAM
/// twice.
struct NpyStages<'a> {
    ctx: &'a HipContext,
    dir: PathBuf,
    written: Vec<String>,
}

impl StageSink for NpyStages<'_> {
    fn stage(&mut self, name: &str, tensor: &Tensor) -> Result<()> {
        let host = self.ctx.to_cpu(tensor)?.to_f32_vec()?;
        write_npy(&self.dir, name, tensor.shape().dims(), &host)
            .map_err(|e| apxinf_core::Error::Other(format!("{name}: {e}")))?;
        self.written.push(name.to_string());
        Ok(())
    }
}

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

fn main() -> AnyResult<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let usage = "usage: hip_gr00t_vision_dump <checkpoint> --fixture <dir> --out <dir>";
    let root = PathBuf::from(args.get(1).filter(|a| !a.starts_with("--")).ok_or(usage)?);
    let fixture = PathBuf::from(flag("--fixture").ok_or(usage)?);
    let out = PathBuf::from(flag("--out").ok_or(usage)?);

    let config = Gr00tConfig::from_json_file(&root.join("config.json"))?;
    let (backbone_config, weights) =
        Gr00tWeights::from_safetensors(&config, &root.join("assets/cosmos"), &root)?;

    // Grid first: it fixes how many patch rows `pixel_values` must hold.
    let grid_bytes = std::fs::read(fixture.join("image_grid_thw.bin"))?;
    if grid_bytes.len() % 12 != 0 || grid_bytes.is_empty() {
        return Err(format!("image_grid_thw.bin is {} bytes, not 12 per grid", grid_bytes.len()).into());
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
    let pixel_bytes = std::fs::read(fixture.join("pixel_values.bin"))?;
    let row_bytes = patch_width * 2;
    if pixel_bytes.len() % row_bytes != 0 {
        return Err(format!(
            "pixel_values.bin is {} bytes, not a whole number of {patch_width}-wide BF16 rows",
            pixel_bytes.len()
        )
        .into());
    }
    let patches = pixel_bytes.len() / row_bytes;
    let token_bytes = std::fs::read(fixture.join("token_ids.bin"))?;
    if token_bytes.len() % 4 != 0 {
        return Err(format!("token_ids.bin is {} bytes, not u32", token_bytes.len()).into());
    }
    let token_ids: Vec<u32> = token_bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let attention_mask = std::fs::read(fixture.join("attention_mask.bin"))?;
    let read_bf16 = |name: &str, dims: Vec<usize>| -> AnyResult<Tensor> {
        let bytes = std::fs::read(fixture.join(name))?;
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
        serde_json::from_slice(&std::fs::read(fixture.join("manifest.json"))?)?;
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
    let (device, embodiment) = HipGr00tWeights::from_checkpoint(
        &weights,
        &config,
        &backbone_config,
        embodiment_id,
        &ctx,
    )?;
    let pixel_values = ctx.to_device(&pixel_host)?;

    let prepared = vision::prepare(
        &ctx,
        &backbone_config,
        &weights.backbone_vision.pos_embed,
        &grid,
        patches,
    )?;
    println!(
        "grid {grid:?} -> {patches} patches in segments {:?}",
        prepared.segments()
    );

    // Generous: 24 blocks x a handful of [512, 4096] activations, plus the
    // mergers. The arena never frees, so this is the whole tower's footprint.
    let arena = HipArena::new(&ctx, 1536 << 20)?;
    let mut sink = NpyStages {
        ctx: &ctx,
        dir: out.clone(),
        written: Vec::new(),
    };
    // Host-side preparation, so a position-table mismatch is distinguishable
    // from a block mismatch.
    sink.stage("pos_embeds", prepared.position_embeddings())?;

    let vision_output = vision::forward_observed(
        &ctx,
        &backbone_config,
        &device,
        &arena,
        &pixel_values,
        &prepared,
        &mut sink,
    )?;
    println!(
        "vision: primary {:?}, {} deepstack",
        vision_output.primary.shape().dims(),
        vision_output.deepstack.len(),
    );

    let text = backbone::prepare(
        &ctx,
        &backbone_config,
        &token_ids,
        &attention_mask,
        &grid,
    )?;
    let hidden = backbone::forward_observed(
        &ctx,
        &backbone_config,
        &device,
        &arena,
        &vision_output,
        &text,
        &mut sink,
    )?;
    // The reference's own names, so the comparison needs no alias table.
    sink.stage("backbone_features_raw", &hidden)?;
    let adapted =
        backbone::forward_adapter_observed(&ctx, &config, &device, &arena, &hidden, &mut sink)?;
    sink.stage("backbone_features_post_adapter", &adapted)?;

    // The static timestep conditioning lives for the model's lifetime, so it
    // gets its own arena rather than the per-request one.
    let constants = HipArena::new(&ctx, 64 << 20)?;
    let conditioning = DitConditioning::new(&ctx, &config, &device, &constants)?;
    let groups = apxinf_model::hip_gr00t::model::token_groups::build_backbone_token_groups(
        &token_ids,
        &attention_mask,
        backbone_config.image_token_id,
    )?;
    let split = action_head::split_backbone(
                &ctx, &arena, &adapted,
                &action_head::DitRowIndices::new(&ctx, &groups.non_image, &groups.image)?,
            )?;
    let actions = action_head::forward_observed(
        &ctx,
        &config,
        &device,
        &embodiment,
        &conditioning,
        &arena,
        &split,
        &ctx.to_device(&state_host)?,
        &ctx.to_device(&noise_host)?,
        &mut sink,
    )?;
    println!(
        "actions {:?} from embodiment {embodiment_id} over {} flow steps",
        actions.shape().dims(),
        conditioning.steps(),
    );

    println!(
        "wrote {} stages to {} ({} tokens, {} image rows, arena peak {:.0} MiB)",
        sink.written.len(),
        out.display(),
        text.tokens(),
        text.image_rows(),
        arena.peak() as f64 / (1 << 20) as f64,
    );
    Ok(())
}

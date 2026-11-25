//! Parity of each stage against ComfyUI traces of the same checkpoints.
//!
//! DiT parity against a ComfyUI trace of the same checkpoint (`scripts/comfy_t2i.py` then `scripts/export_comfy_trace.py`).
//! Each step is teacher-forced: the DiT receives ComfyUI's input of that step and its output is compared with ComfyUI's output.
//!
//!   APXINF_H3_IMAGE_DIT=/path/minimax_h3_image_pruned_int8_convrot.safetensors \
//!   APXINF_H3_IMAGE_TRACE=/path/runs/comfy_bf16_1024/export \
//!   cargo test --release -p apxinf-model --features cuda --lib -- --ignored dit_matches_comfyui_trace --nocapture
use crate::cuda_minimax_h3_image::backend::ops::GemmPolicy;
use crate::cuda_minimax_h3_image::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_minimax_h3_image::config::*;
use crate::cuda_minimax_h3_image::model::blocks::Int8Blocks;
use crate::cuda_minimax_h3_image::model::host;
use crate::cuda_minimax_h3_image::weights::DitWeights;
use apxinf_core::{DType, Shape, Tensor};
use std::path::{Path, PathBuf};
use std::rc::Rc;

fn read_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn upload(ctx: &CudaContext, values: &[f32], shape: Vec<usize>, dtype: DType) -> Tensor {
    let bytes: Vec<u8> = match dtype {
        DType::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        DType::BF16 => values
            .iter()
            .flat_map(|v| half::bf16::from_f32(*v).to_le_bytes())
            .collect(),
        other => panic!("unsupported {other:?}"),
    };
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host_on(&bytes, ctx.stream()).unwrap();
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
}

fn alloc_output(ctx: &CudaContext, rows: usize) -> Tensor {
    ctx.allocate_output(Shape::new(vec![rows, PATCH_DIM]), DType::F32)
        .unwrap()
}

fn download_f32(t: &Tensor) -> Vec<f32> {
    let mut bytes = vec![0u8; t.numel() * 4];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut bytes)
        .unwrap();
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Relative L2 error, cosine similarity and the largest absolute error with the reference's scale.
pub(super) struct Stats {
    pub rel_l2: f64,
    pub cosine: f64,
    pub max_abs: f64,
    pub ref_rms: f64,
}

pub(super) fn compare(actual: &[f32], reference: &[f32]) -> Stats {
    assert_eq!(actual.len(), reference.len());
    let (mut dd, mut rr, mut aa, mut ar, mut max_abs) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&a, &r) in actual.iter().zip(reference) {
        let (a, r) = (a as f64, r as f64);
        dd += (a - r) * (a - r);
        rr += r * r;
        aa += a * a;
        ar += a * r;
        max_abs = max_abs.max((a - r).abs());
    }
    Stats {
        rel_l2: (dd / rr.max(1e-30)).sqrt(),
        cosine: ar / (aa.sqrt() * rr.sqrt()).max(1e-30),
        max_abs,
        ref_rms: (rr / actual.len() as f64).sqrt(),
    }
}

#[test]
fn compare_reports_exact_match_and_scaled_error() {
    let r: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
    let same = compare(&r, &r);
    assert_eq!((same.rel_l2, same.max_abs), (0.0, 0.0));
    assert!((same.cosine - 1.0).abs() < 1e-12);
    let scaled: Vec<f32> = r.iter().map(|v| v * 1.1).collect();
    assert!((compare(&scaled, &r).rel_l2 - 0.1).abs() < 1e-6);
}

#[test]
#[ignore = "requires the INT8 ConvRot DiT checkpoint, a ComfyUI trace export and a CUDA GPU"]
fn dit_matches_comfyui_trace() {
    let dit = PathBuf::from(std::env::var("APXINF_H3_IMAGE_DIT").expect("APXINF_H3_IMAGE_DIT"));
    let trace =
        PathBuf::from(std::env::var("APXINF_H3_IMAGE_TRACE").expect("APXINF_H3_IMAGE_TRACE"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join("meta.json")).unwrap()).unwrap();
    let text_rows = meta["text_rows"].as_u64().unwrap() as usize;
    let latent: Vec<usize> = meta["latent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let (h, w) = (latent[1], latent[2]);
    // ComfyUI sees timestep = sigma * 1000 in float32
    let sigmas: Vec<f32> = meta["timesteps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32 / 1000.0)
        .collect();
    assert_eq!(sigmas.len(), PDD_STEPS);

    let canvas = Canvas::new(w * VAE_RATIO, h * VAE_RATIO).unwrap();
    let layout = Layout::new(text_rows, canvas).unwrap();
    let ctx = Rc::new(CudaContext::new(0).unwrap());
    let started = std::time::Instant::now();
    let weights = Rc::new(DitWeights::load(&dit, &ctx).unwrap());
    eprintln!("weights loaded in {:.1} s", started.elapsed().as_secs_f64());
    let blocks = Int8Blocks::new(ctx.clone(), weights, layout, canvas, &sigmas).unwrap();
    let text = upload(
        &ctx,
        &read_f32(&trace.join("context.f32")),
        vec![text_rows, HIDDEN],
        DType::BF16,
    );
    let (gemm, attention) = (GemmPolicy::default(), AttentionPolicy::default());

    let mut worst = 0f64;
    for step in 0..PDD_STEPS {
        let x = read_f32(&trace.join(format!("step{step}_x.f32")));
        let reference = read_f32(&trace.join(format!("step{step}_out.f32")));
        let rows = host::patchify(&x, h, w);
        let latent = upload(&ctx, &rows, vec![layout.image_rows, PATCH_DIM], DType::F32);
        let started = std::time::Instant::now();
        blocks.select_step(step).unwrap();
        let head = alloc_output(&ctx, layout.image_rows);
        blocks
            .forward_into(&latent, &text, &head, &gemm, &attention)
            .unwrap();
        ctx.synchronize().unwrap();
        let seconds = started.elapsed().as_secs_f64();
        // the model returns the negated head output in latent layout
        let actual: Vec<f32> = host::unpatchify(&download_f32(&head), h, w)
            .into_iter()
            .map(|v| -v)
            .collect();
        let s = compare(&actual, &reference);
        eprintln!(
            "step {step}: rel_l2 {:.4}  cosine {:.5}  max_abs {:.4}  ref_rms {:.4}  forward {:.2} s",
            s.rel_l2, s.cosine, s.max_abs, s.ref_rms, seconds
        );
        worst = worst.max(s.rel_l2);
    }
    assert!(
        worst < 0.5,
        "DiT output is not close to the ComfyUI trace (worst rel_l2 {worst:.3})"
    );
}

/// Mean and 99.9th percentile absolute difference and the fraction of pixels that differ, between two RGB8 images.
fn pixel_diff(actual: &[u8], reference: &[u8]) -> (f64, u8, f64) {
    assert_eq!(actual.len(), reference.len());
    let mut diffs: Vec<u8> = actual
        .iter()
        .zip(reference)
        .map(|(a, r)| a.abs_diff(*r))
        .collect();
    let mean = diffs.iter().map(|d| *d as f64).sum::<f64>() / diffs.len() as f64;
    let differing = diffs.iter().filter(|d| **d != 0).count() as f64 / diffs.len() as f64;
    diffs.sort_unstable();
    (
        mean,
        diffs[(diffs.len() as f64 * 0.999) as usize],
        differing,
    )
}

#[test]
fn pixel_diff_counts_differences() {
    let (mean, p999, fraction) = pixel_diff(&[0, 10, 20, 30], &[0, 10, 22, 26]);
    assert_eq!((p999, fraction), (4, 0.5));
    assert!((mean - 1.5).abs() < 1e-12);
}

#[test]
#[ignore = "requires the VAE checkpoint, a ComfyUI trace export and a CUDA GPU"]
fn vae_matches_comfyui_decode() {
    use crate::cuda_minimax_h3_image::model::blocks::VaeBlocks;
    use crate::cuda_minimax_h3_image::model::vae_host::Chw;
    use crate::cuda_minimax_h3_image::model_runner::PreparedVae;
    use crate::cuda_minimax_h3_image::weights::VaeWeights;
    let vae = PathBuf::from(std::env::var("APXINF_H3_IMAGE_VAE").expect("APXINF_H3_IMAGE_VAE"));
    let trace =
        PathBuf::from(std::env::var("APXINF_H3_IMAGE_TRACE").expect("APXINF_H3_IMAGE_TRACE"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join("meta.json")).unwrap()).unwrap();
    let latent: Vec<usize> = meta["latent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let (h, w) = (latent[1], latent[2]);
    let samples = Chw {
        c: LATENT_CHANNELS,
        h,
        w,
        data: read_f32(&trace.join("samples.f32")),
    };
    let reference = std::fs::read(trace.join("image.rgb8")).unwrap();
    assert_eq!(reference.len(), h * VAE_RATIO * w * VAE_RATIO * 3);

    // `APXINF_H3_IMAGE_AUTOTUNE=1` decodes with online tuning like the shipped loader's default; tuning can pick other candidates
    let autotune = std::env::var_os("APXINF_H3_IMAGE_AUTOTUNE").is_some();
    let ctx = Rc::new(CudaContext::new_with_autotune(0, autotune).unwrap());
    let (gemm, attention) = (
        GemmPolicy {
            online_tune: autotune,
            ..Default::default()
        },
        AttentionPolicy {
            online_tune: autotune,
            ..Default::default()
        },
    );
    let weights = Rc::new(VaeWeights::load(&vae, &ctx).unwrap());
    let tiles: usize = std::env::var("APXINF_H3_IMAGE_VAE_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let capture = std::env::var_os("APXINF_H3_IMAGE_EAGER").is_none();
    let blocks = Rc::new(VaeBlocks::new(ctx.clone(), weights, tiles).unwrap());
    let blocks = PreparedVae::new(ctx.clone(), blocks, capture, gemm, attention).unwrap();
    let started = std::time::Instant::now();
    let rgb = blocks.decode_image(&samples).unwrap();
    eprintln!("first decode {:.2} s", started.elapsed().as_secs_f64());
    let started = std::time::Instant::now();
    let again = blocks.decode_image(&samples).unwrap();
    eprintln!("second decode {:.2} s", started.elapsed().as_secs_f64());
    assert_eq!(
        rgb, again,
        "decoding the same latent twice must give identical pixels"
    );
    let (mean, p999, fraction) = pixel_diff(&rgb, &reference);
    eprintln!(
        "vs ComfyUI: mean abs {mean:.3} /255, 99.9th percentile {p999}, differing pixels {:.1} %",
        fraction * 100.0
    );
    assert!(
        mean < 6.0,
        "VAE decode is far from ComfyUI (mean abs {mean:.2} /255)"
    );
}

#[test]
#[ignore = "requires a ComfyUI trace export (noise.f32)"]
fn noise_matches_torch_randn() {
    let trace =
        PathBuf::from(std::env::var("APXINF_H3_IMAGE_TRACE").expect("APXINF_H3_IMAGE_TRACE"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join("meta.json")).unwrap()).unwrap();
    let seed = meta["seed"].as_u64().expect("meta.json carries the seed");
    let reference = read_f32(&trace.join("noise.f32"));
    let ours = crate::cuda_minimax_h3_image::model::noise::randn(seed, reference.len());
    let s = compare(&ours, &reference);
    eprintln!(
        "noise vs torch.randn: rel_l2 {:.2e}  max_abs {:.2e}",
        s.rel_l2, s.max_abs
    );
    assert!(
        s.max_abs < 1e-4,
        "noise differs from torch.randn (max abs {})",
        s.max_abs
    );
}

/// Layer-by-layer parity of the streamed text encoder against ComfyUI's captured activations
/// (`scripts/capture_comfy_te.py --layers 0,1,24,49`). ComfyUI runs this stack in FP32, so the comparison measures
/// the BF16 + INT8 stream against an FP32 stream of the same INT8 weights.
///
///   APXINF_H3_IMAGE_TE=/path/qwen3vl_32b_minimax_h3_int8_convrot.safetensors APXINF_H3_IMAGE_TE_TRACE=/path/te_fox \
///   cargo test --release -p apxinf-model --features cuda --lib -- --ignored text_encoder_matches_comfyui
#[test]
#[ignore = "requires the 27 GB text-encoder checkpoint, a ComfyUI TE capture and a CUDA GPU"]
fn text_encoder_matches_comfyui() {
    use crate::cuda_minimax_h3_image::model::blocks::TextBlocks;
    use crate::cuda_minimax_h3_image::weights::{Pinning, TextWeights};
    let te = PathBuf::from(std::env::var("APXINF_H3_IMAGE_TE").expect("APXINF_H3_IMAGE_TE"));
    let trace =
        PathBuf::from(std::env::var("APXINF_H3_IMAGE_TE_TRACE").expect("APXINF_H3_IMAGE_TE_TRACE"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join("te_meta.json")).unwrap()).unwrap();
    let ids: Vec<u32> = meta["ids"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let real = ids.len();
    let padded = real.div_ceil(TE_ROW_BUCKET) * TE_ROW_BUCKET;
    let ctx = Rc::new(CudaContext::new(0).unwrap());
    let blocks = TextBlocks::new(ctx.clone(), padded).unwrap();
    let (gemm, attention) = (GemmPolicy::default(), AttentionPolicy::default());

    let weights = TextWeights::open(&te, ctx.clone(), Pinning::from_env()).unwrap();
    let embeddings = weights.embed(&ids).unwrap();
    let embedded: Vec<f32> = embeddings.iter().map(|v| v.to_f32()).collect();
    let s = compare(&embedded, &read_f32(&trace.join("embed_out.f32")));
    eprintln!(
        "embedding rows: rel_l2 {:.2e} (BF16 table vs ComfyUI's float32 copy of it)",
        s.rel_l2
    );
    assert!(s.rel_l2 < 1e-6, "embedding rows differ from ComfyUI");

    let x = blocks.embed(&embeddings, real).unwrap();
    let started = std::time::Instant::now();
    let mut worst = 0f64;
    weights
        .run_layers(|layer, device| {
            blocks.layer(device, &x, &gemm, &attention)?;
            let reference = trace.join(format!("layer{layer}_out.f32"));
            if reference.exists() || layer + 1 == TE_LAYERS {
                ctx.synchronize().map_err(apxinf_core::Error::Cuda)?;
                let all = download_bf16(&x);
                let ours = &all[..real * TE_DIM];
                let theirs = read_f32(&if reference.exists() { reference } else { trace.join("te_out.f32") });
                let s = compare(ours, &theirs);
                eprintln!(
                    "layer {layer:2}: rel_l2 {:.4}  cosine {:.5}  max_abs {:.3}  ref_rms {:.3}  ({:.1} s elapsed)",
                    s.rel_l2, s.cosine, s.max_abs, s.ref_rms, started.elapsed().as_secs_f64()
                );
                worst = worst.max(s.rel_l2);
            }
            Ok(())
        })
        .unwrap();
    assert!(
        worst < 0.5,
        "text encoder is not close to the ComfyUI capture (worst rel_l2 {worst:.3})"
    );
}

fn download_bf16(t: &Tensor) -> Vec<f32> {
    let mut bytes = vec![0u8; t.numel() * 2];
    CudaBuffer::from_tensor(t)
        .unwrap()
        .copy_to_host(&mut bytes)
        .unwrap();
    bytes
        .chunks_exact(2)
        .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
        .collect()
}

//! Parity of the DiT against a ComfyUI trace of the same fl2va checkpoint (`scripts/comfy_t2va.py`, `scripts/export_h3_trace.py`).
//! Each step is teacher-forced: the DiT receives ComfyUI's inputs of that step and both stream outputs are compared with ComfyUI's.
//!
//!   APXINF_H3_DIT=/path/minimax_h3_fl2va_pruned_int8_convrot.safetensors \
//!   APXINF_H3_TRACE=/path/runs/comfy_t2va/export \
//!   cargo test --release -p apxinf-model --features cuda --lib -- --ignored dit_matches_comfyui_trace --nocapture
use crate::cuda_minimax_h3::backend::ops::GemmPolicy;
use crate::cuda_minimax_h3::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_minimax_h3::config::*;
use crate::cuda_minimax_h3::model::blocks::{Int8Blocks, StepTimes};
use crate::cuda_minimax_h3::model::host;
use crate::cuda_minimax_h3::model::layout::Layout;
use crate::cuda_minimax_h3::weights::DitWeights;
use apxinf_core::{DType, Shape, Tensor};
use std::path::{Path, PathBuf};
use std::rc::Rc;

pub(super) fn read_f32(path: &Path) -> Vec<f32> {
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

fn output(ctx: &CudaContext, rows: usize, width: usize) -> Tensor {
    ctx.allocate_output(Shape::new(vec![rows, width]), DType::F32)
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
#[ignore = "requires the INT8 ConvRot fl2va DiT checkpoint, a ComfyUI trace export and a CUDA GPU"]
fn dit_matches_comfyui_trace() {
    let dit = PathBuf::from(std::env::var("APXINF_H3_DIT").expect("APXINF_H3_DIT"));
    let trace = PathBuf::from(std::env::var("APXINF_H3_TRACE").expect("APXINF_H3_TRACE"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join("meta.json")).unwrap()).unwrap();
    let text_rows = meta["text_rows"].as_u64().unwrap() as usize;
    let latent: Vec<usize> = meta["latent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let (t, h, w) = (latent[0], latent[1], latent[2]);
    let audio_t = meta["audio_t"].as_u64().unwrap() as usize;
    // ComfyUI hands the model timestep = sigma * 1000 in float32
    let timesteps: Vec<f32> = meta["timesteps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let schedule: Vec<StepTimes> = timesteps
        .iter()
        .map(|ts| StepTimes {
            sigma_video: ts / 1000.0,
        })
        .collect();

    let layout = Layout::new(text_rows, t, h, w, audio_t).unwrap();
    let ctx = Rc::new(CudaContext::new(0).unwrap());
    let started = std::time::Instant::now();
    let weights = Rc::new(DitWeights::load(&dit, &ctx).unwrap());
    eprintln!("weights loaded in {:.1} s", started.elapsed().as_secs_f64());
    let blocks = Int8Blocks::new(ctx.clone(), weights, layout, &schedule).unwrap();
    let text = upload(
        &ctx,
        &read_f32(&trace.join("context.f32")),
        vec![text_rows, HIDDEN],
        DType::BF16,
    );
    let (gemm, attention) = (GemmPolicy::default(), AttentionPolicy::default());

    let mut worst = (0f64, 0f64);
    for step in 0..schedule.len() {
        let video_in = read_f32(&trace.join(format!("step{step}_video_in.f32")));
        let audio_in = read_f32(&trace.join(format!("step{step}_audio_in.f32")));
        let video_ref = read_f32(&trace.join(format!("step{step}_video_out.f32")));
        let audio_ref = read_f32(&trace.join(format!("step{step}_audio_out.f32")));
        let video = upload(
            &ctx,
            &host::patchify_video(&video_in, t, h, w),
            vec![layout.video_rows(), PATCH_DIM],
            DType::F32,
        );
        let audio = upload(
            &ctx,
            &host::pack_audio(&audio_in, audio_t),
            vec![layout.audio_rows(), AUDIO_CHANNELS],
            DType::F32,
        );
        let video_out = output(&ctx, layout.video_rows(), PATCH_DIM);
        let audio_out = output(&ctx, layout.audio_rows(), AUDIO_CHANNELS);
        let started = std::time::Instant::now();
        blocks.select_step(step).unwrap();
        blocks
            .forward_into(
                &video, &audio, &text, &video_out, &audio_out, &gemm, &attention,
            )
            .unwrap();
        ctx.synchronize().unwrap();
        let seconds = started.elapsed().as_secs_f64();
        // the model returns the negated head output in latent layout
        let video_actual: Vec<f32> = host::unpatchify_video(&download_f32(&video_out), t, h, w)
            .into_iter()
            .map(|v| -v)
            .collect();
        let audio_actual: Vec<f32> = host::unpack_audio(&download_f32(&audio_out), audio_t)
            .into_iter()
            .map(|v| -v)
            .collect();
        let (sv, sa) = (
            compare(&video_actual, &video_ref),
            compare(&audio_actual, &audio_ref),
        );
        eprintln!(
            "step {step}: video rel_l2 {:.4} cos {:.5} max_abs {:.3} (ref rms {:.3}) | audio rel_l2 {:.4} cos {:.5} max_abs {:.3} (ref rms {:.3}) | {:.2} s",
            sv.rel_l2, sv.cosine, sv.max_abs, sv.ref_rms, sa.rel_l2, sa.cosine, sa.max_abs, sa.ref_rms, seconds
        );
        worst = (worst.0.max(sv.rel_l2), worst.1.max(sa.rel_l2));
    }
    assert!(
        worst.0 < 0.5 && worst.1 < 0.5,
        "DiT output is not close to the ComfyUI trace (worst rel_l2 {worst:?})"
    );
}

fn read_bf16(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
        .collect()
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

/// Where does a difference start? Compares the residual stream after the embedding and after chosen blocks with ComfyUI's, per segment
/// (text, audio, video), for each captured step. `APXINF_H3_TAPS` holds `stepS_embed.bf16` and `stepS_blockB.bf16` from
/// `scripts/comfy_h3_taps.py`; `APXINF_H3_TAP_BLOCKS` lists the blocks (default `0,1,4,9,24,49`).
#[test]
#[ignore = "requires the fl2va DiT checkpoint, a ComfyUI trace export, ComfyUI hidden-state taps and a CUDA GPU"]
fn hidden_states_match_comfyui_per_segment() {
    let dit = PathBuf::from(std::env::var("APXINF_H3_DIT").expect("APXINF_H3_DIT"));
    let trace = PathBuf::from(std::env::var("APXINF_H3_TRACE").expect("APXINF_H3_TRACE"));
    let taps = PathBuf::from(std::env::var("APXINF_H3_TAPS").expect("APXINF_H3_TAPS"));
    let blocks: Vec<usize> = std::env::var("APXINF_H3_TAP_BLOCKS")
        .unwrap_or_else(|_| "0,1,4,9,24,49".into())
        .split(',')
        .map(|b| b.trim().parse().unwrap())
        .collect();
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join("meta.json")).unwrap()).unwrap();
    let text_rows = meta["text_rows"].as_u64().unwrap() as usize;
    let latent: Vec<usize> = meta["latent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let (t, h, w) = (latent[0], latent[1], latent[2]);
    let audio_t = meta["audio_t"].as_u64().unwrap() as usize;
    let timesteps: Vec<f32> = meta["timesteps"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect();
    let schedule: Vec<StepTimes> = timesteps
        .iter()
        .map(|ts| StepTimes {
            sigma_video: ts / 1000.0,
        })
        .collect();
    let layout = Layout::new(text_rows, t, h, w, audio_t).unwrap();
    let ctx = Rc::new(CudaContext::new(0).unwrap());
    let weights = Rc::new(DitWeights::load(&dit, &ctx).unwrap());
    let blocks_dit = Int8Blocks::new(ctx.clone(), weights, layout, &schedule).unwrap();
    let text = upload(
        &ctx,
        &read_f32(&trace.join("context.f32")),
        vec![text_rows, HIDDEN],
        DType::BF16,
    );
    let (gemm, attention) = (GemmPolicy::default(), AttentionPolicy::default());
    let segments = layout.segments();
    let names = ["text ", "audio", "video"];
    for step in 0..schedule.len() {
        let video_in = read_f32(&trace.join(format!("step{step}_video_in.f32")));
        let audio_in = read_f32(&trace.join(format!("step{step}_audio_in.f32")));
        let video = upload(
            &ctx,
            &host::patchify_video(&video_in, t, h, w),
            vec![layout.video_rows(), PATCH_DIM],
            DType::F32,
        );
        let audio = upload(
            &ctx,
            &host::pack_audio(&audio_in, audio_t),
            vec![layout.audio_rows(), AUDIO_CHANNELS],
            DType::F32,
        );
        let (video_out, audio_out) = (
            output(&ctx, layout.video_rows(), PATCH_DIM),
            output(&ctx, layout.audio_rows(), AUDIO_CHANNELS),
        );
        blocks_dit.select_step(step).unwrap();
        blocks_dit
            .forward_tapped(
                &video,
                &audio,
                &text,
                &video_out,
                &audio_out,
                &gemm,
                &attention,
                &mut |index, x| {
                    let name = if index == usize::MAX {
                        format!("step{step}_embed")
                    } else if blocks.contains(&index) {
                        format!("step{step}_block{index}")
                    } else {
                        return Ok(());
                    };
                    ctx.synchronize().map_err(apxinf_core::Error::Cuda)?;
                    let ours = download_bf16(x);
                    let theirs = read_bf16(&taps.join(format!("{name}.bf16")));
                    let mut line = format!("{name:<16}");
                    for (seg, label) in segments.iter().zip(names) {
                        let range = seg.start * HIDDEN..seg.stop * HIDDEN;
                        let s = compare(&ours[range.clone()], &theirs[range]);
                        line.push_str(&format!(
                            " | {label} rel_l2 {:.4} cos {:.5} (ref rms {:.2})",
                            s.rel_l2, s.cosine, s.ref_rms
                        ));
                    }
                    eprintln!("{line}");
                    Ok(())
                },
            )
            .unwrap();
    }
}

#[test]
#[ignore = "requires a ComfyUI trace export (noise_video.f32, noise_audio.f32)"]
fn seed_noise_equals_comfyui_nested_noise() {
    use crate::cuda_minimax_h3::model::noise::Generator;
    let trace = PathBuf::from(std::env::var("APXINF_H3_TRACE").expect("APXINF_H3_TRACE"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join("meta.json")).unwrap()).unwrap();
    let latent: Vec<usize> = meta["latent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let audio_t = meta["audio_t"].as_u64().unwrap() as usize;
    let seed = std::env::var("APXINF_H3_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(42u64);
    // `comfy.sample.prepare_noise` draws the video tensor and then the audio tensor from one generator
    let mut generator = Generator::new(seed);
    let video = generator.randn(LATENT_CHANNELS * latent[0] * latent[1] * latent[2]);
    let audio = generator.randn(AUDIO_CHANNELS * AUDIO_STEREO * audio_t);
    let (sv, sa) = (
        compare(&video, &read_f32(&trace.join("noise_video.f32"))),
        compare(&audio, &read_f32(&trace.join("noise_audio.f32"))),
    );
    eprintln!(
        "video noise max_abs {:.2e} rel_l2 {:.2e} | audio noise max_abs {:.2e} rel_l2 {:.2e}",
        sv.max_abs, sv.rel_l2, sa.max_abs, sa.rel_l2
    );
    // torch's AVX2 log/cos/sin differ from libm in the last bits; anything above 1e-4 is a different stream
    assert!(
        sv.max_abs < 1e-4 && sa.max_abs < 1e-4,
        "noise differs from ComfyUI's (video {:.2e}, audio {:.2e})",
        sv.max_abs,
        sa.max_abs
    );
}

/// Free-running: starting from ComfyUI's own noise and text context, APXInf's sampler (graph unless `APXINF_H3_EAGER` is set) runs every
/// step on its own latents. The final latents are compared with ComfyUI's; unlike the teacher-forced test the errors of the steps compound.
#[test]
#[ignore = "requires the INT8 ConvRot fl2va DiT checkpoint, a ComfyUI trace export and a CUDA GPU"]
fn sampler_final_latents_match_comfyui() {
    use crate::cuda_minimax_h3::model_runner::{PlanKey, PreparedDenoise};
    let dit = PathBuf::from(std::env::var("APXINF_H3_DIT").expect("APXINF_H3_DIT"));
    let trace = PathBuf::from(std::env::var("APXINF_H3_TRACE").expect("APXINF_H3_TRACE"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join("meta.json")).unwrap()).unwrap();
    let text_rows = meta["text_rows"].as_u64().unwrap() as usize;
    let latent: Vec<usize> = meta["latent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let (t, h, w) = (latent[0], latent[1], latent[2]);
    let audio_t = meta["audio_t"].as_u64().unwrap() as usize;
    let steps = meta["steps"].as_u64().unwrap() as usize;
    let capture = std::env::var_os("APXINF_H3_EAGER").is_none();

    let layout = Layout::new(text_rows, t, h, w, audio_t).unwrap();
    let ctx = Rc::new(CudaContext::new(0).unwrap());
    let weights = Rc::new(DitWeights::load(&dit, &ctx).unwrap());
    let started = std::time::Instant::now();
    let plan = PreparedDenoise::new(
        ctx.clone(),
        weights,
        PlanKey { layout, steps },
        capture,
        GemmPolicy::default(),
        AttentionPolicy::default(),
    )
    .unwrap();
    ctx.synchronize().unwrap();
    eprintln!(
        "plan ready in {:.2} s (captured: {})",
        started.elapsed().as_secs_f64(),
        plan.is_captured()
    );

    let text = upload(
        &ctx,
        &read_f32(&trace.join("context.f32")),
        vec![text_rows, HIDDEN],
        DType::BF16,
    );
    let video_noise = host::patchify_video(&read_f32(&trace.join("noise_video.f32")), t, h, w);
    let audio_noise = host::pack_audio(&read_f32(&trace.join("noise_audio.f32")), audio_t);
    // every repeat re-binds the same noise on the same plan, so each ends at the same latents; the first one includes first-replay costs
    let repeats: usize = std::env::var("APXINF_H3_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let mut times = Vec::new();
    for _ in 0..repeats {
        plan.bind(&text, &video_noise, &audio_noise).unwrap();
        ctx.synchronize().unwrap();
        let started = std::time::Instant::now();
        for step in 0..steps {
            plan.step(step).unwrap();
        }
        ctx.synchronize().unwrap();
        times.push(started.elapsed().as_secs_f64());
    }
    eprintln!(
        "denoise seconds per repeat: {:?}",
        times
            .iter()
            .map(|t| (t * 1000.0).round() / 1000.0)
            .collect::<Vec<_>>()
    );
    let seconds = *times.last().unwrap();
    let (video, audio) = plan.latents().unwrap();
    let video = host::unpatchify_video(&video, t, h, w);
    let audio = host::unpack_audio(&audio, audio_t);
    let (sv, sa) = (
        compare(&video, &read_f32(&trace.join("final_video.f32"))),
        compare(&audio, &read_f32(&trace.join("final_audio.f32"))),
    );
    eprintln!(
        "{steps} steps in {seconds:.2} s | video rel_l2 {:.4} cos {:.5} (ref rms {:.3}) | audio rel_l2 {:.4} cos {:.5} (ref rms {:.3})",
        sv.rel_l2, sv.cosine, sv.ref_rms, sa.rel_l2, sa.cosine, sa.ref_rms
    );
    assert!(
        sv.rel_l2 < 0.5 && sa.rel_l2 < 0.5,
        "sampled latents are not close to ComfyUI's (video {:.3}, audio {:.3})",
        sv.rel_l2,
        sa.rel_l2
    );
}

/// Everything from a prompt and a seed, no ComfyUI input except the reference to compare with: the tokenizer, the streamed text encoder,
/// the refiner, the noise and the sampler. The free-running ComfyUI run used the same prompt, seed and size.
///
///   APXINF_H3_PACKAGE=/path/h3-pkg APXINF_H3_TRACE=/path/export (meta.json carries the geometry; APXINF_H3_PROMPT overrides the prompt)
#[test]
#[ignore = "requires the full H3 package (DiT, 27 GB text encoder, tokenizer), a ComfyUI trace export and a CUDA GPU"]
fn prompt_to_latents_matches_comfyui() {
    use crate::cuda_minimax_h3::model_runner::{GenerationRequest, ModelRunner};
    use crate::cuda_minimax_h3::weights::{Pinning, TextWeights};
    use apxinf_tokenizer::Tokenizer;
    let package = crate::cuda_minimax_h3::package::Package::open(&PathBuf::from(
        std::env::var("APXINF_H3_PACKAGE").expect("APXINF_H3_PACKAGE"),
    ))
    .unwrap();
    let trace = PathBuf::from(std::env::var("APXINF_H3_TRACE").expect("APXINF_H3_TRACE"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join("meta.json")).unwrap()).unwrap();
    let latent: Vec<usize> = meta["latent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let steps = meta["steps"].as_u64().unwrap() as usize;
    let prompt = std::env::var("APXINF_H3_PROMPT").unwrap_or_else(|_| {
        "A red fox trotting through a snowy pine forest, birds singing.".into()
    });
    let (width, height) = (latent[2] * VAE_RATIO, latent[1] * VAE_RATIO);
    let frames = (latent[0] - 2) / 5 * 17 + 5;

    let ctx = Rc::new(CudaContext::new(0).unwrap());
    let tokenizer = Tokenizer::from_qwen2_vocab(&package.tokenizer).unwrap();
    // ComfyUI's ids for this prompt (clip.tokenize), captured once on the host
    let expected_ids: Vec<u32> = vec![
        32, 2518, 38835, 56577, 1280, 1526, 264, 89773, 33597, 13638, 11, 19654, 25083, 13,
    ];
    let ids = tokenizer
        .encode_with_special_tokens(&prompt, false)
        .unwrap();
    if prompt.starts_with("A red fox trotting") {
        assert_eq!(ids, expected_ids, "token ids differ from ComfyUI's");
    }
    let dit = Rc::new(DitWeights::load(&package.transformer, &ctx).unwrap());
    let text = TextWeights::open(&package.text_encoder, ctx.clone(), Pinning::from_env()).unwrap();
    let video = crate::cuda_minimax_h3::model_runner::VideoDecoder::new(
        ctx.clone(),
        crate::cuda_minimax_h3::model::blocks::VideoVaeBlocks::new(
            ctx.clone(),
            Rc::new(
                crate::cuda_minimax_h3::weights::VideoVaeWeights::load(
                    &package.video_vae,
                    ctx.clone(),
                )
                .unwrap(),
            ),
        ),
        crate::cuda_minimax_h3::model_runner::TILE_BATCH,
        GemmPolicy::default(),
        AttentionPolicy::default(),
    );
    let audio = crate::cuda_minimax_h3::model_runner::AudioDecoder::new(
        ctx.clone(),
        crate::cuda_minimax_h3::model::blocks::AudioVaeBlocks::new(
            ctx.clone(),
            Rc::new(
                crate::cuda_minimax_h3::weights::AudioVaeWeights::load(&package.audio_vae, &ctx)
                    .unwrap(),
            ),
        ),
    );
    let mut runner = ModelRunner::new(
        ctx.clone(),
        dit,
        tokenizer,
        text,
        video,
        audio,
        GemmPolicy::default(),
        AttentionPolicy::default(),
    );
    let request = GenerationRequest {
        prompt: &prompt,
        width,
        height,
        frames,
        seed: 42,
        steps,
    };
    let mut results = Vec::new();
    for pass in 0..2 {
        let started = std::time::Instant::now();
        let latents = runner.denoise(&request).unwrap();
        eprintln!(
            "pass {pass} ({}): {:.2} s",
            if pass == 0 {
                "cold prompt"
            } else {
                "cached prompt"
            },
            started.elapsed().as_secs_f64()
        );
        results.push(latents);
    }
    // the refined text rows (what the DiT receives as context) against ComfyUI's, which the earlier free-running test fed in directly
    let context = runner.refined_text_rows(&prompt).unwrap();
    let sc = compare(&context, &read_f32(&trace.join("context.f32")));
    eprintln!(
        "refined text context vs ComfyUI: rel_l2 {:.4} cos {:.5} max_abs {:.3} (ref rms {:.3})",
        sc.rel_l2, sc.cosine, sc.max_abs, sc.ref_rms
    );
    assert_eq!(
        results[0].video, results[1].video,
        "a cached prompt must reproduce the same video latents"
    );
    assert_eq!(results[0].audio, results[1].audio);
    let l = &results[0];
    assert_eq!(
        (
            l.layout.latent_t,
            l.layout.latent_h,
            l.layout.latent_w,
            l.layout.audio_t
        ),
        (
            latent[0],
            latent[1],
            latent[2],
            meta["audio_t"].as_u64().unwrap() as usize
        )
    );
    let (sv, sa) = (
        compare(&l.video, &read_f32(&trace.join("final_video.f32"))),
        compare(&l.audio, &read_f32(&trace.join("final_audio.f32"))),
    );
    eprintln!(
        "prompt -> latents vs ComfyUI: video rel_l2 {:.4} cos {:.5} | audio rel_l2 {:.4} cos {:.5}",
        sv.rel_l2, sv.cosine, sa.rel_l2, sa.cosine
    );
    assert!(
        sv.rel_l2 < 0.5 && sa.rel_l2 < 0.5,
        "latents are not close to ComfyUI's"
    );
}

/// One 256x256-pixel tile of 7 latent steps through the video VAE decoder against ComfyUI's activations of the same tile
/// (`scripts/comfy_h3_vae_tile_taps.py`): the embedding, the residual stream after blocks 0, 17 and 35, and the pixel patches.
///
///   APXINF_H3_VAE=/path/minimax_h3_video_vae_fp16.safetensors APXINF_H3_VAE_TILE=/path/vae_tile \
///   cargo test --release -p apxinf-model --features cuda --lib -- --ignored video_vae_tile_matches_comfyui --nocapture
#[test]
#[ignore = "requires the video VAE checkpoint, a ComfyUI one-tile capture and a CUDA GPU"]
fn video_vae_tile_matches_comfyui() {
    use crate::cuda_minimax_h3::model::blocks::VideoVaeBlocks;
    use crate::cuda_minimax_h3::model::vae_host::tile_tokens;
    use crate::cuda_minimax_h3::weights::VideoVaeWeights;
    let vae = PathBuf::from(std::env::var("APXINF_H3_VAE").expect("APXINF_H3_VAE"));
    let dir = PathBuf::from(std::env::var("APXINF_H3_VAE_TILE").expect("APXINF_H3_VAE_TILE"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("meta.json")).unwrap()).unwrap();
    let steps = meta["tile_z"][2].as_u64().unwrap() as usize;
    let ctx = Rc::new(CudaContext::new(0).unwrap());
    let started = std::time::Instant::now();
    let weights = Rc::new(VideoVaeWeights::load(&vae, ctx.clone()).unwrap());
    eprintln!(
        "video VAE weights converted and uploaded in {:.1} s (pinned: {})",
        started.elapsed().as_secs_f64(),
        weights.is_pinned()
    );
    let blocks = VideoVaeBlocks::new(ctx.clone(), weights);
    let (gemm, attention) = (GemmPolicy::default(), AttentionPolicy::default());

    // z is [24, T, 16, 16]; the decoder takes token rows (time, row, column) of channel vectors
    let z = read_f32(&dir.join("tile_z.f32"));
    let (n, mut tokens) = (VV_TILE_LATENT, Vec::new());
    for ft in 0..steps {
        for y in 0..n {
            for x in 0..n {
                for c in 0..LATENT_CHANNELS {
                    tokens.push(z[((c * steps + ft) * n + y) * n + x]);
                }
            }
        }
    }
    let latents = upload(
        &ctx,
        &tokens,
        vec![steps * n * n, LATENT_CHANNELS],
        DType::F32,
    );
    let rows = tile_tokens(steps);
    let mut worst = 0f64;
    let check = |label: &str, blocks_run: usize, file: &str| {
        let ours = blocks
            .residual_stream(&latents, 1, steps, &gemm, &attention, blocks_run)
            .unwrap();
        ctx.synchronize().unwrap();
        let ours = download_bf16(&ours);
        let reference = read_f32(&dir.join(file));
        // the reference carries only the latent tokens for the embedding tap and all `rows` tokens for blocks
        let take = reference.len() / VV_DIM;
        let stats = compare(&ours[..take * VV_DIM], &reference);
        eprintln!("{label:<10} rel_l2 {:.4} cos {:.5} max_abs {:.3} (ref rms {:.3}, {take} of {rows} rows)", stats.rel_l2, stats.cosine, stats.max_abs, stats.ref_rms);
        stats.rel_l2
    };
    worst = worst.max(check("embedding", 0, "tile_embed.f32"));
    for (label, run, file) in [
        ("block 0", 1, "tile_block0.f32"),
        ("block 17", 18, "tile_block17.f32"),
        ("block 35", 36, "tile_block35.f32"),
    ] {
        worst = worst.max(check(label, run, file));
    }
    // pixel patches: decoder output [3, T*4, 256, 256] in ComfyUI's unpatchified layout
    let started = std::time::Instant::now();
    let out = blocks
        .decode(&latents, 1, steps, &gemm, &attention)
        .unwrap();
    ctx.synchronize().unwrap();
    let seconds = started.elapsed().as_secs_f64();
    let patches = download_bf16(&out);
    let tile = crate::cuda_minimax_h3::model::vae_host::unpatchify_pixels(
        &patches[..steps * n * n * VV_OUT_DIM],
        steps,
        n,
        n,
    );
    let stats = compare(&tile.data, &read_f32(&dir.join("tile_out.f32")));
    eprintln!(
        "pixels     rel_l2 {:.4} cos {:.5} max_abs {:.3} (ref rms {:.3}) | decode {:.2} s",
        stats.rel_l2, stats.cosine, stats.max_abs, stats.ref_rms, seconds
    );
    assert!(
        worst < 0.5 && stats.rel_l2 < 0.5,
        "video VAE tile is not close to ComfyUI's (worst stream {worst:.3}, pixels {:.3})",
        stats.rel_l2
    );
}

/// The whole video decode of ComfyUI's final latents (every tile of every chunk, both blends, the pixel denormalisation) against
/// ComfyUI's decode of the same latents (`scripts/comfy_h3_vae_ref.py`: `video_pixels.f32`, `[3, T, H, W]` in `[0, 1]`).
///
///   APXINF_H3_VAE=... APXINF_H3_TRACE=/path/export APXINF_H3_VAE_REF=/path/vae_ref
#[test]
#[ignore = "requires the video VAE checkpoint, ComfyUI's final latents and decode, and a CUDA GPU"]
fn video_decode_matches_comfyui() {
    use crate::cuda_minimax_h3::model::blocks::VideoVaeBlocks;
    use crate::cuda_minimax_h3::model_runner::VideoDecoder;
    use crate::cuda_minimax_h3::weights::VideoVaeWeights;
    let vae = PathBuf::from(std::env::var("APXINF_H3_VAE").expect("APXINF_H3_VAE"));
    let trace = PathBuf::from(std::env::var("APXINF_H3_TRACE").expect("APXINF_H3_TRACE"));
    let reference = PathBuf::from(std::env::var("APXINF_H3_VAE_REF").expect("APXINF_H3_VAE_REF"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join("meta.json")).unwrap()).unwrap();
    let dims: Vec<usize> = meta["latent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let ctx = Rc::new(CudaContext::new(0).unwrap());
    let weights = Rc::new(VideoVaeWeights::load(&vae, ctx.clone()).unwrap());
    let batch: usize = std::env::var("APXINF_H3_VAE_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(crate::cuda_minimax_h3::model_runner::TILE_BATCH);
    let decoder = VideoDecoder::new(
        ctx.clone(),
        VideoVaeBlocks::new(ctx.clone(), weights),
        batch,
        GemmPolicy::default(),
        AttentionPolicy::default(),
    );
    let latent = read_f32(&trace.join("final_video.f32"));
    let mut times = Vec::new();
    let mut video = None;
    for _ in 0..3 {
        let started = std::time::Instant::now();
        video = Some(
            decoder
                .decode(&latent, [dims[0], dims[1], dims[2]])
                .unwrap(),
        );
        times.push((started.elapsed().as_secs_f64() * 1000.0).round() / 1000.0);
    }
    let video = video.unwrap();
    eprintln!(
        "video decode (batch {batch}) seconds per pass: {times:?} -> [{}, {}, {}, {}]",
        video.c, video.t, video.h, video.w
    );
    let want = read_f32(&reference.join("video_pixels.f32"));
    assert_eq!(
        video.data.len(),
        want.len(),
        "decoded volume has a different size than ComfyUI's"
    );
    let stats = compare(&video.data, &want);
    let mean_abs = video
        .data
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs() as f64)
        .sum::<f64>()
        / want.len() as f64;
    eprintln!(
        "video pixels vs ComfyUI: rel_l2 {:.4} cos {:.5} mean abs {:.5} ({:.2} / 255) max_abs {:.3}",
        stats.rel_l2, stats.cosine, mean_abs, mean_abs * 255.0, stats.max_abs
    );
    assert!(
        stats.rel_l2 < 0.2,
        "decoded video is not close to ComfyUI's"
    );
}

/// The audio VAE decoder against ComfyUI's, stage by stage and as a waveform. ComfyUI's references are taken with cuDNN TF32 on (a stock run)
/// and off; the difference between those two is the reference's own precision-mode spread, which the comparison is read against.
///
///   APXINF_H3_AVAE=/path/minimax_h3_audio_vae_fp32.safetensors APXINF_H3_TRACE=/path/export \
///   APXINF_H3_AVAE_REF=/root/runs/comfy_t2va   (directories avae_tf32 and avae_notf32 from scripts/comfy_h3_avae_taps.py)
#[test]
#[ignore = "requires the audio VAE checkpoint, ComfyUI's final latents and decode taps, and a CUDA GPU"]
fn audio_decode_matches_comfyui() {
    use crate::cuda_minimax_h3::model::blocks::AudioVaeBlocks;
    use crate::cuda_minimax_h3::model_runner::AudioDecoder;
    use crate::cuda_minimax_h3::weights::AudioVaeWeights;
    let vae = PathBuf::from(std::env::var("APXINF_H3_AVAE").expect("APXINF_H3_AVAE"));
    let trace = PathBuf::from(std::env::var("APXINF_H3_TRACE").expect("APXINF_H3_TRACE"));
    let reference = PathBuf::from(std::env::var("APXINF_H3_AVAE_REF").expect("APXINF_H3_AVAE_REF"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join("meta.json")).unwrap()).unwrap();
    let frames = meta["audio_t"].as_u64().unwrap() as usize;
    let ctx = Rc::new(CudaContext::new(0).unwrap());
    let started = std::time::Instant::now();
    let weights = Rc::new(AudioVaeWeights::load(&vae, &ctx).unwrap());
    eprintln!(
        "audio VAE weights loaded in {:.2} s",
        started.elapsed().as_secs_f64()
    );
    let decoder = AudioDecoder::new(ctx.clone(), AudioVaeBlocks::new(ctx.clone(), weights));
    let latent = read_f32(&trace.join("final_audio.f32"));

    // stage-by-stage against both references, on the first pass
    let mut rows: Vec<(String, f64, f64)> = Vec::new();
    let mut tap = |name: &str, t: &Tensor| -> apxinf_core::Result<()> {
        ctx.synchronize().map_err(apxinf_core::Error::Cuda)?;
        let ours = download_f32(t);
        let per = ours.len() / 2;
        let (mut vs_tf32, mut vs_fp32) = (0f64, 0f64);
        for s in 0..2 {
            let file = format!("{name}_s{s}.f32");
            let ours_s = &ours[s * per..(s + 1) * per];
            vs_tf32 = vs_tf32
                .max(compare(ours_s, &read_f32(&reference.join("avae_tf32").join(&file))).rel_l2);
            vs_fp32 = vs_fp32.max(
                compare(
                    ours_s,
                    &read_f32(&reference.join("avae_notf32").join(&file)),
                )
                .rel_l2,
            );
        }
        rows.push((name.to_owned(), vs_tf32, vs_fp32));
        Ok(())
    };
    let wave = decoder.decode_with_taps(&latent, frames, &mut tap).unwrap();
    eprintln!("stage          rel_l2 vs ComfyUI (TF32 on) | vs ComfyUI (strict FP32)   [worst of the two stereo channels]");
    for (name, a, b) in &rows {
        eprintln!("{name:<12} {a:>12.2e} | {b:>12.2e}");
    }
    // the waveform: [2, L] against ComfyUI's per-channel files
    let l = wave.len() / 2;
    for (label, dir) in [("TF32 on", "avae_tf32"), ("strict FP32", "avae_notf32")] {
        let mut line = format!("waveform vs ComfyUI ({label}):");
        for s in 0..2 {
            let want = read_f32(&reference.join(dir).join(format!("final_s{s}.f32")));
            let stats = compare(&wave[s * l..(s + 1) * l], &want);
            line.push_str(&format!(
                "  ch{s} rel_l2 {:.2e} max_abs {:.2e} (ref rms {:.2e})",
                stats.rel_l2, stats.max_abs, stats.ref_rms
            ));
        }
        eprintln!("{line}");
    }
    let mut times = Vec::new();
    for _ in 0..5 {
        let started = std::time::Instant::now();
        let again = decoder.decode(&latent, frames).unwrap();
        times.push((started.elapsed().as_secs_f64() * 1000.0).round() / 1000.0);
        assert_eq!(again, wave, "the decode must be deterministic");
    }
    eprintln!("audio decode seconds per pass (eager): {times:?}");
    let want0 = read_f32(&reference.join("avae_notf32").join("final_s0.f32"));
    assert!(
        compare(&wave[..l], &want0).rel_l2 < 0.05,
        "audio waveform is not close to ComfyUI's"
    );
}

/// Attribution of the end-to-end pixel gap: ComfyUI's own noise and text context go through APXInf's sampler and then APXInf's video decoder,
/// and the frames are compared with ComfyUI's decode of ComfyUI's own final latents. Whatever remains is the DiT and decoder; the rest of the
/// gap of `prompt_to_latents_matches_comfyui` is the text context.
#[test]
#[ignore = "requires the DiT, the video VAE, a ComfyUI trace export and decode reference, and a CUDA GPU"]
fn pixels_from_comfyui_context_match_comfyui() {
    use crate::cuda_minimax_h3::model::blocks::VideoVaeBlocks;
    use crate::cuda_minimax_h3::model_runner::{PlanKey, PreparedDenoise, VideoDecoder};
    use crate::cuda_minimax_h3::weights::VideoVaeWeights;
    let dit = PathBuf::from(std::env::var("APXINF_H3_DIT").expect("APXINF_H3_DIT"));
    let vae = PathBuf::from(std::env::var("APXINF_H3_VAE").expect("APXINF_H3_VAE"));
    let trace = PathBuf::from(std::env::var("APXINF_H3_TRACE").expect("APXINF_H3_TRACE"));
    let reference = PathBuf::from(std::env::var("APXINF_H3_VAE_REF").expect("APXINF_H3_VAE_REF"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(trace.join("meta.json")).unwrap()).unwrap();
    let text_rows = meta["text_rows"].as_u64().unwrap() as usize;
    let latent: Vec<usize> = meta["latent"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let (t, h, w) = (latent[0], latent[1], latent[2]);
    let audio_t = meta["audio_t"].as_u64().unwrap() as usize;
    let steps = meta["steps"].as_u64().unwrap() as usize;
    let layout = Layout::new(text_rows, t, h, w, audio_t).unwrap();
    let ctx = Rc::new(CudaContext::new(0).unwrap());
    let weights = Rc::new(DitWeights::load(&dit, &ctx).unwrap());
    let plan = PreparedDenoise::new(
        ctx.clone(),
        weights,
        PlanKey { layout, steps },
        true,
        GemmPolicy::default(),
        AttentionPolicy::default(),
    )
    .unwrap();
    let text = upload(
        &ctx,
        &read_f32(&trace.join("context.f32")),
        vec![text_rows, HIDDEN],
        DType::BF16,
    );
    plan.bind(
        &text,
        &host::patchify_video(&read_f32(&trace.join("noise_video.f32")), t, h, w),
        &host::pack_audio(&read_f32(&trace.join("noise_audio.f32")), audio_t),
    )
    .unwrap();
    for step in 0..steps {
        plan.step(step).unwrap();
    }
    let (video_rows, _) = plan.latents().unwrap();
    let video_latent = host::unpatchify_video(&video_rows, t, h, w);
    // the final latents themselves, against ComfyUI's
    let latent_stats = compare(&video_latent, &read_f32(&trace.join("final_video.f32")));
    drop(plan); // the decoder needs the arena and the graph's memory
    let video_weights = Rc::new(VideoVaeWeights::load(&vae, ctx.clone()).unwrap());
    let decoder = VideoDecoder::new(
        ctx.clone(),
        VideoVaeBlocks::new(ctx.clone(), video_weights),
        crate::cuda_minimax_h3::model_runner::TILE_BATCH,
        GemmPolicy::default(),
        AttentionPolicy::default(),
    );
    let frames = decoder.decode(&video_latent, [t, h, w]).unwrap();
    let want = read_f32(&reference.join("video_pixels.f32"));
    assert_eq!(frames.data.len(), want.len());
    // ComfyUI's SaveVideo path: (x * 255).clamp(0, 255).byte() truncates
    let to_byte = |v: f32| (v * 255.0).clamp(0.0, 255.0) as u8 as f64;
    let mean_abs_8bit = frames
        .data
        .iter()
        .zip(&want)
        .map(|(a, b)| (to_byte(*a) - to_byte(*b)).abs())
        .sum::<f64>()
        / want.len() as f64;
    let pixels = compare(&frames.data, &want);
    eprintln!(
        "ComfyUI context + noise through APXInf: latents rel_l2 {:.4} | pixels rel_l2 {:.4} cos {:.5} | mean abs {:.2} / 255 after 8-bit conversion",
        latent_stats.rel_l2, pixels.rel_l2, pixels.cosine, mean_abs_8bit
    );
}

/// Where the 3.3 % gap of the refined text context comes from. Two separate checks against ComfyUI's captures
/// (`scripts/capture_comfy_te.py --layers 0,1,24,49`, and `context.f32` of the trace export):
///  1. the streamed Qwen3 stack layer by layer (ComfyUI runs it in FP32 over the same INT8 weights);
///  2. the token refiner fed ComfyUI's OWN encoder output, against ComfyUI's refined context.
///
///   APXINF_H3_TE=/path/qwen3vl_32b_minimax_h3_int8_convrot.safetensors APXINF_H3_TE_TRACE=/path/te_fox \
///   APXINF_H3_DIT=... APXINF_H3_TRACE=/path/export
#[test]
#[ignore = "requires the 27 GB text-encoder checkpoint, the DiT, ComfyUI captures and a CUDA GPU"]
fn text_path_matches_comfyui_in_two_halves() {
    use crate::cuda_minimax_h3::model::blocks::{refine_text, TextBlocks};
    use crate::cuda_minimax_h3::weights::{Pinning, TextWeights};
    let te = PathBuf::from(std::env::var("APXINF_H3_TE").expect("APXINF_H3_TE"));
    let te_trace = PathBuf::from(std::env::var("APXINF_H3_TE_TRACE").expect("APXINF_H3_TE_TRACE"));
    let dit = PathBuf::from(std::env::var("APXINF_H3_DIT").expect("APXINF_H3_DIT"));
    let trace = PathBuf::from(std::env::var("APXINF_H3_TRACE").expect("APXINF_H3_TRACE"));
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(te_trace.join("te_meta.json")).unwrap()).unwrap();
    let ids: Vec<u32> = meta["ids"][0]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    let real = ids.len();
    let padded = real.div_ceil(TE_ROW_BUCKET) * TE_ROW_BUCKET;
    let ctx = Rc::new(CudaContext::new(0).unwrap());
    let (gemm, attention) = (GemmPolicy::default(), AttentionPolicy::default());
    let blocks = TextBlocks::new(ctx.clone(), padded).unwrap();
    let weights = TextWeights::open(&te, ctx.clone(), Pinning::from_env()).unwrap();
    let embeddings = weights.embed(&ids).unwrap();
    let embedded: Vec<f32> = embeddings.iter().map(|v| v.to_f32()).collect();
    let s = compare(&embedded, &read_f32(&te_trace.join("embed_out.f32")));
    eprintln!("embedding rows: rel_l2 {:.2e}", s.rel_l2);
    let x = blocks.embed(&embeddings, real).unwrap();
    let mut layer_stats = Vec::new();
    weights
        .run_layers(|layer, device| {
            blocks.layer(device, &x, &gemm, &attention)?;
            let file = te_trace.join(format!("layer{layer}_out.f32"));
            if file.exists() || layer + 1 == TE_LAYERS {
                ctx.synchronize().map_err(apxinf_core::Error::Cuda)?;
                let all = download_bf16(&x);
                let theirs = read_f32(&if file.exists() {
                    file
                } else {
                    te_trace.join("te_out.f32")
                });
                layer_stats.push((layer, compare(&all[..real * TE_DIM], &theirs)));
            }
            Ok(())
        })
        .unwrap();
    for (layer, s) in &layer_stats {
        eprintln!(
            "text layer {layer:2}: rel_l2 {:.4} cos {:.5} max_abs {:.3} (ref rms {:.3})",
            s.rel_l2, s.cosine, s.max_abs, s.ref_rms
        );
    }
    // half 2: the refiner on ComfyUI's own encoder output
    let dit_weights = crate::cuda_minimax_h3::weights::DitWeights::load(&dit, &ctx).unwrap();
    let states = upload(
        &ctx,
        &read_f32(&te_trace.join("te_out.f32")),
        vec![real, TE_DIM],
        DType::BF16,
    );
    let refined = refine_text(&ctx, &dit_weights, &states, &gemm, &attention).unwrap();
    ctx.synchronize().unwrap();
    let ours = download_bf16(&refined);
    let theirs = read_f32(&trace.join("context.f32"));
    assert_eq!(ours.len(), theirs.len(), "refiner output rows");
    let s = compare(&ours, &theirs);
    eprintln!("refiner on ComfyUI's encoder output vs ComfyUI's context: rel_l2 {:.4} cos {:.5} max_abs {:.3} (ref rms {:.3})", s.rel_l2, s.cosine, s.max_abs, s.ref_rms);
}

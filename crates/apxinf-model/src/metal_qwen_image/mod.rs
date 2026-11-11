//! Native Qwen-Image-2.1 text-to-image on Apple Metal, following the mflux
//! `QwenImage21` txt2img path (MIT) with its `--low-ram` lifecycle: the
//! text encoder is read one layer at a time, and the DiT is released before
//! VAE decoding and read again by the next request. No Python execution.
//!
//! The checkpoint is the mflux 4-bit export (affine Q4 DiT, BF16 text
//! encoder, F32 VAE). Its weights are under the Qwen Research License; this
//! family ships no weights.
mod dit;
mod ops;
mod prompt_cache;
mod tables;
mod text;
mod vae;
mod weights;

use crate::image::{GeneratedImage, ImageRequest, ImageRuntime};
use crate::{LoadOptions, LoadedModel, ModelPrecision};
use apxinf_core::{Backend, Device, Error, Result};
use apxinf_mlx::fusions::MppBf16GemmNt;
use apxinf_mlx::{Array, MlxBackend, MlxDType, Stream};
use apxinf_tokenizer::Tokenizer;
use ops::Ops;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use weights::{Decoder, TextEncoderFiles, Transformer, LATENT, MLP};

/// The only schedule this family has validated: 40 linear steps, guidance 1.
pub const STEPS: usize = 40;
/// Pixels per latent; the RoPE tables cover 2048 latent rows or columns.
const SCALE: usize = 16;
const MAX_SIDE: usize = 2048 * SCALE;
const MAX_IMAGE_TOKENS: usize = (i32::MAX as usize + 1) / MLP;

pub struct QwenImage {
    dir: PathBuf,
    tokenizer: Tokenizer,
    text: TextEncoderFiles,
    /// Loaded until the first VAE decode, then read again per request.
    transformer: Option<Transformer>,
    decoder: Decoder,
    ops: Ops,
    /// Present in auto mode when the synthetic MPP probe matches quantized matmul.
    gemm: Option<Rc<MppBf16GemmNt>>,
    /// `Qwen21Rope` cos/sin tables for the 16- and 56-wide axes.
    rope16: (Vec<f32>, Vec<f32>),
    rope56: (Vec<f32>, Vec<f32>),
    prompt_cache: Option<prompt_cache::PromptCache>,
}

/// The prompt as the DiT consumes it: text-encoder states to project, or the
/// projected per-block K/V from the prompt cache.
enum Prompt {
    Encoded(Array),
    Cached(dit::TextCache),
}

/// Device-independent option admission, checked before the checkpoint is read.
fn admit(options: &LoadOptions) -> Result<()> {
    if !matches!(
        options.model_variant.as_deref(),
        None | Some("auto" | "reference")
    ) || !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16
    ) || options.text_weight_dtype.is_some()
        || options.config.is_some()
        || options.synthetic.is_some()
        || options.calibration_path.is_some()
        || options.tuning_path.is_some()
        || options.autotune
        || options.uniform_fp8_scale.is_some()
        || !options.assets.is_empty()
    {
        return Err(Error::Contract(
            "Qwen-Image MLX loads the mflux 4-bit checkpoint as stored; only auto and reference \
             variants are supported, without CUDA tuning, calibration, synthetic weights or auxiliary assets",
        ));
    }
    Ok(())
}

pub fn load(
    path: &Path,
    device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if !matches!(device, Device::Metal(_)) {
        return Err(Error::UnsupportedDevice(device));
    }
    admit(options)?;
    let native = backend
        .as_any()
        .downcast_ref::<MlxBackend>()
        .ok_or(Error::Contract(
            "Qwen-Image requires the native MLX backend",
        ))?
        .clone();
    if native.device() != device {
        return Err(Error::DeviceMismatch {
            expected: device,
            got: native.device(),
        });
    }
    if !path.is_dir() {
        return Err(Error::Contract("Qwen-Image loads a checkpoint directory"));
    }
    let stream = native.stream().clone();
    let tokenizer = Tokenizer::from_file(path.join("processor/tokenizer.json"))?;
    let text = TextEncoderFiles::open(&path.join("text_encoder"))?;
    let transformer = Transformer::load(&stream, &path.join("transformer"))?;
    let decoder = Decoder::load(&stream, &path.join("vae"))?;
    let ops = Ops::new(&stream)?;
    let gemm = if options.model_variant.as_deref() == Some("reference") {
        None
    } else {
        probed_gemm(&stream).map(Rc::new)
    };
    let prompt_cache = options
        .prompt_cache
        .as_ref()
        .map(|cache| prompt_cache::PromptCache::open(cache, path, &stream, gemm.is_some()))
        .transpose()?;
    Ok(LoadedModel::Image(Box::new(QwenImage {
        dir: path.to_path_buf(),
        tokenizer,
        text,
        transformer: Some(transformer),
        decoder,
        ops,
        gemm,
        rope16: tables::rope_axis(16, 10000.0),
        rope56: tables::rope_axis(56, 10000.0),
        prompt_cache,
    })))
}

/// MLX architectures where MPP image projections were measured faster than
/// quantized matmul: the M4 (paired whole-image timing). Other architectures keep quantized matmul until
/// measured there; the selection is structural, so it stays in this family.
const MPP_MEASURED_ARCHITECTURES: &[&str] = &["applegpu_g16g"];

/// Enables MPP on a measured architecture after a bit-identical probe of the
/// three block projection shapes `(K, N)` at `dit::MPP_MIN_TOKENS` rows, on
/// seeded device data. This does not qualify checkpoint weights, larger token
/// counts or image outputs.
fn probed_gemm(stream: &Stream) -> Option<MppBf16GemmNt> {
    let architecture = stream
        .metal_caps()
        .map_or("", |caps| caps.architecture.as_str());
    if !MPP_MEASURED_ARCHITECTURES.contains(&architecture) {
        eprintln!(
            "Qwen-Image: MPP GEMM not used (not measured on {architecture:?}); image projections use quantized matmul"
        );
        return None;
    }
    let probe = || -> Result<Option<MppBf16GemmNt>> {
        let gemm = MppBf16GemmNt::new(stream)?;
        let normal = |shape: &[usize], seed: u64, scale: f32| -> Result<Array> {
            Array::random_normal(shape, MlxDType::F32, &Array::random_key(stream, seed)?)?
                .mul(&Array::scalar(stream, scale, MlxDType::F32)?)?
                .cast(MlxDType::BF16)
        };
        let m = dit::MPP_MIN_TOKENS;
        for (seed, (k, n)) in [(4096, 4096), (4096, MLP), (MLP, 4096)]
            .into_iter()
            .enumerate()
        {
            let [q, scales, biases] = normal(&[n, k], 2 * seed as u64, 0.02)?.quantize(64, 4)?;
            let x = normal(&[1, m, k], 2 * seed as u64 + 1, 1.0)?;
            let reference = x.quantized_matmul(&q, &scales, &biases, true, 64, 4)?;
            let out = gemm.call(&x, &q.dequantize(&scales, &biases, 64, 4)?)?;
            if out.to_bytes()? != reference.to_bytes()? {
                return Ok(None);
            }
        }
        Ok(Some(gemm))
    };
    match probe() {
        Ok(Some(gemm)) => Some(gemm),
        outcome => {
            let reason = match outcome {
                Err(e) => e.to_string(),
                _ => "results differ from quantized matmul".to_owned(),
            };
            eprintln!(
                "Qwen-Image: MPP GEMM not used ({reason}); image projections use quantized matmul"
            );
            None
        }
    }
}

/// Request admission, before any device work.
fn admit_request(request: &ImageRequest<'_>) -> Result<()> {
    if request.prompt.trim().is_empty() {
        return Err(Error::Contract("Qwen-Image needs a nonempty prompt"));
    }
    for side in [request.width, request.height] {
        if side == 0 || side % SCALE != 0 || side > MAX_SIDE {
            return Err(Error::Contract(
                "Qwen-Image width and height must be positive multiples of 16, at most 32768",
            ));
        }
    }
    (request.width / SCALE)
        .checked_mul(request.height / SCALE)
        .filter(|&tokens| tokens <= MAX_IMAGE_TOKENS)
        .ok_or(Error::Contract(
            "Qwen-Image area exceeds the MLP flattened i32 execution bound",
        ))?;
    if request.steps != STEPS {
        return Err(Error::Contract(
            "Qwen-Image supports exactly 40 denoising steps",
        ));
    }
    Ok(())
}

/// `LinearScheduler` sigmas with the Qwen-Image-2.1 shift and terminal
/// rescale, computed on the device as MLX computes them: FP32 `[steps + 1]`.
fn sigmas(ops: &Ops, width: usize, height: usize, steps: usize) -> Result<Array> {
    let s = ops.stream();
    let int = |v: usize| Array::from_i32(s, &[], &[v as i32]);
    // mx.linspace(1.0, 1 / steps, steps): t = arange / (n - 1); (1 - t) * a + t * b.
    let t = Array::arange(s, 0.0, steps as f32, 1.0, MlxDType::I32)?.div(&int(steps - 1)?)?;
    let linear = int(1)?
        .sub(&t)?
        .mul(&ops.f32(1.0)?)?
        .add(&t.mul(&ops.f32((1.0 / steps as f64) as f32)?)?)?;
    // shift_sigmas: mu from the image area, then the 0.02 terminal rescale.
    let m = (0.9 - 0.5) / (8192.0 - 256.0);
    let b = 0.5 - m * 256.0;
    let mu = m * width as f64 * height as f64 / 256.0 + b;
    let e = ops.f32(mu as f32)?.exp()?;
    let shifted = e.div(&e.add(&ops.f32(1.0)?.div(&linear)?.sub(&ops.f32(1.0)?)?)?)?;
    let one_minus = ops.f32(1.0)?.sub(&shifted)?;
    let scale = one_minus
        .slice(&[steps - 1], &[steps])?
        .reshape(&[])?
        .div(&ops.f32((1.0 - 0.02f64) as f32)?)?;
    let shifted = ops.f32(1.0)?.sub(&one_minus.div(&scale)?)?;
    let sigmas = Array::concat(&[&shifted, &Array::zeros(s, &[1], MlxDType::F32)?], 0)?;
    sigmas.eval()?;
    Ok(sigmas)
}

/// `Qwen21Rope` rows for `[text | image]`: FP32 `[rows, 64]` cos and sin.
/// Text advances one position on all axes; the image fixes the frame axis at
/// the text length and centers its height/width grid on zero.
fn rope(
    rope16: &(Vec<f32>, Vec<f32>),
    rope56: &(Vec<f32>, Vec<f32>),
    text: usize,
    lh: usize,
    lw: usize,
) -> (Vec<f32>, Vec<f32>) {
    let rows = text + lh * lw;
    let (mut cos, mut sin) = (Vec::with_capacity(rows * 64), Vec::with_capacity(rows * 64));
    let mut push = |position: [i64; 3]| {
        for (table, width, p) in [
            (rope16, 8, position[0]),
            (rope56, 28, position[1]),
            (rope56, 28, position[2]),
        ] {
            let r = tables::rope_row(p).expect("admitted sizes stay inside the RoPE tables");
            cos.extend_from_slice(&table.0[r * width..(r + 1) * width]);
            sin.extend_from_slice(&table.1[r * width..(r + 1) * width]);
        }
    };
    for p in 0..text as i64 {
        push([p; 3]);
    }
    let (top, left) = (-((lh - lh / 2) as i64), -((lw - lw / 2) as i64));
    for i in 0..lh as i64 {
        for j in 0..lw as i64 {
            push([text as i64, top + i, left + j]);
        }
    }
    (cos, sin)
}

impl ImageRuntime for QwenImage {
    fn generate(&mut self, request: &ImageRequest<'_>) -> Result<GeneratedImage> {
        admit_request(request)?;
        let ops = &self.ops;
        let stream = ops.stream();
        let (width, height) = (request.width, request.height);
        let (lh, lw) = (height / SCALE, width / SCALE);
        let tokens = text::tokenize(&self.tokenizer, request.prompt)?;
        let cached = self
            .prompt_cache
            .as_ref()
            .and_then(|cache| cache.load(&tokens, stream));
        let prompt = match cached {
            Some(cache) => Prompt::Cached(cache),
            None => {
                let prompt = text::encode(&self.text, &tokens, ops)?;
                // The text encoder layers are freed; release their cached buffers as
                // mflux --low-ram does after deleting the encoder.
                apxinf_mlx::clear_cache()?;
                Prompt::Encoded(prompt)
            }
        };
        let key = Array::random_key(stream, request.seed)?;
        let mut latents = Array::random_normal(&[1, lh * lw, LATENT], MlxDType::F32, &key)?
            .cast(MlxDType::BF16)?;
        let sigmas = sigmas(ops, width, height, request.steps)?;
        let text_len = tokens.ids.len() - tokens.drop;
        let (cos, sin) = rope(&self.rope16, &self.rope56, text_len, lh, lw);
        let split = text_len * 64;
        let table = |v: &[f32], rows: usize| Array::from_f32(stream, &[rows, 64], v);
        let transformer = match self.transformer.take() {
            Some(t) => t,
            None => Transformer::load(stream, &self.dir.join("transformer"))?,
        };
        let dit = dit::Dit::new(transformer, ops, self.gemm.clone())?;
        let cache = match prompt {
            Prompt::Cached(cache) => cache,
            Prompt::Encoded(prompt) => {
                let cache = dit.text_cache(
                    &prompt,
                    &table(&cos[..split], text_len)?,
                    &table(&sin[..split], text_len)?,
                )?;
                drop(prompt);
                if let Some(prompt_cache) = &self.prompt_cache {
                    prompt_cache.store(&tokens, &cache);
                }
                cache
            }
        };
        let (cos, sin) = (
            table(&cos[split..], lh * lw)?,
            table(&sin[split..], lh * lw)?,
        );
        latents.eval()?;
        let sigma = |i: usize| sigmas.slice(&[i], &[i + 1]);
        let (step, first) = dit::Step::prepare(
            dit,
            &dit::Step::inputs(&latents, &sigma(0)?, &cos, &sin, &cache),
        )?;
        let mut first = Some(first);
        for t in 0..request.steps {
            let velocity = match first.take() {
                Some(velocity) => velocity,
                None => step.call(&dit::Step::inputs(&latents, &sigma(t)?, &cos, &sin, &cache))?,
            };
            let dt = sigma(t + 1)?
                .sub(&sigma(t)?)?
                .reshape(&[])?
                .cast(MlxDType::BF16)?;
            latents = latents.add(&velocity.cast(MlxDType::BF16)?.mul(&dt)?)?;
            latents.eval()?;
        }
        // Release the DiT and its cached buffers before decoding, as mflux
        // --low-ram does.
        drop((cache, step));
        apxinf_mlx::clear_cache()?;
        let rgb = vae::decode(&self.decoder, ops, &latents, width, height)?;
        Ok(GeneratedImage { width, height, rgb })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_variant_admission_needs_no_device() {
        for variant in [None, Some("auto"), Some("reference")] {
            assert!(
                admit(&LoadOptions {
                    model_variant: variant.map(str::to_owned),
                    ..LoadOptions::default()
                })
                .is_ok(),
                "rejected {variant:?}"
            );
        }
    }

    #[test]
    fn request_execution_bounds_need_no_device() {
        let request = ImageRequest {
            prompt: "a red bicycle",
            width: 1024,
            height: 1024,
            seed: 7,
            steps: STEPS,
        };
        for (width, height) in [
            (16, 16),
            (512, 512),
            (1024, 576),
            (1024, 1024),
            (32768, 16),
            (16, 32768),
            (1824, 24528),
            (24528, 1824),
            (26880, 1664),
        ] {
            assert!(
                admit_request(&ImageRequest {
                    width,
                    height,
                    ..request
                })
                .is_ok(),
                "rejected {width}x{height}"
            );
        }
        for (width, height) in [
            (8192, 8192),
            (16384, 16384),
            (32768, 32768),
            (1824, 24544),
            (24544, 1824),
            (26880, 1680),
            (usize::MAX, usize::MAX),
            (usize::MAX - 15, 16),
            (16, usize::MAX - 15),
        ] {
            assert!(
                admit_request(&ImageRequest {
                    width,
                    height,
                    ..request
                })
                .is_err(),
                "accepted execution-invalid {width}x{height}"
            );
        }
    }

    #[test]
    fn admitted_schedules_remain_finite_and_decreasing_on_cpu() {
        let stream = Stream::cpu().unwrap();
        let ops = Ops::new(&stream).unwrap();
        for (width, height) in [
            (16, 16),
            (512, 512),
            (1024, 576),
            (1024, 1024),
            (1824, 24528),
            (26880, 1664),
        ] {
            let schedule = sigmas(&ops, width, height, STEPS)
                .unwrap()
                .to_f32_vec()
                .unwrap();
            assert_eq!(schedule.len(), STEPS + 1);
            assert_eq!(schedule[0], 1.0);
            assert!((schedule[STEPS - 1] - 0.02).abs() < 1e-6);
            assert_eq!(schedule[STEPS], 0.0);
            assert!(schedule.iter().all(|sigma| sigma.is_finite()));
            assert!(
                schedule.windows(2).all(|pair| pair[0] > pair[1]),
                "nondecreasing schedule for {width}x{height}: {schedule:?}"
            );
        }
    }

    #[test]
    fn admission_needs_no_device_or_checkpoint() {
        assert!(admit(&LoadOptions::default()).is_ok());
        for options in [
            LoadOptions {
                model_variant: Some("bf16".into()),
                ..LoadOptions::default()
            },
            LoadOptions {
                autotune: true,
                ..LoadOptions::default()
            },
        ] {
            assert!(admit(&options).is_err());
        }
        let request = ImageRequest {
            prompt: "a red bicycle",
            width: 1024,
            height: 576,
            seed: 7,
            steps: STEPS,
        };
        assert!(admit_request(&request).is_ok());
        for bad in [
            ImageRequest {
                prompt: " \n",
                ..request
            },
            ImageRequest {
                width: 1000,
                ..request
            },
            ImageRequest {
                height: 0,
                ..request
            },
            ImageRequest {
                width: MAX_SIDE + 16,
                ..request
            },
            ImageRequest {
                steps: 20,
                ..request
            },
        ] {
            assert!(admit_request(&bad).is_err());
        }
    }

    #[test]
    #[ignore = "requires exclusive Apple Silicon Metal test slot and pinned MLX SDK"]
    fn mpp_gemm_is_probed_only_on_measured_architectures() {
        let stream = Stream::metal(0).unwrap();
        let architecture = stream.metal_caps().unwrap().architecture.clone();
        let measured = MPP_MEASURED_ARCHITECTURES.contains(&architecture.as_str());
        assert_eq!(probed_gemm(&stream).is_some(), measured, "{architecture}");
    }

    #[test]
    #[ignore = "requires the pinned Qwen-Image checkpoint and an exclusive Apple Silicon Metal slot"]
    fn prompt_cache_hits_give_the_image_of_the_uncached_path() {
        let checkpoint = PathBuf::from(
            std::env::var("APXINF_QWEN_IMAGE_CHECKPOINT")
                .expect("set APXINF_QWEN_IMAGE_CHECKPOINT"),
        );
        let dir =
            std::env::temp_dir().join(format!("apxinf-qwen-image-cache-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let backend = crate::accelerator::create_backend(Device::Metal(0)).unwrap();
        let load = |prompt_cache| {
            let options = LoadOptions {
                prompt_cache,
                ..LoadOptions::default()
            };
            match super::load(&checkpoint, Device::Metal(0), backend.clone(), &options).unwrap() {
                LoadedModel::Image(model) => model,
                _ => unreachable!("Qwen-Image loads an image model"),
            }
        };
        let cached = Some(crate::PromptCacheOptions {
            dir: dir.clone(),
            max_bytes: std::num::NonZeroU64::new(1 << 30).unwrap(),
        });
        let request = ImageRequest {
            prompt: "a red bicycle",
            width: 64,
            height: 64,
            seed: 7,
            steps: STEPS,
        };
        let uncached = load(None).generate(&request).unwrap();
        let entries = || -> Vec<_> {
            std::fs::read_dir(&dir)
                .unwrap()
                .map(|item| item.unwrap().path())
                .collect()
        };
        let mut model = load(cached.clone());
        assert_eq!(model.generate(&request).unwrap(), uncached, "miss");
        let [entry] = entries().try_into().expect("one stored entry");
        let stored = std::fs::metadata(&entry).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        assert_eq!(model.generate(&request).unwrap(), uncached, "hit");
        assert_eq!(
            load(cached).generate(&request).unwrap(),
            uncached,
            "hit after reload"
        );
        assert_eq!(entries(), std::slice::from_ref(&entry));
        assert!(
            std::fs::metadata(&entry).unwrap().modified().unwrap() > stored,
            "hit not recorded"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rope_positions_center_the_image_grid() {
        let tables = |dim| tables::rope_axis(dim, 10000.0);
        let (r16, r56) = (tables(16), tables(56));
        let (cos, _) = rope(&r16, &r56, 3, 2, 3);
        assert_eq!(cos.len(), (3 + 6) * 64);
        let row = |position: [i64; 3]| -> Vec<f32> {
            let mut out = Vec::new();
            for (t, w, p) in [
                (&r16, 8, position[0]),
                (&r56, 28, position[1]),
                (&r56, 28, position[2]),
            ] {
                let r = tables::rope_row(p).unwrap();
                out.extend_from_slice(&t.0[r * w..(r + 1) * w]);
            }
            out
        };
        assert_eq!(cos[2 * 64..3 * 64], row([2, 2, 2]));
        // First image token: frame 3, height -1, width -2; last: height 0, width 0.
        assert_eq!(cos[3 * 64..4 * 64], row([3, -1, -2]));
        assert_eq!(cos[8 * 64..9 * 64], row([3, 0, 0]));
    }
}

use apxinf_core::{Error, Result};
use serde::Deserialize;

/// Geometry of the released MiniMax-H3-Image DiT. Every other module takes its
/// dimensions from here, and `DitMetadata::validate` plus the weight loader reject any checkpoint that
/// disagrees, so a different model never runs under these shapes by accident.
pub(crate) const HIDDEN: usize = 5376;
pub(crate) const BLOCKS: usize = 50;
pub(crate) const HEADS: usize = 56;
pub(crate) const HEAD_DIM: usize = 128;
pub(crate) const INNER: usize = HEADS * HEAD_DIM;
pub(crate) const FFN: usize = 14336;
pub(crate) const TEXT_DIM: usize = 5120;
pub(crate) const REFINER_BLOCKS: usize = 2;
pub(crate) const LATENT_CHANNELS: usize = 24;
pub(crate) const PATCH: usize = 2;
pub(crate) const PATCH_DIM: usize = LATENT_CHANNELS * PATCH * PATCH;
/// Pixels per latent pixel along each side.
pub(crate) const VAE_RATIO: usize = 16;
/// Canvas sides are multiples of this (latent patch of 2 times the VAE ratio).
pub(crate) const CANVAS_MULTIPLE: usize = PATCH * VAE_RATIO;
/// Distilled sampling steps: one output head and one forward per step.
pub const PDD_STEPS: usize = 6;
/// Rotated channels of every head are `2 * 3 * ROPE_FREQS = 96`: three axes of `ROPE_FREQS` frequencies, duplicated.
/// The frequencies themselves (theta 10000) come from the checkpoint's `rope.inv_freq` buffer.
pub(crate) const ROPE_FREQS: usize = 16;
pub(crate) const NORM_EPS: f32 = 1e-5;
/// Rows of the AdaLN curve table and its rank.
pub(crate) const ADALN_GRID: usize = 1025;
pub(crate) const ADALN_RANK: usize = 8;
/// Modalities that address AdaLN rows: image 0, text 1 (audio 2 is video-only).
pub(crate) const MODALITIES: usize = 3;
pub(crate) const MODALITY_IMAGE: usize = 0;
pub(crate) const MODALITY_TEXT: usize = 1;
/// ConvRot group size shared by activation rotation and the weight files.
pub(crate) const CONVROT_GROUP: usize = 256;

/// The six PDD sigmas of the released checkpoint (`pdd_config.json`). `sigmas[i]`
/// to `sigmas[i + 1]` is step `i`; the model sees `t = 1 - sigmas[i]`.
pub(crate) const PDD_SIGMAS: [f32; PDD_STEPS + 1] = [
    1.0,
    0.967_523_7,
    0.887_548_55,
    0.75,
    0.532_773_9,
    0.232_008_87,
    0.0,
];

/// Checkpoint-level metadata read from the DiT file's `__metadata__.config`.
#[derive(Clone, Debug, Deserialize, PartialEq)]
pub(crate) struct DitMetadata {
    pub pdd_steps: usize,
    pub pdd_sigmas: Vec<f32>,
}

#[derive(Deserialize)]
struct MetadataEnvelope {
    transformer: DitMetadata,
}

impl DitMetadata {
    /// Parse `{"transformer": {"pdd_steps": 6, "pdd_sigmas": [...]}}`.
    pub fn parse(json: &str) -> Result<Self> {
        let envelope: MetadataEnvelope = serde_json::from_str(json)
            .map_err(|e| Error::Other(format!("MiniMax-H3-Image DiT metadata: {e}")))?;
        let metadata = envelope.transformer;
        metadata.validate()?;
        Ok(metadata)
    }

    pub fn validate(&self) -> Result<()> {
        if self.pdd_steps != PDD_STEPS
            || self.pdd_sigmas.len() != PDD_STEPS + 1
            || self
                .pdd_sigmas
                .iter()
                .zip(PDD_SIGMAS)
                .any(|(a, b)| (a - b).abs() > 1e-6)
        {
            return Err(Error::Other(
                "MiniMax-H3-Image checkpoint is not the 6-step PDD release this family supports"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Segment layout of one packed sequence: `[text | image]`. Text rows come first and share the image
/// timestep; rows differ only in the modality tag (image 0, text 1) that addresses AdaLN rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub text_rows: usize,
    pub image_rows: usize,
}

impl Layout {
    pub fn new(text_rows: usize, canvas: Canvas) -> Result<Self> {
        if text_rows == 0 || text_rows > MAX_TEXT_TOKENS {
            return Err(Error::Other(format!(
                "MiniMax-H3-Image needs 1..={MAX_TEXT_TOKENS} text tokens, got {text_rows}"
            )));
        }
        Ok(Self {
            text_rows,
            image_rows: canvas.rows(),
        })
    }

    pub fn total_rows(&self) -> usize {
        self.text_rows + self.image_rows
    }
}

/// Text-to-image geometry for one request: the canvas the DiT generates on and
/// the number of latent patches. The released pipeline never generates below
/// `min_generation_area` pixels; a smaller request is generated on a larger canvas
/// and resized back, which this family does not implement, so such sizes are rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Canvas {
    pub width: usize,
    pub height: usize,
}

impl Canvas {
    pub fn new(width: usize, height: usize) -> Result<Self> {
        if width == 0
            || height == 0
            || width % CANVAS_MULTIPLE != 0
            || height % CANVAS_MULTIPLE != 0
        {
            return Err(Error::Other(format!(
                "MiniMax-H3-Image sizes must be positive multiples of {CANVAS_MULTIPLE}, got {width}x{height}"
            )));
        }
        if width < VAE_TILE_PIXELS || height < VAE_TILE_PIXELS {
            return Err(Error::Other(format!(
                "MiniMax-H3-Image sides must be at least {VAE_TILE_PIXELS} pixels (one VAE tile), got {width}x{height}"
            )));
        }
        let area = width
            .checked_mul(height)
            .ok_or_else(|| Error::Other("image size overflow".into()))?;
        if !(MIN_AREA..=MAX_AREA).contains(&area) {
            return Err(Error::Other(format!(
                "MiniMax-H3-Image supports {MIN_AREA}..={MAX_AREA} pixels, got {width}x{height}"
            )));
        }
        Ok(Self { width, height })
    }

    pub fn latent_width(&self) -> usize {
        self.width / VAE_RATIO
    }

    pub fn latent_height(&self) -> usize {
        self.height / VAE_RATIO
    }

    /// Patch grid `(rows, cols)` of the generated image.
    pub fn grid(&self) -> (usize, usize) {
        (self.latent_height() / PATCH, self.latent_width() / PATCH)
    }

    pub fn rows(&self) -> usize {
        let (h, w) = self.grid();
        h * w
    }
}

/// The model is tuned for 2048x2048 and is never asked to generate less than 1024^2.
pub(crate) const MIN_AREA: usize = 1024 * 1024;
pub(crate) const MAX_AREA: usize = 4096 * 4096;

/// Longest prompt this family accepts, in tokens. The released pipelines impose no limit (the tokenizer allows 131072);
/// this is a deployment bound chosen to keep the text stage's activations and the DiT's packed sequence small, and a
/// longer prompt is rejected before any device work instead of being truncated.
pub(crate) const MAX_TEXT_TOKENS: usize = 512;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metadata_matches_the_released_schedule() {
        let json = r#"{"transformer": {"pdd_steps": 6, "pdd_sigmas": [1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0]}}"#;
        let metadata = DitMetadata::parse(json).unwrap();
        assert_eq!(metadata.pdd_steps, PDD_STEPS);
        let altered = json.replace("0.75", "0.76");
        assert!(DitMetadata::parse(&altered).is_err());
        let fewer =
            r#"{"transformer": {"pdd_steps": 4, "pdd_sigmas": [1.0, 0.75, 0.5, 0.25, 0.0]}}"#;
        assert!(DitMetadata::parse(fewer).is_err());
    }

    #[test]
    fn canvas_rejects_unsupported_sizes_before_device_work() {
        assert_eq!(Canvas::new(1024, 1024).unwrap().rows(), 32 * 32);
        assert_eq!(Canvas::new(2048, 2048).unwrap().rows(), 64 * 64);
        assert_eq!(Canvas::new(2048, 1024).unwrap().grid(), (32, 64));
        assert!(Canvas::new(1000, 1024).is_err(), "not a multiple of 32");
        assert!(Canvas::new(512, 512).is_err(), "below the generation area");
        assert!(Canvas::new(8192, 4096).is_err(), "above the supported area");
        assert!(Canvas::new(0, 1024).is_err());
        assert!(
            Canvas::new(224, 8192).is_err(),
            "narrower than one VAE tile"
        );
        assert!(Canvas::new(256, 4096).is_ok(), "exactly one tile wide");
    }

    #[test]
    fn rope_width_is_three_axes_duplicated() {
        let rope_dim = 2 * 3 * ROPE_FREQS;
        assert_eq!(rope_dim, 96);
        assert!(rope_dim <= HEAD_DIM);
        assert_eq!(INNER, 7168);
        assert_eq!(PATCH_DIM, 96);
    }
}

/// VAE decoder geometry (a 2D ViT over latent pixels, decoded in 256-pixel tiles).
pub(crate) const VAE_DIM: usize = 1024;
pub(crate) const VAE_HEADS: usize = 16;
pub(crate) const VAE_HEAD_DIM: usize = 64;
pub(crate) const VAE_FFN: usize = 4096;
pub(crate) const VAE_BLOCKS: usize = 24;
pub(crate) const VAE_REGISTERS: usize = 4;
/// Tokens appended to every tile: the register tokens and one zero token.
pub(crate) const VAE_SUFFIX: usize = VAE_REGISTERS + 1;
pub(crate) const VAE_ROPE_THETA: f32 = 100.0;
/// Rotated channels per head: half of the head, over two axes.
pub(crate) const VAE_ROPE_DIM: usize = VAE_HEAD_DIM / 2;
pub(crate) const VAE_TILE_PIXELS: usize = 256;
pub(crate) const VAE_TILE_OVERLAP: usize = 64;
pub(crate) const VAE_TILE_LATENT: usize = VAE_TILE_PIXELS / VAE_RATIO;
pub(crate) const PIXEL_CHANNELS: usize = 3;
pub(crate) const VAE_OUT_DIM: usize = PIXEL_CHANNELS * VAE_RATIO * VAE_RATIO;

/// Per-channel latent statistics of the VAE (`comfy/ldm/minimax/vae.py` `LATENTS_MEAN`, `LATENTS_STD`), exact float32 values.
pub(crate) const LATENTS_MEAN: [f32; LATENT_CHANNELS] = [
    0.85809034,
    -0.96065915,
    1.066164,
    -0.50903255,
    -0.2727582,
    -1.3675414,
    -0.2553255,
    -0.26907554,
    -0.5376841,
    -0.04640973,
    0.66573703,
    0.19690128,
    -0.5460608,
    -0.4035342,
    -0.23683025,
    0.25928453,
    -0.30133945,
    0.21134199,
    -1.1206849,
    0.35819334,
    -0.042251438,
    0.260483,
    0.22864093,
    0.7056032,
];
pub(crate) const LATENTS_STD: [f32; LATENT_CHANNELS] = [
    1.2223774, 1.2767264, 1.6831775, 1.7549455, 1.5636216, 2.1941435, 0.9653138, 1.0569886,
    0.8419489, 0.7729953, 1.8955938, 0.94684184, 0.79968095, 0.449889, 0.71974, 0.6936293,
    2.961095, 2.76942, 3.0496185, 2.1088054, 3.2762263, 3.1627357, 2.2816813, 2.6127844,
];
pub(crate) const PIXEL_MEAN: [f32; PIXEL_CHANNELS] = [0.485, 0.456, 0.406];
pub(crate) const PIXEL_STD: [f32; PIXEL_CHANNELS] = [0.229, 0.224, 0.225];

/// Text encoder (Qwen3-VL 32B language stack, first 50 of 64 layers) geometry.
pub(crate) const TE_DIM: usize = 5120;
pub(crate) const TE_LAYERS: usize = 50;
pub(crate) const TE_HEADS: usize = 64;
pub(crate) const TE_KV_HEADS: usize = 8;
pub(crate) const TE_HEAD_DIM: usize = 128;
pub(crate) const TE_Q: usize = TE_HEADS * TE_HEAD_DIM;
pub(crate) const TE_KV: usize = TE_KV_HEADS * TE_HEAD_DIM;
pub(crate) const TE_FFN: usize = 25600;
pub(crate) const TE_VOCAB: usize = 151_936;
pub(crate) const TE_EPS: f32 = 1e-6;
pub(crate) const TE_ROPE_THETA: f32 = 5_000_000.0;
/// Prompt rows are padded up to a multiple of this so the INT8 GEMMs see few distinct shapes; the model is causal,
/// so padding rows at the end never influence the real rows.
pub(crate) const TE_ROW_BUCKET: usize = 64;

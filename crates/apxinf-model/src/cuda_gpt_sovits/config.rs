//! GPT-SoVITS V5 model geometry and checkpoint defaults.

use std::path::Path;

use apxinf_core::{Error, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct GptSovitsConfig {
    pub model_type: String,
    pub variant: String,
    pub sample_rate: u32,
    pub mel_frames_per_second: usize,
    pub semantic_frames_per_second: usize,
}

impl GptSovitsConfig {
    pub fn from_dir(directory: &Path) -> Result<Self> {
        let path = directory.join("config.json");
        let raw = std::fs::read_to_string(&path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        let config: Self = serde_json::from_str(&raw)
            .map_err(|error| Error::Other(format!("parse {}: {error}", path.display())))?;
        if config.model_type != "gpt_sovits" {
            return Err(Error::Other(format!(
                "{} is not a gpt_sovits checkpoint (model_type={})",
                directory.display(),
                config.model_type
            )));
        }
        if config.variant != "v5turbo" {
            return Err(Error::Other(format!(
                "unsupported GPT-SoVITS variant {}; this build supports v5turbo",
                config.variant
            )));
        }
        Ok(config)
    }
}

pub const GPT_HIDDEN: usize = 512;
pub const GPT_HEADS: usize = 16;
pub const GPT_LAYERS: usize = 24;
pub const GPT_FF: usize = 2048;
pub const GPT_PHONE_VOCAB: usize = 732;
pub const GPT_AUDIO_VOCAB: usize = 1025;
pub const GPT_EOS: u32 = 1024;
pub const GPT_BERT_DIM: usize = 1024;

pub const HUBERT_LAYERS: usize = 12;
pub const HUBERT_HIDDEN: usize = 768;
pub const HUBERT_HEADS: usize = 12;
pub const HUBERT_FF: usize = 3072;
pub const HUBERT_CONV_DIM: usize = 512;

pub const BERT_LAYERS: usize = 24;
pub const BERT_HIDDEN: usize = 1024;
pub const BERT_HEADS: usize = 16;
pub const BERT_FF: usize = 4096;
pub const BERT_OUTPUT_LAYER: usize = 21;

pub const SOVITS_INTER: usize = 192;
pub const SOVITS_HIDDEN: usize = 192;
// Width of the decoder stack (WaveNet and the MRTE style block), which is the
// same 512 the checkpoint's `dec.*` and `mrte.*` tensors use.
pub const SOVITS_DECODER: usize = 512;
pub const SOVITS_HEADS: usize = 2;
pub const SOVITS_FF: usize = 768;
pub const SOVITS_WINDOW: usize = 4;
pub const SOVITS_WN_LAYERS: usize = 8;

pub const CFM_DIM: usize = 1024;
pub const CFM_DEPTH: usize = 22;
pub const CFM_HEADS: usize = 16;
pub const CFM_HEAD_DIM: usize = 64;
pub const CFM_FF: usize = 2048;
pub const CFM_MEL: usize = 100;
pub const CFM_TEXT: usize = 512;
pub const CFM_TEXT_BLOCKS: usize = 4;
pub const CFM_STEPS: usize = 4;

pub const VOC_UPSAMPLE_RATES: [usize; 5] = [10, 6, 2, 2, 2];
pub const VOC_UPSAMPLE_KERNELS: [usize; 5] = [20, 12, 4, 4, 4];
pub const VOC_RESBLOCK_KERNELS: [usize; 3] = [3, 7, 11];

pub const MEL_FILTER_LENGTH: usize = 1280;
pub const MEL_HOP: usize = 320;
pub const MEL_WIN: usize = 1280;
pub const MEL_SAMPLE_RATE: usize = 32000;
pub const SPEC_BINS: usize = 704;

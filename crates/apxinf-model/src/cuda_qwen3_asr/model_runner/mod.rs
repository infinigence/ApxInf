//! AsrTrait contract and request validation for the qwen3-asr family, plus
//! the decode-side execution strategies (graph replay and the eager loop).

pub(crate) mod decode;

use apxinf_core::{Error, Result};

use crate::asr_trait::{AsrContract, AsrRequest, AsrTrait, AsrTranscription};

use crate::cuda_qwen3_asr::config::*;
use crate::cuda_qwen3_asr::model::SUPPORTED_LANGUAGES;
use crate::cuda_qwen3_asr::model::Qwen3AsrModel;

impl AsrTrait for Qwen3AsrModel {
    fn contract(&self) -> AsrContract {
        AsrContract {
            sample_rate: SAMPLE_RATE,
            num_mel_bins: MEL_BINS,
            max_audio_samples: SAMPLES,
            max_decoder_positions: KV_CAPACITY,
            multilingual: true,
            languages: SUPPORTED_LANGUAGES.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn transcribe(&self, request: &AsrRequest<'_>) -> Result<AsrTranscription> {
        if !request.prompt_token_ids.is_empty() {
            return Err(Error::Other(
                "qwen3-asr builds its prompt internally; pass an empty prompt_token_ids".into(),
            ));
        }
        let language = request.language.ok_or_else(|| {
            Error::Other("qwen3-asr needs an explicit language (no auto-detection)".into())
        })?;
        if let Some(task) = request.task {
            if task != "transcribe" {
                return Err(Error::Other(format!(
                    "qwen3-asr supports task \"transcribe\", got {task:?}"
                )));
            }
        }
        self.transcribe_internal(request.waveform, language, request.max_new_tokens)
    }
}

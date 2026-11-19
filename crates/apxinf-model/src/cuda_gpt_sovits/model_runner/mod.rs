//! TtsTrait contract, request validation, caches, and the prepared plan.

pub(super) mod cache;
pub(super) mod gpt_decode;
mod graph;
mod prepared;

use apxinf_core::Result;

use super::model::{GptSovitsModel, OUTPUT_SAMPLE_RATE};
use crate::tts_trait::{
    GeneratedAudio, PreparedTts, TtsCacheConfig, TtsCacheInfo, TtsContract, TtsRequest, TtsSpec,
    TtsTrait,
};

impl TtsTrait for GptSovitsModel {
    fn contract(&self) -> Result<TtsContract> {
        Ok(TtsContract {
            sample_rate: OUTPUT_SAMPLE_RATE,
            mel_frames_per_second: self.config.mel_frames_per_second,
            semantic_frames_per_second: self.config.semantic_frames_per_second,
        })
    }

    fn synthesize(&mut self, request: &TtsRequest<'_>) -> Result<GeneratedAudio> {
        GptSovitsModel::synthesize(self, request)
    }

    fn configure_cache(&self, config: TtsCacheConfig) -> Result<()> {
        self.cache.borrow_mut().configure(config);
        Ok(())
    }
    fn clear_cache(&self, reset_stats: bool) -> Result<()> {
        self.cache.borrow_mut().clear(reset_stats);
        Ok(())
    }
    fn cache_info(&self) -> TtsCacheInfo {
        self.cache.borrow().info()
    }

    fn prepare(&self, spec: &TtsSpec) -> Result<Box<dyn PreparedTts>> {
        Ok(Box::new(self.prepare_plan(spec)?))
    }
}

//! TtsTrait contract, request validation, caches, and the prepared plan.

pub(super) mod cache;
pub(super) mod eager;
pub(super) mod gpt_decode;
mod graph;
mod prepared;

use std::cell::RefCell;
use std::rc::Rc;

use apxinf_core::Result;

use super::model::{GptSovitsModel, OUTPUT_SAMPLE_RATE};
use crate::tts_trait::{
    GeneratedAudio, PreparedTts, TtsCacheConfig, TtsCacheInfo, TtsContract, TtsRequest, TtsSpec,
    TtsTrait,
};

use cache::ConditioningCache;
use eager::EagerExecutor;
use gpt_decode::GptExecutionState;

/// The model-neutral TTS runtime for the GPT-SoVITS V5 family. Owns the
/// loaded model, the conditioning cache and the request state (semantic
/// decode resources) that eager synthesis needs across calls.
pub struct GptSovitsTts {
    model: std::sync::Arc<GptSovitsModel>,
    cache: Rc<RefCell<ConditioningCache>>,
    gpt_state: RefCell<GptExecutionState>,
}

impl GptSovitsTts {
    pub fn load_with_autotune(
        directory: &std::path::Path,
        device_id: usize,
        autotune: bool,
    ) -> apxinf_core::Result<Self> {
        Ok(Self {
            model: std::sync::Arc::new(GptSovitsModel::load_with_autotune(
                directory, device_id, autotune,
            )?),
            cache: Rc::new(RefCell::new(ConditioningCache::default())),
            gpt_state: RefCell::new(GptExecutionState::new()),
        })
    }
}

impl TtsTrait for GptSovitsTts {
    fn contract(&self) -> Result<TtsContract> {
        Ok(TtsContract {
            sample_rate: OUTPUT_SAMPLE_RATE,
            mel_frames_per_second: self.model.config.mel_frames_per_second,
            semantic_frames_per_second: self.model.config.semantic_frames_per_second,
        })
    }

    fn synthesize(&mut self, request: &TtsRequest<'_>) -> Result<GeneratedAudio> {
        let executor = EagerExecutor {
            model: &self.model,
            cache: &self.cache,
            gpt_state: &self.gpt_state,
        };
        self.model.synthesize_with(request, &executor)
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

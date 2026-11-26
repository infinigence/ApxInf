//! The model-defined execution interface behind the single synthesis
//! orchestration.
//!
//! `GptSovitsModel::synthesize_with` owns the network math and its ordering;
//! an executor decides how each capturable stage is *executed* — eager
//! operator calls or CUDA-graph replay — and where conditioning results are
//! cached. The trait lives in the model layer so runners depend on the
//! model, never the other way around.

use std::rc::Rc;

use apxinf_core::{Result, Tensor};

use super::cfm::ChunkArgs;
use super::conditioning::ReferenceFeatures;
use crate::tts_trait::GptSovitsInputs;

pub(crate) trait SynthesisExecutor {
    /// Conditioning for the reference prompt: BERT features, HuBERT prompt
    /// codes, style vector, decoded reference features and the prompt mel.
    fn reference_features(&self, inputs: &GptSovitsInputs<'_>) -> Result<Rc<ReferenceFeatures>>;
    /// BERT features gathered per phone for the target text.
    fn target_bert(&self, inputs: &GptSovitsInputs<'_>) -> Result<Tensor>;
    /// Fixed output length a prepared plan pins, if any; `None` for eager.
    fn expected_target_frames(&self) -> Option<usize>;
    /// Semantic-token generation over the prompt and target conditioning.
    #[allow(clippy::too_many_arguments)]
    fn generate_semantic(
        &self,
        phones: &[u32],
        bert: &Tensor,
        prompt: &[u32],
        seed: u64,
        top_k: usize,
        top_p: f32,
        temperature: f32,
        repetition_penalty: f32,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>>;
    /// Decode target features from generated semantic codes.
    fn target_features(
        &self,
        codes: &Tensor,
        phones: &[u32],
        style: &Tensor,
        speed_factor: f32,
    ) -> Result<Tensor>;
    /// Execute one CFM chunk. `seed` already folds in the chunk index; do
    /// not add it again.
    fn cfm_chunk(&self, args: ChunkArgs<'_>, steps: usize, seed: u64) -> Result<Tensor>;
    /// Run the vocoder over the denormalized mel.
    fn vocode(&self, mel: &Tensor) -> Result<Vec<f32>>;
}

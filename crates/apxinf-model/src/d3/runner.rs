//! Owning text-decision runtime and fixed-shape CUDA Graph profiles.

use std::collections::{HashMap, VecDeque};

use apxinf_core::{Error, Result};
use apxinf_cuda_new::{ops, CudaContext, PreparedPhase};

use crate::d3::backend::read_f32;
use crate::d3::model::{
    forward_multimodal_device, forward_text_device, write_text_inputs, D3Policies, D3TextScratch,
};
use crate::d3::runtime::{DecisionInput, DecisionOutput, DecisionRuntime, DecisionScore};
use crate::d3::vision::D3VisionScratch;
use crate::d3::weights::bf16::D3DeviceWeights;
use crate::d3::{
    D3DecisionConfig, D3PackageConfig, D3_DEFAULT_TEXT_PROFILE_LIMIT,
    D3_DEFAULT_TEXT_PROFILE_TOKEN_BUDGET,
};
use crate::llm_trait::{ImageInput, LlmCapabilities};

struct TextProfile {
    scratch: D3TextScratch,
    graph: Option<PreparedPhase>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct VisionProfileKey {
    tokens: usize,
    patches: usize,
    modality_layout: Vec<u8>,
    grid_thw: Vec<[u32; 3]>,
}

struct VisionProfile {
    text: D3TextScratch,
    vision: D3VisionScratch,
    graph: Option<PreparedPhase>,
}

pub(crate) struct D3Runner {
    ctx: CudaContext,
    config: D3PackageConfig,
    weights: D3DeviceWeights,
    policies: D3Policies,
    text_profiles: HashMap<usize, TextProfile>,
    text_profile_order: VecDeque<usize>,
    text_profile_limit: usize,
    text_profile_token_budget: usize,
    vision_profiles: HashMap<VisionProfileKey, VisionProfile>,
}

impl D3Runner {
    pub(crate) fn new(
        ctx: CudaContext,
        config: D3PackageConfig,
        weights: D3DeviceWeights,
        policies: D3Policies,
        text_profile_limit: Option<usize>,
        text_profile_token_budget: Option<usize>,
    ) -> Result<Self> {
        let text_profile_limit = text_profile_limit.unwrap_or(D3_DEFAULT_TEXT_PROFILE_LIMIT);
        let text_profile_token_budget =
            text_profile_token_budget.unwrap_or(D3_DEFAULT_TEXT_PROFILE_TOKEN_BUDGET);
        if text_profile_limit == 0 {
            return Err(Error::Other(
                "d3 text profile cache limit must be positive".into(),
            ));
        }
        if text_profile_token_budget == 0 {
            return Err(Error::Other(
                "d3 text profile token budget must be positive".into(),
            ));
        }
        Ok(Self {
            ctx,
            config,
            weights,
            policies,
            text_profiles: HashMap::new(),
            text_profile_order: VecDeque::new(),
            text_profile_limit,
            text_profile_token_budget,
            vision_profiles: HashMap::new(),
        })
    }

    fn prepare_text_profile(&mut self, token_ids: &[u32]) -> Result<()> {
        let tokens = token_ids.len();
        if self.text_profiles.contains_key(&tokens) {
            self.text_profile_order.retain(|length| *length != tokens);
            self.text_profile_order.push_back(tokens);
            return Ok(());
        }
        while !self.text_profiles.is_empty()
            && (self.text_profiles.len() >= self.text_profile_limit
                || self
                    .text_profiles
                    .keys()
                    .copied()
                    .fold(0usize, usize::saturating_add)
                    .saturating_add(tokens)
                    > self.text_profile_token_budget)
        {
            let oldest = self.text_profile_order.pop_front().ok_or_else(|| {
                Error::Other("d3 text profile cache order is inconsistent".into())
            })?;
            self.text_profiles.remove(&oldest);
        }
        let mut scratch = D3TextScratch::new(&self.ctx, &self.config, tokens)?;
        write_text_inputs(&self.ctx, &self.config, &scratch, token_ids)?;
        let graph = if std::env::var_os("APXINF_D3_EAGER").is_some()
            || std::env::var_os("APXINF_D3_TRACE").is_some()
        {
            None
        } else {
            let session =
                ops::ExecutionSession::with_capacity(256 * 1024 * 1024, self.ctx.device_id())?;
            Some(PreparedPhase::prepare_and_capture(
                &self.ctx,
                session,
                || {
                    forward_text_device(
                        &self.ctx,
                        &self.config,
                        &self.weights,
                        &self.policies,
                        &mut scratch,
                        tokens,
                    )
                },
            )?)
        };
        if scratch.capacity() != tokens {
            return Err(Error::Other("d3 prepared profile capacity mismatch".into()));
        }
        self.text_profiles
            .insert(tokens, TextProfile { scratch, graph });
        self.text_profile_order.push_back(tokens);
        Ok(())
    }

    fn decide_text(&mut self, token_ids: &[u32], candidate_count: usize) -> Result<DecisionOutput> {
        self.prepare_text_profile(token_ids)?;
        let profile = self
            .text_profiles
            .get_mut(&token_ids.len())
            .ok_or_else(|| Error::Other("d3 text profile disappeared after preparation".into()))?;
        write_text_inputs(&self.ctx, &self.config, &profile.scratch, token_ids)?;
        if let Some(graph) = &profile.graph {
            if std::env::var_os("APXINF_D3_PREPARED_EAGER").is_some() {
                graph.run(|| {
                    forward_text_device(
                        &self.ctx,
                        &self.config,
                        &self.weights,
                        &self.policies,
                        &mut profile.scratch,
                        token_ids.len(),
                    )
                })?;
            } else {
                graph.replay()?;
            }
        } else {
            forward_text_device(
                &self.ctx,
                &self.config,
                &self.weights,
                &self.policies,
                &mut profile.scratch,
                token_ids.len(),
            )?;
        }
        let logits = read_f32(&self.ctx, profile.scratch.logits())?;
        probabilities(&self.config.decision, &logits, candidate_count)
    }

    fn vision_key(&self, token_ids: &[u32], image: ImageInput<'_>) -> Result<VisionProfileKey> {
        let dims = image.pixel_values.shape().dims();
        if dims.len() != 2 {
            return Err(Error::Other(format!(
                "d3 vision pixels must be rank 2, got {dims:?}"
            )));
        }
        let modality_layout = token_ids
            .iter()
            .map(|token| {
                if *token == self.config.model.image_token_id {
                    1
                } else if *token == self.config.model.video_token_id {
                    2
                } else {
                    0
                }
            })
            .collect();
        Ok(VisionProfileKey {
            tokens: token_ids.len(),
            patches: dims[0],
            modality_layout,
            grid_thw: image.grid_thw.to_vec(),
        })
    }

    fn prepare_vision_profile(
        &mut self,
        key: &VisionProfileKey,
        token_ids: &[u32],
        image: ImageInput<'_>,
    ) -> Result<()> {
        if self.vision_profiles.contains_key(key) {
            return Ok(());
        }
        let mut text = D3TextScratch::new(&self.ctx, &self.config, key.tokens)?;
        let mut vision = D3VisionScratch::new(
            &self.ctx,
            &self.config,
            &self.weights,
            token_ids,
            image.grid_thw,
            key.patches,
        )?;
        write_text_inputs(&self.ctx, &self.config, &text, token_ids)?;
        vision.write_pixels(&self.ctx, image.pixel_values)?;
        let graph = if std::env::var_os("APXINF_D3_EAGER").is_some()
            || std::env::var_os("APXINF_D3_TRACE").is_some()
        {
            None
        } else {
            let session =
                ops::ExecutionSession::with_capacity(512 * 1024 * 1024, self.ctx.device_id())?;
            Some(PreparedPhase::prepare_and_capture(
                &self.ctx,
                session,
                || {
                    forward_multimodal_device(
                        &self.ctx,
                        &self.config,
                        &self.weights,
                        &self.policies,
                        &mut text,
                        &mut vision,
                        key.tokens,
                    )
                },
            )?)
        };
        self.vision_profiles.insert(
            key.clone(),
            VisionProfile {
                text,
                vision,
                graph,
            },
        );
        Ok(())
    }

    fn decide_vision(
        &mut self,
        token_ids: &[u32],
        image: ImageInput<'_>,
        candidate_count: usize,
    ) -> Result<DecisionOutput> {
        let key = self.vision_key(token_ids, image)?;
        self.prepare_vision_profile(&key, token_ids, image)?;
        let profile = self
            .vision_profiles
            .get_mut(&key)
            .ok_or_else(|| Error::Other("d3 vision profile disappeared".into()))?;
        write_text_inputs(&self.ctx, &self.config, &profile.text, token_ids)?;
        profile.vision.write_pixels(&self.ctx, image.pixel_values)?;
        if let Some(graph) = &profile.graph {
            if std::env::var_os("APXINF_D3_PREPARED_EAGER").is_some() {
                graph.run(|| {
                    forward_multimodal_device(
                        &self.ctx,
                        &self.config,
                        &self.weights,
                        &self.policies,
                        &mut profile.text,
                        &mut profile.vision,
                        key.tokens,
                    )
                })?;
            } else {
                graph.replay()?;
            }
        } else {
            forward_multimodal_device(
                &self.ctx,
                &self.config,
                &self.weights,
                &self.policies,
                &mut profile.text,
                &mut profile.vision,
                key.tokens,
            )?;
        }
        let logits = read_f32(&self.ctx, profile.text.logits())?;
        probabilities(&self.config.decision, &logits, candidate_count)
    }
}

impl DecisionRuntime for D3Runner {
    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::TEXT_ONLY
    }

    fn decide(&mut self, input: DecisionInput<'_>) -> Result<DecisionOutput> {
        if let Some(image) = input.image {
            self.decide_vision(input.token_ids, image, input.candidate_count)
        } else {
            self.decide_text(input.token_ids, input.candidate_count)
        }
    }

    fn code_names(&self) -> &[String] {
        self.config.decision.code_names()
    }

    fn token_ids(&self) -> &[u32] {
        self.config.decision.token_ids()
    }
}

fn probabilities(
    config: &D3DecisionConfig,
    logits: &[f32],
    candidate_count: usize,
) -> Result<DecisionOutput> {
    if logits.len() != config.codes.len() {
        return Err(Error::Other(format!(
            "d3 readout returned {} logits for {} codes",
            logits.len(),
            config.codes.len()
        )));
    }
    if candidate_count == 0 || candidate_count > config.codes.len() {
        return Err(Error::Other(format!(
            "d3 candidate count {candidate_count} is outside 1..={}",
            config.codes.len()
        )));
    }
    let logits = &logits[..candidate_count];
    let inv_temperature = 1.0 / config.temperature;
    if logits.iter().any(|value| !value.is_finite()) {
        return Err(Error::Other("d3 readout produced non-finite logits".into()));
    }
    let maximum = logits
        .iter()
        .copied()
        .map(|value| value * inv_temperature)
        .reduce(f32::max)
        .ok_or_else(|| Error::Other("d3 readout returned no logits".into()))?;
    let mut values = logits
        .iter()
        .map(|value| (value * inv_temperature - maximum).exp())
        .collect::<Vec<_>>();
    let denominator = values.iter().sum::<f32>();
    if !denominator.is_finite() || denominator <= 0.0 {
        return Err(Error::Other("d3 softmax normalization failed".into()));
    }
    for value in &mut values {
        *value /= denominator;
    }
    let mut selected_index = 0usize;
    for index in 1..values.len() {
        if values[index] > values[selected_index] {
            selected_index = index;
        }
    }
    let scores = values
        .into_iter()
        .enumerate()
        .map(|(index, probability)| DecisionScore {
            code: config.codes[index].clone(),
            token_id: config.token_ids[index],
            probability,
        })
        .collect();
    Ok(DecisionOutput {
        selected_index,
        scores,
    })
}

#[cfg(test)]
mod tests {
    use super::probabilities;
    use crate::d3::D3DecisionConfig;

    #[test]
    fn readout_softmax_is_stable_and_selects_maximum() {
        let config = D3DecisionConfig {
            format_version: 1,
            format_id: "d3-code-readout-v1".into(),
            prompt: "d3".into(),
            base_model: "Qwen/Qwen3.5-4B".into(),
            revision: "test".into(),
            codes: vec!["a".into(), "b".into(), "c".into()],
            token_ids: vec![1, 2, 3],
            temperature: 1.0,
            attention_mode: "noncausal_full_attention".into(),
            pooling: "last".into(),
            max_length: None,
            readout_dtype: "float32".into(),
        };
        let output = probabilities(&config, &[1_000.0, 1_002.0, 999.0], 2).unwrap();
        assert_eq!(output.selected_index, 1);
        assert_eq!(output.scores.len(), 2);
        assert!(
            (output
                .scores
                .iter()
                .map(|score| score.probability)
                .sum::<f32>()
                - 1.0)
                .abs()
                < 1e-6
        );
    }

    #[test]
    fn readout_ties_select_first_code() {
        let config = D3DecisionConfig {
            format_version: 1,
            format_id: "d3-code-readout-v1".into(),
            prompt: "d3".into(),
            base_model: "test".into(),
            revision: "test".into(),
            codes: vec!["a".into(), "b".into()],
            token_ids: vec![1, 2],
            temperature: 1.0,
            attention_mode: "noncausal_full_attention".into(),
            pooling: "last".into(),
            max_length: None,
            readout_dtype: "float32".into(),
        };
        assert_eq!(
            probabilities(&config, &[2.0, 2.0], 2)
                .unwrap()
                .selected_index,
            0
        );
    }

    #[test]
    fn readout_rejects_non_finite_valid_logits() {
        let config = D3DecisionConfig {
            format_version: 1,
            format_id: "d3-code-readout-v1".into(),
            prompt: "d3".into(),
            base_model: "test".into(),
            revision: "test".into(),
            codes: vec!["a".into(), "b".into(), "c".into()],
            token_ids: vec![1, 2, 3],
            temperature: 1.0,
            attention_mode: "noncausal_full_attention".into(),
            pooling: "last".into(),
            max_length: None,
            readout_dtype: "float32".into(),
        };
        assert!(probabilities(&config, &[2.0, f32::NAN, 100.0], 2).is_err());
        assert!(probabilities(&config, &[2.0, 1.0, f32::NAN], 2).is_ok());
    }
}

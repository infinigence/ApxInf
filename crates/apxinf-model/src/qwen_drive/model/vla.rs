//! Planning computation: multimodal prefix, optional reasoning, then flow sampling.
//! The caller owns backbone state and chooses how GDN blocks execute.
use super::blocks::bf16::{expert, upload_u32, BackboneBf16, BackboneState, PlannerBf16};
use super::blocks::GdnExecution;
use super::{PlanningState, VisionState};
use crate::qwen_drive::backend::kernels::linear_attention;
use crate::qwen_drive::inputs::ExpertConditioning;
use apxinf_core::{
    NextTokenLogits, Result, RngKey, SamplingBackend, Tensor, TokenSamplingInit,
    TokenSamplingParams, TokenSamplingSpec,
};

pub(crate) struct ReasoningInput<'a> {
    pub max_new_tokens: usize,
    pub min_new_tokens: usize,
    pub terminator_ids: &'a [u32],
    pub closing_ids: &'a [u32],
}
pub(crate) struct PlanningInput<'a> {
    pub token_ids: &'a [u32],
    pub pixels: &'a Tensor,
    pub grids: &'a [[u32; 3]],
    pub conditioning: &'a ExpertConditioning,
    pub noise: &'a Tensor,
    pub steps: usize,
    pub reasoning: Option<ReasoningInput<'a>>,
}
pub(crate) struct QwenDriveModel {
    backbone: BackboneBf16,
    planner: PlannerBf16,
}
impl QwenDriveModel {
    pub fn new(
        config: crate::qwen_drive::config::QwenDriveConfig,
        cuda: std::sync::Arc<crate::qwen_drive::backend::RuntimeBackend>,
        backbone: crate::qwen_drive::weights::bf16::BackboneDeviceWeights,
        planner: crate::qwen_drive::weights::bf16::ExpertDeviceWeights,
    ) -> Self {
        Self {
            backbone: BackboneBf16 {
                config: config.clone(),
                cuda: cuda.clone(),
                weights: backbone,
            },
            planner: PlannerBf16 {
                config,
                cuda,
                weights: planner,
            },
        }
    }
    pub fn config(&self) -> &crate::qwen_drive::config::QwenDriveConfig {
        &self.backbone.config
    }
    pub fn backend(&self) -> &std::sync::Arc<crate::qwen_drive::backend::RuntimeBackend> {
        &self.backbone.cuda
    }
    pub fn new_state(&self, vision: std::rc::Rc<VisionState>) -> Result<PlanningState> {
        self.backbone.new_state(vision)
    }

    pub fn infer(
        &self,
        state: &mut BackboneState,
        execution: &mut dyn GdnExecution,
        input: &PlanningInput<'_>,
    ) -> Result<Tensor> {
        let b = &self.backbone;
        let hidden = b.prefill(state, execution, input.token_ids, input.pixels, input.grids)?;
        let anchor = if let Some(reasoning) = &input.reasoning {
            let mut generated = Vec::new();
            let mut sampler = b.cuda.create_token_sampler(TokenSamplingSpec {
                vocab_size: b.config.text.vocab_size,
                max_sequence_len: input.token_ids.len() + reasoning.max_new_tokens + 1,
            })?;
            sampler.begin(TokenSamplingInit {
                prompt_token_ids: input.token_ids,
                params: &TokenSamplingParams::greedy(),
                rng: RngKey::default(),
            })?;
            let mut logits = b.next_logits(&hidden)?;
            let prompt_anchor = state.last_position;
            let eos = upload_u32(b.ctx(), reasoning.terminator_ids)?;
            for step in 0..reasoning.max_new_tokens {
                state.decode_step = Some(step);
                if step < reasoning.min_new_tokens {
                    linear_attention::suppress_logits(
                        b.ctx(),
                        &logits,
                        logits.shape().dims()[0] - 1,
                        &eos,
                    )?;
                }
                let sample =
                    sampler.sample(NextTokenLogits::last(&logits, b.config.text.vocab_size)?)?;
                generated.push(sample.token_id);
                if reasoning.terminator_ids.contains(&sample.token_id)
                    || step + 1 == reasoning.max_new_tokens
                {
                    break;
                }
                let hidden = b.forward_tokens(state, execution, &[sample.token_id])?;
                logits = b.next_logits(&hidden)?;
            }
            let mut closed = generated.clone();
            if let Some(index) = closed
                .iter()
                .position(|id| reasoning.terminator_ids.contains(id))
            {
                closed.truncate(index);
            }
            closed.extend_from_slice(reasoning.closing_ids);
            let cached = generated.len().saturating_sub(1);
            if cached < closed.len() {
                b.forward_tokens(state, execution, &closed[cached..])?;
            }
            prompt_anchor + closed.len() as i64
        } else {
            state.last_position
        };
        // Direct planning and turn completion need caches, not language logits.
        let scene = b.scene_caches(state)?;
        let expert_input = expert::ExpertPlan {
            scene: &scene,
            scene_len: state.cache_len,
            anchor,
            cond: input.conditioning,
            noise: input.noise,
            num_steps: input.steps,
        };
        let flow = self.planner.prepare(&expert_input)?;
        for step in 0..input.steps {
            self.planner.step(&expert_input, &flow, step)?;
        }
        Ok(self.planner.output(flow))
    }
}

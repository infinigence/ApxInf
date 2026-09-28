//! Planning computation: multimodal prefix, optional reasoning, then flow sampling.
//! The caller owns backbone state and chooses how GDN blocks execute.
use super::blocks::bf16::{expert, upload_u32, BackboneBf16, BackboneState, PlannerBf16};
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
    /// Unmasked tokens in `token_ids`. Equal to `token_ids.len()` for an unpadded
    /// prompt, smaller when the caller padded to a fixed width to keep the
    /// shape-specialised kernels reachable. Everything semantic — the expert's
    /// scene extent, the planner KV offset, the attention key count — uses this;
    /// buffer shapes use `token_ids.len()`.
    pub prompt_len: usize,
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
    pub fn reset_state(&self, state: &mut PlanningState) -> Result<()> {
        self.backbone.reset_state(state)
    }

    pub(crate) fn validate_layout(&self, token_ids: &[u32], grids: &[[u32; 3]]) -> Result<()> {
        self.backbone.rope_index(token_ids, grids).map(|_| ())
    }
    pub(crate) fn prepare_direct_inputs(
        &self,
        state: &mut PlanningState,
        vision: &VisionState,
        token_ids: &[u32],
        prompt_len: usize,
        pixels: &Tensor,
        grids: &[[u32; 3]],
        cond: ExpertConditioning,
        steps: usize,
    ) -> Result<super::DirectInputs> {
        super::DirectInputs::new(
            &self.backbone,
            &self.planner,
            state,
            vision,
            token_ids,
            prompt_len,
            pixels,
            grids,
            cond,
            steps,
        )
    }
    pub(crate) fn forward_direct(
        &self,
        inputs: &super::DirectInputs,
        state: &mut PlanningState,
        execution: &mut dyn super::DirectExecution,
    ) -> Result<Tensor> {
        inputs.forward(&self.backbone, &self.planner, state, execution)
    }

    pub fn infer(&self, state: &mut BackboneState, input: &PlanningInput<'_>) -> Result<Tensor> {
        let b = &self.backbone;
        let hidden = b.prefill(state, input.token_ids, input.pixels, input.grids)?;
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
                let hidden = b.forward_tokens(state, &[sample.token_id])?;
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
                b.forward_tokens(state, &closed[cached..])?;
            }
            prompt_anchor + closed.len() as i64
        } else {
            let positions = b.rope_index(input.token_ids, input.grids)?;
            *positions[input.prompt_len - 1].iter().max().unwrap() as i64
        };
        // Direct planning and turn completion need caches, not language logits.
        let scene = b.scene_caches(state)?;
        let expert_input = expert::ExpertPlan {
            scene: &scene,
            scene_len: if input.reasoning.is_some() {
                state.cache_len
            } else {
                input.prompt_len
            },
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

//! Sub-network connection and flow schedule of the MolmoAct2 BF16 family.
//!
//! ```text
//! uint8 RGB + token ids
//!   -> vision / pooling / connector / VLM prefill        (Bf16Blocks::encode, 36 layers of K/V)
//!   -> context adapter (visible rows, shared K/V projections)   -> one (K, V) per expert block
//!   -> action expert x `flow_steps` forward Euler steps  (Bf16Blocks::expert_flow)
//!   -> normalized padded actions, BF16 [horizon, max_action_dim]
//! ```
//! The VLM context and the adapter output belong to one observation: [`MolmoAct2Model::encode`] runs
//! them once and the flow reuses the result for every step; a new observation always re-encodes.
mod blocks;
pub(super) use blocks::*;

use crate::cuda_molmoact2::backend::ops::{AttentionNumerics, GemmPolicy};
use crate::cuda_molmoact2::backend::{AttentionPolicy, CudaBuffer, CudaContext};
use crate::cuda_molmoact2::config::{MolmoAct2Config, LIBERO_ACTION_DIM};
use crate::cuda_molmoact2::weights::Bf16Weights;
use apxinf_core::{Error, Result, Tensor};
use std::rc::Rc;

/// Token-derived inputs of one request: prefill geometry plus the context rows the expert may see.
pub(in crate::cuda_molmoact2) struct RequestPlan {
    prepared: Prepared,
    rows: ContextRows,
}

pub(in crate::cuda_molmoact2) struct MolmoAct2Model {
    blocks: Bf16Blocks,
    cache: ExpertCache,
    cfg: MolmoAct2Config,
}

impl MolmoAct2Model {
    pub fn new(
        ctx: Rc<CudaContext>,
        weights: Rc<Bf16Weights>,
        cfg: MolmoAct2Config,
        gemm: GemmPolicy,
        attention: AttentionPolicy,
    ) -> Result<Self> {
        let blocks = Bf16Blocks::new(ctx, weights, cfg.clone(), gemm, attention, AttentionNumerics::Flexible, true)?;
        let ex = &cfg.expert;
        let cache = blocks.expert_cache(ex.flow_steps, ex.max_action_horizon, LIBERO_ACTION_DIM)?;
        Ok(Self { blocks, cache, cfg })
    }

    pub fn config(&self) -> &MolmoAct2Config {
        &self.cfg
    }

    /// Geometry and visible context rows from the prompt token ids. The prompt of this profile has no
    /// padding, so every position is valid.
    pub fn plan(&self, token_ids: &[u32]) -> Result<RequestPlan> {
        self.plan_from(&self.analyze(token_ids)?)
    }

    /// Host-only analysis of the prompt tokens (structure and token-dependent contents).
    pub fn analyze(&self, token_ids: &[u32]) -> Result<HostPlan> {
        self.blocks.analyze(token_ids, &vec![true; token_ids.len()])
    }

    /// Device tables of an analysed request.
    pub fn plan_from(&self, host: &HostPlan) -> Result<RequestPlan> {
        Ok(RequestPlan { prepared: self.blocks.prepare_host(host)?, rows: self.blocks.context_rows_host(host)? })
    }

    /// Rewrite a prepared plan's token-dependent contents in place; errors when the structure differs.
    pub fn update_plan(&self, plan: &RequestPlan, host: &HostPlan) -> Result<()> {
        self.blocks.update_inputs(&plan.prepared, &plan.rows, host)
    }

    /// The whole observation-to-action computation, in the one order eager execution and graph capture
    /// share: vision -> connector -> VLM prefill -> context adapter -> every flow step -> publish the
    /// normalized padded actions as F32 into `out`. Nothing here synchronizes or reads back.
    pub fn forward(&self, rgb: &CudaBuffer, plan: &RequestPlan, noise_bf16: &Tensor, out: &mut Tensor) -> Result<()> {
        let context = self.encode(rgb, plan)?;
        let trajectory = self.flow(noise_bf16, &context)?;
        self.publish(&trajectory, out)
    }

    /// The observation's context: VLM prefill then the expert's context adapter.
    pub fn encode(&self, rgb: &CudaBuffer, plan: &RequestPlan) -> Result<ExpertContext> {
        let kv = self.blocks.encode(rgb, &plan.prepared)?;
        self.blocks.expert_context(&kv, &plan.rows)
    }

    /// Initial noise `[horizon, max_action_dim]` as F32 -> BF16 (the trajectory dtype of the official run).
    pub fn noise_to_bf16(&self, noise: &Tensor) -> Result<Tensor> {
        self.blocks.cast_f32_to_bf16(noise)
    }

    /// The flow over the already encoded context.
    pub fn flow(&self, noise_bf16: &Tensor, context: &ExpertContext) -> Result<Tensor> {
        if context.layers.len() != self.cfg.expert.layers {
            return Err(Error::Other("MolmoAct2 flow: context does not have one entry per expert block".into()));
        }
        self.blocks.expert_flow(noise_bf16, &self.cache, context)
    }

    /// Normalized padded actions as F32 into `out` (`[horizon, max_action_dim]`).
    pub fn publish(&self, trajectory: &Tensor, out: &mut Tensor) -> Result<()> {
        self.blocks.cast_bf16_to_f32(trajectory, out)
    }
}

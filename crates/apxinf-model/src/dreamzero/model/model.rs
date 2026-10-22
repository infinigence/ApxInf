//! Model-domain stages and storage. No request, preparation or graph policy.
use super::{
    dit::{DitContext, DitWorkspace, VideoCache},
    dit_block::rows,
    sample::JointWorkspace,
    text::{self, TextWorkspace},
    vision::VisionWorkspace,
};
use crate::dreamzero::{
    backend::{allocate, binary, ops, transfers, Context, Policies},
    schedule::Schedule,
    weights::DreamZeroWeights,
};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::rc::Rc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::dreamzero) struct Profile {
    pub input_frames: usize,
    /// Position of the next generated video block, after committing observations.
    pub start_frame: usize,
}
impl Profile {
    pub fn initial(self) -> bool {
        self.start_frame == 1
    }
    pub fn validate(self) -> Result<()> {
        if ![1, 4, 9].contains(&self.input_frames)
            || ![1, 3, 5, 7].contains(&self.start_frame)
            || (!self.initial() && self.input_frames == 1)
        {
            return Err(Error::Other(
                "unsupported DreamZero frame/history profile".into(),
            ));
        }
        Ok(())
    }
    fn previous_rows(self) -> usize {
        if self.initial() {
            0
        } else {
            (self.start_frame - 2) * 880
        }
    }
}

#[derive(Clone)]
pub(in crate::dreamzero) struct History {
    positive: VideoCache,
    negative: VideoCache,
    clip: Tensor,
    conditioning: Tensor,
}
pub(in crate::dreamzero) struct Workspace {
    pub profile: Profile,
    pub ids: [Tensor; 2],
    pub masks: [Tensor; 2],
    pub valid_rows: [Tensor; 2],
    pub vision: VisionWorkspace,
    pub joint: JointWorkspace,
    text: TextWorkspace,
    masked_text: Tensor,
    contexts: [DitContext; 2],
    clip: Tensor,
    conditioning: Tensor,
    observed: DitWorkspace,
    prior: Option<[VideoCache; 2]>,
    candidate: [VideoCache; 2],
    zero_time: Tensor,
}
pub(in crate::dreamzero) struct DreamZeroModel {
    pub ctx: Rc<Context>,
    weights: DreamZeroWeights,
    policies: Policies,
}
impl DreamZeroModel {
    pub fn new(ctx: Rc<Context>, weights: DreamZeroWeights, policies: Policies) -> Self {
        Self {
            ctx,
            weights,
            policies,
        }
    }
    fn execution_policies(&self, tune: bool) -> Policies {
        let mut p = self.policies.clone();
        p.gemm.online_tune &= tune;
        p.attention.online_tune &= tune;
        p
    }
    /// Every activation has explicit model-owned storage; prepared native
    /// GEMM/attention executions own their opaque provider workspaces. The
    /// session arena is unused but the backend requires nonzero capacity.
    pub fn phase_scratch_bytes(&self) -> [usize; 3] {
        [256; 3]
    }
    pub fn workspace(&self, profile: Profile) -> Result<Workspace> {
        profile.validate()?;
        let ctx = &self.ctx;
        let previous = profile.previous_rows();
        let observed_frames = if profile.initial() { 1 } else { 2 };
        let observed_start = if profile.initial() {
            0
        } else {
            profile.start_frame - 2
        };
        let mut zero_time = vec![half::bf16::ZERO; 512];
        zero_time[..128].fill(half::bf16::ONE);
        zero_time[256..384].fill(half::bf16::ONE);
        let ids = || ctx.allocate_output(Shape::new(vec![512]), DType::I32);
        Ok(Workspace {
            profile,
            ids: [ids()?, ids()?],
            masks: [allocate(ctx, &[1, 512])?, allocate(ctx, &[1, 512])?],
            valid_rows: [allocate(ctx, &[512, 1])?, allocate(ctx, &[512, 1])?],
            vision: VisionWorkspace::new(ctx, profile.input_frames, profile.initial())?,
            joint: JointWorkspace::new(
                ctx,
                profile.start_frame,
                profile.start_frame * 880,
                Schedule::standard()?,
                Schedule::standard()?,
            )?,
            text: TextWorkspace::new(ctx)?,
            masked_text: allocate(ctx, &[512, 4096])?,
            contexts: [DitContext::new(ctx)?, DitContext::new(ctx)?],
            clip: allocate(ctx, &[257, 1280])?,
            conditioning: allocate(ctx, &[20, 9, 44, 80])?,
            observed: DitWorkspace::new(ctx, observed_frames, observed_start, previous, false)?,
            prior: if previous == 0 {
                None
            } else {
                Some([
                    VideoCache::new(ctx, previous)?,
                    VideoCache::new(ctx, previous)?,
                ])
            },
            candidate: [
                VideoCache::new(ctx, profile.start_frame * 880)?,
                VideoCache::new(ctx, profile.start_frame * 880)?,
            ],
            zero_time: transfers::to_cuda(
                &Tensor::from_bf16(vec![2, 256], &zero_time)?,
                ctx.device_id(),
            )?,
        })
    }
    /// Bind an immutable episode snapshot into stable graph input storage.
    pub fn bind_history(&self, w: &Workspace, history: Option<&History>) -> Result<()> {
        match (&w.prior, history) {
            (None, None) => Ok(()),
            (Some(prior), Some(history)) => {
                copy_cache(&self.ctx, &history.positive, &prior[0])?;
                copy_cache(&self.ctx, &history.negative, &prior[1])?;
                transfers::copy_cuda_to_cuda(&self.ctx, &history.clip, &w.clip)?;
                transfers::copy_cuda_to_cuda(&self.ctx, &history.conditioning, &w.conditioning)
            }
            _ => Err(Error::Other(
                "DreamZero history does not match workspace profile".into(),
            )),
        }
    }
    pub fn vision(&self, w: &mut Workspace, tune: bool) -> Result<()> {
        let p = self.execution_policies(tune);
        w.vision
            .forward(&self.ctx, &p, &self.weights.clip, &self.weights.vae)?;
        if w.profile.initial() {
            transfers::copy_cuda_to_cuda(&self.ctx, &w.vision.clip.output, &w.clip)?;
            transfers::copy_cuda_to_cuda(&self.ctx, &w.vision.conditioning, &w.conditioning)?;
        }
        Ok(())
    }
    pub fn language(&self, w: &mut Workspace, tune: bool) -> Result<()> {
        let p = self.execution_policies(tune);
        for i in 0..2 {
            transfers::copy_cuda_to_cuda(&self.ctx, &w.ids[i], &w.text.ids)?;
            transfers::copy_cuda_to_cuda(&self.ctx, &w.masks[i], &w.text.mask)?;
            text::encode(&self.ctx, &p, &self.weights.text, &mut w.text)?;
            // Official encode_prompt zeros padded output rows before the DiT MLP.
            binary(
                &self.ctx,
                ops::PointwiseSemantic::RowMultiply,
                &w.text.output,
                &w.valid_rows[i],
                &mut w.masked_text,
            )?;
            w.contexts[i].encode(&self.ctx, &p, &self.weights.dit, &w.masked_text, &w.clip)?;
        }
        Ok(())
    }
    pub fn action(&self, w: &mut Workspace, tune: bool) -> Result<()> {
        let p = self.execution_policies(tune);
        transfers::copy_cuda_to_cuda(&self.ctx, &w.vision.observed, &w.observed.video)?;
        transfers::copy_cuda_to_cuda(&self.ctx, &w.zero_time, &w.observed.time_sinusoidal)?;
        let count = if w.profile.initial() { 1 } else { 2 };
        let start = if w.profile.initial() {
            0
        } else {
            w.profile.start_frame - 2
        };
        let source = w.conditioning.reshape(vec![20 * 9, 44 * 80])?;
        let target = w.observed.conditioning.reshape(vec![20 * count, 44 * 80])?;
        for c in 0..20 {
            transfers::copy_cuda_to_cuda(
                &self.ctx,
                &rows(&source, c * 9 + start, count)?,
                &rows(&target, c * count, count)?,
            )?;
        }
        for i in 0..2 {
            w.observed.forward(
                &self.ctx,
                &p,
                &self.weights.dit,
                &w.contexts[i],
                w.prior.as_ref().map(|p| &p[i]),
                Some(&w.candidate[i]),
            )?;
        }
        w.joint.sample(
            &self.ctx,
            &p,
            &self.weights.dit,
            &w.contexts[0],
            &w.contexts[1],
            &w.candidate[0],
            &w.candidate[1],
            &w.conditioning,
        )
    }
    /// Independent storage prevents a later warmup or failed inference from
    /// overwriting committed history through a prepared workspace alias.
    pub fn snapshot(&self, w: &Workspace) -> Result<History> {
        let h = History {
            positive: VideoCache::new(&self.ctx, w.candidate[0].rows)?,
            negative: VideoCache::new(&self.ctx, w.candidate[1].rows)?,
            clip: allocate(&self.ctx, &[257, 1280])?,
            conditioning: allocate(&self.ctx, &[20, 9, 44, 80])?,
        };
        copy_cache(&self.ctx, &w.candidate[0], &h.positive)?;
        copy_cache(&self.ctx, &w.candidate[1], &h.negative)?;
        transfers::copy_cuda_to_cuda(&self.ctx, &w.clip, &h.clip)?;
        transfers::copy_cuda_to_cuda(&self.ctx, &w.conditioning, &h.conditioning)?;
        Ok(h)
    }
}
fn copy_cache(ctx: &Context, source: &VideoCache, target: &VideoCache) -> Result<()> {
    if source.rows != target.rows {
        return Err(Error::Other(
            "DreamZero committed history has the wrong length".into(),
        ));
    }
    for ((sk, sv), (tk, tv)) in source
        .keys
        .iter()
        .zip(&source.values)
        .zip(target.keys.iter().zip(&target.values))
    {
        transfers::copy_cuda_to_cuda(ctx, sk, tk)?;
        transfers::copy_cuda_to_cuda(ctx, sv, tv)?;
    }
    Ok(())
}

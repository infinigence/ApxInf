//! Joint model topology and standard sampling order; no graph/request policy.
pub(super) mod blocks;
use crate::cuda_dreamzero::{
    backend::Context, config::Policies, schedule::Schedule, weights::Bf16Weights,
};
use apxinf_core::{Error, Result, Tensor};
use blocks::Bf16Blocks;
pub(super) use blocks::History;
use std::rc::Rc;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::cuda_dreamzero) struct Profile {
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
    pub(super) fn previous_rows(self) -> usize {
        if self.initial() {
            0
        } else {
            (self.start_frame - 2) * 880
        }
    }
}

pub(super) struct Workspace {
    inner: blocks::Workspace,
}
impl Workspace {
    pub fn images(&self) -> &Tensor {
        &self.inner.vision.images
    }
    pub fn state(&self) -> &Tensor {
        &self.inner.joint.state
    }
    pub fn noise(&self) -> &Tensor {
        &self.inner.joint.joint_noise
    }
    pub fn ids(&self, i: usize) -> &Tensor {
        &self.inner.ids[i]
    }
    pub fn mask(&self, i: usize) -> &Tensor {
        &self.inner.masks[i]
    }
    pub fn valid_rows(&self, i: usize) -> &Tensor {
        &self.inner.valid_rows[i]
    }
    pub fn actions(&self) -> &Tensor {
        &self.inner.joint.action.sample
    }
    #[cfg(test)]
    pub fn video(&self) -> &Tensor {
        &self.inner.joint.video.sample
    }
}
pub(super) struct DreamZeroModel {
    pub ctx: Rc<Context>,
    blocks: Bf16Blocks,
    steps: usize,
}
impl DreamZeroModel {
    pub fn new(ctx: Rc<Context>, weights: Bf16Weights, policies: Policies) -> Self {
        Self {
            blocks: Bf16Blocks::new(ctx.clone(), weights, policies),
            ctx,
            steps: Schedule::standard()
                .expect("fixed standard schedule")
                .steps
                .len(),
        }
    }
    pub fn phase_scratch_bytes(&self) -> [usize; 3] {
        self.blocks.phase_scratch_bytes()
    }
    pub fn workspace(&self, profile: Profile) -> Result<Workspace> {
        Ok(Workspace {
            inner: self.blocks.workspace(profile)?,
        })
    }
    pub fn bind_history(&self, w: &Workspace, h: Option<&History>) -> Result<()> {
        self.blocks.bind_history(&w.inner, h)
    }
    pub fn vision(&self, w: &mut Workspace, tune: bool) -> Result<()> {
        self.blocks.vision(&mut w.inner, tune)
    }
    pub fn language(&self, w: &mut Workspace, tune: bool) -> Result<()> {
        self.blocks.language(&mut w.inner, tune)
    }
    pub fn action(&self, w: &mut Workspace, tune: bool) -> Result<()> {
        self.blocks.action(&mut w.inner, tune)?;
        for step in 0..self.steps {
            self.blocks.denoise_step(&mut w.inner, step, tune)?;
        }
        Ok(())
    }
    pub fn snapshot(&self, w: &Workspace) -> Result<History> {
        self.blocks.snapshot(&w.inner)
    }
}

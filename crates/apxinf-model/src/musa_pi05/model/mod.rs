//! One PI0.5 dataflow: vision -> prefix/KV -> fixed-step action generation.
//! Precision, fused topology and physical weight representations live in Blocks.
//! ModelRunner and prepare own execution policy, capture, workspaces and input binding.

use super::backend::DeviceBuffer;
use apxinf_core::{Error, Result, Tensor};
use blocks::Blocks;

mod blocks;
mod model;
pub(in crate::musa_pi05) use blocks::L3Policies;
#[cfg(test)]
pub(in crate::musa_pi05) use blocks::L3PolicySnapshot;
pub use model::{
    build_bf16_model, build_int8_dynamic_model, upload_time_embeddings_bf16,
    upload_time_embeddings_int8_dynamic,
};
pub(in crate::musa_pi05) use model::{build_bf16_model_with_policies, build_int8_dynamic_model_with_policies};
pub(super) use model::{ModelOperation, ModelVariant};

use crate::musa_pi05::backend::Context;
use crate::musa_pi05::Pi05ImageLayout;
use apxinf_core::DType;
use std::sync::Arc;
/// Resource requirements supplied by a model implementation. Execution owns
/// allocation, warmup, capture and lifetime; this description allocates nothing.
pub struct WorkspaceRequirements {
    pub bytes: usize,
}
pub trait PrepareBlocks: Blocks + 'static {
    fn backend(&self) -> &Arc<Context>;
    fn workspace_requirements(&self, tokens: usize) -> Result<WorkspaceRequirements>;
    /// RGB bytes to patch-major BF16.
    ///
    /// Both precisions preprocess identically — the image never passes through
    /// a quantized layer — so the traversal lives here and the impls say only
    /// which arena and how many bytes it needs.
    fn preprocess(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: Pi05ImageLayout,
    ) -> Result<()> {
        use crate::musa_pi05::backend::ops;

        let config = self.config();
        let mut patches = patches.clone();
        ops::gather(
            self.backend(),
            ops::GatherArgs::rgb_to_patches(
                images,
                &mut patches,
                ops::GatherPatchGeometry {
                    views: config.num_views,
                    image_size: config.image_size,
                    patch_size: config.patch_size,
                    nhwc: matches!(layout, Pi05ImageLayout::Nhwc),
                },
            ),
        )
    }
}

impl<B: PrepareBlocks> Pi05Model<B> {
    pub(in crate::musa_pi05) fn backend(&self) -> &Arc<Context> {
        self.blocks.backend()
    }
    pub(in crate::musa_pi05) fn config(&self) -> &crate::musa_pi05::Pi05Config {
        self.blocks.config()
    }
    pub(in crate::musa_pi05) fn workspace_requirements(
        &self,
        tokens: usize,
    ) -> Result<WorkspaceRequirements> {
        self.blocks.workspace_requirements(tokens)
    }
}

pub use blocks::bf16::{
    action_layer_bf16, language_layer_bf16, vision_layer_bf16, vision_patch_embed_bf16,
    Bf16ActionLayerOutput, Bf16LanguageLayerOutput,
};
pub use blocks::int8_dynamic::{
    action_layer_int8_dynamic, language_layer_int8_dynamic, vision_layer_int8_dynamic,
    vision_patch_embed_int8_dynamic, Int8DynamicActionLayerOutput, Int8DynamicLanguageLayerOutput,
};
pub use blocks::Bf16PrefixKvCache;
pub use blocks::Int8DynamicPrefixKvCache;
pub type Bf16Model = std::sync::Arc<Pi05Model<blocks::Bf16Blocks>>;
pub type Int8DynamicModel = std::sync::Arc<Pi05Model<blocks::Int8DynamicBlocks>>;

impl<B: PrepareBlocks> Pi05Model<B> {
    pub(in crate::musa_pi05) fn preprocess(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: Pi05ImageLayout,
    ) -> Result<()> {
        self.blocks.preprocess(images, patches, layout)
    }
}

/// Every activation, input and output of this port is BF16.
///
/// The precision lives in the linear layers' representation, not in the tensors
/// that flow between them; a second precision would have to change this const
/// and the kernels behind it, not add a branch at each call site.
pub(in crate::musa_pi05) const ACTIVATION_DTYPE: DType = DType::BF16;

pub struct Pi05Model<B: Blocks> {
    blocks: B,
    /// Every step's modulation, built once by [`Self::new`].
    ///
    /// The flow schedule comes from config and the projection weights from the
    /// checkpoint, so this is one of the model's own constants — not a request
    /// cache, and it cannot go stale. Holding it here is also what lets a
    /// captured graph read it: these are ordinary driver allocations, which is
    /// only safe because something owns them for the life of the graph, and the
    /// captured graph owning this model is that something.
    modulation: Vec<B::StepModulation>,
}

impl<B: Blocks> Pi05Model<B> {
    /// Assemble the model together with its per-step constants.
    ///
    /// This is the only place modulation is computed. It runs while the loader
    /// has no traversal workspace bound, so the tensors land in ordinary device
    /// memory — an arena allocation would not survive the reset at the head of
    /// the next traversal, and a captured graph would replay against whatever
    /// took its addresses.
    pub fn new(blocks: B, embeddings: &[Tensor]) -> Result<Self> {
        let modulation = blocks.compute_modulation(embeddings)?;
        Ok(Self { blocks, modulation })
    }

    pub fn modulation(&self) -> &[B::StepModulation] {
        &self.modulation
    }

    pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
        self.blocks.vision(patches, false)
    }

    pub fn embed_prefix(&self, vision: &Tensor, ids: &DeviceBuffer, count: usize) -> Result<Tensor> {
        self.blocks.embed_prefix(vision, ids, count)
    }
    pub fn prefix_forward(&self, prefix: &Tensor) -> Result<B::Prefix> {
        self.blocks.prefix(prefix)
    }

    pub fn denoise_all_steps(&self, noise: &Tensor, prefix: &B::Prefix) -> Result<Tensor> {
        let config = self.blocks.config();
        if self.modulation.len() != config.num_flow_steps {
            return Err(Error::Other(format!(
                "PI0.5 expected {} precomputed modulation sets, got {}",
                config.num_flow_steps,
                self.modulation.len()
            )));
        }
        let mut state = noise.clone();
        let dt = -config.flow_start_time / config.num_flow_steps as f32;
        for modulation in &self.modulation {
            state = self
                .blocks
                .step_with_modulation(&state, modulation, prefix, dt)?;
        }
        Ok(state)
    }

    fn infer_impl(
        &self,
        patches: &Tensor,
        ids: &DeviceBuffer,
        count: usize,
        noise: &Tensor,
        native: bool,
    ) -> Result<Tensor> {
        let vision = self.blocks.vision(patches, native)?;
        let prefix = self.blocks.embed_prefix(&vision, ids, count)?;
        let prefix = self.blocks.prefix(&prefix)?;
        self.denoise_all_steps(noise, &prefix)
    }

    pub fn infer(
        &self,
        patches: &Tensor,
        ids: &DeviceBuffer,
        count: usize,
        noise: &Tensor,
    ) -> Result<Tensor> {
        self.infer_impl(patches, ids, count, noise, false)
    }

    pub fn infer_native(
        &self,
        patches: &Tensor,
        ids: &DeviceBuffer,
        count: usize,
        noise: &Tensor,
    ) -> Result<Tensor> {
        self.infer_impl(patches, ids, count, noise, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::musa_pi05::Pi05Config;
    use apxinf_core::DType;
    use std::{cell::RefCell, rc::Rc};

    struct Probe {
        config: Pi05Config,
        events: Rc<RefCell<Vec<&'static str>>>,
    }
    fn tensor() -> Tensor {
        Tensor::zeros((1, 1), DType::BF16)
    }
    impl Probe {
        fn record(&self, event: &'static str) {
            self.events.borrow_mut().push(event);
        }
    }
    impl Blocks for Probe {
        type Prefix = ();
        type StepModulation = ();
        fn config(&self) -> &Pi05Config {
            &self.config
        }
        fn vision(&self, _: &Tensor, native: bool) -> Result<Tensor> {
            self.record(if native { "native vision" } else { "vision" });
            Ok(tensor())
        }
        fn embed_prefix(&self, _: &Tensor, _: &DeviceBuffer, _: usize) -> Result<Tensor> {
            self.record("embed");
            Ok(tensor())
        }
        fn prefix(&self, _: &Tensor) -> Result<()> {
            self.record("prefix");
            Ok(())
        }
        fn compute_modulation(&self, _: &[Tensor]) -> Result<Vec<()>> {
            self.record("modulation");
            Ok(vec![(); 2])
        }
        fn step_with_modulation(&self, _: &Tensor, _: &(), _: &(), dt: f32) -> Result<Tensor> {
            assert_eq!(dt, -0.5);
            self.record("step");
            Ok(tensor())
        }
    }

    fn probed(num_flow_steps: usize) -> (Pi05Model<Probe>, Rc<RefCell<Vec<&'static str>>>) {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut config = Pi05Config::default();
        config.num_flow_steps = num_flow_steps;
        config.flow_start_time = 1.0;
        let model = Pi05Model::new(
            Probe {
                config,
                events: events.clone(),
            },
            &[tensor(), tensor()],
        )
        .unwrap();
        (model, events)
    }

    /// The whole point of holding modulation on the model: it is built while the
    /// model is assembled and no traversal ever recomputes it, captured ones
    /// included.
    #[test]
    fn modulation_is_built_once_at_construction() {
        let (model, events) = probed(2);
        assert_eq!(*events.borrow(), vec!["modulation"]);
        assert_eq!(model.modulation().len(), 2);

        let ids = DeviceBuffer::alloc_zeros(4, 0).unwrap();
        for _ in 0..3 {
            model.infer(&tensor(), &ids, 1, &tensor()).unwrap();
        }
        assert_eq!(
            *events.borrow(),
            vec![
                "modulation",
                "vision",
                "embed",
                "prefix",
                "step",
                "step",
                "vision",
                "embed",
                "prefix",
                "step",
                "step",
                "vision",
                "embed",
                "prefix",
                "step",
                "step",
            ]
        );
    }

    #[test]
    fn native_infer_skips_the_host_patchify_step() {
        let (model, events) = probed(2);
        events.borrow_mut().clear();
        let ids = DeviceBuffer::alloc_zeros(4, 0).unwrap();
        model.infer_native(&tensor(), &ids, 1, &tensor()).unwrap();
        assert_eq!(
            *events.borrow(),
            vec!["native vision", "embed", "prefix", "step", "step"]
        );
    }

    #[test]
    fn a_modulation_table_that_does_not_cover_the_schedule_is_refused() {
        let (model, _) = probed(3);
        let ids = DeviceBuffer::alloc_zeros(4, 0).unwrap();
        assert!(model.infer(&tensor(), &ids, 1, &tensor()).is_err());
    }
}

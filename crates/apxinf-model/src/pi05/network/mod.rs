//! One PI0.5 dataflow: vision -> prefix/KV -> fixed-step action generation.
//! Precision, fused topology and physical weight representations live in Blocks.
//! Session and prepare own execution policy, capture, workspaces and input binding.

use super::backend::DeviceBuffer as CudaBuffer;
use apxinf_core::{Error, Result, Tensor};
use blocks::Blocks;

mod blocks;
mod calibration;
mod compute;
pub use calibration::Pi05CalibrationObserver;
pub use compute::{
    build_bf16_network, build_fp8_static_network, build_int8_dynamic_network,
    upload_time_embeddings_bf16, upload_time_embeddings_fp8_static,
    upload_time_embeddings_int8_dynamic,
};
pub(super) use compute::{LoadedCompute, NetworkOperation};

use crate::pi05::backend::{DeviceBuffer, RuntimeBackend};
use crate::pi05::Pi05ImageLayout;
use apxinf_core::DType;
use std::sync::Arc;
/// Resource requirements supplied by a compute implementation. Execution owns
/// allocation, warmup, capture and lifetime; this description allocates nothing.
pub struct WorkspaceRequirements {
    pub bytes: usize,
    pub fp8_scratch: Option<(usize, usize)>,
}
pub trait PrepareBlocks: Blocks + 'static {
    fn backend(&self) -> &Arc<RuntimeBackend>;
    fn workspace_requirements(&self, tokens: usize) -> Result<WorkspaceRequirements>;
    fn raw_patch_dtype(&self) -> DType;
    fn preprocess(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: Pi05ImageLayout,
    ) -> Result<()>;
}

impl<B: PrepareBlocks> Pi05Network<B> {
    pub(in crate::pi05) fn backend(&self) -> &Arc<RuntimeBackend> {
        self.blocks.backend()
    }
    pub(in crate::pi05) fn config(&self) -> &crate::pi05::Pi05Config {
        self.blocks.config()
    }
    pub(in crate::pi05) fn workspace_requirements(
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
pub use blocks::fp8_static::{
    action_layer_fp8_static, language_layer_fp8_static, vision_layer_fp8_static,
    vision_patch_embed_fp8_static, vision_patch_embed_fp8_static_native,
    vision_qkv_packed_from_env, Fp8StaticActionLayerOutput, Fp8StaticLanguageLayerOutput,
};
pub use blocks::int8_dynamic::{
    action_layer_int8_dynamic, language_layer_int8_dynamic, vision_layer_int8_dynamic,
    vision_patch_embed_int8_dynamic, Int8DynamicActionLayerOutput, Int8DynamicLanguageLayerOutput,
};
pub use blocks::{Bf16PrefixKvCache, Fp8StaticPrefixKvCache, Int8DynamicPrefixKvCache};
pub type Bf16Network = std::sync::Arc<Pi05Network<blocks::Bf16Blocks>>;
pub type Fp8StaticNetwork = std::sync::Arc<Pi05Network<blocks::Fp8StaticBlocks>>;
pub type Int8DynamicNetwork = std::sync::Arc<Pi05Network<blocks::Int8DynamicBlocks>>;

impl<B: PrepareBlocks> Pi05Network<B> {
    pub(in crate::pi05) fn preprocess(
        &self,
        images: &DeviceBuffer,
        patches: &Tensor,
        layout: Pi05ImageLayout,
    ) -> Result<()> {
        self.blocks.preprocess(images, patches, layout)
    }
    pub(in crate::pi05) fn raw_patch_dtype(&self) -> DType {
        self.blocks.raw_patch_dtype()
    }
}

pub struct Pi05Network<B: Blocks> {
    blocks: B,
}

impl<B: Blocks> Pi05Network<B> {
    pub fn from_blocks(blocks: B) -> Self {
        Self { blocks }
    }

    pub fn encode_vision(&self, patches: &Tensor) -> Result<Tensor> {
        self.blocks.vision(patches, false)
    }

    pub fn embed_prefix(&self, vision: &Tensor, ids: &CudaBuffer, count: usize) -> Result<Tensor> {
        self.blocks.embed_prefix(vision, ids, count)
    }
    pub fn prefix_forward(&self, prefix: &Tensor) -> Result<B::Prefix> {
        self.blocks.prefix(prefix)
    }
    pub fn prepare_all_styles(&self, embeddings: &[Tensor]) -> Result<Vec<B::Styles>> {
        self.blocks.prepare_styles(embeddings)
    }
    pub fn denoise_step(
        &self,
        state: &Tensor,
        embedding: &Tensor,
        prefix: &B::Prefix,
        dt: f32,
    ) -> Result<Tensor> {
        self.blocks.step(state, embedding, prefix, dt)
    }

    pub fn denoise_all_steps_with_styles(
        &self,
        noise: &Tensor,
        styles: &[B::Styles],
        prefix: &B::Prefix,
    ) -> Result<Tensor> {
        let config = self.blocks.config();
        if styles.len() != config.num_flow_steps {
            return Err(Error::Other(format!(
                "π0.5 expected {} precomputed style sets, got {}",
                config.num_flow_steps,
                styles.len()
            )));
        }
        let mut state = noise.clone();
        let dt = -config.flow_start_time / config.num_flow_steps as f32;
        for styles in styles {
            state = self.blocks.step_with_styles(&state, styles, prefix, dt)?;
        }
        Ok(state)
    }
    pub fn denoise_all_steps(
        &self,
        noise: &Tensor,
        embeddings: &[Tensor],
        prefix: &B::Prefix,
    ) -> Result<Tensor> {
        if let Some(styles) = self.blocks.eager_styles(embeddings)? {
            return self.denoise_all_steps_with_styles(noise, &styles, prefix);
        }
        self.denoise_embeddings(noise, embeddings, prefix)
    }

    fn denoise_embeddings(
        &self,
        noise: &Tensor,
        embeddings: &[Tensor],
        prefix: &B::Prefix,
    ) -> Result<Tensor> {
        let config = self.blocks.config();
        if embeddings.len() != config.num_flow_steps {
            return Err(Error::Other(format!(
                "π0.5 expected {} timestep embeddings, got {}",
                config.num_flow_steps,
                embeddings.len()
            )));
        }
        let mut state = noise.clone();
        let dt = -config.flow_start_time / config.num_flow_steps as f32;
        for embedding in embeddings {
            state = self.blocks.step(&state, embedding, prefix, dt)?;
        }
        Ok(state)
    }
    fn infer_impl(
        &self,
        patches: &Tensor,
        ids: &CudaBuffer,
        count: usize,
        noise: &Tensor,
        embeddings: &[Tensor],
        native: bool,
    ) -> Result<Tensor> {
        let styles = self.blocks.eager_styles(embeddings)?;
        let vision = self.blocks.vision(patches, native)?;
        let prefix = self.blocks.embed_prefix(&vision, ids, count)?;
        let prefix = self.blocks.prefix(&prefix)?;
        match styles {
            Some(styles) => self.denoise_all_steps_with_styles(noise, &styles, &prefix),
            None => self.denoise_embeddings(noise, embeddings, &prefix),
        }
    }
    pub fn infer(
        &self,
        patches: &Tensor,
        ids: &CudaBuffer,
        count: usize,
        noise: &Tensor,
        embeddings: &[Tensor],
    ) -> Result<Tensor> {
        self.infer_impl(patches, ids, count, noise, embeddings, false)
    }
    pub fn infer_native(
        &self,
        patches: &Tensor,
        ids: &CudaBuffer,
        count: usize,
        noise: &Tensor,
        embeddings: &[Tensor],
    ) -> Result<Tensor> {
        self.infer_impl(patches, ids, count, noise, embeddings, true)
    }
    fn infer_styles_impl(
        &self,
        patches: &Tensor,
        ids: &CudaBuffer,
        count: usize,
        noise: &Tensor,
        styles: &[B::Styles],
        native: bool,
    ) -> Result<Tensor> {
        let vision = self.blocks.vision(patches, native)?;
        let prefix = self.blocks.embed_prefix(&vision, ids, count)?;
        let prefix = self.blocks.prefix(&prefix)?;
        self.denoise_all_steps_with_styles(noise, styles, &prefix)
    }
    pub fn infer_with_styles(
        &self,
        patches: &Tensor,
        ids: &CudaBuffer,
        count: usize,
        noise: &Tensor,
        styles: &[B::Styles],
    ) -> Result<Tensor> {
        self.infer_styles_impl(patches, ids, count, noise, styles, false)
    }
    pub fn infer_with_native_styles(
        &self,
        patches: &Tensor,
        ids: &CudaBuffer,
        count: usize,
        noise: &Tensor,
        styles: &[B::Styles],
    ) -> Result<Tensor> {
        self.infer_styles_impl(patches, ids, count, noise, styles, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pi05::Pi05Config;
    use apxinf_core::DType;
    use std::{cell::RefCell, rc::Rc};

    struct Probe<const PRECOMPUTE: bool> {
        config: Pi05Config,
        events: Rc<RefCell<Vec<&'static str>>>,
    }
    fn tensor() -> Tensor {
        Tensor::zeros((1, 1), DType::BF16)
    }
    impl<const P: bool> Probe<P> {
        fn record(&self, event: &'static str) {
            self.events.borrow_mut().push(event);
        }
    }
    impl<const P: bool> Blocks for Probe<P> {
        type Prefix = ();
        type Styles = ();
        fn config(&self) -> &Pi05Config {
            &self.config
        }
        fn vision(&self, _: &Tensor, native: bool) -> Result<Tensor> {
            self.record(if native { "native vision" } else { "vision" });
            Ok(tensor())
        }
        fn embed_prefix(&self, _: &Tensor, _: &CudaBuffer, _: usize) -> Result<Tensor> {
            self.record("embed");
            Ok(tensor())
        }
        fn prefix(&self, _: &Tensor) -> Result<()> {
            self.record("prefix");
            Ok(())
        }
        fn prepare_styles(&self, _: &[Tensor]) -> Result<Vec<()>> {
            self.record("styles");
            Ok(vec![(); 2])
        }
        fn eager_styles(&self, e: &[Tensor]) -> Result<Option<Vec<()>>> {
            if P {
                self.prepare_styles(e).map(Some)
            } else {
                Ok(None)
            }
        }
        fn step(&self, _: &Tensor, _: &Tensor, _: &(), dt: f32) -> Result<Tensor> {
            assert_eq!(dt, -0.5);
            self.record("inline step");
            Ok(tensor())
        }
        fn step_with_styles(&self, _: &Tensor, _: &(), _: &(), dt: f32) -> Result<Tensor> {
            assert_eq!(dt, -0.5);
            self.record("prepared step");
            Ok(tensor())
        }
    }

    fn check_order<const PRECOMPUTE: bool>() {
        let events = Rc::new(RefCell::new(Vec::new()));
        let mut config = Pi05Config::default();
        config.num_flow_steps = 2;
        config.flow_start_time = 1.0;
        let network = Pi05Network::from_blocks(Probe::<PRECOMPUTE> {
            config,
            events: events.clone(),
        });
        let ids = CudaBuffer::alloc_zeros(4, 0).unwrap();
        let embeddings = [tensor(), tensor()];
        network
            .infer(&tensor(), &ids, 1, &tensor(), &embeddings)
            .unwrap();
        let expected = if PRECOMPUTE {
            vec![
                "styles",
                "vision",
                "embed",
                "prefix",
                "prepared step",
                "prepared step",
            ]
        } else {
            vec!["vision", "embed", "prefix", "inline step", "inline step"]
        };
        assert_eq!(*events.borrow(), expected);
        events.borrow_mut().clear();
        network
            .infer_with_native_styles(&tensor(), &ids, 1, &tensor(), &[(), ()])
            .unwrap();
        assert_eq!(
            *events.borrow(),
            vec![
                "native vision",
                "embed",
                "prefix",
                "prepared step",
                "prepared step"
            ]
        );
        assert!(network
            .denoise_all_steps_with_styles(&tensor(), &[()], &())
            .is_err());
    }
    #[test]
    fn shared_network_preserves_precomputed_eager_order() {
        check_order::<true>();
    }
    #[test]
    fn shared_network_preserves_inline_eager_order() {
        check_order::<false>();
    }
}

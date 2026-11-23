//! Shared PI0.5 preparation and captured resource ownership.
use crate::pi05::backend::{
    self, capture as capture_cuda_graph, ops, transfers, Context, DeviceBuffer as CudaBuffer,
    ExecutionSession, GraphWorkspace, ProviderWorkspace,
};
use crate::pi05::model::Pi05Model;
use crate::pi05::model::{ModelOperation, ModelVariant, PrepareBlocks, WorkspaceRequirements};
use crate::pi05::{Pi05Config, Pi05ImageLayout};
use apxinf_core::{Error, Graph, Result, Tensor};
use std::sync::Arc;

fn allocate_session(
    requirements: &WorkspaceRequirements,
    device: usize,
    scratch: &ProviderWorkspace,
) -> Result<ExecutionSession> {
    Ok(ExecutionSession::with_provider_workspace(
        GraphWorkspace::new(arena_capacity(requirements.bytes)?, device)?,
        scratch.clone(),
    ))
}

/// Environment variable that replaces the static workspace capacity of a PI0.5 session, in MiB.
pub(in crate::pi05) const ARENA_MIB_ENV: &str = "APXINF_PI05_ARENA_MIB";

/// Capacity of the session arena for a model whose conservative bound is `bound` bytes.
///
/// The bound counts every intermediate as live at once. On the INT8 path it is twice the BF16
/// bound (7660 MiB for two views, a horizon of ten and ten prompt tokens), which leaves no room on
/// an 8 GB card where the weights take 4024 MiB. Setting `APXINF_PI05_ARENA_MIB` replaces the bound
/// with a measured capacity. A value that is not a positive integer is an error, not a silent
/// fallback to the bound, so a typo cannot be mistaken for a setting that took effect. A capacity
/// that is too small fails the first request with "static inference workspace exhausted", which
/// names the bytes needed.
pub(in crate::pi05) fn arena_capacity(bound: usize) -> Result<usize> {
    arena_capacity_from(bound, std::env::var(ARENA_MIB_ENV).ok().as_deref())
}

fn arena_capacity_from(bound: usize, mib: Option<&str>) -> Result<usize> {
    let Some(text) = mib else {
        return Ok(bound);
    };
    let invalid = || {
        Error::Other(format!(
            "{ARENA_MIB_ENV}={text:?} is not a positive number of MiB that fits in the address space"
        ))
    };
    let mib: usize = text.trim().parse().map_err(|_| invalid())?;
    if mib == 0 {
        return Err(invalid());
    }
    mib.checked_mul(1 << 20).ok_or_else(invalid)
}

pub struct CapturedGraph {
    graph: Box<dyn Graph>,
    output: Tensor,
    patches: Tensor,
    raw_images: Option<CudaBuffer>,
    raw_image_layout: Option<Pi05ImageLayout>,
    noise: Tensor,
    token_ids: CudaBuffer,
    token_count: usize,
    backend: Arc<Context>,
    // Retain every fixed weight referenced by the captured computation.
    _fixed: Box<dyn std::any::Any>,
    session: ExecutionSession,
}

impl CapturedGraph {
    pub fn replay(&self) -> Result<()> {
        self.graph.replay()
    }

    pub fn replay_and_synchronize(&self) -> Result<()> {
        self.graph.replay()?;
        backend::synchronize(&self.backend)
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    pub fn raw_image_layout(&self) -> Option<Pi05ImageLayout> {
        self.raw_image_layout
    }

    fn update_tokens(&self, token_ids: &[u32]) -> Result<()> {
        let bytes = token_ids
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        self.token_ids.copy_from_host(&bytes).map_err(Error::Cuda)
    }

    pub fn update_inputs(&self, patches: &Tensor, token_ids: &[u32], noise: &Tensor) -> Result<()> {
        self.update_inputs_without_noise(patches, token_ids)?;
        transfers::copy_cpu_to_cuda(noise, &self.noise)
    }

    pub fn update_inputs_without_noise(&self, patches: &Tensor, token_ids: &[u32]) -> Result<()> {
        if self.raw_images.is_some() {
            return Err(Error::Other(
                "π0.5 graph uses raw RGB input; call update_raw_image_inputs".into(),
            ));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 graph expects {} token IDs, got {}",
                self.token_count,
                token_ids.len()
            )));
        }
        backend::synchronize(&self.backend)?;
        transfers::copy_cpu_to_cuda(patches, &self.patches)?;
        self.update_tokens(token_ids)
    }

    pub fn update_raw_image_inputs(
        &self,
        images: &[u8],
        token_ids: &[u32],
        noise: &Tensor,
    ) -> Result<()> {
        self.update_raw_image_inputs_without_noise(images, token_ids)?;
        transfers::copy_cpu_to_cuda(noise, &self.noise)
    }

    pub fn update_raw_image_inputs_without_noise(
        &self,
        images: &[u8],
        token_ids: &[u32],
    ) -> Result<()> {
        let raw_images = self.raw_images.as_ref().ok_or_else(|| {
            Error::Other("π0.5 graph uses patch input; call update_inputs".into())
        })?;
        if images.len() != raw_images.len() {
            return Err(Error::Other(format!(
                "π0.5 graph expects {} raw image bytes, got {}",
                raw_images.len(),
                images.len()
            )));
        }
        if token_ids.len() != self.token_count {
            return Err(Error::Other(format!(
                "π0.5 graph expects {} token IDs, got {}",
                self.token_count,
                token_ids.len()
            )));
        }
        backend::synchronize(&self.backend)?;
        raw_images.copy_from_host(images).map_err(Error::Cuda)?;
        self.update_tokens(token_ids)
    }

    pub fn workspace_bytes(&self) -> usize {
        self.session.workspace().capacity()
    }

    pub fn workspace_used_bytes(&self) -> usize {
        self.session.workspace().used()
    }
}

struct CaptureBuilder<'a, B: PrepareBlocks> {
    model: &'a Arc<Pi05Model<B>>,
    backend: &'a Arc<Context>,
    config: &'a Pi05Config,
    scratch: &'a ProviderWorkspace,
}
impl<B: PrepareBlocks> CaptureBuilder<'_, B> {
    fn ctx(&self) -> &Context {
        self.backend
    }
    #[allow(clippy::too_many_arguments)]
    fn infer_captured_inputs(
        &self,
        patches: &Tensor,
        raw_images: Option<&CudaBuffer>,
        raw_image_layout: Option<Pi05ImageLayout>,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        modulation: &[B::StepModulation],
    ) -> Result<Tensor> {
        match (raw_images, raw_image_layout) {
            (None, None) => {
                self.model
                    .infer_with_modulation(patches, token_ids, token_count, noise, modulation)
            }
            (Some(images), Some(layout)) => {
                self.model.preprocess(images, patches, layout)?;
                self.model.infer_with_native_modulation(
                    patches,
                    token_ids,
                    token_count,
                    noise,
                    modulation,
                )
            }
            _ => Err(Error::Other(
                "π0.5 raw image capture state is inconsistent".into(),
            )),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_infer_impl(
        &self,
        patches: Tensor,
        raw_images: Option<CudaBuffer>,
        raw_image_layout: Option<Pi05ImageLayout>,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<CapturedGraph> {
        let backend = self.backend;
        if raw_images.is_some() != raw_image_layout.is_some() {
            return Err(Error::Other(
                "π0.5 raw image capture state is inconsistent".into(),
            ));
        }
        let modulation = self.model.prepare_all_modulation(time_embeddings)?;
        backend::synchronize(backend)?;
        let session = allocate_session(
            &self.model.workspace_requirements(token_count)?,
            self.ctx().device_id(),
            self.scratch,
        )?;
        let eager_output = ops::prepare_with_session(&session, || {
            self.infer_captured_inputs(
                &patches,
                raw_images.as_ref(),
                raw_image_layout,
                token_ids,
                token_count,
                noise,
                &modulation,
            )
        })?;
        backend::synchronize(backend)?;
        drop(eager_output);

        let captured_output = std::cell::RefCell::new(None);
        let graph = capture_cuda_graph(self.ctx(), || {
            let output = ops::with_session(&session, || {
                self.infer_captured_inputs(
                    &patches,
                    raw_images.as_ref(),
                    raw_image_layout,
                    token_ids,
                    token_count,
                    noise,
                    &modulation,
                )
            })?;
            captured_output.replace(Some(output));
            Ok(())
        })?;
        let output = captured_output
            .into_inner()
            .ok_or_else(|| Error::Other("PI0.5 graph capture produced no output tensor".into()))?;
        Ok(CapturedGraph {
            graph: Box::new(graph),
            output,
            patches,
            raw_images,
            raw_image_layout,
            noise: noise.clone(),
            token_ids: token_ids.clone(),
            token_count,
            backend: Arc::clone(&self.backend),
            _fixed: Box::new((Arc::clone(&self.model), modulation)),
            session,
        })
    }

    pub fn capture_infer(
        &self,
        patches: &Tensor,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<CapturedGraph> {
        self.capture_infer_impl(
            patches.clone(),
            None,
            None,
            token_ids,
            token_count,
            noise,
            time_embeddings,
        )
    }

    pub fn capture_infer_rgb_u8(
        &self,
        layout: Pi05ImageLayout,
        token_ids: &CudaBuffer,
        token_count: usize,
        noise: &Tensor,
        time_embeddings: &[Tensor],
    ) -> Result<CapturedGraph> {
        let backend = &self.backend;
        let raw_image_bytes = self
            .config
            .num_views
            .checked_mul(3)
            .and_then(|n| n.checked_mul(self.config.image_size))
            .and_then(|n| n.checked_mul(self.config.image_size))
            .ok_or_else(|| Error::Other("PI0.5 raw image size overflow".into()))?;
        let raw_images = CudaBuffer::alloc_zeros(raw_image_bytes, self.ctx().device_id())
            .map_err(Error::Cuda)?;
        let patch_rows = self.config.num_views * self.config.patches_per_view();
        let patch_width = 3 * self.config.patch_size * self.config.patch_size;
        let patches = backend::to_device(
            backend,
            &Tensor::zeros(vec![patch_rows, patch_width], self.model.raw_patch_dtype()),
        )?;
        self.capture_infer_impl(
            patches,
            Some(raw_images),
            Some(layout),
            token_ids,
            token_count,
            noise,
            time_embeddings,
        )
    }
}

pub(super) enum CaptureInput<'a> {
    Patches(&'a Tensor),
    Rgb(Pi05ImageLayout),
}
pub(super) fn capture<B: PrepareBlocks>(
    model: &Arc<Pi05Model<B>>,
    input: CaptureInput<'_>,
    tokens: &CudaBuffer,
    count: usize,
    noise: &Tensor,
    embeddings: &[Tensor],
    scratch: &ProviderWorkspace,
) -> Result<CapturedGraph> {
    let builder = CaptureBuilder {
        model,
        backend: model.backend(),
        config: model.config(),
        scratch,
    };
    match input {
        CaptureInput::Rgb(layout) => {
            builder.capture_infer_rgb_u8(layout, tokens, count, noise, embeddings)
        }
        CaptureInput::Patches(patches) => {
            builder.capture_infer(patches, tokens, count, noise, embeddings)
        }
    }
}
/// Prepare a low-level graph with caller-owned stable patch/token/noise tensors.
/// Their storage is retained; sizes and addresses cannot change between replays.
/// These standalone entry points use a fresh [`ProviderWorkspace`], so operator
/// scratch is not shared across separately captured graphs; use
/// [`Pi05ModelRunner`](super::Pi05ModelRunner) to share one workspace across
/// profiles.
pub fn capture_patches<B: PrepareBlocks>(
    model: &Arc<Pi05Model<B>>,
    patches: &Tensor,
    tokens: &CudaBuffer,
    count: usize,
    noise: &Tensor,
    embeddings: &[Tensor],
) -> Result<CapturedGraph> {
    capture(
        model,
        CaptureInput::Patches(patches),
        tokens,
        count,
        noise,
        embeddings,
        &ProviderWorkspace::default(),
    )
}
/// Prepare a graph that owns RGB input storage and device preprocessing.
pub fn capture_rgb<B: PrepareBlocks>(
    model: &Arc<Pi05Model<B>>,
    layout: Pi05ImageLayout,
    tokens: &CudaBuffer,
    count: usize,
    noise: &Tensor,
    embeddings: &[Tensor],
) -> Result<CapturedGraph> {
    capture(
        model,
        CaptureInput::Rgb(layout),
        tokens,
        count,
        noise,
        embeddings,
        &ProviderWorkspace::default(),
    )
}

pub(super) fn capture_loaded(
    model: &ModelVariant,
    scratch: &ProviderWorkspace,
    spec: &crate::vla::InferenceSpec,
    patches: &Tensor,
    tokens: &CudaBuffer,
    noise: &Tensor,
) -> Result<CapturedGraph> {
    struct CaptureOperation<'a> {
        input: CaptureInput<'a>,
        scratch: &'a ProviderWorkspace,
        tokens: &'a CudaBuffer,
        count: usize,
        noise: &'a Tensor,
    }
    impl ModelOperation for CaptureOperation<'_> {
        type Output = CapturedGraph;
        fn run<B: PrepareBlocks>(
            self,
            model: &Arc<Pi05Model<B>>,
            embeddings: &[Tensor],
        ) -> Result<Self::Output> {
            capture(
                model,
                self.input,
                self.tokens,
                self.count,
                self.noise,
                embeddings,
                self.scratch,
            )
        }
    }
    let input = match spec.image_layout {
        Some(layout) => CaptureInput::Rgb(super::runner::kernel_image_layout(layout)),
        None => CaptureInput::Patches(patches),
    };
    model.with_model(CaptureOperation {
        input,
        scratch,
        tokens,
        count: spec.token_count,
        noise,
    })
}

#[cfg(test)]
mod tests {
    use super::arena_capacity_from;

    const BOUND: usize = 7660 << 20;

    #[test]
    fn unset_keeps_the_conservative_bound() {
        assert_eq!(arena_capacity_from(BOUND, None).unwrap(), BOUND);
    }

    #[test]
    fn a_positive_value_replaces_the_bound_in_mib() {
        assert_eq!(arena_capacity_from(BOUND, Some("256")).unwrap(), 256 << 20);
        assert_eq!(arena_capacity_from(BOUND, Some(" 91 ")).unwrap(), 91 << 20);
        // A capacity above the bound is allowed: the variable states what the user measured.
        assert_eq!(
            arena_capacity_from(BOUND, Some("9000")).unwrap(),
            9000 << 20
        );
    }

    #[test]
    fn anything_else_is_an_error_naming_the_variable() {
        let bad_values = [
            "",
            "0",
            "-1",
            "256MiB",
            "1.5",
            "abc",
            "99999999999999999999",
        ];
        for bad in bad_values {
            let message = arena_capacity_from(BOUND, Some(bad))
                .unwrap_err()
                .to_string();
            assert!(
                message.contains("APXINF_PI05_ARENA_MIB"),
                "{bad:?}: {message}"
            );
        }
    }

    #[test]
    fn a_value_that_overflows_bytes_is_an_error() {
        assert!(arena_capacity_from(BOUND, Some(&usize::MAX.to_string())).is_err());
    }
}

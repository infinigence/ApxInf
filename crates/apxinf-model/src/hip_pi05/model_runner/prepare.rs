//! The prepared plan: fixed device inputs, the activation arena, and — when
//! the policy asks and the capture succeeds — the forward recorded as one HIP
//! graph launch.
//!
//! Preparation runs the forward once eagerly as a warmup, whatever the
//! policy: the first execution of these shapes performs whatever first-touch
//! work the libraries defer (rocBLAS's per-shape kernel selection, hipBLASLt
//! heuristics, CK's instance check), and recording that into a graph would
//! replay it forever. A graph policy then records the forward as one
//! [`HipGraph`] launch. The preconditions are what the plan was already built
//! on: inputs live at fixed addresses, every activation comes from the arena
//! in a fixed order, and a run never synchronizes before its output is read.
//! Input upload and the RGB-to-patches conversion stay outside the graph: they
//! read host memory the caller rebuilds per request, which a captured copy
//! would replay from a dangling pointer.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use apxinf_core::{DType, Device, Error, NormalGenerator, Result, Tensor};
use apxinf_hip::graph::HipGraph;
use apxinf_hip::native::{self, HipArena, HipDeviceBytes};
use half::bf16;

use super::super::config::Config;
use super::super::model::HipPi05Model;
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent,
    PreparationStatus, PreparedInference, VisionObservation, VlaRequest,
};

/// One fixed-shape plan: device inputs at fixed addresses, the arena, the
/// generator that fills the noise buffer in place, and — when the policy asks
/// and the capture succeeds — the forward recorded as one [`HipGraph`] launch.
pub struct HipPi05Prepared {
    spec: InferenceSpec,
    model: Rc<HipPi05Model>,
    arena: HipArena,
    patches: Tensor,
    images: Option<HipDeviceBytes>,
    token_ids: HipDeviceBytes,
    noise: Tensor,
    normal_generator: RefCell<Box<dyn NormalGenerator>>,
    /// The forward as one launch, with the arena slot it writes. `None` when
    /// the policy did not ask for a graph, or when capture fell back;
    /// [`Self::fallback_reason`] says which.
    graph: RefCell<Option<(HipGraph, Tensor)>>,
    fallback_reason: Option<String>,
    /// Inferences served by replaying the graph. A capture that fell back, or
    /// a policy that never asked, both leave the eager path in use and the
    /// right answer returned — so without a count, a measurement that never
    /// reached the graph is indistinguishable from one that found no gain.
    replays: Cell<usize>,
}

impl HipPi05Prepared {
    pub(in crate::hip_pi05) fn new(
        model: Rc<HipPi05Model>,
        spec: InferenceSpec,
        policy: ExecutionPolicy,
    ) -> Result<Self> {
        spec.validate()?;
        let config = model.config();
        if spec.token_count > config.max_token_len {
            return Err(Error::Other(format!(
                "π0.5 token count {} exceeds maximum {}",
                spec.token_count, config.max_token_len
            )));
        }
        let ctx = model.context();
        let patches = ctx.to_device(&Tensor::zeros(patch_shape(config), DType::BF16))?;
        let noise = ctx.to_device(&Tensor::zeros(noise_shape(config), DType::BF16))?;
        let images = spec
            .image_layout
            .map(|_| HipDeviceBytes::new(ctx, image_bytes(config)))
            .transpose()?;
        let token_ids = HipDeviceBytes::new(ctx, spec.token_count * 4)?;
        // The warmup below runs before any request uploads its ids, and the
        // memory `HipDeviceBytes` hands back is uninitialized. The embedding
        // kernel gathers `table[ids[i] * width]` with no bounds check, so a
        // leftover byte reinterpreted as an id would read far outside the
        // table. Zero the ids so the warmup reads a valid row: id 0, like the
        // zeroed patches and noise around it.
        token_ids.upload_u32(&vec![0u32; spec.token_count])?;
        let normal_generator = native::normal_generator(ctx, noise.clone())?;
        let arena = HipArena::new(ctx, HipPi05Model::arena_bytes(config, spec.token_count))?;
        let mut prepared = Self {
            spec,
            model,
            arena,
            patches,
            images,
            token_ids,
            noise,
            normal_generator: RefCell::new(normal_generator),
            graph: RefCell::new(None),
            fallback_reason: None,
            replays: Cell::new(0),
        };

        // Warm up on the zero-initialized inputs, whatever the policy. The
        // first execution of these shapes runs the libraries' first-touch
        // work — rocBLAS's per-shape kernel selection, CK's instance checks —
        // which must not end up recorded into a graph. The forward is a fixed
        // DAG over shapes, so zeros exercise exactly the kernels a real
        // request does.
        prepared.forward()?;
        let (graph, fallback_reason) = match policy {
            ExecutionPolicy::Eager => (None, None),
            ExecutionPolicy::RequireGraph | ExecutionPolicy::PreferGraph => {
                match prepared.capture() {
                    Ok(graph) => (Some(graph), None),
                    Err(error) => {
                        if policy == ExecutionPolicy::RequireGraph {
                            return Err(error);
                        }
                        // PreferGraph falls back — but only onto a stream that
                        // still works. A failure inside the captured region
                        // latches an error on the stream itself, so probe
                        // before promising eager execution.
                        prepared.model.context().stream_status()?;
                        (None, Some(error.to_string()))
                    }
                }
            }
        };
        *prepared.graph.borrow_mut() = graph;
        prepared.fallback_reason = fallback_reason;
        if let Some(reason) = &prepared.fallback_reason {
            eprintln!("[apxinf] PI0.5 HIP graph capture unavailable, using eager: {reason}");
        }
        // Leave the plan as `new` found it: inputs zeroed, arena at zero, so
        // the first real run's uploads and replay start from the same state
        // the capture recorded against.
        prepared.model.context().synchronize()?;
        prepared.arena.reset();
        Ok(prepared)
    }

    /// The forward against this plan's fixed inputs. What `run` replays and
    /// what the warmup and capture execute.
    fn forward(&self) -> Result<Tensor> {
        self.model.infer(
            &self.arena,
            &self.patches,
            &self.token_ids,
            self.spec.token_count,
            &self.noise,
        )
    }

    /// Record this plan's forward, so later runs replay it as one launch.
    ///
    /// Runs the forward once more under capture. That run's result is
    /// discarded — the warmup already validated the path — but the arena slot
    /// it writes is the slot every replay writes, so it is kept alongside the
    /// graph.
    fn capture(&self) -> Result<(HipGraph, Tensor)> {
        let ctx = self.model.context();
        // The eager warmup left work queued against the arena; capture must
        // not interleave with it.
        ctx.synchronize()?;
        self.arena.reset();
        // SAFETY: the captured region allocates nothing on the device (every
        // activation, the prefix K/V cache, and the attention scratch come
        // from the arena; the weights and modulation constants were
        // allocated at load), copies nothing from host memory, and neither
        // synchronizes nor downloads. Every replay is preceded by
        // `arena.reset()` in `run`, so the addresses the graph records are
        // the addresses it replays against.
        let (graph, output) = unsafe { HipGraph::capture(ctx, || self.forward()) }?;
        Ok((graph, output))
    }

    /// `(replays, graph nodes)`; `(0, 0)` when the plan runs eagerly.
    pub fn graph_usage(&self) -> (usize, usize) {
        (
            self.replays.get(),
            self.graph.borrow().as_ref().map_or(0, |(g, _)| g.nodes()),
        )
    }

    fn bind_inputs(&self, request: &VlaRequest<'_>) -> Result<()> {
        let observation = request.observation;
        let config = self.model.config();
        let ctx = self.model.context();
        match &observation.vision {
            VisionObservation::Patches(patches) => {
                let patches = host_bf16(patches, patch_shape(config), "patches")?;
                native::upload(ctx, &patches, &self.patches)?;
            }
            VisionObservation::RgbU8 { bytes, .. } => {
                let images = self
                    .images
                    .as_ref()
                    .expect("an RGB plan holds an image buffer");
                if bytes.len() != images.len() {
                    return Err(Error::Other(format!(
                        "π0.5 expected {} raw image bytes, got {}",
                        images.len(),
                        bytes.len()
                    )));
                }
                images.upload(bytes)?;
            }
        }
        self.token_ids.upload_u32(&observation.token_ids)?;
        match request.initial_latent {
            InitialLatent::Provided(latent) => {
                let latent = host_bf16(latent, noise_shape(config), "initial latent")?;
                native::upload(ctx, &latent, &self.noise)?;
            }
            InitialLatent::Generate { rng } => {
                self.normal_generator.borrow_mut().generate(rng)?;
            }
        }
        Ok(())
    }
}

impl PreparedInference for HipPi05Prepared {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: if self.graph.borrow().is_some() {
                ExecutionMode::Graph
            } else {
                ExecutionMode::Eager
            },
            fallback_reason: self.fallback_reason.clone(),
        }
    }

    /// The returned action lives in the arena: the next run overwrites it.
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let observation = request.observation;
        observation.validate()?;
        if !self.spec.matches(observation) {
            return Err(Error::Other(format!(
                "prepared π0.5 spec {:?} does not match observation {:?}",
                self.spec,
                observation.inference_spec()
            )));
        }
        // Everything from here to the replay/forward reads and writes only
        // this plan's fixed device buffers, so the graph's recorded addresses
        // stay valid: `bind_inputs` fills them in place, `arena.reset` puts
        // the bump pointer back where the capture started, and the RGB
        // conversion writes the patch buffer the forward reads.
        self.bind_inputs(request)?;
        let config = self.model.config();
        let ctx = self.model.context();
        self.arena.reset();
        if let (Some(images), Some(layout)) = (&self.images, self.spec.image_layout) {
            native::rgb_to_patches(
                ctx,
                images,
                kernel_layout(layout),
                config.num_views,
                config.image_size,
                config.patch_size,
                &self.patches,
            )?;
        }
        if let Some((graph, output)) = self.graph.borrow().as_ref() {
            self.replays.set(self.replays.get() + 1);
            graph.launch(ctx)?;
            return Ok(Action::new(output.clone()));
        }
        let actions = self.forward()?;
        Ok(Action::new(actions))
    }
}

fn kernel_layout(layout: ImageLayout) -> native::ImageLayout {
    match layout {
        ImageLayout::Nhwc => native::ImageLayout::Nhwc,
        ImageLayout::Nchw => native::ImageLayout::Nchw,
    }
}

fn patch_shape(config: &Config) -> Vec<usize> {
    vec![
        config.num_views * config.patches_per_view(),
        3 * config.patch_size * config.patch_size,
    ]
}

fn noise_shape(config: &Config) -> Vec<usize> {
    vec![config.action_horizon, config.action_dim]
}

fn image_bytes(config: &Config) -> usize {
    config.num_views * 3 * config.image_size * config.image_size
}

/// A host tensor of `shape` as BF16, converting from any float dtype.
fn host_bf16(tensor: &Tensor, shape: Vec<usize>, label: &str) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other(format!("π0.5 {label} must be a CPU tensor")));
    }
    if tensor.shape().dims() != shape {
        return Err(Error::Other(format!(
            "π0.5 {label} shape {:?} does not match {shape:?}",
            tensor.shape().dims()
        )));
    }
    if tensor.dtype() == DType::BF16 {
        return Ok(tensor.clone());
    }
    let values: Vec<bf16> = tensor
        .to_f32_vec()?
        .into_iter()
        .map(bf16::from_f32)
        .collect();
    Tensor::from_bf16(shape, &values)
}

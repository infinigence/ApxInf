//! Fixed buffers, explicit request binding, RNG, and whole-model CUDA Graph
//! ownership. Forward computation is exclusively in FastWamModel.
use crate::cuda_fastwam::weights::host_tensor;
use crate::cuda_fastwam::{
    backend::{
        self, ops, sampling, transfers, Context, DeviceBuffer, ExecutionSession, PreparedPhase,
    },
    model::{FastWamModel, Inputs},
};
use crate::vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent,
    PreparationStatus, PreparedInference, VisionObservation, VlaRequest,
};
use apxinf_core::{DType, Error, NormalGenerator, Result, Shape, Tensor};
use std::{cell::RefCell, rc::Rc};

use super::runner::{check_spec, validate, FastWamModelRunner};
impl Inputs {
    fn allocate(ctx: &Context, layout: ImageLayout) -> Result<Self> {
        let zeros =
            |dims: Vec<usize>, dtype| backend::upload(ctx, &Tensor::zeros(Shape::new(dims), dtype));
        Ok(Self {
            rgb: DeviceBuffer::alloc_zeros(224 * 448 * 3, ctx.device_id()).map_err(Error::Cuda)?,
            image_layout: match layout {
                ImageLayout::Nhwc => ops::ImageMemoryLayout::Nhwc,
                ImageLayout::Nchw => ops::ImageMemoryLayout::Nchw,
            },
            ids: zeros(vec![128], DType::I32)?,
            padding: zeros(vec![128], DType::F32)?,
            mask: zeros(vec![128], DType::BF16)?,
            state: zeros(vec![1, 8], DType::BF16)?,
            noise: zeros(vec![32, 7], DType::BF16)?,
        })
    }
    fn bind(
        &self,
        request: &VlaRequest<'_>,
        rng: &RefCell<Box<dyn NormalGenerator>>,
    ) -> Result<()> {
        let VisionObservation::RgbU8 { bytes, .. } = &request.observation.vision else {
            unreachable!()
        };
        self.rgb.copy_from_host(bytes).map_err(Error::Cuda)?;
        transfers::copy_cpu_to_cuda(
            &Tensor::from_i32(
                vec![128],
                &request
                    .observation
                    .token_ids
                    .iter()
                    .map(|&x| x as i32)
                    .collect::<Vec<_>>(),
            )?,
            &self.ids,
        )?;
        let mask = request
            .metadata
            .attention_mask
            .map(|x| x.to_vec())
            .unwrap_or_else(|| {
                request
                    .observation
                    .token_ids
                    .iter()
                    .map(|&id| u8::from(id != 0))
                    .collect()
            });
        let mask_values = mask.iter().map(|&x| x as f32).collect::<Vec<_>>();
        let padding = mask
            .iter()
            .map(|&x| if x == 0 { -3.3895314e38f32 } else { 0. })
            .collect::<Vec<_>>();
        transfers::copy_cpu_to_cuda(
            &host_tensor(vec![128], &mask_values, DType::BF16)?,
            &self.mask,
        )?;
        transfers::copy_cpu_to_cuda(&Tensor::from_f32(vec![128], &padding)?, &self.padding)?;
        transfers::copy_cpu_to_cuda(
            &host_tensor(
                vec![1, 8],
                &request.observation.state.as_ref().unwrap().to_f32_vec()?,
                DType::BF16,
            )?,
            &self.state,
        )?;
        match request.initial_latent {
            InitialLatent::Provided(x) => transfers::copy_cpu_to_cuda(
                &host_tensor(vec![32, 7], &x.to_f32_vec()?, DType::BF16)?,
                &self.noise,
            )?,
            InitialLatent::Generate { rng: key } => {
                rng.borrow_mut().generate(key)?;
            }
        }
        Ok(())
    }
}
enum Execution {
    Eager(ExecutionSession),
    Graph(PreparedPhase),
}
pub(super) struct Plan {
    execution: Execution,
    inputs: Inputs,
    output: Tensor,
    rng: RefCell<Box<dyn NormalGenerator>>,
    spec: InferenceSpec,
    ctx: Rc<Context>,
    model: Rc<FastWamModel>,
    fallback: Option<String>,
}
impl Plan {
    pub(super) fn build(
        runner: &FastWamModelRunner,
        spec: &InferenceSpec,
        policy: ExecutionPolicy,
        sample: Option<&VlaRequest<'_>>,
    ) -> Result<Self> {
        check_spec(spec)?;
        if let Some(sample) = sample {
            validate(sample, spec)?;
        }
        let ctx = &runner.ctx;
        let inputs = Inputs::allocate(ctx, spec.image_layout.unwrap())?;
        let rng = RefCell::new(sampling::create_normal_generator(
            ctx,
            inputs.noise.clone(),
        )?);
        if let Some(sample) = sample {
            ctx.synchronize().map_err(Error::Cuda)?;
            inputs.bind(sample, &rng)?;
        }
        let output = ctx.allocate_output(Shape::new(vec![32, 7]), DType::F32)?;
        let operation = |autotune| {
            let action = runner.model.forward(ctx, &inputs, autotune)?;
            transfers::copy_cuda_to_cuda(ctx, &action, &output)
        };
        let make_session =
            || ExecutionSession::with_capacity(runner.model.workspace_bytes(), ctx.device_id());
        // Prepare/tune in the same stable arena that capture will reuse. An
        // unscoped warm forward would create and destroy thousands of native
        // GEMM resources, then repeat that work during actual preparation.
        let make_graph = || {
            let mut preparing = true;
            PreparedPhase::prepare_and_capture(ctx, make_session()?, || {
                let autotune = std::mem::replace(&mut preparing, false);
                operation(autotune)
            })
        };
        let mut fallback = None;
        let execution = match policy {
            ExecutionPolicy::Eager => {
                let session = make_session()?;
                ops::prepare_with_session(&session, || operation(true))?;
                Execution::Eager(session)
            }
            ExecutionPolicy::RequireGraph => Execution::Graph(make_graph()?),
            ExecutionPolicy::PreferGraph => match make_graph() {
                Ok(graph) => Execution::Graph(graph),
                Err(e) => {
                    fallback = Some(e.to_string());
                    let session = make_session()?;
                    ops::prepare_with_session(&session, || operation(true))?;
                    Execution::Eager(session)
                }
            },
        };
        Ok(Self {
            execution,
            inputs,
            output,
            rng,
            spec: *spec,
            ctx: Rc::clone(ctx),
            model: Rc::clone(&runner.model),
            fallback,
        })
    }
}
impl PreparedInference for Plan {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }
    fn status(&self) -> PreparationStatus {
        PreparationStatus::Ready {
            mode: match self.execution {
                Execution::Eager(_) => ExecutionMode::Eager,
                Execution::Graph(_) => ExecutionMode::Graph,
            },
            fallback_reason: self.fallback.clone(),
        }
    }
    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        validate(request, &self.spec)?;
        // Host uploads must not overwrite buffers still read by the previous
        // asynchronous execution on this nonblocking context stream.
        self.ctx.synchronize().map_err(Error::Cuda)?;
        self.inputs.bind(request, &self.rng)?;
        match &self.execution {
            Execution::Graph(graph) => graph.replay()?,
            Execution::Eager(session) => ops::with_session(session, || {
                let x = self.model.forward(&self.ctx, &self.inputs, false)?;
                transfers::copy_cuda_to_cuda(&self.ctx, &x, &self.output)
            })?,
        }
        Ok(Action::new(self.output.clone()))
    }
}

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use apxinf_core::{Backend, DType, Device, Error, Result, SamplingBackend, Shape, Tensor};
use apxinf_cuda::kernels;
use apxinf_cuda::tensorrt::{Engine, TensorMode};
use apxinf_cuda::{CudaBuffer, CudaContext};
use half::bf16;

use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::vla::{
    Action, InferenceSpec, InitialLatent, PreparedInference, VisionObservation, VlaRequest,
    VlaRuntime,
};

use super::{EngineBundle, Gr00tN17Config};

const IMAGE_TOKEN_ID: u32 = 151_655;
const VLLN_EPS: f32 = 1e-5;

struct Stage {
    engine: Engine,
    outputs: Vec<(String, apxinf_cuda::tensorrt::TensorDType)>,
    buffers: RefCell<HashMap<String, CudaBuffer>>,
}

struct StageInput<'a> {
    name: &'a str,
    shape: &'a [i64],
    buffer: &'a CudaBuffer,
}

impl Stage {
    fn load(path: &Path) -> Result<Self> {
        let engine = Engine::load(path).map_err(Error::Cuda)?;
        let outputs = engine
            .tensors()
            .map_err(Error::Cuda)?
            .into_iter()
            .filter(|info| info.mode == TensorMode::Output)
            .map(|info| (info.name, info.dtype))
            .collect();
        Ok(Self {
            engine,
            outputs,
            buffers: RefCell::new(HashMap::new()),
        })
    }

    fn run(
        &self,
        context: &CudaContext,
        inputs: &[StageInput<'_>],
    ) -> Result<HashMap<String, CudaBuffer>> {
        if inputs.is_empty() {
            return Err(Error::Other(
                "TensorRT stage requires at least one input".into(),
            ));
        }
        for input in inputs {
            self.engine
                .set_input_shape(input.name, input.shape)
                .map_err(Error::Cuda)?;
            self.engine
                .set_address(input.name, input.buffer)
                .map_err(Error::Cuda)?;
        }
        let device = inputs[0].buffer.device();
        if inputs.iter().any(|input| input.buffer.device() != device) {
            return Err(Error::Other(
                "TensorRT stage inputs span multiple CUDA devices".into(),
            ));
        }
        let mut outputs = HashMap::new();
        let mut cached = self.buffers.borrow_mut();
        for (name, dtype) in &self.outputs {
            let shape = self.engine.shape(name).map_err(Error::Cuda)?;
            let elements = shape.iter().try_fold(1usize, |count, &dim| {
                usize::try_from(dim)
                    .ok()
                    .and_then(|dim| count.checked_mul(dim))
                    .ok_or_else(|| {
                        Error::Other(format!(
                            "TensorRT output {} has unresolved/overflowing shape {shape:?}",
                            name
                        ))
                    })
            })?;
            let bytes = elements
                .checked_mul(dtype.size_in_bytes())
                .ok_or_else(|| Error::Other("TensorRT output byte size overflow".into()))?;
            let buffer = match cached.get(name) {
                Some(buffer) if buffer.len() == bytes && buffer.device() == device => {
                    buffer.clone()
                }
                _ => {
                    let buffer = CudaBuffer::alloc(bytes, device).map_err(Error::Cuda)?;
                    cached.insert(name.clone(), buffer.clone());
                    buffer
                }
            };
            self.engine
                .set_address(name, &buffer)
                .map_err(Error::Cuda)?;
            outputs.insert(name.clone(), buffer);
        }
        self.engine.enqueue(context).map_err(Error::Cuda)?;
        Ok(outputs)
    }
}

fn take_output(outputs: &mut HashMap<String, CudaBuffer>, name: &str) -> Result<CudaBuffer> {
    outputs
        .remove(name)
        .ok_or_else(|| Error::Other(format!("TensorRT stage did not produce `{name}`")))
}

struct Stages {
    vit: Stage,
    llm: Stage,
    vl_self_attention: Stage,
    state_encoder: Stage,
    action_encoder: Stage,
    dit: Stage,
    action_decoder: Stage,
}

struct GlueWeights {
    token_embedding: Tensor,
    vlln_weight: Tensor,
    vlln_bias: Tensor,
    action_position: Tensor,
}

pub struct Gr00tN17Runtime {
    config: Gr00tN17Config,
    bundle: EngineBundle,
    backend: Arc<apxinf_cuda::CudaBackend>,
    stages: Stages,
    weights: GlueWeights,
}

impl Gr00tN17Runtime {
    fn load(
        checkpoint: &Path,
        backend: Arc<apxinf_cuda::CudaBackend>,
        options: &LoadOptions,
    ) -> Result<Self> {
        if !matches!(
            options.precision,
            ModelPrecision::Auto | ModelPrecision::Bf16
        ) {
            return Err(Error::Other(
                "GR00T N1.7 currently supports the released BF16 TensorRT contract only".into(),
            ));
        }
        let config = Gr00tN17Config::from_json_file(&checkpoint.join("config.json"))?;
        let bundle = EngineBundle::load(checkpoint, &config)?;
        let stages = Stages {
            vit: Stage::load(&bundle.path("vit.engine"))?,
            llm: Stage::load(&bundle.path("llm_bf16.engine"))?,
            vl_self_attention: Stage::load(&bundle.path("vl_self_attention.engine"))?,
            state_encoder: Stage::load(&bundle.path("state_encoder.engine"))?,
            action_encoder: Stage::load(&bundle.path("action_encoder.engine"))?,
            dit: Stage::load(&bundle.path("dit_bf16.engine"))?,
            action_decoder: Stage::load(&bundle.path("action_decoder.engine"))?,
        };
        let names = [
            "backbone.model.model.language_model.embed_tokens.weight",
            "action_head.vlln.weight",
            "action_head.vlln.bias",
            "action_head.position_embedding.weight",
        ];
        let mut host = apxinf_loader::safetensors::load_native_selected_path(checkpoint, &names)
            .map_err(|error| Error::Other(format!("load GR00T glue weights: {error}")))?;
        let upload = |name: &str, values: &mut HashMap<String, Tensor>| -> Result<Tensor> {
            let value = values
                .remove(name)
                .ok_or_else(|| Error::Other(format!("missing GR00T weight `{name}`")))?;
            backend.to_device(&value)
        };
        let token_embedding = upload(names[0], &mut host)?;
        let vlln_weight = upload(names[1], &mut host)?;
        let vlln_bias = upload(names[2], &mut host)?;
        let action_position_full = upload(names[3], &mut host)?;
        let position_bytes = config.action_horizon * config.input_embedding_dim * 2;
        let action_position = CudaBuffer::from_tensor(&action_position_full)
            .map_err(Error::Cuda)?
            .view(0, position_bytes)
            .map_err(Error::Cuda)?
            .as_tensor(
                Shape::new(vec![config.action_horizon, config.input_embedding_dim]),
                DType::BF16,
            )
            .map_err(Error::Cuda)?;
        Ok(Self {
            config,
            bundle,
            backend,
            stages,
            weights: GlueWeights {
                token_embedding,
                vlln_weight,
                vlln_bias,
                action_position,
            },
        })
    }

    fn context(&self) -> &CudaContext {
        self.backend.context()
    }

    fn device(&self) -> usize {
        self.backend.device_id()
    }

    fn upload_bytes(&self, bytes: &[u8]) -> Result<CudaBuffer> {
        let buffer = CudaBuffer::alloc(bytes.len(), self.device()).map_err(Error::Cuda)?;
        buffer.copy_from_host(bytes).map_err(Error::Cuda)?;
        Ok(buffer)
    }

    fn upload_u32(&self, values: &[u32]) -> Result<CudaBuffer> {
        self.upload_bytes(bytemuck::cast_slice(values))
    }

    fn upload_i64(&self, values: &[i64]) -> Result<CudaBuffer> {
        self.upload_bytes(bytemuck::cast_slice(values))
    }

    fn ensure_device_f32(&self, value: &Tensor) -> Result<Tensor> {
        if value.dtype() != DType::F32 {
            return Err(Error::Other(format!(
                "GR00T ViT expects f32 pixel values, got {}",
                value.dtype()
            )));
        }
        if value.device() == Device::Cuda(self.device()) {
            Ok(value.clone())
        } else if value.device() == Device::Cpu {
            self.backend.to_device(value)
        } else {
            Err(Error::DeviceMismatch {
                expected: Device::Cuda(self.device()),
                got: value.device(),
            })
        }
    }

    fn ensure_device_bf16(&self, value: &Tensor, shape: &[usize], label: &str) -> Result<Tensor> {
        if value.shape().dims() != shape {
            return Err(Error::Other(format!(
                "GR00T {label} expected {shape:?}, got {:?}",
                value.shape().dims()
            )));
        }
        match (value.device(), value.dtype()) {
            (Device::Cuda(device), DType::BF16) if device == self.device() => Ok(value.clone()),
            (Device::Cuda(device), DType::F32) if device == self.device() => {
                kernels::elementwise::cast_f32_to_bf16(self.context(), value)
            }
            (Device::Cpu, DType::BF16) => self.backend.to_device(value),
            (Device::Cpu, DType::F32) => {
                let converted = value
                    .as_f32()?
                    .iter()
                    .map(|&item| bf16::from_f32(item))
                    .collect::<Vec<_>>();
                self.backend
                    .to_device(&Tensor::from_bf16(shape.to_vec(), &converted)?)
            }
            (device, dtype) => Err(Error::Other(format!(
                "GR00T {label} does not support {dtype} on {device}"
            ))),
        }
    }

    fn initial_actions(&self, latent: InitialLatent<'_>) -> Result<Tensor> {
        let shape = [self.config.action_horizon, self.config.max_action_dim];
        match latent {
            InitialLatent::Provided(value) => self.ensure_device_bf16(value, &shape, "noise"),
            InitialLatent::Generate { rng } => {
                let storage = CudaBuffer::alloc(shape.iter().product::<usize>() * 2, self.device())
                    .map_err(Error::Cuda)?;
                let output = storage
                    .as_tensor(Shape::new(shape.to_vec()), DType::BF16)
                    .map_err(Error::Cuda)?;
                let mut generator = self.backend.create_normal_generator(output)?;
                Ok(generator.generate(rng)?.clone())
            }
        }
    }

    fn mrope_positions(&self, token_ids: &[u32]) -> Result<Vec<i64>> {
        let grids = &self.bundle.metadata.vit_grid_thw;
        let mut token_major = Vec::<[i64; 3]>::with_capacity(token_ids.len());
        let mut start = 0usize;
        let mut image_index = 0usize;
        let mut next = 0i64;
        loop {
            let image_start =
                (start..token_ids.len()).find(|&index| token_ids[index] == IMAGE_TOKEN_ID);
            let Some(image_start) = image_start else {
                for _ in start..token_ids.len() {
                    token_major.push([next, next, next]);
                    next += 1;
                }
                break;
            };
            for _ in start..image_start {
                token_major.push([next, next, next]);
                next += 1;
            }
            let [t, h, w] = *grids.get(image_index).ok_or_else(|| {
                Error::Other("GR00T token stream contains more images than the engine grid".into())
            })?;
            image_index += 1;
            let (t, h, w) = (t as i64, (h / 2) as i64, (w / 2) as i64);
            for ti in 0..t {
                for hi in 0..h {
                    for wi in 0..w {
                        token_major.push([next + ti, next + hi, next + wi]);
                    }
                }
            }
            next += t.max(h).max(w);
            start = image_start + (t * h * w) as usize;
        }
        if image_index != grids.len() || token_major.len() != token_ids.len() {
            return Err(Error::Other(format!(
                "GR00T image-token/grid mismatch: images={}, grids={}, positions={}",
                image_index,
                grids.len(),
                token_major.len()
            )));
        }
        let mut axis_major = Vec::with_capacity(token_ids.len() * 3);
        for axis in 0..3 {
            axis_major.extend(token_major.iter().map(|position| position[axis]));
        }
        Ok(axis_major)
    }

    fn infer_device(&self, request: &VlaRequest<'_>) -> Result<Tensor> {
        request.observation.validate()?;
        let token_ids = &request.observation.token_ids;
        if token_ids.len() > 312 {
            return Err(Error::Other(format!(
                "GR00T token length {} exceeds TensorRT profile maximum 312",
                token_ids.len()
            )));
        }
        let image_positions = token_ids
            .iter()
            .enumerate()
            .filter_map(|(index, &token)| (token == IMAGE_TOKEN_ID).then_some(index as u32))
            .collect::<Vec<_>>();
        if image_positions.len() != self.bundle.metadata.num_merged_patches {
            return Err(Error::Other(format!(
                "GR00T expected {} image tokens, got {}",
                self.bundle.metadata.num_merged_patches,
                image_positions.len()
            )));
        }
        let pixels = match &request.observation.vision {
            VisionObservation::Patches(value) => self.ensure_device_f32(value)?,
            VisionObservation::RgbU8 { .. } => {
                return Err(Error::Other(
                    "GR00T native runtime expects canonical Qwen3-VL pixel_values".into(),
                ))
            }
        };
        if pixels.shape().dims()
            != [
                self.bundle.metadata.num_patches,
                self.config.input_embedding_dim,
            ]
        {
            return Err(Error::Other(format!(
                "GR00T pixel_values expected [{}, {}], got {:?}",
                self.bundle.metadata.num_patches,
                self.config.input_embedding_dim,
                pixels.shape().dims()
            )));
        }
        let pixel_buffer = CudaBuffer::from_tensor(&pixels).map_err(Error::Cuda)?;
        let mut vit = self.stages.vit.run(
            self.context(),
            &[StageInput {
                name: "pixel_values",
                shape: &[
                    self.bundle.metadata.num_patches as i64,
                    self.config.input_embedding_dim as i64,
                ],
                buffer: &pixel_buffer,
            }],
        )?;
        let primary_f32 = take_output(&mut vit, "image_embeds")?
            .as_tensor(
                Shape::new(vec![
                    self.bundle.metadata.num_merged_patches,
                    self.config.backbone_embedding_dim,
                ]),
                DType::F32,
            )
            .map_err(Error::Cuda)?;
        let deepstack_f32 = take_output(&mut vit, "deepstack_features")?;
        let primary = kernels::elementwise::cast_f32_to_bf16(self.context(), &primary_f32)?;
        let deepstack_bytes = self.bundle.metadata.num_merged_patches
            * self.config.backbone_embedding_dim
            * DType::F32.size_in_bytes();
        let mut deepstack = Vec::with_capacity(self.bundle.metadata.num_deepstack);
        for layer in 0..self.bundle.metadata.num_deepstack {
            let value = deepstack_f32
                .view(layer * deepstack_bytes, deepstack_bytes)
                .map_err(Error::Cuda)?
                .as_tensor(
                    Shape::new(vec![
                        self.bundle.metadata.num_merged_patches,
                        self.config.backbone_embedding_dim,
                    ]),
                    DType::F32,
                )
                .map_err(Error::Cuda)?;
            deepstack.push(kernels::elementwise::cast_f32_to_bf16(
                self.context(),
                &value,
            )?);
        }

        let token_buffer = self.upload_u32(token_ids)?;
        let embeddings = kernels::embedding::lookup(
            self.context(),
            &self.weights.token_embedding,
            &token_buffer,
            token_ids.len(),
        )?;
        let positions = self.upload_u32(&image_positions)?;
        kernels::elementwise::scatter_rows_bf16(self.context(), &primary, &positions, &embeddings)?;

        let sequence = token_ids.len();
        let attention = self.upload_i64(&vec![1i64; sequence])?;
        let position_ids = self.upload_i64(&self.mrope_positions(token_ids)?)?;
        let image_mask_host = token_ids
            .iter()
            .map(|&token| u8::from(token == IMAGE_TOKEN_ID))
            .collect::<Vec<_>>();
        let image_mask = self.upload_bytes(&image_mask_host)?;
        let full_mask = self.upload_bytes(&vec![1u8; sequence])?;
        let embeddings_buffer = CudaBuffer::from_tensor(&embeddings).map_err(Error::Cuda)?;
        let deepstack_buffers = deepstack
            .iter()
            .map(CudaBuffer::from_tensor)
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(Error::Cuda)?;
        let mut llm = self.stages.llm.run(
            self.context(),
            &[
                StageInput {
                    name: "inputs_embeds",
                    shape: &[1, sequence as i64, 2048],
                    buffer: &embeddings_buffer,
                },
                StageInput {
                    name: "attention_mask",
                    shape: &[1, sequence as i64],
                    buffer: &attention,
                },
                StageInput {
                    name: "position_ids",
                    shape: &[3, 1, sequence as i64],
                    buffer: &position_ids,
                },
                StageInput {
                    name: "visual_pos_masks",
                    shape: &[1, sequence as i64],
                    buffer: &image_mask,
                },
                StageInput {
                    name: "deepstack_0",
                    shape: &[128, 2048],
                    buffer: &deepstack_buffers[0],
                },
                StageInput {
                    name: "deepstack_1",
                    shape: &[128, 2048],
                    buffer: &deepstack_buffers[1],
                },
                StageInput {
                    name: "deepstack_2",
                    shape: &[128, 2048],
                    buffer: &deepstack_buffers[2],
                },
            ],
        )?;
        let llm_output = take_output(&mut llm, "embeddings")?
            .as_tensor(Shape::new(vec![sequence, 2048]), DType::BF16)
            .map_err(Error::Cuda)?;
        let normalized = kernels::norm::layer_bf16(
            self.context(),
            &llm_output,
            &self.weights.vlln_weight,
            &self.weights.vlln_bias,
            VLLN_EPS,
        )?;
        let normalized_buffer = CudaBuffer::from_tensor(&normalized).map_err(Error::Cuda)?;
        let mut vl_sa = self.stages.vl_self_attention.run(
            self.context(),
            &[StageInput {
                name: "hidden_states",
                shape: &[1, sequence as i64, 2048],
                buffer: &normalized_buffer,
            }],
        )?;
        let vl_embeds = take_output(&mut vl_sa, "output")?
            .as_tensor(Shape::new(vec![sequence, 2048]), DType::BF16)
            .map_err(Error::Cuda)?;

        let state =
            request.observation.state.as_ref().ok_or_else(|| {
                Error::Other("GR00T observation is missing normalized state".into())
            })?;
        let state = self.ensure_device_bf16(state, &[1, 1, self.config.max_state_dim], "state")?;
        let embodiment = request
            .observation
            .embodiment_id
            .ok_or_else(|| Error::Other("GR00T observation is missing embodiment_id".into()))?;
        if embodiment as usize >= self.config.max_num_embodiments {
            return Err(Error::Other(format!(
                "GR00T embodiment {embodiment} is outside 0..{}",
                self.config.max_num_embodiments
            )));
        }
        let embodiment_buffer = self.upload_i64(&[embodiment as i64])?;
        let state_buffer = CudaBuffer::from_tensor(&state).map_err(Error::Cuda)?;
        let mut state_output = self.stages.state_encoder.run(
            self.context(),
            &[
                StageInput {
                    name: "state",
                    shape: &[1, 1, 132],
                    buffer: &state_buffer,
                },
                StageInput {
                    name: "embodiment_id",
                    shape: &[1],
                    buffer: &embodiment_buffer,
                },
            ],
        )?;
        let state_features = take_output(&mut state_output, "output")?
            .as_tensor(Shape::new(vec![1, 1536]), DType::BF16)
            .map_err(Error::Cuda)?;

        let mut actions = self.initial_actions(request.initial_latent)?;
        let vl_buffer = CudaBuffer::from_tensor(&vl_embeds).map_err(Error::Cuda)?;
        for step in 0..self.config.num_inference_timesteps {
            let bucket =
                step * self.config.num_timestep_buckets / self.config.num_inference_timesteps;
            let timestep = self.upload_i64(&[bucket as i64])?;
            let actions_buffer = CudaBuffer::from_tensor(&actions).map_err(Error::Cuda)?;
            let mut encoded = self.stages.action_encoder.run(
                self.context(),
                &[
                    StageInput {
                        name: "actions",
                        shape: &[1, 40, 132],
                        buffer: &actions_buffer,
                    },
                    StageInput {
                        name: "timesteps",
                        shape: &[1],
                        buffer: &timestep,
                    },
                    StageInput {
                        name: "embodiment_id",
                        shape: &[1],
                        buffer: &embodiment_buffer,
                    },
                ],
            )?;
            let action_features = take_output(&mut encoded, "output")?
                .as_tensor(Shape::new(vec![40, 1536]), DType::BF16)
                .map_err(Error::Cuda)?;
            let action_features = self
                .backend
                .add(&action_features, &self.weights.action_position)?;
            let state_action = kernels::elementwise::concat_rows_bf16(
                self.context(),
                &state_features,
                &action_features,
            )?;
            let sa_buffer = CudaBuffer::from_tensor(&state_action).map_err(Error::Cuda)?;
            let mut dit = self.stages.dit.run(
                self.context(),
                &[
                    StageInput {
                        name: "sa_embs",
                        shape: &[1, 41, 1536],
                        buffer: &sa_buffer,
                    },
                    StageInput {
                        name: "vl_embs",
                        shape: &[1, sequence as i64, 2048],
                        buffer: &vl_buffer,
                    },
                    StageInput {
                        name: "timestep",
                        shape: &[1],
                        buffer: &timestep,
                    },
                    StageInput {
                        name: "image_mask",
                        shape: &[1, sequence as i64],
                        buffer: &image_mask,
                    },
                    StageInput {
                        name: "backbone_attention_mask",
                        shape: &[1, sequence as i64],
                        buffer: &full_mask,
                    },
                ],
            )?;
            let model_output = take_output(&mut dit, "output")?;
            let mut decoded = self.stages.action_decoder.run(
                self.context(),
                &[
                    StageInput {
                        name: "model_output",
                        shape: &[1, 41, 1024],
                        buffer: &model_output,
                    },
                    StageInput {
                        name: "embodiment_id",
                        shape: &[1],
                        buffer: &embodiment_buffer,
                    },
                ],
            )?;
            let decoded = take_output(&mut decoded, "output")?;
            let velocity = decoded
                .view(
                    self.config.max_action_dim * 2,
                    self.config.action_horizon * self.config.max_action_dim * 2,
                )
                .map_err(Error::Cuda)?
                .as_tensor(
                    Shape::new(vec![self.config.action_horizon, self.config.max_action_dim]),
                    DType::BF16,
                )
                .map_err(Error::Cuda)?;
            actions = kernels::elementwise::euler_update_bf16(
                self.context(),
                &actions,
                &velocity,
                1.0 / self.config.num_inference_timesteps as f32,
            )?;
        }
        Ok(actions)
    }
}

struct Gr00tPrepared<'a> {
    runtime: &'a Gr00tN17Runtime,
    spec: InferenceSpec,
}

impl PreparedInference for Gr00tPrepared<'_> {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        if !self.spec.matches(request.observation) {
            return Err(Error::Other(
                "GR00T prepared inference spec mismatch".into(),
            ));
        }
        Ok(Action::new(self.runtime.infer_device(request)?))
    }
}

impl VlaRuntime for Gr00tN17Runtime {
    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        Ok(Action::new(self.infer_device(request)?))
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference + '_>> {
        spec.validate()?;
        if spec.image_layout.is_some() {
            return Err(Error::Other(
                "GR00T prepared inference expects canonical pixel_values patches".into(),
            ));
        }
        Ok(Box::new(Gr00tPrepared {
            runtime: self,
            spec: *spec,
        }))
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        self.backend
            .to_cpu(&self.infer_device(request)?)?
            .to_f32_vec()
    }
}

pub(crate) fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    let backend = crate::accelerator::cuda::downcast_arc(backend)
        .ok_or_else(|| Error::Other("GR00T N1.7 requires the CUDA backend".into()))?;
    Ok(LoadedModel::Vla(Box::new(Gr00tN17Runtime::load(
        path, backend, options,
    )?)))
}

//! CUDA graph owners for the fixed-shape prepared path.

use std::cell::RefCell;
use std::sync::Arc;

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::{ops as cuda_ops, transfers, CudaContext as Context, PreparedPhase};

use crate::cuda_gpt_sovits::config::{BERT_HIDDEN, CFM_DIM};
use crate::cuda_gpt_sovits::model::{
    repeat_row_ids, Bert, Cfm, Hubert, Sovits, StepModulation, Vocoder,
};
use crate::cuda_gpt_sovits::ops;
use crate::cuda_gpt_sovits::weights::{fp16_tensor, ids_tensor};

const MIB: usize = 1024 * 1024;

fn cfm_workspace_bytes(prompt_len: usize, target_len: usize) -> Result<usize> {
    let total = prompt_len
        .checked_add(target_len)
        .ok_or_else(|| Error::Other("prepared CFM length overflow".into()))?;
    // One reusable solver traversal retains one step's intermediates and native
    // executions, independently of sample_steps. Initialization has its own arena.
    (2usize * 1024 * MIB)
        .checked_mul(total)
        .and_then(|bytes| bytes.checked_add(999))
        .map(|bytes| (bytes / 1000).max(512 * MIB))
        .ok_or_else(|| Error::Other("prepared CFM workspace size overflow".into()))
}

fn vocoder_workspace_bytes(frames: usize) -> Result<usize> {
    // A 500-frame vocoder profile measures below 2 GiB on Orin. Its retained
    // activations grow linearly with mel frames.
    (2usize * 1024 * MIB)
        .checked_mul(frames)
        .and_then(|bytes| bytes.checked_add(499))
        .map(|bytes| (bytes / 500).max(512 * MIB))
        .ok_or_else(|| Error::Other("prepared vocoder workspace size overflow".into()))
}

pub(super) struct CfmChunkGraph {
    conditioning: PreparedPhase,
    solver: PreparedPhase,
    prompt_features: Tensor,
    target_features: Tensor,
    prompt_mel: Tensor,
    output: Tensor,
    // Both allocations are created outside the solver arena. Captured nodes
    // retain their addresses, so the plan must retain the owning tensors.
    _state: Tensor,
    _static_condition: Tensor,
    modulation: StepModulation,
    steps: usize,
    cfm: Arc<Cfm>,
    noise_generator: RefCell<Box<dyn apxinf_core::NormalGenerator>>,
}

impl CfmChunkGraph {
    pub(super) fn prepare(
        ctx: &Context,
        cfm: &Arc<Cfm>,
        prompt_len: usize,
        target_len: usize,
        steps: usize,
    ) -> Result<Self> {
        let total = prompt_len
            .checked_add(target_len)
            .ok_or_else(|| Error::Other("prepared CFM length overflow".into()))?;
        let prompt_features = ops::alloc(ctx, &[prompt_len, 512])?;
        let target_features = ops::alloc(ctx, &[target_len, 512])?;
        let prompt_mel = ops::alloc(ctx, &[prompt_len, 100])?;
        let noise = ops::alloc(ctx, &[total, 100])?;
        let x = ops::alloc(ctx, &[total, 100])?;
        let static_condition = ops::alloc(ctx, &[total, CFM_DIM])?;
        let output = ops::alloc(ctx, &[target_len, 100])?;
        let noise_generator = RefCell::new(apxinf_cuda_new::sampling::create_normal_generator(
            ctx,
            noise.clone(),
        )?);
        let conditioning = PreparedPhase::prepare_and_capture(
            ctx,
            cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(
                128 * MIB,
                ctx.device_id(),
            )?,
            || {
                let (condition, initial) = cfm.prepared_chunk_inputs(
                    ctx,
                    &prompt_features,
                    &target_features,
                    &prompt_mel,
                    &noise,
                )?;
                transfers::copy_cuda_to_cuda(ctx, &condition, &static_condition)?;
                transfers::copy_cuda_to_cuda(ctx, &initial, &x)
            },
        )?;
        let modulation = cfm.allocate_step_modulation(ctx)?;
        cfm.bind_step_modulation(ctx, 0, &modulation)?;
        let solver = PreparedPhase::prepare_and_capture(
            ctx,
            cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(
                cfm_workspace_bytes(prompt_len, target_len)?,
                ctx.device_id(),
            )?,
            || {
                let next = cfm.prepared_solver_step(
                    ctx,
                    &x,
                    &static_condition,
                    &modulation,
                    steps,
                    prompt_len,
                    target_len,
                )?;
                transfers::copy_cuda_to_cuda(ctx, &next, &x)?;
                let target = ops::slice_rows(ctx, &next, prompt_len, target_len)?;
                transfers::copy_cuda_to_cuda(ctx, &target, &output)
            },
        )?;
        Ok(Self {
            conditioning,
            solver,
            prompt_features,
            target_features,
            prompt_mel,
            output,
            _state: x,
            _static_condition: static_condition,
            modulation,
            steps,
            cfm: Arc::clone(cfm),
            noise_generator,
        })
    }

    pub(super) fn replay(
        &self,
        ctx: &Context,
        prompt_features: &Tensor,
        target_features: &Tensor,
        prompt_mel: &Tensor,
        seed: u64,
    ) -> Result<Tensor> {
        transfers::copy_cuda_to_cuda(ctx, prompt_features, &self.prompt_features)?;
        transfers::copy_cuda_to_cuda(ctx, target_features, &self.target_features)?;
        transfers::copy_cuda_to_cuda(ctx, prompt_mel, &self.prompt_mel)?;
        self.noise_generator
            .borrow_mut()
            .generate(apxinf_core::RngKey::new(seed, 0, 1))?;
        self.replay_bound(ctx)?;
        let copy = ops::alloc(ctx, self.output.shape().dims())?;
        transfers::copy_cuda_to_cuda(ctx, &self.output, &copy)?;
        Ok(copy)
    }
    fn replay_bound(&self, ctx: &Context) -> Result<()> {
        self.conditioning.replay()?;
        for index in 0..self.steps {
            self.cfm
                .bind_step_modulation(ctx, index * 32 / self.steps, &self.modulation)?;
            self.solver.replay()?;
        }
        Ok(())
    }
}

pub(super) struct VocoderGraph {
    phase: PreparedPhase,
    input: Tensor,
    output: Tensor,
}

impl VocoderGraph {
    pub(super) fn prepare(ctx: &Context, vocoder: &Vocoder, frames: usize) -> Result<Self> {
        let input = ops::alloc(ctx, &[frames, 100])?;
        let output = ops::alloc(ctx, &[1, frames * 480, 1])?;
        let session = cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(
            vocoder_workspace_bytes(frames)?,
            ctx.device_id(),
        )?;
        let phase = PreparedPhase::prepare_and_capture(ctx, session, || {
            let generated = vocoder.forward(ctx, &input)?;
            transfers::copy_cuda_to_cuda(ctx, &generated, &output)
        })?;
        Ok(Self {
            phase,
            input,
            output,
        })
    }

    pub(super) fn replay(&self, ctx: &Context, mel: &Tensor) -> Result<Vec<f32>> {
        transfers::copy_cuda_to_cuda(ctx, mel, &self.input)?;
        self.phase.replay()?;
        Vocoder::pcm_from_output(ctx, &self.output)
    }
}

// Frontend and acoustic encoder graphs own stable mutable inputs. Public
// requests rebind values; no reference content is baked into their profiles.
pub(super) struct BertGraph {
    phase: PreparedPhase,
    _positions: Tensor,
    _types: Tensor,
    ids: Tensor,
    output: Tensor,
}
impl BertGraph {
    pub(super) fn prepare(ctx: &Context, bert: &Bert, tokens: usize) -> Result<Self> {
        let ids = ids_tensor(ctx, &vec![0; tokens])?;
        let positions = ids_tensor(ctx, &(0..tokens as u32).collect::<Vec<_>>())?;
        let types = ids_tensor(ctx, &vec![0; tokens])?;
        let output = ops::alloc(ctx, &[tokens - 2, BERT_HIDDEN])?;
        let session = cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(
            (tokens * MIB).max(64 * MIB),
            ctx.device_id(),
        )?;
        let phase = PreparedPhase::prepare_and_capture(ctx, session, || {
            let hidden = bert.forward_device(ctx, &ids, &positions, &types)?;
            transfers::copy_cuda_to_cuda(ctx, &hidden, &output)
        })?;
        Ok(Self {
            phase,
            _positions: positions,
            _types: types,
            ids,
            output,
        })
    }
    pub(super) fn replay(
        &self,
        ctx: &Context,
        ids: &[u32],
        word2ph: &[u32],
        phones: usize,
    ) -> Result<Tensor> {
        let values = ids.iter().map(|v| *v as i32).collect::<Vec<_>>();
        transfers::copy_cpu_to_cuda(&Tensor::from_i32(vec![values.len()], &values)?, &self.ids)?;
        if word2ph.len() != self.output.shape().dims()[0] {
            return Err(Error::Other("BERT token/word2ph mismatch".into()));
        }
        let mut map = Vec::with_capacity(phones);
        for (index, count) in word2ph.iter().enumerate() {
            map.extend(std::iter::repeat(index as i32).take(*count as usize));
        }
        if map.len() != phones {
            return Err(Error::Other("word2ph phone count mismatch".into()));
        }
        self.phase.replay()?;
        let map = transfers::to_cuda(&Tensor::from_i32(vec![phones], &map)?, ctx.device_id())?;
        ops::gather_rows(ctx, &self.output, &map)
    }
}

pub(super) struct HubertGraph {
    phase: PreparedPhase,
    input: Tensor,
    codes: Tensor,
}
impl HubertGraph {
    pub(super) fn prepare(ctx: &Context, hubert: &Hubert, samples: usize) -> Result<Self> {
        let input = ops::alloc(ctx, &[1, samples, 1])?;
        ops::fill_zero(ctx, &input)?;
        let frames = hubert.semantic_frames(samples)?;
        let codes = ctx.allocate_output(
            apxinf_core::Shape::new(vec![frames]),
            apxinf_core::DType::I32,
        )?;
        // The first 512-channel stage dominates the retained activation arena.
        let workspace = samples
            .checked_mul(1500)
            .and_then(|v| v.checked_add(128 * MIB))
            .ok_or_else(|| Error::Other("HuBERT workspace overflow".into()))?;
        let phase = PreparedPhase::prepare_and_capture(
            ctx,
            cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(
                workspace,
                ctx.device_id(),
            )?,
            || {
                let features = hubert.encode_features_device(ctx, &input)?;
                let generated = hubert.quantize(ctx, &features)?;
                transfers::copy_cuda_to_cuda(ctx, &generated, &codes)
            },
        )?;
        Ok(Self {
            phase,
            input,
            codes,
        })
    }
    pub(super) fn replay(&self, ctx: &Context, audio: &[f32]) -> Result<Tensor> {
        let input = fp16_tensor(ctx, self.input.shape().dims(), audio)?;
        transfers::copy_cuda_to_cuda(ctx, &input, &self.input)?;
        self.phase.replay()?;
        Ok(self.codes.clone())
    }
}

pub(super) struct StyleGraph {
    phase: PreparedPhase,
    input: Tensor,
    output: Tensor,
}
impl StyleGraph {
    pub(super) fn prepare(ctx: &Context, sovits: &Sovits, frames: usize) -> Result<Self> {
        let input = ops::alloc(ctx, &[frames, 704])?;
        let output = ops::alloc(ctx, &[512])?;
        ops::fill_zero(ctx, &input)?;
        let phase = PreparedPhase::prepare_and_capture(
            ctx,
            cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(
                64 * MIB,
                ctx.device_id(),
            )?,
            || {
                let style = sovits.style(ctx, &input)?;
                transfers::copy_cuda_to_cuda(ctx, &style, &output)
            },
        )?;
        Ok(Self {
            phase,
            input,
            output,
        })
    }
    pub(super) fn replay(&self, ctx: &Context, spec: &Tensor) -> Result<Tensor> {
        transfers::copy_cuda_to_cuda(ctx, spec, &self.input)?;
        self.phase.replay()?;
        Ok(self.output.clone())
    }
}

pub(super) struct SovitsGraph {
    phase: PreparedPhase,
    codes: Tensor,
    phones: Tensor,
    _upsample_codes: Tensor,
    _upsample_bridge: Tensor,
    style: Tensor,
    output: Tensor,
}
impl SovitsGraph {
    pub(super) fn prepare(
        ctx: &Context,
        sovits: &Sovits,
        codes: usize,
        phones: usize,
    ) -> Result<Self> {
        let codes_tensor = ids_tensor(ctx, &vec![0; codes])?;
        let phone_tensor = ids_tensor(ctx, &vec![0; phones])?;
        let upsample_codes = repeat_row_ids(ctx, codes)?;
        let upsample_bridge = repeat_row_ids(ctx, codes * 2)?;
        let style = ops::alloc(ctx, &[512])?;
        ops::fill_zero(ctx, &style)?;
        let output = ops::alloc(ctx, &[codes * 4, 512])?;
        let workspace = codes
            .checked_mul(4 * MIB)
            .and_then(|v| v.checked_add(128 * MIB))
            .ok_or_else(|| Error::Other("SoVITS workspace overflow".into()))?;
        let phase = PreparedPhase::prepare_and_capture(
            ctx,
            cuda_ops::ExecutionSession::with_capacity_pinning_prepared_buffers(
                workspace,
                ctx.device_id(),
            )?,
            || {
                let generated = sovits.decode_features_device(
                    ctx,
                    &codes_tensor,
                    &phone_tensor,
                    &style,
                    &upsample_codes,
                    &upsample_bridge,
                    1.0,
                )?;
                transfers::copy_cuda_to_cuda(ctx, &generated, &output)
            },
        )?;
        Ok(Self {
            phase,
            codes: codes_tensor,
            phones: phone_tensor,
            _upsample_codes: upsample_codes,
            _upsample_bridge: upsample_bridge,
            style,
            output,
        })
    }
    pub(super) fn replay(
        &self,
        ctx: &Context,
        codes: &Tensor,
        phones: &[u32],
        style: &Tensor,
    ) -> Result<Tensor> {
        transfers::copy_cuda_to_cuda(ctx, codes, &self.codes)?;
        let values = phones.iter().map(|v| *v as i32).collect::<Vec<_>>();
        transfers::copy_cpu_to_cuda(
            &Tensor::from_i32(vec![values.len()], &values)?,
            &self.phones,
        )?;
        transfers::copy_cuda_to_cuda(ctx, style, &self.style)?;
        self.phase.replay()?;
        Ok(self.output.clone())
    }
}

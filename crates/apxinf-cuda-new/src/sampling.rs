//! CUDA implementations of the model-neutral sampling contracts.

use apxinf_core::{
    DType, Device, Error, NextTokenLogits, NormalGenerator, Result, RngKey, Tensor, TokenSample,
    TokenSampler, TokenSamplingInit, TokenSamplingParams, TokenSamplingSpec, TokenSelection,
};

use crate::buffer::CudaBuffer;
use crate::ffi;
use crate::ffi::raw::sampling as sampling_ffi;
use crate::CudaContext;

const OUTPUT_BYTES: usize = 16;

fn dtype_tag(dtype: DType) -> Result<i32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        dtype => Err(Error::Other(format!(
            "CUDA sampling does not support {dtype}"
        ))),
    }
}

struct CudaTokenSampler {
    spec: TokenSamplingSpec,
    device_id: usize,
    counts: CudaBuffer,
    adjusted: CudaBuffer,
    token_ids: CudaBuffer,
    sorted_logits: CudaBuffer,
    sorted_tokens: CudaBuffer,
    weights: CudaBuffer,
    cdf: CudaBuffer,
    partial_values: CudaBuffer,
    partial_tokens: CudaBuffer,
    partial_count: usize,
    sort_workspace: CudaBuffer,
    scan_workspace: CudaBuffer,
    output: CudaBuffer,
    params: Option<TokenSamplingParams>,
    sequence_len: usize,
    rng: RngKey,
    stream: crate::ffi::cudaStream_t,
}

impl CudaTokenSampler {
    fn new(ctx: &CudaContext, spec: TokenSamplingSpec) -> Result<Self> {
        spec.validate()?;
        let device_id = ctx.device_id();
        let vocab_size = u32::try_from(spec.vocab_size)
            .map_err(|_| Error::Other("sampling vocabulary exceeds the CUDA ABI range".into()))?;
        let vocab_bytes = spec
            .vocab_size
            .checked_mul(4)
            .ok_or_else(|| Error::Other("sampling workspace size overflow".into()))?;
        let partial_count = spec.vocab_size.div_ceil(256).min(1024).max(1);
        let partial_bytes = partial_count * 4;
        let mut sort_bytes = 0usize;
        let mut scan_bytes = 0usize;
        let status = unsafe {
            sampling_ffi::apxinf_cuda_new_token_sampling_workspace_sizes(
                vocab_size,
                &mut sort_bytes,
                &mut scan_bytes,
            )
        };
        ffi::check_cuda(status).map_err(Error::Cuda)?;
        let alloc = |bytes: usize| CudaBuffer::alloc(bytes.max(1), device_id).map_err(Error::Cuda);
        Ok(Self {
            spec,
            device_id,
            counts: alloc(vocab_bytes)?,
            adjusted: alloc(vocab_bytes)?,
            token_ids: alloc(vocab_bytes)?,
            sorted_logits: alloc(vocab_bytes)?,
            sorted_tokens: alloc(vocab_bytes)?,
            weights: alloc(vocab_bytes)?,
            cdf: alloc(vocab_bytes)?,
            partial_values: alloc(partial_bytes)?,
            partial_tokens: alloc(partial_bytes)?,
            partial_count,
            sort_workspace: alloc(sort_bytes)?,
            scan_workspace: alloc(scan_bytes)?,
            output: alloc(OUTPUT_BYTES)?,
            params: None,
            sequence_len: 0,
            rng: RngKey::default(),
            stream: ctx.stream().handle(),
        })
    }
}

impl TokenSampler for CudaTokenSampler {
    fn spec(&self) -> TokenSamplingSpec {
        self.spec
    }

    fn begin(&mut self, init: TokenSamplingInit<'_>) -> Result<()> {
        init.params.validate(self.spec.vocab_size)?;
        if init.prompt_token_ids.len() > self.spec.max_sequence_len {
            return Err(Error::Other(format!(
                "prompt length {} exceeds sampler capacity {}",
                init.prompt_token_ids.len(),
                self.spec.max_sequence_len
            )));
        }
        let mut counts = vec![0u32; self.spec.vocab_size];
        for &token_id in init.prompt_token_ids {
            let count = counts.get_mut(token_id as usize).ok_or_else(|| {
                Error::Other(format!(
                    "prompt token {token_id} is outside vocabulary {}",
                    self.spec.vocab_size
                ))
            })?;
            *count = count
                .checked_add(1)
                .ok_or_else(|| Error::Other("token occurrence count overflow".into()))?;
        }
        let count_bytes: Vec<_> = counts
            .iter()
            .flat_map(|count| count.to_ne_bytes())
            .collect();
        self.counts
            .copy_from_host(&count_bytes)
            .map_err(Error::Cuda)?;
        self.params = Some(init.params.clone());
        self.sequence_len = init.prompt_token_ids.len();
        self.rng = init.rng;
        Ok(())
    }

    fn sample(&mut self, logits: NextTokenLogits<'_>) -> Result<TokenSample> {
        let params = self
            .params
            .as_ref()
            .ok_or_else(|| Error::Other("token sampler must be initialized with begin()".into()))?;
        if logits.vocab_size() != self.spec.vocab_size {
            return Err(Error::Other(format!(
                "sampler vocabulary is {}, logits vocabulary is {}",
                self.spec.vocab_size,
                logits.vocab_size()
            )));
        }
        if logits.tensor().device() != Device::Cuda(self.device_id) {
            return Err(Error::DeviceMismatch {
                expected: Device::Cuda(self.device_id),
                got: logits.tensor().device(),
            });
        }
        if self.sequence_len >= self.spec.max_sequence_len {
            return Err(Error::Other(format!(
                "token sampler reached sequence capacity {}",
                self.spec.max_sequence_len
            )));
        }
        let mut next_rng = self.rng;
        next_rng.advance()?;

        let dtype = logits.tensor().dtype();
        let dtype_tag = dtype_tag(dtype)?;
        let row_bytes = self
            .spec
            .vocab_size
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::Other("sampling logits row size overflow".into()))?;
        let row_offset = logits
            .row_index()
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::Other("sampling logits row offset overflow".into()))?;
        let logits_buffer = CudaBuffer::from_tensor(logits.tensor()).map_err(Error::Cuda)?;
        let logits_row = logits_buffer
            .view(row_offset, row_bytes)
            .map_err(Error::Cuda)?;
        let (selection, temperature, top_k, top_p) = match params.selection {
            TokenSelection::Greedy => (0, 1.0, 0, 1.0),
            TokenSelection::Random {
                temperature,
                top_k,
                top_p,
            } => (
                1,
                temperature,
                u32::try_from(top_k.unwrap_or(0)).map_err(|_| {
                    Error::Other("sampling top-k exceeds the CUDA ABI range".into())
                })?,
                top_p,
            ),
        };
        let penalties = params.penalties;
        let status = unsafe {
            sampling_ffi::apxinf_cuda_new_sample_token(
                logits_row.ptr(),
                dtype_tag,
                u32::try_from(self.spec.vocab_size).map_err(|_| {
                    Error::Other("sampling vocabulary exceeds the CUDA ABI range".into())
                })?,
                self.counts.ptr().cast(),
                penalties.repetition,
                penalties.frequency,
                penalties.presence,
                selection,
                temperature,
                top_k,
                top_p,
                self.rng.seed,
                self.rng.sequence,
                self.rng.draw,
                u32::from(params.return_logprob),
                self.adjusted.ptr().cast(),
                self.token_ids.ptr().cast(),
                self.sorted_logits.ptr().cast(),
                self.sorted_tokens.ptr().cast(),
                self.weights.ptr().cast(),
                self.cdf.ptr().cast(),
                self.partial_values.ptr().cast(),
                self.partial_tokens.ptr().cast(),
                u32::try_from(self.partial_count).map_err(|_| {
                    Error::Other("sampling partial count exceeds the CUDA ABI range".into())
                })?,
                self.sort_workspace.ptr(),
                self.sort_workspace.len(),
                self.scan_workspace.ptr(),
                self.scan_workspace.len(),
                self.output.ptr(),
                self.stream,
            )
        };
        ffi::check_cuda(status).map_err(Error::Cuda)?;
        // The context stream is deliberately non-blocking, so a synchronous
        // D2H copy issued on the default stream does not order itself after
        // the sampling kernels. Sampling returns a host token and therefore
        // must explicitly wait for its own stream before reading `output`.
        unsafe {
            ffi::check_cuda(ffi::cudaStreamSynchronize(self.stream)).map_err(Error::Cuda)?;
        }
        let mut output = [0u8; OUTPUT_BYTES];
        self.output.copy_to_host(&mut output).map_err(Error::Cuda)?;
        let token_id = u32::from_ne_bytes(output[0..4].try_into().unwrap());
        let status = u32::from_ne_bytes(output[4..8].try_into().unwrap());
        let logprob = f32::from_ne_bytes(output[8..12].try_into().unwrap());
        match status {
            0 => {}
            1 => return Err(Error::Other("all token logits are invalid".into())),
            2 => return Err(Error::Other("token probability mass is invalid".into())),
            status => {
                return Err(Error::Other(format!(
                    "CUDA token sampler returned status {status}"
                )))
            }
        }
        self.sequence_len += 1;
        self.rng = next_rng;
        Ok(TokenSample {
            token_id,
            logprob: params.return_logprob.then_some(logprob),
        })
    }
}

struct CudaNormalGenerator {
    output: Tensor,
    stream: crate::ffi::cudaStream_t,
}

impl CudaNormalGenerator {
    fn new(ctx: &CudaContext, output: Tensor) -> Result<Self> {
        if output.device() != Device::Cuda(ctx.device_id()) {
            return Err(Error::DeviceMismatch {
                expected: Device::Cuda(ctx.device_id()),
                got: output.device(),
            });
        }
        dtype_tag(output.dtype())?;
        Ok(Self {
            output,
            stream: ctx.stream().handle(),
        })
    }
}

impl NormalGenerator for CudaNormalGenerator {
    fn output(&self) -> &Tensor {
        &self.output
    }

    fn generate(&mut self, rng: RngKey) -> Result<&Tensor> {
        let output = CudaBuffer::from_tensor(&self.output).map_err(Error::Cuda)?;
        let status = unsafe {
            sampling_ffi::apxinf_cuda_new_fill_standard_normal(
                output.ptr(),
                dtype_tag(self.output.dtype())?,
                u64::try_from(self.output.numel()).map_err(|_| {
                    Error::Other("normal output size exceeds the CUDA ABI range".into())
                })?,
                rng.seed,
                rng.sequence,
                rng.draw,
                self.stream,
            )
        };
        ffi::check_cuda(status).map_err(Error::Cuda)?;
        Ok(&self.output)
    }
}

pub fn create_token_sampler(
    ctx: &CudaContext,
    spec: TokenSamplingSpec,
) -> Result<Box<dyn TokenSampler>> {
    Ok(Box::new(CudaTokenSampler::new(ctx, spec)?))
}

pub fn create_normal_generator(
    ctx: &CudaContext,
    output: Tensor,
) -> Result<Box<dyn NormalGenerator>> {
    Ok(Box::new(CudaNormalGenerator::new(ctx, output)?))
}

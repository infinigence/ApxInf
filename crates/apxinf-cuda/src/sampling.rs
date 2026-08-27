//! CUDA implementations of the model-neutral sampling contracts.

use apxinf_core::{
    DType, Device, Error, NextTokenLogits, NormalGenerator, Result, RngKey, SamplingBackend,
    Tensor, TokenSample, TokenSampler, TokenSamplingInit, TokenSamplingParams, TokenSamplingSpec,
    TokenSelection,
};

use crate::buffer::CudaBuffer;
use crate::ffi;
use crate::CudaBackend;

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
    fn new(backend: &CudaBackend, spec: TokenSamplingSpec) -> Result<Self> {
        spec.validate()?;
        let device_id = backend.device_id();
        let vocab_bytes = spec
            .vocab_size
            .checked_mul(4)
            .ok_or_else(|| Error::Other("sampling workspace size overflow".into()))?;
        let partial_count = spec.vocab_size.div_ceil(256).min(1024).max(1);
        let partial_bytes = partial_count * 4;
        let mut sort_bytes = 0usize;
        let mut scan_bytes = 0usize;
        let status = unsafe {
            ffi::apxinf_token_sampling_workspace_sizes(
                spec.vocab_size as u32,
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
            stream: backend.context().stream().handle(),
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
        self.counts
            .copy_from_host(bytemuck::cast_slice(&counts))
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
        let row_bytes = self.spec.vocab_size * dtype.size_in_bytes();
        let logits_buffer = CudaBuffer::from_tensor(logits.tensor()).map_err(Error::Cuda)?;
        let logits_row = logits_buffer
            .view(logits.row_index() * row_bytes, row_bytes)
            .map_err(Error::Cuda)?;
        let (selection, temperature, top_k, top_p) = match params.selection {
            TokenSelection::Greedy => (0, 1.0, 0, 1.0),
            TokenSelection::Random {
                temperature,
                top_k,
                top_p,
            } => (1, temperature, top_k.unwrap_or(0) as u32, top_p),
        };
        let penalties = params.penalties;
        let status = unsafe {
            ffi::apxinf_sample_token(
                logits_row.ptr(),
                dtype_tag,
                self.spec.vocab_size as u32,
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
                self.partial_count as u32,
                self.sort_workspace.ptr(),
                self.sort_workspace.len(),
                self.scan_workspace.ptr(),
                self.scan_workspace.len(),
                self.output.ptr(),
                self.stream,
            )
        };
        ffi::check_cuda(status).map_err(Error::Cuda)?;
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
    fn new(backend: &CudaBackend, output: Tensor) -> Result<Self> {
        if output.device() != Device::Cuda(backend.device_id()) {
            return Err(Error::DeviceMismatch {
                expected: Device::Cuda(backend.device_id()),
                got: output.device(),
            });
        }
        dtype_tag(output.dtype())?;
        Ok(Self {
            output,
            stream: backend.context().stream().handle(),
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
            ffi::apxinf_fill_standard_normal(
                output.ptr(),
                dtype_tag(self.output.dtype())?,
                self.output.numel() as u64,
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

impl SamplingBackend for CudaBackend {
    fn create_token_sampler(&self, spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        Ok(Box::new(CudaTokenSampler::new(self, spec)?))
    }

    fn create_normal_generator(&self, output: Tensor) -> Result<Box<dyn NormalGenerator>> {
        Ok(Box::new(CudaNormalGenerator::new(self, output)?))
    }
}

//! Token sampling and standard-normal generation.
//!
//! Both delegate to the host implementations, reached through `CpuBackend`'s
//! public `SamplingBackend` impl, so sampling semantics — penalties, top-k/p,
//! RNG streams — are the host's exactly, not a second implementation that
//! could drift. The cost is one logits-row download per sampled token and one
//! upload per generated noise tensor. Moving either onto the device is
//! performance work for a later milestone.

use std::sync::Arc;

use apxinf_core::{
    CpuBackend, Device, Error, NextTokenLogits, NormalGenerator, Result, RngKey, SamplingBackend,
    Shape, Tensor, TokenSample, TokenSampler, TokenSamplingInit, TokenSamplingSpec,
};

use crate::runtime::HipContext;

pub(crate) struct HipTokenSampler {
    ctx: Arc<HipContext>,
    host: Box<dyn TokenSampler>,
}

impl HipTokenSampler {
    pub(crate) fn new(ctx: Arc<HipContext>, spec: TokenSamplingSpec) -> Result<Self> {
        Ok(Self { ctx, host: CpuBackend.create_token_sampler(spec)? })
    }
}

impl TokenSampler for HipTokenSampler {
    fn spec(&self) -> TokenSamplingSpec {
        self.host.spec()
    }

    fn begin(&mut self, init: TokenSamplingInit<'_>) -> Result<()> {
        self.host.begin(init)
    }

    fn sample(&mut self, logits: NextTokenLogits<'_>) -> Result<TokenSample> {
        let tensor = logits.tensor();
        if tensor.device() == Device::Cpu {
            return self.host.sample(logits);
        }
        // Download only the selected row, not the whole logits tensor.
        let vocab = logits.vocab_size();
        let element = tensor.dtype().size_in_bytes();
        let base = self.ctx.ptr(tensor)?;
        self.ctx.bind()?;
        let mut row = vec![0u8; vocab * element];
        // SAFETY: `NextTokenLogits` validated the row against the tensor's
        // shape, and `ptr` checked the storage covers that shape.
        let src = unsafe { base.cast::<u8>().add(logits.row_offset() * element) };
        self.ctx.download(&mut row, src.cast())?;
        let host = Tensor::from_raw(Shape::new(vec![1, vocab]), tensor.dtype(), Device::Cpu, row)?;
        self.host.sample(NextTokenLogits::row(&host, 0, vocab)?)
    }
}

pub(crate) struct HipNormalGenerator {
    ctx: Arc<HipContext>,
    output: Tensor,
    host: Box<dyn NormalGenerator>,
}

impl HipNormalGenerator {
    pub(crate) fn new(ctx: Arc<HipContext>, output: Tensor) -> Result<Self> {
        // Validates residency and storage extent; the host generator below
        // validates the dtype.
        ctx.ptr(&output)?;
        let staging = Tensor::zeros(output.shape().dims().to_vec(), output.dtype());
        let host = CpuBackend.create_normal_generator(staging)?;
        Ok(Self { ctx, output, host })
    }
}

impl NormalGenerator for HipNormalGenerator {
    fn output(&self) -> &Tensor {
        &self.output
    }

    /// Fill the bound device tensor in place, so a graph that captured its
    /// address sees the new values.
    fn generate(&mut self, rng: RngKey) -> Result<&Tensor> {
        let values = self.host.generate(rng)?;
        let bytes = values.storage().as_cpu().ok_or_else(|| {
            Error::Other("apxinf-hip: host normal generator returned device storage".into())
        })?;
        let dst = self.ctx.ptr(&self.output)?;
        self.ctx.bind()?;
        self.ctx.upload(dst, &bytes[..self.output.size_in_bytes()])?;
        Ok(&self.output)
    }
}

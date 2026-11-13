//! Device-side token selection operators.

use apxinf_core::{DType, Device, Error, Result, Tensor};

use crate::ffi::abi::{elementwise as elementwise_abi, sampling as sampling_abi, status};
use crate::{CudaBuffer, CudaContext};

mod suppress;

pub use suppress::suppress_logits;

/// Device argmax. `remap`, when present, maps the winning local column to the
/// token id written to `index`; without it, `index` receives the column index.
pub struct ArgmaxArgs<'a> {
    pub logits: &'a Tensor,
    pub index: &'a Tensor,
    pub remap: Option<&'a Tensor>,
}

impl<'a> ArgmaxArgs<'a> {
    pub fn new(logits: &'a Tensor, index: &'a Tensor) -> Self {
        Self {
            logits,
            index,
            remap: None,
        }
    }

    pub fn with_remap(mut self, remap: &'a Tensor) -> Self {
        self.remap = Some(remap);
        self
    }
}

fn tensor_buffer(
    ctx: &CudaContext,
    tensor: &Tensor,
    dtype: DType,
    elements: usize,
    name: &str,
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id())
        || tensor.dtype() != dtype
        || tensor.shape().numel() != elements
    {
        return Err(Error::Other(format!(
            "argmax {name} must contain {elements} {dtype} values on CUDA{}",
            ctx.device_id()
        )));
    }
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    let required = elements
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other(format!("argmax {name} size overflow")))?;
    if buffer.len() < required {
        return Err(Error::Other(format!(
            "argmax {name} storage has {} bytes, need {required}",
            buffer.len()
        )));
    }
    Ok(buffer)
}

/// Index of the largest logit, reduced on device. Input dtype is inferred from
/// `logits`; the current kernel supports BF16 and rejects other dtypes.
pub fn argmax(ctx: &CudaContext, args: ArgmaxArgs<'_>) -> Result<()> {
    let count = args.logits.shape().numel();
    if count == 0 || count > i32::MAX as usize {
        return Err(Error::Other(
            "argmax logit count must be in 1..=i32::MAX".into(),
        ));
    }
    let logits = tensor_buffer(ctx, args.logits, DType::BF16, count, "logits")?;
    let index = tensor_buffer(ctx, args.index, DType::I32, 1, "index")?;
    if let Some(remap) = args.remap {
        let remap = tensor_buffer(ctx, remap, DType::I32, count, "remap")?;
        unsafe {
            status::check(elementwise_abi::apxinf_elementwise_argmax_remap_bf16(
                logits.ptr(),
                count as u32,
                remap.ptr(),
                index.ptr(),
                ctx.stream().handle(),
            ))
        }
    } else {
        unsafe {
            status::check(sampling_abi::apxinf_sampling_argmax(
                logits.ptr(),
                index.ptr(),
                count as i64,
                ctx.stream().handle(),
            ))
        }
    }
}

/// Largest vocabulary the exponential-race kernel can resolve in one block.
pub const EXPONENTIAL_RACE_MAX_VOCAB: usize = 2048;

/// Largest `top_k` the kernel resolves with masked reduction passes; larger
/// values fall back to the host sampler.
pub const EXPONENTIAL_RACE_MAX_TOP_K: u32 = 128;

/// Arguments for [`exponential_race_sample`]. All tensors live on `ctx`'s
/// device; `output` receives `[sampled, adjusted_argmax, status, 0]`.
pub struct ExponentialRaceArgs<'a> {
    pub logits: &'a Tensor,
    /// One byte flag per vocabulary entry (`I8` storage, 0 or 1); flagged
    /// scores take the repetition penalty. The kernel flags the sampled token
    /// for the next step.
    pub seen: &'a Tensor,
    /// `I32` output words, at least four: `[sampled, adjusted_argmax, status,
    /// 0]`, reinterpreted as unsigned on the device.
    pub output: &'a Tensor,
    pub seed: u64,
    /// Philox counter offset; the reference advances one draw per step.
    pub step: u64,
    /// `0` disables top-k truncation.
    pub top_k: u32,
    pub temperature: f32,
    pub repetition_penalty: f32,
}

/// Device-side exponential-race sampling. Applies the repetition penalty,
/// derives the adjusted argmax, scales by temperature, truncates to the top-k
/// pivot, softmaxes with a sequential FP32 sum, and races the probabilities
/// against per-candidate Philox exponentials — reproducing the reference host
/// sampler bit for bit, including FP16 rounding boundaries. Nucleus (top_p)
/// truncation is not supported here; callers keep that path on the host.
pub fn exponential_race_sample(
    ctx: &CudaContext,
    args: ExponentialRaceArgs<'_>,
) -> Result<()> {
    let vocab = args.logits.shape().numel();
    if vocab == 0 || vocab > EXPONENTIAL_RACE_MAX_VOCAB {
        return Err(Error::Other(format!(
            "exponential-race vocabulary must be in 1..={EXPONENTIAL_RACE_MAX_VOCAB}"
        )));
    }
    if args.top_k > EXPONENTIAL_RACE_MAX_TOP_K {
        return Err(Error::Other(format!(
            "exponential-race top_k {} exceeds {EXPONENTIAL_RACE_MAX_TOP_K}",
            args.top_k
        )));
    }
    if args.temperature <= 0.0 {
        return Err(Error::Other(
            "exponential-race temperature must be positive".into(),
        ));
    }
    let dtype = match args.logits.dtype() {
        DType::F16 => 1u32,
        DType::BF16 => 2,
        other => {
            return Err(Error::Other(format!(
                "exponential-race logits must be F16 or BF16, got {other}"
            )))
        }
    };
    let logits = tensor_buffer(ctx, args.logits, args.logits.dtype(), vocab, "logits")?;
    // The seen-flag buffer outlives the active vocabulary: before EOS joins
    // the candidates the logits cover 1024 entries while the buffer holds all
    // GPT_AUDIO_VOCAB slots, so only require full coverage of the logits.
    let seen_count = args.seen.shape().numel();
    if seen_count < vocab {
        return Err(Error::Other(format!(
            "exponential-race seen buffer ({seen_count}) must cover the vocabulary ({vocab})"
        )));
    }
    let seen = tensor_buffer(ctx, args.seen, DType::I8, seen_count, "seen")?;
    let output = tensor_buffer(ctx, args.output, DType::I32, 4, "output")?;
    unsafe {
        status::check(sampling_abi::apxinf_sampling_exponential_race(
            logits.ptr(),
            seen.ptr(),
            output.ptr(),
            vocab as i64,
            args.seed,
            args.step,
            args.top_k,
            args.temperature,
            args.repetition_penalty,
            dtype,
            ctx.stream().handle(),
        ))
    }
}

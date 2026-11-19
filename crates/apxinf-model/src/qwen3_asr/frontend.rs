//! GPU log-mel frontend for qwen3-asr.
//!
//! The mel transform is numerically identical to Whisper's (128 bins,
//! n_fft 400, hop 160, Slaney filters, log10, floor(max-8), (x+4)/4), so the
//! pipeline reuses the whisper native helpers: reflect-padded frames, DFT
//! via F32 GEMMs, power, filterbank, log-mel. The HF feature extractor pads
//! the mel time axis with 0.0 beyond the valid frames; we zero those rows on
//! the host after the F16 log-mel readback and upload BF16 for the conv stem.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::ops::whisper_frontend::{
    frames as whisper_frames, log_mel as whisper_log_mel, power as whisper_power, to_device_f32,
};
use apxinf_cuda_new::ops::GemmArgs;
use apxinf_cuda_new::{transfers, CudaContext as Context};

use super::config::{FRAMES, MEL_BINS, SAMPLES};
use super::weights::Weights;

pub const N_FFT: usize = 400;
pub const HOP: usize = 160;
pub const PAD: usize = 200;
pub const BINS: usize = 201;

/// Device-resident frontend constants, uploaded once at load.
pub struct Frontend {
    /// Periodic Hann window, `[400]` F32.
    hann: Tensor,
    /// Cos projection matrix `[400, 201]` F32 (GEMM b operand).
    cos_mat: Tensor,
    /// Sin projection matrix `[400, 201]` F32.
    sin_mat: Tensor,
    /// Mel filterbank `[201, 128]` F32.
    mel_filter: Tensor,
    /// F32 scratch for the log-mel global-max reduction.
    max_scratch: Tensor,
}

impl Frontend {
    pub fn load(weights: &mut Weights, ctx: &Context) -> Result<Self> {
        let mel_filter_host = weights.host_clone("frontend.mel_filter")?;
        if mel_filter_host.dtype() != DType::F32 {
            return Err(Error::Other(format!(
                "mel_filter must be F32, got {:?}",
                mel_filter_host.dtype()
            )));
        }
        if mel_filter_host.shape().dims() != [BINS, MEL_BINS] {
            return Err(Error::Other(format!(
                "mel_filter has shape {:?}, expected [{BINS}, {MEL_BINS}]",
                mel_filter_host.shape().dims()
            )));
        }
        let mel_filter = transfers::to_cuda_on(ctx, &mel_filter_host)?;

        let hann: Vec<f32> = (0..N_FFT)
            .map(|n| {
                (std::f32::consts::PI * (n as f32) / N_FFT as f32)
                    .sin()
                    .powi(2)
            })
            .collect();
        let hann = to_device_f32(ctx, &[N_FFT], &hann)?;

        let mut cos_rows = Vec::with_capacity(N_FFT * BINS);
        let mut sin_rows = Vec::with_capacity(N_FFT * BINS);
        for j in 0..N_FFT {
            for k in 0..BINS {
                let angle = 2.0 * std::f32::consts::PI * (k as f32) * (j as f32) / N_FFT as f32;
                cos_rows.push(angle.cos());
                sin_rows.push(angle.sin());
            }
        }
        let cos_mat = to_device_f32(ctx, &[N_FFT, BINS], &cos_rows)?;
        let sin_mat = to_device_f32(ctx, &[N_FFT, BINS], &sin_rows)?;

        let blocks = ((FRAMES * MEL_BINS + 255) / 256).min(4096).max(1);
        let max_scratch = to_device_f32(ctx, &[blocks + 1], &vec![0.0; blocks + 1])?;

        Ok(Self {
            hann,
            cos_mat,
            sin_mat,
            mel_filter,
            max_scratch,
        })
    }

    /// Run the frontend on a device-resident F32 waveform (`[480000]`,
    /// zero-padded by the caller) with `valid_frames` leading mel frames
    /// kept; the rest are zeroed. Returns the `[3000, 128]` BF16 mel.
    pub fn run(&self, ctx: &Context, waveform: &Tensor, valid_frames: usize) -> Result<Tensor> {
        if waveform.dtype() != DType::F32 {
            return Err(Error::Other(format!(
                "qwen3-asr frontend expects an F32 waveform, got {:?}",
                waveform.dtype()
            )));
        }
        if waveform.shape().numel() != SAMPLES {
            return Err(Error::Other(format!(
                "qwen3-asr frontend expects {SAMPLES} samples, got {}",
                waveform.shape().numel()
            )));
        }
        if waveform.device() != Device::Cuda(ctx.device_id()) {
            return Err(Error::Other("qwen3-asr frontend waveform is off-device".into()));
        }

        // 1. Reflect-pad + Hann-windowed frames: [3000, 400] F32.
        let mut frames = ctx.allocate_output(Shape::new(vec![FRAMES, N_FFT]), DType::F32)?;
        whisper_frames(ctx, waveform, &self.hann, &mut frames, N_FFT, HOP, PAD)?;

        // 2. DFT projections: [3000, 201] each, F32 GEMM.
        let re = self.gemm_f32(ctx, &frames, &self.cos_mat)?;
        let im = self.gemm_f32(ctx, &frames, &self.sin_mat)?;

        // 3. Power: [3000, 201].
        let mut power =
            ctx.allocate_output(Shape::new(vec![FRAMES, BINS]), DType::F32)?;
        whisper_power(ctx, &re, &im, &mut power)?;

        // 4. Mel filterbank: [3000, 201] @ [201, 128] -> [3000, 128].
        let mel_spec = self.gemm_f32(ctx, &power, &self.mel_filter)?;

        // 5. Log-mel transform + F16 cast: [3000, 128].
        let mut mel_f16 =
            ctx.allocate_output(Shape::new(vec![FRAMES, MEL_BINS]), DType::F16)?;
        let mut scratch = self.max_scratch.clone();
        whisper_log_mel(ctx, &mel_spec, &mut mel_f16, &mut scratch)?;

        // 6. Host roundtrip: read F16, zero padding frames, upload BF16.
        ctx.synchronize().map_err(|e| Error::Other(format!("sync: {e}")))?;
        let bytes = {
            use apxinf_cuda_new::CudaBuffer;
            let buffer = CudaBuffer::from_tensor(&mel_f16).map_err(Error::Cuda)?;
            let mut bytes = vec![0u8; FRAMES * MEL_BINS * 2];
            buffer
                .copy_to_host(&mut bytes)
                .map_err(|e| Error::Other(format!("mel readback: {e}")))?;
            bytes
        };
        let mut out = Vec::with_capacity(FRAMES * MEL_BINS);
        for (frame, chunk) in bytes.chunks_exact(MEL_BINS * 2).enumerate() {
            let keep = frame < valid_frames;
            for b in chunk.chunks_exact(2) {
                if keep {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    let value = half::f16::from_bits(bits).to_f32();
                    out.push(half::bf16::from_f32(value));
                } else {
                    out.push(half::bf16::from_f32(0.0));
                }
            }
        }
        let host = Tensor::from_bf16(vec![FRAMES, MEL_BINS], &out)?;
        transfers::to_cuda_on(ctx, &host)
    }

    fn gemm_f32(&self, ctx: &Context, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
            return Err(Error::Other(format!(
                "qwen3-asr frontend gemm shape mismatch: a={a_dims:?} b={b_dims:?}"
            )));
        }
        let mut out = ctx.allocate_output(
            Shape::new(vec![a_dims[0], b_dims[1]]),
            DType::F32,
        )?;
        apxinf_cuda_new::ops::gemm(ctx, GemmArgs::new(a, b, &mut out))?;
        Ok(out)
    }
}

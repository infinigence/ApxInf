//! Audio decode of the denoised latents: denormalise, run the BigVGAN decoder on the two stereo channels as a batch of two mono signals,
//! clamp to `[-1, 1]`.
use crate::cuda_minimax_h3::backend::{CudaBuffer, CudaContext};
use crate::cuda_minimax_h3::config::*;
use crate::cuda_minimax_h3::model::blocks::AudioVaeBlocks;
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::rc::Rc;

pub(in crate::cuda_minimax_h3) struct AudioDecoder {
    ctx: Rc<CudaContext>,
    blocks: AudioVaeBlocks,
}

fn upload_f32(ctx: &CudaContext, values: &[f32], shape: Vec<usize>) -> Result<Tensor> {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).map_err(Error::Cuda)?;
    buffer
        .copy_from_host_on(&bytes, ctx.stream())
        .map_err(Error::Cuda)?;
    buffer
        .as_tensor(Shape::new(shape), DType::F32)
        .map_err(Error::Cuda)
}

pub(in crate::cuda_minimax_h3) fn download_f32(
    ctx: &CudaContext,
    tensor: &Tensor,
) -> Result<Vec<f32>> {
    let mut bytes = vec![0u8; tensor.size_in_bytes()];
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .copy_to_host_on(&mut bytes, ctx.stream())
        .map_err(Error::Cuda)?;
    ctx.synchronize().map_err(Error::Cuda)?;
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

impl AudioDecoder {
    pub fn new(ctx: Rc<CudaContext>, blocks: AudioVaeBlocks) -> Self {
        Self { ctx, blocks }
    }

    /// Denormalised latents as the decoder's batch `[2, AUDIO_CHANNELS, frames]`, from the sampler's `[AUDIO_CHANNELS, 2, frames]`.
    pub fn batch_of(&self, latent: &[f32], frames: usize) -> Result<Vec<f32>> {
        if latent.len() != AUDIO_CHANNELS * AUDIO_STEREO * frames {
            return Err(Error::Other(format!(
                "audio latent has {} values, expected {}",
                latent.len(),
                AUDIO_CHANNELS * AUDIO_STEREO * frames
            )));
        }
        let w = self.blocks.weights();
        let mut batch = vec![0f32; latent.len()];
        for c in 0..AUDIO_CHANNELS {
            for s in 0..AUDIO_STEREO {
                for t in 0..frames {
                    batch[(s * AUDIO_CHANNELS + c) * frames + t] =
                        latent[(c * AUDIO_STEREO + s) * frames + t] * w.latents_std[c]
                            + w.latents_mean[c];
                }
            }
        }
        Ok(batch)
    }

    /// Decode a normalised audio latent `[AUDIO_CHANNELS, 2, frames]` (host, FP32, as the sampler returns it) into `[2, frames * 800]`
    /// samples in `[-1, 1]` (channel-major). `tap` sees the decoder's stage activations (see `AudioVaeBlocks::decode`).
    pub fn decode_with_taps(
        &self,
        latent: &[f32],
        frames: usize,
        tap: &mut dyn FnMut(&str, &Tensor) -> Result<()>,
    ) -> Result<Vec<f32>> {
        let batch = self.batch_of(latent, frames)?;
        let input = upload_f32(
            &self.ctx,
            &batch,
            vec![AUDIO_STEREO, AUDIO_CHANNELS, frames],
        )?;
        let wave = self.blocks.decode(&input, tap)?;
        let mut samples = download_f32(&self.ctx, &wave)?;
        samples.iter_mut().for_each(|v| *v = v.clamp(-1.0, 1.0));
        Ok(samples)
    }

    pub fn decode(&self, latent: &[f32], frames: usize) -> Result<Vec<f32>> {
        self.decode_with_taps(latent, frames, &mut |_, _| Ok(()))
    }
}

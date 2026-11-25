//! Video decode of the denoised latents: temporal chunks, spatial tiles in batches, the blends of both, and the final pixel
//! denormalisation. The decoder weights stream once per batch of tiles, so a batch is as large as device memory allows.
use crate::cuda_minimax_h3::backend::ops;
use crate::cuda_minimax_h3::backend::ops::GemmPolicy;
use crate::cuda_minimax_h3::backend::{AttentionPolicy, CudaBuffer, CudaContext, ExecutionSession};
use crate::cuda_minimax_h3::config::*;
use crate::cuda_minimax_h3::model::blocks::VideoVaeBlocks;
use crate::cuda_minimax_h3::model::vae_host::{
    assemble, latent_tile_tokens, splice_chunks, tile_tokens, unpatchify_pixels, TemporalPlan,
    TilePlan, Vid,
};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::rc::Rc;

/// Default tiles decoded by one pass over the decoder weights. Each tile of a 7-step chunk holds 1797 tokens and its arena share is about
/// 170 MiB; two tiles fit next to the resident DiT, its denoising plan and the text-encoder ring on a 24 GiB card, four do not (measured:
/// out of memory at 768x448). Larger batches decode faster when memory allows (`APXINF_H3_VAE_BATCH`).
pub(in crate::cuda_minimax_h3) const TILE_BATCH: usize = 2;

/// Seconds spent in each part of one decode, accumulated over its chunks and tile batches (printed with `APXINF_H3_TIMINGS`).
#[derive(Default, Clone, Copy, Debug)]
pub(in crate::cuda_minimax_h3) struct DecodeTimes {
    pub gather_upload: f64,
    pub device: f64,
    pub download_unpatchify: f64,
    pub assemble: f64,
    pub splice: f64,
}

/// One batch shape prepared for repeated eager runs: a session arena that sub-allocates every intermediate at fixed offsets, the
/// native GEMM / attention executions bound to those addresses, and persistent input, output and rotary tables. Every chunk of a clip is
/// `VV_CHUNK_LATENT` steps, so one prepared batch serves every chunk of every clip.
struct PreparedBatch {
    session: ExecutionSession,
    input: Tensor,
    output: Tensor,
    rope: Tensor,
}

pub(in crate::cuda_minimax_h3) struct VideoDecoder {
    times: std::cell::Cell<DecodeTimes>,
    prepared: std::cell::RefCell<Option<PreparedBatch>>,
    batch: usize,
    ctx: Rc<CudaContext>,
    blocks: VideoVaeBlocks,
    gemm: GemmPolicy,
    attention: AttentionPolicy,
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

impl VideoDecoder {
    pub fn new(
        ctx: Rc<CudaContext>,
        blocks: VideoVaeBlocks,
        batch: usize,
        gemm: GemmPolicy,
        attention: AttentionPolicy,
    ) -> Self {
        Self {
            times: Default::default(),
            prepared: Default::default(),
            batch: batch.max(1),
            ctx,
            blocks,
            gemm,
            attention,
        }
    }

    /// Arena bytes for one batch: the measured high water of the block loop is 96 KiB per token (the feed-forward's gate, up and
    /// activation beside the residual stream and the attention operands), so 112 KiB per token plus a fixed 128 MiB of headroom.
    fn arena_bytes(tiles: usize) -> usize {
        tiles * tile_tokens(VV_CHUNK_LATENT) * 112 * 1024 + 128 * 1024 * 1024
    }

    /// `APXINF_H3_VAE_SESSION=0` runs every batch with per-call allocation instead of a prepared session (comparison, debugging).
    fn use_session() -> bool {
        !matches!(
            std::env::var("APXINF_H3_VAE_SESSION").as_deref(),
            Ok("0" | "off")
        )
    }

    fn prepare(&self) -> Result<PreparedBatch> {
        let tiles = self.batch;
        let steps = VV_CHUNK_LATENT;
        let per_tile = tile_tokens(steps);
        let n = VV_TILE_LATENT;
        let persistent = |shape: Vec<usize>, dtype: DType| -> Result<Tensor> {
            let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
            CudaBuffer::alloc_zeros(bytes, self.ctx.device_id())
                .map_err(Error::Cuda)?
                .as_tensor(Shape::new(shape), dtype)
                .map_err(Error::Cuda)
        };
        let input = persistent(vec![tiles * steps * n * n, LATENT_CHANNELS], DType::F32)?;
        let output = persistent(vec![tiles * per_tile, VV_OUT_DIM], DType::BF16)?;
        let rope = self.blocks.rope_table(tiles, steps)?;
        let session =
            ExecutionSession::with_capacity(Self::arena_bytes(tiles), self.ctx.device_id())?;
        // preparation traverses the real forward once (tuning GEMM shapes, binding executions); zero latents are finite and harmless
        session.prepare(|| self.forward(&input, &output, &rope, tiles, steps))?;
        self.ctx.synchronize().map_err(Error::Cuda)?;
        if std::env::var_os("APXINF_H3_TIMINGS").is_some() {
            let w = session.workspace();
            eprintln!(
                "  video decode arena: {:.0} MiB high water of {:.0} MiB for {tiles} tiles x {} tokens ({:.1} KiB per token)",
                w.peak_used() as f64 / 1048576.0,
                w.capacity() as f64 / 1048576.0,
                tile_tokens(steps),
                w.peak_used() as f64 / 1024.0 / (tiles * tile_tokens(steps)) as f64
            );
        }
        Ok(PreparedBatch {
            session,
            input,
            output,
            rope,
        })
    }

    fn forward(
        &self,
        input: &Tensor,
        output: &Tensor,
        rope: &Tensor,
        tiles: usize,
        steps: usize,
    ) -> Result<()> {
        let rows =
            self.blocks
                .decode_with(input, rope, tiles, steps, &self.gemm, &self.attention)?;
        ops::copy_tensor(&self.ctx, &rows, output)
    }

    /// Decode one batch of `tiles` real tiles of `VV_CHUNK_LATENT` steps (`tokens` is their latent tokens) through the prepared session;
    /// the batch is padded with zero latents, which never influence the real tiles because tiles do not attend to each other.
    fn run_prepared(&self, tokens: &[f32], tiles: usize) -> Result<Vec<f32>> {
        let mut slot = self.prepared.borrow_mut();
        if slot.is_none() {
            *slot = Some(self.prepare()?);
        }
        let prepared = slot.as_ref().expect("prepared batch was just installed");
        let per_tile_in = VV_CHUNK_LATENT * VV_TILE_LATENT * VV_TILE_LATENT * LATENT_CHANNELS;
        let mut padded = tokens.to_vec();
        padded.resize(self.batch * per_tile_in, 0.0);
        let bytes: Vec<u8> = padded.iter().flat_map(|v| v.to_le_bytes()).collect();
        CudaBuffer::from_tensor(&prepared.input)
            .map_err(Error::Cuda)?
            .copy_from_host_on(&bytes, self.ctx.stream())
            .map_err(Error::Cuda)?;
        let clock = std::time::Instant::now();
        prepared.session.run(|| {
            self.forward(
                &prepared.input,
                &prepared.output,
                &prepared.rope,
                self.batch,
                VV_CHUNK_LATENT,
            )
        })?;
        self.ctx.synchronize().map_err(Error::Cuda)?;
        self.add(|t| t.device += clock.elapsed().as_secs_f64());
        let clock = std::time::Instant::now();
        let mut out = vec![0u8; prepared.output.size_in_bytes()];
        CudaBuffer::from_tensor(&prepared.output)
            .map_err(Error::Cuda)?
            .copy_to_host_on(&mut out, self.ctx.stream())
            .map_err(Error::Cuda)?;
        self.ctx.synchronize().map_err(Error::Cuda)?;
        let per_tile_out = tile_tokens(VV_CHUNK_LATENT) * VV_OUT_DIM;
        let values: Vec<f32> = out[..tiles * per_tile_out * 2]
            .chunks_exact(2)
            .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect();
        self.add(|t| t.download_unpatchify += clock.elapsed().as_secs_f64());
        Ok(values)
    }

    fn add(&self, f: impl FnOnce(&mut DecodeTimes)) {
        let mut t = self.times.get();
        f(&mut t);
        self.times.set(t);
    }

    /// Decode a normalised latent `[LATENT_CHANNELS, t, h, w]` (host, FP32, as the sampler returns it) into
    /// `[PIXEL_CHANNELS, frames, h * 16, w * 16]` pixels in `[0, 1]`.
    pub fn decode(&self, latent: &[f32], dims: [usize; 3]) -> Result<Vid> {
        let [t, h, w] = dims;
        if latent.len() != LATENT_CHANNELS * t * h * w {
            return Err(Error::Other(format!(
                "video latent has {} values, expected {}",
                latent.len(),
                LATENT_CHANNELS * t * h * w
            )));
        }
        let (height, width) = (h * VAE_RATIO, w * VAE_RATIO);
        if height < VV_TILE_PIXELS || width < VV_TILE_PIXELS {
            return Err(Error::Other(format!(
                "MiniMax-H3 video sides must be at least {VV_TILE_PIXELS} pixels (one VAE tile), got {width}x{height}"
            )));
        }
        // a clip is padded with copies of its last latent step so the chunks tile it
        let plan = TemporalPlan::new(t)?;
        let padded_steps = t + plan.pad_tokens;
        self.times.set(DecodeTimes::default());
        let started = std::time::Instant::now();
        let video = splice_chunks(t, &plan, |start, end| {
            let steps = end - start;
            let source = |step: usize| step.min(t - 1);
            // latent steps of this chunk, with the padding copies of the last step expanded
            let mut chunk = Vec::with_capacity(LATENT_CHANNELS * steps * h * w);
            for c in 0..LATENT_CHANNELS {
                for step in start..end {
                    let from = ((c * t + source(step)) * h) * w;
                    chunk.extend_from_slice(&latent[from..from + h * w]);
                }
            }
            debug_assert!(end <= padded_steps);
            self.decode_chunk(&chunk, [steps, h, w])
        });
        let total = started.elapsed().as_secs_f64();
        let t = self.times.get();
        self.add(|t| {
            t.splice =
                (total - t.gather_upload - t.device - t.download_unpatchify - t.assemble).max(0.0)
        });
        if std::env::var_os("APXINF_H3_TIMINGS").is_some() {
            let t = self.times.get();
            eprintln!(
                "  video decode {total:6.3} s: gather+upload {:.3}, device {:.3}, download+unpatchify {:.3}, tile blend {:.3}, splice+pixels {:.3}",
                t.gather_upload, t.device, t.download_unpatchify, t.assemble, t.splice
            );
        }
        let _ = t;
        video
    }

    /// One chunk: all its spatial tiles, in batches, blended into one `[3, 4 * steps, height, width]` volume.
    fn decode_chunk(&self, chunk: &[f32], dims: [usize; 3]) -> Result<Vid> {
        let [steps, h, w] = dims;
        let (height, width) = (h * VAE_RATIO, w * VAE_RATIO);
        let (ys, xs) = (TilePlan::new(height), TilePlan::new(width));
        let mut grid: Vec<(usize, usize)> = Vec::new();
        for row in 0..ys.count() {
            for col in 0..xs.count() {
                grid.push((row, col));
            }
        }
        let n = VV_TILE_LATENT;
        let mut decoded: Vec<Option<Vid>> = (0..grid.len()).map(|_| None).collect();
        for batch_start in (0..grid.len()).step_by(self.batch) {
            let batch = &grid[batch_start..(batch_start + self.batch).min(grid.len())];
            let clock = std::time::Instant::now();
            let mut tokens = Vec::with_capacity(batch.len() * steps * n * n * LATENT_CHANNELS);
            for &(row, col) in batch {
                latent_tile_tokens(
                    chunk,
                    [steps, h, w],
                    0,
                    steps,
                    ys.starts[row] / VAE_RATIO,
                    xs.starts[col] / VAE_RATIO,
                    [n, n],
                    &mut tokens,
                );
            }
            let per_tile = tile_tokens(steps);
            let values: Vec<f32> = if Self::use_session() && steps == VV_CHUNK_LATENT {
                self.add(|t| t.gather_upload += clock.elapsed().as_secs_f64());
                self.run_prepared(&tokens, batch.len())?
            } else {
                let input = upload_f32(
                    &self.ctx,
                    &tokens,
                    vec![batch.len() * steps * n * n, LATENT_CHANNELS],
                )?;
                self.ctx.synchronize().map_err(Error::Cuda)?;
                self.add(|t| t.gather_upload += clock.elapsed().as_secs_f64());
                let clock = std::time::Instant::now();
                let out =
                    self.blocks
                        .decode(&input, batch.len(), steps, &self.gemm, &self.attention)?;
                self.ctx.synchronize().map_err(Error::Cuda)?;
                self.add(|t| t.device += clock.elapsed().as_secs_f64());
                let mut bytes = vec![0u8; out.size_in_bytes()];
                CudaBuffer::from_tensor(&out)
                    .map_err(Error::Cuda)?
                    .copy_to_host_on(&mut bytes, self.ctx.stream())
                    .map_err(Error::Cuda)?;
                self.ctx.synchronize().map_err(Error::Cuda)?;
                bytes
                    .chunks_exact(2)
                    .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                    .collect()
            };
            let clock = std::time::Instant::now();
            for (i, _) in batch.iter().enumerate() {
                let rows = &values[i * per_tile * VV_OUT_DIM..][..steps * n * n * VV_OUT_DIM];
                decoded[batch_start + i] = Some(unpatchify_pixels(rows, steps, n, n));
            }
            self.add(|t| t.download_unpatchify += clock.elapsed().as_secs_f64());
        }
        let clock = std::time::Instant::now();
        let volume = assemble(height, width, |row, columns| {
            (0..columns.count())
                .map(|col| {
                    decoded[row * columns.count() + col]
                        .take()
                        .ok_or_else(|| Error::Other("video tile decoded twice".into()))
                })
                .collect()
        });
        self.add(|t| t.assemble += clock.elapsed().as_secs_f64());
        volume
    }
}

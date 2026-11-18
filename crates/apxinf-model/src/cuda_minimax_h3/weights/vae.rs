//! Device weights of the video VAE decoder (`minimax_h3_video_vae_fp16.safetensors`): a 36-block ViT of 2.4 B parameters.
//!
//! The decoder is 4.8 GB as BF16, which the card cannot hold beside the 19.5 GiB DiT, so the weights live on the host, converted once at
//! load into the layout the kernels execute (BF16 `[in, out]`, de-interleaved q/k/v with the rotated channels permuted, gate and up as
//! separate linears) and cut into one contiguous byte range per block. A video decode streams them to a two-slot device ring on a copy
//! stream while the compute stream works on the previous block, as the text encoder does. The embedding, the registers and the head are
//! small and stay on the device.
use super::bf16::f16_vec;
use super::host::Checkpoint;
use crate::cuda_minimax_h3::backend::{
    CudaBuffer, CudaContext, CudaEvent, CudaStream, PinnedRegion,
};
use crate::cuda_minimax_h3::config::*;
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

/// Element counts of the tensors of one block, in the order they are packed in its byte range.
const BLOCK_PARTS: [(&str, usize); 14] = [
    ("norm1", VV_DIM),
    ("norm2", VV_DIM),
    ("q.weight", VV_DIM * VV_DIM),
    ("q.bias", VV_DIM),
    ("k.weight", VV_DIM * VV_DIM),
    ("k.bias", VV_DIM),
    ("v.weight", VV_DIM * VV_DIM),
    ("v.bias", VV_DIM),
    ("out.weight", VV_DIM * VV_DIM),
    ("out.bias", VV_DIM),
    ("gate.weight", VV_DIM * VV_FFN),
    ("gate.bias", VV_FFN),
    ("up.weight", VV_DIM * VV_FFN),
    ("up.bias", VV_FFN),
];
/// `down` and the two residual scales close the block (kept apart so the table above stays a single list of 14).
const DOWN: (usize, usize) = (VV_FFN * VV_DIM, VV_DIM);

/// BF16 elements of one block.
pub(in crate::cuda_minimax_h3) fn block_elements() -> usize {
    BLOCK_PARTS.iter().map(|(_, n)| n).sum::<usize>() + DOWN.0 + DOWN.1 + 2 * 3 * VV_DIM
}

/// One linear of a resident block: BF16 weight `[in, out]` and bias `[out]`.
pub(in crate::cuda_minimax_h3) struct VvLinear {
    pub weight: Tensor,
    pub bias: Tensor,
}

/// The tensors of one block, as views into one device buffer.
pub(in crate::cuda_minimax_h3) struct VvBlock {
    pub norm1: Tensor,
    pub norm2: Tensor,
    pub q: VvLinear,
    pub k: VvLinear,
    pub v: VvLinear,
    pub out: VvLinear,
    pub gate: VvLinear,
    pub up: VvLinear,
    pub down: VvLinear,
    /// Residual scales as packed styles `[0; 0; scale]` (BF16 `[3 * VV_DIM]`): `x + scale * branch` is a gated residual.
    pub style1: Tensor,
    pub style2: Tensor,
    /// Keeps the backing allocation alive; every tensor above is a view of it.
    _buffer: CudaBuffer,
}

impl VvBlock {
    /// The one allocation every tensor of the block is a view of; a refill is a single copy into it.
    fn buffer(&self) -> &CudaBuffer {
        &self._buffer
    }

    fn new(ctx: &CudaContext) -> Result<Self> {
        let bytes = block_elements() * 2;
        let buffer = CudaBuffer::alloc(bytes, ctx.device_id()).map_err(Error::Cuda)?;
        let mut cursor = 0usize;
        let mut take = |shape: Vec<usize>| -> Result<Tensor> {
            let len = shape.iter().product::<usize>() * 2;
            let view = buffer.view(cursor, len).map_err(Error::Cuda)?;
            cursor += len;
            view.as_tensor(Shape::new(shape), DType::BF16)
                .map_err(Error::Cuda)
        };
        let norm1 = take(vec![VV_DIM])?;
        let norm2 = take(vec![VV_DIM])?;
        let mut linear = |n_in: usize, n_out: usize| -> Result<VvLinear> {
            Ok(VvLinear {
                weight: take(vec![n_in, n_out])?,
                bias: take(vec![n_out])?,
            })
        };
        let q = linear(VV_DIM, VV_DIM)?;
        let k = linear(VV_DIM, VV_DIM)?;
        let v = linear(VV_DIM, VV_DIM)?;
        let out = linear(VV_DIM, VV_DIM)?;
        let gate = linear(VV_DIM, VV_FFN)?;
        let up = linear(VV_DIM, VV_FFN)?;
        let down = linear(VV_FFN, VV_DIM)?;
        let style1 = take(vec![3 * VV_DIM])?;
        let style2 = take(vec![3 * VV_DIM])?;
        debug_assert_eq!(cursor, bytes);
        Ok(Self {
            norm1,
            norm2,
            q,
            k,
            v,
            out,
            gate,
            up,
            down,
            style1,
            style2,
            _buffer: buffer,
        })
    }
}

/// The small tensors that stay on the device.
pub(in crate::cuda_minimax_h3) struct VvResident {
    /// `x_embedder(post_quant_conv(z * std + mean))` as one FP32 linear on normalised latents `[LATENT_CHANNELS, VV_DIM]` + `[VV_DIM]`.
    pub embed_weight: Tensor,
    pub embed_bias: Tensor,
    pub ones: Tensor,
    pub registers: Tensor,
    pub norm_out_weight: Tensor,
    pub norm_out_bias: Tensor,
    pub proj_out_weight: Tensor,
    pub proj_out_bias: Tensor,
}

pub(in crate::cuda_minimax_h3) struct VideoVaeWeights {
    ctx: Rc<CudaContext>,
    pub resident: VvResident,
    /// One pinned host range per block, `block_elements()` BF16 elements each.
    blob: PinnedRegion,
    ring: Vec<VvBlock>,
    copy: CudaStream,
    copied: Vec<CudaEvent>,
    consumed: Vec<CudaEvent>,
}

const RING: usize = 2;

/// Channel permutation of one 64-channel head so the decoder's rotation of the first 48 channels (split-half pairs `(i, i + 24)`)
/// becomes the half-split rotation over all 64 channels `rms_norm_apply_precomputed` implements: pairs `(j, j + 32)` rotate, the
/// pass-through channels `48..64` are paired with themselves under identity coefficients. Permuted channel `c` holds original `P[c]`.
pub(in crate::cuda_minimax_h3) fn vae_rope_permutation() -> [usize; VV_HEAD_DIM] {
    let half = VV_ROPE_DIM / 2;
    let mut p = [0usize; VV_HEAD_DIM];
    for j in 0..VV_HEAD_DIM / 2 {
        if j < half {
            p[j] = j;
            p[j + VV_HEAD_DIM / 2] = j + half;
        } else {
            // pass-through channels 48..64 pair with themselves: put two of them at (j, j + 32)
            p[j] = j + half;
            p[j + VV_HEAD_DIM / 2] = j + VV_HEAD_DIM / 2;
        }
    }
    p
}

/// Row indices (into a `[heads * 3 * head_dim, k]` per-head-interleaved qkv matrix) of projection `part` (0 q, 1 k, 2 v). Q and K rows
/// are channel-permuted within each head for the rotation; V rows keep their order.
pub(super) fn qkv_rows(part: usize) -> Vec<usize> {
    let permutation = vae_rope_permutation();
    (0..VV_HEADS)
        .flat_map(move |h| {
            (0..VV_HEAD_DIM).map(move |d| {
                let channel = if part < 2 { permutation[d] } else { d };
                h * 3 * VV_HEAD_DIM + part * VV_HEAD_DIM + channel
            })
        })
        .collect()
}

/// Compose `x_embedder(post_quant_conv(z * std + mean))` into one affine map `z -> z @ W + b` in FP64.
/// `pq_weight` is `[out, in]` (a 1x1x1 convolution), `embed_weight` is `[VV_DIM, LATENT_CHANNELS]`. Returns `W` as row-major
/// `[LATENT_CHANNELS, VV_DIM]` and `b` as `[VV_DIM]`.
pub(super) fn fold_embedding(
    pq_weight: &[f32],
    pq_bias: &[f32],
    embed_weight: &[f32],
    embed_bias: &[f32],
    mean: &[f32],
    std: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let c = LATENT_CHANNELS;
    let mut weight = vec![0f32; c * VV_DIM];
    let mut bias = vec![0f32; VV_DIM];
    for d in 0..VV_DIM {
        let mut b = embed_bias[d] as f64;
        let mut column = vec![0f64; c];
        for o in 0..c {
            let e = embed_weight[d * c + o] as f64;
            b += e * pq_bias[o] as f64;
            for i in 0..c {
                let p = pq_weight[o * c + i] as f64;
                column[i] += e * p * std[i] as f64;
                b += e * p * mean[i] as f64;
            }
        }
        for i in 0..c {
            weight[i * VV_DIM + d] = column[i] as f32;
        }
        bias[d] = b as f32;
    }
    (weight, bias)
}

fn bf16_of(values: &[half::f16]) -> Vec<half::bf16> {
    values
        .iter()
        .map(|v| half::bf16::from_f32(v.to_f32()))
        .collect()
}

fn to_vec_bf16(values: &[half::bf16], out: &mut Vec<u8>) {
    out.extend(values.iter().flat_map(|v| v.to_le_bytes()));
}

/// `[rows, inp]` F16 matrix restricted to `selection` rows and transposed to `[inp, selection.len()]` BF16 bytes, appended to `out`.
fn push_transposed(all: &[half::f16], inp: usize, selection: &[usize], out: &mut Vec<u8>) {
    let n = selection.len();
    let mut transposed = vec![half::bf16::ZERO; inp * n];
    for (col, &row) in selection.iter().enumerate() {
        let source = &all[row * inp..(row + 1) * inp];
        for (k, v) in source.iter().enumerate() {
            transposed[k * n + col] = half::bf16::from_f32(v.to_f32());
        }
    }
    to_vec_bf16(&transposed, out);
}

fn push_bias(all: &[half::f16], selection: &[usize], out: &mut Vec<u8>) {
    let picked: Vec<half::f16> = selection.iter().map(|&r| all[r]).collect();
    to_vec_bf16(&bf16_of(&picked), out);
}

/// Pack block `index` of the checkpoint into the byte layout of `VvBlock::new`.
fn pack_block(file: &mut Checkpoint, index: usize) -> Result<Vec<u8>> {
    let p = format!("decoder.transformer_blocks.{index}");
    let mut out = Vec::with_capacity(block_elements() * 2);
    let mut vector = |name: &str, len: usize, out: &mut Vec<u8>| -> Result<()> {
        let raw = file.read(name, Some(DType::F16), &[len])?;
        to_vec_bf16(&bf16_of(&f16_vec(&raw)), out);
        Ok(())
    };
    vector(&format!("{p}.norm1.weight"), VV_DIM, &mut out)?;
    vector(&format!("{p}.norm2.weight"), VV_DIM, &mut out)?;
    let qkv = f16_vec(&file.read(
        &format!("{p}.attn.to_qkv.weight"),
        Some(DType::F16),
        &[3 * VV_DIM, VV_DIM],
    )?);
    let qkv_bias = f16_vec(&file.read(
        &format!("{p}.attn.to_qkv.bias"),
        Some(DType::F16),
        &[3 * VV_DIM],
    )?);
    for part in 0..3 {
        let rows = qkv_rows(part);
        push_transposed(&qkv, VV_DIM, &rows, &mut out);
        push_bias(&qkv_bias, &rows, &mut out);
    }
    let all: Vec<usize> = (0..VV_DIM).collect();
    let o = f16_vec(&file.read(
        &format!("{p}.attn.to_out.weight"),
        Some(DType::F16),
        &[VV_DIM, VV_DIM],
    )?);
    let ob = f16_vec(&file.read(
        &format!("{p}.attn.to_out.bias"),
        Some(DType::F16),
        &[VV_DIM],
    )?);
    push_transposed(&o, VV_DIM, &all, &mut out);
    push_bias(&ob, &all, &mut out);
    // `w1` is `[gate; up]` in contiguous halves
    let w1 = f16_vec(&file.read(
        &format!("{p}.ff.w1.weight"),
        Some(DType::F16),
        &[2 * VV_FFN, VV_DIM],
    )?);
    let b1 = f16_vec(&file.read(&format!("{p}.ff.w1.bias"), Some(DType::F16), &[2 * VV_FFN])?);
    for half_index in 0..2 {
        let rows: Vec<usize> = (half_index * VV_FFN..(half_index + 1) * VV_FFN).collect();
        push_transposed(&w1, VV_DIM, &rows, &mut out);
        push_bias(&b1, &rows, &mut out);
    }
    let w2 = f16_vec(&file.read(
        &format!("{p}.ff.w2.weight"),
        Some(DType::F16),
        &[VV_DIM, VV_FFN],
    )?);
    let b2 = f16_vec(&file.read(&format!("{p}.ff.w2.bias"), Some(DType::F16), &[VV_DIM])?);
    push_transposed(&w2, VV_FFN, &all, &mut out);
    push_bias(&b2, &all, &mut out);
    for name in ["scale1", "scale2"] {
        let scale = bf16_of(&f16_vec(&file.read(
            &format!("{p}.{name}"),
            Some(DType::F16),
            &[VV_DIM],
        )?));
        to_vec_bf16(&vec![half::bf16::ZERO; 2 * VV_DIM], &mut out);
        to_vec_bf16(&scale, &mut out);
    }
    debug_assert_eq!(out.len(), block_elements() * 2);
    Ok(out)
}

fn device_tensor(
    ctx: &CudaContext,
    values: &[u8],
    shape: Vec<usize>,
    dtype: DType,
) -> Result<Tensor> {
    let buffer = CudaBuffer::alloc(values.len(), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(values).map_err(Error::Cuda)?;
    buffer
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

impl VideoVaeWeights {
    /// Read the checkpoint, convert every block to the device layout in one host buffer, page-lock it (best effort: a refusal falls
    /// back to pageable copies, three times slower) and allocate the device ring.
    pub fn load(path: &Path, ctx: Rc<CudaContext>) -> Result<Self> {
        let mut file = Checkpoint::open(path)?;
        // the encoder and its quantizer are only needed to encode reference frames, which this family does not do
        let skipped = file.ignore_prefix("encoder.") + file.ignore_prefix("quant_conv.");
        file.ignore_prefix("decoder.mask_token"); // unused at inference; present so the file loads without leftovers
        if skipped == 0 {
            return Err(Error::Other(format!(
                "{}: not a MiniMax-H3 video VAE (no encoder tensors)",
                path.display()
            )));
        }
        let to_f32 =
            |bytes: Vec<u8>| -> Vec<f32> { f16_vec(&bytes).iter().map(|v| v.to_f32()).collect() };
        // the file's F16 statistics are what ComfyUI decodes with (they replace the module's float32 constants on load)
        let mean = to_f32(file.read("latents_mean", Some(DType::F16), &[LATENT_CHANNELS])?);
        let std = to_f32(file.read("latents_std", Some(DType::F16), &[LATENT_CHANNELS])?);
        let pq_w = to_f32(file.read(
            "post_quant_conv.weight",
            Some(DType::F16),
            &[LATENT_CHANNELS, LATENT_CHANNELS, 1, 1, 1],
        )?);
        let pq_b =
            to_f32(file.read("post_quant_conv.bias", Some(DType::F16), &[LATENT_CHANNELS])?);
        let e_w = to_f32(file.read(
            "decoder.x_embedder.weight",
            Some(DType::F16),
            &[VV_DIM, LATENT_CHANNELS],
        )?);
        let e_b = to_f32(file.read("decoder.x_embedder.bias", Some(DType::F16), &[VV_DIM])?);
        let (folded_w, folded_b) = fold_embedding(&pq_w, &pq_b, &e_w, &e_b, &mean, &std);
        let f32_bytes = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let mut bf16_tensor = |name: &str, shape: &[usize]| -> Result<Vec<u8>> {
            let raw = file.read(name, Some(DType::F16), shape)?;
            let mut out = Vec::with_capacity(raw.len());
            to_vec_bf16(&bf16_of(&f16_vec(&raw)), &mut out);
            Ok(out)
        };
        let registers = bf16_tensor("decoder.register_tokens", &[1, VV_REGISTERS, VV_DIM])?;
        let norm_w = bf16_tensor("decoder.norm_out.weight", &[VV_DIM])?;
        let norm_b = bf16_tensor("decoder.norm_out.bias", &[VV_DIM])?;
        let proj_w = f16_vec(&file.read(
            "decoder.proj_out.weight",
            Some(DType::F16),
            &[VV_OUT_DIM, VV_DIM],
        )?);
        let proj_b =
            f16_vec(&file.read("decoder.proj_out.bias", Some(DType::F16), &[VV_OUT_DIM])?);
        let mut proj_out = Vec::new();
        push_transposed(
            &proj_w,
            VV_DIM,
            &(0..VV_OUT_DIM).collect::<Vec<_>>(),
            &mut proj_out,
        );
        let mut proj_bias = Vec::new();
        push_bias(
            &proj_b,
            &(0..VV_OUT_DIM).collect::<Vec<_>>(),
            &mut proj_bias,
        );
        let ones = vec![half::bf16::ONE; VV_HEAD_DIM];
        let mut ones_bytes = Vec::new();
        to_vec_bf16(&ones, &mut ones_bytes);
        let resident = VvResident {
            embed_weight: device_tensor(
                &ctx,
                &f32_bytes(&folded_w),
                vec![LATENT_CHANNELS, VV_DIM],
                DType::F32,
            )?,
            embed_bias: device_tensor(&ctx, &f32_bytes(&folded_b), vec![VV_DIM], DType::F32)?,
            ones: device_tensor(&ctx, &ones_bytes, vec![VV_HEAD_DIM], DType::BF16)?,
            registers: device_tensor(&ctx, &registers, vec![VV_REGISTERS, VV_DIM], DType::BF16)?,
            norm_out_weight: device_tensor(&ctx, &norm_w, vec![VV_DIM], DType::BF16)?,
            norm_out_bias: device_tensor(&ctx, &norm_b, vec![VV_DIM], DType::BF16)?,
            proj_out_weight: device_tensor(&ctx, &proj_out, vec![VV_DIM, VV_OUT_DIM], DType::BF16)?,
            proj_out_bias: device_tensor(&ctx, &proj_bias, vec![VV_OUT_DIM], DType::BF16)?,
        };

        let per_block = block_elements() * 2;
        let mut blob = vec![0u8; per_block * VV_BLOCKS];
        for index in 0..VV_BLOCKS {
            let packed = pack_block(&mut file, index)?;
            blob[index * per_block..(index + 1) * per_block].copy_from_slice(&packed);
        }
        file.finish()?;
        let owner: Arc<dyn AsRef<[u8]> + Send + Sync> = Arc::new(blob);
        let blob = match PinnedRegion::register(Arc::clone(&owner), ctx.device_id()) {
            Ok(region) => region,
            Err(e) => {
                eprintln!("MiniMax-H3: page-locking the video VAE weights failed ({e}); they stream through the pageable path");
                PinnedRegion::unregistered(owner, ctx.device_id())
            }
        };
        let ring = (0..RING)
            .map(|_| VvBlock::new(&ctx))
            .collect::<Result<Vec<_>>>()?;
        let copy = CudaStream::new_on(ctx.device_id()).map_err(Error::Cuda)?;
        let events = || {
            (0..RING)
                .map(|_| CudaEvent::new(ctx.device_id()).map_err(Error::Cuda))
                .collect::<Result<Vec<_>>>()
        };
        let (copied, consumed) = (events()?, events()?);
        Ok(Self {
            ctx,
            resident,
            blob,
            ring,
            copy,
            copied,
            consumed,
        })
    }

    /// Whether block copies are DMA transfers (reported by the parity tests).
    #[cfg(test)]
    pub fn is_pinned(&self) -> bool {
        self.blob.is_pinned()
    }

    fn enqueue(&self, block: usize) -> Result<()> {
        let set = block % RING;
        let per_block = block_elements() * 2;
        self.copy
            .wait_for(&self.consumed[set])
            .map_err(Error::Cuda)?;
        // the block's tensors are views of one allocation: a single copy fills them all
        self.ring[set]
            .buffer()
            .copy_from_region_async(0, &self.blob, block * per_block, per_block, &self.copy)
            .map_err(Error::Cuda)?;
        self.copy.record(&self.copied[set]).map_err(Error::Cuda)
    }

    /// Run `block_fn(index, block)` for blocks 0..36 in order. Each call must enqueue all of the block's work on the context stream
    /// before returning (the slot is refilled as soon as the compute stream passes that point), and may not keep the block.
    pub fn run_blocks(
        &self,
        mut block_fn: impl FnMut(usize, &VvBlock) -> Result<()>,
    ) -> Result<()> {
        let result = self.pass(&mut block_fn);
        let drained = self.copy.synchronize().map_err(Error::Cuda);
        result.and(drained)
    }

    fn pass(&self, block_fn: &mut impl FnMut(usize, &VvBlock) -> Result<()>) -> Result<()> {
        self.enqueue(0)?;
        for index in 0..VV_BLOCKS {
            if index + 1 < VV_BLOCKS {
                self.enqueue(index + 1)?;
            }
            let set = index % RING;
            self.ctx
                .stream()
                .wait_for(&self.copied[set])
                .map_err(Error::Cuda)?;
            block_fn(index, &self.ring[set])?;
            self.ctx
                .stream()
                .record(&self.consumed[set])
                .map_err(Error::Cuda)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_packs_to_the_documented_element_count() {
        let parts: usize = BLOCK_PARTS.iter().map(|(_, n)| n).sum();
        assert_eq!(
            parts,
            2 * VV_DIM + 4 * (VV_DIM * VV_DIM + VV_DIM) + 2 * (VV_DIM * VV_FFN + VV_FFN)
        );
        assert_eq!(
            block_elements(),
            parts + VV_FFN * VV_DIM + VV_DIM + 6 * VV_DIM
        );
        // 36 blocks stay below 4.9 GB, the whole decoder's BF16 weight
        assert!(block_elements() * 2 * VV_BLOCKS < 5_000_000_000);
    }

    #[test]
    fn qkv_rows_are_the_per_head_interleaved_thirds() {
        let (q, k, v) = (qkv_rows(0), qkv_rows(1), qkv_rows(2));
        for part in [&q, &k, &v] {
            assert_eq!(part.len(), VV_DIM);
        }
        // head 0: channels 0..24 stay, 24..32 hold the pass-through 48..56, 32..56 hold the second halves 24..48
        assert_eq!((q[0], q[24], q[32], q[63], q[64]), (0, 48, 24, 63, 192));
        let mut all: Vec<usize> = q.into_iter().chain(k).chain(v).collect();
        all.sort_unstable();
        assert_eq!(all, (0..3 * VV_DIM).collect::<Vec<_>>());
    }

    #[test]
    fn rope_permutation_pairs_rotated_channels_half_a_head_apart() {
        let p = vae_rope_permutation();
        let mut seen = [false; VV_HEAD_DIM];
        for &c in &p {
            assert!(!seen[c]);
            seen[c] = true;
        }
        assert!(seen.iter().all(|s| *s));
        // the decoder rotates (i, i + 24) for i < 24; after permutation those sit at (j, j + 32)
        for i in 0..24 {
            assert_eq!(p[i + 32], p[i] + 24);
        }
        // the 16 pass-through channels 48..64 pair up inside the unrotated halves under identity coefficients
        for j in 24..32 {
            assert!(
                p[j] >= 48 && p[j + 32] >= 56,
                "j={j}: {} {}",
                p[j],
                p[j + 32]
            );
        }
    }

    #[test]
    fn folded_embedding_equals_the_three_stages() {
        let c = LATENT_CHANNELS;
        let pq_w: Vec<f32> = (0..c * c)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.03)
            .collect();
        let pq_b: Vec<f32> = (0..c).map(|i| (i as f32 - 12.0) * 0.01).collect();
        let e_w: Vec<f32> = (0..VV_DIM * c)
            .map(|i| ((i * 13 % 17) as f32 - 8.0) * 0.02)
            .collect();
        let e_b: Vec<f32> = (0..VV_DIM).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
        let mean: Vec<f32> = (0..c).map(|i| (i as f32 - 11.5) * 0.07).collect();
        let std: Vec<f32> = (0..c).map(|i| 0.5 + (i % 7) as f32 * 0.3).collect();
        let (w, b) = fold_embedding(&pq_w, &pq_b, &e_w, &e_b, &mean, &std);
        let z: Vec<f32> = (0..c).map(|i| ((i * 5 % 9) as f32 - 4.0) * 0.3).collect();
        for d in [0, 1, 1023, VV_DIM - 1] {
            let conv: Vec<f64> = (0..c)
                .map(|o| {
                    pq_b[o] as f64
                        + (0..c)
                            .map(|i| {
                                pq_w[o * c + i] as f64
                                    * (z[i] as f64 * std[i] as f64 + mean[i] as f64)
                            })
                            .sum::<f64>()
                })
                .collect();
            let direct =
                e_b[d] as f64 + (0..c).map(|o| e_w[d * c + o] as f64 * conv[o]).sum::<f64>();
            let folded = b[d] as f64
                + (0..c)
                    .map(|i| z[i] as f64 * w[i * VV_DIM + d] as f64)
                    .sum::<f64>();
            assert!(
                (direct - folded).abs() < 1e-4 * direct.abs().max(1.0),
                "d={d}: {direct} vs {folded}"
            );
        }
    }
}

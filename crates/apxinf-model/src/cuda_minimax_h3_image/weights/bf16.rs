//! Device weights of the INT8 ConvRot H3-Image DiT. Mapping from checkpoint keys, one-time packing and upload.
//! No activation computation lives here.
use super::host::Checkpoint;
use crate::cuda_minimax_h3_image::backend::ops::RepackedWeight;
use crate::cuda_minimax_h3_image::backend::{CudaBuffer, CudaContext};
use crate::cuda_minimax_h3_image::config::*;
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::path::Path;

/// One W8A8 linear: INT8 weight (logical `[K, N]`, physical layout owned by `layout`) and
/// one FP32 scale per output channel. ConvRot is already folded into the weight bytes.
pub(in crate::cuda_minimax_h3_image) struct Int8Linear {
    pub weight: Tensor,
    pub layout: RepackedWeight,
    pub channel_scales: Tensor,
    pub k: usize,
    pub n: usize,
}

/// A BF16 or FP32 dense linear whose weight is stored `[in, out]` after load.
pub(in crate::cuda_minimax_h3_image) struct DenseLinear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

/// AdaLN curve projection of one block: F16 weight `[96768, 8]` and BF16 bias.
/// Evaluated once per request step, never per forward (see `model` for the table build).
pub(in crate::cuda_minimax_h3_image) struct AdalnCurve {
    /// Row-major `[expand * modalities * hidden, rank]` as loaded (host copy used to build tables).
    pub weight: Vec<half::f16>,
    pub bias: Vec<half::bf16>,
}

/// The checkpoint's fused `qkv_proj` (`chunk(3)` rows: Q, K, V) is loaded as three linears so every projection
/// output is a contiguous `[tokens, heads, head_dim]` tensor for the Q/K norm+RoPE and attention kernels.
pub(in crate::cuda_minimax_h3_image) struct Attention {
    pub q: Int8Linear,
    pub k: Int8Linear,
    pub v: Int8Linear,
    pub out: Int8Linear,
    /// Per-head RMSNorm weights, BF16 `[128]`, after the RoPE channel permutation.
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

/// SwiGLU feed-forward. The Comfy file stores `fc1` as `[gate; up]` (first half gate),
/// loaded as two linears so each projection is one GEMM and `silu_mul` takes the halves as separate tensors.
pub(in crate::cuda_minimax_h3_image) struct Mlp {
    pub gate: Int8Linear,
    pub up: Int8Linear,
    pub fc2: Int8Linear,
}

pub(in crate::cuda_minimax_h3_image) struct DitBlock {
    pub norm1: Tensor,
    pub norm2: Tensor,
    pub attention: Attention,
    pub mlp: Mlp,
    pub adaln: AdalnCurve,
}

/// BF16 token-refiner block: dense (not INT8) linears, no modulation, no rotary embedding.
pub(in crate::cuda_minimax_h3_image) struct RefinerBlock {
    pub norm1: Tensor,
    pub norm2: Tensor,
    pub q: DenseLinear,
    pub k: DenseLinear,
    pub v: DenseLinear,
    pub out: DenseLinear,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    /// `fc1` rows `[gate; up]`, as in the DiT blocks.
    pub gate: DenseLinear,
    pub up: DenseLinear,
    pub fc2: DenseLinear,
}

/// One PDD output head kept on the host: the step's AdaLN scale and shift are folded into it when the step
/// tables are built, so the device never holds an unmodulated head.
pub(in crate::cuda_minimax_h3_image) struct HostHead {
    /// Row-major `[PATCH_DIM, HIDDEN]`, block 0 plus the step's offset block.
    pub weight: Vec<f32>,
    /// `[PATCH_DIM]`.
    pub bias: Vec<f32>,
}

/// Final layer: modulation curve, norm, and the six per-step FP32 heads.
pub(in crate::cuda_minimax_h3_image) struct FinalLayer {
    pub norm: Tensor,
    pub adaln: AdalnCurve,
    pub heads: Vec<HostHead>,
}

pub(in crate::cuda_minimax_h3_image) struct DitWeights {
    pub patch: DenseLinear,
    pub condition: DenseLinear,
    pub refiner: Vec<RefinerBlock>,
    pub refiner_final_norm: Tensor,
    pub blocks: Vec<DitBlock>,
    pub final_layer: FinalLayer,
    /// `[ADALN_GRID, ADALN_RANK]` FP32 curve table.
    pub adaln_table: Vec<f32>,
    /// Rotary inverse frequencies, FP32 `[ROPE_FREQS]` (checkpoint buffer).
    pub rope_inv_freq: Vec<f32>,
}

pub(in crate::cuda_minimax_h3_image) fn upload(tensor: &Tensor, device: usize) -> Result<Tensor> {
    let bytes = tensor
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("MiniMax-H3-Image upload requires host weights".into()))?;
    upload_bytes(bytes, tensor.shape().clone(), tensor.dtype(), device)
}

pub(in crate::cuda_minimax_h3_image) fn upload_bytes(
    bytes: &[u8],
    shape: Shape,
    dtype: DType,
    device: usize,
) -> Result<Tensor> {
    let buffer = CudaBuffer::alloc(bytes.len(), device).map_err(Error::Cuda)?;
    buffer.copy_from_host(bytes).map_err(Error::Cuda)?;
    buffer.as_tensor(shape, dtype).map_err(Error::Cuda)
}

pub(super) fn f32_vec(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
pub(super) fn f16_vec(bytes: &[u8]) -> Vec<half::f16> {
    bytes
        .chunks_exact(2)
        .map(|c| half::f16::from_le_bytes(c.try_into().unwrap()))
        .collect()
}
pub(super) fn bf16_vec(bytes: &[u8]) -> Vec<half::bf16> {
    bytes
        .chunks_exact(2)
        .map(|c| half::bf16::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

/// Transpose a row-major `[rows, cols]` array of `T` into `[cols, rows]`.
pub(super) fn transpose<T: Copy + Default>(source: &[T], rows: usize, cols: usize) -> Vec<T> {
    let mut out = vec![T::default(); source.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = source[r * cols + c];
        }
    }
    out
}

struct Reader<'a> {
    file: Checkpoint,
    ctx: &'a CudaContext,
}

impl<'a> Reader<'a> {
    fn device(&self) -> usize {
        self.ctx.device_id()
    }

    fn vector_bf16(&mut self, name: &str, len: usize) -> Result<Tensor> {
        let bytes = self.file.read(name, Some(DType::BF16), &[len])?;
        upload_bytes(&bytes, Shape::new(vec![len]), DType::BF16, self.device())
    }

    /// `[out, in]` BF16 -> device `[in, out]`.
    fn dense_bf16(
        &mut self,
        prefix: &str,
        out: usize,
        inp: usize,
        bias: bool,
    ) -> Result<DenseLinear> {
        let raw = self
            .file
            .read(&format!("{prefix}.weight"), Some(DType::BF16), &[out, inp])?;
        let packed = transpose(&bf16_vec(&raw), out, inp);
        let weight = upload(&Tensor::from_bf16(vec![inp, out], &packed)?, self.device())?;
        let bias = if bias {
            Some(self.vector_bf16(&format!("{prefix}.bias"), out)?)
        } else {
            None
        };
        Ok(DenseLinear { weight, bias })
    }

    /// `[parts * out, in]` BF16 without bias -> `parts` device linears `[in, out]`, one per consecutive row block.
    fn dense_bf16_parts(
        &mut self,
        prefix: &str,
        parts: usize,
        out: usize,
        inp: usize,
    ) -> Result<Vec<DenseLinear>> {
        let raw = self.file.read(
            &format!("{prefix}.weight"),
            Some(DType::BF16),
            &[parts * out, inp],
        )?;
        let all = bf16_vec(&raw);
        drop(raw);
        (0..parts)
            .map(|part| {
                let packed = transpose(&all[part * out * inp..(part + 1) * out * inp], out, inp);
                let weight = upload(&Tensor::from_bf16(vec![inp, out], &packed)?, self.device())?;
                Ok(DenseLinear { weight, bias: None })
            })
            .collect()
    }

    /// `[out, in]` FP32 -> device `[in, out]` FP32 plus FP32 bias.
    fn dense_f32(&mut self, prefix: &str, out: usize, inp: usize) -> Result<DenseLinear> {
        let raw = self
            .file
            .read(&format!("{prefix}.weight"), Some(DType::F32), &[out, inp])?;
        let packed = transpose(&f32_vec(&raw), out, inp);
        let weight = upload(&Tensor::from_f32(vec![inp, out], &packed)?, self.device())?;
        let braw = self
            .file
            .read(&format!("{prefix}.bias"), Some(DType::F32), &[out])?;
        let bias = upload(
            &Tensor::from_f32(vec![out], &f32_vec(&braw))?,
            self.device(),
        )?;
        Ok(DenseLinear {
            weight,
            bias: Some(bias),
        })
    }

    /// Read and validate one ConvRot INT8 linear: weight bytes `[n, k]` and FP32 scales `[n]`.
    fn read_int8(&mut self, prefix: &str, n: usize, k: usize) -> Result<(Vec<u8>, Vec<f32>)> {
        let meta = self.file.read_u8(&format!("{prefix}.comfy_quant"))?;
        check_convrot_meta(prefix, &meta)?;
        let weight = self
            .file
            .read(&format!("{prefix}.weight"), Some(DType::I8), &[n, k])?;
        let scales =
            self.file
                .read(&format!("{prefix}.weight_scale"), Some(DType::F32), &[n, 1])?;
        Ok((weight, f32_vec(&scales)))
    }

    fn upload_int8(&self, weight: &[u8], scales: &[f32], n: usize, k: usize) -> Result<Int8Linear> {
        upload_int8(self.ctx, weight, scales, n, k)
    }

    /// W8A8 linear with no row reordering.
    fn int8(&mut self, prefix: &str, n: usize, k: usize) -> Result<Int8Linear> {
        let (weight, scales) = self.read_int8(prefix, n, k)?;
        self.upload_int8(&weight, &scales, n, k)
    }

    /// A `[parts * rows, k]` projection stored as stacked row blocks, returned as `parts` linears. `permuted[i]`
    /// reorders block `i`'s output channels inside every `HEAD_DIM` group (partial-RoPE channel permutation).
    fn int8_parts(
        &mut self,
        prefix: &str,
        rows: usize,
        k: usize,
        permuted: &[bool],
    ) -> Result<Vec<Int8Linear>> {
        let (weight, scales) = self.read_int8(prefix, permuted.len() * rows, k)?;
        let permutation = rope_permutation();
        permuted
            .iter()
            .enumerate()
            .map(|(part, &permute)| {
                let w = &weight[part * rows * k..(part + 1) * rows * k];
                let sc = &scales[part * rows..(part + 1) * rows];
                if !permute {
                    return self.upload_int8(w, sc, rows, k);
                }
                if rows % HEAD_DIM != 0 {
                    return Err(Error::Other(format!(
                        "{prefix}: {rows} rows are not whole heads"
                    )));
                }
                let mut pw = vec![0u8; w.len()];
                let mut ps = vec![0f32; rows];
                for head in (0..rows).step_by(HEAD_DIM) {
                    for (c, &original) in permutation.iter().enumerate() {
                        let (dst, src) = (head + c, head + original);
                        pw[dst * k..(dst + 1) * k].copy_from_slice(&w[src * k..(src + 1) * k]);
                        ps[dst] = sc[src];
                    }
                }
                self.upload_int8(&pw, &ps, rows, k)
            })
            .collect()
    }

    /// `blocks.N.attn.comfy_attention.config` records the attention the checkpoint's author tuned for
    /// (`comfy_kitchen_int8`: INT8 Q/K/V attention). This family computes attention in BF16 (FA2) and does not
    /// reproduce that choice, so the tensor is validated and then deliberately ignored; any other value is a
    /// different checkpoint contract and is rejected.
    fn attention_hint(&mut self, prefix: &str) -> Result<()> {
        let raw = self
            .file
            .read_u8(&format!("{prefix}.attn.comfy_attention.config"))?;
        let text = std::str::from_utf8(&raw).map_err(|_| {
            Error::Other(format!("{prefix}.attn.comfy_attention.config is not UTF-8"))
        })?;
        let parsed: serde_json::Value = serde_json::from_str(text)
            .map_err(|e| Error::Other(format!("{prefix}.attn.comfy_attention.config: {e}")))?;
        if parsed != serde_json::json!([{"attention": "comfy_kitchen_int8"}]) {
            return Err(Error::Other(format!(
                "{prefix}: unsupported attention hint {text}; this family supports the released INT8 ConvRot file"
            )));
        }
        Ok(())
    }

    fn adaln(&mut self, prefix: &str, rows: usize) -> Result<AdalnCurve> {
        let w = self.file.read(
            &format!("{prefix}.linear.weight"),
            Some(DType::F16),
            &[rows, ADALN_RANK],
        )?;
        let b = self
            .file
            .read(&format!("{prefix}.linear.bias"), Some(DType::BF16), &[rows])?;
        Ok(AdalnCurve {
            weight: f16_vec(&w),
            bias: bf16_vec(&b),
        })
    }
}

/// Accept only the quantization recipe this family implements: INT8, tensor-wise scale per output channel, ConvRot 256.
pub(super) fn check_convrot_meta(prefix: &str, meta: &[u8]) -> Result<()> {
    let text = std::str::from_utf8(meta)
        .map_err(|_| Error::Other(format!("{prefix}.comfy_quant is not UTF-8")))?;
    let parsed: serde_json::Value = serde_json::from_str(text)
        .map_err(|e| Error::Other(format!("{prefix}.comfy_quant: {e}")))?;
    if parsed["format"] != "int8_tensorwise"
        || parsed["convrot"] != true
        || parsed["convrot_groupsize"] != CONVROT_GROUP
    {
        return Err(Error::Other(format!(
            "{prefix}: only int8 ConvRot with group size {CONVROT_GROUP} is supported, got {text}"
        )));
    }
    Ok(())
}

/// Upload one `[n, k]` INT8 weight (already ConvRot-rotated, output-major) with its per-channel FP32 scales.
pub(super) fn upload_int8(
    ctx: &CudaContext,
    weight: &[u8],
    scales: &[f32],
    n: usize,
    k: usize,
) -> Result<Int8Linear> {
    let (tensor, layout) = RepackedWeight::upload_row_channel_output_major(ctx, weight, n, k)?;
    let channel_scales = upload(&Tensor::from_f32(vec![n], scales)?, ctx.device_id())?;
    Ok(Int8Linear {
        weight: tensor,
        layout,
        channel_scales,
        k,
        n,
    })
}

/// Channel permutation `P` of one head so that the half-split rotation of the model becomes the
/// half-split rotation over all 128 channels that `rms_norm_apply_precomputed` implements.
/// `P[i] = i` and `P[i + 64] = i + 48` for `i < 48`; `P[48 + j] = 96 + j` and `P[48 + j + 64] = 112 + j`
/// for `j < 16` (`rope_permutation_is_a_bijection_that_keeps_the_pass_through_tail` checks this layout; the DiT parity test
/// against the ComfyUI trace checks the rotation it produces).
/// The permuted channel `c` holds the original channel `P[c]`.
pub(in crate::cuda_minimax_h3_image) fn rope_permutation() -> [usize; HEAD_DIM] {
    let mut p = [0usize; HEAD_DIM];
    for i in 0..48 {
        p[i] = i;
        p[i + 64] = i + 48;
    }
    for j in 0..16 {
        p[48 + j] = 96 + j;
        p[48 + j + 64] = 112 + j;
    }
    p
}

impl DitWeights {
    /// Open the DiT file, validate its schedule metadata, upload every tensor, and fail on any
    /// tensor the loader did not use.
    pub fn load(path: &Path, ctx: &CudaContext) -> Result<Self> {
        let file = Checkpoint::open(path)?;
        let meta = file.metadata("config").ok_or_else(|| {
            Error::Other(format!("{}: missing __metadata__.config", path.display()))
        })?;
        DitMetadata::parse(meta)?;
        let mut r = Reader { file, ctx };

        let patch = r.dense_f32("video_patch_proj", HIDDEN, PATCH_DIM)?;
        let condition = r.dense_bf16("condition_proj", HIDDEN, TEXT_DIM, true)?;
        let mut refiner = Vec::with_capacity(REFINER_BLOCKS);
        for i in 0..REFINER_BLOCKS {
            let p = format!("token_refiner.blocks.{i}");
            let mut qkv = r
                .dense_bf16_parts(&format!("{p}.attn.qkv_proj"), 3, INNER, HIDDEN)?
                .into_iter();
            let mut fc1 = r
                .dense_bf16_parts(&format!("{p}.mlp.fc1"), 2, FFN, HIDDEN)?
                .into_iter();
            refiner.push(RefinerBlock {
                norm1: r.vector_bf16(&format!("{p}.norm1.weight"), HIDDEN)?,
                norm2: r.vector_bf16(&format!("{p}.norm2.weight"), HIDDEN)?,
                q: qkv.next().unwrap(),
                k: qkv.next().unwrap(),
                v: qkv.next().unwrap(),
                out: r.dense_bf16(&format!("{p}.attn.out_proj"), HIDDEN, INNER, false)?,
                q_norm: r.vector_bf16(&format!("{p}.attn.q_norm.weight"), HEAD_DIM)?,
                k_norm: r.vector_bf16(&format!("{p}.attn.k_norm.weight"), HEAD_DIM)?,
                gate: fc1.next().unwrap(),
                up: fc1.next().unwrap(),
                fc2: r.dense_bf16(&format!("{p}.mlp.fc2"), HIDDEN, FFN, false)?,
            });
        }
        let refiner_final_norm = r.vector_bf16("token_refiner.final_norm.weight", HIDDEN)?;

        let permutation = rope_permutation();
        let mut blocks = Vec::with_capacity(BLOCKS);
        for i in 0..BLOCKS {
            let p = format!("blocks.{i}");
            r.attention_hint(&p)?;
            let q_norm = r.permuted_norm(&format!("{p}.attn.q_norm.weight"), &permutation)?;
            let k_norm = r.permuted_norm(&format!("{p}.attn.k_norm.weight"), &permutation)?;
            blocks.push(DitBlock {
                norm1: r.vector_bf16(&format!("{p}.norm1.weight"), HIDDEN)?,
                norm2: r.vector_bf16(&format!("{p}.norm2.weight"), HIDDEN)?,
                attention: {
                    // Q and K rows are channel-permuted for partial RoPE; V rows are left in place.
                    let mut qkv = r
                        .int8_parts(
                            &format!("{p}.attn.qkv_proj"),
                            INNER,
                            HIDDEN,
                            &[true, true, false],
                        )?
                        .into_iter();
                    Attention {
                        q: qkv.next().unwrap(),
                        k: qkv.next().unwrap(),
                        v: qkv.next().unwrap(),
                        out: r.int8(&format!("{p}.attn.out_proj"), HIDDEN, INNER)?,
                        q_norm,
                        k_norm,
                    }
                },
                mlp: {
                    let mut fc1 = r
                        .int8_parts(&format!("{p}.mlp.fc1"), FFN, HIDDEN, &[false, false])?
                        .into_iter();
                    Mlp {
                        gate: fc1.next().unwrap(),
                        up: fc1.next().unwrap(),
                        fc2: r.int8(&format!("{p}.mlp.fc2"), HIDDEN, FFN)?,
                    }
                },
                adaln: r.adaln(&format!("{p}.adaln_proj"), 6 * HIDDEN * MODALITIES)?,
            });
        }

        let final_norm = r.vector_bf16("final_layer.norm.weight", HIDDEN)?;
        let final_adaln = r.adaln("final_layer.adaln_proj", 2 * HIDDEN)?;
        let heads = r.pdd_heads()?;
        let table = r
            .file
            .read("adaln_t_table", Some(DType::F32), &[ADALN_GRID, ADALN_RANK])?;
        let inv_freq = r
            .file
            .read("rope.inv_freq", Some(DType::F32), &[ROPE_FREQS])?;
        r.file.finish()?;
        Ok(Self {
            patch,
            condition,
            refiner,
            refiner_final_norm,
            blocks,
            final_layer: FinalLayer {
                norm: final_norm,
                adaln: final_adaln,
                heads,
            },
            adaln_table: f32_vec(&table),
            rope_inv_freq: f32_vec(&inv_freq),
        })
    }
}

impl Reader<'_> {
    fn permuted_norm(&mut self, name: &str, permutation: &[usize; HEAD_DIM]) -> Result<Tensor> {
        let raw = self.file.read(name, Some(DType::BF16), &[HEAD_DIM])?;
        let source = bf16_vec(&raw);
        let permuted: Vec<half::bf16> = permutation.iter().map(|&c| source[c]).collect();
        upload(
            &Tensor::from_bf16(vec![HEAD_DIM], &permuted)?,
            self.device(),
        )
    }

    /// The Comfy file stores the six step heads as one `[576, 5376]` FP32 matrix: block 0 is head 0 and
    /// block `i >= 1` is an OFFSET, so head `i = block0 + block_i`.
    fn pdd_heads(&mut self) -> Result<Vec<HostHead>> {
        let w = f32_vec(&self.file.read(
            "final_layer.video_out.weight",
            Some(DType::F32),
            &[PDD_STEPS * PATCH_DIM, HIDDEN],
        )?);
        let b = f32_vec(&self.file.read(
            "final_layer.video_out.bias",
            Some(DType::F32),
            &[PDD_STEPS * PATCH_DIM],
        )?);
        let (wb, bb) = (PATCH_DIM * HIDDEN, PATCH_DIM);
        let mut heads = Vec::with_capacity(PDD_STEPS);
        for step in 0..PDD_STEPS {
            let mut weight: Vec<f32> = w[..wb].to_vec();
            let mut bias: Vec<f32> = b[..bb].to_vec();
            if step > 0 {
                for (dst, add) in weight.iter_mut().zip(&w[step * wb..(step + 1) * wb]) {
                    *dst += *add;
                }
                for (dst, add) in bias.iter_mut().zip(&b[step * bb..(step + 1) * bb]) {
                    *dst += *add;
                }
            }
            heads.push(HostHead { weight, bias });
        }
        Ok(heads)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_permutation_is_a_bijection_that_keeps_the_pass_through_tail() {
        let p = rope_permutation();
        let mut seen = [false; HEAD_DIM];
        for &c in &p {
            assert!(!seen[c], "channel {c} used twice");
            seen[c] = true;
        }
        assert!(seen.iter().all(|s| *s));
        // pair partners of the model (i, i + 48) sit half a head apart after permutation
        for i in 0..48 {
            assert_eq!(p[i + 64], p[i] + 48);
        }
        // pass-through channels 96..128 pair with themselves under identity coefficients
        for j in 0..16 {
            assert_eq!(p[48 + j], 96 + j);
            assert_eq!(p[48 + j + 64], 112 + j);
        }
    }

    #[test]
    fn transpose_roundtrip() {
        let m: Vec<u32> = (0..12).collect();
        let t = transpose(&m, 3, 4);
        assert_eq!(t[0..3], [0, 4, 8]);
        assert_eq!(transpose(&t, 4, 3), m);
    }
}

#[cfg(test)]
mod checkpoint_tests {
    use super::*;

    /// Needs the real checkpoint and a CUDA device:
    /// `APXINF_H3_IMAGE_DIT=/path/minimax_h3_image_pruned_int8_convrot.safetensors cargo test --release \
    ///  -p apxinf-model --features cuda --lib -- --ignored loads_the_released_int8_dit`
    #[test]
    #[ignore = "requires the Comfy-Org INT8 ConvRot DiT checkpoint and a CUDA GPU"]
    fn loads_the_released_int8_dit() {
        let path = std::env::var("APXINF_H3_IMAGE_DIT").expect("APXINF_H3_IMAGE_DIT");
        let ctx = CudaContext::new(0).unwrap();
        let started = std::time::Instant::now();
        let weights = DitWeights::load(Path::new(&path), &ctx).unwrap();
        ctx.synchronize().unwrap();
        eprintln!("loaded in {:.1} s", started.elapsed().as_secs_f64());
        assert_eq!(weights.blocks.len(), BLOCKS);
        assert_eq!(weights.refiner.len(), REFINER_BLOCKS);
        assert_eq!(weights.final_layer.heads.len(), PDD_STEPS);
        assert_eq!(weights.adaln_table.len(), ADALN_GRID * ADALN_RANK);
        assert!(weights
            .rope_inv_freq
            .iter()
            .all(|f| f.is_finite() && *f > 0.0));
        let block = &weights.blocks[0];
        for projection in [&block.attention.q, &block.attention.k, &block.attention.v] {
            assert_eq!((projection.k, projection.n), (HIDDEN, INNER));
        }
        assert_eq!((block.mlp.gate.k, block.mlp.gate.n), (HIDDEN, FFN));
        assert_eq!((block.mlp.up.k, block.mlp.up.n), (HIDDEN, FFN));
        assert_eq!((block.mlp.fc2.k, block.mlp.fc2.n), (FFN, HIDDEN));
        assert_eq!(
            block.attention.q.weight.shape().dims(),
            [HIDDEN, INNER],
            "logical [K, N]"
        );
        assert_eq!(block.attention.q.channel_scales.shape().dims(), [INNER]);
        assert_eq!(
            block.adaln.weight.len(),
            6 * HIDDEN * MODALITIES * ADALN_RANK
        );
    }
}

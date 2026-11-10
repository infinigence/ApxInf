//! Qwen3.5 vision tower: 27-block ViT plus the patch merger, executed with
//! the CUDA backend primitives already used by the Qwen3-VL vision path.
//!
//! Structure (verified against the pinned checkpoint's `config.json`, its
//! weight index, and the HF `Qwen3VLVisionModel` reference):
//! - patch_embed: Conv3d [2,16,16] == matmul of `[N, 1536]` by `[1536, 1152]`
//!   plus bias, where each pixel row is `[C=3][T=2][ph=16][pw=16]`.
//! - learned pos_embed `[2304, 1152]` (48x48 grid), bilinearly interpolated
//!   with `align_corners=True` to the image grid, permuted to
//!   spatial-merge-block order, then added.
//! - 27 blocks of: LayerNorm -> fused QKV (16 heads x 72) -> 2D vision RoPE
//!   (theta 10000, FP32 rotation) -> non-causal SDPA -> proj -> residual;
//!   LayerNorm -> fc1 -> tanh-GELU -> fc2 -> residual. All LayerNorms carry
//!   bias, eps 1e-6.
//! - merger: LayerNorm(1152) -> reshape `[N,1152] -> [N/4,4608]` (rows are
//!   contiguous, so this is a pure reshape) -> fc1 -> **exact erf GELU**
//!   (`nn.GELU()`, not the tanh form the block MLPs use) -> fc2 to 5120.
//! - `deepstack_visual_indexes` is empty for this checkpoint: no deepstack.
//!
//! The tower weighs ~880 MiB in BF16 and is only loaded when
//! `APXINF_ENABLE_MULTIMODAL=1`, so the text-only configuration is untouched.

#[cfg(feature = "cuda")]
use apxinf_core::{Backend, Shape, Tensor};
#[cfg(feature = "cuda")]
use apxinf_cuda::CudaBackend;
#[cfg(feature = "cuda")]
use half::bf16;

#[cfg(feature = "cuda")]
use super::loader::Qwen35CheckpointInventory;

pub const VISION_DEPTH: usize = 27;
pub const VISION_HIDDEN: usize = 1152;
pub const VISION_HEADS: usize = 16;
pub const VISION_HEAD_DIM: usize = 72;
pub const VISION_INTERMEDIATE: usize = 4304;
/// in_channels(3) * temporal_patch_size(2) * patch_size(16)^2.
pub const VISION_PATCH_DIM: usize = 1536;
pub const VISION_MERGE: usize = 2;
/// 48 x 48 learned position grid.
pub const VISION_NUM_POS: usize = 2304;
pub const VISION_POS_GRID_SIDE: usize = 48;
pub const VISION_OUT_HIDDEN: usize = 5120;
#[cfg(feature = "cuda")]
const VISION_EPS: f32 = 1e-6;
#[cfg(feature = "cuda")]
const VISION_ROPE_THETA: f32 = 10_000.0;
/// hidden * merge^2, the merger's working width.
pub const VISION_MERGED_DIM: usize = VISION_HIDDEN * VISION_MERGE * VISION_MERGE;

#[cfg(feature = "cuda")]
struct VisionBlock {
    norm1_w: Tensor,
    norm1_b: Tensor,
    /// `[1152, 3456]`, transposed for row-major matmul.
    qkv_w: Tensor,
    qkv_b: Tensor,
    /// `[1152, 1152]`, transposed.
    proj_w: Tensor,
    proj_b: Tensor,
    norm2_w: Tensor,
    norm2_b: Tensor,
    /// `[1152, 4304]`, transposed.
    fc1_w: Tensor,
    fc1_b: Tensor,
    /// `[4304, 1152]`, transposed.
    fc2_w: Tensor,
    fc2_b: Tensor,
}

#[cfg(feature = "cuda")]
pub struct Qwen35VisionTower {
    /// `[1536, 1152]`, transposed patch projection.
    patch_embed_w: Tensor,
    patch_embed_b: Tensor,
    /// Host copy of the `[2304, 1152]` learned position table in f32; the
    /// per-image bilinear interpolation runs on host (once per request).
    pos_embed_f32: Vec<f32>,
    blocks: Vec<VisionBlock>,
    merger_norm_w: Tensor,
    merger_norm_b: Tensor,
    /// `[4608, 4608]`, transposed.
    merger_fc1_w: Tensor,
    merger_fc1_b: Tensor,
    /// `[4608, 5120]`, transposed.
    merger_fc2_w: Tensor,
    merger_fc2_b: Tensor,
}

/// Per-stage snapshots for oracle comparison in tests.
#[cfg(feature = "cuda")]
pub struct VisionProbe {
    pub post_patch_embed: Vec<f32>,
    pub post_pos_embed: Vec<f32>,
    pub block_outputs: Vec<(usize, Vec<f32>)>,
    pub merged: Vec<f32>,
}

/// Resident BF16 bytes of the tower (weights only), for VRAM accounting.
pub fn resident_weight_bytes() -> usize {
    let block = 2 * VISION_HIDDEN                       // norm1 w+b
        + VISION_HIDDEN * 3 * VISION_HIDDEN + 3 * VISION_HIDDEN // qkv
        + VISION_HIDDEN * VISION_HIDDEN + VISION_HIDDEN // proj
        + 2 * VISION_HIDDEN                             // norm2
        + VISION_HIDDEN * VISION_INTERMEDIATE + VISION_INTERMEDIATE
        + VISION_INTERMEDIATE * VISION_HIDDEN + VISION_HIDDEN;
    let merger = 2 * VISION_HIDDEN
        + VISION_MERGED_DIM * VISION_MERGED_DIM
        + VISION_MERGED_DIM
        + VISION_MERGED_DIM * VISION_OUT_HIDDEN
        + VISION_OUT_HIDDEN;
    let patch = VISION_PATCH_DIM * VISION_HIDDEN + VISION_HIDDEN;
    2 * (VISION_DEPTH * block + merger + patch)
}

#[cfg(feature = "cuda")]
impl Qwen35VisionTower {
    pub fn from_inventory(
        backend: &CudaBackend,
        inventory: &Qwen35CheckpointInventory,
    ) -> Result<Self, String> {
        let device = backend.device_id();
        let read = |name: &str, shape: &[usize]| -> Result<Vec<bf16>, String> {
            inventory
                .read_bf16_tensor_payload(name, shape)
                .map(|payload| payload.values)
                .map_err(|error| format!("vision tensor {name}: {error}"))
        };
        let upload = |values: &[bf16], shape: Vec<usize>| -> Result<Tensor, String> {
            let host = Tensor::from_bf16(Shape::new(shape), values)
                .map_err(|error| format!("vision tensor create: {error}"))?;
            apxinf_cuda::transfers::to_cuda(&host, device)
                .map_err(|error| format!("vision tensor upload: {error}"))
        };
        // HF Linear weights are `[out, in]`; transpose to `[in, out]` once on
        // host for the row-major cuBLAS matmul.
        let upload_linear =
            |name: &str, out_features: usize, in_features: usize| -> Result<Tensor, String> {
                let values = read(name, &[out_features, in_features])?;
                let mut transposed = vec![bf16::from_f32(0.0); values.len()];
                for row in 0..out_features {
                    for col in 0..in_features {
                        transposed[col * out_features + row] = values[row * in_features + col];
                    }
                }
                upload(&transposed, vec![in_features, out_features])
            };

        let mut blocks = Vec::with_capacity(VISION_DEPTH);
        for index in 0..VISION_DEPTH {
            let prefix = format!("model.visual.blocks.{index}");
            blocks.push(VisionBlock {
                norm1_w: upload(
                    &read(&format!("{prefix}.norm1.weight"), &[VISION_HIDDEN])?,
                    vec![VISION_HIDDEN],
                )?,
                norm1_b: upload(
                    &read(&format!("{prefix}.norm1.bias"), &[VISION_HIDDEN])?,
                    vec![VISION_HIDDEN],
                )?,
                qkv_w: upload_linear(
                    &format!("{prefix}.attn.qkv.weight"),
                    3 * VISION_HIDDEN,
                    VISION_HIDDEN,
                )?,
                qkv_b: upload(
                    &read(&format!("{prefix}.attn.qkv.bias"), &[3 * VISION_HIDDEN])?,
                    vec![3 * VISION_HIDDEN],
                )?,
                proj_w: upload_linear(
                    &format!("{prefix}.attn.proj.weight"),
                    VISION_HIDDEN,
                    VISION_HIDDEN,
                )?,
                proj_b: upload(
                    &read(&format!("{prefix}.attn.proj.bias"), &[VISION_HIDDEN])?,
                    vec![VISION_HIDDEN],
                )?,
                norm2_w: upload(
                    &read(&format!("{prefix}.norm2.weight"), &[VISION_HIDDEN])?,
                    vec![VISION_HIDDEN],
                )?,
                norm2_b: upload(
                    &read(&format!("{prefix}.norm2.bias"), &[VISION_HIDDEN])?,
                    vec![VISION_HIDDEN],
                )?,
                fc1_w: upload_linear(
                    &format!("{prefix}.mlp.linear_fc1.weight"),
                    VISION_INTERMEDIATE,
                    VISION_HIDDEN,
                )?,
                fc1_b: upload(
                    &read(
                        &format!("{prefix}.mlp.linear_fc1.bias"),
                        &[VISION_INTERMEDIATE],
                    )?,
                    vec![VISION_INTERMEDIATE],
                )?,
                fc2_w: upload_linear(
                    &format!("{prefix}.mlp.linear_fc2.weight"),
                    VISION_HIDDEN,
                    VISION_INTERMEDIATE,
                )?,
                fc2_b: upload(
                    &read(&format!("{prefix}.mlp.linear_fc2.bias"), &[VISION_HIDDEN])?,
                    vec![VISION_HIDDEN],
                )?,
            });
        }

        // patch_embed: Conv3d weight [1152, 3, 2, 16, 16] == [1152, 1536].
        let patch_values = read(
            "model.visual.patch_embed.proj.weight",
            &[VISION_HIDDEN, 3, 2, 16, 16],
        )?;
        let mut patch_transposed = vec![bf16::from_f32(0.0); patch_values.len()];
        for row in 0..VISION_HIDDEN {
            for col in 0..VISION_PATCH_DIM {
                patch_transposed[col * VISION_HIDDEN + row] =
                    patch_values[row * VISION_PATCH_DIM + col];
            }
        }
        let patch_embed_w = upload(&patch_transposed, vec![VISION_PATCH_DIM, VISION_HIDDEN])?;
        let patch_embed_b = upload(
            &read("model.visual.patch_embed.proj.bias", &[VISION_HIDDEN])?,
            vec![VISION_HIDDEN],
        )?;

        let pos_values = read(
            "model.visual.pos_embed.weight",
            &[VISION_NUM_POS, VISION_HIDDEN],
        )?;
        let pos_embed_f32: Vec<f32> = pos_values.iter().map(|v| v.to_f32()).collect();

        Ok(Self {
            patch_embed_w,
            patch_embed_b,
            pos_embed_f32,
            blocks,
            merger_norm_w: upload(
                &read("model.visual.merger.norm.weight", &[VISION_HIDDEN])?,
                vec![VISION_HIDDEN],
            )?,
            merger_norm_b: upload(
                &read("model.visual.merger.norm.bias", &[VISION_HIDDEN])?,
                vec![VISION_HIDDEN],
            )?,
            merger_fc1_w: upload_linear(
                "model.visual.merger.linear_fc1.weight",
                VISION_MERGED_DIM,
                VISION_MERGED_DIM,
            )?,
            merger_fc1_b: upload(
                &read("model.visual.merger.linear_fc1.bias", &[VISION_MERGED_DIM])?,
                vec![VISION_MERGED_DIM],
            )?,
            merger_fc2_w: upload_linear(
                "model.visual.merger.linear_fc2.weight",
                VISION_OUT_HIDDEN,
                VISION_MERGED_DIM,
            )?,
            merger_fc2_b: upload(
                &read("model.visual.merger.linear_fc2.bias", &[VISION_OUT_HIDDEN])?,
                vec![VISION_OUT_HIDDEN],
            )?,
        })
    }

    /// Run the tower on preprocessed pixel values. Returns the merged image
    /// embedding `[t*h*w/4, 5120]` as a BF16 device tensor.
    pub fn forward(
        &self,
        backend: &CudaBackend,
        pixel_values: &[bf16],
        grid: [u32; 3],
    ) -> Result<Tensor, String> {
        self.forward_impl(backend, pixel_values, grid, None)
            .map(|(merged, _)| merged)
    }

    /// Forward with per-stage f32 snapshots for golden comparison.
    pub fn forward_probe(
        &self,
        backend: &CudaBackend,
        pixel_values: &[bf16],
        grid: [u32; 3],
        probe_blocks: &[usize],
    ) -> Result<VisionProbe, String> {
        self.forward_impl(backend, pixel_values, grid, Some(probe_blocks))
            .map(|(_, probe)| probe.expect("probe requested"))
    }

    fn forward_impl(
        &self,
        backend: &CudaBackend,
        pixel_values: &[bf16],
        grid: [u32; 3],
        probe_blocks: Option<&[usize]>,
    ) -> Result<(Tensor, Option<VisionProbe>), String> {
        let (t, h, w) = (grid[0] as usize, grid[1] as usize, grid[2] as usize);
        let n_patches = t * h * w;
        if n_patches == 0
            || h % VISION_MERGE != 0
            || w % VISION_MERGE != 0
            || pixel_values.len() != n_patches * VISION_PATCH_DIM
        {
            return Err(format!(
                "vision grid {grid:?} does not match {} pixel values",
                pixel_values.len()
            ));
        }
        let stage = |label: &str, error: apxinf_core::Error| format!("vision {label}: {error}");

        let pixels_host =
            Tensor::from_bf16(Shape::new(vec![n_patches, VISION_PATCH_DIM]), pixel_values)
                .map_err(|error| format!("vision pixel tensor: {error}"))?;
        let pixels = backend
            .to_device(&pixels_host)
            .map_err(|error| stage("pixel upload", error))?;

        // Patch embedding.
        let mut x = backend
            .matmul(&pixels, &self.patch_embed_w)
            .map_err(|error| stage("patch_embed matmul", error))?;
        x = backend
            .add_bias(&x, &self.patch_embed_b)
            .map_err(|error| stage("patch_embed bias", error))?;

        let mut probe = probe_blocks.map(|_| VisionProbe {
            post_patch_embed: Vec::new(),
            post_pos_embed: Vec::new(),
            block_outputs: Vec::new(),
            merged: Vec::new(),
        });
        if let Some(probe) = probe.as_mut() {
            probe.post_patch_embed = download_f32(backend, &x)?;
        }

        // Interpolated positional embedding (host f32, once per image).
        let pos_host = self.interpolate_pos_embed(t, h, w);
        let pos_bf16: Vec<bf16> = pos_host.iter().map(|&v| bf16::from_f32(v)).collect();
        let pos_tensor_host =
            Tensor::from_bf16(Shape::new(vec![n_patches, VISION_HIDDEN]), &pos_bf16)
                .map_err(|error| format!("vision pos tensor: {error}"))?;
        let pos_tensor = backend
            .to_device(&pos_tensor_host)
            .map_err(|error| stage("pos upload", error))?;
        x = backend
            .add(&x, &pos_tensor)
            .map_err(|error| stage("pos add", error))?;
        if let Some(probe) = probe.as_mut() {
            probe.post_pos_embed = download_f32(backend, &x)?;
        }

        // 2D RoPE position ids in spatial-merge-block order.
        let pos_ids = vision_pos_ids(t, h, w);

        for (index, block) in self.blocks.iter().enumerate() {
            let normed = backend
                .layer_norm(&x, &block.norm1_w, &block.norm1_b, VISION_EPS)
                .map_err(|error| stage("norm1", error))?;
            let qkv = backend
                .matmul(&normed, &block.qkv_w)
                .map_err(|error| stage("qkv matmul", error))?;
            let qkv = backend
                .add_bias(&qkv, &block.qkv_b)
                .map_err(|error| stage("qkv bias", error))?;
            let q = slice_columns(backend, &qkv, 0, VISION_HIDDEN, n_patches)?;
            let k = slice_columns(backend, &qkv, VISION_HIDDEN, VISION_HIDDEN, n_patches)?;
            let v = slice_columns(backend, &qkv, 2 * VISION_HIDDEN, VISION_HIDDEN, n_patches)?;
            let q = backend
                .rope_vision_2d(
                    &q,
                    VISION_HEADS,
                    VISION_HEAD_DIM,
                    VISION_ROPE_THETA,
                    &pos_ids,
                )
                .map_err(|error| stage("rope q", error))?;
            let k = backend
                .rope_vision_2d(
                    &k,
                    VISION_HEADS,
                    VISION_HEAD_DIM,
                    VISION_ROPE_THETA,
                    &pos_ids,
                )
                .map_err(|error| stage("rope k", error))?;
            let attn = backend
                .vision_sdpa(&q, &k, &v, n_patches, VISION_HEADS, VISION_HEAD_DIM)
                .map_err(|error| stage("sdpa", error))?;
            let attn = backend
                .matmul(&attn, &block.proj_w)
                .map_err(|error| stage("proj matmul", error))?;
            let attn = backend
                .add_bias(&attn, &block.proj_b)
                .map_err(|error| stage("proj bias", error))?;
            x = backend
                .add(&x, &attn)
                .map_err(|error| stage("attn residual", error))?;

            let normed = backend
                .layer_norm(&x, &block.norm2_w, &block.norm2_b, VISION_EPS)
                .map_err(|error| stage("norm2", error))?;
            let hidden = backend
                .matmul(&normed, &block.fc1_w)
                .map_err(|error| stage("fc1 matmul", error))?;
            let hidden = backend
                .add_bias(&hidden, &block.fc1_b)
                .map_err(|error| stage("fc1 bias", error))?;
            let hidden = backend
                .gelu_tanh(&hidden)
                .map_err(|error| stage("mlp gelu", error))?;
            let hidden = backend
                .matmul(&hidden, &block.fc2_w)
                .map_err(|error| stage("fc2 matmul", error))?;
            let hidden = backend
                .add_bias(&hidden, &block.fc2_b)
                .map_err(|error| stage("fc2 bias", error))?;
            x = backend
                .add(&x, &hidden)
                .map_err(|error| stage("mlp residual", error))?;

            if let Some(probe) = probe.as_mut() {
                if probe_blocks.is_some_and(|blocks| blocks.contains(&index)) {
                    probe
                        .block_outputs
                        .push((index, download_f32(backend, &x)?));
                }
            }
        }

        // Merger. Rows are contiguous, so the [N,1152] -> [N/4,4608] merge is
        // a pure reshape of the same storage.
        let normed = backend
            .layer_norm(&x, &self.merger_norm_w, &self.merger_norm_b, VISION_EPS)
            .map_err(|error| stage("merger norm", error))?;
        let merged_rows = n_patches / (VISION_MERGE * VISION_MERGE);
        let reshaped = normed
            .reshape(vec![merged_rows, VISION_MERGED_DIM])
            .map_err(|error| format!("merger reshape: {error}"))?;
        let hidden = backend
            .matmul(&reshaped, &self.merger_fc1_w)
            .map_err(|error| stage("merger fc1", error))?;
        let hidden = backend
            .add_bias(&hidden, &self.merger_fc1_b)
            .map_err(|error| stage("merger fc1 bias", error))?;
        // The merger uses PyTorch's default nn.GELU() — the exact erf form.
        let hidden = apxinf_cuda::kernels::activation::gelu_erf(backend.context(), &hidden)
            .map_err(|error| stage("merger gelu", error))?;
        let out = backend
            .matmul(&hidden, &self.merger_fc2_w)
            .map_err(|error| stage("merger fc2", error))?;
        let out = backend
            .add_bias(&out, &self.merger_fc2_b)
            .map_err(|error| stage("merger fc2 bias", error))?;

        if let Some(probe) = probe.as_mut() {
            probe.merged = download_f32(backend, &out)?;
        }
        Ok((out, probe))
    }

    /// Bilinear interpolation (align_corners=True) of the 48x48 learned grid
    /// to (h, w), emitted in spatial-merge-block order. Matches HF's
    /// `get_vision_interpolation_indices_and_weights` + embedding sum.
    fn interpolate_pos_embed(&self, t: usize, h: usize, w: usize) -> Vec<f32> {
        let side = VISION_POS_GRID_SIDE;
        let hidden = VISION_HIDDEN;
        let merge = VISION_MERGE;
        let table = &self.pos_embed_f32;

        let axis = |index: usize, size: usize| -> (usize, usize, f32) {
            let denominator = (size - 1).max(1) as f32;
            let src = index as f32 * (side - 1) as f32 / denominator;
            let floor = src.floor();
            let tap0 = (floor as isize).clamp(0, (side - 1) as isize) as usize;
            let tap1 = ((floor as isize) + 1).clamp(0, (side - 1) as isize) as usize;
            // Hat-kernel weights: w0 = 1 - |src - floor|, w1 = 1 - |src - floor - 1|.
            let d0 = (src - floor).abs();
            (tap0, tap1, d0)
        };

        let merged_h = h / merge;
        let merged_w = w / merge;
        let mut out = vec![0.0f32; t * h * w * hidden];
        let mut row_index = 0usize;
        for _t in 0..t {
            for bh in 0..merged_h {
                for bw in 0..merged_w {
                    for ih in 0..merge {
                        for iw in 0..merge {
                            let row = bh * merge + ih;
                            let col = bw * merge + iw;
                            let (h0, h1, dh) = axis(row, h);
                            let (w0, w1, dw) = axis(col, w);
                            let w00 = (1.0 - dh) * (1.0 - dw);
                            let w01 = (1.0 - dh) * dw;
                            let w10 = dh * (1.0 - dw);
                            let w11 = dh * dw;
                            let dst = row_index * hidden;
                            let s00 = (h0 * side + w0) * hidden;
                            let s01 = (h0 * side + w1) * hidden;
                            let s10 = (h1 * side + w0) * hidden;
                            let s11 = (h1 * side + w1) * hidden;
                            for c in 0..hidden {
                                out[dst + c] = table[s00 + c] * w00
                                    + table[s01 + c] * w01
                                    + table[s10 + c] * w10
                                    + table[s11 + c] * w11;
                            }
                            row_index += 1;
                        }
                    }
                }
            }
        }
        out
    }
}

/// (h, w) RoPE position ids in spatial-merge-block order, flattened as
/// `[h0, w0, h1, w1, ...]`. Matches HF's `get_vision_position_ids`.
fn vision_pos_ids(t: usize, h: usize, w: usize) -> Vec<u32> {
    let merge = VISION_MERGE;
    let merged_h = h / merge;
    let merged_w = w / merge;
    let mut ids = Vec::with_capacity(t * h * w * 2);
    for _t in 0..t {
        for bh in 0..merged_h {
            for bw in 0..merged_w {
                for ih in 0..merge {
                    for iw in 0..merge {
                        ids.push((bh * merge + ih) as u32);
                        ids.push((bw * merge + iw) as u32);
                    }
                }
            }
        }
    }
    ids
}

/// Extract contiguous columns `[col_start, col_start+width)` of a `[rows, 3*width]`
/// BF16 tensor and reshape to `[rows, heads, head_dim]`. Host round-trip: the
/// vision tower runs once per image request, so the ~5 MB transfer is not on
/// any steady-state path.
#[cfg(feature = "cuda")]
fn slice_columns(
    backend: &CudaBackend,
    qkv: &Tensor,
    col_start: usize,
    width: usize,
    rows: usize,
) -> Result<Tensor, String> {
    let cpu = backend
        .to_cpu(qkv)
        .map_err(|error| format!("vision qkv download: {error}"))?;
    let data = cpu
        .as_bf16()
        .map_err(|error| format!("vision qkv dtype: {error}"))?;
    let total_cols = qkv.shape().dims()[1];
    let mut out = vec![bf16::from_f32(0.0); rows * width];
    for row in 0..rows {
        let src = row * total_cols + col_start;
        out[row * width..(row + 1) * width].copy_from_slice(&data[src..src + width]);
    }
    let host = Tensor::from_bf16(Shape::new(vec![rows, VISION_HEADS, VISION_HEAD_DIM]), &out)
        .map_err(|error| format!("vision qkv slice tensor: {error}"))?;
    backend
        .to_device(&host)
        .map_err(|error| format!("vision qkv slice upload: {error}"))
}

#[cfg(feature = "cuda")]
fn download_f32(backend: &CudaBackend, tensor: &Tensor) -> Result<Vec<f32>, String> {
    let cpu = backend
        .to_cpu(tensor)
        .map_err(|error| format!("vision probe download: {error}"))?;
    cpu.to_f32_vec()
        .map_err(|error| format!("vision probe convert: {error}"))
}

#[cfg(all(test, feature = "cuda"))]
mod oracle_tests {
    use super::*;

    struct Golden {
        values: Vec<f32>,
        rows: usize,
        cols: usize,
    }

    fn load_golden(directory: &std::path::Path, name: &str, rows: usize, cols: usize) -> Golden {
        let path = directory.join(format!("{name}.f32.bin"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        assert_eq!(bytes.len(), rows * cols * 4, "{name} golden size");
        let values = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        Golden { values, rows, cols }
    }

    fn compare(label: &str, actual: &[f32], golden: &Golden) -> (f32, f32) {
        assert_eq!(actual.len(), golden.rows * golden.cols, "{label} shape");
        let mut max_abs_diff = 0.0f32;
        let mut sum_abs_diff = 0.0f64;
        let mut golden_max_abs = 0.0f32;
        for (a, g) in actual.iter().zip(&golden.values) {
            let diff = (a - g).abs();
            max_abs_diff = max_abs_diff.max(diff);
            sum_abs_diff += f64::from(diff);
            golden_max_abs = golden_max_abs.max(g.abs());
        }
        let mean_abs_diff = (sum_abs_diff / actual.len() as f64) as f32;
        eprintln!(
            "{label}: max_abs_diff={max_abs_diff:.6} mean_abs_diff={mean_abs_diff:.6} \
             golden_max_abs={golden_max_abs:.3}"
        );
        (max_abs_diff, mean_abs_diff)
    }

    /// Stage-by-stage comparison of the CUDA vision tower against the
    /// offline FP32 Transformers oracle on the deterministic 448x448 probe.
    /// BF16-vs-FP32 drift bounds were set from the first measured run and
    /// then tightened to sit just above it.
    #[test]
    #[ignore = "requires GPU, the pinned checkpoint, and the vision oracle directory"]
    fn vision_tower_matches_oracle_stage_by_stage() {
        use apxinf_loader::QWEN35_MODEL_REVISION;

        let oracle_dir = std::path::PathBuf::from(
            std::env::var("APXINF_VISION_ORACLE_DIR")
                .unwrap_or_else(|_| "/tmp/apxinf-vision-oracle".to_string()),
        );
        let checkpoint = std::env::var_os("APXINF_QWEN35_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from("/mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4")
            });
        let device = std::env::var("APXINF_CUDA_DEVICE")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(0);

        let pixel_golden = load_golden(&oracle_dir, "pixel_values", 784, VISION_PATCH_DIM);
        let pixels_bf16: Vec<bf16> = pixel_golden
            .values
            .iter()
            .map(|&v| bf16::from_f32(v))
            .collect();

        let inventory = super::super::loader::Qwen35CheckpointInventory::from_checkpoint_dir(
            &checkpoint,
            QWEN35_MODEL_REVISION,
        )
        .expect("checkpoint inventory");
        let backend = CudaBackend::new(device).expect("CUDA backend");
        let start = std::time::Instant::now();
        let tower = Qwen35VisionTower::from_inventory(&backend, &inventory).expect("vision tower");
        eprintln!(
            "vision tower loaded in {:.1}s",
            start.elapsed().as_secs_f32()
        );

        let start = std::time::Instant::now();
        let probe = tower
            .forward_probe(&backend, &pixels_bf16, [1, 28, 28], &[0, 13, 26])
            .expect("vision forward");
        eprintln!("vision forward in {:.2}s", start.elapsed().as_secs_f32());

        let block00 = load_golden(&oracle_dir, "block00_out", 784, VISION_HIDDEN);
        let block13 = load_golden(&oracle_dir, "block13_out", 784, VISION_HIDDEN);
        let block26 = load_golden(&oracle_dir, "block26_out", 784, VISION_HIDDEN);
        let merged = load_golden(&oracle_dir, "merged", 196, VISION_OUT_HIDDEN);

        let outputs: std::collections::HashMap<usize, &Vec<f32>> = probe
            .block_outputs
            .iter()
            .map(|(index, values)| (*index, values))
            .collect();

        // Measured BF16-vs-FP32 drift (2026-08-26 first run on GPU0):
        //   block00 max 0.069 (golden max 5.6)   -> ~1.2% on the largest values
        //   block13 max 4.87  (golden max 372.7) -> ~1.3%
        //   block26 max 352   (golden max 4477)  -> ~7.9%, mean_abs_diff 0.195
        // The drift concentrates on the huge-magnitude outlier channels and
        // compounds through 27 blocks; the merger LayerNorm renormalizes
        // before the LLM consumes anything, so the functional bound is the
        // merged stage plus its cosine, with per-stage bounds set ~2x above
        // the measured run to catch regressions without flaking.
        let (b0_max, _) = compare("block00", outputs[&0], &block00);
        let (b13_max, _) = compare("block13", outputs[&13], &block13);
        let (b26_max, _) = compare("block26", outputs[&26], &block26);
        let (merged_max, merged_mean) = compare("merged", &probe.merged, &merged);
        // Measured merged drift: max 4.21 (golden max_abs 142.8, ~2.9%),
        // mean 0.0078 (golden mean_abs 0.278, ~2.8%) — the expected BF16
        // compounding level, bounded at ~2x measured.
        assert!(b0_max <= 0.15, "block00 drift {b0_max} too large");
        assert!(b13_max <= 10.0, "block13 drift {b13_max} too large");
        assert!(b26_max <= 700.0, "block26 drift {b26_max} too large");
        assert!(merged_max <= 8.0, "merged drift {merged_max} too large");
        assert!(
            merged_mean <= 0.02,
            "merged mean drift {merged_mean} too large"
        );

        // Cosine similarity of the merged embedding is the functional
        // criterion: the LLM consumes directions in embedding space.
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_g = 0.0f64;
        for (a, g) in probe.merged.iter().zip(&merged.values) {
            dot += f64::from(*a) * f64::from(*g);
            norm_a += f64::from(*a) * f64::from(*a);
            norm_g += f64::from(*g) * f64::from(*g);
        }
        let cosine = dot / (norm_a.sqrt() * norm_g.sqrt());
        eprintln!("merged cosine similarity: {cosine:.8}");
        assert!(cosine >= 0.999, "merged cosine {cosine} too low");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pos_ids_follow_spatial_merge_block_order() {
        // 4x4 grid, merge 2: blocks are (rows 0-1, cols 0-1), (rows 0-1,
        // cols 2-3), (rows 2-3, cols 0-1), (rows 2-3, cols 2-3).
        let ids = vision_pos_ids(1, 4, 4);
        let pairs: Vec<(u32, u32)> = ids.chunks(2).map(|c| (c[0], c[1])).collect();
        assert_eq!(
            pairs,
            vec![
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 0),
                (2, 1),
                (3, 0),
                (3, 1),
                (2, 2),
                (2, 3),
                (3, 2),
                (3, 3),
            ]
        );
    }

    #[test]
    fn resident_weight_estimate_matches_checkpoint_arithmetic() {
        // 27 blocks + merger + patch embed, BF16: ~915 MB. Guards against a
        // silently wrong constant when accounting VRAM headroom.
        let bytes = resident_weight_bytes();
        assert!(
            (850 * 1024 * 1024..1000 * 1024 * 1024).contains(&bytes),
            "vision tower resident bytes {bytes} outside expected range"
        );
    }
}

//! The Clef (Qwen3.5) vision tower and multimodal position preparation.
//!
//! Numeric order follows the MLX-VLM modules the reference checkpoint
//! (`clef_mlx.py`) executes: `models/qwen3_vl/vision.py` for the tower and
//! `models/qwen3_5/language.py::get_rope_index` for mRoPE positions. The patch
//! embed is a plain matmul because the checkpoint's Conv3d weight flattens to
//! `[out, channel, temporal, row, column]`, the processor's patch order; no
//! conv3d is needed. Everything runs on the GPU stream in every composition:
//! the ANE column split only replaces text MLP columns.
use super::config::Config;
use super::preprocess::{PATCH_WIDTH, SPATIAL_MERGE_SIZE};
use super::weights::Take;
use apxinf_core::{Error, Result};
use apxinf_mlx::fusions::MropeHalfSplitRope;
use apxinf_mlx::{Array, MlxDType, Stream};
use std::path::Path;

/// LayerNorm epsilon of the vision tower and its merger.
const EPS: f32 = 1e-6;
const VISION_THETA: f32 = 10_000.0;
const MERGE: usize = SPATIAL_MERGE_SIZE;

/// Python scalars combine with arrays in the array's dtype.
fn scalar(like: &Array, value: f64) -> Result<Array> {
    Array::scalar(like.stream(), value as f32, like.dtype())
}

/// `nn.Linear` without bias: `x @ W.T`.
fn project(x: &Array, weight_transposed: &Array) -> Result<Array> {
    x.matmul(weight_transposed)
}

/// `nn.Linear` with bias.
fn linear(x: &Array, weight_transposed: &Array, bias: &Array) -> Result<Array> {
    project(x, weight_transposed)?.add(bias)
}

fn layer_norm(x: &Array, weight: &Array, bias: &Array) -> Result<Array> {
    x.fast_layer_norm(Some(weight), Some(bias), EPS)
}

/// `nn.GELU(approx="tanh")`, in the BF16 op order MLX's elementwise chain uses:
/// `0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x**3)))`.
fn gelu_tanh(x: &Array) -> Result<Array> {
    let cube = x.mul(x)?.mul(x)?;
    let inner = x.add(&scalar(x, 0.044715)?.mul(&cube)?)?;
    let tanh = inner.mul(&scalar(x, 0.797_884_560_802_865_4)?)?.tanh()?;
    let gate = scalar(x, 1.0)?.add(&tanh)?;
    x.mul(&scalar(x, 0.5)?)?.mul(&gate)
}

/// `nn.GELU()` (exact, `erf`), in the order MLX uses:
/// `x * (1 + erf(x / sqrt(2))) / 2`.
fn gelu_exact(x: &Array) -> Result<Array> {
    let inner = x.div(&scalar(x, std::f64::consts::SQRT_2)?)?.erf()?;
    let gate = scalar(x, 1.0)?.add(&inner)?;
    x.mul(&gate)?.div(&scalar(x, 2.0)?)
}

/// `APXINF_CLEF_DUMP_STAGES=<prefix>` writes one tower stage as
/// `<prefix>.<name>.f32` (native-endian FP32) plus a `.json` shape record, for
/// the stage-by-stage parity table. Off by default and never on the timed path
/// unless the harness asks for it.
fn dump_stage(name: &str, array: &Array) -> Result<()> {
    let Some(prefix) = std::env::var_os("APXINF_CLEF_DUMP_STAGES") else {
        return Ok(());
    };
    let prefix = std::path::PathBuf::from(prefix);
    array.eval()?;
    let shape = array.shape().to_vec();
    let values = array.to_f32_vec()?;
    let path = std::path::PathBuf::from(format!("{}.{name}.f32", prefix.display()));
    let mut bytes = Vec::with_capacity(values.len() * 4);
    for value in &values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    std::fs::write(&path, bytes)
        .map_err(|e| Error::Other(format!("dump {}: {e}", path.display())))?;
    let meta = format!("{{\"shape\": {shape:?}, \"stage\": \"{name}\"}}");
    std::fs::write(
        std::path::PathBuf::from(format!("{}.{name}.json", prefix.display())),
        meta,
    )
    .map_err(|e| Error::Other(format!("dump {name} meta: {e}")))?;
    Ok(())
}

/// One vision block: bias LayerNorms, axial-2D-RoPE self attention over the
/// image's own patches, then a GELU-tanh MLP.
struct Block {
    norm1_weight: Array,
    norm1_bias: Array,
    norm2_weight: Array,
    norm2_bias: Array,
    /// `qkv` weight already transposed for `x @ W.T`, and its bias.
    qkv_weight: Array,
    qkv_bias: Array,
    proj_weight: Array,
    proj_bias: Array,
    fc1_weight: Array,
    fc1_bias: Array,
    fc2_weight: Array,
    fc2_bias: Array,
}

fn norm(take: &mut Take, name: &str, width: usize) -> Result<(Array, Array)> {
    Ok((
        take.bf16(&format!("{name}.weight"), &[width])?,
        take.bf16(&format!("{name}.bias"), &[width])?,
    ))
}

/// A `torch.nn.Linear` weight stored as the checkpoint does (`[out, in]`) plus
/// its bias, with the weight transposed for `x @ W.T`.
fn dense(take: &mut Take, name: &str, out: usize, input: usize) -> Result<(Array, Array)> {
    let weight = take
        .bf16(&format!("{name}.weight"), &[out, input])?
        .transpose(&[1, 0])?;
    let bias = take.bf16(&format!("{name}.bias"), &[out])?;
    Ok((weight, bias))
}

pub struct Vision {
    hidden: usize,
    heads: usize,
    head_dim: usize,
    /// Sigmoid-free `48 * 48` position table, `[num_positions, hidden]`.
    position: Array,
    patch_weight: Array,
    patch_bias: Array,
    blocks: Vec<Block>,
    merger_norm_weight: Array,
    merger_norm_bias: Array,
    merger_fc1_weight: Array,
    merger_fc1_bias: Array,
    merger_fc2_weight: Array,
    merger_fc2_bias: Array,
    /// Vision rotary: `head_dim / 2` frequencies, theta 10000, applied twice
    /// (height then width) over a 72-wide head.
    rope: MropeHalfSplitRope,
    rope_inv_freq: Array,
    rope_selector: Array,
}

impl Vision {
    pub fn load(config: &Config, stream: &Stream, checkpoint: &Path) -> Result<Self> {
        let files =
            apxinf_loader::safetensors::safetensors_files(checkpoint).map_err(Error::Other)?;
        let map =
            apxinf_mlx::load_safetensors(stream, &files, |name| name.starts_with("vision_tower."))?;
        let v = &config.vision;
        let hidden = v.hidden_size;
        let head_dim = hidden / v.num_heads;
        if head_dim % 2 != 0 {
            return Err(Error::Other(
                "Clef vision head width must be even for its rotary embedding".into(),
            ));
        }
        let mut take = Take {
            map,
            what: "Clef vision checkpoint",
        };
        // The checkpoint stores MLX `nn.Conv3d` weights in channels-last order
        // `[out, temporal, row, column, channel]`; the processor's patch vector
        // is `(channel, temporal, row, column)`, so permute before the matmul.
        let patch_weight = take
            .bf16(
                "vision_tower.patch_embed.proj.weight",
                &[
                    hidden,
                    v.temporal_patch_size,
                    v.patch_size,
                    v.patch_size,
                    v.in_channels,
                ],
            )?
            .transpose(&[0, 4, 1, 2, 3])?
            .reshape(&[hidden, PATCH_WIDTH])?
            .transpose(&[1, 0])?;
        let patch_bias = take.bf16("vision_tower.patch_embed.proj.bias", &[hidden])?;
        let position = take.bf16(
            "vision_tower.pos_embed.weight",
            &[v.num_position_embeddings, hidden],
        )?;
        let mut blocks = Vec::with_capacity(v.depth);
        for index in 0..v.depth {
            let prefix = format!("vision_tower.blocks.{index}");
            let (norm1_weight, norm1_bias) = norm(&mut take, &format!("{prefix}.norm1"), hidden)?;
            let (norm2_weight, norm2_bias) = norm(&mut take, &format!("{prefix}.norm2"), hidden)?;
            let (qkv_weight, qkv_bias) =
                dense(&mut take, &format!("{prefix}.attn.qkv"), 3 * hidden, hidden)?;
            let (proj_weight, proj_bias) =
                dense(&mut take, &format!("{prefix}.attn.proj"), hidden, hidden)?;
            let (fc1_weight, fc1_bias) = dense(
                &mut take,
                &format!("{prefix}.mlp.linear_fc1"),
                v.intermediate_size,
                hidden,
            )?;
            let (fc2_weight, fc2_bias) = dense(
                &mut take,
                &format!("{prefix}.mlp.linear_fc2"),
                hidden,
                v.intermediate_size,
            )?;
            blocks.push(Block {
                norm1_weight,
                norm1_bias,
                norm2_weight,
                norm2_bias,
                qkv_weight,
                qkv_bias,
                proj_weight,
                proj_bias,
                fc1_weight,
                fc1_bias,
                fc2_weight,
                fc2_bias,
            });
        }
        let merged_width = hidden * MERGE * MERGE;
        let (merger_norm_weight, merger_norm_bias) =
            norm(&mut take, "vision_tower.merger.norm", hidden)?;
        let (merger_fc1_weight, merger_fc1_bias) = dense(
            &mut take,
            "vision_tower.merger.linear_fc1",
            merged_width,
            merged_width,
        )?;
        let (merger_fc2_weight, merger_fc2_bias) = dense(
            &mut take,
            "vision_tower.merger.linear_fc2",
            v.out_hidden_size,
            merged_width,
        )?;
        take.finish()?;

        let rope_inv_freq =
            Array::from_f32(stream, &[head_dim / 2], &vision_rope_inv_freq(head_dim))?;
        let rope_selector =
            Array::from_i32(stream, &[head_dim / 2], &vision_rope_selector(head_dim))?;
        Ok(Self {
            hidden,
            heads: v.num_heads,
            head_dim,
            position,
            patch_weight,
            patch_bias,
            blocks,
            merger_norm_weight,
            merger_norm_bias,
            merger_fc1_weight,
            merger_fc1_bias,
            merger_fc2_weight,
            merger_fc2_bias,
            rope: MropeHalfSplitRope::new(stream, head_dim)?,
            rope_inv_freq,
            rope_selector,
        })
    }

    /// Vision features `[merged_rows, out_hidden]` for concatenated patch rows.
    ///
    /// `pixels` is F32 `[rows, PATCH_WIDTH]` in processor patch order; `grids`
    /// is `[1, height / patch, width / patch]` per image, in prompt order.
    pub fn forward(&self, pixels: &Array, grids: &[[u32; 3]]) -> Result<Array> {
        let rows = patch_rows(grids)?;
        if grids.is_empty() {
            return Err(Error::Contract("Clef vision needs at least one image"));
        }
        if pixels.dtype() != MlxDType::F32 || pixels.shape() != [rows, PATCH_WIDTH] {
            return Err(Error::Other(format!(
                "Clef vision pixels must be F32 [{rows}, {PATCH_WIDTH}], got {:?} {:?}",
                pixels.dtype(),
                pixels.shape()
            )));
        }
        let patches = pixels.cast(MlxDType::BF16)?;
        let mut hidden = linear(&patches, &self.patch_weight, &self.patch_bias)?;
        dump_stage("patch_embed", &hidden)?;
        let pos_embeddings = self.position_embeddings(grids, rows)?;
        dump_stage("pos_embed", &pos_embeddings)?;
        hidden = hidden.add(&pos_embeddings)?;
        dump_stage("pos_added", &hidden)?;

        let (positions, segments) = self.rope_positions(grids, rows)?;
        for (block_index, block) in self.blocks.iter().enumerate() {
            let normalized = layer_norm(&hidden, &block.norm1_weight, &block.norm1_bias)?;
            let qkv = linear(&normalized, &block.qkv_weight, &block.qkv_bias)?;
            let qkv = qkv
                .reshape(&[rows, 3, self.heads, self.head_dim])?
                .transpose(&[1, 0, 2, 3])?;
            let rotate = |part: usize| -> Result<Array> {
                // MLX applies the rotary embedding to `[1, seq, heads, dim]`;
                // elementwise it is the same as rotating `[1, heads, seq, dim]`.
                let part = qkv
                    .slice_axis(0, part, part + 1)?
                    .transpose(&[0, 2, 1, 3])?;
                self.rope
                    .call(&part, &positions, &self.rope_inv_freq, &self.rope_selector)
            };
            let queries = rotate(0)?;
            let keys = rotate(1)?;
            let values = qkv.slice_axis(0, 2, 3)?.transpose(&[0, 2, 1, 3])?;
            // One attention segment per image (frames are never > 1 here).
            let scale = (self.head_dim as f32).powf(-0.5);
            let mut attended = Vec::with_capacity(segments.len());
            let mut start = 0;
            for &length in &segments {
                let segment =
                    |x: &Array| -> Result<Array> { x.slice_axis(2, start, start + length) };
                attended.push(segment(&queries)?.sdpa(
                    &segment(&keys)?,
                    &segment(&values)?,
                    scale,
                    false,
                    None,
                )?);
                start += length;
            }
            let attended = Array::concat(&attended.iter().collect::<Vec<_>>(), 2)?
                .transpose(&[0, 2, 1, 3])?
                .reshape(&[rows, self.hidden])?;
            hidden = hidden.add(&linear(&attended, &block.proj_weight, &block.proj_bias)?)?;
            let normalized = layer_norm(&hidden, &block.norm2_weight, &block.norm2_bias)?;
            let activated = gelu_tanh(&linear(&normalized, &block.fc1_weight, &block.fc1_bias)?)?;
            hidden = hidden.add(&linear(&activated, &block.fc2_weight, &block.fc2_bias)?)?;
            if std::env::var_os("APXINF_CLEF_DUMP_STAGES").is_some() {
                dump_stage(&format!("block{block_index:02}"), &hidden)?;
            }
        }

        // Merger: LayerNorm per patch row, then the 2x2 merge block flattened.
        let normalized = layer_norm(&hidden, &self.merger_norm_weight, &self.merger_norm_bias)?;
        let merged_rows = rows / (MERGE * MERGE);
        let merged = normalized.reshape(&[merged_rows, self.hidden * MERGE * MERGE])?;
        let activated = gelu_exact(&linear(
            &merged,
            &self.merger_fc1_weight,
            &self.merger_fc1_bias,
        )?)?;
        let features = linear(&activated, &self.merger_fc2_weight, &self.merger_fc2_bias)?;
        dump_stage("vision_rows", &features)?;
        Ok(features)
    }

    fn stream_handle(&self) -> &Stream {
        self.position.stream()
    }

    /// `fast_pos_embed_interpolate` for images: bilinear over the 48x48 table
    /// with BF16 weights, then the `(merged, merged, intra, intra)` order.
    fn position_embeddings(&self, grids: &[[u32; 3]], rows: usize) -> Result<Array> {
        let side = (self.position.shape()[0] as f64).sqrt() as usize;
        let tokens = grid_tokens(grids, side)?.0;
        if tokens.len() != rows {
            return Err(Error::Other(format!(
                "Clef vision grids cover {} patch rows, expected {rows}",
                tokens.len()
            )));
        }
        let mut indices = [
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
        ];
        let mut weights = [
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
            Vec::with_capacity(rows),
        ];
        for token in &tokens {
            let (h, w) = (token.interpolation.0, token.interpolation.1);
            let h0 = h.floor() as usize;
            let w0 = w.floor() as usize;
            let h1 = (h0 + 1).min(side - 1);
            let w1 = (w0 + 1).min(side - 1);
            let dh = h - h0 as f32;
            let dw = w - w0 as f32;
            indices[0].push((h0 * side + w0) as i32);
            indices[1].push((h0 * side + w1) as i32);
            indices[2].push((h1 * side + w0) as i32);
            indices[3].push((h1 * side + w1) as i32);
            weights[0].push((1.0 - dh) * (1.0 - dw));
            weights[1].push((1.0 - dh) * dw);
            weights[2].push(dh * (1.0 - dw));
            weights[3].push(dh * dw);
        }
        let mut gathered = Vec::with_capacity(4);
        for corner in 0..4 {
            let index = Array::from_i32(self.position.stream(), &[rows], &indices[corner])?;
            let weight = Array::from_f32(self.position.stream(), &[rows, 1], &weights[corner])?
                .cast(MlxDType::BF16)?;
            gathered.push(self.position.take(&index, 0)?.mul(&weight)?);
        }
        // `pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]`, BF16.
        gathered[0]
            .add(&gathered[1])?
            .add(&gathered[2])?
            .add(&gathered[3])
    }

    /// `rot_pos_emb`: per patch row `[height_position, width_position]` as I32
    /// mRoPE rows `[3, 1, rows]`, plus the per-image attention segment lengths.
    fn rope_positions(&self, grids: &[[u32; 3]], rows: usize) -> Result<(Array, Vec<usize>)> {
        let side = (self.position.shape()[0] as f64).sqrt() as usize;
        let (tokens, segments) = grid_tokens(grids, side)?;
        if tokens.len() != rows {
            return Err(Error::Other(format!(
                "Clef vision grids cover {} patch rows, expected {rows}",
                tokens.len()
            )));
        }
        let position_ids = vision_rope_positions(&tokens);
        let positions = Array::from_i32(self.position.stream(), &[3, 1, rows], &position_ids)?;
        Ok((positions, segments))
    }
}

/// One vision patch row's positions, in prompt order.
struct GridToken {
    /// Table row/column before flooring: `linspace(0, side - 1, patch_grid)`.
    interpolation: (f32, f32),
    /// Rotary row/column: the patch's row/column inside its image.
    rotary: (i32, i32),
}

/// Per patch row positions, in prompt order, and the per-image attention
/// segments. `grids` is `image_grid_thw` in **patch units**
/// (`[1, height / 16, width / 16]`), the same unit `preprocess::PreparedImages`
/// and the processor use.
///
/// The traversal is one shared order for the position embedding, the rotary
/// table and the patchify loop:
/// `merged_row -> merged_column -> intra_row -> intra_column`; it is the
/// `(merged, merged, intra, intra)` permutation `fast_pos_embed_interpolate`
/// applies, and it is what makes four consecutive patch rows one merge block for
/// the merger reshape.
fn grid_tokens(grids: &[[u32; 3]], side: usize) -> Result<(Vec<GridToken>, Vec<usize>)> {
    let mut tokens = Vec::new();
    let mut segments = Vec::with_capacity(grids.len());
    for &[temporal, height, width] in grids {
        if temporal != 1 {
            return Err(Error::Other(
                "Clef vision prepares one frame per image; video grids are unsupported".into(),
            ));
        }
        let (height, width) = (height as usize, width as usize);
        if height == 0 || width == 0 || height % MERGE != 0 || width % MERGE != 0 {
            return Err(Error::Other(format!(
                "Clef vision grid [{temporal}, {height}, {width}] needs patch dimensions \
                 divisible by the merge size {MERGE}"
            )));
        }
        segments.push(height * width);
        let heights = linspace(side, height);
        let widths = linspace(side, width);
        for merged_row in 0..height / MERGE {
            for merged_column in 0..width / MERGE {
                for inner_row in 0..MERGE {
                    for inner_column in 0..MERGE {
                        let row = merged_row * MERGE + inner_row;
                        let column = merged_column * MERGE + inner_column;
                        tokens.push(GridToken {
                            interpolation: (heights[row], widths[column]),
                            rotary: (row as i32, column as i32),
                        });
                    }
                }
            }
        }
    }
    Ok((tokens, segments))
}

/// `VisionRotaryEmbedding(head_dim // 2)` frequencies, duplicated for the
/// kernel's per-pair layout.
///
/// The reference builds `dim = head_dim / 2 = 36` frequencies
/// `1 / theta^(2k / dim)` (18 of them, `k = 0..17`), `freq_table = outer(seq,
/// inv_freq)` of shape `[max_h, 18]`, `h_emb = table[h]`, `w_emb = table[w]`,
/// `concat([h_emb, w_emb])` -> `[tokens, 36]`, then `tile(1, 1, 2)` -> 72 and
/// `rotate_half` pairing `(i, i + 36)`. Pair `i < 18` therefore uses the height
/// angle and pair `18..36` the width angle, each applied to both halves of the
/// pair; the kernel reads one inverse frequency per pair.
pub(crate) fn vision_rope_inv_freq(head_dim: usize) -> Vec<f32> {
    let dim = head_dim / 2;
    let step = dim / 2;
    let inverse = (0..step)
        .map(|k| 1.0f32 / VISION_THETA.powf((2 * k) as f32 / dim as f32))
        .collect::<Vec<_>>();
    let mut table = Vec::with_capacity(dim);
    table.extend_from_slice(&inverse);
    table.extend_from_slice(&inverse);
    table
}

/// Which position axis feeds each rotary pair: height for the first 18 pairs,
/// width for the last 18 (indices into the kernel's `[3, 1, rows]` positions).
pub(crate) fn vision_rope_selector(head_dim: usize) -> Vec<i32> {
    let dim = head_dim / 2;
    let step = dim / 2;
    let mut selector = vec![0i32; dim];
    for slot in step..dim {
        selector[slot] = 1;
    }
    selector
}

/// Kernel layout `[3, 1, rows]`: height positions, width positions, then an
/// unused zero row. An entry of the third row never feeds a pair because the
/// selector only names axes 0 and 1.
pub(crate) fn vision_rope_positions(tokens: &[GridToken]) -> Vec<i32> {
    let rows = tokens.len();
    let mut positions = Vec::with_capacity(rows * 3);
    positions.extend(tokens.iter().map(|token| token.rotary.0));
    positions.extend(tokens.iter().map(|token| token.rotary.1));
    positions.extend(std::iter::repeat(0i32).take(rows));
    positions
}

/// Patch rows of `grids` (patch units), i.e. the vision tower's sequence length.
fn patch_rows(grids: &[[u32; 3]]) -> Result<usize> {
    let mut rows = 0usize;
    for &[temporal, height, width] in grids {
        if temporal == 0 || height == 0 || width == 0 {
            return Err(Error::Other(format!(
                "Clef vision grid [{temporal}, {height}, {width}] is empty"
            )));
        }
        if height as usize % MERGE != 0 || width as usize % MERGE != 0 {
            return Err(Error::Other(format!(
                "Clef vision grid [{temporal}, {height}, {width}] needs patch dimensions \
                 divisible by the merge size {MERGE}"
            )));
        }
        rows = rows
            .checked_add(temporal as usize * height as usize * width as usize)
            .ok_or_else(|| Error::Other("Clef vision patch rows overflow".into()))?;
    }
    Ok(rows)
}

/// `mx.linspace(0, side - 1, count)` in FP32: `i * step`, last point exact.
fn linspace(side: usize, count: usize) -> Vec<f32> {
    if count == 0 {
        return Vec::new();
    }
    if count == 1 {
        return vec![0.0];
    }
    let step = (side - 1) as f32 / (count - 1) as f32;
    let mut values = (0..count).map(|i| i as f32 * step).collect::<Vec<_>>();
    values[count - 1] = (side - 1) as f32;
    values
}

/// `get_rope_index` for images: text positions advance by one; each image run
/// gets `[0, h, w] + base` over its merged grid and advances the base by
/// `max(merged_height, merged_width)`.
pub fn mrope_position_ids(
    input_ids: &[u32],
    grids: &[[u32; 3]],
    image_token_id: u32,
) -> Result<Vec<[i32; 3]>> {
    let mut positions = Vec::with_capacity(input_ids.len());
    let mut grid_index = 0usize;
    let mut current = 0i32;
    let mut index = 0usize;
    while index < input_ids.len() {
        if input_ids[index] != image_token_id {
            positions.push([current; 3]);
            current += 1;
            index += 1;
            continue;
        }
        let start = index;
        while index < input_ids.len() && input_ids[index] == image_token_id {
            index += 1;
        }
        let run = index - start;
        let &[temporal, height, width] = grids.get(grid_index).ok_or_else(|| {
            Error::Other("Clef prompt has more image runs than image grids".into())
        })?;
        if temporal != 1 {
            return Err(Error::Other(
                "Clef prepares one frame per image; video prompts are unsupported".into(),
            ));
        }
        let merged_height = (height / SPATIAL_MERGE_SIZE as u32) as i32;
        let merged_width = (width / SPATIAL_MERGE_SIZE as u32) as i32;
        let per_row = (merged_height * merged_width) as usize;
        if run != per_row {
            return Err(Error::Other(format!(
                "Clef image run has {run} placeholder tokens for {per_row} merged rows"
            )));
        }
        for row in 0..merged_height {
            for column in 0..merged_width {
                positions.push([current, current + row, current + column]);
            }
        }
        current += merged_height.max(merged_width);
        grid_index += 1;
    }
    if grid_index != grids.len() {
        return Err(Error::Other(format!(
            "Clef prompt uses {grid_index} image grids, {} images were preprocessed",
            grids.len()
        )));
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_positions_follow_the_reference_index() {
        // "<|vision_start|>" + 8 image pads (a 4x8 patch image merges to 2x4)
        // + "<|vision_end|>" + three text tokens.
        let grids = [[1u32, 4, 8]];
        let mut ids = vec![248_053u32];
        ids.extend(std::iter::repeat(248_056).take(8));
        ids.push(248_054);
        ids.extend_from_slice(&[1, 2, 3]);
        let positions = mrope_position_ids(&ids, &grids, 248_056).unwrap();
        assert_eq!(positions.len(), ids.len());
        assert_eq!(positions[0], [0, 0, 0]);
        // merged 2x4 grid at base 1.
        assert_eq!(
            positions[1..9],
            [
                [1, 1, 1],
                [1, 1, 2],
                [1, 1, 3],
                [1, 1, 4],
                [1, 2, 1],
                [1, 2, 2],
                [1, 2, 3],
                [1, 2, 4]
            ]
        );
        // next text position: 1 + max(2, 4) = 5, then 6 and 7.
        assert_eq!(positions[9], [5, 5, 5]);
        assert_eq!(positions[10], [6, 6, 6]);
        assert_eq!(positions[12], [8, 8, 8]);
    }

    #[test]
    fn three_image_positions_follow_the_reference_index() {
        // Three images of different sizes, each wrapped in
        // vision_start/vision_end, as the processor emits them. Merged rows:
        // 2x4=8, 3x2=6, 1x2=2. Bases advance by max(merged_h, merged_w).
        let grids = [[1u32, 4, 8], [1, 6, 4], [1, 2, 4]];
        let mut ids = Vec::new();
        for rows in [8usize, 6, 2] {
            ids.push(248_053u32);
            ids.extend(std::iter::repeat(248_056).take(rows));
            ids.push(248_054);
        }
        ids.extend_from_slice(&[7, 8]);
        let positions = mrope_position_ids(&ids, &grids, 248_056).unwrap();
        assert_eq!(positions.len(), ids.len());
        // Image 1: base 1, merged 2x4 -> rows 1..=2, columns 1..=4.
        assert_eq!(positions[1], [1, 1, 1]);
        assert_eq!(positions[8], [1, 2, 4]);
        // vision_end at 5, vision_start at 6, image 2 base 7, merged 3x2.
        assert_eq!(positions[9], [5, 5, 5]);
        assert_eq!(positions[10], [6, 6, 6]);
        assert_eq!(positions[11], [7, 7, 7]);
        assert_eq!(positions[12], [7, 7, 8]);
        assert_eq!(positions[13], [7, 8, 7]);
        assert_eq!(positions[16], [7, 9, 8]);
        // vision_end at 10, vision_start at 11, image 3 base 12, merged 1x2.
        assert_eq!(positions[17], [10, 10, 10]);
        assert_eq!(positions[18], [11, 11, 11]);
        assert_eq!(positions[19], [12, 12, 12]);
        assert_eq!(positions[20], [12, 12, 13]);
        // vision_end at 14, then text at 15 and 16.
        assert_eq!(positions[21], [14, 14, 14]);
        assert_eq!(positions[22], [15, 15, 15]);
        assert_eq!(positions[23], [16, 16, 16]);
    }

    #[test]
    fn vision_rope_layout_matches_the_reference() {
        // head_dim 72 -> VisionRotaryEmbedding(36) -> 18 frequencies, tiled.
        let inverse = vision_rope_inv_freq(72);
        assert_eq!(inverse.len(), 36);
        assert_eq!(&inverse[..18], &inverse[18..]);
        assert_eq!(inverse[0], 1.0);
        for (index, expected) in [
            (1, 0.599_484_205_245_971_7f32),
            (9, 0.009_999_999_776_482_582),
            (17, 0.000_166_810_103_110_037_74),
        ] {
            assert!(
                (inverse[index] - expected).abs() < 1e-6 * expected.abs().max(1e-6),
                "inverse[{index}] = {}, expected {expected}",
                inverse[index]
            );
        }
        let selector = vision_rope_selector(72);
        assert_eq!(selector.len(), 36);
        assert_eq!(selector[..18], [0i32; 18]);
        assert_eq!(selector[18..], [1i32; 18]);
    }

    #[test]
    fn vision_rope_position_pairs_match_the_reference() {
        // A 4x8 patch image: 32 patch rows. Kernel layout [3, 1, rows] with the
        // height position row, then the width position row, then an unused row.
        let (tokens, _) = grid_tokens(&[[1, 4, 8]], 48).unwrap();
        let positions = vision_rope_positions(&tokens);
        assert_eq!(positions.len(), 32 * 3);
        // Token i = (merged_row, merged_column, intra_row, intra_column), so
        // the height position is the patch row and the width position the
        // patch column.
        assert_eq!(
            &positions[..32],
            &[
                0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2,
                3, 3, 2, 2, 3, 3
            ]
        );
        assert_eq!(
            &positions[32..64],
            &[
                0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 0, 1, 0, 1, 2, 3, 2, 3, 4, 5,
                4, 5, 6, 7, 6, 7
            ]
        );
        assert_eq!(&positions[64..], &[0i32; 32]);

        // Pair i < 18 takes the height position, pair 18..36 the width one, and
        // both halves of a pair share one angle: `(i, i + 36)`.
        let inverse = vision_rope_inv_freq(72);
        let selector = vision_rope_selector(72);
        let angle = |token: usize, pair: usize| -> f32 {
            let row = match selector[pair] {
                0 => positions[token],
                _ => positions[32 + token],
            };
            row as f32 * inverse[pair]
        };
        // Token 1 is (h=0, w=1): pairs 0..17 see the height position, 18..35
        // the width one.
        assert!((angle(1, 0) - 0.0).abs() < 1e-12);
        assert!((angle(1, 18) - 1.0).abs() < 1e-12);
        assert!((angle(1, 35) - inverse[17]).abs() < 1e-12);
        // Token 2 is (h=1, w=0).
        assert!((angle(2, 0) - 1.0).abs() < 1e-12);
        assert!((angle(2, 17) - inverse[17]).abs() < 1e-12);
        assert!((angle(2, 18) - 0.0).abs() < 1e-12);
        // Token 31 is (h=3, w=7).
        assert!((angle(31, 0) - 3.0).abs() < 1e-12);
        assert!((angle(31, 35) - 7.0 * inverse[17]).abs() < 1e-6);

        // Reference cos/sin fixtures (numpy float32, same positions/inverse):
        // angle 1 -> (0.5403023, 0.8414710), angle 1/10000^(34/36) ->
        // (1.0, 0.00016681), angle 0 -> (1.0, 0.0).
        let check = |value: f32, cos_expected: f32, sin_expected: f32, label: &str| {
            assert!(
                (value.cos() - cos_expected).abs() < 1e-5,
                "{label} cos {} vs {cos_expected}",
                value.cos()
            );
            assert!(
                (value.sin() - sin_expected).abs() < 1e-5,
                "{label} sin {} vs {sin_expected}",
                value.sin()
            );
        };
        check(angle(2, 0), 0.540_302_3, 0.841_470_96, "h=1 pair 0");
        check(angle(1, 18), 0.540_302_3, 0.841_470_96, "w=1 pair 18");
        check(angle(2, 18), 1.0, 0.0, "w=0 pair 18");
        check(angle(1, 35), 1.0, 0.000_166_810_1, "w=1 pair 35");
    }

    #[test]
    fn merger_reshape_groups_four_consecutive_patch_rows() -> Result<()> {
        // 8 patch rows of 2 features become 2 merged rows of 8 values; row r
        // is [r, 10r]. The merger must take patch rows 0..3, then 4..7.
        let stream = Stream::cpu()?;
        let values = (0..8u32)
            .flat_map(|row| [row as f32, 10.0 * row as f32])
            .collect::<Vec<_>>();
        let hidden = Array::from_f32(&stream, &[8, 2], &values)?;
        let merged = hidden.reshape(&[2, 8])?;
        assert_eq!(
            merged.to_f32_vec()?,
            [
                0.0, 0.0, 1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, 5.0, 50.0, 6.0, 60.0, 7.0,
                70.0
            ]
        );
        Ok(())
    }

    #[test]
    fn linspace_matches_mlx_values() {
        assert_eq!(linspace(48, 1), vec![0.0]);
        assert_eq!(linspace(48, 2), vec![0.0, 47.0]);
        let values = linspace(48, 9);
        assert_eq!(values.len(), 9);
        assert_eq!(values[1], 5.875);
        assert_eq!(*values.last().unwrap(), 47.0);
    }

    #[test]
    fn patch_rows_and_merged_rows_use_patch_units() {
        // Suite crops after preprocessing, `image_grid_thw` in patch units.
        for (grid, rows, merged) in [
            ([1u32, 24, 20], 480usize, 120usize),
            ([1, 18, 24], 432, 108),
            ([1, 20, 30], 600, 150),
            ([1, 16, 16], 256, 64),
        ] {
            assert_eq!(patch_rows(&[grid]).unwrap(), rows, "{grid:?}");
            assert_eq!(rows / (MERGE * MERGE), merged, "{grid:?}");
        }
        assert_eq!(
            patch_rows(&[[1, 24, 20], [1, 18, 24], [1, 20, 30]]).unwrap(),
            1512
        );
        assert!(patch_rows(&[[2, 8, 12]]).is_ok());
        assert!(patch_rows(&[[1, 0, 32]]).is_err());
        // A patch dimension that is not merge-divisible is rejected.
        assert!(patch_rows(&[[1, 23, 20]]).is_err());
        assert!(grid_tokens(&[[1, 23, 20]], 48).is_err());
    }

    #[test]
    fn grid_tokens_lock_the_patch_order() {
        // 4x8 patch image: 2x4 merge blocks, 32 patch rows. Tokens are one per
        // patch row in the reference order `merged_row -> merged_column ->
        // intra_row -> intra_column`; `linspace(0, 47, 4)` is
        // [0, 47/3, 94/3, 47] and `linspace(0, 47, 8)` steps by 47/7.
        let (tokens, segments) = grid_tokens(&[[1, 4, 8]], 48).unwrap();
        assert_eq!(segments, vec![32]);
        assert_eq!(tokens.len(), 32);
        let rotary: Vec<(i32, i32)> = tokens.iter().map(|token| token.rotary).collect();
        assert_eq!(
            &rotary[..8],
            &[(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (1, 2), (1, 3)]
        );
        // Merged row 1: patch rows 2 and 3, same intra order.
        assert_eq!(
            &rotary[16..24],
            &[(2, 0), (2, 1), (3, 0), (3, 1), (2, 2), (2, 3), (3, 2), (3, 3)]
        );
        let interpolation: Vec<(f32, f32)> =
            tokens.iter().map(|token| token.interpolation).collect();
        assert_eq!(interpolation[0], (0.0, 0.0));
        assert_eq!(interpolation[1].0, 0.0);
        assert!((interpolation[1].1 - 47.0 / 7.0).abs() < 1e-5);
        assert!((interpolation[2].0 - 47.0 / 3.0).abs() < 1e-5);
        assert_eq!(interpolation[2].1, 0.0);
        // Token 4 is the next merged column, first patch row of the block.
        assert_eq!(interpolation[4].0, 0.0);
        assert!((interpolation[4].1 - 2.0 * 47.0 / 7.0).abs() < 1e-5);
        // Token 16 is merged row 1, so its patch row is 2.
        assert!((interpolation[16].0 - 2.0 * 47.0 / 3.0).abs() < 1e-5);
        assert_eq!(interpolation[16].1, 0.0);
    }
}

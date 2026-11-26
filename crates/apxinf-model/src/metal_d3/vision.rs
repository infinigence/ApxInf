//! Qwen3.5 vision tower on native MLX: patch embed, learned position-table
//! interpolation, axial RoPE, `depth` pre-norm blocks with non-causal SDPA
//! over each image's patches, and the spatial merger that turns every 2 x 2
//! patch group into one LLM image token.
//!
//! Semantics follow the checkpoint's reference (transformers 5.17.0
//! `Qwen3_5VisionModel`, which `d3_runtime` loads): the Conv3d patch embed is
//! the matrix product its kernel = stride equals; the 48 x 48 learned table
//! is resampled with bilinear, align-corners interpolation and *then*
//! reordered into spatial-merge-block order; RoPE uses Keys' cubic with
//! `a = -0.5` recomposed as `[H | W | H | W]`; every norm, activation and
//! attention runs in FP32 opmath and rounds once to BF16.
//!
//! The tower runs eagerly (the reference does too): the host knows each
//! image's grid, so the interpolation taps and RoPE coordinates are built per
//! request on the host and uploaded as inputs to this graph. Only the text
//! layers plus readout are compiled, as one whole-request callable.

use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType, Stream};

use super::config::{Config, VisionContract};
use super::weights::{BiasedLinear, VisionBlock, VisionWeights};

const EPS: f32 = 1e-6;
/// `M_SQRT2 * M_2_SQRTPI * 0.5` in the reference's opmath (double constexpr
/// narrowed to float), so the activation matches torch bit for bit in F32.
const GELU_TANH_BETA: f32 = 0.797_884_560_802_865_4;
const GELU_TANH_KAPPA: f32 = 0.044_715;

/// One vision execution over shared weights.
pub struct Vision {
    weights: VisionWeights,
    contract: VisionContract,
    /// Axial inverse frequencies `[head_dim / 4]` in F32 (16 for d3-lite).
    inv_freq: Array,
    stream: Stream,
}

/// `[pixel | weight]` helpers a linear layer's bias needs.
fn biased_project(input: &Array, layer: &BiasedLinear) -> Result<Array> {
    input
        .matmul(&layer.weight.transpose(&[1, 0])?)?
        .add(&layer.bias)
}

/// `0.5 x (1 + tanh(β (x + κ x³)))`, the `gelu_pytorch_tanh` activation,
/// computed in F32 and rounded once to the input dtype (as torch's CPU
/// kernel does with `opmath_t`).
fn gelu_tanh(x: &Array) -> Result<Array> {
    let compute = x.cast(MlxDType::F32)?;
    let square = compute.mul(&compute)?;
    let cube = square.mul(&compute)?;
    let shifted = compute.add(&cube.mul(&scalar(&compute, GELU_TANH_KAPPA)?)?)?;
    let inner = shifted.mul(&scalar(&compute, GELU_TANH_BETA)?)?;
    // tanh(z) = 1 - 2·sigmoid(-2z), evaluated in F32.
    let tanh = scalar(&inner, 1.0)?.sub(
        &inner
            .mul(&scalar(&inner, -2.0)?)?
            .sigmoid()?
            .mul(&scalar(&inner, 2.0)?)?,
    )?;
    let output = scalar(&compute, 0.5)?
        .mul(&compute)?
        .mul(&scalar(&compute, 1.0)?.add(&tanh)?)?;
    output.cast(x.dtype())
}

/// Exact `0.5 x (1 + erf(x / √2))` GELU, the merger's `nn.GELU()`.
fn gelu_exact(x: &Array) -> Result<Array> {
    let compute = x.cast(MlxDType::F32)?;
    let argument = compute.div(&scalar(&compute, std::f32::consts::SQRT_2)?)?;
    let output = scalar(&compute, 0.5)?
        .mul(&compute)?
        .mul(&scalar(&compute, 1.0)?.add(&argument.erf()?)?)?;
    output.cast(x.dtype())
}

fn scalar(like: &Array, value: f32) -> Result<Array> {
    Array::scalar(like.stream(), value, like.dtype())
}

/// Retain one stage when a diagnostic run asks for it.
fn record(stages: &mut Option<&mut Vec<(&'static str, Array)>>, name: &'static str, array: &Array) {
    if let Some(list) = stages.as_deref_mut() {
        list.push((name, array.clone()));
    }
}

impl Vision {
    pub fn new(config: &Config, stream: Stream, weights: VisionWeights) -> Result<Self> {
        let contract = config.vision;
        let values = axial_inverse_frequencies(contract);
        let inv_freq = Array::from_f32(&stream, &[contract.head_dim / 4], &values)?;
        stream.eval(&[inv_freq.clone()])?;
        Ok(Self {
            weights,
            contract,
            inv_freq,
            stream,
        })
    }

    /// Decoded processor input → merged image tokens `[total, out_hidden]` BF16.
    ///
    /// `pixel_values` is the F32 `[sum(t·h·w), C·T·P·P]` matrix; `grids` is
    /// the request's `[t, h, w]` rows. Still images have `t = 1` and each
    /// image attends only its own patches, exactly as the reference's
    /// `cu_seqlens` splits them.
    pub fn forward(&self, pixel_values: &[f32], grids: &[[u32; 3]]) -> Result<Array> {
        self.run(pixel_values, grids, None)
    }

    /// The same forward with every named stage retained, for the reference
    /// comparison that localizes a divergence (patch embed, position add,
    /// each block, merger norm, merged output). Diagnostic only; used by the
    /// device acceptance test.
    #[cfg(test)]
    pub fn forward_stages(
        &self,
        pixel_values: &[f32],
        grids: &[[u32; 3]],
    ) -> Result<Vec<(&'static str, Vec<f32>)>> {
        let mut stages = Vec::new();
        let output = self.run(pixel_values, grids, Some(&mut stages))?;
        stages.push(("merged", output));
        let arrays = stages
            .iter()
            .map(|(_, array)| array.clone())
            .collect::<Vec<_>>();
        self.stream.eval(&arrays)?;
        stages
            .into_iter()
            .map(|(name, array)| Ok((name, array.to_f32_vec()?)))
            .collect()
    }

    fn run(
        &self,
        pixel_values: &[f32],
        grids: &[[u32; 3]],
        mut stages: Option<&mut Vec<(&'static str, Array)>>,
    ) -> Result<Array> {
        let contract = self.contract;
        let patches = expected_patches(grids)?;
        if pixel_values.len()
            != patches * 3 * contract.temporal_patch * contract.patch * contract.patch
        {
            return Err(Error::Other(format!(
                "d3-lite vision got {} pixel values for {patches} patches of {} floats",
                pixel_values.len(),
                3 * contract.temporal_patch * contract.patch * contract.patch
            )));
        }
        let pixels = Array::from_f32(
            &self.stream,
            &[
                patches,
                3 * contract.temporal_patch * contract.patch * contract.patch,
            ],
            pixel_values,
        )?
        .cast(MlxDType::BF16)?;
        let projected = pixels
            .matmul(&self.weights.patch_weight.transpose(&[1, 0])?)?
            .add(&self.weights.patch_bias)?;
        record(&mut stages, "patch_embed", &projected);

        // Learned position table, bilinear (align corners) onto each grid,
        // summed over the four taps in F32 and rounded once to BF16.
        let (indices, weights) = self.interpolation(grids)?;
        let taken = self.weights.pos_embed.take(&indices, 0)?;
        let weighted = taken
            .cast(MlxDType::F32)?
            .mul(&weights.reshape(&[patches, 4, 1])?)?;
        let position = weighted.sum(&[1], false)?.cast(MlxDType::BF16)?;
        let hidden = projected.add(&position)?;
        record(&mut stages, "block_input", &hidden);

        // Axial RoPE: angles for H and W recomposed as `[H | W | H | W]`.
        let coordinates = self.rope_coordinates(grids)?;
        let (cos, sin) = vision_rotary(&coordinates, &self.inv_freq, patches)?;
        record(&mut stages, "b0_cos", &cos);
        record(&mut stages, "b0_sin", &sin);

        let segments = attention_segments(grids)?;
        let mut hidden = hidden;
        for (index, block) in self.weights.blocks.iter().enumerate() {
            hidden = vision_block(
                &self.contract,
                block,
                &hidden,
                &cos,
                &sin,
                &segments,
                index == 0,
                &mut stages,
            )?;
            let name: &'static str = match index {
                0 => "block_0",
                1 => "block_1",
                2 => "block_2",
                3 => "block_3",
                4 => "block_4",
                5 => "block_5",
                6 => "block_6",
                7 => "block_7",
                8 => "block_8",
                9 => "block_9",
                10 => "block_10",
                _ => "block_11",
            };
            record(&mut stages, name, &hidden);
        }
        merger_forward(
            &contract,
            &self.weights,
            &hidden,
            patches,
            stages.as_deref_mut(),
        )
    }
}

fn vision_block(
    contract: &VisionContract,
    block: &VisionBlock,
    hidden: &Array,
    cos: &Array,
    sin: &Array,
    segments: &[(usize, usize)],
    details: bool,
    stages: &mut Option<&mut Vec<(&'static str, Array)>>,
) -> Result<Array> {
    let patches = hidden.shape()[0];
    let normalized =
        hidden.fast_layer_norm(Some(&block.norm1_weight), Some(&block.norm1_bias), EPS)?;
    if details {
        record(stages, "b0_norm1", &normalized);
    }
    let projected = biased_project(&normalized, &block.qkv)?;
    if details {
        record(stages, "b0_qkv", &projected);
    }
    let qkv = projected
        .reshape(&[patches, 3, contract.heads, contract.head_dim])?
        .transpose(&[1, 0, 2, 3])?;
    let queries =
        qkv.slice_axis(0, 0, 1)?
            .reshape(&[patches, contract.heads, contract.head_dim])?;
    let keys = qkv
        .slice_axis(0, 1, 2)?
        .reshape(&[patches, contract.heads, contract.head_dim])?;
    let values = qkv
        .slice_axis(0, 2, 3)?
        .reshape(&[patches, contract.heads, contract.head_dim])?;
    // FP32 opmath with a single rounding back to BF16, as
    // `apply_rotary_pos_emb_vision` does.
    let rotate = |tensor: &Array| -> Result<Array> {
        let tensor = tensor.cast(MlxDType::F32)?;
        let first = tensor.slice_axis(2, contract.head_dim / 2, contract.head_dim)?;
        let second = tensor.slice_axis(2, 0, contract.head_dim / 2)?;
        let rotated = Array::concat(&[&first.neg()?, &second], 2)?;
        tensor
            .mul(&cos.reshape(&[patches, 1, contract.head_dim])?)?
            .add(&rotated.mul(&sin.reshape(&[patches, 1, contract.head_dim])?)?)?
            .cast(MlxDType::BF16)
    };
    let queries = rotate(&queries)?;
    let keys = rotate(&keys)?;
    if details {
        record(stages, "b0_rope_q", &queries);
    }
    let queries = queries.transpose(&[1, 0, 2])?.contiguous()?.reshape(&[
        1,
        contract.heads,
        patches,
        contract.head_dim,
    ])?;
    let keys = keys.transpose(&[1, 0, 2])?.contiguous()?.reshape(&[
        1,
        contract.heads,
        patches,
        contract.head_dim,
    ])?;
    let values = values.transpose(&[1, 0, 2])?.contiguous()?.reshape(&[
        1,
        contract.heads,
        patches,
        contract.head_dim,
    ])?;

    // Per-image (or per-frame) non-causal attention; the reference splits
    // `qkv` by `cu_seqlens` and concatenates along the sequence axis.
    let mut attended = Vec::with_capacity(segments.len());
    for &(start, length) in segments {
        let q = queries.slice_axis(2, start, start + length)?;
        let k = keys.slice_axis(2, start, start + length)?;
        let v = values.slice_axis(2, start, start + length)?;
        attended.push(q.sdpa(&k, &v, (contract.head_dim as f32).powf(-0.5), false, None)?);
    }
    let joined = Array::concat(&attended.iter().collect::<Vec<_>>(), 2)?
        .transpose(&[0, 2, 1, 3])?
        .contiguous()?
        .reshape(&[patches, contract.heads * contract.head_dim])?;
    if details {
        record(stages, "b0_attn", &joined);
    }
    let projected = biased_project(&joined, &block.proj)?;
    if details {
        record(stages, "b0_proj", &projected);
    }
    let residual = hidden.add(&projected)?;

    let normalized =
        residual.fast_layer_norm(Some(&block.norm2_weight), Some(&block.norm2_bias), EPS)?;
    if details {
        record(stages, "b0_norm2", &normalized);
    }
    let fc1 = biased_project(&normalized, &block.fc1)?;
    if details {
        record(stages, "b0_fc1", &fc1);
    }
    let inner = gelu_tanh(&fc1)?;
    if details {
        record(stages, "b0_act", &inner);
    }
    let outer = biased_project(&inner, &block.fc2)?;
    if details {
        record(stages, "b0_fc2", &outer);
    }
    residual.add(&outer)
}

fn merger_forward(
    contract: &VisionContract,
    weights: &VisionWeights,
    hidden: &Array,
    patches: usize,
    mut stages: Option<&mut Vec<(&'static str, Array)>>,
) -> Result<Array> {
    let merge_unit = contract.spatial_merge * contract.spatial_merge;
    if patches % merge_unit != 0 {
        return Err(Error::Contract(
            "d3-lite vision patch count is not a multiple of the spatial merge unit",
        ));
    }
    let merged_width = contract.hidden * merge_unit;
    let normalized = hidden.fast_layer_norm(
        Some(&weights.merger.norm_weight),
        Some(&weights.merger.norm_bias),
        EPS,
    )?;
    record(&mut stages, "merger_norm", &normalized);
    let grouped = normalized.reshape(&[patches / merge_unit, merged_width])?;
    let activated = gelu_exact(&biased_project(&grouped, &weights.merger.fc1)?)?;
    biased_project(&activated, &weights.merger.fc2)
}

/// `1 / base ** (torch.arange(0, spatial_dim, 2) / spatial_dim)` with
/// `spatial_dim = head_dim / 2`, the vision tower's axial inverse frequencies.
/// The exponent advances by two (the even indices), which is what pairs the
/// 16 stored frequencies with the 32-wide H/W halves of each head.
fn axial_inverse_frequencies(contract: VisionContract) -> Vec<f32> {
    (0..contract.head_dim / 4)
        .map(|index| {
            let exponent = (2 * index) as f32 / (contract.head_dim / 2) as f32;
            // f32 `powf` reproduces the reference's `inv_freq` exactly.
            1.0f32 / contract.rope_theta.powf(exponent)
        })
        .collect()
}

/// Axial RoPE for the vision tower: `position_ids [N, 2]` (H, W) times the
/// 16 inverse frequencies, recomposed by the reference as
/// `[H | W | H | W]` over the 64-wide head, all in F32.
fn vision_rotary(coordinates: &Array, inv_freq: &Array, patches: usize) -> Result<(Array, Array)> {
    let angles = coordinates
        .cast(MlxDType::F32)?
        .reshape(&[patches, 2, 1])?
        .mul(&inv_freq.reshape(&[1, 1, inv_freq.shape()[0]])?)?;
    let recompose = |values: &Array| -> Result<Array> {
        let rows = values
            .slice_axis(1, 0, 1)?
            .reshape(&[patches, inv_freq.shape()[0]])?;
        let columns = values
            .slice_axis(1, 1, 2)?
            .reshape(&[patches, inv_freq.shape()[0]])?;
        let paired = Array::concat(&[&rows, &columns], 1)?;
        Array::concat(&[&paired, &paired], 1)
    };
    let cos = recompose(&angles.cos()?)?;
    let sin = recompose(&angles.sin()?)?;
    Ok((cos, sin))
}

/// The reference's `get_image_features` input size: `Σ t·h·w` patches.
fn expected_patches(grids: &[[u32; 3]]) -> Result<usize> {
    if grids.is_empty() {
        return Err(Error::Contract("d3-lite vision needs at least one grid"));
    }
    grids
        .iter()
        .map(|&[t, h, w]| {
            if t == 0 || h == 0 || w == 0 {
                return Err(Error::Contract(
                    "d3-lite vision grid entries must be nonzero [t, h, w]",
                ));
            }
            Ok(t as usize * h as usize * w as usize)
        })
        .sum()
}

/// Per-image `(start, length)` ranges for the reference's `cu_seqlens` split.
/// d3-lite images have `t = 1`, so one segment per image.
fn attention_segments(grids: &[[u32; 3]]) -> Result<Vec<(usize, usize)>> {
    let mut start = 0usize;
    let mut segments = Vec::with_capacity(grids.len());
    for &[t, h, w] in grids {
        if t != 1 {
            return Err(Error::Contract(
                "d3-lite Metal vision runs still images (grid t = 1); video inputs are not wired",
            ));
        }
        let length = h as usize * w as usize;
        segments.push((start, length));
        start += length;
    }
    Ok(segments)
}

impl Vision {
    /// Bilinear, align-corners taps of the learned 48 x 48 table for every
    /// patch of every image, in merge-block order: `[N, 4]` I32 indices and
    /// `[N, 4]` F32 weights (the reference's gather/weight contract).
    fn interpolation(&self, grids: &[[u32; 3]]) -> Result<(Array, Array)> {
        let contract = self.contract;
        let side = contract.num_grid_per_side;
        let merge = contract.spatial_merge;
        let mut indices = Vec::new();
        let mut weights = Vec::new();
        for &[t, h, w] in grids {
            if t != 1 {
                return Err(Error::Contract(
                    "d3-lite Metal vision runs still images (grid t = 1); video inputs are not wired",
                ));
            }
            let h = h as usize;
            let w = w as usize;
            for block_row in 0..h / merge {
                for block_column in 0..w / merge {
                    for inner_row in 0..merge {
                        for inner_column in 0..merge {
                            let row = block_row * merge + inner_row;
                            let column = block_column * merge + inner_column;
                            let row_source = row as f32 * (side - 1) as f32 / (h - 1).max(1) as f32;
                            let column_source =
                                column as f32 * (side - 1) as f32 / (w - 1).max(1) as f32;
                            let row_floor = row_source.floor();
                            let column_floor = column_source.floor();
                            let mut row_taps = [0i32; 2];
                            let mut column_taps = [0i32; 2];
                            let mut row_weights = [0.0f32; 2];
                            let mut column_weights = [0.0f32; 2];
                            for (tap, offset) in [0.0f32, 1.0].into_iter().enumerate() {
                                row_taps[tap] =
                                    (row_floor as i32 + offset as i32).clamp(0, side as i32 - 1);
                                column_taps[tap] =
                                    (column_floor as i32 + offset as i32).clamp(0, side as i32 - 1);
                                row_weights[tap] =
                                    (1.0 - (row_source - row_floor - offset).abs()).max(0.0);
                                column_weights[tap] =
                                    (1.0 - (column_source - column_floor - offset).abs()).max(0.0);
                            }
                            for row_tap in 0..2 {
                                for column_tap in 0..2 {
                                    indices.push(
                                        row_taps[row_tap] * side as i32 + column_taps[column_tap],
                                    );
                                    weights.push(row_weights[row_tap] * column_weights[column_tap]);
                                }
                            }
                        }
                    }
                }
            }
        }
        let patches = indices.len() / 4;
        let indices = Array::from_i32(&self.stream, &[patches, 4], &indices)?;
        let weights = Array::from_f32(&self.stream, &[patches, 4], &weights)?;
        Ok((indices, weights))
    }

    /// Axial RoPE coordinates `(h, w)` per patch, merge-block order `[N, 2] I32`.
    fn rope_coordinates(&self, grids: &[[u32; 3]]) -> Result<Array> {
        let contract = self.contract;
        let merge = contract.spatial_merge;
        let mut coordinates = Vec::new();
        for &[t, h, w] in grids {
            if t != 1 {
                return Err(Error::Contract(
                    "d3-lite Metal vision runs still images (grid t = 1); video inputs are not wired",
                ));
            }
            let h = h as usize;
            let w = w as usize;
            for block_row in 0..h / merge {
                for block_column in 0..w / merge {
                    for inner_row in 0..merge {
                        for inner_column in 0..merge {
                            coordinates.push((block_row * merge + inner_row) as i32);
                            coordinates.push((block_column * merge + inner_column) as i32);
                        }
                    }
                }
            }
        }
        let patches = coordinates.len() / 2;
        Array::from_i32(&self.stream, &[patches, 2], &coordinates)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn contract() -> VisionContract {
        VisionContract {
            depth: 12,
            hidden: 768,
            intermediate: 3_072,
            heads: 12,
            head_dim: 64,
            out_hidden: 1_024,
            patch: 16,
            temporal_patch: 2,
            spatial_merge: 2,
            num_position_embeddings: 2_304,
            num_grid_per_side: 48,
            rope_theta: 10_000.0,
            tensors: 153,
        }
    }

    #[test]
    fn axial_inverse_frequencies_advance_by_two_exponents() {
        let values = axial_inverse_frequencies(contract());
        assert_eq!(values.len(), 16);
        assert_eq!(values[0], 1.0);
        // 10000^(-1/16) and 10000^(-1/8): the step is 2 / spatial_dim, not 1.
        assert!((values[1] - 0.562_341_33).abs() < 1e-6, "{}", values[1]);
        assert!((values[2] - 0.316_227_77).abs() < 1e-6, "{}", values[2]);
        // Monotone decay, one entry per H/W half-frequency.
        assert!(values.windows(2).all(|pair| pair[0] > pair[1]));
    }
}

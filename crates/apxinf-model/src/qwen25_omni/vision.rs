//! Native Qwen2.5-Omni vision encoder.

use std::collections::HashMap;

use apxinf_core::{Backend, Error, Result, Tensor};

use super::config::Qwen25OmniConfig;
use super::weights::{flatten_and_transpose, transpose_2d};
#[cfg(feature = "cuda")]
use crate::accelerator::cuda::downcast as cuda_backend;

pub struct Qwen25OmniVisionWeights {
    patch_embed: Tensor,
    blocks: Vec<VisionBlock>,
    merger_norm: Tensor,
    merger_fc1: Tensor,
    merger_fc1_bias: Tensor,
    merger_fc2: Tensor,
    merger_fc2_bias: Tensor,
}

struct VisionBlock {
    norm1: Tensor,
    qkv: VisionQkvWeights,
    bq: Tensor,
    bk: Tensor,
    bv: Tensor,
    wo: Tensor,
    bo: Tensor,
    norm2: Tensor,
    gate_up: VisionGateUpWeights,
    b_gate: Tensor,
    b_up: Tensor,
    w_down: Tensor,
    b_down: Tensor,
}

enum VisionGateUpWeights {
    Separate { gate: Tensor, up: Tensor },
    Packed(Tensor),
}

impl VisionGateUpWeights {
    fn separate(&self) -> Result<(&Tensor, &Tensor)> {
        match self {
            Self::Separate { gate, up } => Ok((gate, up)),
            Self::Packed(_) => Err(Error::Other(
                "Qwen2.5-Omni vision separate Gate/Up path has packed weights".into(),
            )),
        }
    }

    #[cfg(feature = "cuda")]
    fn packed(&self) -> Result<&Tensor> {
        match self {
            Self::Packed(weight) => Ok(weight),
            Self::Separate { .. } => Err(Error::Other(
                "Qwen2.5-Omni vision packed Gate/Up path has separate weights".into(),
            )),
        }
    }
}

enum VisionQkvWeights {
    Separate { wq: Tensor, wk: Tensor, wv: Tensor },
    Packed(Tensor),
}

impl VisionQkvWeights {
    fn separate(&self) -> Result<(&Tensor, &Tensor, &Tensor)> {
        match self {
            Self::Separate { wq, wk, wv } => Ok((wq, wk, wv)),
            Self::Packed(_) => Err(Error::Other(
                "Qwen2.5-Omni vision separate QKV path has packed weights".into(),
            )),
        }
    }

    #[cfg(feature = "cuda")]
    fn packed(&self) -> Result<&Tensor> {
        match self {
            Self::Packed(weight) => Ok(weight),
            Self::Separate { .. } => Err(Error::Other(
                "Qwen2.5-Omni vision packed QKV path has separate weights".into(),
            )),
        }
    }
}

impl Qwen25OmniVisionWeights {
    pub fn from_map(
        config: &Qwen25OmniConfig,
        tensors: &mut HashMap<String, Tensor>,
    ) -> Result<Self> {
        let take = |name: &str, map: &mut HashMap<String, Tensor>| {
            map.remove(name)
                .ok_or_else(|| Error::Other(format!("missing {name}")))
        };
        let vision = &config.vision;
        let patch_width =
            vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size;
        let patch = take("thinker.visual.patch_embed.proj.weight", tensors)?;
        let patch_embed = flatten_and_transpose(&patch, vision.hidden_size, patch_width)?;
        let mut blocks = Vec::with_capacity(vision.depth);
        for index in 0..vision.depth {
            let prefix = format!("thinker.visual.blocks.{index}");
            blocks.push(VisionBlock {
                norm1: take(&format!("{prefix}.norm1.weight"), tensors)?,
                qkv: VisionQkvWeights::Separate {
                    wq: transpose_2d(&take(&format!("{prefix}.attn.q.weight"), tensors)?)?,
                    wk: transpose_2d(&take(&format!("{prefix}.attn.k.weight"), tensors)?)?,
                    wv: transpose_2d(&take(&format!("{prefix}.attn.v.weight"), tensors)?)?,
                },
                bq: take(&format!("{prefix}.attn.q.bias"), tensors)?,
                bk: take(&format!("{prefix}.attn.k.bias"), tensors)?,
                bv: take(&format!("{prefix}.attn.v.bias"), tensors)?,
                wo: transpose_2d(&take(&format!("{prefix}.attn.proj.weight"), tensors)?)?,
                bo: take(&format!("{prefix}.attn.proj.bias"), tensors)?,
                norm2: take(&format!("{prefix}.norm2.weight"), tensors)?,
                gate_up: VisionGateUpWeights::Separate {
                    gate: transpose_2d(&take(
                        &format!("{prefix}.mlp.gate_proj.weight"),
                        tensors,
                    )?)?,
                    up: transpose_2d(&take(
                        &format!("{prefix}.mlp.up_proj.weight"),
                        tensors,
                    )?)?,
                },
                b_gate: take(&format!("{prefix}.mlp.gate_proj.bias"), tensors)?,
                b_up: take(&format!("{prefix}.mlp.up_proj.bias"), tensors)?,
                w_down: transpose_2d(&take(&format!("{prefix}.mlp.down_proj.weight"), tensors)?)?,
                b_down: take(&format!("{prefix}.mlp.down_proj.bias"), tensors)?,
            });
        }
        Ok(Self {
            patch_embed,
            blocks,
            merger_norm: take("thinker.visual.merger.ln_q.weight", tensors)?,
            merger_fc1: transpose_2d(&take("thinker.visual.merger.mlp.0.weight", tensors)?)?,
            merger_fc1_bias: take("thinker.visual.merger.mlp.0.bias", tensors)?,
            merger_fc2: transpose_2d(&take("thinker.visual.merger.mlp.2.weight", tensors)?)?,
            merger_fc2_bias: take("thinker.visual.merger.mlp.2.bias", tensors)?,
        })
    }

    pub fn to_device(self, backend: &dyn Backend) -> Result<Self> {
        let blocks = self
            .blocks
            .into_iter()
            .map(|block| {
                let qkv = match block.qkv {
                    VisionQkvWeights::Separate { wq, wk, wv } => VisionQkvWeights::Separate {
                        wq: backend.to_device(&wq)?,
                        wk: backend.to_device(&wk)?,
                        wv: backend.to_device(&wv)?,
                    },
                    VisionQkvWeights::Packed(weight) => {
                        VisionQkvWeights::Packed(backend.to_device(&weight)?)
                    }
                };
                let gate_up = match block.gate_up {
                    VisionGateUpWeights::Separate { gate, up } => {
                        VisionGateUpWeights::Separate {
                            gate: backend.to_device(&gate)?,
                            up: backend.to_device(&up)?,
                        }
                    }
                    VisionGateUpWeights::Packed(weight) => {
                        VisionGateUpWeights::Packed(backend.to_device(&weight)?)
                    }
                };
                Ok(VisionBlock {
                    norm1: backend.to_device(&block.norm1)?,
                    qkv,
                    bq: backend.to_device(&block.bq)?,
                    bk: backend.to_device(&block.bk)?,
                    bv: backend.to_device(&block.bv)?,
                    wo: backend.to_device(&block.wo)?,
                    bo: backend.to_device(&block.bo)?,
                    norm2: backend.to_device(&block.norm2)?,
                    gate_up,
                    b_gate: backend.to_device(&block.b_gate)?,
                    b_up: backend.to_device(&block.b_up)?,
                    w_down: backend.to_device(&block.w_down)?,
                    b_down: backend.to_device(&block.b_down)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            patch_embed: backend.to_device(&self.patch_embed)?,
            blocks,
            merger_norm: backend.to_device(&self.merger_norm)?,
            merger_fc1: backend.to_device(&self.merger_fc1)?,
            merger_fc1_bias: backend.to_device(&self.merger_fc1_bias)?,
            merger_fc2: backend.to_device(&self.merger_fc2)?,
            merger_fc2_bias: backend.to_device(&self.merger_fc2_bias)?,
        })
    }

    pub fn into_packed_qkv(self, backend: &dyn Backend) -> Result<Self> {
        let blocks = self
            .blocks
            .into_iter()
            .map(|block| {
                let qkv = match block.qkv {
                    VisionQkvWeights::Separate { wq, wk, wv } => {
                        VisionQkvWeights::Packed(backend.concat_2d(&[&wq, &wk, &wv])?)
                    }
                    VisionQkvWeights::Packed(_) => {
                        return Err(Error::Other(
                            "Qwen2.5-Omni vision QKV weights were packed twice".into(),
                        ))
                    }
                };
                Ok(VisionBlock {
                    norm1: block.norm1,
                    qkv,
                    bq: block.bq,
                    bk: block.bk,
                    bv: block.bv,
                    wo: block.wo,
                    bo: block.bo,
                    norm2: block.norm2,
                    gate_up: block.gate_up,
                    b_gate: block.b_gate,
                    b_up: block.b_up,
                    w_down: block.w_down,
                    b_down: block.b_down,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        backend.synchronize()?;
        Ok(Self {
            patch_embed: self.patch_embed,
            blocks,
            merger_norm: self.merger_norm,
            merger_fc1: self.merger_fc1,
            merger_fc1_bias: self.merger_fc1_bias,
            merger_fc2: self.merger_fc2,
            merger_fc2_bias: self.merger_fc2_bias,
        })
    }

    pub fn into_packed_gate_up(self, backend: &dyn Backend) -> Result<Self> {
        let blocks = self
            .blocks
            .into_iter()
            .map(|block| {
                let gate_up = match block.gate_up {
                    VisionGateUpWeights::Separate { gate, up } => {
                        VisionGateUpWeights::Packed(backend.concat_2d(&[&gate, &up])?)
                    }
                    VisionGateUpWeights::Packed(_) => {
                        return Err(Error::Other(
                            "Qwen2.5-Omni vision Gate/Up weights were packed twice".into(),
                        ))
                    }
                };
                Ok(VisionBlock {
                    norm1: block.norm1,
                    qkv: block.qkv,
                    bq: block.bq,
                    bk: block.bk,
                    bv: block.bv,
                    wo: block.wo,
                    bo: block.bo,
                    norm2: block.norm2,
                    gate_up,
                    b_gate: block.b_gate,
                    b_up: block.b_up,
                    w_down: block.w_down,
                    b_down: block.b_down,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        backend.synchronize()?;
        Ok(Self {
            patch_embed: self.patch_embed,
            blocks,
            merger_norm: self.merger_norm,
            merger_fc1: self.merger_fc1,
            merger_fc1_bias: self.merger_fc1_bias,
            merger_fc2: self.merger_fc2,
            merger_fc2_bias: self.merger_fc2_bias,
        })
    }
}

pub fn forward(
    config: &Qwen25OmniConfig,
    weights: &Qwen25OmniVisionWeights,
    backend: &dyn Backend,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
    use_fused_qkv_bias_rope: bool,
    use_fused_silu_mul: bool,
    use_fused_bias_residual: bool,
    use_fused_gate_up_bias_silu_mul: bool,
    use_grouped_qkv_layout: bool,
    use_packed_qkv: bool,
    use_packed_gate_up: bool,
) -> Result<Tensor> {
    #[cfg(not(feature = "cuda"))]
    let _ = (use_packed_qkv, use_packed_gate_up);
    let vision = &config.vision;
    let raw_tokens = validate_input(config, pixel_values, grid_thw)?;
    let uploaded = if pixel_values.device() != backend.device() {
        Some(backend.to_device(pixel_values)?)
    } else {
        None
    };
    let pixels = uploaded.as_ref().unwrap_or(pixel_values);
    let mut hidden = backend.matmul(pixels, &weights.patch_embed)?;
    let positions = vision_positions(grid_thw, vision.spatial_merge_size);
    let groups = vision_window_groups(config, grid_thw)?;
    for (index, block) in weights.blocks.iter().enumerate() {
        let full_attention = vision.full_attention_blocks.contains(&index);
        let grouped_qkv_layout = use_grouped_qkv_layout && !full_attention;
        let normalized = backend.rms_norm(&hidden, &block.norm1, 1e-6)?;
        let (q, k, v) = if use_fused_qkv_bias_rope {
            #[cfg(feature = "cuda")]
            {
                let cuda = cuda_backend(backend).ok_or_else(|| {
                    Error::Other("vision QKV bias/RoPE fusion requires CudaBackend".into())
                })?;
                if use_packed_qkv {
                    let weight = block.qkv.packed()?;
                    let packed = backend.matmul(&normalized, weight)?;
                    cuda.qwen25_omni_vision_packed_qkv_bias_rope(
                        &packed,
                        &block.bq,
                        &block.bk,
                        &block.bv,
                        10_000.0,
                        &positions,
                        grouped_qkv_layout.then_some(groups.as_slice()),
                    )?
                } else if grouped_qkv_layout {
                    let (wq, wk, wv) = block.qkv.separate()?;
                    let query = backend.matmul(&normalized, wq)?;
                    let key = backend.matmul(&normalized, wk)?;
                    let value = backend.matmul(&normalized, wv)?;
                    cuda.qwen25_omni_vision_grouped_qkv_bias_rope(
                        &query,
                        &key,
                        &value,
                        &block.bq,
                        &block.bk,
                        &block.bv,
                        10_000.0,
                        &positions,
                        &groups,
                    )?
                } else {
                    let (wq, wk, wv) = block.qkv.separate()?;
                    let query = backend.matmul(&normalized, wq)?;
                    let key = backend.matmul(&normalized, wk)?;
                    let value = backend.matmul(&normalized, wv)?;
                    cuda.qwen25_omni_vision_qkv_bias_rope(
                        &query,
                        &key,
                        &value,
                        &block.bq,
                        &block.bk,
                        &block.bv,
                        10_000.0,
                        &positions,
                    )?
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(Error::Other(
                    "vision QKV bias/RoPE fusion requires CUDA".into(),
                ));
            }
        } else {
            let (wq, wk, wv) = block.qkv.separate()?;
            let q = backend.add_bias(&backend.matmul(&normalized, wq)?, &block.bq)?;
            let k = backend.add_bias(&backend.matmul(&normalized, wk)?, &block.bk)?;
            let v = backend.add_bias(&backend.matmul(&normalized, wv)?, &block.bv)?;
            let q = q.reshape(vec![raw_tokens, vision.n_heads, vision.head_dim])?;
            let k = k.reshape(vec![raw_tokens, vision.n_heads, vision.head_dim])?;
            let v = v.reshape(vec![raw_tokens, vision.n_heads, vision.head_dim])?;
            let q = backend.rope_vision_2d(
                &q,
                vision.n_heads,
                vision.head_dim,
                10_000.0,
                &positions,
            )?;
            let k = backend.rope_vision_2d(
                &k,
                vision.n_heads,
                vision.head_dim,
                10_000.0,
                &positions,
            )?;
            (q, k, v)
        };
        let attention = if full_attention {
            backend.vision_sdpa(&q, &k, &v, raw_tokens, vision.n_heads, vision.head_dim)?
        } else if grouped_qkv_layout {
            #[cfg(feature = "cuda")]
            {
                let cuda = cuda_backend(backend).ok_or_else(|| {
                    Error::Other("prepacked grouped vision attention requires CudaBackend".into())
                })?;
                cuda.qwen25_omni_vision_grouped_sdpa_prepacked(
                    &q,
                    &k,
                    &v,
                    raw_tokens,
                    vision.n_heads,
                    vision.head_dim,
                    &groups,
                )?
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(Error::Other(
                    "prepacked grouped vision attention requires CUDA".into(),
                ));
            }
        } else {
            backend.grouped_sdpa(
                &q,
                &k,
                &v,
                raw_tokens,
                vision.n_heads,
                vision.head_dim,
                &groups,
            )?
        };
        let attention = backend.matmul(&attention, &block.wo)?;
        hidden = projection_bias_residual(
            backend,
            &attention,
            &block.bo,
            &hidden,
            use_fused_bias_residual,
        )?;
        let normalized = backend.rms_norm(&hidden, &block.norm2, 1e-6)?;
        let mlp = if use_fused_gate_up_bias_silu_mul {
            #[cfg(feature = "cuda")]
            {
                let cuda = cuda_backend(backend).ok_or_else(|| {
                    Error::Other(
                        "vision Gate/Up bias SiLU/multiply fusion requires CudaBackend".into(),
                    )
                })?;
                if use_packed_gate_up {
                    let weight = block.gate_up.packed()?;
                    let packed = backend.matmul(&normalized, weight)?;
                    cuda.qwen25_omni_vision_packed_gate_up_bias_silu_mul_exact(
                        &packed,
                        &block.b_gate,
                        &block.b_up,
                    )?
                } else {
                    let (gate_weight, up_weight) = block.gate_up.separate()?;
                    let gate = backend.matmul(&normalized, gate_weight)?;
                    let up = backend.matmul(&normalized, up_weight)?;
                    cuda.qwen25_omni_vision_gate_up_bias_silu_mul_exact(
                        &gate,
                        &block.b_gate,
                        &up,
                        &block.b_up,
                    )?
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(Error::Other(
                    "vision Gate/Up bias SiLU/multiply fusion requires CUDA".into(),
                ));
            }
        } else {
            let (gate_weight, up_weight) = block.gate_up.separate()?;
            let gate = backend.matmul(&normalized, gate_weight)?;
            let gate = backend.add_bias(&gate, &block.b_gate)?;
            let up = backend.add_bias(&backend.matmul(&normalized, up_weight)?, &block.b_up)?;
            if use_fused_silu_mul {
                backend.silu_mul(&gate, &up)?
            } else {
                backend.mul(&backend.silu(&gate)?, &up)?
            }
        };
        let mlp = backend.matmul(&mlp, &block.w_down)?;
        hidden = projection_bias_residual(
            backend,
            &mlp,
            &block.b_down,
            &hidden,
            use_fused_bias_residual,
        )?;
    }

    let normalized = backend.rms_norm(&hidden, &weights.merger_norm, 1e-6)?;
    let merged_tokens = merged_token_count(grid_thw, vision.spatial_merge_size)?;
    let merged_width = vision.hidden_size * vision.spatial_merge_size.pow(2);
    let merged = normalized.reshape(vec![merged_tokens, merged_width])?;
    let merged = backend.add_bias(
        &backend.matmul(&merged, &weights.merger_fc1)?,
        &weights.merger_fc1_bias,
    )?;
    let merged = backend.gelu_tanh(&merged)?;
    backend.add_bias(
        &backend.matmul(&merged, &weights.merger_fc2)?,
        &weights.merger_fc2_bias,
    )
}

fn projection_bias_residual(
    backend: &dyn Backend,
    projection: &Tensor,
    bias: &Tensor,
    residual: &Tensor,
    use_fused: bool,
) -> Result<Tensor> {
    if use_fused {
        #[cfg(feature = "cuda")]
        {
            let cuda = cuda_backend(backend).ok_or_else(|| {
                Error::Other("vision exact bias/residual fusion requires CudaBackend".into())
            })?;
            return cuda.qwen25_omni_vision_bias_residual_exact(projection, bias, residual);
        }
        #[cfg(not(feature = "cuda"))]
        {
            return Err(Error::Other(
                "vision exact bias/residual fusion requires CUDA".into(),
            ));
        }
    }
    backend.add(&backend.add_bias(projection, bias)?, residual)
}

/// Validate processor-owned image views without executing a backend operator.
pub fn validate_input(
    config: &Qwen25OmniConfig,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<usize> {
    if grid_thw.len() != 1 {
        return Err(Error::Other(
            "qwen2.5-omni first deployment slice accepts exactly one image grid".into(),
        ));
    }
    let vision = &config.vision;
    let patch_width =
        vision.in_channels * vision.temporal_patch_size * vision.patch_size * vision.patch_size;
    let raw_tokens = validate_grids(grid_thw, vision.spatial_merge_size)?;
    if pixel_values.shape().dims() != [raw_tokens, patch_width] {
        return Err(Error::Other(format!(
            "qwen2.5-omni pixel_values shape {:?}, expected [{raw_tokens}, {patch_width}]",
            pixel_values.shape().dims()
        )));
    }
    Ok(raw_tokens)
}

pub fn merged_token_count(grid_thw: &[[u32; 3]], merge: usize) -> Result<usize> {
    let merge = u32::try_from(merge)
        .map_err(|_| Error::Other("qwen2.5-omni merge size exceeds u32".into()))?;
    grid_thw
        .iter()
        .try_fold(0usize, |total, &[time, height, width]| {
            let count = (time as usize)
                .checked_mul((height / merge) as usize)
                .and_then(|value| value.checked_mul((width / merge) as usize))
                .ok_or_else(|| Error::Other("qwen2.5-omni merged image token overflow".into()))?;
            total
                .checked_add(count)
                .ok_or_else(|| Error::Other("qwen2.5-omni merged image total overflow".into()))
        })
}

fn validate_grids(grid_thw: &[[u32; 3]], merge: usize) -> Result<usize> {
    let merge = u32::try_from(merge)
        .map_err(|_| Error::Other("qwen2.5-omni merge size exceeds u32".into()))?;
    if merge == 0 {
        return Err(Error::Other("qwen2.5-omni merge size is zero".into()));
    }
    grid_thw
        .iter()
        .try_fold(0usize, |total, &[time, height, width]| {
            if time == 0 || height == 0 || width == 0 || height % merge != 0 || width % merge != 0 {
                return Err(Error::Other(format!(
                    "qwen2.5-omni invalid image grid [{time}, {height}, {width}] for merge {merge}"
                )));
            }
            let count = (time as usize)
                .checked_mul(height as usize)
                .and_then(|value| value.checked_mul(width as usize))
                .ok_or_else(|| Error::Other("qwen2.5-omni image token overflow".into()))?;
            total
                .checked_add(count)
                .ok_or_else(|| Error::Other("qwen2.5-omni image token total overflow".into()))
        })
}

fn vision_positions(grid_thw: &[[u32; 3]], merge: usize) -> Vec<u32> {
    let mut positions = Vec::new();
    for &[time, height, width] in grid_thw {
        for _ in 0..time {
            for merged_row in 0..height as usize / merge {
                for merged_col in 0..width as usize / merge {
                    for inner_row in 0..merge {
                        for inner_col in 0..merge {
                            positions.extend_from_slice(&[
                                (merged_row * merge + inner_row) as u32,
                                (merged_col * merge + inner_col) as u32,
                            ]);
                        }
                    }
                }
            }
        }
    }
    positions
}

fn vision_window_groups(config: &Qwen25OmniConfig, grid_thw: &[[u32; 3]]) -> Result<Vec<u32>> {
    let patches_per_window = config.vision.window_size / config.vision.patch_size;
    if patches_per_window == 0 {
        return Err(Error::Other(
            "qwen2.5-omni vision window is smaller than a patch".into(),
        ));
    }
    let mut groups = Vec::new();
    let mut group_base = 0_u32;
    for &[time, height, width] in grid_thw {
        let window_columns = (width as usize).div_ceil(patches_per_window);
        let window_rows = (height as usize).div_ceil(patches_per_window);
        let merge = config.vision.spatial_merge_size;
        for temporal in 0..time as usize {
            for merged_row in 0..height as usize / merge {
                for merged_col in 0..width as usize / merge {
                    for inner_row in 0..merge {
                        for inner_col in 0..merge {
                            let row = merged_row * merge + inner_row;
                            let col = merged_col * merge + inner_col;
                            let group = temporal * window_rows * window_columns
                                + (row / patches_per_window) * window_columns
                                + col / patches_per_window;
                            groups.push(group_base + group as u32);
                        }
                    }
                }
            }
        }
        group_base = group_base
            .checked_add((time as usize * window_rows * window_columns) as u32)
            .ok_or_else(|| Error::Other("qwen2.5-omni vision group id overflow".into()))?;
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_merged_grid_contract() {
        assert_eq!(merged_token_count(&[[1, 4, 6]], 2).unwrap(), 6);
        assert!(validate_grids(&[[1, 3, 4]], 2).is_err());
    }

    #[test]
    fn validates_one_processor_image_before_backend_work() {
        let raw = include_str!("../../tests/data/qwen25_omni_config_minimal.json");
        let config = Qwen25OmniConfig::from_json_str(raw).unwrap();
        let pixels = Tensor::from_f32(vec![16, 1176], &vec![0.0; 16 * 1176]).unwrap();
        assert_eq!(validate_input(&config, &pixels, &[[1, 4, 4]]).unwrap(), 16);
        assert!(validate_input(&config, &pixels, &[]).is_err());
        assert!(validate_input(&config, &pixels, &[[1, 4, 4], [1, 4, 4]]).is_err());
        let wrong = Tensor::from_f32(vec![15, 1176], &vec![0.0; 15 * 1176]).unwrap();
        assert!(validate_input(&config, &wrong, &[[1, 4, 4]]).is_err());
    }

    #[test]
    fn real_png_grid_has_bounded_nonempty_windows() {
        let raw = include_str!("../../tests/data/qwen25_omni_config_minimal.json");
        let config = Qwen25OmniConfig::from_json_str(raw).unwrap();
        let groups = vision_window_groups(&config, &[[1, 64, 108]]).unwrap();
        assert_eq!(groups.len(), 6_912);
        assert_eq!(groups.iter().copied().max(), Some(111));
        let mut counts = vec![0usize; 112];
        for group in groups {
            counts[group as usize] += 1;
        }
        assert_eq!(counts.iter().filter(|&&count| count == 64).count(), 104);
        assert_eq!(counts.iter().filter(|&&count| count == 32).count(), 8);
        assert!(counts.iter().all(|&count| count > 0 && count <= 64));
    }
}

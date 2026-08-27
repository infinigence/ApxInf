//! Helpers for Hugging Face `compressed-tensors` pack-quantized weights.
//!
//! The Qwen3.8-27B AWQ checkpoint stores W4A16 linear weights as four tensors
//! per logical weight:
//!
//! - `<prefix>.weight_packed`: `I32 [out_features, in_features / 8]`
//! - `<prefix>.weight_scale`: `BF16 [out_features, in_features / group_size]`
//! - `<prefix>.weight_zero_point`: `I32 [out_features / 8, in_features / group_size]`
//! - `<prefix>.weight_shape`: `I64 [2]`, logical `[out_features, in_features]`
//!
//! Each `I32` packs eight 4-bit unsigned nibbles, low nibble first. Zero-points
//! are packed across output rows for each input group; weights are packed across
//! input columns for each output row.

use std::collections::HashMap;

use apxinf_core::{DType, Tensor};

/// CUDA decode layout identifier. Values cross the Rust/CUDA FFI boundary and
/// therefore form a versioned ABI rather than an implementation detail.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum W4KernelLayout {
    /// Hugging Face compressed-tensors output-major tensors.
    RawCompressedTensors = 0,
    /// K-major 16-column subtiles inside 64-output-row tiles.
    RepackedN64K16V1 = 1,
    /// Upstream vLLM Marlin AWQ U4 layout with group size 32.
    MarlinAwqU4G32V1 = 2,
}

pub const W4_REPACKED_N64_K16_V1_SUFFIX: &str = "repacked_n64k16_v1";
/// Exact upstream Marlin AWQ layout produced on the host.
pub const W4_MARLIN_AWQ_U4_G32_V1_SUFFIX: &str = "marlin_awq_u4_g32_v1";
/// Runtime-generated persistent device-side transform cache suffix.
pub const W4_TRANSFORM_CACHE_SUFFIX: &str = "transform_cache_n64k16_v1";

/// Complete geometry and byte accounting for one physical W4 representation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct W4LayoutMetadata {
    pub layout: W4KernelLayout,
    pub version: u32,
    pub logical_rows: usize,
    pub logical_cols: usize,
    pub padded_rows: usize,
    pub padded_cols: usize,
    pub group_size: usize,
    pub groups: usize,
    pub packed_bytes: usize,
    pub scale_bytes: usize,
    pub zero_point_bytes: usize,
    pub total_bytes: usize,
    pub source_total_bytes: usize,
}

impl W4LayoutMetadata {
    /// Bytes uploaded for this physical representation. The raw checkpoint
    /// tensors are CPU-owned fallback data and are not assumed to be uploaded.
    pub fn device_bytes(&self) -> usize {
        self.total_bytes
    }

    /// Size delta between representations, not an allocation requirement.
    pub fn additional_device_bytes(&self) -> usize {
        self.total_bytes.saturating_sub(self.source_total_bytes)
    }
}

/// Host-owned repack result. Uploading these tensors transfers allocation
/// ownership to the backend's device tensors; the checkpoint tensors remain the
/// authoritative raw fallback and are never silently reinterpreted.
pub struct RepackedW4Weight {
    pub packed: Tensor,
    pub scale: Tensor,
    pub zero_point: Tensor,
    pub metadata: W4LayoutMetadata,
}

/// Physically transpose compressed-tensors W4 metadata for coalesced decode.
///
/// The V1 indices are:
/// `qword[n/64][k/16][n%64][(k%16)/8]`,
/// `scale[n/64][group][n%64]`, and
/// `zero_point[n/64][group][(n%64)/8]`. N is padded to 64 rows. K must
/// already be a multiple of 128 and groups must be exactly 32 columns so the
/// decode kernel can preserve the established eight m16n8k16 MMA sequence.
/// Unsupported geometry returns `Ok(None)` and must use the raw ABI.
pub fn repack_w4_n64_k16_v1(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
) -> Result<Option<RepackedW4Weight>, String> {
    let packed = get(tensors, prefix, "weight_packed")?;
    let scale = get(tensors, prefix, "weight_scale")?;
    let zero_point = get(tensors, prefix, "weight_zero_point")?;
    let shape = get(tensors, prefix, "weight_shape")?;
    expect_dtype(packed, DType::I32, "weight_packed")?;
    expect_dtype(scale, DType::BF16, "weight_scale")?;
    expect_dtype(zero_point, DType::I32, "weight_zero_point")?;
    expect_dtype(shape, DType::I64, "weight_shape")?;

    let logical = shape.as_i64().map_err(|error| error.to_string())?;
    if logical.len() != 2 || logical.iter().any(|&dim| dim <= 0) {
        return Err(format!(
            "{prefix}.weight_shape must contain two positive dimensions, got {logical:?}"
        ));
    }
    let rows = logical[0] as usize;
    let cols = logical[1] as usize;
    let groups = scale.shape().dims().get(1).copied().unwrap_or(0);
    if groups == 0 || cols % groups != 0 || cols / groups != 32 || cols % 128 != 0 {
        return Ok(None);
    }
    let packed_cols = cols / 8;
    if packed.shape().dims() != [rows, packed_cols]
        || scale.shape().dims() != [rows, groups]
        || zero_point.shape().dims() != [rows.div_ceil(8), groups]
    {
        return Err(format!("{prefix}: invalid compressed-tensors W4 geometry"));
    }

    let padded_rows = rows.div_ceil(64) * 64;
    let n_tiles = padded_rows / 64;
    let k_tiles = cols / 16;
    let source_packed = packed.as_i32().map_err(|error| error.to_string())?;
    let source_scales = scale.as_bf16().map_err(|error| error.to_string())?;
    let source_zp = zero_point.as_i32().map_err(|error| error.to_string())?;
    let mut repacked = vec![0i32; padded_rows * packed_cols];
    let mut repacked_scales = vec![half::bf16::ZERO; padded_rows * groups];
    let mut repacked_zp = vec![0i32; padded_rows / 8 * groups];

    for row in 0..rows {
        let n_tile = row / 64;
        let row_in_tile = row % 64;
        for k_tile in 0..k_tiles {
            let dst = (((n_tile * k_tiles + k_tile) * 64 + row_in_tile) * 2) as usize;
            let src = row * packed_cols + k_tile * 2;
            repacked[dst..dst + 2].copy_from_slice(&source_packed[src..src + 2]);
        }
        for group in 0..groups {
            repacked_scales[(n_tile * groups + group) * 64 + row_in_tile] =
                source_scales[row * groups + group];
        }
    }
    for n_tile in 0..n_tiles {
        let valid_packs = rows.saturating_sub(n_tile * 64).min(64).div_ceil(8);
        for group in 0..groups {
            for row_pack in 0..valid_packs {
                repacked_zp[(n_tile * groups + group) * 8 + row_pack] =
                    source_zp[(n_tile * 8 + row_pack) * groups + group];
            }
        }
    }

    let packed_bytes = repacked.len() * std::mem::size_of::<i32>();
    let scale_bytes = repacked_scales.len() * std::mem::size_of::<half::bf16>();
    let zero_point_bytes = repacked_zp.len() * std::mem::size_of::<i32>();
    let source_total_bytes = packed.size_in_bytes() + scale.size_in_bytes()
        + zero_point.size_in_bytes();
    let metadata = W4LayoutMetadata {
        layout: W4KernelLayout::RepackedN64K16V1,
        version: 1,
        logical_rows: rows,
        logical_cols: cols,
        padded_rows,
        padded_cols: cols,
        group_size: 32,
        groups,
        packed_bytes,
        scale_bytes,
        zero_point_bytes,
        total_bytes: packed_bytes + scale_bytes + zero_point_bytes,
        source_total_bytes,
    };
    let packed = Tensor::from_raw(
        vec![n_tiles, k_tiles, 64, 2].into(),
        DType::I32,
        apxinf_core::Device::Cpu,
        bytemuck::cast_slice(&repacked).to_vec(),
    )
    .map_err(|error| error.to_string())?;
    let scale = Tensor::from_bf16(vec![n_tiles, groups, 64], &repacked_scales)
        .map_err(|error| error.to_string())?;
    let zero_point = Tensor::from_raw(
        vec![n_tiles, groups, 8].into(),
        DType::I32,
        apxinf_core::Device::Cpu,
        bytemuck::cast_slice(&repacked_zp).to_vec(),
    )
    .map_err(|error| error.to_string())?;
    Ok(Some(RepackedW4Weight { packed, scale, zero_point, metadata }))
}

/// Convert compressed-tensors U4 group-32 tensors to the exact vLLM Marlin
/// AWQ representation on the CPU.
///
/// The returned physical shapes are qweight `[padded_K / 16, padded_N * 2]`
/// I32, scales `[padded_K / 32, padded_N]` BF16, and zero-points
/// `[padded_K / 32, padded_N / 8]` I32. `logical_rows/logical_cols` in the
/// metadata mean logical N/K, while `padded_rows/padded_cols` mean padded N/K.
/// Upload only these returned tensors for the Marlin path; source tensors stay
/// CPU checkpoint data for the raw fallback.
///
/// Qweight indexing is a literal CPU transcription of vLLM v0.27.1
/// `awq_marlin_repack_kernel<..., 4, false>` (Apache-2.0): within each 16x64
/// K/N tile, word `thread * 4 + warp` gathers `n0=warp*16+thread/4` and
/// `k0=(thread%4)*2`, then packs values in kernel order. Scales and zero-points
/// use vLLM's `get_scale_perms`; zero-points additionally use Marlin's
/// `[0,2,4,6,1,3,5,7]` interleave before low-nibble packing.
pub fn repack_w4_marlin_awq_u4_g32_v1(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
) -> Result<Option<RepackedW4Weight>, String> {
    const GROUP_SIZE: usize = 32;
    // SPDX-derived mapping: vLLM v0.27.1 marlin_utils.py::get_scale_perms.
    const SCALE_PERM: [usize; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57,
        2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
        4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61,
        6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
    ];
    const INTERLEAVE: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

    let packed = get(tensors, prefix, "weight_packed")?;
    let scale = get(tensors, prefix, "weight_scale")?;
    let zero_point = get(tensors, prefix, "weight_zero_point")?;
    let shape = get(tensors, prefix, "weight_shape")?;
    expect_dtype(packed, DType::I32, "weight_packed")?;
    expect_dtype(scale, DType::BF16, "weight_scale")?;
    expect_dtype(zero_point, DType::I32, "weight_zero_point")?;
    expect_dtype(shape, DType::I64, "weight_shape")?;
    let logical = shape.as_i64().map_err(|error| error.to_string())?;
    if logical.len() != 2 || logical.iter().any(|&dim| dim <= 0) {
        return Err(format!("{prefix}.weight_shape must contain two positive dimensions, got {logical:?}"));
    }
    let logical_n = logical[0] as usize;
    let logical_k = logical[1] as usize;
    if logical_n % 8 != 0 || logical_k % GROUP_SIZE != 0 {
        return Ok(None);
    }
    let logical_groups = logical_k / GROUP_SIZE;
    if packed.shape().dims() != [logical_n, logical_k / 8]
        || scale.shape().dims() != [logical_n, logical_groups]
        || zero_point.shape().dims() != [logical_n / 8, logical_groups]
    {
        return Err(format!("{prefix}: invalid compressed-tensors W4 geometry"));
    }

    let a = (logical_n.div_ceil(64) * 64, logical_k.div_ceil(128) * 128);
    let b = (logical_n.div_ceil(128) * 128, logical_k.div_ceil(64) * 64);
    let (padded_n, padded_k) = if (a.0 * a.1, a.0 + a.1) <= (b.0 * b.1, b.0 + b.1) { a } else { b };
    let padded_groups = padded_k / GROUP_SIZE;
    let source_q = packed.as_i32().map_err(|error| error.to_string())?;
    let source_scales = scale.as_bf16().map_err(|error| error.to_string())?;
    let source_zp = zero_point.as_i32().map_err(|error| error.to_string())?;
    let q_at = |n: usize, k: usize| -> u32 {
        if n >= logical_n || k >= logical_k { return 0; }
        ((source_q[n * (logical_k / 8) + k / 8] as u32) >> ((k % 8) * 4)) & 0xF
    };
    let mut marlin_q = vec![0i32; padded_k / 16 * padded_n * 2];
    let n_tiles = padded_n / 64;
    for k_tile in 0..padded_k / 16 {
        for n_tile in 0..n_tiles {
            let tile_base = (k_tile * n_tiles + n_tile) * 128;
            for warp in 0..4 {
                for thread in 0..32 {
                    let n0 = n_tile * 64 + warp * 16 + thread / 4;
                    let k0 = k_tile * 16 + (thread % 4) * 2;
                    let vals = [q_at(n0,k0), q_at(n0,k0+1), q_at(n0,k0+8), q_at(n0,k0+9), q_at(n0+8,k0), q_at(n0+8,k0+1), q_at(n0+8,k0+8), q_at(n0+8,k0+9)];
                    let mut word = 0u32;
                    for (nibble, &source) in INTERLEAVE.iter().enumerate() { word |= vals[source] << (nibble * 4); }
                    marlin_q[tile_base + thread * 4 + warp] = word as i32;
                }
            }
        }
    }
    let mut marlin_scales = vec![half::bf16::ZERO; padded_groups * padded_n];
    for group in 0..logical_groups {
        for block in 0..padded_n / 64 {
            for (dst, &src) in SCALE_PERM.iter().enumerate() {
                let n = block * 64 + src;
                if n < logical_n { marlin_scales[group * padded_n + block * 64 + dst] = source_scales[n * logical_groups + group]; }
            }
        }
    }
    let zp_at = |group: usize, n: usize| -> u32 {
        if group >= logical_groups || n >= logical_n { return 0; }
        ((source_zp[(n / 8) * logical_groups + group] as u32) >> ((n % 8) * 4)) & 0xF
    };
    let mut marlin_zp = vec![0i32; padded_groups * (padded_n / 8)];
    for group in 0..padded_groups {
        for block in 0..padded_n / 64 {
            for word_index in 0..8 {
                let mut word = 0u32;
                for (nibble, &interleaved) in INTERLEAVE.iter().enumerate() {
                    let source_n = block * 64 + SCALE_PERM[word_index * 8 + interleaved];
                    word |= zp_at(group, source_n) << (nibble * 4);
                }
                marlin_zp[group * (padded_n / 8) + block * 8 + word_index] = word as i32;
            }
        }
    }
    let packed_bytes = marlin_q.len() * 4;
    let scale_bytes = marlin_scales.len() * 2;
    let zero_point_bytes = marlin_zp.len() * 4;
    let metadata = W4LayoutMetadata { layout: W4KernelLayout::MarlinAwqU4G32V1, version: 1, logical_rows: logical_n, logical_cols: logical_k, padded_rows: padded_n, padded_cols: padded_k, group_size: GROUP_SIZE, groups: padded_groups, packed_bytes, scale_bytes, zero_point_bytes, total_bytes: packed_bytes + scale_bytes + zero_point_bytes, source_total_bytes: packed.size_in_bytes() + scale.size_in_bytes() + zero_point.size_in_bytes() };
    let packed = Tensor::from_raw(vec![padded_k / 16, padded_n * 2].into(), DType::I32, apxinf_core::Device::Cpu, bytemuck::cast_slice(&marlin_q).to_vec()).map_err(|error| error.to_string())?;
    let scale = Tensor::from_bf16(vec![padded_groups, padded_n], &marlin_scales).map_err(|error| error.to_string())?;
    let zero_point = Tensor::from_raw(vec![padded_groups, padded_n / 8].into(), DType::I32, apxinf_core::Device::Cpu, bytemuck::cast_slice(&marlin_zp).to_vec()).map_err(|error| error.to_string())?;
    Ok(Some(RepackedW4Weight { packed, scale, zero_point, metadata }))
}

/// Repack several same-K projections into one physical Marlin N dimension.
///
/// Concatenation is exact when every member's logical N already satisfies the
/// Marlin N alignment. In that case whole repacked N tiles, scale blocks, and
/// zero-point blocks can be concatenated without unpacking or changing any
/// quantized value. Unsupported groups return `Ok(None)` so callers retain
/// their established individual representation and launch path.
pub fn repack_w4_marlin_awq_u4_g32_v1_concat(
    tensors: &HashMap<String, Tensor>,
    prefixes: &[&str],
) -> Result<Option<(RepackedW4Weight, Vec<usize>)>, String> {
    if prefixes.len() < 2 {
        return Ok(None);
    }
    let mut members = Vec::with_capacity(prefixes.len());
    for prefix in prefixes {
        let Some(weight) = repack_w4_marlin_awq_u4_g32_v1(tensors, prefix)? else {
            return Ok(None);
        };
        members.push(weight);
    }
    let padded_k = members[0].metadata.padded_cols;
    if members.iter().any(|member| {
        member.metadata.logical_cols != members[0].metadata.logical_cols
            || member.metadata.padded_cols != padded_k
            || member.metadata.logical_rows != member.metadata.padded_rows
    }) {
        return Ok(None);
    }

    let padded_n = members.iter().map(|member| member.metadata.padded_rows).sum::<usize>();
    if padded_n % 64 != 0 {
        return Ok(None);
    }
    let groups = padded_k / 32;
    let k_rows = padded_k / 16;
    let mut packed_words = Vec::with_capacity(k_rows * padded_n * 2);
    for k_row in 0..k_rows {
        for member in &members {
            let member_n = member.metadata.padded_rows;
            let words = member.packed.as_i32().map_err(|error| error.to_string())?;
            let begin = k_row * member_n * 2;
            packed_words.extend_from_slice(&words[begin..begin + member_n * 2]);
        }
    }
    let mut scales = Vec::with_capacity(groups * padded_n);
    let mut zero_points = Vec::with_capacity(groups * padded_n / 8);
    for group in 0..groups {
        for member in &members {
            let member_n = member.metadata.padded_rows;
            let values = member.scale.as_bf16().map_err(|error| error.to_string())?;
            let begin = group * member_n;
            scales.extend_from_slice(&values[begin..begin + member_n]);
        }
        for member in &members {
            let member_n = member.metadata.padded_rows;
            let values = member.zero_point.as_i32().map_err(|error| error.to_string())?;
            let begin = group * (member_n / 8);
            zero_points.extend_from_slice(&values[begin..begin + member_n / 8]);
        }
    }

    let mut member_offsets = Vec::with_capacity(members.len());
    let mut logical_n = 0usize;
    for member in &members {
        member_offsets.push(logical_n);
        logical_n += member.metadata.logical_rows;
    }
    let packed_bytes = packed_words.len() * 4;
    let scale_bytes = scales.len() * 2;
    let zero_point_bytes = zero_points.len() * 4;
    let source_total_bytes = members.iter().map(|member| member.metadata.source_total_bytes).sum();
    let metadata = W4LayoutMetadata {
        layout: W4KernelLayout::MarlinAwqU4G32V1,
        version: 1,
        logical_rows: logical_n,
        logical_cols: members[0].metadata.logical_cols,
        padded_rows: padded_n,
        padded_cols: padded_k,
        group_size: 32,
        groups,
        packed_bytes,
        scale_bytes,
        zero_point_bytes,
        total_bytes: packed_bytes + scale_bytes + zero_point_bytes,
        source_total_bytes,
    };
    let packed = Tensor::from_raw(
        vec![k_rows, padded_n * 2].into(),
        DType::I32,
        apxinf_core::Device::Cpu,
        bytemuck::cast_slice(&packed_words).to_vec(),
    ).map_err(|error| error.to_string())?;
    let scale = Tensor::from_bf16(vec![groups, padded_n], &scales)
        .map_err(|error| error.to_string())?;
    let zero_point = Tensor::from_raw(
        vec![groups, padded_n / 8].into(),
        DType::I32,
        apxinf_core::Device::Cpu,
        bytemuck::cast_slice(&zero_points).to_vec(),
    ).map_err(|error| error.to_string())?;
    Ok(Some((
        RepackedW4Weight { packed, scale, zero_point, metadata },
        member_offsets,
    )))
}

/// Quantize one dense row-major BF16 matrix to asymmetric unsigned W4A16
/// group-32, then emit the existing Marlin physical layout directly.
///
/// This is an explicit lossy candidate for weights excluded by the checkpoint's
/// quantization policy (currently the LM head). Callers must validate complete
/// output trajectories before promotion. No raw W4 or second device layout is
/// produced.
pub fn quantize_bf16_marlin_awq_u4_g32_v1(
    dense: &Tensor,
) -> Result<Option<RepackedW4Weight>, String> {
    use rayon::prelude::*;

    const GROUP_SIZE: usize = 32;
    const SCALE_PERM: [usize; 64] = [
        0, 8, 16, 24, 32, 40, 48, 56, 1, 9, 17, 25, 33, 41, 49, 57,
        2, 10, 18, 26, 34, 42, 50, 58, 3, 11, 19, 27, 35, 43, 51, 59,
        4, 12, 20, 28, 36, 44, 52, 60, 5, 13, 21, 29, 37, 45, 53, 61,
        6, 14, 22, 30, 38, 46, 54, 62, 7, 15, 23, 31, 39, 47, 55, 63,
    ];
    const INTERLEAVE: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];

    if dense.dtype() != DType::BF16 || dense.shape().dims().len() != 2 {
        return Ok(None);
    }
    let logical_n = dense.shape().dims()[0];
    let logical_k = dense.shape().dims()[1];
    if logical_n == 0 || logical_k == 0 || logical_k % GROUP_SIZE != 0 {
        return Ok(None);
    }
    let source = dense.as_bf16().map_err(|error| error.to_string())?;
    let logical_groups = logical_k / GROUP_SIZE;
    let padded_n = logical_n.div_ceil(64) * 64;
    let padded_k = logical_k.div_ceil(128) * 128;
    let padded_groups = padded_k / GROUP_SIZE;

    struct QuantizedRow {
        q: Vec<u8>,
        scales: Vec<half::bf16>,
        zero_points: Vec<u8>,
    }
    let rows = source
        .par_chunks_exact(logical_k)
        .map(|row| {
            let mut q = vec![0u8; logical_k];
            let mut scales = Vec::with_capacity(logical_groups);
            let mut zero_points = Vec::with_capacity(logical_groups);
            for group in 0..logical_groups {
                let values = &row[group * GROUP_SIZE..(group + 1) * GROUP_SIZE];
                let (mut min_value, mut max_value) = (f32::INFINITY, f32::NEG_INFINITY);
                for value in values {
                    let value = value.to_f32();
                    min_value = min_value.min(value);
                    max_value = max_value.max(value);
                }
                let scale_f32 = ((max_value - min_value) / 15.0).max(f32::MIN_POSITIVE);
                let scale = half::bf16::from_f32(scale_f32);
                let effective_scale = scale.to_f32();
                let zero_point = (-min_value / effective_scale).round().clamp(0.0, 15.0) as u8;
                for (index, value) in values.iter().enumerate() {
                    q[group * GROUP_SIZE + index] =
                        (value.to_f32() / effective_scale + f32::from(zero_point))
                            .round()
                            .clamp(0.0, 15.0) as u8;
                }
                scales.push(scale);
                zero_points.push(zero_point);
            }
            QuantizedRow { q, scales, zero_points }
        })
        .collect::<Vec<_>>();

    let n_tiles = padded_n / 64;
    let mut marlin_q = vec![0i32; padded_k / 16 * padded_n * 2];
    for k_tile in 0..padded_k / 16 {
        for n_tile in 0..n_tiles {
            let tile_base = (k_tile * n_tiles + n_tile) * 128;
            for warp in 0..4 {
                for thread in 0..32 {
                    let n0 = n_tile * 64 + warp * 16 + thread / 4;
                    let k0 = k_tile * 16 + (thread % 4) * 2;
                    let q_at = |n: usize, k: usize| -> u32 {
                        rows.get(n)
                            .and_then(|row| row.q.get(k))
                            .copied()
                            .unwrap_or(0) as u32
                    };
                    let values = [
                        q_at(n0, k0), q_at(n0, k0 + 1), q_at(n0, k0 + 8), q_at(n0, k0 + 9),
                        q_at(n0 + 8, k0), q_at(n0 + 8, k0 + 1),
                        q_at(n0 + 8, k0 + 8), q_at(n0 + 8, k0 + 9),
                    ];
                    let mut word = 0u32;
                    for (nibble, source_index) in INTERLEAVE.into_iter().enumerate() {
                        word |= values[source_index] << (nibble * 4);
                    }
                    marlin_q[tile_base + thread * 4 + warp] = word as i32;
                }
            }
        }
    }

    let mut marlin_scales = vec![half::bf16::ZERO; padded_groups * padded_n];
    let mut marlin_zp = vec![0i32; padded_groups * padded_n / 8];
    for group in 0..padded_groups {
        for block in 0..padded_n / 64 {
            for (destination, source_index) in SCALE_PERM.into_iter().enumerate() {
                let n = block * 64 + source_index;
                if let Some(row) = rows.get(n) {
                    if let Some(scale) = row.scales.get(group) {
                        marlin_scales[group * padded_n + block * 64 + destination] = *scale;
                    }
                }
            }
            for word_index in 0..8 {
                let mut word = 0u32;
                for (nibble, interleaved) in INTERLEAVE.into_iter().enumerate() {
                    let source_n = block * 64 + SCALE_PERM[word_index * 8 + interleaved];
                    let zero_point = rows.get(source_n)
                        .and_then(|row| row.zero_points.get(group))
                        .copied()
                        .unwrap_or(0);
                    word |= u32::from(zero_point) << (nibble * 4);
                }
                marlin_zp[group * (padded_n / 8) + block * 8 + word_index] = word as i32;
            }
        }
    }

    let packed_bytes = marlin_q.len() * 4;
    let scale_bytes = marlin_scales.len() * 2;
    let zero_point_bytes = marlin_zp.len() * 4;
    let metadata = W4LayoutMetadata {
        layout: W4KernelLayout::MarlinAwqU4G32V1,
        version: 1,
        logical_rows: logical_n,
        logical_cols: logical_k,
        padded_rows: padded_n,
        padded_cols: padded_k,
        group_size: GROUP_SIZE,
        groups: padded_groups,
        packed_bytes,
        scale_bytes,
        zero_point_bytes,
        total_bytes: packed_bytes + scale_bytes + zero_point_bytes,
        source_total_bytes: dense.size_in_bytes(),
    };
    let packed = Tensor::from_raw(
        vec![padded_k / 16, padded_n * 2].into(), DType::I32,
        apxinf_core::Device::Cpu, bytemuck::cast_slice(&marlin_q).to_vec(),
    ).map_err(|error| error.to_string())?;
    let scale = Tensor::from_bf16(vec![padded_groups, padded_n], &marlin_scales)
        .map_err(|error| error.to_string())?;
    let zero_point = Tensor::from_raw(
        vec![padded_groups, padded_n / 8].into(), DType::I32,
        apxinf_core::Device::Cpu, bytemuck::cast_slice(&marlin_zp).to_vec(),
    ).map_err(|error| error.to_string())?;
    Ok(Some(RepackedW4Weight { packed, scale, zero_point, metadata }))
}

/// Dequantize a group-wise asymmetric INT4 weight to BF16.
///
/// This is a correctness/reference path for loading and validation. Fast CUDA
/// inference should consume the packed tensors directly or fuse this operation
/// with GEMM; materializing full BF16 weights is too memory-heavy for the target
/// 27B checkpoint.
pub fn dequantize_w4a16_grouped(
    tensors: &HashMap<String, Tensor>,
    prefix: &str,
    group_size: usize,
) -> Result<Tensor, String> {
    if group_size == 0 {
        return Err("group_size must be greater than zero".into());
    }
    let packed = get(tensors, prefix, "weight_packed")?;
    let scale = get(tensors, prefix, "weight_scale")?;
    let zero_point = get(tensors, prefix, "weight_zero_point")?;
    let shape = get(tensors, prefix, "weight_shape")?;

    expect_dtype(packed, DType::I32, "weight_packed")?;
    expect_dtype(scale, DType::BF16, "weight_scale")?;
    expect_dtype(zero_point, DType::I32, "weight_zero_point")?;
    expect_dtype(shape, DType::I64, "weight_shape")?;

    let logical_shape = shape.as_i64().map_err(|error| error.to_string())?;
    if logical_shape.len() != 2 || logical_shape.iter().any(|&dim| dim <= 0) {
        return Err(format!(
            "{prefix}.weight_shape must contain two positive dimensions, got {logical_shape:?}"
        ));
    }
    let rows = logical_shape[0] as usize;
    let cols = logical_shape[1] as usize;
    let packed_cols = cols.div_ceil(8);
    let groups = cols.div_ceil(group_size);
    let zp_rows = rows.div_ceil(8);

    if packed.shape().dims() != [rows, packed_cols] {
        return Err(format!(
            "{prefix}.weight_packed shape {:?} does not match expected [{rows}, {packed_cols}]",
            packed.shape().dims()
        ));
    }
    if scale.shape().dims() != [rows, groups] {
        return Err(format!(
            "{prefix}.weight_scale shape {:?} does not match expected [{rows}, {groups}]",
            scale.shape().dims()
        ));
    }
    if zero_point.shape().dims() != [zp_rows, groups] {
        return Err(format!(
            "{prefix}.weight_zero_point shape {:?} does not match expected [{zp_rows}, {groups}]",
            zero_point.shape().dims()
        ));
    }

    let packed = packed.as_i32().map_err(|error| error.to_string())?;
    let scales = scale.as_bf16().map_err(|error| error.to_string())?;
    let zero_points = zero_point.as_i32().map_err(|error| error.to_string())?;
    let mut output = Vec::with_capacity(rows * cols);

    for row in 0..rows {
        let zp_row = row / 8;
        let zp_shift = (row % 8) * 4;
        for col in 0..cols {
            let group = col / group_size;
            let word = packed[row * packed_cols + col / 8] as u32;
            let q = ((word >> ((col % 8) * 4)) & 0xF) as i32;
            let zp_word = zero_points[zp_row * groups + group] as u32;
            let zp = ((zp_word >> zp_shift) & 0xF) as i32;
            let value = (q - zp) as f32 * scales[row * groups + group].to_f32();
            output.push(half::bf16::from_f32(value));
        }
    }

    Tensor::from_bf16(vec![rows, cols], &output).map_err(|error| error.to_string())
}

fn get<'a>(
    tensors: &'a HashMap<String, Tensor>,
    prefix: &str,
    suffix: &str,
) -> Result<&'a Tensor, String> {
    let name = format!("{prefix}.{suffix}");
    tensors
        .get(&name)
        .ok_or_else(|| format!("missing compressed-tensors entry {name}"))
}

fn expect_dtype(tensor: &Tensor, expected: DType, suffix: &str) -> Result<(), String> {
    let actual = tensor.dtype();
    if actual != expected {
        return Err(format!(
            "{suffix} must be {expected}, got {actual}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::Shape;

    fn tensor_i32(shape: &[usize], values: &[i32]) -> Tensor {
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        Tensor::from_raw(Shape::from(shape.to_vec()), DType::I32, apxinf_core::Device::Cpu, bytes)
            .unwrap()
    }

    fn tensor_i64(shape: &[usize], values: &[i64]) -> Tensor {
        let bytes = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        Tensor::from_raw(Shape::from(shape.to_vec()), DType::I64, apxinf_core::Device::Cpu, bytes)
            .unwrap()
    }

    fn pack_nibbles(values: &[u8]) -> i32 {
        assert!(values.len() <= 8);
        values
            .iter()
            .enumerate()
            .fold(0u32, |word, (index, value)| {
                word | (((value & 0xF) as u32) << (index * 4))
            }) as i32
    }

    #[test]
    fn dequantizes_grouped_int4_weight() {
        let prefix = "layer.q_proj";
        let mut tensors = HashMap::new();
        let packed_rows = [
            pack_nibbles(&[1, 2, 3, 4, 5, 6, 7, 8]),
            pack_nibbles(&[2, 3, 4, 5, 6, 7, 8, 9]),
            pack_nibbles(&[3, 4, 5, 6, 7, 8, 9, 10]),
            pack_nibbles(&[4, 5, 6, 7, 8, 9, 10, 11]),
            pack_nibbles(&[5, 6, 7, 8, 9, 10, 11, 12]),
            pack_nibbles(&[6, 7, 8, 9, 10, 11, 12, 13]),
            pack_nibbles(&[7, 8, 9, 10, 11, 12, 13, 14]),
            pack_nibbles(&[8, 9, 10, 11, 12, 13, 14, 15]),
        ];
        tensors.insert(format!("{prefix}.weight_packed"), tensor_i32(&[8, 1], &packed_rows));
        tensors.insert(
            format!("{prefix}.weight_scale"),
            Tensor::from_bf16(vec![8, 1], &[half::bf16::from_f32(0.5); 8]).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.weight_zero_point"),
            tensor_i32(&[1, 1], &[pack_nibbles(&[1, 2, 3, 4, 5, 6, 7, 8])]),
        );
        tensors.insert(format!("{prefix}.weight_shape"), tensor_i64(&[2], &[8, 8]));

        let dequantized = dequantize_w4a16_grouped(&tensors, prefix, 8).unwrap();
        assert_eq!(dequantized.shape().dims(), &[8, 8]);
        let values = dequantized.to_f32_vec().unwrap();
        assert_eq!(values[0], 0.0);
        assert_eq!(values[1], 0.5);
        assert_eq!(values[8], 0.0);
        assert_eq!(values[15], 3.5);
    }

    #[test]
    fn repacks_n64_k16_v1_without_changing_quantized_values() {
        let prefix = "layer.proj";
        let rows = 64;
        let cols = 128;
        let groups = 4;
        let packed_cols = cols / 8;
        let packed: Vec<i32> = (0..rows * packed_cols)
            .map(|index| 0x1111_1111u32.wrapping_mul(index as u32 + 1) as i32)
            .collect();
        let scales: Vec<half::bf16> = (0..rows * groups)
            .map(|index| half::bf16::from_f32(index as f32 / 256.0 + 0.5))
            .collect();
        let zp: Vec<i32> = (0..rows / 8 * groups)
            .map(|index| 0x0123_4567u32.rotate_left(index as u32) as i32)
            .collect();
        let mut tensors = HashMap::new();
        tensors.insert(format!("{prefix}.weight_packed"), tensor_i32(&[rows, packed_cols], &packed));
        tensors.insert(format!("{prefix}.weight_scale"), Tensor::from_bf16(vec![rows, groups], &scales).unwrap());
        tensors.insert(format!("{prefix}.weight_zero_point"), tensor_i32(&[rows / 8, groups], &zp));
        tensors.insert(format!("{prefix}.weight_shape"), tensor_i64(&[2], &[rows as i64, cols as i64]));

        let result = repack_w4_n64_k16_v1(&tensors, prefix).unwrap().unwrap();
        assert_eq!(result.metadata.layout, W4KernelLayout::RepackedN64K16V1);
        assert_eq!(result.metadata.additional_device_bytes(), 0);
        let q = result.packed.as_i32().unwrap();
        let s = result.scale.as_bf16().unwrap();
        let z = result.zero_point.as_i32().unwrap();
        for row in 0..rows {
            for word in 0..packed_cols {
                let k_tile = word / 2;
                assert_eq!(q[(k_tile * 64 + row) * 2 + word % 2], packed[row * packed_cols + word]);
            }
            for group in 0..groups {
                assert_eq!(s[group * 64 + row], scales[row * groups + group]);
            }
        }
        for row_pack in 0..rows / 8 {
            for group in 0..groups {
                assert_eq!(z[group * 8 + row_pack], zp[row_pack * groups + group]);
            }
        }
    }

    #[test]
    fn repacks_exact_marlin_awq_layout_with_padding() {
        let prefix = "layer.marlin";
        let n = 72;
        let k = 32;
        let groups = 1;
        let mut qwords = Vec::with_capacity(n * k / 8);
        for row in 0..n {
            for word in 0..k / 8 {
                let values: Vec<u8> = (0..8)
                    .map(|nibble| ((row + word * 8 + nibble) & 0xF) as u8)
                    .collect();
                qwords.push(pack_nibbles(&values));
            }
        }
        let scales: Vec<half::bf16> = (0..n)
            .map(|row| half::bf16::from_f32(row as f32 + 1.0))
            .collect();
        let zps: Vec<i32> = (0..n / 8)
            .map(|word| {
                let values: Vec<u8> = (0..8).map(|i| ((word * 8 + i) & 0xF) as u8).collect();
                pack_nibbles(&values)
            })
            .collect();
        let mut tensors = HashMap::new();
        tensors.insert(format!("{prefix}.weight_packed"), tensor_i32(&[n, k / 8], &qwords));
        tensors.insert(format!("{prefix}.weight_scale"), Tensor::from_bf16(vec![n, groups], &scales).unwrap());
        tensors.insert(format!("{prefix}.weight_zero_point"), tensor_i32(&[n / 8, groups], &zps));
        tensors.insert(format!("{prefix}.weight_shape"), tensor_i64(&[2], &[n as i64, k as i64]));

        let result = repack_w4_marlin_awq_u4_g32_v1(&tensors, prefix).unwrap().unwrap();
        assert_eq!(W4_MARLIN_AWQ_U4_G32_V1_SUFFIX, "marlin_awq_u4_g32_v1");
        assert_eq!(result.metadata.layout, W4KernelLayout::MarlinAwqU4G32V1);
        assert_eq!((result.metadata.logical_rows, result.metadata.logical_cols), (72, 32));
        assert_eq!((result.metadata.padded_rows, result.metadata.padded_cols), (128, 64));
        assert_eq!(result.metadata.groups, 2);
        assert_eq!(result.packed.shape().dims(), &[4, 256]);
        assert_eq!(result.scale.shape().dims(), &[2, 128]);
        assert_eq!(result.zero_point.shape().dims(), &[2, 16]);
        assert_eq!(result.metadata.packed_bytes, 4096);
        assert_eq!(result.metadata.scale_bytes, 512);
        assert_eq!(result.metadata.zero_point_bytes, 128);
        assert_eq!(result.metadata.total_bytes, result.metadata.device_bytes());
        assert_eq!(result.metadata.total_bytes, 4736);
        assert_eq!(result.metadata.source_total_bytes, 1332);

        let q = result.packed.as_i32().unwrap();
        let expected_first = pack_nibbles(&[0, 8, 8, 0, 1, 9, 9, 1]);
        assert_eq!(q[0], expected_first);
        assert_eq!(q[512], pack_nibbles(&[0; 8]));
        let s = result.scale.as_bf16().unwrap();
        assert_eq!(s[0], scales[0]);
        assert_eq!(s[1], scales[8]);
        assert_eq!(s[8], scales[1]);
        assert_eq!(s[73], half::bf16::ZERO);
        assert!(s[128..].iter().all(|value| *value == half::bf16::ZERO));
        let zp = result.zero_point.as_i32().unwrap();
        assert_eq!(zp[0], pack_nibbles(&[0, 0, 0, 0, 8, 8, 8, 8]));
        assert!(zp[16..].iter().all(|&word| word == 0));
    }
}

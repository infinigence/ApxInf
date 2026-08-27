//! CUDA backend implementing the Backend trait.

use std::sync::{Mutex, OnceLock};

use apxinf_core::{Backend, DType, Device, Error, Graph, KvCache, Result, Shape, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::cublas::CublasHandle;
use crate::kernels;
use crate::transfers;
use crate::workspace::output_buffer;
use crate::CudaKVCache;

struct CudaGraph {
    graph: crate::graph::CapturedGraph,
}

impl Graph for CudaGraph {
    fn replay(&self) -> Result<()> {
        self.graph.replay().map_err(Error::Cuda)
    }
}

/// CUDA backend — all ops execute on GPU via cuBLAS + custom kernels.
///
/// Implements the portable `Backend` trait. Also provides CUDA-specific
/// extension methods via `CudaBackend` directly.
pub struct CudaBackend {
    tmrope_positions: Mutex<Option<TmropePositionCache>>,
    vision_positions: Mutex<Option<TmropePositionCache>>,
    vision_groups: Mutex<Option<VisionGroupCache>>,
    ctx: CudaContext,
}

struct TmropePositionCache {
    values: Vec<u32>,
    _source_bytes: Vec<u8>,
    buffer: CudaBuffer,
}

struct VisionGroupCache {
    values: Vec<u32>,
    _group_source_bytes: Vec<u8>,
    _offset_source_bytes: Vec<u8>,
    _index_source_bytes: Vec<u8>,
    groups: CudaBuffer,
    offsets: CudaBuffer,
    indices: CudaBuffer,
    group_count: usize,
    max_group_size: usize,
}

fn tmrope_position_cache_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| match std::env::var("APXINF_TMROPE_POSITION_CACHE") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            Ok(value) => Err(format!(
                "APXINF_TMROPE_POSITION_CACHE must be 0 or 1, got `{value}`"
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("APXINF_TMROPE_POSITION_CACHE must be UTF-8".into())
            }
        })
        .clone()
        .map_err(Error::Other)
}

fn tmrope_prefill_position_cache_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(
            || match std::env::var("APXINF_TMROPE_POSITION_CACHE_PREFILL") {
                Err(std::env::VarError::NotPresent) => Ok(false),
                Ok(value) if value == "0" => Ok(false),
                Ok(value) if value == "1" => Ok(true),
                Ok(value) => Err(format!(
                    "APXINF_TMROPE_POSITION_CACHE_PREFILL must be 0 or 1, got `{value}`"
                )),
                Err(std::env::VarError::NotUnicode(_)) => {
                    Err("APXINF_TMROPE_POSITION_CACHE_PREFILL must be UTF-8".into())
                }
            },
        )
        .clone()
        .map_err(Error::Other)
}

fn vision_grouped_sparse_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| match std::env::var("APXINF_VISION_GROUPED_SPARSE") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            Ok(value) => Err(format!(
                "APXINF_VISION_GROUPED_SPARSE must be 0 or 1, got `{value}`"
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("APXINF_VISION_GROUPED_SPARSE must be UTF-8".into())
            }
        })
        .clone()
        .map_err(Error::Other)
}

fn vision_grouped_fa2_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| match std::env::var("APXINF_VISION_GROUPED_FA2") {
            Err(std::env::VarError::NotPresent) => Ok(false),
            Ok(value) if value == "0" => Ok(false),
            Ok(value) if value == "1" => Ok(true),
            Ok(value) => Err(format!(
                "APXINF_VISION_GROUPED_FA2 must be 0 or 1, got `{value}`"
            )),
            Err(std::env::VarError::NotUnicode(_)) => {
                Err("APXINF_VISION_GROUPED_FA2 must be UTF-8".into())
            }
        })
        .clone()
        .map_err(Error::Other)
}

pub(crate) fn vision_group_plan(group_ids: &[u32]) -> Result<(Vec<u32>, Vec<u32>)> {
    if group_ids.is_empty() {
        return Err(Error::Other("vision group plan is empty".into()));
    }
    let max_group = *group_ids
        .iter()
        .max()
        .ok_or_else(|| Error::Other("vision group plan is empty".into()))?;
    let group_count = usize::try_from(max_group)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| Error::Other("vision group count overflow".into()))?;
    if group_count > group_ids.len() {
        return Err(Error::Other(format!(
            "vision group count {group_count} exceeds sequence {}",
            group_ids.len()
        )));
    }
    let mut counts = vec![0usize; group_count];
    for &group in group_ids {
        let group = group as usize;
        counts[group] = counts[group]
            .checked_add(1)
            .ok_or_else(|| Error::Other("vision group size overflow".into()))?;
    }
    let mut offsets = Vec::with_capacity(group_count + 1);
    offsets.push(0u32);
    let mut total = 0usize;
    for count in counts {
        total = total
            .checked_add(count)
            .ok_or_else(|| Error::Other("vision group offset overflow".into()))?;
        offsets.push(
            u32::try_from(total)
                .map_err(|_| Error::Other("vision group offset exceeds u32".into()))?,
        );
    }
    let mut cursors = offsets[..group_count]
        .iter()
        .map(|value| *value as usize)
        .collect::<Vec<_>>();
    let mut indices = vec![0u32; group_ids.len()];
    for (index, &group) in group_ids.iter().enumerate() {
        let group = group as usize;
        let cursor = cursors[group];
        indices[cursor] = u32::try_from(index)
            .map_err(|_| Error::Other("vision key index exceeds u32".into()))?;
        cursors[group] += 1;
    }
    Ok((offsets, indices))
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

impl CudaBackend {
    /// Create a CUDA backend for the given device.
    pub fn new(device_id: usize) -> Result<Self> {
        let ctx = CudaContext::new(device_id).map_err(Error::Cuda)?;
        eprintln!(
            "CUDA {}: {} (compute {}.{}, {}, {} SMs)",
            device_id,
            ctx.caps().device_name,
            ctx.caps().compute_major,
            ctx.caps().compute_minor,
            ctx.caps().arch_family,
            ctx.caps().multiprocessor_count,
        );
        Ok(Self {
            tmrope_positions: Mutex::new(None),
            vision_positions: Mutex::new(None),
            vision_groups: Mutex::new(None),
            ctx,
        })
    }

    /// Access the CUDA context.
    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Access the cuBLAS handle.
    pub fn cublas(&self) -> &CublasHandle {
        self.ctx.cublas()
    }

    // ── CUDA-specific extensions (not in Backend trait) ──────────────

    /// Get the device ID.
    pub fn device_id(&self) -> usize {
        self.ctx.device_id()
    }

    fn vision_position_cache(
        &self,
        positions: &[u32],
        sequence: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<TmropePositionCache>>> {
        if positions.len() != sequence * 2 {
            return Err(Error::Other(format!(
                "Qwen2.5-Omni vision position length {} != {sequence} * 2",
                positions.len()
            )));
        }
        let mut cache = self
            .vision_positions
            .lock()
            .map_err(|_| Error::Other("vision position cache lock poisoned".into()))?;
        if cache
            .as_ref()
            .is_none_or(|cached| cached.values.as_slice() != positions)
        {
            let bytes = u32_bytes(positions);
            let buffer = CudaBuffer::alloc_stream_ordered(
                bytes.len(),
                self.ctx.device_id(),
                self.ctx.stream(),
            )
            .map_err(Error::Cuda)?;
            buffer
                .copy_from_host_async(&bytes, self.ctx.stream())
                .map_err(Error::Cuda)?;
            *cache = Some(TmropePositionCache {
                values: positions.to_vec(),
                _source_bytes: bytes,
                buffer,
            });
        }
        Ok(cache)
    }

    fn tmrope_position_cache(
        &self,
        positions: &[u32],
        sequence: usize,
    ) -> Result<std::sync::MutexGuard<'_, Option<TmropePositionCache>>> {
        let expected = sequence
            .checked_mul(3)
            .ok_or_else(|| Error::Other("Qwen2.5-Omni TMRoPE position size overflow".into()))?;
        if positions.len() != expected {
            return Err(Error::Other(format!(
                "Qwen2.5-Omni TMRoPE position length {} != {sequence} * 3",
                positions.len()
            )));
        }
        let mut cache = self
            .tmrope_positions
            .lock()
            .map_err(|_| Error::Other("TMRoPE position cache lock poisoned".into()))?;
        if cache
            .as_ref()
            .is_none_or(|cached| cached.values.as_slice() != positions)
        {
            let bytes = u32_bytes(positions);
            let buffer = CudaBuffer::alloc_stream_ordered(
                bytes.len(),
                self.ctx.device_id(),
                self.ctx.stream(),
            )
            .map_err(Error::Cuda)?;
            buffer
                .copy_from_host_async(&bytes, self.ctx.stream())
                .map_err(Error::Cuda)?;
            *cache = Some(TmropePositionCache {
                values: positions.to_vec(),
                _source_bytes: bytes,
                buffer,
            });
        }
        Ok(cache)
    }

    fn vision_group_cache(
        &self,
        group_ids: &[u32],
        grouped_fa2: bool,
    ) -> Result<std::sync::MutexGuard<'_, Option<VisionGroupCache>>> {
        let mut cache = self
            .vision_groups
            .lock()
            .map_err(|_| Error::Other("vision group cache lock poisoned".into()))?;
        if cache
            .as_ref()
            .is_none_or(|cached| cached.values.as_slice() != group_ids)
        {
            let (offset_values, index_values) = vision_group_plan(group_ids)?;
            let group_count = offset_values.len() - 1;
            let max_group_size = offset_values
                .windows(2)
                .map(|window| (window[1] - window[0]) as usize)
                .max()
                .ok_or_else(|| Error::Other("vision group plan has no groups".into()))?;
            if grouped_fa2
                && offset_values
                    .windows(2)
                    .any(|window| window[0] >= window[1])
            {
                return Err(Error::Other(
                    "grouped vision FA2 requires nonempty contiguous groups".into(),
                ));
            }
            if grouped_fa2 {
                eprintln!(
                    "ApxInf grouped vision FA2: {group_count} groups, max {max_group_size} tokens, total {}",
                    group_ids.len()
                );
            }
            let group_bytes = u32_bytes(group_ids);
            let offset_bytes = u32_bytes(&offset_values);
            let index_bytes = u32_bytes(&index_values);
            let groups = CudaBuffer::alloc_stream_ordered(
                group_bytes.len(),
                self.ctx.device_id(),
                self.ctx.stream(),
            )
            .map_err(Error::Cuda)?;
            let offsets = CudaBuffer::alloc_stream_ordered(
                offset_bytes.len(),
                self.ctx.device_id(),
                self.ctx.stream(),
            )
            .map_err(Error::Cuda)?;
            let indices = CudaBuffer::alloc_stream_ordered(
                index_bytes.len(),
                self.ctx.device_id(),
                self.ctx.stream(),
            )
            .map_err(Error::Cuda)?;
            groups
                .copy_from_host_async(&group_bytes, self.ctx.stream())
                .map_err(Error::Cuda)?;
            offsets
                .copy_from_host_async(&offset_bytes, self.ctx.stream())
                .map_err(Error::Cuda)?;
            indices
                .copy_from_host_async(&index_bytes, self.ctx.stream())
                .map_err(Error::Cuda)?;
            *cache = Some(VisionGroupCache {
                values: group_ids.to_vec(),
                _group_source_bytes: group_bytes,
                _offset_source_bytes: offset_bytes,
                _index_source_bytes: index_bytes,
                groups,
                offsets,
                indices,
                group_count,
                max_group_size,
            });
        }
        Ok(cache)
    }

    pub fn qwen25_omni_vision_bias_residual_exact(
        &self,
        projection: &Tensor,
        bias: &Tensor,
        residual: &Tensor,
    ) -> Result<Tensor> {
        kernels::qwen25_omni_vision::bias_residual_exact(
            &self.ctx,
            projection,
            bias,
            residual,
        )
    }

    pub fn qwen25_omni_vision_gate_up_bias_silu_mul_exact(
        &self,
        gate: &Tensor,
        gate_bias: &Tensor,
        up: &Tensor,
        up_bias: &Tensor,
    ) -> Result<Tensor> {
        kernels::qwen25_omni_vision::gate_up_bias_silu_mul_exact(
            &self.ctx,
            gate,
            gate_bias,
            up,
            up_bias,
        )
    }

    pub fn qwen25_omni_vision_packed_gate_up_bias_silu_mul_exact(
        &self,
        packed_gate_up: &Tensor,
        gate_bias: &Tensor,
        up_bias: &Tensor,
    ) -> Result<Tensor> {
        kernels::qwen25_omni_vision::packed_gate_up_bias_silu_mul_exact(
            &self.ctx,
            packed_gate_up,
            gate_bias,
            up_bias,
        )
    }

    pub fn qwen25_omni_prefill_packed_silu_mul_exact(
        &self,
        gate_up: &Tensor,
        intermediate: usize,
    ) -> Result<Tensor> {
        kernels::activation::silu_mul_packed_rows_exact(&self.ctx, gate_up, intermediate)
    }

    pub fn qwen25_omni_prefill_packed_qkv_prelude(
        &self,
        packed_qkv: &Tensor,
        bias: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        positions: &[u32],
    ) -> Result<Tensor> {
        let rows = packed_qkv.shape().dims().first().copied().unwrap_or(0);
        if !tmrope_position_cache_enabled()? || !tmrope_prefill_position_cache_enabled()? {
            return Err(Error::Other(
                "Qwen2.5-Omni prefill packed QKV prelude requires cached prefill positions"
                    .into(),
            ));
        }
        if layer_idx >= kv.n_layers() {
            return Err(Error::Other(format!(
                "Qwen2.5-Omni prefill packed QKV layer {layer_idx} exceeds cache depth {}",
                kv.n_layers()
            )));
        }
        let start_position = kv.seq_len();
        let position_cache = self.tmrope_position_cache(positions, rows)?;
        let position_buffer = &position_cache
            .as_ref()
            .ok_or_else(|| Error::Other("TMRoPE position cache is missing".into()))?
            .buffer;
        let cache = kv
            .as_any()
            .downcast_ref::<CudaKVCache>()
            .ok_or_else(|| Error::Other("prefill packed QKV requires CudaKVCache".into()))?;
        kernels::qwen25_omni_attention::prefill_packed_qkv_prelude_write(
            &self.ctx,
            packed_qkv,
            bias,
            cache.k_buffer(layer_idx),
            cache.v_buffer(layer_idx),
            position_buffer,
            start_position,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qwen25_omni_vision_qkv_bias_rope(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        query_bias: &Tensor,
        key_bias: &Tensor,
        value_bias: &Tensor,
        theta: f32,
        positions: &[u32],
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let sequence = query.shape().dims().first().copied().unwrap_or(0);
        let cache = self.vision_position_cache(positions, sequence)?;
        kernels::qwen25_omni_vision::qkv_bias_rope(
            &self.ctx,
            query,
            key,
            value,
            query_bias,
            key_bias,
            value_bias,
            theta,
            &cache.as_ref().expect("vision position cache populated").buffer,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qwen25_omni_vision_grouped_qkv_bias_rope(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        query_bias: &Tensor,
        key_bias: &Tensor,
        value_bias: &Tensor,
        theta: f32,
        positions: &[u32],
        group_ids: &[u32],
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let sequence = query.shape().dims().first().copied().unwrap_or(0);
        if group_ids.len() != sequence {
            return Err(Error::Other(format!(
                "Qwen2.5-Omni vision group length {} != sequence {sequence}",
                group_ids.len()
            )));
        }
        let groups = self.vision_group_cache(group_ids, true)?;
        let positions = self.vision_position_cache(positions, sequence)?;
        kernels::qwen25_omni_vision::grouped_qkv_bias_rope(
            &self.ctx,
            query,
            key,
            value,
            query_bias,
            key_bias,
            value_bias,
            theta,
            &positions
                .as_ref()
                .expect("vision position cache populated")
                .buffer,
            &groups
                .as_ref()
                .expect("vision group cache populated")
                .indices,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qwen25_omni_vision_packed_qkv_bias_rope(
        &self,
        packed_qkv: &Tensor,
        query_bias: &Tensor,
        key_bias: &Tensor,
        value_bias: &Tensor,
        theta: f32,
        positions: &[u32],
        group_ids: Option<&[u32]>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let sequence = packed_qkv
            .shape()
            .dims()
            .first()
            .copied()
            .unwrap_or(0);
        if group_ids.is_some_and(|groups| groups.len() != sequence) {
            return Err(Error::Other(
                "Qwen2.5-Omni packed vision QKV group length mismatch".into(),
            ));
        }
        let groups = group_ids
            .map(|groups| self.vision_group_cache(groups, true))
            .transpose()?;
        let positions = self.vision_position_cache(positions, sequence)?;
        kernels::qwen25_omni_vision::packed_qkv_bias_rope(
            &self.ctx,
            packed_qkv,
            query_bias,
            key_bias,
            value_bias,
            theta,
            &positions
                .as_ref()
                .expect("vision position cache populated")
                .buffer,
            groups.as_ref().map(|groups| {
                &groups
                    .as_ref()
                    .expect("vision group cache populated")
                    .indices
            }),
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qwen25_omni_vision_grouped_sdpa_prepacked(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        sequence: usize,
        heads: usize,
        head_dim: usize,
        group_ids: &[u32],
    ) -> Result<Tensor> {
        if !vision_grouped_sparse_enabled()? || !vision_grouped_fa2_enabled()? {
            return Err(Error::Other(
                "prepacked grouped vision attention requires accepted grouped FA2 selectors"
                    .into(),
            ));
        }
        if group_ids.len() != sequence {
            return Err(Error::Other(format!(
                "prepacked grouped vision attention has {} groups for {sequence} tokens",
                group_ids.len()
            )));
        }
        let cache = self.vision_group_cache(group_ids, true)?;
        let cached = cache.as_ref().expect("vision group cache populated");
        #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_vision_sm80))]
        {
            kernels::attention::grouped_varlen_fa2(
                &self.ctx,
                query,
                key,
                value,
                sequence,
                heads,
                head_dim,
                &cached.offsets,
                &cached.indices,
                cached.group_count,
                cached.max_group_size,
                true,
            )
        }
        #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_vision_sm80)))]
        {
            Err(Error::Other(
                "prepacked grouped vision attention requires an SM80-family FA2 build".into(),
            ))
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn qwen25_omni_grouped_split_cta_decode(
        &self,
        query: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        workspace: &kernels::qwen25_omni_attention::SplitCtaWorkspace,
        split_count: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let cache = kv
            .as_any()
            .downcast_ref::<CudaKVCache>()
            .ok_or_else(|| Error::Other("Qwen2.5-Omni split-CTA expects CudaKVCache".into()))?;
        let positions = self
            .tmrope_positions
            .lock()
            .map_err(|_| Error::Other("TMRoPE position cache lock poisoned".into()))?;
        let positions = positions.as_ref().ok_or_else(|| {
            Error::Other("Qwen2.5-Omni split-CTA requires cached TMRoPE position".into())
        })?;
        if positions.values.first().copied().map(|value| value as usize + 1) != Some(kv_len) {
            return Err(Error::Other(format!(
                "Qwen2.5-Omni split-CTA position does not own KV length {kv_len}"
            )));
        }
        let output = output_buffer(
            &self.ctx,
            kernels::qwen25_omni_attention::WIDTH * DType::BF16.size_in_bytes(),
        )?
        .into_tensor(
            Shape::new(vec![1, kernels::qwen25_omni_attention::WIDTH]),
            DType::BF16,
        );
        kernels::qwen25_omni_attention::grouped4_split_cta_write(
            &self.ctx,
            query,
            cache.k_buffer(layer_idx),
            cache.v_buffer(layer_idx),
            &output,
            workspace,
            split_count,
            kv_len,
            max_seq_len,
            (kernels::qwen25_omni_attention::HEAD_DIM as f32)
                .sqrt()
                .recip(),
            positions.buffer.address(),
        )?;
        Ok(output)
    }

    /// Execute suffix prefill through causal FA2 without the default long-KV
    /// scheduling threshold. The model owns the request-level eligibility
    /// gate; the CUDA attention contract still validates dtype, shape, cache,
    /// suffix position and the explicit process selector.
    #[allow(clippy::too_many_arguments)]
    pub fn sdpa_prefill_causal_fa2(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let seq_len = q.shape().dims()[0];
        let kv_offset = kv_len
            .checked_sub(seq_len)
            .ok_or_else(|| Error::Other("causal FA2 query exceeds KV length".into()))?;
        kernels::attention::sdpa_with_batched_prefill_fa2_min_kv(
            &self.ctx,
            q,
            kv,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            max_seq_len,
            u32::try_from(kv_offset)
                .map_err(|_| Error::Other("causal FA2 KV offset exceeds u32".into()))?,
            true,
            1,
            false,
        )
    }

    /// Execute the pinned Qwen2.5-Omni short prefill path with one exact
    /// scaled numerator-cache softmax owner instead of materializing scaled
    /// scores before softmax.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen25_omni_sdpa_prefill_scaled_exp_cache(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let dims = q.shape().dims();
        if q.dtype() != DType::BF16
            || q.device() != Device::Cuda(self.ctx.device_id())
            || dims.len() != 3
            || !(2..=4_096).contains(&dims[0])
            || dims[1] != 16
            || dims[2] != 128
            || n_heads != 16
            || n_kv_heads != 2
            || head_dim != 128
            || kv_len > 4_096
            || max_seq_len != 32_768
        {
            return Err(Error::Other(
                "Qwen2.5-Omni scaled exp-cache prefill contract mismatch".into(),
            ));
        }
        let kv_offset = kv_len
            .checked_sub(dims[0])
            .ok_or_else(|| Error::Other("scaled exp-cache query exceeds KV length".into()))?;
        kernels::attention::sdpa_with_batched_prefill_fa2_min_kv(
            &self.ctx,
            q,
            kv,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            max_seq_len,
            u32::try_from(kv_offset)
                .map_err(|_| Error::Other("scaled exp-cache KV offset exceeds u32".into()))?,
            true,
            4_097,
            true,
        )
    }

    /// Begin a relaxed stream capture for decode graphs which call vendor
    /// libraries with internal thread-local state.
    pub fn begin_capture_relaxed(&self) -> Result<()> {
        crate::graph::begin(&self.ctx, crate::graph::CaptureMode::Relaxed).map_err(Error::Cuda)
    }
}

impl Backend for CudaBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        kernels::norm::rms(&self.ctx, input, weight, eps)
    }

    fn silu(&self, x: &Tensor) -> Result<Tensor> {
        kernels::activation::silu(&self.ctx, x)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        kernels::elementwise::add(&self.ctx, a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        kernels::elementwise::mul(&self.ctx, a, b)
    }

    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        kernels::activation::silu_mul(&self.ctx, gate, up)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        kernels::elementwise::scale(&self.ctx, input, factor)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        kernels::gemm::matmul(&self.ctx, a, b)
    }

    fn rope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_offset: u32,
    ) -> Result<Tensor> {
        kernels::rope::apply_batched(&self.ctx, input, n_heads, head_dim, theta, pos_offset)
    }

    fn rope_mrope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        sections: [usize; 3],
        pos_ids: &[u32],
    ) -> Result<Tensor> {
        let dims = input.shape().dims();
        let seq_len = if dims.len() == 2 { 1 } else { dims[0] };
        if pos_ids.len() != seq_len * 3 {
            return Err(Error::Other(format!(
                "rope_mrope: pos_ids len {} != seq_len {} * 3",
                pos_ids.len(),
                seq_len
            )));
        }
        let ids_bytes: Vec<u8> = pos_ids.iter().flat_map(|&v| v.to_ne_bytes()).collect();
        let ids_buf =
            CudaBuffer::alloc(ids_bytes.len(), self.ctx.device_id()).map_err(Error::Cuda)?;
        ids_buf.copy_from_host(&ids_bytes).map_err(Error::Cuda)?;
        kernels::rope::apply_mrope(
            &self.ctx, input, n_heads, head_dim, theta, sections, &ids_buf,
        )
    }

    fn rope_tmrope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        sections: [usize; 3],
        pos_ids: &[u32],
    ) -> Result<Tensor> {
        let dims = input.shape().dims();
        let seq_len = if dims.len() == 2 { 1 } else { dims[0] };
        if pos_ids.len() != seq_len * 3 {
            return Err(Error::Other(format!(
                "rope_tmrope: pos_ids len {} != seq_len {} * 3",
                pos_ids.len(),
                seq_len
            )));
        }
        let cache_positions = tmrope_position_cache_enabled()?
            && (seq_len == 1 || tmrope_prefill_position_cache_enabled()?);
        if !cache_positions {
            let bytes = pos_ids
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect::<Vec<_>>();
            let ids = CudaBuffer::alloc(bytes.len(), self.ctx.device_id()).map_err(Error::Cuda)?;
            ids.copy_from_host(&bytes).map_err(Error::Cuda)?;
            return kernels::rope::apply_tmrope(
                &self.ctx, input, n_heads, head_dim, theta, sections, &ids,
            );
        }

        let cache = self.tmrope_position_cache(pos_ids, seq_len)?;
        let positions = &cache.as_ref().expect("position cache populated").buffer;
        kernels::rope::apply_tmrope(
            &self.ctx, input, n_heads, head_dim, theta, sections, positions,
        )
    }

    fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        kernels::norm::layer(&self.ctx, input, weight, bias, eps)
    }

    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        kernels::activation::gelu_tanh(&self.ctx, input)
    }

    fn add_bias(&self, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
        kernels::elementwise::add_bias(&self.ctx, input, bias)
    }

    fn rope_vision_2d(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_ids: &[u32],
    ) -> Result<Tensor> {
        let dims = input.shape().dims();
        let seq_len = if dims.len() == 2 { 1 } else { dims[0] };
        if pos_ids.len() != seq_len * 2 {
            return Err(Error::Other(format!(
                "rope_vision_2d: pos_ids len {} != seq_len {} * 2",
                pos_ids.len(),
                seq_len
            )));
        }
        let ids_bytes: Vec<u8> = pos_ids.iter().flat_map(|&v| v.to_ne_bytes()).collect();
        let ids_buf =
            CudaBuffer::alloc(ids_bytes.len(), self.ctx.device_id()).map_err(Error::Cuda)?;
        ids_buf.copy_from_host(&ids_bytes).map_err(Error::Cuda)?;
        kernels::rope::apply_vision_2d(&self.ctx, input, n_heads, head_dim, theta, &ids_buf)
    }

    fn vision_sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        kernels::attention::vision(&self.ctx, q, k, v, seq_len, n_heads, head_dim)
    }

    fn grouped_sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
        group_ids: &[u32],
    ) -> Result<Tensor> {
        if group_ids.len() != seq_len {
            return Err(Error::Other(format!(
                "grouped_sdpa: {} group ids for {seq_len} tokens",
                group_ids.len()
            )));
        }
        let grouped_sparse = vision_grouped_sparse_enabled()?;
        let grouped_fa2 = vision_grouped_fa2_enabled()?;
        if grouped_fa2 && !grouped_sparse {
            return Err(Error::Other(
                "APXINF_VISION_GROUPED_FA2 requires APXINF_VISION_GROUPED_SPARSE=1".into(),
            ));
        }
        if !grouped_sparse {
            let bytes = u32_bytes(group_ids);
            let ids = CudaBuffer::alloc(bytes.len(), self.ctx.device_id()).map_err(Error::Cuda)?;
            ids.copy_from_host(&bytes).map_err(Error::Cuda)?;
            return kernels::attention::grouped(
                &self.ctx, q, k, v, seq_len, n_heads, head_dim, &ids,
            );
        }

        let cache = self.vision_group_cache(group_ids, grouped_fa2)?;
        let cached = cache.as_ref().expect("vision group cache populated");
        if grouped_fa2 {
            #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_vision_sm80))]
            {
                return kernels::attention::grouped_varlen_fa2(
                    &self.ctx,
                    q,
                    k,
                    v,
                    seq_len,
                    n_heads,
                    head_dim,
                    &cached.offsets,
                    &cached.indices,
                    cached.group_count,
                    cached.max_group_size,
                    false,
                );
            }
            #[cfg(not(any(apxinf_fa2_sm80, apxinf_fa2_vision_sm80)))]
            {
                return Err(Error::Other(
                    "APXINF_VISION_GROUPED_FA2 requires an SM80-family FA2 build".into(),
                ));
            }
        }
        kernels::attention::grouped_indexed(
            &self.ctx,
            q,
            k,
            v,
            seq_len,
            n_heads,
            head_dim,
            &cached.groups,
            &cached.offsets,
            &cached.indices,
            cached.group_count,
        )
    }

    fn im2col1d(
        &self,
        input: &Tensor,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Tensor> {
        kernels::preprocess::im2col1d_bf16(&self.ctx, input, kernel, stride, padding)
    }

    fn avg_pool1d(&self, input: &Tensor, kernel: usize, stride: usize) -> Result<Tensor> {
        kernels::preprocess::avg_pool1d_bf16(&self.ctx, input, kernel, stride)
    }

    fn concat_2d(&self, tensors: &[&Tensor]) -> Result<Tensor> {
        use apxinf_core::Shape;
        if tensors.is_empty() {
            return Err(Error::Other("concat_2d: empty input".into()));
        }
        let device_id = self.ctx.device_id();
        let dtype = tensors[0].dtype();
        let elem = dtype.size_in_bytes();
        let dims0 = tensors[0].shape().dims();
        if dims0.len() != 2 {
            return Err(Error::Other(format!(
                "concat_2d: expected 2D, got {}D",
                dims0.len()
            )));
        }
        let rows = dims0[0];
        let total_cols: usize = tensors.iter().map(|t| t.shape().dims()[1]).sum();
        for t in tensors {
            let d = t.shape().dims();
            if d.len() != 2 || d[0] != rows || t.dtype() != dtype {
                return Err(Error::Other("concat_2d: shape/dtype mismatch".into()));
            }
        }
        let out_bytes = rows * total_cols * elem;
        let out_buf = CudaBuffer::alloc_zeros(out_bytes, device_id).map_err(Error::Cuda)?;
        let dst_pitch = total_cols * elem;
        let mut col_offset = 0usize;
        for t in tensors {
            let cols = t.shape().dims()[1];
            let width = cols * elem;
            let spitch = cols * elem;
            crate::transfers::copy_tensor_2d_to_buffer(
                &self.ctx,
                t,
                &out_buf,
                col_offset * elem,
                dst_pitch,
                spitch,
                width,
                rows,
            )?;
            col_offset += cols;
        }
        Ok(out_buf.into_tensor(Shape::new(vec![rows, total_cols]), dtype))
    }

    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        let device_id = self.ctx.device_id();
        let ids_bytes: Vec<u8> = ids.iter().flat_map(|&v| v.to_ne_bytes()).collect();
        let ids_buf = CudaBuffer::alloc(ids_bytes.len(), device_id).map_err(Error::Cuda)?;
        ids_buf.copy_from_host(&ids_bytes).map_err(Error::Cuda)?;

        kernels::embedding::lookup(&self.ctx, table, &ids_buf, ids.len())
    }

    fn sdpa_decode(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        // For decode (seq_len=1), the new token is at position kv_len-1
        // and must attend to all kv_len positions (including itself).
        // attention_softmax kernel computes valid_cols = seq_pos + kv_offset + 1.
        // With seq_pos=0, we need kv_offset = kv_len - 1.
        let kv_offset = (kv_len - 1) as u32;
        kernels::attention::sdpa(
            &self.ctx,
            q,
            kv,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            max_seq_len,
            kv_offset,
        )
    }

    fn sdpa_prefill(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let seq_len = q.shape().dims()[0];
        let kv_offset = kv_len - seq_len;
        kernels::attention::sdpa(
            &self.ctx,
            q,
            kv,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            max_seq_len,
            kv_offset as u32,
        )
    }

    fn create_kv_cache(
        &self,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Box<dyn KvCache> {
        Box::new(
            CudaKVCache::new(
                self.ctx.device_id(),
                n_layers,
                n_kv_heads,
                head_dim,
                max_seq_len,
            )
            .unwrap(),
        )
    }

    fn kv_append(
        &self,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        append_len: usize,
    ) -> Result<()> {
        let cache = kv
            .as_any()
            .downcast_ref::<CudaKVCache>()
            .ok_or_else(|| Error::Other("expected CudaKVCache".into()))?;
        cache.append(&self.ctx, layer_idx, k, v, append_len)
    }

    fn synchronize(&self) -> Result<()> {
        self.ctx.synchronize().map_err(Error::Cuda)
    }

    fn begin_capture(&self) -> Result<()> {
        // PI/VLA capture is driven entirely by this calling thread.
        // Thread-local mode preserves the captured work while avoiding
        // unrelated CUDA activity in other service/test threads from
        // invalidating the capture.
        crate::graph::begin(&self.ctx, crate::graph::CaptureMode::ThreadLocal).map_err(Error::Cuda)
    }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        Ok(Box::new(CudaGraph {
            graph: crate::graph::end(&self.ctx).map_err(Error::Cuda)?,
        }))
    }

    fn device(&self) -> Device {
        Device::Cuda(self.ctx.device_id())
    }

    fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        transfers::to_cuda(tensor, self.ctx.device_id())
    }

    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        transfers::to_cpu(tensor)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;

    #[test]
    fn backend_graph_capture_replays_preallocated_work() {
        let backend = CudaBackend::new(0).unwrap();
        let buffer = CudaBuffer::alloc_zeros(64, 0).unwrap();
        backend.begin_capture().unwrap();
        crate::graph::captured_memset(backend.context(), &buffer, 0x5a).unwrap();
        let graph = backend.end_capture().unwrap();
        graph.replay().unwrap();
        backend.synchronize().unwrap();
        let mut output = vec![0u8; 64];
        buffer.copy_to_host(&mut output).unwrap();
        assert!(output.iter().all(|value| *value == 0x5a));
    }
}

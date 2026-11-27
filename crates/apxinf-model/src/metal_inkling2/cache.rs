//! Per-visit KV and short-conv caches with chunked preallocation and MTP rollback.

use apxinf_core::{Error, Result};
use apxinf_mlx::{Array, MlxDType, Stream};

/// Step size for KVCache allocation (matches the 256-token cache allocation grain).
pub const KV_STEP: usize = 256;

/// Preallocated grow-by-step KV cache for a single layer visit.
pub struct KvVisitCache {
    pub keys: Option<Array>,
    pub values: Option<Array>,
    pub offset: usize,
    pub capacity: usize,
    pub n_kv: usize,
    pub head_dim: usize,
}

impl KvVisitCache {
    pub fn new(n_kv: usize, head_dim: usize) -> Self {
        Self {
            keys: None,
            values: None,
            offset: 0,
            capacity: 0,
            n_kv,
            head_dim,
        }
    }

    pub fn reset(&mut self) {
        self.keys = None;
        self.values = None;
        self.offset = 0;
        self.capacity = 0;
    }

    /// Preallocate the KV buffer for `tokens` (rounded up to `KV_STEP`).
    pub fn reserve_capacity(&mut self, stream: &Stream, tokens: usize, dtype: MlxDType) -> Result<()> {
        let want = tokens.div_ceil(KV_STEP) * KV_STEP;
        if self.keys.is_some() && self.capacity >= want {
            return Ok(());
        }
        if self.offset == 0 {
            self.keys = Some(Array::zeros(stream, &[1, self.n_kv, want, self.head_dim], dtype)?);
            self.values = Some(Array::zeros(stream, &[1, self.n_kv, want, self.head_dim], dtype)?);
            self.capacity = want;
            return Ok(());
        }
        self.grow_to(stream, want, dtype)
    }

    fn grow_to(&mut self, stream: &Stream, want: usize, dtype: MlxDType) -> Result<()> {
        let new_k = Array::zeros(stream, &[1, self.n_kv, want, self.head_dim], dtype)?;
        let new_v = Array::zeros(stream, &[1, self.n_kv, want, self.head_dim], dtype)?;
        if let (Some(old_k), Some(old_v)) = (&self.keys, &self.values) {
            if self.offset > 0 {
                let prev_k = old_k.slice_axis(2, 0, self.offset)?;
                let prev_v = old_v.slice_axis(2, 0, self.offset)?;
                self.keys = Some(new_k.slice_update_static(
                    &prev_k,
                    &[0, 0, 0, 0],
                    &[1, self.n_kv, self.offset, self.head_dim],
                )?);
                self.values = Some(new_v.slice_update_static(
                    &prev_v,
                    &[0, 0, 0, 0],
                    &[1, self.n_kv, self.offset, self.head_dim],
                )?);
            } else {
                self.keys = Some(new_k);
                self.values = Some(new_v);
            }
        } else {
            self.keys = Some(new_k);
            self.values = Some(new_v);
        }
        self.capacity = want;
        Ok(())
    }

    /// Append `k`, `v` (`[1, n_kv, L, D]`) in place and return views `[1, n_kv, offset, D]`.
    pub fn update_and_fetch(&mut self, k: &Array, v: &Array) -> Result<(Array, Array)> {
        let b = k.shape()[0];
        let n_kv = k.shape()[1];
        let length = k.shape()[2];
        let d = k.shape()[3];
        if b != 1 || n_kv != self.n_kv || d != self.head_dim {
            return Err(Error::Contract("KV update shape mismatch"));
        }
        let prev = self.offset;
        let next = prev + length;
        if self.keys.is_none() || next > self.capacity {
            let want = next.div_ceil(KV_STEP) * KV_STEP;
            self.grow_to(k.stream(), want, k.dtype())?;
        }
        let keys_buf = self.keys.take().unwrap();
        let vals_buf = self.values.take().unwrap();
        let keys_buf = keys_buf.slice_update_static(k, &[0, 0, prev, 0], &[1, n_kv, next, d])?;
        let vals_buf = vals_buf.slice_update_static(v, &[0, 0, prev, 0], &[1, n_kv, next, d])?;
        let k_view = keys_buf.slice_axis(2, 0, next)?;
        let v_view = vals_buf.slice_axis(2, 0, next)?;
        self.keys = Some(keys_buf);
        self.values = Some(vals_buf);
        self.offset = next;
        Ok((k_view, v_view))
    }

    pub fn trim(&mut self, rejected: usize) -> usize {
        let trimmed = rejected.min(self.offset);
        self.offset -= trimmed;
        trimmed
    }

    pub fn is_allocated(&self) -> bool {
        self.keys.is_some()
    }
}

/// Short-conv state for one of the 4 conv slots (`k`, `v`, `attn`, `mlp`).
///
/// Prefill stores a single `[1, K-1, C]` FP32 array; single-token decode and
/// exact MTP verify store `[Array; 3]` row views (`[1, 1, C]` FP32 each) to avoid
/// per-step concatenation and copy launches.
#[derive(Clone)]
pub enum ConvSlotState {
    Empty,
    Array(Array),
    Rows([Array; 3]),
}

impl ConvSlotState {
    /// Materialize as `[1, K-1, C]` FP32 array (for prefill chunks).
    pub fn as_array(&self, stream: &Stream, keep: usize, width: usize) -> Result<Array> {
        match self {
            Self::Empty => Array::zeros(stream, &[1, keep, width], MlxDType::F32),
            Self::Array(a) => Ok(a.clone()),
            Self::Rows(rows) => Array::concat(&[&rows[0], &rows[1], &rows[2]], 1),
        }
    }

    /// Materialize as 3 `[1, 1, C]` FP32 row views (for decode and exact verify).
    pub fn as_rows(&self, stream: &Stream, width: usize) -> Result<[Array; 3]> {
        match self {
            Self::Empty => {
                let z = Array::zeros(stream, &[1, 3, width], MlxDType::F32)?;
                Ok([
                    z.slice_axis(1, 0, 1)?,
                    z.slice_axis(1, 1, 2)?,
                    z.slice_axis(1, 2, 3)?,
                ])
            }
            Self::Array(a) => Ok([
                a.slice_axis(1, 0, 1)?,
                a.slice_axis(1, 1, 2)?,
                a.slice_axis(1, 2, 3)?,
            ]),
            Self::Rows(rows) => Ok(rows.clone()),
        }
    }

    pub fn push_eval_arrays(&self, out: &mut Vec<Array>) {
        match self {
            Self::Empty => {}
            Self::Array(a) => out.push(a.clone()),
            Self::Rows(rows) => {
                out.push(rows[0].clone());
                out.push(rows[1].clone());
                out.push(rows[2].clone());
            }
        }
    }
}

/// Complete state for a single layer visit (one of 48 visits).
pub struct VisitCache {
    pub kv: KvVisitCache,
    pub conv: [ConvSlotState; 4],
    /// Saved `[1, 3 + L, C]` FP32 conv windows from the most recent verify step for rollback.
    pub verify_windows: Option<[Array; 4]>,
}

impl VisitCache {
    pub fn new(n_kv: usize, head_dim: usize) -> Self {
        Self {
            kv: KvVisitCache::new(n_kv, head_dim),
            conv: [
                ConvSlotState::Empty,
                ConvSlotState::Empty,
                ConvSlotState::Empty,
                ConvSlotState::Empty,
            ],
            verify_windows: None,
        }
    }

    pub fn reset(&mut self) {
        self.kv.reset();
        self.conv = [
            ConvSlotState::Empty,
            ConvSlotState::Empty,
            ConvSlotState::Empty,
            ConvSlotState::Empty,
        ];
        self.verify_windows = None;
    }
}

/// Request-owned cache across all layer visits (48 visits for Inkling-2-Mini).
pub struct ModelCache {
    pub visits: Vec<VisitCache>,
}

impl ModelCache {
    pub fn new(schedule: &[(usize, usize)], n_kv_per_layer: &[usize], head_dim_per_layer: &[usize]) -> Self {
        let visits = schedule
            .iter()
            .map(|&(layer_idx, _)| {
                VisitCache::new(n_kv_per_layer[layer_idx], head_dim_per_layer[layer_idx])
            })
            .collect();
        Self { visits }
    }

    pub fn reset(&mut self) {
        for v in &mut self.visits {
            v.reset();
        }
    }

    pub fn offset(&self) -> usize {
        self.visits.first().map_or(0, |v| v.kv.offset)
    }

    pub fn capacity(&self) -> usize {
        self.visits.first().map_or(0, |v| v.kv.capacity)
    }

    /// Trim all KV caches back to `tokens` total offset, discarding any
    /// decode tokens appended after the prompt. Call before reusing the cache
    /// for a new prompt that shares a prefix with the previous one.
    pub fn trim_to(&mut self, tokens: usize) {
        for v in &mut self.visits {
            let excess = v.kv.offset.saturating_sub(tokens);
            if excess > 0 {
                v.kv.trim(excess);
            }
        }
    }

    /// Preallocate all 48 KV caches to `tokens` (rounded to `KV_STEP`) and evaluate them once.
    pub fn reserve_capacity(&mut self, stream: &Stream, tokens: usize, dtype: MlxDType) -> Result<()> {
        let mut to_eval = Vec::with_capacity(self.visits.len() * 2);
        for v in &mut self.visits {
            let had = v.kv.is_allocated();
            let prev_cap = v.kv.capacity;
            v.kv.reserve_capacity(stream, tokens, dtype)?;
            if !had || v.kv.capacity != prev_cap {
                if let Some(k) = &v.kv.keys {
                    to_eval.push(k.clone());
                }
                if let Some(val) = &v.kv.values {
                    to_eval.push(val.clone());
                }
            }
        }
        if !to_eval.is_empty() {
            stream.eval(&to_eval)?;
        }
        Ok(())
    }

    /// Collect all live KV and conv state arrays (for chunk-boundary `stream.eval`).
    pub fn state_arrays(&self) -> Vec<Array> {
        let mut out = Vec::with_capacity(self.visits.len() * 6);
        for v in &self.visits {
            if let Some(k) = &v.kv.keys {
                out.push(k.clone());
            }
            if let Some(val) = &v.kv.values {
                out.push(val.clone());
            }
            for slot in &v.conv {
                slot.push_eval_arrays(&mut out);
            }
        }
        out
    }

    /// Roll back `rejected = num_drafts - accepted` tokens after an MTP verify forward.
    pub fn mtp_partial_rollback(&mut self, accepted: usize, num_drafts: usize) -> Result<bool> {
        let rejected = num_drafts.saturating_sub(accepted);
        if rejected == 0 {
            for v in &mut self.visits {
                v.verify_windows = None;
            }
            return Ok(true);
        }
        let keep = accepted + 1;
        for v in &mut self.visits {
            let Some(windows) = v.verify_windows.take() else {
                return Ok(false);
            };
            if v.kv.trim(rejected) != rejected {
                return Ok(false);
            }
            for (slot, window) in windows.into_iter().enumerate() {
                v.conv[slot] = ConvSlotState::Rows([
                    window.slice_axis(1, keep, keep + 1)?,
                    window.slice_axis(1, keep + 1, keep + 2)?,
                    window.slice_axis(1, keep + 2, keep + 3)?,
                ]);
            }
        }
        Ok(true)
    }
}

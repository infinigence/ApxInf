//! Backend abstraction trait and Graph trait for execution capture/replay.

use crate::{Device, Result, SamplingBackend, Tensor};
use crate::kv_cache::KvCache;

/// Backend-agnostic interface for tensor compute and device management.
///
/// Combines:
/// - Compute ops (tensor → tensor kernels)
/// - Execution control (synchronize, graph capture)
/// - Device management (transfer, cache creation)
///
/// Object-safe so models can hold `dyn Backend`.
pub trait Backend: SamplingBackend {
    // ── Primitive compute ops ────────────────────────────────────────

    /// RMS normalization: output = input * rsqrt(mean(input^2) + eps) * weight
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;

    /// SiLU activation: output = input / (1 + exp(-input))
    fn silu(&self, input: &Tensor) -> Result<Tensor>;

    /// Element-wise add.
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Element-wise multiply.
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Fused SiLU(gate) * up. Backends may override this complete-write
    /// boundary; the default preserves composition and numerical order.
    fn silu_mul(&self, gate: &Tensor, up: &Tensor) -> Result<Tensor> {
        let activated = self.silu(gate)?;
        self.mul(&activated, up)
    }

    /// Scale by scalar: output = input * factor
    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor>;

    /// Matrix multiplication: output = a @ b
    /// a: [m, k], b: [k, n] -> output: [m, n]
    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Rotary Position Embedding (half-split / Llama-style).
    /// input shape: [seq_len, n_heads, head_dim]
    fn rope(&self, input: &Tensor, n_heads: usize, head_dim: usize,
            theta: f32, pos_offset: u32) -> Result<Tensor>;

    /// Multimodal (3-D) RoPE for Qwen3-VL. `pos_ids` is a flat u32 slice of
    /// length `seq_len * 3` holding `(t, h, w)` per token; `sections` is
    /// the `[T, H, W]` split of the `head_dim/2` frequency pairs.
    ///
    /// The backend is responsible for uploading `pos_ids` to device memory
    /// (this keeps callers dtype-agnostic — Tensor doesn't have a u32
    /// dtype and we don't want to smuggle bytes through a F32 tensor).
    ///
    /// Default: `Err(Unsupported)`. CUDA overrides.
    fn rope_mrope(&self, input: &Tensor, _n_heads: usize, _head_dim: usize,
                  _theta: f32, _sections: [usize; 3], _pos_ids: &[u32]) -> Result<Tensor> {
        let _ = input;
        Err(crate::Error::Other("rope_mrope: not supported on this backend".into()))
    }

    /// Qwen2.5-Omni TMRoPE. Unlike Qwen3-VL's interleaved pair-axis
    /// assignment, T/H/W own contiguous full-dimension sections. Paired
    /// rotate-half dimensions may consequently use different axes.
    fn rope_tmrope(&self, input: &Tensor, _n_heads: usize, _head_dim: usize,
                   _theta: f32, _sections: [usize; 3], _pos_ids: &[u32]) -> Result<Tensor> {
        let _ = input;
        Err(crate::Error::Other("rope_tmrope: not supported on this backend".into()))
    }

    /// LayerNorm with weight + bias (Qwen3-VL vision tower).
    /// `input` shape `[..., cols]`; `weight` and `bias` shape `[cols]`.
    /// Default: `Err(Unsupported)`. CUDA overrides.
    fn layer_norm(&self, input: &Tensor, _weight: &Tensor, _bias: &Tensor,
                  _eps: f32) -> Result<Tensor> {
        let _ = input;
        Err(crate::Error::Other("layer_norm: not supported on this backend".into()))
    }

    /// GELU with tanh approximation (Qwen3-VL vision MLP).
    /// Default: `Err(Unsupported)`. CUDA overrides.
    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        let _ = input;
        Err(crate::Error::Other("gelu_tanh: not supported on this backend".into()))
    }

    /// Broadcast-add a `[cols]` bias vector over rows of a `[rows, cols]`
    /// activation. Used after every vision linear layer.
    /// Default: `Err(Unsupported)`. CUDA overrides.
    fn add_bias(&self, input: &Tensor, _bias: &Tensor) -> Result<Tensor> {
        let _ = input;
        Err(crate::Error::Other("add_bias: not supported on this backend".into()))
    }

    /// Vision 2D-RoPE for Qwen3-VL's ViT. `pos_ids` is a flat u32 slice of
    /// length `seq_len * 2` holding `(h, w)` per token; head_dim is 64 and
    /// the first half of freq pairs uses h, the second half uses w.
    /// Default: `Err(Unsupported)`. CUDA overrides.
    fn rope_vision_2d(&self, input: &Tensor, _n_heads: usize, _head_dim: usize,
                      _theta: f32, _pos_ids: &[u32]) -> Result<Tensor> {
        let _ = input;
        Err(crate::Error::Other("rope_vision_2d: not supported on this backend".into()))
    }

    /// Concatenate 2D tensors along the column axis (dim 1).
    /// All inputs must have the same row count; outputs have
    /// `sum(input.col_counts)` columns. Used at load time to build fused
    /// QKV and Gate/Up weight matrices for the fused GEMM path.
    /// Default: `Err(Unsupported)`. CUDA overrides (D2D memcpy).
    fn concat_2d(&self, _tensors: &[&Tensor]) -> Result<Tensor> {
        Err(crate::Error::Other("concat_2d: not supported on this backend".into()))
    }

    /// Non-causal full attention for the vision tower. Q/K/V each
    /// `[seq, n_heads, head_dim]`; returns `[seq, n_heads * head_dim]`.
    /// Default: `Err(Unsupported)`. CUDA overrides.
    fn vision_sdpa(&self, _q: &Tensor, _k: &Tensor, _v: &Tensor,
                   _seq_len: usize, _n_heads: usize, _head_dim: usize) -> Result<Tensor> {
        Err(crate::Error::Other("vision_sdpa: not supported on this backend".into()))
    }

    /// Non-causal attention constrained to processor/model-owned groups.
    /// Query `i` may attend to key `j` exactly when `group_ids[i] ==
    /// group_ids[j]`. This represents windowed vision attention without
    /// smuggling a model-specific window schedule into the backend.
    fn grouped_sdpa(&self, _q: &Tensor, _k: &Tensor, _v: &Tensor,
                    _seq_len: usize, _n_heads: usize, _head_dim: usize,
                    _group_ids: &[u32]) -> Result<Tensor> {
        Err(crate::Error::Other("grouped_sdpa: not supported on this backend".into()))
    }

    /// Lower a `[frames, channels]` sequence into contiguous convolution
    /// patches `[out_frames, channels * kernel]`. A model composes this one
    /// kernel with the ordinary matmul and bias primitives.
    fn im2col1d(&self, _input: &Tensor, _kernel: usize, _stride: usize,
                _padding: usize) -> Result<Tensor> {
        Err(crate::Error::Other("im2col1d: not supported on this backend".into()))
    }

    /// Average-pool a `[frames, channels]` sequence along its frame axis.
    fn avg_pool1d(&self, _input: &Tensor, _kernel: usize, _stride: usize) -> Result<Tensor> {
        Err(crate::Error::Other("avg_pool1d: not supported on this backend".into()))
    }

    /// Embedding lookup: table[ids] -> output [seq_len, embed_dim]
    /// table: [vocab_size, embed_dim], ids: u32 token IDs
    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor>;

    // ── Composite compute ops ────────────────────────────────────────

    /// Scaled dot-product attention (decode: seq_len=1).
    /// q: [1, n_heads, head_dim]
    /// Returns: [1, n_heads * head_dim]
    fn sdpa_decode(&self, q: &Tensor, kv: &mut dyn KvCache,
                   layer_idx: usize, n_heads: usize, n_kv_heads: usize,
                   head_dim: usize, kv_len: usize, max_seq_len: usize) -> Result<Tensor>;

    /// Scaled dot-product attention (prefill: seq_len>1).
    /// q: [seq_len, n_heads, head_dim]
    fn sdpa_prefill(&self, q: &Tensor, kv: &mut dyn KvCache,
                    layer_idx: usize, n_heads: usize, n_kv_heads: usize,
                    head_dim: usize, kv_len: usize, max_seq_len: usize) -> Result<Tensor>;

    // ── KV Cache ─────────────────────────────────────────────────────

    /// Create a new KV cache for n_layers layers.
    fn create_kv_cache(&self, n_layers: usize, n_kv_heads: usize,
                       head_dim: usize, max_seq_len: usize) -> Box<dyn KvCache>;

    /// Append K/V data for a layer into the cache.
    ///
    /// This is on the Backend (not just KvCache) because GPU backends need
    /// access to their stream/context to encode the append kernel.
    /// k, v: [append_len, n_kv_heads, head_dim]
    fn kv_append(&self, kv: &mut dyn KvCache, layer_idx: usize,
                 k: &Tensor, v: &Tensor, append_len: usize) -> Result<()>;

    // ── Execution control ────────────────────────────────────────────

    /// Block until all queued operations complete.
    fn synchronize(&self) -> Result<()>;

    /// Start recording ops into a capture graph.
    fn begin_capture(&self) -> Result<()>;

    /// End capture and return a replayable graph.
    fn end_capture(&self) -> Result<Box<dyn Graph>>;

    // ── Device management ────────────────────────────────────────────

    /// Which device this backend targets.
    fn device(&self) -> Device;

    /// Copy tensor to this backend's device.
    fn to_device(&self, tensor: &Tensor) -> Result<Tensor>;

    /// Copy tensor to CPU.
    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor>;

    /// Downcast to `Any` — enables `downcast_ref::<CudaBackend>()` on
    /// `&dyn Backend`. Used by models that need the concrete backend type
    /// for the fast path (e.g. decode workspace + graph capture).
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Which flavor of Rotary Position Embedding the decode graph should apply.
///
/// `OneD` is the standard Llama / Qwen2 / TinyLlama scalar-position RoPE.
/// `MRope3D` is the Qwen3-VL multimodal RoPE with a 3-vector (t, h, w) per
/// position and interleaved axis assignment across the 64 frequency pairs
/// according to `sections`.
///
/// The enum is `Copy` (`sections` is a fixed `[usize;3]`) so it can flow
/// into the backend by value like the rest of the config primitives.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RopeKind {
    OneD { theta: f32 },
    MRope3D { theta: f32, sections: [usize; 3] },
}

impl RopeKind {
    /// Backward-compatible default for models that don't know about mRoPE.
    pub fn one_d(theta: f32) -> Self { RopeKind::OneD { theta } }
}

/// A captured execution graph that can be replayed.
///
/// CUDA: wraps cudaGraphExec_t, replayed via cudaGraphLaunch.
/// CPU: no-op (synchronous, nothing to capture).
pub trait Graph {
    /// Replay the captured graph. Inputs must already be updated in-place.
    fn replay(&self) -> Result<()>;
}

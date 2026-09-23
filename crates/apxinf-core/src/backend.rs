//! Backend abstraction trait and Graph trait for execution capture/replay.

use crate::{contracts, Device, Error, Result, SamplingBackend, Tensor};
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
    // Portable composition extensions. These are implementation hooks: callers use
    // the validated `PortableOps` entry points, which check arguments and the
    // returned shape before and after dispatch. Default errors preserve existing
    // backend implementations while phase 2 adds device implementations.
    // See `contracts` and doc/backend-contracts.md for the normative semantics.
    //
    // Implement `*_impl` and assume arguments are already structurally valid.
    // Additive mask *values* are not checked here; see `validate_mask_values`.

    /// Convert F32/F16/BF16 with round-to-nearest-even; preserve shape/device.
    /// Output is functional, including a same-dtype cast. No FP8 reinterpretation.
    fn cast_impl(&self, _input: &Tensor, _dtype: crate::DType) -> Result<Tensor> {
        Err(Error::UnsupportedOp("cast"))
    }

    /// Materialize a contiguous axis slice without mutating or aliasing inputs.
    fn slice_axis_impl(&self, _input: &Tensor, _slice: contracts::AxisSlice) -> Result<Tensor> {
        Err(Error::UnsupportedOp("slice_axis"))
    }

    /// Concatenate on any axis. All inputs have identical dtype/device/rank.
    fn concat_axis_impl(&self, _inputs: &[&Tensor], _axis: usize) -> Result<Tensor> {
        Err(Error::UnsupportedOp("concat_axis"))
    }

    /// Reorder axes and return contiguous storage; unlike reshape this moves data.
    fn permute_impl(&self, _input: &Tensor, _axes: &[usize]) -> Result<Tensor> {
        Err(Error::UnsupportedOp("permute"))
    }

    /// Materialize right-aligned broadcasting, preserving dtype/device/bits.
    fn broadcast_to_impl(&self, _input: &Tensor, _shape: &[usize]) -> Result<Tensor> {
        Err(Error::UnsupportedOp("broadcast_to"))
    }

    /// Stateless MHA/GQA/MQA with explicit masking and intermediate precision.
    /// Q[B,Q,Hq,D], K/V[B,K,Hkv,D] -> output shaped like Q. No KV mutation.
    fn attention_impl(&self, _q: &Tensor, _k: &Tensor, _v: &Tensor,
                      _options: &contracts::AttentionOptions<'_>) -> Result<Tensor> {
        Err(Error::UnsupportedOp("attention"))
    }

    // ── Primitive compute ops ────────────────────────────────────────

    /// RMS normalization: output = input * rsqrt(mean(input^2) + eps) * weight
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor>;

    /// SiLU activation: output = input / (1 + exp(-input))
    fn silu(&self, input: &Tensor) -> Result<Tensor>;

    /// Element-wise add.
    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

    /// Element-wise multiply.
    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor>;

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
        Err(Error::UnsupportedOp("rope_mrope"))
    }

    /// LayerNorm with weight + bias (Qwen3-VL vision tower).
    /// `input` shape `[..., cols]`; `weight` and `bias` shape `[cols]`.
    /// Default: `Err(Unsupported)`. CUDA overrides.
    fn layer_norm(&self, input: &Tensor, _weight: &Tensor, _bias: &Tensor,
                  _eps: f32) -> Result<Tensor> {
        let _ = input;
        Err(Error::UnsupportedOp("layer_norm"))
    }

    /// GELU with tanh approximation (Qwen3-VL vision MLP).
    /// Default: `Err(Unsupported)`. CUDA overrides.
    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        let _ = input;
        Err(Error::UnsupportedOp("gelu_tanh"))
    }

    /// Broadcast-add a `[cols]` bias vector over rows of a `[rows, cols]`
    /// activation. Used after every vision linear layer.
    /// Default: `Err(Unsupported)`. CUDA overrides.
    fn add_bias(&self, input: &Tensor, _bias: &Tensor) -> Result<Tensor> {
        let _ = input;
        Err(Error::UnsupportedOp("add_bias"))
    }

    /// Vision 2D-RoPE for Qwen3-VL's ViT. `pos_ids` is a flat u32 slice of
    /// length `seq_len * 2` holding `(h, w)` per token; head_dim is 64 and
    /// the first half of freq pairs uses h, the second half uses w.
    /// Default: `Err(Unsupported)`. CUDA overrides.
    fn rope_vision_2d(&self, input: &Tensor, _n_heads: usize, _head_dim: usize,
                      _theta: f32, _pos_ids: &[u32]) -> Result<Tensor> {
        let _ = input;
        Err(Error::UnsupportedOp("rope_vision_2d"))
    }

    /// Concatenate 2D tensors along the column axis (dim 1).
    /// All inputs must have the same row count; outputs have
    /// `sum(input.col_counts)` columns. Used at load time to build fused
    /// QKV and Gate/Up weight matrices for the fused GEMM path.
    /// Default: `Err(Unsupported)`. CUDA overrides (D2D memcpy).
    fn concat_2d(&self, _tensors: &[&Tensor]) -> Result<Tensor> {
        Err(Error::UnsupportedOp("concat_2d"))
    }

    /// Non-causal full attention for the vision tower. Q/K/V each
    /// `[seq, n_heads, head_dim]`; returns `[seq, n_heads * head_dim]`.
    /// Default: `Err(Unsupported)`. CUDA overrides.
    fn vision_sdpa(&self, _q: &Tensor, _k: &Tensor, _v: &Tensor,
                   _seq_len: usize, _n_heads: usize, _head_dim: usize) -> Result<Tensor> {
        Err(Error::UnsupportedOp("vision_sdpa"))
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

/// Validated entry points for the portable operators.
///
/// Blanket-implemented for every [`Backend`], including `dyn Backend`, so a
/// backend cannot override or skip validation: it only implements the `*_impl`
/// hooks. Each method validates arguments against [`contracts`], dispatches,
/// then checks the returned shape against the contract before handing it back.
///
/// Import this trait to call the portable ops:
/// `use apxinf_core::{Backend, PortableOps};`
pub trait PortableOps {
    fn cast(&self, input: &Tensor, dtype: crate::DType) -> Result<Tensor>;
    fn slice_axis(&self, input: &Tensor, slice: contracts::AxisSlice) -> Result<Tensor>;
    fn concat_axis(&self, inputs: &[&Tensor], axis: usize) -> Result<Tensor>;
    fn permute(&self, input: &Tensor, axes: &[usize]) -> Result<Tensor>;
    fn broadcast_to(&self, input: &Tensor, shape: &[usize]) -> Result<Tensor>;
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor,
                 options: &contracts::AttentionOptions<'_>) -> Result<Tensor>;
}

/// Confirm an implementation honoured the contract it was given: exact shape and
/// dtype, the backend's own device, and storage large enough for the result it
/// claims. Cheap (rank-sized) and kept in release builds so a backend bug surfaces
/// as an error rather than as silently misinterpreted downstream data.
///
/// Shape alone is not enough: a `cast` that returns its input unchanged preserves
/// the shape while ignoring the requested dtype, and a short-storage result would
/// be read out of bounds downstream.
fn check_output(
    op: &'static str,
    out: Tensor,
    expected: &[usize],
    dtype: crate::DType,
    device: Device,
) -> Result<Tensor> {
    if out.shape().dims() != expected {
        return Err(Error::ShapeMismatch {
            expected: format!("{op} -> {expected:?}"),
            got: format!("{:?}", out.shape().dims()),
        });
    }
    if out.dtype() != dtype {
        return Err(Error::DTypeMismatch {
            expected: dtype,
            got: out.dtype(),
        });
    }
    // Device residency plus storage extent for the shape/dtype just checked.
    contracts::tensor_storage(&out, device)?;
    Ok(out)
}

impl<B: Backend + ?Sized> PortableOps for B {
    fn cast(&self, input: &Tensor, dtype: crate::DType) -> Result<Tensor> {
        let device = self.device();
        contracts::float_tensor(input, device)?;
        if !dtype.is_float() {
            return Err(Error::UnsupportedDType {
                got: dtype,
                allowed: "f32, f16, bf16",
            });
        }
        let expected = input.shape().dims().to_vec();
        // Output dtype may be wider than the input's, so re-check the byte extent.
        contracts::checked_bytes(&expected, dtype)?;
        check_output("cast", self.cast_impl(input, dtype)?, &expected, dtype, device)
    }

    fn slice_axis(&self, input: &Tensor, slice: contracts::AxisSlice) -> Result<Tensor> {
        // Layout ops preserve bits, so any dtype is admissible.
        let device = self.device();
        contracts::tensor_storage(input, device)?;
        let expected = slice.output_shape(input.shape().dims())?;
        contracts::checked_bytes(&expected, input.dtype())?;
        check_output(
            "slice_axis",
            self.slice_axis_impl(input, slice)?,
            &expected,
            input.dtype(),
            device,
        )
    }

    fn concat_axis(&self, inputs: &[&Tensor], axis: usize) -> Result<Tensor> {
        let device = self.device();
        let first = *inputs
            .first()
            .ok_or(Error::Contract("concat requires at least one input"))?;
        for t in inputs {
            contracts::tensor_storage(t, device)?;
            if t.dtype() != first.dtype() {
                return Err(Error::DTypeMismatch {
                    expected: first.dtype(),
                    got: t.dtype(),
                });
            }
        }
        let dims: Vec<&[usize]> = inputs.iter().map(|t| t.shape().dims()).collect();
        let expected = contracts::concatenated_shape(&dims, axis)?;
        // The concatenated result is larger than any single validated input.
        contracts::checked_bytes(&expected, first.dtype())?;
        check_output(
            "concat_axis",
            self.concat_axis_impl(inputs, axis)?,
            &expected,
            first.dtype(),
            device,
        )
    }

    fn permute(&self, input: &Tensor, axes: &[usize]) -> Result<Tensor> {
        let device = self.device();
        contracts::tensor_storage(input, device)?;
        let expected = contracts::permuted_shape(input.shape().dims(), axes)?;
        contracts::checked_bytes(&expected, input.dtype())?;
        check_output(
            "permute",
            self.permute_impl(input, axes)?,
            &expected,
            input.dtype(),
            device,
        )
    }

    fn broadcast_to(&self, input: &Tensor, shape: &[usize]) -> Result<Tensor> {
        let device = self.device();
        contracts::tensor_storage(input, device)?;
        contracts::broadcast_shape(input.shape().dims(), shape)?;
        // Broadcasting expands the element count, so the byte extent can overflow
        // even when the element count itself is representable.
        contracts::checked_bytes(shape, input.dtype())?;
        check_output(
            "broadcast_to",
            self.broadcast_to_impl(input, shape)?,
            shape,
            input.dtype(),
            device,
        )
    }

    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor,
                 options: &contracts::AttentionOptions<'_>) -> Result<Tensor> {
        let device = self.device();
        let expected = options.validate(device, q, k, v)?;
        contracts::checked_bytes(&expected, q.dtype())?;
        check_output(
            "attention",
            self.attention_impl(q, k, v, options)?,
            &expected,
            q.dtype(),
            device,
        )
    }
}

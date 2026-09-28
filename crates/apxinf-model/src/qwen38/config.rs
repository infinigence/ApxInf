//! Model shape constants and the runtime configuration.
//!
//! The shapes are properties of the Qwen3.8-27B-NVFP4 checkpoint (64 layers:
//! 48 Gated DeltaNet + 16 full attention on every 4th layer); they are not
//! tunable. The runtime options that survived the optimization rounds as
//! genuine choices live in [`Qwen38Config`]; everything else that used to be
//! an `APXINF_*` environment flag is now simply the default code path.

pub(crate) const HIDDEN: usize = 5120;
pub(crate) const INTERMEDIATE: usize = 17408;
pub const VOCAB: usize = 248320;
pub(crate) const LAYERS: usize = 64;
pub(crate) const FULL_ATTENTION_INTERVAL: usize = 4;
pub(crate) const BLOCK: u32 = 16;
pub(crate) const EPSILON: f32 = 1e-6;

// Full attention
pub(crate) const HEADS: usize = 24;
pub(crate) const KV_HEADS: usize = 4;
pub(crate) const HEAD_DIM: usize = 256;
pub(crate) const ROPE_THETA: f32 = 1.0e7;
pub(crate) const PARTIAL_ROTARY: f32 = 0.25;

// Gated DeltaNet
pub(crate) const GDN_K_HEADS: usize = 16;
pub(crate) const GDN_V_HEADS: usize = 48;
pub(crate) const GDN_HEAD_DIM: usize = 128;
pub(crate) const CONV_WIDTH: usize = 4;
pub(crate) const CHUNK: usize = 64; // the chunked GDN scan works a chunk at a time
pub(crate) const QKV_WIDTH: usize = 10240; // 16*128 q + 16*128 k + 48*128 v
pub(crate) const Z_WIDTH: usize = 6144;

pub(crate) fn is_full_attention(layer: usize) -> bool {
    (layer + 1) % FULL_ATTENTION_INTERVAL == 0
}

/// Runtime options. Both defaults are the validated fast paths; the
/// alternatives exist for precision comparison, not for performance.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Qwen38Config {
    /// Split-KV FA2 decode attention. Changes the FA2 reduction order, which
    /// since the quantizer fixes produces the same tokens as the non-split
    /// path on the acceptance prompt; kept as an option because the reduction
    /// order is still a numeric degree of freedom.
    pub splitkv: bool,
    /// FlashInfer Cake GDN prefill. `false` falls back to the reference
    /// chunked scan (~75x slower prefill), which is the numeric baseline the
    /// FlashInfer path is shadow-checked against.
    pub flashinfer_gdn: bool,
}

impl Default for Qwen38Config {
    fn default() -> Self {
        Self {
            splitkv: true,
            flashinfer_gdn: true,
        }
    }
}

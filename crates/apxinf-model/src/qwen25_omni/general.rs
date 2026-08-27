//! Native Qwen2.5-Omni Thinker orchestration.

use std::collections::HashMap;
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::OnceLock;

use apxinf_core::{Backend, Device, Error, KvCache, Result, Tensor};

#[cfg(feature = "cuda")]
use crate::accelerator::cuda::{downcast as cuda_backend, kernels as cuda_kernels, DeviceBuffer};
use crate::llm_trait::{LlmCapabilities, LlmInput, LlmTrait};

use super::audio::{self, Qwen25OmniAudioWeights};
use super::config::Qwen25OmniConfig;
#[cfg(feature = "cuda")]
use super::decode_graph::{
    Qwen25OmniDecodeGraph, Qwen25OmniDecodeGraphConfig, Qwen25OmniDecodeGraphWeights,
    Qwen25OmniDecodeLayerWeights, Qwen25OmniDecodeQkvWeights,
};
#[cfg(feature = "cuda")]
use super::parse_binary_env;
use super::vision::{self, Qwen25OmniVisionWeights};
use super::weights::{Qwen25OmniQkvWeights, Qwen25OmniTextWeights};

#[cfg(feature = "cuda")]
const MAX_DECODE_GRAPH_POSITION: u32 = 3_072;
#[cfg(any(feature = "cuda", test))]
const LONG_DECODE_GRAPH_MIN_POSITION: u32 = 32_760;
#[cfg(any(feature = "cuda", test))]
const CHUNKED_PREFILL_MID_SIZE: usize = 256;
#[cfg(any(feature = "cuda", test))]
const CHUNKED_PREFILL_SMALL_SIZE: usize = 512;
#[cfg(any(feature = "cuda", test))]
const CHUNKED_PREFILL_SMALL_MAX_PROMPT: usize = 12_288;
#[cfg(any(feature = "cuda", test))]
const CHUNKED_PREFILL_MID_MIN_PROMPT: usize = 4_096;
#[cfg(any(feature = "cuda", test))]
const CHUNKED_PREFILL_MID_MAX_PROMPT: usize = 12_288;
#[cfg(any(feature = "cuda", test))]
const CHUNKED_PREFILL_LARGE_SIZE: usize = 1_024;
#[cfg(any(feature = "cuda", test))]
const CHUNKED_PREFILL_FA2_LARGE_MIN_PROMPT: usize = 8_192;
#[cfg(any(feature = "cuda", test))]
const CHUNKED_PREFILL_FA2_LARGE_MAX_PROMPT: usize = 12_288;
#[cfg(feature = "cuda")]
const CHUNKED_PREFILL_THRESHOLD: usize = 1_024;
#[cfg(any(feature = "cuda", test))]
const LONG_DECODE_SPLIT_CTA_MIN_KV: usize = 32_761;
#[cfg(any(feature = "cuda", test))]
const LONG_DECODE_SPLIT_CTA_COUNT: usize = 64;

#[cfg(any(feature = "cuda", test))]
fn text_prefill_chunk_size(prompt_tokens: usize, fa2_chunk1024: bool) -> usize {
    if fa2_chunk1024
        && (CHUNKED_PREFILL_FA2_LARGE_MIN_PROMPT..=CHUNKED_PREFILL_FA2_LARGE_MAX_PROMPT)
            .contains(&prompt_tokens)
    {
        CHUNKED_PREFILL_LARGE_SIZE
    } else if (CHUNKED_PREFILL_MID_MIN_PROMPT..=CHUNKED_PREFILL_MID_MAX_PROMPT)
        .contains(&prompt_tokens)
    {
        CHUNKED_PREFILL_MID_SIZE
    } else if prompt_tokens <= CHUNKED_PREFILL_SMALL_MAX_PROMPT {
        CHUNKED_PREFILL_SMALL_SIZE
    } else {
        CHUNKED_PREFILL_LARGE_SIZE
    }
}

#[cfg(any(feature = "cuda", test))]
fn use_all_chunk_fa2(prompt_tokens: usize, enabled: bool) -> bool {
    enabled
        && (CHUNKED_PREFILL_FA2_LARGE_MIN_PROMPT..=CHUNKED_PREFILL_FA2_LARGE_MAX_PROMPT)
            .contains(&prompt_tokens)
}

#[cfg(any(feature = "cuda", test))]
fn use_long_decode_split_cta(kv_len: usize, enabled: bool) -> bool {
    enabled && kv_len >= LONG_DECODE_SPLIT_CTA_MIN_KV
}

#[cfg(any(feature = "cuda", test))]
fn use_long_decode_graph(position: u32, enabled: bool) -> bool {
    enabled && position >= LONG_DECODE_GRAPH_MIN_POSITION
}

#[cfg(any(feature = "cuda", test))]
fn use_scaled_exp_cache_prefill(
    sequence: usize,
    kv_len: usize,
    enabled: bool,
    all_chunk_fa2: bool,
) -> bool {
    enabled && !all_chunk_fa2 && sequence > 1 && kv_len <= 4_096
}

#[cfg(any(feature = "cuda", test))]
fn use_prefill_packed_mlp(sequence: usize, enabled: bool) -> bool {
    enabled && matches!(sequence, 512 | 1_024 | 1_760)
}

#[cfg(any(feature = "cuda", test))]
fn use_prefill_packed_qkv_prelude(sequence: usize, enabled: bool) -> bool {
    enabled && sequence == 1_024
}

pub struct GeneralQwen25Omni {
    config: Qwen25OmniConfig,
    text: Qwen25OmniTextWeights,
    vision: Qwen25OmniVisionWeights,
    audio: Qwen25OmniAudioWeights,
    backend: Arc<dyn Backend>,
    kv: Box<dyn KvCache>,
    rope_delta: i64,
    vision_qkv_bias_rope: bool,
    vision_fused_silu_mul: bool,
    vision_bias_residual: bool,
    vision_gate_up_bias_silu_mul: bool,
    vision_grouped_qkv_layout: bool,
    vision_packed_qkv: bool,
    vision_packed_gate_up: bool,
    #[cfg(feature = "cuda")]
    decode_graph: Option<Qwen25OmniDecodeGraph>,
    #[cfg(feature = "cuda")]
    long_decode_graph: Option<Qwen25OmniDecodeGraph>,
    #[cfg(feature = "cuda")]
    long_decode_split_cta: Option<cuda_kernels::qwen25_omni_attention::SplitCtaWorkspace>,
    #[cfg(feature = "cuda")]
    scaled_exp_cache_prefill: bool,
    #[cfg(feature = "cuda")]
    prefill_packed_mlp: bool,
    #[cfg(feature = "cuda")]
    prefill_packed_qkv_prelude: bool,
}

#[cfg(feature = "cuda")]
fn decode_graph_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_DECODE_GRAPH"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn long_decode_graph_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_LONG_DECODE_GRAPH"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn gpu_argmax_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_GPU_ARGMAX"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn packed_qkv_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_PACKED_QKV"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn m1_packed_mlp_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_M1_PACKED_MLP"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn short_decode_exact_residual_norm_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_SHORT_DECODE_EXACT_RESIDUAL_NORM"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn short_decode_w32_attention_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_SHORT_DECODE_W32_ATTENTION"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn short_decode_fused_qkv_prelude_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_SHORT_DECODE_FUSED_QKV_PRELUDE"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn short_decode_pack8_residual_norm_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_SHORT_DECODE_PACK8_RESIDUAL_NORM"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn scaled_exp_cache_prefill_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_SCALED_EXP_CACHE_PREFILL"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn fused_tmrope_kv_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_FUSED_TMROPE_KV"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn chunked_prefill_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_CHUNKED_PREFILL"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn fa2_chunk1024_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| {
            let enabled = parse_binary_env("APXINF_QWEN25_FA2_CHUNK1024")?;
            if enabled && !parse_binary_env("APXINF_FA2_GQA_PREFILL")? {
                return Err(
                    "APXINF_QWEN25_FA2_CHUNK1024=1 requires APXINF_FA2_GQA_PREFILL=1".into(),
                );
            }
            Ok(enabled)
        })
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn all_chunk_fa2_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| {
            let enabled = parse_binary_env("APXINF_QWEN25_FA2_ALL_CHUNKS")?;
            if enabled && !fa2_chunk1024_enabled().map_err(|error| error.to_string())? {
                return Err(
                    "APXINF_QWEN25_FA2_ALL_CHUNKS=1 requires APXINF_QWEN25_FA2_CHUNK1024=1".into(),
                );
            }
            Ok(enabled)
        })
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn gpu_last_row_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_GPU_LAST_ROW"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn eager_gpu_argmax_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_EAGER_GPU_ARGMAX"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn fused_silu_mul_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_FUSED_SILU_MUL"))
        .clone()
        .map_err(Error::Other)
}

#[cfg(feature = "cuda")]
fn long_decode_split_cta_enabled() -> Result<bool> {
    static ENABLED: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    ENABLED
        .get_or_init(|| parse_binary_env("APXINF_QWEN25_LONG_DECODE_SPLIT_CTA"))
        .clone()
        .map_err(Error::Other)
}

impl GeneralQwen25Omni {
    pub(crate) fn from_selected_weights(
        config: Qwen25OmniConfig,
        mut tensors: HashMap<String, Tensor>,
        backend: Arc<dyn Backend>,
    ) -> Result<Self> {
        let text = Qwen25OmniTextWeights::from_map(&config, &mut tensors)?.to_device(&*backend)?;
        #[cfg(feature = "cuda")]
        let mut text = text;
        let vision =
            Qwen25OmniVisionWeights::from_map(&config, &mut tensors)?.to_device(&*backend)?;
        #[cfg(feature = "cuda")]
        let mut vision = vision;
        let audio =
            Qwen25OmniAudioWeights::from_map(&config, &mut tensors)?.to_device(&*backend)?;
        if !tensors.is_empty() {
            let mut names = tensors.keys().cloned().collect::<Vec<_>>();
            names.sort();
            return Err(Error::Other(format!(
                "qwen2.5-omni selected loader left unowned tensors: {}",
                names.into_iter().take(8).collect::<Vec<_>>().join(", ")
            )));
        }
        let kv = backend.create_kv_cache(
            config.text.n_layers,
            config.text.n_kv_heads,
            config.text.head_dim,
            config.text.max_position_embeddings,
        );
        #[cfg(feature = "cuda")]
        let vision_qkv_bias_rope = {
            let enabled =
                parse_binary_env("APXINF_QWEN25_VISION_QKV_BIAS_ROPE").map_err(Error::Other)?;
            if enabled {
                let cuda = cuda_backend(&*backend).ok_or_else(|| {
                    Error::Other("Qwen2.5-Omni vision QKV bias/RoPE requires CudaBackend".into())
                })?;
                let vision = &config.vision;
                if cuda.context().caps().sm != 89
                    || vision.depth != 32
                    || vision.hidden_size != 1_280
                    || vision.n_heads != 16
                    || vision.head_dim != 80
                    || !parse_binary_env("APXINF_VISION_GROUPED_SPARSE").map_err(Error::Other)?
                    || !parse_binary_env("APXINF_VISION_GROUPED_FA2").map_err(Error::Other)?
                    || !parse_binary_env("APXINF_VISION_FULL_FA2").map_err(Error::Other)?
                {
                    return Err(Error::Other(
                        "APXINF_QWEN25_VISION_QKV_BIAS_ROPE=1 requires the pinned SM89 vision stack and all accepted vision selectors"
                            .into(),
                    ));
                }
                eprintln!(
                    "ApxInf Qwen2.5-Omni vision QKV bias/RoPE: exact one-owner projection epilogue"
                );
            }
            enabled
        };
        #[cfg(not(feature = "cuda"))]
        let vision_qkv_bias_rope = false;
        #[cfg(feature = "cuda")]
        let vision_fused_silu_mul = {
            let enabled = parse_binary_env("APXINF_QWEN25_VISION_FUSED_SILU_MUL")
                .map_err(Error::Other)?;
            if enabled {
                if !vision_qkv_bias_rope {
                    return Err(Error::Other(
                        "APXINF_QWEN25_VISION_FUSED_SILU_MUL=1 requires APXINF_QWEN25_VISION_QKV_BIAS_ROPE=1"
                            .into(),
                    ));
                }
                eprintln!("ApxInf Qwen2.5-Omni vision fused SiLU/multiply: exact MLP activation");
            }
            enabled
        };
        #[cfg(not(feature = "cuda"))]
        let vision_fused_silu_mul = false;
        #[cfg(feature = "cuda")]
        let vision_bias_residual = {
            let enabled = parse_binary_env("APXINF_QWEN25_VISION_BIAS_RESIDUAL")
                .map_err(Error::Other)?;
            if enabled {
                if !vision_fused_silu_mul {
                    return Err(Error::Other(
                        "APXINF_QWEN25_VISION_BIAS_RESIDUAL=1 requires APXINF_QWEN25_VISION_FUSED_SILU_MUL=1"
                            .into(),
                    ));
                }
                eprintln!("ApxInf Qwen2.5-Omni vision exact bias/residual: two-round epilogue");
            }
            enabled
        };
        #[cfg(not(feature = "cuda"))]
        let vision_bias_residual = false;
        #[cfg(feature = "cuda")]
        let vision_gate_up_bias_silu_mul = {
            let enabled = parse_binary_env("APXINF_QWEN25_VISION_GATE_UP_BIAS_SILU_MUL")
                .map_err(Error::Other)?;
            if enabled {
                if !vision_bias_residual {
                    return Err(Error::Other(
                        "APXINF_QWEN25_VISION_GATE_UP_BIAS_SILU_MUL=1 requires APXINF_QWEN25_VISION_BIAS_RESIDUAL=1"
                            .into(),
                    ));
                }
                eprintln!(
                    "ApxInf Qwen2.5-Omni vision Gate/Up bias SiLU/multiply: exact MLP owner"
                );
            }
            enabled
        };
        #[cfg(not(feature = "cuda"))]
        let vision_gate_up_bias_silu_mul = false;
        #[cfg(feature = "cuda")]
        let vision_grouped_qkv_layout = {
            let enabled = parse_binary_env("APXINF_QWEN25_VISION_GROUPED_QKV_LAYOUT")
                .map_err(Error::Other)?;
            if enabled {
                if !vision_gate_up_bias_silu_mul {
                    return Err(Error::Other(
                        "APXINF_QWEN25_VISION_GROUPED_QKV_LAYOUT=1 requires APXINF_QWEN25_VISION_GATE_UP_BIAS_SILU_MUL=1"
                            .into(),
                    ));
                }
                eprintln!(
                    "ApxInf Qwen2.5-Omni grouped vision QKV layout: direct FA2 consumer order"
                );
            }
            enabled
        };
        #[cfg(not(feature = "cuda"))]
        let vision_grouped_qkv_layout = false;
        #[cfg(feature = "cuda")]
        let vision_packed_qkv = {
            let enabled = parse_binary_env("APXINF_QWEN25_VISION_PACKED_QKV")
                .map_err(Error::Other)?;
            if enabled {
                if !vision_grouped_qkv_layout {
                    return Err(Error::Other(
                        "APXINF_QWEN25_VISION_PACKED_QKV=1 requires APXINF_QWEN25_VISION_GROUPED_QKV_LAYOUT=1"
                            .into(),
                    ));
                }
                vision = vision.into_packed_qkv(&*backend)?;
                eprintln!(
                    "ApxInf Qwen2.5-Omni vision packed QKV: one projection with direct epilogue consumption"
                );
            }
            enabled
        };
        #[cfg(not(feature = "cuda"))]
        let vision_packed_qkv = false;
        #[cfg(feature = "cuda")]
        let vision_packed_gate_up = {
            let enabled = parse_binary_env("APXINF_QWEN25_VISION_PACKED_GATE_UP")
                .map_err(Error::Other)?;
            if enabled {
                if !vision_packed_qkv {
                    return Err(Error::Other(
                        "APXINF_QWEN25_VISION_PACKED_GATE_UP=1 requires APXINF_QWEN25_VISION_PACKED_QKV=1"
                            .into(),
                    ));
                }
                vision = vision.into_packed_gate_up(&*backend)?;
                eprintln!(
                    "ApxInf Qwen2.5-Omni vision packed Gate/Up: one projection with direct activation consumption"
                );
            }
            enabled
        };
        #[cfg(not(feature = "cuda"))]
        let vision_packed_gate_up = false;
        #[cfg(feature = "cuda")]
        let long_decode_split_cta = if long_decode_split_cta_enabled()? {
            if !parse_binary_env("APXINF_TMROPE_POSITION_CACHE").map_err(Error::Other)? {
                return Err(Error::Other(
                    "APXINF_QWEN25_LONG_DECODE_SPLIT_CTA=1 requires APXINF_TMROPE_POSITION_CACHE=1"
                        .into(),
                ));
            }
            let cuda = cuda_backend(&*backend).ok_or_else(|| {
                Error::Other("Qwen2.5-Omni split-CTA decode requires CudaBackend".into())
            })?;
            if cuda.context().caps().sm != 89
                || config.text.n_heads != 16
                || config.text.n_kv_heads != 2
                || config.text.head_dim != 128
                || config.text.max_position_embeddings != 32_768
            {
                return Err(Error::Other(
                    "Qwen2.5-Omni split-CTA decode requires SM89 QH/KVH/D=16/2/128 and max context 32768"
                        .into(),
                ));
            }
            Some(cuda_kernels::qwen25_omni_attention::SplitCtaWorkspace::new(
                cuda.context(),
            )?)
        } else {
            None
        };
        #[cfg(feature = "cuda")]
        let (
            decode_graph,
            long_decode_graph,
            scaled_exp_cache_prefill,
            prefill_packed_mlp,
            prefill_packed_qkv_prelude,
        ) = {
            let fa2_chunk1024 = fa2_chunk1024_enabled()?;
            let _all_chunk_fa2 = all_chunk_fa2_enabled()?;
            if fa2_chunk1024 && !chunked_prefill_enabled()? {
                return Err(Error::Other(
                    "APXINF_QWEN25_FA2_CHUNK1024=1 requires APXINF_QWEN25_CHUNKED_PREFILL=1".into(),
                ));
            }
            let graph_enabled = decode_graph_enabled()?;
            let long_graph_enabled = long_decode_graph_enabled()?;
            let select_token = gpu_argmax_enabled()?;
            let eager_select_token = eager_gpu_argmax_enabled()?;
            let packed_qkv = packed_qkv_enabled()?;
            let m1_packed_mlp = m1_packed_mlp_enabled()?;
            let prefill_packed_mlp =
                parse_binary_env("APXINF_QWEN25_PREFILL_PACKED_MLP").map_err(Error::Other)?;
            let prefill_packed_qkv_prelude =
                parse_binary_env("APXINF_QWEN25_PREFILL_PACKED_QKV_PRELUDE")
                    .map_err(Error::Other)?;
            let short_exact_residual_norm = short_decode_exact_residual_norm_enabled()?;
            let short_w32_attention = short_decode_w32_attention_enabled()?;
            let short_fused_qkv_prelude = short_decode_fused_qkv_prelude_enabled()?;
            let short_pack8_residual_norm = short_decode_pack8_residual_norm_enabled()?;
            let scaled_exp_cache_prefill = scaled_exp_cache_prefill_enabled()?;
            let m1_gemv_tactics =
                parse_binary_env("APXINF_QWEN25_M1_GEMV_TACTICS").map_err(Error::Other)?;
            let fused_tmrope_kv = fused_tmrope_kv_enabled()?;
            if select_token && !graph_enabled {
                return Err(Error::Other(
                    "APXINF_QWEN25_GPU_ARGMAX requires APXINF_QWEN25_DECODE_GRAPH=1".into(),
                ));
            }
            if eager_select_token && (!select_token || !gpu_last_row_enabled()?) {
                return Err(Error::Other(
                    "APXINF_QWEN25_EAGER_GPU_ARGMAX requires GPU_ARGMAX and GPU_LAST_ROW".into(),
                ));
            }
            if packed_qkv && !graph_enabled {
                return Err(Error::Other(
                    "APXINF_QWEN25_PACKED_QKV requires APXINF_QWEN25_DECODE_GRAPH=1".into(),
                ));
            }
            if m1_packed_mlp && !graph_enabled {
                return Err(Error::Other(
                    "APXINF_QWEN25_M1_PACKED_MLP=1 requires APXINF_QWEN25_DECODE_GRAPH=1".into(),
                ));
            }
            if prefill_packed_mlp && (!m1_packed_mlp || !fused_silu_mul_enabled()?) {
                return Err(Error::Other(
                    "APXINF_QWEN25_PREFILL_PACKED_MLP=1 requires APXINF_QWEN25_M1_PACKED_MLP=1 and APXINF_QWEN25_FUSED_SILU_MUL=1"
                        .into(),
                ));
            }
            if prefill_packed_qkv_prelude
                && (!packed_qkv
                    || !parse_binary_env("APXINF_TMROPE_POSITION_CACHE")
                        .map_err(Error::Other)?
                    || !parse_binary_env("APXINF_TMROPE_POSITION_CACHE_PREFILL")
                        .map_err(Error::Other)?)
            {
                return Err(Error::Other(
                    "APXINF_QWEN25_PREFILL_PACKED_QKV_PRELUDE=1 requires APXINF_QWEN25_PACKED_QKV=1, APXINF_TMROPE_POSITION_CACHE=1 and APXINF_TMROPE_POSITION_CACHE_PREFILL=1"
                        .into(),
                ));
            }
            if prefill_packed_qkv_prelude
                && (config.text.n_layers != 36
                    || config.text.hidden_size != 2_048
                    || config.text.n_heads != 16
                    || config.text.n_kv_heads != 2
                    || config.text.head_dim != 128
                    || config.text.max_position_embeddings != 32_768
                    || config.text.rope_theta.to_bits() != 1_000_000.0f32.to_bits()
                    || config.text.mrope_section != [16, 24, 24])
            {
                return Err(Error::Other(
                    "APXINF_QWEN25_PREFILL_PACKED_QKV_PRELUDE requires the pinned Qwen2.5-Omni-3B text architecture"
                        .into(),
                ));
            }
            if short_exact_residual_norm
                && (!graph_enabled
                    || !select_token
                    || !packed_qkv
                    || !m1_packed_mlp
                    || !m1_gemv_tactics
                    || !fused_tmrope_kv)
            {
                return Err(Error::Other(
                    "APXINF_QWEN25_SHORT_DECODE_EXACT_RESIDUAL_NORM=1 requires DECODE_GRAPH, GPU_ARGMAX, PACKED_QKV, M1_PACKED_MLP, M1_GEMV_TACTICS and FUSED_TMROPE_KV"
                        .into(),
                ));
            }
            if short_w32_attention && !short_exact_residual_norm {
                return Err(Error::Other(
                    "APXINF_QWEN25_SHORT_DECODE_W32_ATTENTION=1 requires APXINF_QWEN25_SHORT_DECODE_EXACT_RESIDUAL_NORM=1"
                        .into(),
                ));
            }
            if short_fused_qkv_prelude && !short_w32_attention {
                return Err(Error::Other(
                    "APXINF_QWEN25_SHORT_DECODE_FUSED_QKV_PRELUDE=1 requires APXINF_QWEN25_SHORT_DECODE_W32_ATTENTION=1"
                        .into(),
                ));
            }
            if short_pack8_residual_norm && !short_fused_qkv_prelude {
                return Err(Error::Other(
                    "APXINF_QWEN25_SHORT_DECODE_PACK8_RESIDUAL_NORM=1 requires APXINF_QWEN25_SHORT_DECODE_FUSED_QKV_PRELUDE=1"
                        .into(),
                ));
            }
            if scaled_exp_cache_prefill
                && (!short_pack8_residual_norm
                    || !parse_binary_env("APXINF_BATCHED_GQA_PREFILL").map_err(Error::Other)?
                    || !parse_binary_env("APXINF_SOFTMAX_EXP_CACHE").map_err(Error::Other)?)
            {
                return Err(Error::Other(
                    "APXINF_QWEN25_SCALED_EXP_CACHE_PREFILL=1 requires SHORT_DECODE_PACK8_RESIDUAL_NORM, BATCHED_GQA_PREFILL and SOFTMAX_EXP_CACHE"
                        .into(),
                ));
            }
            if fused_tmrope_kv && !graph_enabled {
                return Err(Error::Other(
                    "APXINF_QWEN25_FUSED_TMROPE_KV requires APXINF_QWEN25_DECODE_GRAPH=1".into(),
                ));
            }
            if long_graph_enabled
                && (!graph_enabled
                    || !select_token
                    || !packed_qkv
                    || !fused_tmrope_kv
                    || long_decode_split_cta.is_none())
            {
                return Err(Error::Other(
                    "APXINF_QWEN25_LONG_DECODE_GRAPH=1 requires DECODE_GRAPH, GPU_ARGMAX, PACKED_QKV, FUSED_TMROPE_KV and LONG_DECODE_SPLIT_CTA"
                        .into(),
                ));
            }
            if graph_enabled {
                let cuda = cuda_backend(&*backend).ok_or_else(|| {
                    Error::Other("Qwen2.5-Omni decode graph requires CudaBackend".into())
                })?;
                if (select_token
                    || packed_qkv
                    || fused_tmrope_kv
                    || short_exact_residual_norm
                    || short_w32_attention
                    || short_fused_qkv_prelude
                    || short_pack8_residual_norm
                    || scaled_exp_cache_prefill)
                    && cuda.context().caps().sm != 89
                {
                    return Err(Error::Other(format!(
                        "Qwen2.5-Omni graph probes require SM89, got SM{}",
                        cuda.context().caps().sm
                    )));
                }
                if short_exact_residual_norm
                    && (config.text.n_layers != 36
                        || config.text.n_heads != 16
                        || config.text.n_kv_heads != 2
                        || config.text.head_dim != 128
                        || config.text.hidden_size != 2048
                        || config.text.intermediate_size != 11008
                        || config.text.rms_norm_eps.to_bits() != 1e-6f32.to_bits())
                {
                    return Err(Error::Other(
                        "short exact residual RMSNorm requires the pinned Qwen2.5-Omni-3B BF16 text graph"
                            .into(),
                    ));
                }
                if packed_qkv {
                    text = text.into_packed_qkv(&*backend)?;
                }
                if m1_packed_mlp {
                    text = text.into_packed_gate_up(&*backend)?;
                    eprintln!(
                        "ApxInf Qwen2.5-Omni M1 packed MLP: combined Gate/Up with exact SM89 cuBLASLt tactic"
                    );
                }
                if short_exact_residual_norm {
                    eprintln!(
                        "ApxInf Qwen2.5-Omni short-decode exact residual RMSNorm: BF16-rounded graph path"
                    );
                }
                if short_w32_attention {
                    eprintln!("ApxInf Qwen2.5-Omni short-decode W32 attention: 32 split-K warps");
                }
                if short_fused_qkv_prelude {
                    eprintln!(
                        "ApxInf Qwen2.5-Omni short-decode fused QKV prelude: bias, TMRoPE and KV publish"
                    );
                }
                if short_pack8_residual_norm {
                    eprintln!(
                        "ApxInf Qwen2.5-Omni short-decode pack8 residual RMSNorm: aligned H2048 path"
                    );
                }
                if scaled_exp_cache_prefill {
                    eprintln!(
                        "ApxInf Qwen2.5-Omni scaled exp-cache prefill: exact fused scale and numerator-cache softmax for 2..=4096 KV tokens"
                    );
                }
                if prefill_packed_mlp {
                    eprintln!(
                        "ApxInf Qwen2.5-Omni prefill packed MLP: exact rows 512, 1024 and 1760"
                    );
                }
                if prefill_packed_qkv_prelude {
                    eprintln!(
                        "ApxInf Qwen2.5-Omni prefill packed QKV prelude: exact rows 1024"
                    );
                }
                let short = Qwen25OmniDecodeGraph::new(
                    cuda,
                    Self::decode_graph_config(&config),
                    select_token,
                    fused_tmrope_kv,
                    short_exact_residual_norm,
                    short_w32_attention,
                    short_fused_qkv_prelude,
                    short_pack8_residual_norm,
                    false,
                )?;
                let long = if long_graph_enabled {
                    Some(Qwen25OmniDecodeGraph::new(
                        cuda,
                        Self::decode_graph_config(&config),
                        select_token,
                        fused_tmrope_kv,
                        false,
                        false,
                        false,
                        false,
                        true,
                    )?)
                } else {
                    None
                };
                (
                    Some(short),
                    long,
                    scaled_exp_cache_prefill,
                    prefill_packed_mlp,
                    prefill_packed_qkv_prelude,
                )
            } else {
                (None, None, false, false, false)
            }
        };
        let model = Self {
            config,
            text,
            vision,
            audio,
            backend,
            kv,
            rope_delta: 0,
            vision_qkv_bias_rope,
            vision_fused_silu_mul,
            vision_bias_residual,
            vision_gate_up_bias_silu_mul,
            vision_grouped_qkv_layout,
            vision_packed_qkv,
            vision_packed_gate_up,
            #[cfg(feature = "cuda")]
            decode_graph,
            #[cfg(feature = "cuda")]
            long_decode_graph,
            #[cfg(feature = "cuda")]
            long_decode_split_cta,
            #[cfg(feature = "cuda")]
            scaled_exp_cache_prefill,
            #[cfg(feature = "cuda")]
            prefill_packed_mlp,
            #[cfg(feature = "cuda")]
            prefill_packed_qkv_prelude,
        };
        #[cfg(feature = "cuda")]
        {
            let mut model = model;
            model.prewarm_decode_graph()?;
            Ok(model)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Ok(model)
        }
    }

    pub fn config(&self) -> &Qwen25OmniConfig {
        &self.config
    }

    #[cfg(feature = "cuda")]
    fn decode_graph_config(config: &Qwen25OmniConfig) -> Qwen25OmniDecodeGraphConfig {
        let text = &config.text;
        Qwen25OmniDecodeGraphConfig {
            n_layers: text.n_layers,
            n_heads: text.n_heads,
            n_kv_heads: text.n_kv_heads,
            head_dim: text.head_dim,
            hidden_size: text.hidden_size,
            intermediate_size: text.intermediate_size,
            vocab_size: text.vocab_size,
            max_seq_len: text.max_position_embeddings,
            rope_theta: text.rope_theta,
            mrope_section: text.mrope_section,
            rms_norm_eps: text.rms_norm_eps,
        }
    }

    #[cfg(feature = "cuda")]
    fn decode_graph_weights(text: &Qwen25OmniTextWeights) -> Qwen25OmniDecodeGraphWeights<'_> {
        Qwen25OmniDecodeGraphWeights {
            token_embedding: &text.token_embedding,
            layers: text
                .layers
                .iter()
                .map(|layer| Qwen25OmniDecodeLayerWeights {
                    attn_norm: &layer.attn_norm,
                    qkv: match &layer.qkv {
                        Qwen25OmniQkvWeights::Separate {
                            wq,
                            bq,
                            wk,
                            bk,
                            wv,
                            bv,
                        } => Qwen25OmniDecodeQkvWeights::Separate {
                            wq,
                            bq,
                            wk,
                            bk,
                            wv,
                            bv,
                        },
                        Qwen25OmniQkvWeights::Packed { weight, bias } => {
                            Qwen25OmniDecodeQkvWeights::Packed { weight, bias }
                        }
                    },
                    wo: &layer.wo,
                    ffn_norm: &layer.ffn_norm,
                    w_gate: &layer.w_gate,
                    w_up: &layer.w_up,
                    gate_up_packed: layer.gate_up_packed.as_ref(),
                    w_down: &layer.w_down,
                })
                .collect(),
            output_norm: &text.output_norm,
            lm_head: &text.lm_head,
        }
    }

    #[cfg(feature = "cuda")]
    fn prewarm_decode_graph(&mut self) -> Result<()> {
        let weights = Self::decode_graph_weights(&self.text);
        let cuda =
            cuda_backend(&*self.backend).expect("Qwen2.5-Omni decode graph owns a CudaBackend");
        if let Some(graph) = self.decode_graph.as_mut() {
            graph.prewarm(cuda, &weights, &mut *self.kv)?;
        }
        if let Some(graph) = self.long_decode_graph.as_mut() {
            graph.prewarm(cuda, &weights, &mut *self.kv)?;
        }
        Ok(())
    }

    fn forward_inner(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        self.forward_inner_with_fa2_policy(token_ids, start_pos, false)
    }

    fn forward_inner_with_fa2_policy(
        &mut self,
        token_ids: &[u32],
        start_pos: u32,
        use_all_chunk_fa2: bool,
    ) -> Result<Tensor> {
        self.validate_forward_input(token_ids, start_pos)?;
        if use_all_chunk_fa2 && token_ids.len() <= 1 {
            return Err(Error::Other(
                "all-chunk causal FA2 requires multi-token prefill".into(),
            ));
        }
        #[cfg(feature = "cuda")]
        if token_ids.len() == 1
            && use_long_decode_graph(start_pos, self.long_decode_graph.is_some())
        {
            static PATH_LOGGED: OnceLock<()> = OnceLock::new();
            if PATH_LOGGED.set(()).is_ok() {
                eprintln!(
                    "ApxInf Qwen2.5-Omni long-decode CUDA Graph: pos={start_pos}, query_heads_per_cta=4, splits=64"
                );
            }
            let positions = linear_positions(1, start_pos, self.rope_delta)?;
            let coordinates = [positions[0], positions[1], positions[2]];
            let weights = Self::decode_graph_weights(&self.text);
            let cuda = cuda_backend(&*self.backend)
                .expect("Qwen2.5-Omni long decode graph owns a CudaBackend");
            let logits = self.long_decode_graph.as_mut().unwrap().decode(
                cuda,
                &weights,
                &mut *self.kv,
                token_ids[0],
                coordinates,
                start_pos,
            )?;
            self.kv.advance(1);
            return Ok(logits);
        }
        #[cfg(feature = "cuda")]
        if token_ids.len() == 1
            && start_pos < MAX_DECODE_GRAPH_POSITION
            && self.decode_graph.is_some()
        {
            let positions = linear_positions(1, start_pos, self.rope_delta)?;
            let coordinates = [positions[0], positions[1], positions[2]];
            let weights = Self::decode_graph_weights(&self.text);
            let cuda =
                cuda_backend(&*self.backend).expect("Qwen2.5-Omni decode graph owns a CudaBackend");
            let logits = self.decode_graph.as_mut().unwrap().decode(
                cuda,
                &weights,
                &mut *self.kv,
                token_ids[0],
                coordinates,
                start_pos,
            )?;
            self.kv.advance(1);
            return Ok(logits);
        }
        let hidden = self.forward_text_hidden_validated_with_fa2_policy(
            token_ids,
            start_pos,
            use_all_chunk_fa2,
        )?;
        self.logits_last_row(&hidden)
    }

    fn validate_forward_input(&self, token_ids: &[u32], start_pos: u32) -> Result<()> {
        if token_ids.is_empty() {
            return Err(Error::Other("qwen2.5-omni forward: empty token_ids".into()));
        }
        let expected_start = self.kv.seq_len();
        if start_pos as usize != expected_start {
            return Err(Error::Other(format!(
                "qwen2.5-omni cache position mismatch: start_pos={start_pos}, cache={expected_start}"
            )));
        }
        if start_pos as usize + token_ids.len() > self.config.text.max_position_embeddings {
            return Err(Error::Other(
                "qwen2.5-omni forward exceeds context capacity".into(),
            ));
        }
        reject_video(token_ids, self.config.video_token_id)?;
        Ok(())
    }

    /// Run an already-validated ordinary text slice through KV publication.
    /// The caller decides whether the final hidden row needs an LM-head result.
    fn forward_text_hidden_validated(
        &mut self,
        token_ids: &[u32],
        start_pos: u32,
    ) -> Result<Tensor> {
        self.forward_text_hidden_validated_with_fa2_policy(token_ids, start_pos, false)
    }

    fn forward_text_hidden_validated_with_fa2_policy(
        &mut self,
        token_ids: &[u32],
        start_pos: u32,
        use_all_chunk_fa2: bool,
    ) -> Result<Tensor> {
        let mut hidden = self
            .backend
            .embedding(&self.text.token_embedding, token_ids)?;
        let positions = linear_positions(token_ids.len(), start_pos, self.rope_delta)?;
        for index in 0..self.config.text.n_layers {
            hidden = self.forward_layer(&hidden, index, &positions, use_all_chunk_fa2)?;
        }
        self.kv.advance(token_ids.len());
        Ok(hidden)
    }

    fn prefill_inner(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        if self.kv.seq_len() != 0 {
            return Err(Error::Other(
                "qwen2.5-omni prefill requires reset state".into(),
            ));
        }
        if input.token_ids.is_empty() {
            return Err(Error::Other("qwen2.5-omni prefill: empty token_ids".into()));
        }
        if input.token_ids.len() > self.config.text.max_position_embeddings {
            return Err(Error::Other(
                "qwen2.5-omni prompt exceeds context capacity".into(),
            ));
        }
        reject_unsupported_media_combination(input)?;
        reject_video(input.token_ids, self.config.video_token_id)?;
        let image_positions = if let Some(image) = input.image {
            vision::validate_input(&self.config, image.pixel_values, image.grid_thw)?;
            let expected =
                vision::merged_token_count(image.grid_thw, self.config.vision.spatial_merge_size)?;
            let positions = token_positions(input.token_ids, self.config.image_token_id);
            if positions.len() != expected {
                return Err(Error::Other(format!(
                    "qwen2.5-omni image placeholders {} != encoded tokens {expected}",
                    positions.len()
                )));
            }
            Some(positions)
        } else {
            if input.token_ids.contains(&self.config.image_token_id) {
                return Err(Error::Other(
                    "qwen2.5-omni image placeholders require image input".into(),
                ));
            }
            None
        };
        let audio_positions = if let Some(audio_input) = input.audio {
            let expected = audio::validate_input(&self.config, audio_input)?;
            let positions = token_positions(input.token_ids, self.config.audio_token_id);
            if positions.len() != expected {
                return Err(Error::Other(format!(
                    "qwen2.5-omni audio placeholders {} != encoded tokens {expected}",
                    positions.len()
                )));
            }
            let boundaries = audio_boundary_positions(
                input.token_ids,
                self.config.audio_start_token_id,
                self.config.audio_token_id,
                self.config.audio_end_token_id,
                expected,
            )?;
            Some((positions, boundaries))
        } else {
            if input.token_ids.contains(&self.config.audio_token_id)
                || input.token_ids.contains(&self.config.audio_start_token_id)
                || input.token_ids.contains(&self.config.audio_end_token_id)
            {
                return Err(Error::Other(
                    "qwen2.5-omni audio markers require audio input".into(),
                ));
            }
            None
        };

        #[cfg(feature = "cuda")]
        if image_positions.is_none()
            && audio_positions.is_none()
            && input.token_ids.len() > CHUNKED_PREFILL_THRESHOLD
            && chunked_prefill_enabled()?
        {
            return self.prefill_text_chunked(input.token_ids);
        }

        // Every processor shape, placeholder count, and modality marker is
        // validated above so malformed media fails before any backend work.
        let mut hidden = self
            .backend
            .embedding(&self.text.token_embedding, input.token_ids)?;

        if let (Some(image), Some(positions)) = (input.image, image_positions.as_ref()) {
            let encoded = vision::forward(
                &self.config,
                &self.vision,
                &*self.backend,
                image.pixel_values,
                image.grid_thw,
                self.vision_qkv_bias_rope,
                self.vision_fused_silu_mul,
                self.vision_bias_residual,
                self.vision_gate_up_bias_silu_mul,
                self.vision_grouped_qkv_layout,
                self.vision_packed_qkv,
                self.vision_packed_gate_up,
            )?;
            hidden = scatter_replace(&hidden, &positions, &encoded, &*self.backend)?;
        }

        if let (Some(audio_input), Some((positions, boundaries))) =
            (input.audio, audio_positions.as_ref())
        {
            let encoded = audio::forward(&self.config, &self.audio, &*self.backend, audio_input)?;
            hidden = scatter_replace(&hidden, &positions, &encoded, &*self.backend)?;
            hidden = scatter_replace(
                &hidden,
                boundaries,
                self.audio.boundary_embeddings(),
                &*self.backend,
            )?;
        }

        let positions = multimodal_positions(&self.config, input)?;
        let max_position = positions
            .chunks_exact(3)
            .map(|position| position[0].max(position[1]).max(position[2]))
            .max()
            .unwrap_or(0) as i64;
        self.rope_delta = max_position + 1 - input.token_ids.len() as i64;
        for index in 0..self.config.text.n_layers {
            hidden = self.forward_layer(&hidden, index, &positions, false)?;
        }
        self.kv.advance(input.token_ids.len());
        self.logits_last_row(&hidden)
    }

    #[cfg(feature = "cuda")]
    fn prefill_text_chunked(&mut self, token_ids: &[u32]) -> Result<Tensor> {
        if self.rope_delta != 0 || self.kv.seq_len() != 0 {
            return Err(Error::Other(
                "Qwen2.5-Omni chunked prefill requires reset text-only state".into(),
            ));
        }
        let fa2_chunk1024 = fa2_chunk1024_enabled()?;
        let all_chunk_fa2 = use_all_chunk_fa2(token_ids.len(), all_chunk_fa2_enabled()?);
        let chunk_size = text_prefill_chunk_size(token_ids.len(), fa2_chunk1024);
        if fa2_chunk1024
            && (CHUNKED_PREFILL_FA2_LARGE_MIN_PROMPT..=CHUNKED_PREFILL_FA2_LARGE_MAX_PROMPT)
                .contains(&token_ids.len())
        {
            static PATH_LOGGED: OnceLock<()> = OnceLock::new();
            if PATH_LOGGED.set(()).is_ok() {
                eprintln!(
                    "ApxInf Qwen2.5-Omni FA2 chunk1024 prefill: prompt={}, chunk={chunk_size}",
                    token_ids.len()
                );
            }
        }
        if all_chunk_fa2 {
            static ALL_CHUNKS_PATH_LOGGED: OnceLock<()> = OnceLock::new();
            if ALL_CHUNKS_PATH_LOGGED.set(()).is_ok() {
                eprintln!(
                    "ApxInf Qwen2.5-Omni request-scoped all-chunk FA2: prompt={}, chunk={chunk_size}",
                    token_ids.len()
                );
            }
        }
        let chunks = token_ids.len().div_ceil(chunk_size);
        for (index, chunk) in token_ids.chunks(chunk_size).enumerate() {
            let start = u32::try_from(self.kv.seq_len())
                .map_err(|_| Error::Other("chunked prefill position exceeds u32".into()))?;
            if index + 1 == chunks {
                return self.forward_inner_with_fa2_policy(chunk, start, all_chunk_fa2);
            }
            self.forward_text_hidden_validated_with_fa2_policy(chunk, start, all_chunk_fa2)?;
        }
        Err(Error::Other("chunked prefill received no tokens".into()))
    }

    fn forward_layer(
        &mut self,
        hidden: &Tensor,
        index: usize,
        positions: &[u32],
        use_all_chunk_fa2: bool,
    ) -> Result<Tensor> {
        let text = &self.config.text;
        let sequence = hidden.shape().dims()[0];
        let layer = &self.text.layers[index];
        let normalized = self
            .backend
            .rms_norm(hidden, &layer.attn_norm, text.rms_norm_eps)?;
        #[cfg(feature = "cuda")]
        let fuse_prefill_packed_qkv = use_prefill_packed_qkv_prelude(
            sequence,
            self.prefill_packed_qkv_prelude,
        );
        #[cfg(not(feature = "cuda"))]
        let fuse_prefill_packed_qkv = false;
        #[cfg(not(feature = "cuda"))]
        let _ = fuse_prefill_packed_qkv;
        let (q, k, v, kv_published) = match &layer.qkv {
            Qwen25OmniQkvWeights::Separate {
                wq,
                bq,
                wk,
                bk,
                wv,
                bv,
            } => (
                self.backend
                    .add_bias(&self.backend.matmul(&normalized, wq)?, bq)?,
                Some(
                    self.backend
                        .add_bias(&self.backend.matmul(&normalized, wk)?, bk)?,
                ),
                Some(
                    self.backend
                        .add_bias(&self.backend.matmul(&normalized, wv)?, bv)?,
                ),
                false,
            ),
            #[cfg(feature = "cuda")]
            Qwen25OmniQkvWeights::Packed { weight, bias } => {
                let packed = self.backend.matmul(&normalized, weight)?;
                let cuda = cuda_backend(&*self.backend).ok_or_else(|| {
                    Error::Other("Qwen2.5-Omni packed QKV requires CudaBackend".into())
                })?;
                if fuse_prefill_packed_qkv {
                    let query = cuda.qwen25_omni_prefill_packed_qkv_prelude(
                        &packed,
                        bias,
                        &mut *self.kv,
                        index,
                        positions,
                    )?;
                    (query, None, None, true)
                } else {
                    let split = cuda_kernels::attention::split_gqa_qkv_bias_bf16(
                        cuda.context(),
                        &packed,
                        Some(bias),
                        text.n_heads,
                        text.n_kv_heads,
                        text.head_dim,
                    )?;
                    (split.q, Some(split.k), Some(split.v), false)
                }
            }
            #[cfg(not(feature = "cuda"))]
            Qwen25OmniQkvWeights::Packed { .. } => {
                return Err(Error::Other("Qwen2.5-Omni packed QKV requires CUDA".into()))
            }
        };
        let q = if kv_published {
            q
        } else {
            self.backend.rope_tmrope(
                &q.reshape(vec![sequence, text.n_heads, text.head_dim])?,
                text.n_heads,
                text.head_dim,
                text.rope_theta,
                text.mrope_section,
                positions,
            )?
        };
        if !kv_published {
            let k = k
                .ok_or_else(|| Error::Other("unpublished QKV path is missing K".into()))?
                .reshape(vec![sequence, text.n_kv_heads, text.head_dim])?;
            let v = v
                .ok_or_else(|| Error::Other("unpublished QKV path is missing V".into()))?
                .reshape(vec![sequence, text.n_kv_heads, text.head_dim])?;
            let k = self.backend.rope_tmrope(
                &k,
                text.n_kv_heads,
                text.head_dim,
                text.rope_theta,
                text.mrope_section,
                positions,
            )?;
            self.backend
                .kv_append(&mut *self.kv, index, &k, &v, sequence)?;
        }
        let kv_len = self.kv.seq_len() + sequence;
        let attention = if sequence == 1 {
            #[cfg(feature = "cuda")]
            {
                if let Some(workspace) = self.long_decode_split_cta.as_ref() {
                    if use_long_decode_split_cta(kv_len, true) {
                        static PATH_LOGGED: OnceLock<()> = OnceLock::new();
                        if PATH_LOGGED.set(()).is_ok() {
                            eprintln!(
                                "ApxInf Qwen2.5-Omni long-decode grouped-GQA split-CTA: kv={kv_len}, query_heads_per_cta=4, splits={LONG_DECODE_SPLIT_CTA_COUNT}"
                            );
                        }
                        let cuda = cuda_backend(&*self.backend).ok_or_else(|| {
                            Error::Other("split-CTA decode requires CudaBackend".into())
                        })?;
                        cuda.qwen25_omni_grouped_split_cta_decode(
                            &q,
                            &mut *self.kv,
                            index,
                            workspace,
                            LONG_DECODE_SPLIT_CTA_COUNT,
                            kv_len,
                            text.max_position_embeddings,
                        )?
                    } else {
                        self.backend.sdpa_decode(
                            &q,
                            &mut *self.kv,
                            index,
                            text.n_heads,
                            text.n_kv_heads,
                            text.head_dim,
                            kv_len,
                            text.max_position_embeddings,
                        )?
                    }
                } else {
                    self.backend.sdpa_decode(
                        &q,
                        &mut *self.kv,
                        index,
                        text.n_heads,
                        text.n_kv_heads,
                        text.head_dim,
                        kv_len,
                        text.max_position_embeddings,
                    )?
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                self.backend.sdpa_decode(
                    &q,
                    &mut *self.kv,
                    index,
                    text.n_heads,
                    text.n_kv_heads,
                    text.head_dim,
                    kv_len,
                    text.max_position_embeddings,
                )?
            }
        } else {
            #[cfg(feature = "cuda")]
            {
                if use_all_chunk_fa2 {
                    let cuda = cuda_backend(&*self.backend).ok_or_else(|| {
                        Error::Other("all-chunk causal FA2 requires CudaBackend".into())
                    })?;
                    cuda.sdpa_prefill_causal_fa2(
                        &q,
                        &mut *self.kv,
                        index,
                        text.n_heads,
                        text.n_kv_heads,
                        text.head_dim,
                        kv_len,
                        text.max_position_embeddings,
                    )?
                } else if use_scaled_exp_cache_prefill(
                    sequence,
                    kv_len,
                    self.scaled_exp_cache_prefill,
                    use_all_chunk_fa2,
                ) {
                    let cuda = cuda_backend(&*self.backend).ok_or_else(|| {
                        Error::Other("scaled exp-cache prefill requires CudaBackend".into())
                    })?;
                    cuda.qwen25_omni_sdpa_prefill_scaled_exp_cache(
                        &q,
                        &mut *self.kv,
                        index,
                        text.n_heads,
                        text.n_kv_heads,
                        text.head_dim,
                        kv_len,
                        text.max_position_embeddings,
                    )?
                } else {
                    self.backend.sdpa_prefill(
                        &q,
                        &mut *self.kv,
                        index,
                        text.n_heads,
                        text.n_kv_heads,
                        text.head_dim,
                        kv_len,
                        text.max_position_embeddings,
                    )?
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = use_all_chunk_fa2;
                self.backend.sdpa_prefill(
                    &q,
                    &mut *self.kv,
                    index,
                    text.n_heads,
                    text.n_kv_heads,
                    text.head_dim,
                    kv_len,
                    text.max_position_embeddings,
                )?
            }
        };
        let attention = self.backend.matmul(&attention, &layer.wo)?;
        let residual = self.backend.add(hidden, &attention)?;
        let normalized = self
            .backend
            .rms_norm(&residual, &layer.ffn_norm, text.rms_norm_eps)?;
        #[cfg(feature = "cuda")]
        let activated = if use_prefill_packed_mlp(sequence, self.prefill_packed_mlp) {
            let weight = layer.gate_up_packed.as_ref().ok_or_else(|| {
                Error::Other("prefill packed MLP selector has no packed Gate/Up weight".into())
            })?;
            let gate_up = self.backend.matmul(&normalized, weight)?;
            let cuda = cuda_backend(&*self.backend).ok_or_else(|| {
                Error::Other("prefill packed MLP requires CudaBackend".into())
            })?;
            cuda.qwen25_omni_prefill_packed_silu_mul_exact(
                &gate_up,
                text.intermediate_size,
            )?
        } else {
            let gate = self.backend.matmul(&normalized, &layer.w_gate)?;
            let up = self.backend.matmul(&normalized, &layer.w_up)?;
            if fused_silu_mul_enabled()? {
                self.backend.silu_mul(&gate, &up)?
            } else {
                self.backend.mul(&self.backend.silu(&gate)?, &up)?
            }
        };
        #[cfg(not(feature = "cuda"))]
        let activated = {
            let gate = self.backend.matmul(&normalized, &layer.w_gate)?;
            let up = self.backend.matmul(&normalized, &layer.w_up)?;
            self.backend.mul(&self.backend.silu(&gate)?, &up)?
        };
        let mlp = self.backend.matmul(&activated, &layer.w_down)?;
        self.backend.add(&residual, &mlp)
    }

    fn logits_last_row(&self, hidden: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if gpu_last_row_enabled()? {
            if cuda_backend(&*self.backend).is_none() {
                return Err(Error::Other(
                    "Qwen2.5-Omni GPU last-row path requires CUDA".into(),
                ));
            }
            return self.logits_last_row_cuda_device(hidden);
        }
        let hidden = self.backend.rms_norm(
            hidden,
            &self.text.output_norm,
            self.config.text.rms_norm_eps,
        )?;
        // Avoid materializing `[prompt, vocab]`: only the last position owns
        // the next-token distribution. The small hidden row is copied, then
        // returned to the model device for the separate native LM head.
        let cpu = self.backend.to_cpu(&hidden)?;
        let rows = cpu.shape().dims()[0];
        let width = self.config.text.hidden_size;
        let values = cpu.to_f32_vec()?;
        let last = &values[(rows - 1) * width..rows * width];
        let row = match cpu.dtype() {
            apxinf_core::DType::BF16 => Tensor::from_bf16(
                vec![1, width],
                &last
                    .iter()
                    .copied()
                    .map(half::bf16::from_f32)
                    .collect::<Vec<_>>(),
            )?,
            apxinf_core::DType::F32 => Tensor::from_f32(vec![1, width], last)?,
            dtype => {
                return Err(Error::Other(format!(
                    "qwen2.5-omni final hidden dtype {dtype} is unsupported"
                )))
            }
        };
        let row = self.backend.to_device(&row)?;
        let logits = self.backend.matmul(&row, &self.text.lm_head)?;
        self.backend.synchronize()?;
        let logits = self.backend.to_cpu(&logits)?;
        Tensor::from_f32(vec![1, self.config.text.vocab_size], &logits.to_f32_vec()?)
    }

    #[cfg(feature = "cuda")]
    fn logits_last_row_cuda_device(&self, hidden: &Tensor) -> Result<Tensor> {
        let dims = hidden.shape().dims();
        let width = self.config.text.hidden_size;
        if hidden.dtype() != apxinf_core::DType::BF16
            || dims.len() != 2
            || dims[0] == 0
            || dims[1] != width
        {
            return Err(Error::Other(format!(
                "Qwen2.5-Omni GPU last row expected nonempty BF16 [rows,{width}], got {:?} {}",
                dims,
                hidden.dtype()
            )));
        }
        let row_bytes = width
            .checked_mul(apxinf_core::DType::BF16.size_in_bytes())
            .ok_or_else(|| Error::Other("Qwen2.5-Omni last-row byte size overflow".into()))?;
        let byte_offset = (dims[0] - 1)
            .checked_mul(row_bytes)
            .ok_or_else(|| Error::Other("Qwen2.5-Omni last-row offset overflow".into()))?;
        let row = DeviceBuffer::from_tensor(hidden)
            .and_then(|buffer| buffer.view(byte_offset, row_bytes))
            .map_err(Error::Cuda)?
            .into_tensor(
                apxinf_core::Shape::new(vec![1, width]),
                apxinf_core::DType::BF16,
            );
        let row =
            self.backend
                .rms_norm(&row, &self.text.output_norm, self.config.text.rms_norm_eps)?;
        self.backend.matmul(&row, &self.text.lm_head)
    }

    #[cfg(feature = "cuda")]
    fn replay_decode_token(&mut self, token: u32, pos: u32, long: bool) -> Result<u32> {
        let expected_start = self.kv.seq_len();
        if pos as usize != expected_start {
            return Err(Error::Other(format!(
                "qwen2.5-omni cache position mismatch: start_pos={pos}, cache={expected_start}"
            )));
        }
        if pos as usize + 1 > self.config.text.max_position_embeddings {
            return Err(Error::Other(
                "qwen2.5-omni forward exceeds context capacity".into(),
            ));
        }
        reject_video(&[token], self.config.video_token_id)?;
        let positions = linear_positions(1, pos, self.rope_delta)?;
        let coordinates = [positions[0], positions[1], positions[2]];
        let weights = Self::decode_graph_weights(&self.text);
        let cuda =
            cuda_backend(&*self.backend).expect("Qwen2.5-Omni decode graph owns a CudaBackend");
        let graph = if long {
            static PATH_LOGGED: OnceLock<()> = OnceLock::new();
            if PATH_LOGGED.set(()).is_ok() {
                eprintln!(
                    "ApxInf Qwen2.5-Omni long-decode CUDA Graph: pos={pos}, query_heads_per_cta=4, splits=64"
                );
            }
            self.long_decode_graph.as_mut().ok_or_else(|| {
                Error::Other("Qwen2.5-Omni long decode graph is unavailable".into())
            })?
        } else {
            self.decode_graph
                .as_mut()
                .ok_or_else(|| Error::Other("Qwen2.5-Omni decode graph is unavailable".into()))?
        };
        let selected =
            graph.decode_token(cuda, &weights, &mut *self.kv, token, coordinates, pos)?;
        self.kv.advance(1);
        Ok(selected)
    }

    #[cfg(feature = "cuda")]
    fn eager_decode_token(&mut self, token: u32, pos: u32) -> Result<u32> {
        self.validate_forward_input(&[token], pos)?;
        let hidden = self.forward_text_hidden_validated(&[token], pos)?;
        let logits = self.logits_last_row_cuda_device(&hidden)?;
        let cuda =
            cuda_backend(&*self.backend).expect("Qwen2.5-Omni eager selection owns CUDA logits");
        self.decode_graph
            .as_ref()
            .expect("Qwen2.5-Omni eager selection owns decode workspace")
            .select_logits(cuda, &logits)
    }

    fn clear_state(&mut self) {
        // An eager failure may leave stream-ordered frees queued behind the
        // last successful kernel. Drain them before rebuilding the KV cache
        // so an OOM request cannot poison the following request.
        let _ = self.backend.synchronize();
        let _ = self.kv.clear();
        self.rope_delta = 0;
    }
}

impl LlmTrait for GeneralQwen25Omni {
    fn load(
        _config: apxinf_loader::ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self>
    where
        Self: Sized,
    {
        Err(Error::Other(
            "GeneralQwen25Omni owns a nested config; load through AutoModel".into(),
        ))
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        let result = self.forward_inner(token_ids, start_pos);
        if result.is_err() {
            self.clear_state();
        }
        result
    }

    fn backend(&self) -> &dyn Backend {
        &*self.backend
    }

    fn capabilities(&self) -> LlmCapabilities {
        LlmCapabilities::OMNI
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        let result = self.prefill_inner(input);
        if result.is_err() {
            self.clear_state();
        }
        result
    }

    fn reset(&mut self) {
        self.clear_state();
    }

    #[cfg(feature = "cuda")]
    fn decode_token(&mut self, token: u32, pos: u32) -> Option<Result<u32>> {
        if !self
            .decode_graph
            .as_ref()
            .is_some_and(Qwen25OmniDecodeGraph::selects_token)
        {
            return None;
        }
        if pos >= MAX_DECODE_GRAPH_POSITION {
            if use_long_decode_graph(pos, self.long_decode_graph.is_some()) {
                let result = self.replay_decode_token(token, pos, true);
                if result.is_err() {
                    self.clear_state();
                }
                return Some(result);
            }
            let enabled = match eager_gpu_argmax_enabled() {
                Ok(enabled) => enabled,
                Err(error) => return Some(Err(error)),
            };
            if !enabled {
                return None;
            }
            let result = self.eager_decode_token(token, pos);
            if result.is_err() {
                self.clear_state();
            }
            return Some(result);
        }
        let result = self.replay_decode_token(token, pos, false);
        if result.is_err() {
            self.clear_state();
        }
        Some(result)
    }

    fn vocab_size(&self) -> usize {
        self.config.text.vocab_size
    }

    fn max_context_len(&self) -> Option<usize> {
        Some(self.config.text.max_position_embeddings)
    }

    fn max_new_tokens_limit(&self) -> Option<usize> {
        Some(128)
    }
}

fn reject_video(token_ids: &[u32], video_token_id: u32) -> Result<()> {
    if token_ids.contains(&video_token_id) {
        return Err(Error::Other(
            "qwen2.5-omni video input is outside the deployed capability".into(),
        ));
    }
    Ok(())
}

fn reject_unsupported_media_combination(input: LlmInput<'_>) -> Result<()> {
    if input.image.is_some() && input.audio.is_some() {
        return Err(Error::Other(
            "qwen2.5-omni simultaneous image and audio input is outside the deployed capability"
                .into(),
        ));
    }
    Ok(())
}

fn token_positions(token_ids: &[u32], token: u32) -> Vec<usize> {
    token_ids
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value == token).then_some(index))
        .collect()
}

fn audio_boundary_positions(
    token_ids: &[u32],
    start_token: u32,
    audio_token: u32,
    end_token: u32,
    audio_count: usize,
) -> Result<Vec<usize>> {
    let starts = token_positions(token_ids, start_token);
    let ends = token_positions(token_ids, end_token);
    if starts.len() != 1 || ends.len() != 1 {
        return Err(Error::Other(format!(
            "qwen2.5-omni one audio clip requires exactly one start/end marker, got {}/{}",
            starts.len(),
            ends.len()
        )));
    }
    let start = starts[0];
    let end = ends[0];
    if end != start + audio_count + 1
        || token_ids[start + 1..end]
            .iter()
            .any(|token| *token != audio_token)
    {
        return Err(Error::Other(
            "qwen2.5-omni audio markers must enclose one contiguous placeholder run".into(),
        ));
    }
    Ok(vec![start, end])
}

fn linear_positions(length: usize, start: u32, delta: i64) -> Result<Vec<u32>> {
    let first = i64::from(start) + delta;
    if first < 0 {
        return Err(Error::Other(
            "qwen2.5-omni negative TMRoPE decode position".into(),
        ));
    }
    let mut positions = Vec::with_capacity(length * 3);
    for offset in 0..length {
        let position = u32::try_from(first + offset as i64)
            .map_err(|_| Error::Other("qwen2.5-omni TMRoPE position overflow".into()))?;
        positions.extend_from_slice(&[position, position, position]);
    }
    Ok(positions)
}

fn multimodal_positions(config: &Qwen25OmniConfig, input: LlmInput<'_>) -> Result<Vec<u32>> {
    let image_counts = input
        .image
        .map(|image| {
            image
                .grid_thw
                .iter()
                .map(|&[time, height, width]| {
                    (time as usize)
                        * (height as usize / config.vision.spatial_merge_size)
                        * (width as usize / config.vision.spatial_merge_size)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let audio_counts = input
        .audio
        .map(|audio| {
            audio
                .token_counts
                .iter()
                .map(|count| *count as usize)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let grids = input.image.map(|image| image.grid_thw).unwrap_or(&[]);
    let mut output = Vec::with_capacity(input.token_ids.len() * 3);
    let mut index = 0;
    let mut image = 0;
    let mut audio = 0;
    let mut next = 0_u32;
    while index < input.token_ids.len() {
        if input.token_ids[index] == config.image_token_id {
            let count = *image_counts
                .get(image)
                .ok_or_else(|| Error::Other("qwen2.5-omni image placeholder has no grid".into()))?;
            if input.token_ids[index..]
                .iter()
                .take(count)
                .any(|token| *token != config.image_token_id)
            {
                return Err(Error::Other(
                    "qwen2.5-omni image placeholders are not contiguous".into(),
                ));
            }
            let [time, height, width] = grids[image];
            let height = height / config.vision.spatial_merge_size as u32;
            let width = width / config.vision.spatial_merge_size as u32;
            for temporal in 0..time {
                for row in 0..height {
                    for col in 0..width {
                        output.extend_from_slice(&[next + temporal, next + row, next + col]);
                    }
                }
            }
            next += time.max(height).max(width);
            index += count;
            image += 1;
        } else if input.token_ids[index] == config.audio_token_id {
            let count = *audio_counts.get(audio).ok_or_else(|| {
                Error::Other("qwen2.5-omni audio placeholder has no feature group".into())
            })?;
            if input.token_ids[index..]
                .iter()
                .take(count)
                .any(|token| *token != config.audio_token_id)
            {
                return Err(Error::Other(
                    "qwen2.5-omni audio placeholders are not contiguous".into(),
                ));
            }
            for temporal in 0..count as u32 {
                output.extend_from_slice(&[next + temporal, next, next]);
            }
            next += count as u32;
            index += count;
            audio += 1;
        } else {
            output.extend_from_slice(&[next, next, next]);
            next += 1;
            index += 1;
        }
    }
    if image != image_counts.len() || audio != audio_counts.len() {
        return Err(Error::Other(
            "qwen2.5-omni unused media group after TMRoPE construction".into(),
        ));
    }
    if output.len() != input.token_ids.len() * 3 {
        return Err(Error::Other(
            "qwen2.5-omni TMRoPE position length drift".into(),
        ));
    }
    Ok(output)
}

fn scatter_replace(
    hidden: &Tensor,
    positions: &[usize],
    replacement: &Tensor,
    backend: &dyn Backend,
) -> Result<Tensor> {
    let hidden_cpu = backend.to_cpu(hidden)?;
    let replacement_cpu = backend.to_cpu(replacement)?;
    let width = *hidden_cpu
        .shape()
        .dims()
        .last()
        .ok_or_else(|| Error::Other("qwen2.5-omni scatter hidden is scalar".into()))?;
    if replacement_cpu.shape().dims() != [positions.len(), width] {
        return Err(Error::Other(format!(
            "qwen2.5-omni replacement shape {:?}, expected [{}, {width}]",
            replacement_cpu.shape().dims(),
            positions.len()
        )));
    }
    let mut values = hidden_cpu.to_f32_vec()?;
    let replacement = replacement_cpu.to_f32_vec()?;
    for (source, position) in positions.iter().enumerate() {
        values[*position * width..(*position + 1) * width]
            .copy_from_slice(&replacement[source * width..(source + 1) * width]);
    }
    let output = match hidden_cpu.dtype() {
        apxinf_core::DType::BF16 => Tensor::from_bf16(
            hidden_cpu.shape().dims().to_vec(),
            &values
                .into_iter()
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>(),
        )?,
        apxinf_core::DType::F32 => Tensor::from_f32(hidden_cpu.shape().dims().to_vec(), &values)?,
        dtype => {
            return Err(Error::Other(format!(
                "qwen2.5-omni scatter does not support {dtype}"
            )))
        }
    };
    backend.to_device(&output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm_trait::{AudioInput, ImageInput};

    fn config() -> Qwen25OmniConfig {
        let raw = include_str!("../../tests/data/qwen25_omni_config_minimal.json");
        let mut config = Qwen25OmniConfig::from_json_str(raw).unwrap();
        config.processor = super::super::config::Qwen25OmniProcessorConfig {
            sampling_rate: 16000,
            n_fft: 400,
            hop_length: 160,
            feature_size: 128,
        };
        config
    }

    #[test]
    fn fa2_chunk1024_changes_only_the_frozen_8k_to_12k_cell() {
        assert_eq!(text_prefill_chunk_size(2_048, false), 512);
        assert_eq!(text_prefill_chunk_size(4_096, false), 256);
        assert_eq!(text_prefill_chunk_size(8_192, false), 256);
        assert_eq!(text_prefill_chunk_size(12_288, false), 256);
        assert_eq!(text_prefill_chunk_size(12_289, false), 1_024);

        assert_eq!(text_prefill_chunk_size(7_168, true), 256);
        assert_eq!(text_prefill_chunk_size(8_192, true), 1_024);
        assert_eq!(text_prefill_chunk_size(12_288, true), 1_024);
        assert_eq!(text_prefill_chunk_size(12_289, true), 1_024);
    }

    #[test]
    fn all_chunk_fa2_changes_only_the_frozen_8k_to_12k_cell() {
        assert!(!use_all_chunk_fa2(7_168, true));
        assert!(use_all_chunk_fa2(8_192, true));
        assert!(use_all_chunk_fa2(12_288, true));
        assert!(!use_all_chunk_fa2(12_289, true));
        assert!(!use_all_chunk_fa2(8_192, false));
    }

    #[test]
    fn scaled_exp_cache_never_overrides_request_scoped_fa2() {
        assert!(use_scaled_exp_cache_prefill(1_024, 1_024, true, false));
        assert!(!use_scaled_exp_cache_prefill(1_024, 1_024, true, true));
        assert!(!use_scaled_exp_cache_prefill(1, 1, true, false));
        assert!(!use_scaled_exp_cache_prefill(1_024, 4_097, true, false));
        assert!(!use_scaled_exp_cache_prefill(1_024, 1_024, false, false));
    }

    #[test]
    fn prefill_packed_owners_use_only_exact_validated_rows() {
        for rows in [512, 1_024, 1_760] {
            assert!(use_prefill_packed_mlp(rows, true));
        }
        assert!(use_prefill_packed_qkv_prelude(1_024, true));
        assert!(!use_prefill_packed_qkv_prelude(512, true));
        assert!(!use_prefill_packed_qkv_prelude(1_760, true));
        for rows in [1, 52, 128, 256, 511, 513, 2_048] {
            assert!(!use_prefill_packed_mlp(rows, true));
            assert!(!use_prefill_packed_qkv_prelude(rows, true));
        }
        assert!(!use_prefill_packed_mlp(1_024, false));
        assert!(!use_prefill_packed_qkv_prelude(1_024, false));
    }

    #[test]
    fn split_cta_decode_changes_only_the_post_32760_cell() {
        assert!(!use_long_decode_split_cta(32_760, true));
        assert!(use_long_decode_split_cta(32_761, true));
        assert!(use_long_decode_split_cta(32_767, true));
        assert!(!use_long_decode_split_cta(32_767, false));
    }

    #[test]
    fn long_decode_graph_changes_only_the_post_32760_cell() {
        assert!(!use_long_decode_graph(32_759, true));
        assert!(use_long_decode_graph(32_760, true));
        assert!(use_long_decode_graph(32_767, true));
        assert!(!use_long_decode_graph(32_767, false));
    }

    #[test]
    fn rejects_combined_media_and_video() {
        let config = config();
        let pixels = Tensor::from_f32(vec![16, 1176], &vec![0.0; 16 * 1176]).unwrap();
        let grids = [[1, 4, 4]];
        let features = Tensor::from_f32(vec![4, 128], &vec![0.0; 4 * 128]).unwrap();
        let mask = Tensor::from_f32(vec![4], &[1.0; 4]).unwrap();
        let lengths = [4];
        let audio_counts = [2];
        let tokens = [1, 151655, 151655, 151655, 151655, 2, 151646, 151646, 3];
        let input = LlmInput::with_media(
            &tokens,
            Some(ImageInput::new(&pixels, &grids)),
            Some(AudioInput::new(&features, &mask, &lengths, &audio_counts)),
        );
        assert!(reject_unsupported_media_combination(input).is_err());
        assert!(reject_video(&[config.video_token_id], config.video_token_id).is_err());
    }

    #[test]
    fn validates_audio_boundaries_and_multimodal_positions() {
        let config = config();
        let audio_tokens = [10, 151647, 151646, 151646, 151646, 151648, 11];
        assert_eq!(
            audio_boundary_positions(&audio_tokens, 151647, 151646, 151648, 3).unwrap(),
            [1, 5]
        );
        assert!(audio_boundary_positions(
            &[10, 151647, 151646, 12, 151646, 151648],
            151647,
            151646,
            151648,
            3
        )
        .is_err());

        let pixels = Tensor::from_f32(vec![16, 1176], &vec![0.0; 16 * 1176]).unwrap();
        let image_grid = [[1, 4, 4]];
        let image_tokens = [10, 151655, 151655, 151655, 151655, 11];
        let image_positions = multimodal_positions(
            &config,
            LlmInput::with_image(&image_tokens, ImageInput::new(&pixels, &image_grid)),
        )
        .unwrap();
        assert_eq!(
            image_positions,
            [0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 3, 3, 3]
        );

        let features = Tensor::from_f32(vec![7, 128], &vec![0.0; 7 * 128]).unwrap();
        let mask = Tensor::from_f32(vec![7], &[1.0; 7]).unwrap();
        let lengths = [7];
        let counts = [2];
        let positioned_audio = [10, 151647, 151646, 151646, 151648, 11];
        let audio_positions = multimodal_positions(
            &config,
            LlmInput::with_audio(
                &positioned_audio,
                AudioInput::new(&features, &mask, &lengths, &counts),
            ),
        )
        .unwrap();
        assert_eq!(
            audio_positions,
            [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 2, 2, 4, 4, 4, 5, 5, 5]
        );
    }
}

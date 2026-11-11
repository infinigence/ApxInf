//! d3 forward on native MLX/Metal: one lazy request graph, compiled per
//! prompt length, evaluated once. No decode and no KV cache: the GDN
//! recurrence state and the attention K/V are created and consumed inside a
//! single request and never survive it.
//!
//! Semantics follow the checkpoint reference (`modeling_d3.py`) op for op,
//! including every rounding point: `1 + w` norms, the depthwise causal conv
//! over three zero rows, the official GDN q/k l2-norm (sum epsilon `1e-6`, not
//! the mlx-lm mean-epsilon variant), the FP32 gated norm in the GDN block,
//! non-causal SDPA over the full prompt, the output gate on the attention
//! path, and the FP32 255-code readout at the last position.
//!
//! `GatedDeltaRecurrence` (four value rows per SIMD group) and `HalfSplitRope`
//! (FP32 angles, one rounding to the input dtype) are the apxinf-mlx kernels
//! for those primitives. The causal convolution is MLX `conv1d` over an
//! explicit zero history.
//!
//! The readout reads only the last position, and the final layer is a
//! full-attention layer, so its query, output projection and MLP run on that
//! one row against the whole prompt's keys and values ([`FinalLayerRows`]).
//! This is the same function of the inputs, but not the same rounding: MLX
//! sends one query row at head dimension 256 to its fused single-query
//! attention kernel, which keeps scores and softmax weights in FP32, where a
//! longer query takes the unfused path that rounds both to BF16; and one-row
//! projections run as GEMV, which sums the same FP32 products in another order.

use std::rc::Rc;

use apxinf_core::{Error, Result};
use apxinf_mlx::fusions::{GatedDeltaRecurrence, HalfSplitRope};
use apxinf_mlx::{Array, Compiled, MlxDType, Stream};

use super::config::{Config, Geometry, MAX_INPUT_TOKENS};
use super::profiles::{ProfileCache, ProfileLimits};
use super::vision::Vision;
use super::weights::{Attention, GatedDelta, Mixer, Mlp, VisionWeights, Weights};

/// Everything the compiled callables capture: immutable geometry, evaluated
/// weights, registered kernels, the eager vision tower and the
/// request-independent inverse frequencies. `Rc` lets every compiled profile
/// and the runner share one copy of the device arrays.
struct Parts {
    geometry: Geometry,
    weights: Weights,
    vision: Vision,
    image_token_id: u32,
    video_token_id: u32,
    vision_merge: usize,
    recurrence: GatedDeltaRecurrence,
    rope: HalfSplitRope,
    inv_freq: Array,
    stream: Stream,
}

/// Which rows the final layer computes. `Readout` is the served shape: only
/// the last position, the one the readout pools. `All` is the reference shape
/// that computes every row and discards all but the last; it exists so the
/// eager path can measure what the pruning changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinalLayerRows {
    All,
    Readout,
}

/// Image profiles are keyed by prompt length and merged image-token count.
type ImageProfileKey = (usize, usize);

/// One d3 execution over shared weights. Each profile owns the compiled
/// callable for one exact input shape: text takes `ids [1, L]` and
/// `count [1]`; image adds `positions [3, L]`, `features [1, M, D]` and
/// `run_start [1]`. The candidate count is an input value, not a shape, so it
/// does not select a profile. Evicting a profile drops its callable.
pub struct Model {
    parts: Rc<Parts>,
    text_profiles: ProfileCache<usize, Compiled>,
    image_profiles: ProfileCache<ImageProfileKey, Compiled>,
}

fn scalar(like: &Array, value: f64) -> Result<Array> {
    Array::scalar(like.stream(), value as f32, like.dtype())
}

/// `x * sigmoid(x)`.
fn silu(x: &Array) -> Result<Array> {
    x.mul(&x.sigmoid()?)
}

/// `nn.Linear` without bias: `x @ W.T` (`W` is `[out, in]` as stored).
fn project(x: &Array, weight: &Array) -> Result<Array> {
    x.matmul(&weight.transpose(&[1, 0])?)
}

/// `compute_g`: `exp(-exp(A_log_f32) * softplus(a + dt_bias))`; beta:
/// `sigmoid(b)`. `A_log` is promoted to F32 before `exp`, exactly as the
/// vendored reference does; the softplus stays in the operand dtype (BF16).
fn decay_beta(a_log: &Array, a: &Array, b: &Array, dt_bias: &Array) -> Result<(Array, Array)> {
    let shifted = a.add(dt_bias)?;
    let softplus = shifted.logaddexp(&scalar(&shifted, 0.0)?)?;
    let decay = a_log
        .cast(MlxDType::F32)?
        .exp()?
        .neg()?
        .mul(&softplus)?
        .exp()?;
    Ok((decay, b.sigmoid()?))
}

impl Model {
    /// `text_limits` bounds the text profiles; image profiles keep the
    /// default bound.
    pub fn new(
        config: &Config,
        stream: Stream,
        weights: Weights,
        vision_weights: VisionWeights,
        text_limits: ProfileLimits,
    ) -> Result<Self> {
        let geometry = config.geometry;
        let vision = Vision::new(config, stream.clone(), vision_weights)?;
        let parts = Rc::new(Parts {
            geometry,
            weights,
            vision,
            image_token_id: config.image_token_id,
            video_token_id: config.video_token_id,
            vision_merge: config.vision.spatial_merge,
            recurrence: GatedDeltaRecurrence::new(&stream)?,
            rope: HalfSplitRope::new(&stream, geometry.rope_dims)?,
            inv_freq: inverse_frequencies(&stream, geometry)?,
            stream,
        });
        Ok(Self {
            parts,
            text_profiles: ProfileCache::new(text_limits),
            image_profiles: ProfileCache::new(ProfileLimits::default()),
        })
    }

    fn stream(&self) -> &Stream {
        &self.parts.stream
    }

    fn validate(&self, token_ids: &[u32], count: usize) -> Result<()> {
        let geometry = self.parts.geometry;
        if token_ids.is_empty() {
            return Err(Error::Other(
                "d3-lite decision needs at least one token".into(),
            ));
        }
        if token_ids.len() > geometry.max_positions || token_ids.len() > MAX_INPUT_TOKENS {
            return Err(Error::Other(format!(
                "d3-lite decision length {} exceeds the admitted maximum {MAX_INPUT_TOKENS}",
                token_ids.len()
            )));
        }
        if count == 0 || count > super::config::CODES {
            return Err(Error::Other(format!(
                "d3-lite candidate count {count} is outside 1..={}",
                super::config::CODES
            )));
        }
        if let Some((position, token)) = token_ids
            .iter()
            .enumerate()
            .find(|(_, token)| **token as usize >= geometry.vocab)
        {
            return Err(Error::Other(format!(
                "d3-lite token ID {token} at position {position} is outside vocabulary {}",
                geometry.vocab
            )));
        }
        Ok(())
    }

    fn ids(&self, token_ids: &[u32]) -> Result<Array> {
        let ids = token_ids
            .iter()
            .map(|&token| {
                i32::try_from(token).map_err(|_| {
                    Error::Other(format!(
                        "d3-lite token ID {token} exceeds the embedding index range"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Array::from_i32(self.stream(), &[1, token_ids.len()], &ids)
    }

    fn count(&self, count: usize) -> Result<Array> {
        Array::from_i32(self.stream(), &[1], &[count as i32])
    }

    /// The text profile for `length`, traced and warmed on a miss.
    /// `Compiled::prepare` evaluates the new profile twice and fails if the
    /// second invocation traces again.
    fn text_profile(&mut self, length: usize) -> Result<&Compiled> {
        let parts = self.parts.clone();
        self.text_profiles.get_or_insert_with(length, length, || {
            let trace_parts = parts.clone();
            let compiled = Compiled::with_shapeless(&parts.stream, 1, false, move |inputs| {
                let [ids, count] = inputs else {
                    return Err(Error::Contract(
                        "d3-lite compiled forward takes ids and candidate count",
                    ));
                };
                forward(&trace_parts, ids, count, FinalLayerRows::Readout)
            })?;
            // The traced graph depends on shapes, not values, so zeros keep
            // the warm-up independent of any request.
            let ids = Array::zeros(&parts.stream, &[1, length], MlxDType::I32)?;
            let count = Array::from_i32(&parts.stream, &[1], &[1])?;
            compiled.prepare(&[ids, count])?;
            Ok(compiled)
        })
    }

    /// One request: prepared compiled execution. The returned vector holds the
    /// first `count` option probabilities, in package code order.
    pub fn probabilities(&mut self, token_ids: &[u32], count: usize) -> Result<Vec<f32>> {
        self.validate(token_ids, count)?;
        let inputs = [self.ids(token_ids)?, self.count(count)?];
        let outputs = self.text_profile(token_ids.len())?.call_and_eval(&inputs)?;
        read_probabilities(&outputs[0], count)
    }

    /// Eager lazy-graph execution of the same `forward`, used to compare the
    /// compiled path (`FinalLayerRows::Readout`) against its own body, or the
    /// final-layer pruning against the full layer, and to diagnose a
    /// divergence.
    pub fn eager_probabilities(
        &self,
        token_ids: &[u32],
        count: usize,
        rows: FinalLayerRows,
    ) -> Result<Vec<f32>> {
        self.validate(token_ids, count)?;
        let outputs = forward(
            &self.parts,
            &self.ids(token_ids)?,
            &self.count(count)?,
            rows,
        )?;
        self.stream().eval(&outputs)?;
        read_probabilities(&outputs[0], count)
    }

    /// Retained text profile lengths, least recently used first.
    pub fn text_profiles(&self) -> Vec<usize> {
        self.text_profiles.keys()
    }

    /// The image profile for `(length, image_tokens)`, traced and warmed on a
    /// miss with zero inputs.
    fn image_profile(&mut self, length: usize, image_tokens: usize) -> Result<&Compiled> {
        let parts = self.parts.clone();
        self.image_profiles
            .get_or_insert_with((length, image_tokens), length, || {
                let trace_parts = parts.clone();
                let compiled = Compiled::with_shapeless(&parts.stream, 1, false, move |inputs| {
                    let [ids, positions, count, features, run_start] = inputs else {
                        return Err(Error::Contract(
                            "d3-lite compiled image forward takes ids, mrope positions, \
                                 candidate count, merged image features and the placeholder \
                                 run start",
                        ));
                    };
                    image_forward(
                        &trace_parts,
                        ids,
                        positions,
                        count,
                        features,
                        run_start,
                        FinalLayerRows::Readout,
                    )
                })?;
                let stream = &parts.stream;
                let ids = Array::zeros(stream, &[1, length], MlxDType::I32)?;
                let positions = Array::zeros(stream, &[3, length], MlxDType::I32)?;
                let count = Array::from_i32(stream, &[1], &[1])?;
                let features = Array::zeros(
                    stream,
                    &[1, image_tokens, parts.geometry.hidden],
                    MlxDType::BF16,
                )?;
                let run_start = Array::zeros(stream, &[1], MlxDType::I32)?;
                compiled.prepare(&[ids, positions, count, features, run_start])?;
                Ok(compiled)
            })
    }

    /// One image request: the eager vision tower produces merged image
    /// tokens, the compiled image callable scatters them into the prompt and
    /// runs the text layers plus FP32 readout in one forward. Returns the
    /// first `count` option probabilities, in package code order.
    pub fn image_probabilities(
        &mut self,
        pixel_values: &[f32],
        grids: &[[u32; 3]],
        token_ids: &[u32],
        count: usize,
    ) -> Result<Vec<f32>> {
        self.validate(token_ids, count)?;
        let merge = self.parts.vision_merge;
        if merge == 0 {
            return Err(Error::Contract(
                "d3-lite vision merge size must be positive",
            ));
        }
        let (positions, slots) = multimodal_layout(
            token_ids,
            grids,
            self.parts.image_token_id,
            self.parts.video_token_id,
            merge,
        )?;
        let length = token_ids.len();
        let image_tokens = slots.len();
        // The processor expands one image into one contiguous placeholder run;
        // MLX's dynamic slice_update takes one start index, so that run is the
        // only scatter this path supports.
        let run_start = slots[0];
        if slots.windows(2).any(|window| window[1] != window[0] + 1) {
            return Err(Error::Other(
                "d3-lite Metal supports one image per request; multiple images are not wired"
                    .into(),
            ));
        }
        let features = self.parts.vision.forward(pixel_values, grids)?;
        let features = features.reshape(&[1, image_tokens, self.parts.geometry.hidden])?;
        // A concrete input keeps the prepared profile replaying without
        // re-tracing, because preparation passed concrete zero arrays.
        self.stream().eval(&[features.clone()])?;
        let inputs = [
            self.ids(token_ids)?,
            Array::from_i32(self.stream(), &[3, length], &positions)?,
            self.count(count)?,
            features,
            Array::from_i32(self.stream(), &[1], &[run_start])?,
        ];
        let outputs = self
            .image_profile(length, image_tokens)?
            .call_and_eval(&inputs)?;
        read_probabilities(&outputs[0], count)
    }

    /// Retained `(length, image_tokens)` image profiles, least recently used
    /// first.
    pub fn image_profiles(&self) -> Vec<ImageProfileKey> {
        self.image_profiles.keys()
    }
}

/// The multimodal request layout the reference builds from `mm_token_type_ids`
/// and `image_grid_thw`: mRoPE positions `[3, L]` (text positions advance by
/// one per token; an image run takes its grid's H/W coordinates and advances
/// by `max(h, w) / merge`) and the placeholder slot indices `[M]` the merged
/// image tokens scatter into. A run of `image_token_id` is one image, in
/// request order, and must match the next grid exactly.
fn multimodal_layout(
    token_ids: &[u32],
    grids: &[[u32; 3]],
    image_token_id: u32,
    video_token_id: u32,
    merge: usize,
) -> Result<(Vec<i32>, Vec<i32>)> {
    if grids.is_empty() {
        return Err(Error::Other(
            "d3-lite image request needs at least one grid".into(),
        ));
    }
    let length = token_ids.len();
    let mut positions = vec![vec![0i32; length]; 3];
    let mut slots = Vec::new();
    let mut current = 0i32;
    let mut index = 0usize;
    let mut grid_index = 0usize;
    while index < length {
        if token_ids[index] == video_token_id {
            return Err(Error::Other(
                "d3-lite Metal image path does not read video tokens; video is not wired".into(),
            ));
        }
        let image_run = token_ids[index] == image_token_id;
        let mut end = index + 1;
        while end < length && (token_ids[end] == image_token_id) == image_run {
            end += 1;
        }
        if image_run {
            let [temporal, grid_h, grid_w] = *grids.get(grid_index).ok_or_else(|| {
                Error::Other(format!(
                    "d3-lite prompt has more image runs than grids ({grid_index} grids consumed)"
                ))
            })?;
            grid_index += 1;
            if temporal != 1 {
                return Err(Error::Other(
                    "d3-lite Metal vision runs still images (grid t = 1); video is not wired"
                        .into(),
                ));
            }
            let merged_h = grid_h as usize / merge;
            let merged_w = grid_w as usize / merge;
            let expected = merged_h * merged_w;
            if merged_h == 0 || merged_w == 0 {
                return Err(Error::Other(format!(
                    "d3-lite image grid [{temporal}, {grid_h}, {grid_w}] is smaller than one merge block"
                )));
            }
            if end - index != expected {
                return Err(Error::Other(format!(
                    "d3-lite image run at position {index} holds {} placeholders, expected {expected} \
                     for grid [{temporal}, {grid_h}, {grid_w}]",
                    end - index
                )));
            }
            for block_row in 0..merged_h {
                for block_column in 0..merged_w {
                    let slot = index + block_row * merged_w + block_column;
                    positions[0][slot] = current;
                    positions[1][slot] = current + block_row as i32;
                    positions[2][slot] = current + block_column as i32;
                    slots.push(slot as i32);
                }
            }
            current += merged_h.max(merged_w) as i32;
        } else {
            for offset in index..end {
                let value = current + (offset - index) as i32;
                positions[0][offset] = value;
                positions[1][offset] = value;
                positions[2][offset] = value;
            }
            current += (end - index) as i32;
        }
        index = end;
    }
    if grid_index != grids.len() {
        return Err(Error::Other(format!(
            "d3-lite prompt consumed {grid_index} image grids but the request passed {}",
            grids.len()
        )));
    }
    if slots.is_empty() {
        return Err(Error::Other(
            "d3-lite image request found no image_token_id placeholders".into(),
        ));
    }
    let flat = positions.into_iter().flatten().collect();
    Ok((flat, slots))
}

fn read_probabilities(output: &Array, count: usize) -> Result<Vec<f32>> {
    let values = output.to_f32_vec()?;
    if values.len() != super::config::CODES {
        return Err(Error::Other(format!(
            "d3-lite forward returned {} probabilities, expected {}",
            values.len(),
            super::config::CODES
        )));
    }
    Ok(values[..count].to_vec())
}

/// `inv_freq[i] = exp(-(i / rope_dims) * log(rope_base))` in F32, computed as
/// the reference does (`exp(-exponents * log(base))`) and published as one
/// evaluated constant. The host evaluates this once at load; the compiled
/// graph then captures the array.
fn inverse_frequencies(stream: &Stream, geometry: Geometry) -> Result<Array> {
    let values = (0..geometry.rope_dims)
        .step_by(2)
        .map(|index| {
            let exponent = (index as f32) / (geometry.rope_dims as f32);
            (-(exponent * (geometry.rope_base as f32).ln())).exp()
        })
        .collect::<Vec<_>>();
    let inv_freq = Array::from_f32(stream, &[geometry.rope_dims / 2], &values)?;
    stream.eval(&[inv_freq.clone()])?;
    Ok(inv_freq)
}

/// How the attention layers receive rotary positions. The text path uses the
/// fused half-split kernel with 1-D positions; an image
/// request uses mRoPE cos/sin recomposed from the `[3, L]` multimodal
/// positions, exactly as the reference's `Qwen3_5TextRotaryEmbedding` does.
enum RopeMode<'a> {
    HalfSplit(&'a Array),
    Mrope { cos: &'a Array, sin: &'a Array },
}

/// The whole text request graph. `ids` is `[1, L]` I32, `candidate_count` is
/// `[1]` I32; the returned vector is `[probabilities [1, 255]]` in F32, with
/// every code at or past `candidate_count` set to `-inf` before the softmax so
/// the result is exactly the masked softmax the reference produces.
fn forward(
    parts: &Parts,
    ids: &Array,
    candidate_count: &Array,
    rows: FinalLayerRows,
) -> Result<Vec<Array>> {
    let [1, length] = *ids.shape() else {
        return Err(Error::Contract("d3-lite ids must be [1, L]"));
    };
    if length == 0 {
        return Err(Error::Contract("d3-lite ids must be nonempty"));
    }
    let hidden = parts.weights.embed.take(ids, 0)?;
    let positions = Array::arange(&parts.stream, 0.0, length as f32, 1.0, MlxDType::I32)?
        .reshape(&[1, length])?;
    let hidden = stack(parts, hidden, &RopeMode::HalfSplit(&positions), rows)?;
    readout_tail(parts, &hidden, candidate_count)
}

/// The whole image request graph: embedding lookup, the eager vision tower's
/// merged tokens written over the single contiguous `image_token_id` run, the
/// same 24-layer stack under mRoPE, and the FP32 255-code readout. `positions`
/// is `[3, L]` I32, `features` `[1, M, hidden]` BF16 and `run_start` `[1]` I32
/// — MLX's dynamic `slice_update` takes one start index per axis, so the
/// placeholders must form one contiguous run (the processor's expansion of a
/// single image).
fn image_forward(
    parts: &Parts,
    ids: &Array,
    positions: &Array,
    candidate_count: &Array,
    features: &Array,
    run_start: &Array,
    rows: FinalLayerRows,
) -> Result<Vec<Array>> {
    let [1, length] = *ids.shape() else {
        return Err(Error::Contract("d3-lite ids must be [1, L]"));
    };
    let [3, position_length] = *positions.shape() else {
        return Err(Error::Contract("d3-lite mrope positions must be [3, L]"));
    };
    if position_length != length {
        return Err(Error::Other(format!(
            "d3-lite mrope positions cover {position_length} tokens, ids cover {length}"
        )));
    }
    let [1, image_tokens, hidden_width] = *features.shape() else {
        return Err(Error::Contract(
            "d3-lite merged image features must be [1, M, hidden]",
        ));
    };
    if hidden_width != parts.geometry.hidden {
        return Err(Error::Other(format!(
            "d3-lite merged image features have width {hidden_width}, expected {}",
            parts.geometry.hidden
        )));
    }
    if image_tokens == 0 || run_start.shape() != [1] {
        return Err(Error::Other(format!(
            "d3-lite image scatter needs a nonempty feature run and a [1] start, got {:?} and {:?}",
            features.shape(),
            run_start.shape()
        )));
    }
    let mut embeds = parts.weights.embed.take(ids, 0)?;
    embeds = embeds.slice_update(features, run_start, &[1])?;
    let (cos, sin) = mrope_rotary(parts, positions, length)?;
    let hidden = stack(
        parts,
        embeds,
        &RopeMode::Mrope {
            cos: &cos,
            sin: &sin,
        },
        rows,
    )?;
    readout_tail(parts, &hidden, candidate_count)
}

/// The 24-layer stack under one rotary contract, ending at the final RMSNorm.
/// The op order is shared by the text and image callables; only the rotary
/// arguments differ. With `FinalLayerRows::Readout` a full-attention final
/// layer returns only the last row, and the residual, MLP and final norm
/// follow it; a final Gated DeltaNet layer would need every row, so it keeps
/// them.
fn stack(
    parts: &Parts,
    mut hidden: Array,
    rope: &RopeMode<'_>,
    rows: FinalLayerRows,
) -> Result<Array> {
    let geometry = parts.geometry;
    let length = hidden.shape()[1];
    let final_layer = parts.weights.layers.len() - 1;
    for index in 0..parts.weights.layers.len() {
        let first_row = if index == final_layer
            && rows == FinalLayerRows::Readout
            && geometry.is_attention(index)
        {
            length - 1
        } else {
            0
        };
        hidden = layer_forward(parts, index, &hidden, rope, first_row)?;
    }
    hidden.fast_rms_norm(&parts.weights.norm, geometry.eps)
}

/// Decoder layer `index` for the output rows `first_row..L` of `hidden`
/// `[1, L, H]`. Every row still contributes attention keys and values; a
/// Gated DeltaNet layer needs every row, so it takes only `first_row == 0`.
fn layer_forward(
    parts: &Parts,
    index: usize,
    hidden: &Array,
    rope: &RopeMode<'_>,
    first_row: usize,
) -> Result<Array> {
    let geometry = parts.geometry;
    let layer = &parts.weights.layers[index];
    let length = hidden.shape()[1];
    let normalized = hidden.fast_rms_norm(&layer.input_norm, geometry.eps)?;
    let mixed = if geometry.is_attention(index) {
        let Mixer::Attention(attention) = &layer.mixer else {
            unreachable!("the validated schedule selects attention here")
        };
        attention_forward(parts, &normalized, attention, rope, first_row)?
    } else {
        let Mixer::GatedDelta(gated_delta) = &layer.mixer else {
            unreachable!("the validated schedule selects gated delta here")
        };
        debug_assert_eq!(first_row, 0, "a recurrent layer computes every row");
        gated_delta_forward(parts, &normalized, gated_delta, length)?
    };
    let residual = hidden.slice_axis(1, first_row, length)?.add(&mixed)?;
    let mlp_input = residual.fast_rms_norm(&layer.post_norm, geometry.eps)?;
    let mlp_output = mlp_forward(&mlp_input, &layer.mlp)?;
    residual.add(&mlp_output)
}

/// Last position → FP32 → the 255-code readout → masked FP32 softmax.
fn readout_tail(parts: &Parts, hidden: &Array, candidate_count: &Array) -> Result<Vec<Array>> {
    let geometry = parts.geometry;
    let length = hidden.shape()[1];
    let last = hidden
        .slice_axis(1, length - 1, length)?
        .reshape(&[1, geometry.hidden])?
        .cast(MlxDType::F32)?;
    let logits = last.matmul(&parts.weights.readout.transpose(&[1, 0])?)?;
    let indices = Array::arange(
        &parts.stream,
        0.0,
        super::config::CODES as f32,
        1.0,
        MlxDType::I32,
    )?;
    // `candidate_count <= index` is the "code is not an option" mask.
    let masked = candidate_count.less_equal(&indices)?;
    let negative_infinity = Array::scalar(&parts.stream, f32::NEG_INFINITY, MlxDType::F32)?;
    let logits = masked.where_select(&negative_infinity, &logits)?;
    let probabilities = logits.softmax_with(-1, false)?;
    Ok(vec![probabilities])
}

/// mRoPE cos/sin for one image request: `[3, L]` positions × the 32 inverse
/// frequencies, recomposed with the interleaved `[T | H | W]` sections
/// (`i % 3`) and duplicated over the 64-wide rotary block, all in F32 then
/// rounded once to the embedding dtype, exactly as the reference does.
fn mrope_rotary(parts: &Parts, positions: &Array, length: usize) -> Result<(Array, Array)> {
    let width = parts.geometry.rope_dims;
    let frequency_count = parts.inv_freq.shape()[0];
    let frequencies = positions
        .cast(MlxDType::F32)?
        .reshape(&[3, length, 1])?
        .mul(&parts.inv_freq.reshape(&[1, 1, frequency_count])?)?;
    let text = frequencies
        .slice_axis(0, 0, 1)?
        .reshape(&[length, frequency_count])?;
    let height = frequencies
        .slice_axis(0, 1, 2)?
        .reshape(&[length, frequency_count])?;
    let column = frequencies
        .slice_axis(0, 2, 3)?
        .reshape(&[length, frequency_count])?;
    // mrope_section [11, 11, 10] over 32 frequencies is the interleaved
    // `i % 3`: T, H, W, T, H, W, ... .
    let partition_ids = (0..frequency_count as i32)
        .map(|index| index % 3)
        .collect::<Vec<_>>();
    let partition = Array::from_i32(&parts.stream, &[frequency_count], &partition_ids)?;
    let zero = Array::scalar(&parts.stream, 0.0, MlxDType::I32)?;
    let one = Array::scalar(&parts.stream, 1.0, MlxDType::I32)?;
    let is_text = partition.equal(&zero)?;
    let is_height = partition.equal(&one)?;
    let spatial = is_height.where_select(&height, &column)?;
    let recomposed = is_text.where_select(&text, &spatial)?;
    let duplicate = |values: Array| -> Result<Array> { Array::concat(&[&values, &values], 1) };
    let cos = duplicate(recomposed.cos()?.cast(MlxDType::BF16)?)?;
    let sin = duplicate(recomposed.sin()?.cast(MlxDType::BF16)?)?;
    debug_assert_eq!(cos.shape(), &[length, width]);
    Ok((cos, sin))
}

/// Apply mRoPE to one `[B, H, L, D]` tensor: the first `rotary` features are
/// paired half-and-half and rotated by the recomposed cos/sin (BF16 ops, one
/// rounding per operation, as the reference's `apply_rotary_pos_emb` does);
/// the remaining features pass through untouched.
fn mrope_apply(tensor: &Array, cos: &Array, sin: &Array, rotary: usize) -> Result<Array> {
    let [batch, heads, length, width] = *tensor.shape() else {
        return Err(Error::Contract("d3-lite rotary input must be [B, H, L, D]"));
    };
    if rotary == 0 || rotary > width || rotary % 2 != 0 || cos.shape() != [length, rotary] {
        return Err(Error::Other(format!(
            "d3-lite mrope needs even rotary width {rotary} ≤ {width} and cos/sin [{length}, {rotary}]"
        )));
    }
    let active = tensor.slice_axis(3, 0, rotary)?;
    let first = active.slice_axis(3, rotary / 2, rotary)?;
    let second = active.slice_axis(3, 0, rotary / 2)?;
    let rotated = Array::concat(&[&first.neg()?, &second], 3)?;
    let cos = cos.reshape(&[1, 1, length, rotary])?;
    let sin = sin.reshape(&[1, 1, length, rotary])?;
    let active = active.mul(&cos)?.add(&rotated.mul(&sin)?)?;
    if rotary == width {
        debug_assert_eq!(active.shape(), &[batch, heads, length, width]);
        return Ok(active);
    }
    Array::concat(&[&active, &tensor.slice_axis(3, rotary, width)?], 3)
}

fn mlp_forward(input: &Array, mlp: &Mlp) -> Result<Array> {
    let gate = project(input, &mlp.gate.weight)?;
    let up = project(input, &mlp.up.weight)?;
    let activated = silu(&gate)?.mul(&up)?;
    project(&activated, &mlp.down.weight)
}

fn gated_delta_forward(
    parts: &Parts,
    input: &Array,
    gated_delta: &GatedDelta,
    length: usize,
) -> Result<Array> {
    let geometry = parts.geometry;
    let key_dim = geometry.key_dim();
    let value_dim = geometry.value_dim();
    let conv_dim = geometry.conv_dim();
    let qkv = project(input, &gated_delta.qkv.weight)?;
    let z = project(input, &gated_delta.z.weight)?.reshape(&[
        1,
        length,
        geometry.v_heads,
        geometry.v_dim,
    ])?;
    let b = project(input, &gated_delta.b.weight)?;
    let a = project(input, &gated_delta.a.weight)?;
    // Stock reference: a depthwise causal convolution over `S - 1` zero rows.
    let history = Array::zeros(
        &parts.stream,
        &[1, geometry.conv_kernel - 1, conv_dim],
        qkv.dtype(),
    )?;
    let convolved =
        Array::concat(&[&history, &qkv], 1)?.conv1d(&gated_delta.conv, 1, 0, 1, conv_dim)?;
    let convolved = silu(&convolved)?;
    let queries = convolved.slice_axis(2, 0, key_dim)?.reshape(&[
        1,
        length,
        geometry.k_heads,
        geometry.k_dim,
    ])?;
    let keys = convolved.slice_axis(2, key_dim, 2 * key_dim)?.reshape(&[
        1,
        length,
        geometry.k_heads,
        geometry.k_dim,
    ])?;
    let values = convolved.slice_axis(2, 2 * key_dim, conv_dim)?.reshape(&[
        1,
        length,
        geometry.v_heads,
        geometry.v_dim,
    ])?;
    // Official GDN q/k l2-norm: `x * rsqrt(sum(x^2) + 1e-6)`, written as an
    // RMSNorm whose epsilon is on the mean (`1e-6 / k_dim`), then scaled by
    // `k_dim^-1` for q and `k_dim^-0.5` for k. The mlx-lm variant uses a mean
    // epsilon of `1e-6` and differs enough to move the probabilities; the
    // checkpoint's own model uses this form.
    let mean_epsilon = 1e-6 / geometry.k_dim as f32;
    let queries = queries
        .fast_rms_norm_unweighted(mean_epsilon)?
        .mul(&scalar(&queries, (geometry.k_dim as f64).powf(-1.0))?)?;
    let keys = keys
        .fast_rms_norm_unweighted(mean_epsilon)?
        .mul(&scalar(&keys, (geometry.k_dim as f64).powf(-0.5))?)?;
    let (decay, beta) = decay_beta(&gated_delta.a_log, &a, &b, &gated_delta.dt_bias)?;
    let state = Array::zeros(
        &parts.stream,
        &[1, geometry.v_heads, geometry.v_dim, geometry.k_dim],
        MlxDType::F32,
    )?;
    let (recurrent, _next_state) = parts
        .recurrence
        .call(&queries, &keys, &values, &decay, &beta, &state)?;
    let normalized = recurrent.fast_rms_norm(&gated_delta.norm, geometry.eps)?;
    // Gated norm in FP32, rounded once to BF16 before the output projection.
    let gated = silu(&z.cast(MlxDType::F32)?)?
        .mul(&normalized.cast(MlxDType::F32)?)?
        .cast(recurrent.dtype())?;
    project(
        &gated.reshape(&[1, length, value_dim])?,
        &gated_delta.out.weight,
    )
}

/// Full attention for the query rows `first_row..L` against keys and values
/// from all `L` rows. Every query keeps its own position: the rotary inputs
/// are sliced, never rebuilt from zero, so the last row rotates as `L - 1`.
fn attention_forward(
    parts: &Parts,
    input: &Array,
    attention: &Attention,
    rope: &RopeMode<'_>,
    first_row: usize,
) -> Result<Array> {
    let geometry = parts.geometry;
    let [1, length, _] = *input.shape() else {
        return Err(Error::Contract("d3-lite attention input must be [1, L, H]"));
    };
    let query_rows = length - first_row;
    // One packed projection yields queries and the output gate: `[B, T, 2*heads*head_dim]`.
    let projected = project(
        &input.slice_axis(1, first_row, length)?,
        &attention.q.weight,
    )?
    .reshape(&[1, query_rows, geometry.q_heads, 2 * geometry.head_dim])?;
    let queries = projected.slice_axis(3, 0, geometry.head_dim)?;
    let gate = projected
        .slice_axis(3, geometry.head_dim, 2 * geometry.head_dim)?
        .reshape(&[1, query_rows, geometry.q_heads * geometry.head_dim])?;
    let keys = project(input, &attention.k.weight)?;
    let values = project(input, &attention.v.weight)?;
    let queries = queries
        .fast_rms_norm(&attention.q_norm, geometry.eps)?
        .transpose(&[0, 2, 1, 3])?;
    let keys = keys
        .reshape(&[1, length, geometry.kv_heads, geometry.head_dim])?
        .fast_rms_norm(&attention.k_norm, geometry.eps)?
        .transpose(&[0, 2, 1, 3])?;
    let values = values
        .reshape(&[1, length, geometry.kv_heads, geometry.head_dim])?
        .transpose(&[0, 2, 1, 3])?;
    let (queries, keys) = match rope {
        RopeMode::HalfSplit(positions) => (
            parts.rope.call(
                &queries,
                &positions.slice_axis(1, first_row, length)?,
                &parts.inv_freq,
            )?,
            parts.rope.call(&keys, positions, &parts.inv_freq)?,
        ),
        RopeMode::Mrope { cos, sin } => (
            mrope_apply(
                &queries,
                &cos.slice_axis(0, first_row, length)?,
                &sin.slice_axis(0, first_row, length)?,
                geometry.rope_dims,
            )?,
            mrope_apply(&keys, cos, sin, geometry.rope_dims)?,
        ),
    };
    // d3's full-attention layers are explicitly non-causal over the prompt.
    let output = queries.sdpa(
        &keys,
        &values,
        (geometry.head_dim as f32).powf(-0.5),
        false,
        None,
    )?;
    let output = output.transpose(&[0, 2, 1, 3])?.reshape(&[
        1,
        query_rows,
        geometry.q_heads * geometry.head_dim,
    ])?;
    project(&output.mul(&gate.sigmoid()?)?, &attention.o.weight)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stream() -> Stream {
        Stream::cpu().expect("CPU stream for eager semantics")
    }

    #[test]
    fn inverse_frequencies_match_the_reference_formula() {
        let geometry = Geometry {
            hidden: 1_024,
            intermediate: 3_584,
            layers: 24,
            q_heads: 8,
            kv_heads: 2,
            head_dim: 256,
            k_heads: 16,
            v_heads: 16,
            k_dim: 128,
            v_dim: 128,
            conv_kernel: 4,
            full_attention_interval: 4,
            eps: 1e-6,
            rope_dims: 64,
            rope_base: 10_000_000.0,
            vocab: 248_320,
            max_positions: 262_144,
        };
        let expected = (0..64)
            .step_by(2)
            .map(|index| (-(index as f32 / 64.0 * 10_000_000f32.ln())).exp())
            .collect::<Vec<_>>();
        let inv_freq = inverse_frequencies(&stream(), geometry).unwrap();
        assert_eq!(inv_freq.dtype(), MlxDType::F32);
        assert_eq!(inv_freq.shape(), &[32]);
        let actual = inv_freq.to_f32_vec().unwrap();
        assert_eq!(actual, expected);
        assert!(actual[0] > actual[31]);
        assert!((actual[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn decay_beta_follows_compute_g_and_sigmoid() {
        let stream = stream();
        let a_log = Array::from_f32(&stream, &[2], &[0.0, 1.0]).unwrap();
        // Equal inputs across the two heads make `A_log` the only variable:
        // softplus(0) = ln 2, so decay is 0.5 vs exp(-e * ln 2) ≈ 0.152.
        let a = Array::zeros(&stream, &[1, 3, 2], MlxDType::BF16).unwrap();
        let dt_bias = Array::zeros(&stream, &[2], MlxDType::BF16).unwrap();
        let b = Array::from_f32(&stream, &[1, 3, 2], &[-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
            .unwrap()
            .cast(MlxDType::BF16)
            .unwrap();
        let (decay, beta) = decay_beta(&a_log, &a, &b, &dt_bias).unwrap();
        stream.eval(&[decay.clone(), beta.clone()]).unwrap();
        assert_eq!(decay.dtype(), MlxDType::F32);
        assert_eq!(decay.shape(), &[1, 3, 2]);
        assert_eq!(beta.dtype(), MlxDType::BF16);
        let decay = decay.to_f32_vec().unwrap();
        let beta = beta.cast(MlxDType::F32).unwrap().to_f32_vec().unwrap();
        // `exp(-exp(A_log) * softplus(a + dt_bias))` in (0, 1].
        assert!(decay
            .iter()
            .all(|value| value.is_finite() && *value > 0.0 && *value <= 1.0));
        let ln_two = 2f32.ln();
        let expected = [(-ln_two).exp(), (-(std::f32::consts::E * ln_two)).exp()];
        assert!(
            (decay[0] - expected[0]).abs() < 2e-3,
            "{} vs {}",
            decay[0],
            expected[0]
        );
        assert!(
            (decay[1] - expected[1]).abs() < 2e-3,
            "{} vs {}",
            decay[1],
            expected[1]
        );
        assert!(decay[0] > decay[1], "larger A_log must decay faster");
        assert!(beta.iter().all(|value| (0.0..1.0).contains(value)));
        // b = -2 → sigmoid ≈ 0.119; b = 3 → ≈ 0.953.
        assert!((beta[0] - (-2.0f32).exp() / (1.0 + (-2.0f32).exp())).abs() < 3e-3);
        assert!((beta[5] - 3.0f32.exp() / (1.0 + 3.0f32.exp())).abs() < 3e-3);
    }

    #[test]
    fn project_is_linear_without_bias() {
        let stream = stream();
        let weight = Array::from_f32(&stream, &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let input =
            Array::from_f32(&stream, &[1, 2, 3], &[1.0, 0.0, -1.0, 0.5, 2.0, -0.5]).unwrap();
        let output = project(&input, &weight).unwrap();
        stream.eval(&[output.clone()]).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2]);
        assert_eq!(output.to_f32_vec().unwrap(), [-2.0, -2.0, 3.0, 9.0],);
    }

    fn load_device_model(text_limits: ProfileLimits) -> Result<(Config, Stream, Model)> {
        let directory = std::env::var_os("APXINF_D3_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_CHECKPOINT to the d3-lite package directory");
        let config = Config::from_dir(&directory)?;
        let stream = Stream::metal(0)?;
        let weights = Weights::load(config.geometry, &stream, &directory)?;
        let vision_weights = VisionWeights::load(config.vision, &stream, &directory)?;
        let model = Model::new(
            &config,
            stream.clone(),
            weights,
            vision_weights,
            text_limits,
        )?;
        Ok((config, stream, model))
    }

    /// End-to-end device acceptance for the prepared whole-request callable:
    /// compiled output matches the eager body, replay traces nothing new,
    /// changed inputs reach the output, a new candidate count reuses the
    /// profile, and a new length traces exactly once.
    #[test]
    #[ignore = "requires APXINF_D3_CHECKPOINT and an Apple Metal GPU"]
    fn compiled_forward_matches_eager_replays_and_binds_changed_inputs() -> Result<()> {
        let (config, stream, mut model) = load_device_model(ProfileLimits::default())?;
        let count = config.codes.len();
        let tokens = 37usize;
        let first = vec![1u32; tokens];
        let eager = model.eager_probabilities(&first, count, FinalLayerRows::Readout)?;
        let compiled = model.probabilities(&first, count)?;
        let max = eager
            .iter()
            .zip(&compiled)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(max < 1e-4, "compiled vs eager max|Δp| = {max}");

        let before = stream.counters();
        let replay = model.probabilities(&first, count)?;
        assert_eq!(before.trace_callbacks, stream.counters().trace_callbacks);
        assert!(
            replay
                .iter()
                .zip(&compiled)
                .all(|(left, right)| left.to_bits() == right.to_bits()),
            "prepared replay changed values"
        );

        let changed = model.probabilities(&vec![2u32; tokens], count)?;
        let changed_delta = compiled
            .iter()
            .zip(&changed)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(
            changed_delta > 0.0,
            "changed token IDs did not change probabilities"
        );

        let before = stream.counters();
        let two = model.probabilities(&first, 2)?;
        assert_eq!(
            before.trace_callbacks,
            stream.counters().trace_callbacks,
            "a new candidate count retraced the profile"
        );
        let eager_two = model.eager_probabilities(&first, 2, FinalLayerRows::Readout)?;
        let two_delta = two
            .iter()
            .zip(&eager_two)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max);
        assert!(
            two_delta < 1e-4,
            "two-candidate compiled vs eager max|Δp| = {two_delta}"
        );

        let longer = vec![1u32; 64];
        let before = stream.counters();
        model.probabilities(&longer, count)?;
        let traced = stream.counters().trace_callbacks - before.trace_callbacks;
        assert!(traced > 0, "a new length did not trace");
        let before = stream.counters();
        model.probabilities(&longer, count)?;
        assert_eq!(before.trace_callbacks, stream.counters().trace_callbacks);
        assert_eq!(model.text_profiles(), [tokens, 64]);
        Ok(())
    }

    /// The readout-row final layer is the full final layer's last row, not
    /// an approximation of it. Layers before the last run once in BF16; the
    /// final layer then runs on that same input promoted to F32, once for all
    /// rows and once for the readout row. In F32 the two shapes differ only in
    /// summation order, so the rows agree to about 1e-6. The BF16 rounding the
    /// served path sees is about 4e-3, which would hide a query rotated at the
    /// wrong position; in F32 that defect measures 5e-4 at two tokens.
    #[test]
    #[ignore = "requires APXINF_D3_CHECKPOINT and an Apple Metal GPU"]
    fn readout_row_final_layer_is_the_full_layers_last_row_in_f32() -> Result<()> {
        const MAX_RELATIVE_L2: f32 = 1e-4;
        let (_, stream, model) = load_device_model(ProfileLimits::default())?;
        let parts = &model.parts;
        let final_layer = parts.weights.layers.len() - 1;
        assert!(parts.geometry.is_attention(final_layer));
        for length in [1usize, 2, 8, 9, 37, 300] {
            let tokens = (0..length)
                .map(|index| 1_000 + (index as u32 * 7_919) % 50_000)
                .collect::<Vec<_>>();
            let positions = Array::arange(&stream, 0.0, length as f32, 1.0, MlxDType::I32)?
                .reshape(&[1, length])?;
            let rope = RopeMode::HalfSplit(&positions);
            let mut hidden = parts.weights.embed.take(&model.ids(&tokens)?, 0)?;
            for index in 0..final_layer {
                hidden = layer_forward(parts, index, &hidden, &rope, 0)?;
            }
            let hidden = hidden.cast(MlxDType::F32)?;
            let full = layer_forward(parts, final_layer, &hidden, &rope, 0)?.slice_axis(
                1,
                length - 1,
                length,
            )?;
            let readout = layer_forward(parts, final_layer, &hidden, &rope, length - 1)?;
            assert_eq!(readout.shape(), [1, 1, parts.geometry.hidden]);
            stream.eval(&[full.clone(), readout.clone()])?;
            let (full, readout) = (full.to_f32_vec()?, readout.to_f32_vec()?);
            let difference = full
                .iter()
                .zip(&readout)
                .map(|(left, right)| (left - right).powi(2))
                .sum::<f32>()
                .sqrt();
            let relative = difference / full.iter().map(|value| value * value).sum::<f32>().sqrt();
            println!("length {length}: F32 relative L2 = {relative:.2e}");
            assert!(
                relative <= MAX_RELATIVE_L2,
                "length {length}: readout row vs full layer relative L2 = {relative}"
            );
        }
        Ok(())
    }

    /// The served final layer computes only the readout row. It must be the
    /// full layer's last row up to rounding: the probabilities stay within
    /// `MAX_ROW_PRUNE_DELTA` of the full layer's and the decision does not
    /// change. Lengths cover one token, the last length MLX's fused
    /// single-query attention also serves in the full layer (8), the first it
    /// does not (9), and a few longer prompts. Token IDs vary so every row
    /// differs.
    #[test]
    #[ignore = "requires APXINF_D3_CHECKPOINT and an Apple Metal GPU"]
    fn readout_row_final_layer_matches_the_full_final_layer() -> Result<()> {
        const MAX_ROW_PRUNE_DELTA: f32 = 0.01;
        let (config, _, model) = load_device_model(ProfileLimits::default())?;
        let count = config.codes.len();
        for length in [1usize, 2, 8, 9, 37, 300] {
            let tokens = (0..length)
                .map(|index| 1_000 + (index as u32 * 7_919) % 50_000)
                .collect::<Vec<_>>();
            let full = model.eager_probabilities(&tokens, count, FinalLayerRows::All)?;
            let readout = model.eager_probabilities(&tokens, count, FinalLayerRows::Readout)?;
            let delta = full
                .iter()
                .zip(&readout)
                .map(|(left, right)| (left - right).abs())
                .fold(0.0f32, f32::max);
            let decision = |values: &[f32]| {
                (0..values.len())
                    .max_by(|&left, &right| values[left].total_cmp(&values[right]))
                    .unwrap()
            };
            println!("length {length}: max|Δp| = {delta:.6}");
            assert!(
                delta <= MAX_ROW_PRUNE_DELTA,
                "length {length}: readout-row vs full-layer max|Δp| = {delta}"
            );
            assert_eq!(
                decision(&readout),
                decision(&full),
                "length {length}: the readout row changed the decision"
            );
        }
        Ok(())
    }

    /// The profile limit holds on the device: the least recently used length
    /// is evicted and traces again on its next request, while retained
    /// lengths replay without tracing.
    #[test]
    #[ignore = "requires APXINF_D3_CHECKPOINT and an Apple Metal GPU"]
    fn text_profiles_stay_within_the_limit_and_evicted_lengths_retrace() -> Result<()> {
        let limits = ProfileLimits {
            profiles: std::num::NonZeroUsize::new(2).unwrap(),
            tokens: std::num::NonZeroUsize::new(1_000).unwrap(),
        };
        let (_, stream, mut model) = load_device_model(limits)?;
        let traces = |model: &mut Model, length: usize| -> Result<usize> {
            let before = stream.counters().trace_callbacks;
            model.probabilities(&vec![1u32; length], 2)?;
            Ok(stream.counters().trace_callbacks - before)
        };
        for length in [20, 21, 22] {
            assert!(
                traces(&mut model, length)? > 0,
                "length {length} did not trace"
            );
        }
        assert_eq!(model.text_profiles(), [21, 22]);
        assert_eq!(traces(&mut model, 22)?, 0, "a retained length retraced");
        assert!(traces(&mut model, 20)? > 0, "an evicted length replayed");
        assert_eq!(model.text_profiles(), [22, 20]);
        Ok(())
    }

    /// Vision-tower acceptance: the native `visual.*` forward reproduces the
    /// checkpoint's PyTorch pooler output (`image_features`) on the same
    /// preprocessed pixels. This localizes image-path drift between the
    /// vision tower and the text layers + mRoPE + readout.
    #[test]
    #[ignore = "requires APXINF_D3_CHECKPOINT, APXINF_D3_VISION_REF and an Apple Metal GPU"]
    fn vision_tower_matches_pytorch_reference() -> Result<()> {
        use serde::Deserialize;
        use std::collections::HashMap;

        #[derive(Deserialize)]
        struct Receipt {
            cases: Vec<Case>,
        }
        #[derive(Deserialize)]
        struct Case {
            id: String,
            grid_thw: Vec<u32>,
            arrays: Arrays,
        }
        #[derive(Deserialize)]
        struct Arrays {
            pixel_values: Option<ArrayFile>,
            image_features: Option<ArrayFile>,
        }
        #[derive(Deserialize)]
        struct ArrayFile {
            file: String,
        }
        #[derive(Deserialize)]
        struct StageCase {
            stages: HashMap<String, ArrayFile>,
        }

        fn read_f32(base: &std::path::Path, file: &str) -> Result<Vec<f32>> {
            let path = base.join(file);
            let bytes = std::fs::read(&path)
                .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
            Ok(bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect())
        }

        let checkpoint = std::env::var_os("APXINF_D3_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_CHECKPOINT to the d3-lite package directory");
        let reference = std::env::var_os("APXINF_D3_VISION_REF")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_VISION_REF to the reference cases.json");
        let base = reference
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let receipt: Receipt = serde_json::from_str(
            &std::fs::read_to_string(&reference)
                .map_err(|error| Error::Other(format!("read {}: {error}", reference.display())))?,
        )
        .map_err(|error| Error::Other(format!("parse {}: {error}", reference.display())))?;

        let config = Config::from_dir(&checkpoint)?;
        let stream = Stream::metal(0)?;
        let weights = Weights::load(config.geometry, &stream, &checkpoint)?;
        let vision_weights = VisionWeights::load(config.vision, &stream, &checkpoint)?;
        let model = Model::new(
            &config,
            stream.clone(),
            weights,
            vision_weights,
            ProfileLimits::default(),
        )?;

        // Optional per-stage reference, which localizes a divergence to patch
        // embed, position add, one block or the merger.
        let stage_index: HashMap<String, StageCase> =
            match std::env::var_os("APXINF_D3_VISION_STAGES") {
                Some(path) => {
                    let text = std::fs::read_to_string(&path).map_err(|error| {
                        Error::Other(format!("read {}: {error}", path.to_string_lossy()))
                    })?;
                    serde_json::from_str(&text).map_err(|error| {
                        Error::Other(format!("parse {}: {error}", path.to_string_lossy()))
                    })?
                }
                None => HashMap::new(),
            };

        // The tower's late residual stream amplifies BF16 kernel-order
        // differences: on the recorded cases the checkpoint's own PyTorch
        // runtime moves by up to 0.071 relative L2 between MPS and CPU. Native
        // Metal may move three times that, the rule the probability budget of
        // the image decision test also follows.
        const MAX_RELATIVE_L2: f64 = 0.21;
        let mut worst = 0.0f64;
        let mut worst_case = String::new();
        for case in &receipt.cases {
            let pixels =
                case.arrays.pixel_values.as_ref().ok_or_else(|| {
                    Error::Other(format!("{} has no pixel_values array", case.id))
                })?;
            let features =
                case.arrays.image_features.as_ref().ok_or_else(|| {
                    Error::Other(format!("{} has no image_features array", case.id))
                })?;
            let pixels = read_f32(base, &pixels.file)?;
            let expected_final = read_f32(base, &features.file)?;
            if case.grid_thw.len() != 3 {
                return Err(Error::Other(format!(
                    "{} grid_thw must have 3 entries",
                    case.id
                )));
            }
            let grids = [[case.grid_thw[0], case.grid_thw[1], case.grid_thw[2]]];
            let native_stages = model.parts.vision.forward_stages(&pixels, &grids)?;
            let stages = stage_index.get(&case.id);
            let mut case_worst = 0.0f64;
            println!("{}: grid={grids:?}", case.id);
            for (name, native_values) in &native_stages {
                let expected = match *name {
                    "merged" => Some(expected_final.clone()),
                    _ => stages
                        .and_then(|case| case.stages.get(*name))
                        .map(|file| read_f32(base, &file.file))
                        .transpose()?,
                };
                let Some(expected) = expected else { continue };
                if native_values.len() != expected.len() {
                    return Err(Error::Other(format!(
                        "{} {name}: native has {} values, reference has {}",
                        case.id,
                        native_values.len(),
                        expected.len()
                    )));
                }
                let max_delta = native_values
                    .iter()
                    .zip(&expected)
                    .map(|(actual, expected)| (*actual - *expected).abs())
                    .fold(0.0f32, f32::max);
                let norm = |values: &mut dyn Iterator<Item = f64>| {
                    values.map(|value| value * value).sum::<f64>().sqrt()
                };
                let relative_l2 = norm(
                    &mut native_values
                        .iter()
                        .zip(&expected)
                        .map(|(actual, expected)| f64::from(*actual - *expected)),
                ) / norm(&mut expected.iter().map(|value| f64::from(*value)))
                    .max(1e-12);
                println!("  {name:<13} max|Δ|={max_delta:<10.6} relative L2={relative_l2:.6}");
                if *name == "merged" {
                    case_worst = relative_l2;
                }
            }
            if case_worst > worst {
                worst = case_worst;
                worst_case = case.id.clone();
            }
        }
        assert!(
            worst <= MAX_RELATIVE_L2,
            "native vision features differ from the reference by relative L2 {worst} in {worst_case}"
        );
        Ok(())
    }

    /// Image-path acceptance through the **public** runtime: registry load →
    /// `D3Runner::decide` → `Model::image_probabilities`, compared with
    /// the checkpoint's PyTorch reference. The probability budget is three
    /// times the largest MPS-to-CPU movement of the reference itself (0.0201
    /// on an earlier case set, 0.0124 on the current recording), because the
    /// vision tower's residual stream amplifies BF16 kernel-order differences.
    #[test]
    #[ignore = "requires APXINF_D3_CHECKPOINT, APXINF_D3_VISION_REF and an Apple Metal GPU"]
    fn image_decision_matches_pytorch_reference_through_the_public_path() -> Result<()> {
        use crate::{AutoModel, DecisionInput, DecisionModel, ImageInput, LoadOptions};

        const MAX_IMAGE_ABS_DELTA: f32 = 0.06;

        #[derive(serde::Deserialize)]
        struct Receipt {
            cases: Vec<Case>,
        }
        #[derive(serde::Deserialize)]
        struct Case {
            id: String,
            token_ids: Vec<u32>,
            option_count: usize,
            grid_thw: Vec<u32>,
            probabilities: Vec<f32>,
            arrays: Arrays,
        }
        #[derive(serde::Deserialize)]
        struct Arrays {
            pixel_values: ArrayFile,
        }
        #[derive(serde::Deserialize)]
        struct ArrayFile {
            file: String,
        }

        let checkpoint = std::env::var_os("APXINF_D3_CHECKPOINT")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_CHECKPOINT to the d3-lite package directory");
        let reference = std::env::var_os("APXINF_D3_VISION_REF")
            .map(std::path::PathBuf::from)
            .expect("set APXINF_D3_VISION_REF to the reference cases.json");
        let base = reference
            .parent()
            .unwrap_or_else(|| std::path::Path::new("."));
        let receipt: Receipt = serde_json::from_str(
            &std::fs::read_to_string(&reference)
                .map_err(|error| Error::Other(format!("read {}: {error}", reference.display())))?,
        )
        .map_err(|error| Error::Other(format!("parse {}: {error}", reference.display())))?;

        let mut loaded = AutoModel::load_model(
            apxinf_core::Device::Metal(0),
            &checkpoint,
            &LoadOptions::default(),
        )?;
        assert_eq!(
            loaded
                .decision_capabilities()
                .map_err(|error| Error::Other(error.to_string()))?,
            crate::llm_trait::LlmCapabilities::VISION
        );
        let DecisionModel::Code(runtime) = loaded
            .decision_model_mut()
            .map_err(|error| Error::Other(error.to_string()))?
        else {
            return Err(Error::Contract("d3-lite loads as a code decision model"));
        };

        let mut worst = 0.0f32;
        let mut worst_case = String::new();
        let mut flips = Vec::new();
        for case in &receipt.cases {
            if case.grid_thw.len() != 3 {
                return Err(Error::Other(format!(
                    "{} grid_thw must have 3 entries",
                    case.id
                )));
            }
            let path = base.join(&case.arrays.pixel_values.file);
            let bytes = std::fs::read(&path)
                .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
            let pixels = bytes
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                .collect::<Vec<_>>();
            let rows = case.grid_thw[1] as usize * case.grid_thw[2] as usize;
            if pixels.len() != rows * 1_536 {
                return Err(Error::Other(format!(
                    "{}: {} pixel values for {rows} patches",
                    case.id,
                    pixels.len()
                )));
            }
            let pixel_tensor = apxinf_core::Tensor::from_f32(vec![rows, 1_536], &pixels)?;
            let grids = vec![[case.grid_thw[0], case.grid_thw[1], case.grid_thw[2]]];
            let output = runtime.decide(DecisionInput::with_vision(
                &case.token_ids,
                ImageInput::new(&pixel_tensor, &grids),
                case.option_count,
            ))?;
            let probabilities = output
                .scores
                .iter()
                .map(|score| score.probability)
                .collect::<Vec<_>>();
            let expected = &case.probabilities[..case.option_count.min(case.probabilities.len())];
            let decision = {
                let mut best = 0usize;
                for (index, value) in probabilities.iter().enumerate() {
                    if *value > probabilities[best] {
                        best = index;
                    }
                }
                best
            };
            let reference_decision = {
                let mut best = 0usize;
                for (index, value) in expected.iter().enumerate() {
                    if *value > expected[best] {
                        best = index;
                    }
                }
                best
            };
            let delta = probabilities
                .iter()
                .zip(expected)
                .map(|(actual, expected)| (*actual - *expected).abs())
                .fold(0.0f32, f32::max);
            println!(
                "{}: decisions {decision}/{reference_decision} max|Δp|={delta:.6}",
                case.id
            );
            if decision != reference_decision {
                flips.push(case.id.clone());
            }
            if delta > worst {
                worst = delta;
                worst_case = case.id.clone();
            }
        }
        assert!(flips.is_empty(), "image decisions flipped: {flips:?}");
        assert!(
            worst <= MAX_IMAGE_ABS_DELTA,
            "image max|Δp| {worst} in {worst_case} exceeds {MAX_IMAGE_ABS_DELTA}"
        );
        Ok(())
    }
}

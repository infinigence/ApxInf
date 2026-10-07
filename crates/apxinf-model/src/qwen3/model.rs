//! Native Qwen3 forward and request-owned KV state.
//!
//! Ordered RMSNorm/RoPE and the pure decoder-block boundary are adapted from
//! EngineTailor's MIT-licensed target 6d60bf84c949a0f1310f58007f25d206266f41cd,
//! enginetailor/models/mlx/m4/kernels.py. See README.md and LICENSE.engine-tailor.

use super::{
    config::{Qwen3Config, Variant, MAX_CONTEXT},
    weights::{LayerWeights, Weights},
};
use crate::llm_trait::{
    LlmInput, LlmTrait, TextCompilationScope, TextPreparationState, TextPreparationStatus,
};
use apxinf_core::{Backend, Device, Error, Result, Tensor};
use apxinf_loader::ModelConfig;
use apxinf_mlx::fusions::QkNormRope;
use apxinf_mlx::{Array, Compiled, MlxBackend, MlxDType, Stream};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

fn err(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn norm(x: &Array, weight_fp32: &Array, epsilon: f32) -> Result<Array> {
    let f = x.cast(MlxDType::F32)?;
    let variance = f.mul(&f)?.mean(&[-1], true)?;
    let reciprocal = variance
        .add(&Array::scalar(x.stream(), epsilon, MlxDType::F32)?)?
        .rsqrt()?;
    f.mul(&reciprocal)?.mul(weight_fp32)?.cast(x.dtype())
}

fn rotate(x: &Array, cos: &Array, sin: &Array, dim: usize) -> Result<Array> {
    let half = Array::concat(
        &[
            &x.slice_axis(3, dim / 2, dim)?.neg()?,
            &x.slice_axis(3, 0, dim / 2)?,
        ],
        3,
    )?;
    // Both products are BF16 values before addition, matching the frozen oracle.
    x.mul(cos)?.add(&half.mul(sin)?)
}

fn swiglu(gate: &Array, up: &Array) -> Result<Array> {
    gate.mul(&gate.sigmoid()?)?.mul(up)
}

#[derive(Clone)]
struct KvState {
    keys: Array,
    values: Array,
}

struct Seams {
    norm: Compiled,
    rope: Compiled,
    block: Option<Compiled>,
}

impl Seams {
    fn new(stream: &Stream, config: &Qwen3Config, decoder: bool) -> Result<Self> {
        let eps = config.rms_norm_eps;
        let dims = config.head_dim;
        let norm = Compiled::new(stream, 1, move |a| {
            if a.len() != 2 {
                return Err(err("norm trace requires x and FP32 weight"));
            }
            Ok(vec![norm(&a[0], &a[1], eps)?])
        })?;
        let rope = Compiled::new(stream, 1, move |a| {
            if a.len() != 3 {
                return Err(err("RoPE trace requires x, cos, sin"));
            }
            Ok(vec![rotate(&a[0], &a[1], &a[2], dims)?])
        })?;
        let c = config.clone();
        let block = if decoder {
            Some(Compiled::new(stream, 3, move |a| decoder_block(a, &c))?)
        } else {
            None
        };
        Ok(Self { norm, rope, block })
    }
}

/// Pure B=1/L=1 block. Weights, index, table row and KV state are explicit inputs.
/// No mutable request state, host readback, or model object is captured.
fn decoder_block(a: &[Array], c: &Qwen3Config) -> Result<Vec<Array>> {
    if a.len() != 17 {
        return Err(err("Qwen3 decoder block expects 17 arrays"));
    }
    let (x, keys, values, index, cos, sin) = (&a[0], &a[1], &a[2], &a[3], &a[4], &a[5]);
    if x.shape() != [1, 1, c.hidden_size] || x.dtype() != MlxDType::BF16 {
        return Err(err("compiled Qwen3 block requires B1/L1/BF16"));
    }
    let n = norm(x, &a[6], c.rms_norm_eps)?;
    let q = norm(
        &n.matmul(&a[10])?
            .reshape(&[1, 1, c.num_attention_heads, c.head_dim])?,
        &a[7],
        c.rms_norm_eps,
    )?
    .transpose(&[0, 2, 1, 3])?;
    let k = norm(
        &n.matmul(&a[11])?
            .reshape(&[1, 1, c.num_key_value_heads, c.head_dim])?,
        &a[8],
        c.rms_norm_eps,
    )?
    .transpose(&[0, 2, 1, 3])?;
    let v = n
        .matmul(&a[12])?
        .reshape(&[1, 1, c.num_key_value_heads, c.head_dim])?
        .transpose(&[0, 2, 1, 3])?;
    let q = rotate(&q, cos, sin, c.head_dim)?;
    let k = rotate(&k, cos, sin, c.head_dim)?;
    let keys = keys.slice_update(&k, index, &[2])?;
    let values = values.slice_update(&v, index, &[2])?;
    let mask = Array::arange(x.stream(), 0., keys.shape()[2] as f32, 1., MlxDType::I32)?
        .less_equal(index)?
        .reshape(&[1, 1, 1, keys.shape()[2]])?;
    let attention = q
        .sdpa(
            &keys,
            &values,
            (c.head_dim as f64).powf(-0.5) as f32,
            false,
            Some(&mask),
        )?
        .transpose(&[0, 2, 1, 3])?
        .reshape(&[1, 1, c.num_attention_heads * c.head_dim])?
        .matmul(&a[13])?;
    let hidden = x.add(&attention)?;
    let n = norm(&hidden, &a[9], c.rms_norm_eps)?;
    let output = hidden.add(&swiglu(&n.matmul(&a[14])?, &n.matmul(&a[15])?)?.matmul(&a[16])?)?;
    Ok(vec![output, keys, values])
}

pub struct Qwen3Model {
    config: Qwen3Config,
    variant: Variant,
    backend: MlxBackend,
    weights: Weights,
    seams: Option<Seams>,
    qk_fusion: Option<QkNormRope>,
    activation: Compiled,
    initial_kv: Vec<KvState>,
    kv: Vec<KvState>,
    cos: Option<Array>,
    sin: Option<Array>,
    capacity: usize,
    position: usize,
    prepared_lengths: HashSet<usize>,
    invalid: Option<String>,
}

impl Qwen3Model {
    pub fn new(
        config: Qwen3Config,
        tensors: HashMap<String, Tensor>,
        backend: Arc<dyn Backend>,
        variant: Variant,
    ) -> Result<Self> {
        config.validate_checkpoint_scope()?;
        Self::construct(config, tensors, backend, variant)
    }

    fn construct(
        config: Qwen3Config,
        tensors: HashMap<String, Tensor>,
        backend: Arc<dyn Backend>,
        variant: Variant,
    ) -> Result<Self> {
        config.validate()?;
        if variant == Variant::MixedW8 {
            config.validate_checkpoint_scope()?;
        }
        let backend = backend
            .as_any()
            .downcast_ref::<MlxBackend>()
            .ok_or_else(|| err("Qwen3 native implementation requires an MLX backend"))?
            .clone();
        let weights = Weights::load(&config, tensors, backend.stream(), variant)?;
        let seams = if variant != Variant::Bf16Public {
            Some(Seams::new(
                backend.stream(),
                &config,
                variant == Variant::Bf16Compiled,
            )?)
        } else {
            None
        };
        let qk_fusion = if variant == Variant::MixedW8 {
            Some(QkNormRope::new(backend.stream())?)
        } else {
            None
        };
        // mlx-lm's public swiglu is itself shapeless-compiled. Preserve that
        // boundary for both variants instead of introducing extra BF16 stores.
        let activation = Compiled::with_shapeless(backend.stream(), 1, true, |a| {
            if a.len() != 2 {
                return Err(err("SwiGLU expects gate and up arrays"));
            }
            Ok(vec![swiglu(&a[0], &a[1])?])
        })?;
        let mut model = Self {
            config,
            variant,
            backend,
            weights,
            seams,
            qk_fusion,
            activation,
            initial_kv: Vec::new(),
            kv: Vec::new(),
            cos: None,
            sin: None,
            capacity: 0,
            position: 0,
            prepared_lengths: HashSet::new(),
            invalid: None,
        };
        model.prepare(1, 0)?;
        Ok(model)
    }

    pub fn variant(&self) -> &'static str {
        self.variant.name()
    }
    pub fn position(&self) -> usize {
        self.position
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Prepare a complete prompt and the requested decode capacity outside timing.
    /// Growing capacity is only supported on a fresh/reset request.
    pub fn prepare(&mut self, prompt_len: usize, max_new_tokens: usize) -> Result<()> {
        if let Err(error) = self.prepare_inner(prompt_len, max_new_tokens) {
            self.prepared_lengths.clear();
            self.invalid = Some(error.to_string());
            return Err(error);
        }
        Ok(())
    }

    fn prepare_inner(&mut self, prompt_len: usize, max_new_tokens: usize) -> Result<()> {
        let requested = prompt_len
            .checked_add(max_new_tokens)
            .ok_or_else(|| err("Qwen3 context overflow"))?;
        if prompt_len == 0 || requested > self.config.max_position_embeddings.min(MAX_CONTEXT) {
            return Err(err(
                "Qwen3 requested context is empty or exceeds its checkpoint/2048-token migration profile",
            ));
        }
        if let Some(reason) = &self.invalid {
            return Err(err(format!(
                "Qwen3 session invalidated: {reason}; reset first"
            )));
        }
        let stream = self.backend.stream();
        if requested > self.capacity {
            if self.position != 0 {
                return Err(err("reset Qwen3 before growing its prepared KV capacity"));
            }
            let capacity = requested
                .checked_add(255)
                .ok_or_else(|| err("KV capacity overflow"))?
                / 256
                * 256;
            let capacity = capacity
                .min(self.config.max_position_embeddings)
                .min(MAX_CONTEXT);
            let shape = [
                1,
                self.config.num_key_value_heads,
                capacity,
                self.config.head_dim,
            ];
            let mut initial = Vec::with_capacity(self.config.num_hidden_layers);
            for _ in 0..self.config.num_hidden_layers {
                initial.push(KvState {
                    keys: Array::zeros(stream, &shape, MlxDType::BF16)?,
                    values: Array::zeros(stream, &shape, MlxDType::BF16)?,
                });
            }
            let dim = self.config.head_dim;
            let exponents = Array::arange(stream, 0., dim as f32, 2., MlxDType::F32)?
                .div(&Array::scalar(stream, dim as f32, MlxDType::F32)?)?;
            let inv_freq = Array::scalar(stream, 1., MlxDType::F32)?.div(
                &Array::scalar(stream, self.config.rope_theta, MlxDType::F32)?.pow(&exponents)?,
            )?;
            let positions = Array::arange(stream, 0., capacity as f32, 1., MlxDType::F32)?;
            let freqs = positions
                .reshape(&[capacity, 1])?
                .mul(&inv_freq.reshape(&[1, dim / 2])?)?;
            let angles = Array::concat(&[&freqs, &freqs], 1)?;
            let cos = angles
                .cos()?
                .cast(MlxDType::BF16)?
                .reshape(&[1, 1, capacity, dim])?;
            let sin = angles
                .sin()?
                .cast(MlxDType::BF16)?
                .reshape(&[1, 1, capacity, dim])?;
            let mut arrays: Vec<_> = initial
                .iter()
                .flat_map(|kv| [kv.keys.clone(), kv.values.clone()])
                .collect();
            arrays.extend([cos.clone(), sin.clone()]);
            stream.eval(&arrays)?;
            self.initial_kv = initial;
            self.kv = self.initial_kv.clone();
            self.cos = Some(cos);
            self.sin = Some(sin);
            self.capacity = capacity;
            self.prepared_lengths.clear();
        }
        self.warm_length(1)?;
        self.warm_length(prompt_len)?;
        // Source append_logits evaluates cache-only chunks, followed by one
        // last-token forward. Register every prompt shape outside inference.
        let prefix = prompt_len - 1;
        if prefix >= 512 {
            self.warm_length(512)?;
        }
        if prefix % 512 != 0 {
            self.warm_length(prefix % 512)?;
        }
        self.backend.stream().synchronize()?;
        Ok(())
    }

    fn warm_length(&mut self, len: usize) -> Result<()> {
        if self.prepared_lengths.contains(&len) {
            return Ok(());
        }
        let activation_input = Array::zeros(
            self.backend.stream(),
            &[1, len, self.config.intermediate_size],
            MlxDType::BF16,
        )?;
        self.activation
            .prepare(&[activation_input.clone(), activation_input])?;
        if let Some(seams) = &self.seams {
            let s = self.backend.stream();
            let c = &self.config;
            let layer = &self.weights.layers[0];
            let x = Array::zeros(s, &[1, len, c.hidden_size], MlxDType::BF16)?;
            let cos = self.cos.as_ref().unwrap().slice_axis(2, 0, len)?;
            let sin = self.sin.as_ref().unwrap().slice_axis(2, 0, len)?;
            let mut warm = seams.norm.prepare(&[x.clone(), layer.input_norm.clone()])?;
            for (heads, weight) in [
                (c.num_attention_heads, &layer.q_norm),
                (c.num_key_value_heads, &layer.k_norm),
            ] {
                let n = Array::zeros(s, &[1, len, heads, c.head_dim], MlxDType::BF16)?;
                warm.extend(seams.norm.prepare(&[n.clone(), weight.clone()])?);
                warm.extend(seams.rope.prepare(&[
                    n.transpose(&[0, 2, 1, 3])?,
                    cos.clone(),
                    sin.clone(),
                ])?);
            }
            if len == 1 && seams.block.is_some() {
                let index = Array::from_i32(s, &[1], &[0])?;
                let mut args = vec![
                    x,
                    self.initial_kv[0].keys.clone(),
                    self.initial_kv[0].values.clone(),
                    index,
                    cos,
                    sin,
                ];
                args.extend(layer.block_arrays()?);
                warm.extend(seams.block.as_ref().unwrap().prepare(&args)?);
            }
            if len == 1 {
                if let Some(fusion) = &self.qk_fusion {
                    let q = Array::zeros(
                        s,
                        &[1, 1, c.num_attention_heads, c.head_dim],
                        MlxDType::BF16,
                    )?;
                    let k = Array::zeros(
                        s,
                        &[1, 1, c.num_key_value_heads, c.head_dim],
                        MlxDType::BF16,
                    )?;
                    let (q, k) = fusion.call(
                        &q,
                        &k,
                        &layer.q_norm,
                        &layer.k_norm,
                        &self.cos.as_ref().unwrap().slice_axis(2, 0, 1)?,
                        &self.sin.as_ref().unwrap().slice_axis(2, 0, 1)?,
                    )?;
                    warm.extend([q, k]);
                }
            }
            s.eval(&warm)?;
        }
        self.prepared_lengths.insert(len);
        Ok(())
    }

    fn normalized(&self, x: &Array, weight: &Array) -> Result<Array> {
        match &self.seams {
            Some(s) => Ok(s.norm.call(&[x.clone(), weight.clone()])?.remove(0)),
            None => norm(x, weight, self.config.rms_norm_eps),
        }
    }

    fn rotated(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        match &self.seams {
            Some(s) => Ok(s
                .rope
                .call(&[x.clone(), cos.clone(), sin.clone()])?
                .remove(0)),
            None => rotate(x, cos, sin, self.config.head_dim),
        }
    }

    fn eager_layer(
        &self,
        x: &Array,
        layer: &LayerWeights,
        kv: &KvState,
        index: &Array,
        cos: &Array,
        sin: &Array,
        end: usize,
    ) -> Result<(Array, KvState)> {
        let c = &self.config;
        let len = x.shape()[1];
        let n = self.normalized(x, &layer.input_norm)?;
        let q = layer
            .q
            .call(&n)?
            .reshape(&[1, len, c.num_attention_heads, c.head_dim])?;
        let k = layer
            .k
            .call(&n)?
            .reshape(&[1, len, c.num_key_value_heads, c.head_dim])?;
        let (q, k) = if len == 1 && self.qk_fusion.is_some() {
            self.qk_fusion
                .as_ref()
                .unwrap()
                .call(&q, &k, &layer.q_norm, &layer.k_norm, cos, sin)?
        } else {
            let q = self
                .normalized(&q, &layer.q_norm)?
                .transpose(&[0, 2, 1, 3])?;
            let k = self
                .normalized(&k, &layer.k_norm)?
                .transpose(&[0, 2, 1, 3])?;
            (self.rotated(&q, cos, sin)?, self.rotated(&k, cos, sin)?)
        };
        let v = layer
            .v
            .call(&n)?
            .reshape(&[1, len, c.num_key_value_heads, c.head_dim])?
            .transpose(&[0, 2, 1, 3])?;
        let keys = kv.keys.slice_update(&k, index, &[2])?;
        let values = kv.values.slice_update(&v, index, &[2])?;
        let k = keys.slice_axis(2, 0, end)?;
        let v = values.slice_axis(2, 0, end)?;
        let y = q
            .sdpa(&k, &v, (c.head_dim as f64).powf(-0.5) as f32, len > 1, None)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, len, c.num_attention_heads * c.head_dim])?;
        let y = layer.o.call(&y)?;
        let hidden = x.add(&y)?;
        let n = self.normalized(&hidden, &layer.post_norm)?;
        let (gate, up) = layer.gate_up(&n)?;
        let activated = self.activation.call(&[gate, up])?.remove(0);
        let out = hidden.add(&layer.down.call(&activated)?)?;
        Ok((out, KvState { keys, values }))
    }

    fn run(
        &self,
        tokens: &[u32],
        end: usize,
        emit_logits: bool,
    ) -> Result<(Option<Tensor>, Vec<KvState>)> {
        let s = self.backend.stream();
        let len = tokens.len();
        let ids = Array::from_i32(
            s,
            &[len],
            &tokens.iter().map(|&id| id as i32).collect::<Vec<_>>(),
        )?;
        let mut x = self
            .weights
            .embed(&ids)?
            .reshape(&[1, len, self.config.hidden_size])?;
        let index = Array::from_i32(s, &[1], &[self.position as i32])?;
        let cos = self
            .cos
            .as_ref()
            .unwrap()
            .slice_axis(2, self.position, end)?;
        let sin = self
            .sin
            .as_ref()
            .unwrap()
            .slice_axis(2, self.position, end)?;
        let mut pending = Vec::with_capacity(self.kv.len());
        for (layer, kv) in self.weights.layers.iter().zip(&self.kv) {
            // The inherited block replacement applies only after prefill has
            // established request KV. A fresh single-token prompt keeps the
            // public attention boundary, just like a source KVCache with no keys.
            if len == 1 && self.position > 0 && self.variant == Variant::Bf16Compiled {
                let mut args = vec![
                    x,
                    kv.keys.clone(),
                    kv.values.clone(),
                    index.clone(),
                    cos.clone(),
                    sin.clone(),
                ];
                args.extend(layer.block_arrays()?);
                let mut values = self
                    .seams
                    .as_ref()
                    .unwrap()
                    .block
                    .as_ref()
                    .unwrap()
                    .call(&args)?
                    .into_iter();
                x = values.next().unwrap();
                pending.push(KvState {
                    keys: values.next().unwrap(),
                    values: values.next().unwrap(),
                });
            } else {
                let result = self.eager_layer(&x, layer, kv, &index, &cos, &sin, end)?;
                x = result.0;
                pending.push(result.1);
            }
        }
        let output = if emit_logits {
            let logits = self
                .weights
                .project(&self.normalized(&x, &self.weights.final_norm)?)?
                .reshape(&[len, self.config.vocab_size])?;
            Some(self.backend.from_array(logits)?)
        } else {
            None
        };
        let mut required = Vec::with_capacity(pending.len() * 2 + 1);
        if let Some(output) = &output {
            required.push(self.backend.array(output)?);
        }
        required.extend(
            pending
                .iter()
                .flat_map(|kv| [kv.keys.clone(), kv.values.clone()]),
        );
        s.eval(&required)?;
        // Stream::eval includes stream synchronization and exception checking.
        Ok((output, pending))
    }

    fn validate_input(&self, token_ids: &[u32], start_pos: usize) -> Result<usize> {
        if let Some(reason) = &self.invalid {
            return Err(err(format!(
                "Qwen3 session invalidated: {reason}; reset required"
            )));
        }
        let end = self
            .position
            .checked_add(token_ids.len())
            .ok_or_else(|| err("Qwen3 position overflow"))?;
        if token_ids.is_empty()
            || start_pos != self.position
            || end > self.capacity
            || token_ids
                .iter()
                .any(|&id| id as usize >= self.config.vocab_size)
        {
            return Err(err("Qwen3 forward requires nonempty in-vocabulary IDs, matching position and prepared capacity"));
        }
        Ok(end)
    }

    fn execute(
        &mut self,
        token_ids: &[u32],
        start_pos: usize,
        emit_logits: bool,
    ) -> Result<Option<Tensor>> {
        let end = self.validate_input(token_ids, start_pos)?;
        if !self.prepared_lengths.contains(&token_ids.len()) {
            return Err(err("Qwen3 input length was not prepared; call prepare or prewarm_decode before inference"));
        }
        match self.run(token_ids, end, emit_logits) {
            Ok((output, pending)) => {
                self.kv = pending;
                self.position = end;
                Ok(output)
            }
            Err(error) => {
                self.invalid = Some(error.to_string());
                Err(error)
            }
        }
    }
}

impl LlmTrait for Qwen3Model {
    fn preparation_status(&self) -> TextPreparationStatus {
        let implementation_ready = match self.variant {
            Variant::Bf16Public => true,
            Variant::Bf16Compiled => self.seams.as_ref().is_some_and(|s| s.block.is_some()),
            Variant::MixedW8 => self.seams.is_some() && self.qk_fusion.is_some(),
        };
        let ready = self.invalid.is_none()
            && implementation_ready
            && self.capacity > 0
            && self.kv.len() == self.config.num_hidden_layers
            && self.initial_kv.len() == self.config.num_hidden_layers
            && self.cos.is_some()
            && self.sin.is_some()
            && self.prepared_lengths.contains(&1)
            && self
                .prepared_lengths
                .iter()
                .all(|&n| n > 0 && n <= self.capacity);
        let state = if self.invalid.is_some() {
            TextPreparationState::Invalidated
        } else if ready {
            TextPreparationState::Ready
        } else {
            TextPreparationState::Unprepared
        };
        let mut lengths: Vec<usize> = if ready {
            self.prepared_lengths.iter().copied().collect()
        } else {
            Vec::new()
        };
        lengths.sort_unstable();
        let mut scopes = Vec::new();
        if ready {
            // Public SwiGLU is itself compiled in the frozen source.
            scopes.push(TextCompilationScope::LocalSubgraphs);
            if self.variant == Variant::Bf16Compiled {
                scopes.push(TextCompilationScope::DecoderBlock);
            }
        }
        TextPreparationStatus {
            state,
            implementation: "qwen3-mlx",
            variant: Some(self.variant.name()),
            compiled_scopes: scopes,
            prepared_prompt_tokens: None,
            prepared_sequence_lengths: lengths,
            kv_capacity: (self.capacity > 0).then_some(self.capacity),
            max_decode_rows: ready.then_some(1),
            error: self.invalid.clone(),
        }
    }

    fn load(
        _config: ModelConfig,
        _weights: HashMap<String, Tensor>,
        _device: Device,
    ) -> Result<Self> {
        Err(err("Qwen3 requires its checkpoint config with explicit head_dim; use AutoModel or Qwen3Model::new"))
    }

    fn forward(&mut self, token_ids: &[u32], start_pos: u32) -> Result<Tensor> {
        self.execute(token_ids, start_pos as usize, true)?
            .ok_or_else(|| err("missing Qwen3 logits"))
    }

    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        if input.image.is_some() {
            return Err(err("Qwen3 does not support images"));
        }
        self.validate_input(input.token_ids, 0)?;
        let prefix = input.token_ids.len() - 1;
        // Check the entire schedule before consuming any part of the prompt.
        if !self.prepared_lengths.contains(&1)
            || (prefix >= 512 && !self.prepared_lengths.contains(&512))
            || (prefix % 512 != 0 && !self.prepared_lengths.contains(&(prefix % 512)))
        {
            return Err(err("Qwen3 prefill schedule was not prepared"));
        }
        for chunk in input.token_ids[..prefix].chunks(512) {
            self.execute(chunk, self.position, false)?;
        }
        self.execute(&input.token_ids[prefix..], self.position, true)?
            .ok_or_else(|| err("missing Qwen3 prefill logits"))
    }

    fn backend(&self) -> &dyn Backend {
        &self.backend
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn reset(&mut self) {
        self.kv = self.initial_kv.clone();
        self.position = 0;
        self.invalid = None;
    }
    fn prewarm_decode(&mut self, prompt_len: usize, max_new_tokens: usize) {
        if let Err(error) = self.prepare(prompt_len, max_new_tokens) {
            self.invalid = Some(error.to_string());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::bf16;

    fn tiny(variant: Variant) -> Qwen3Model {
        let config = Qwen3Config::from_json(
            r#"{
            "model_type":"qwen3", "hidden_size":8, "intermediate_size":16,
            "num_hidden_layers":2, "num_attention_heads":2, "num_key_value_heads":1,
            "head_dim":8, "vocab_size":16, "max_position_embeddings":512,
            "rms_norm_eps":0.000001, "rope_theta":1000000,
            "tie_word_embeddings":true, "hidden_act":"silu"}"#,
        )
        .unwrap();
        let mut weights = HashMap::new();
        let mut insert = |name: String, shape: &[usize], norm: bool| {
            let n = shape.iter().product();
            let values: Vec<_> = (0..n)
                .map(|i| {
                    bf16::from_f32(if norm {
                        1.
                    } else {
                        ((i * 7 + 3) % 37) as f32 / 37. - 0.5
                    })
                })
                .collect();
            weights.insert(name, Tensor::from_bf16(shape.to_vec(), &values).unwrap());
        };
        insert("model.embed_tokens.weight".into(), &[16, 8], false);
        insert("model.norm.weight".into(), &[8], true);
        for i in 0..2 {
            for suffix in [
                "input_layernorm",
                "post_attention_layernorm",
                "self_attn.q_norm",
                "self_attn.k_norm",
            ] {
                insert(format!("model.layers.{i}.{suffix}.weight"), &[8], true);
            }
            for (suffix, shape) in [
                ("self_attn.q_proj", [16, 8]),
                ("self_attn.k_proj", [8, 8]),
                ("self_attn.v_proj", [8, 8]),
                ("self_attn.o_proj", [8, 16]),
                ("mlp.gate_proj", [16, 8]),
                ("mlp.up_proj", [16, 8]),
                ("mlp.down_proj", [8, 16]),
            ] {
                insert(format!("model.layers.{i}.{suffix}.weight"), &shape, false);
            }
        }
        Qwen3Model::construct(
            config,
            weights,
            Arc::new(MlxBackend::new(0).unwrap()),
            variant,
        )
        .unwrap()
    }

    fn values(model: &Qwen3Model, tensor: &Tensor) -> Vec<f32> {
        model.backend.to_cpu(tensor).unwrap().to_f32_vec().unwrap()
    }

    #[test]
    #[ignore = "requires exclusive Apple Silicon Metal test slot and pinned MLX SDK"]
    fn prepared_state_changed_inputs_and_retained_outputs() {
        let mut model = tiny(Variant::Bf16Compiled);
        model.prepare(3, 4).unwrap();
        let status = model.preparation_status();
        assert_eq!(status.state, TextPreparationState::Ready);
        assert_eq!(status.prepared_sequence_lengths, vec![1, 2, 3]);
        assert_eq!(
            status.compiled_scopes,
            vec![
                TextCompilationScope::LocalSubgraphs,
                TextCompilationScope::DecoderBlock
            ]
        );
        assert_eq!(status.kv_capacity, Some(model.capacity()));
        let first = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(first.shape().dims(), &[3, 16]);
        let retained = values(&model, &first);
        let next = model.forward(&[4], 3).unwrap();
        let expected_next = values(&model, &next);
        assert_eq!(model.position(), 4);
        assert!(model.forward(&[5], 0).is_err());
        assert_eq!(model.position(), 4);
        model.reset();
        let repeated = model.forward(&[1, 2, 3], 0).unwrap();
        assert_eq!(values(&model, &repeated), retained);
        let next_again = model.forward(&[4], 3).unwrap();
        assert_eq!(values(&model, &next_again), expected_next);
        model.reset();
        let changed = model.forward(&[3, 2, 1], 0).unwrap();
        assert_ne!(values(&model, &changed), retained);
        assert_eq!(values(&model, &first), retained);
    }

    #[test]
    #[ignore = "requires exclusive Apple Silicon Metal test slot and pinned MLX SDK"]
    fn prefill_schedule_matches_explicit_cache_prefix_and_final_token() {
        let mut model = tiny(Variant::Bf16Compiled);
        model.prepare(3, 4).unwrap();
        let output = model.prefill(LlmInput::text(&[1, 7, 3])).unwrap();
        assert_eq!(model.position(), 3);
        assert_eq!(output.shape().dims(), &[1, 16]);
        let expected = values(&model, &output);
        model.reset();
        // A full forward keeps its [length,vocab] contract. Evaluating the
        // discarded prefix head cannot alter functional explicit KV state.
        let prefix = model.forward(&[1, 7], 0).unwrap();
        assert_eq!(prefix.shape().dims(), &[2, 16]);
        let last = model.forward(&[3], 2).unwrap();
        assert_eq!(values(&model, &last), expected);
        model.reset();
        assert!(model.prepare(MAX_CONTEXT + 1, 0).is_err());
        let invalid = model.preparation_status();
        assert_eq!(invalid.state, TextPreparationState::Invalidated);
        assert!(invalid.compiled_scopes.is_empty());
        assert!(invalid.prepared_sequence_lengths.is_empty());
        assert!(model.forward(&[1], 0).is_err());
        model.reset();
        assert_eq!(
            model.preparation_status().state,
            TextPreparationState::Unprepared
        );
        model.prepare(3, 4).unwrap();
        assert_eq!(
            model.preparation_status().state,
            TextPreparationState::Ready
        );
        let repeated = model.prefill(LlmInput::text(&[1, 7, 3])).unwrap();
        assert_eq!(values(&model, &repeated), expected);
    }

    #[test]
    #[ignore = "requires exclusive Apple Silicon Metal test slot and pinned MLX SDK"]
    fn public_and_compiled_reference_trajectory_agree() {
        let mut public = tiny(Variant::Bf16Public);
        let mut compiled = tiny(Variant::Bf16Compiled);
        public.prepare(3, 8).unwrap();
        compiled.prepare(3, 8).unwrap();
        let p = public.forward(&[1, 7, 3], 0).unwrap();
        let c = compiled.forward(&[1, 7, 3], 0).unwrap();
        // BF16 rounding and SDPA tiling can differ at the last bit. This checks
        // complete finite logits and per-row greedy parity, not bitwise kernels.
        let check = |p: &Tensor, c: &Tensor| {
            let a = values(&public, p);
            let b = values(&compiled, c);
            assert_eq!(a.len(), b.len());
            for (&x, &y) in a.iter().zip(&b) {
                assert!(
                    x.is_finite() && y.is_finite() && (x - y).abs() <= 0.05 + 0.03 * x.abs(),
                    "{x} != {y}"
                );
            }
            for (a, b) in a.chunks(16).zip(b.chunks(16)) {
                let argmax = |v: &[f32]| {
                    v.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1))
                        .unwrap()
                        .0
                };
                assert_eq!(argmax(a), argmax(b));
            }
        };
        check(&p, &c);
        // Both sides consume this same input trajectory.
        for (position, token) in [(3, 4), (4, 9), (5, 2)] {
            let p = public.forward(&[token], position).unwrap();
            let c = compiled.forward(&[token], position).unwrap();
            // Each pair is checked after actual evaluation, including new KV.
            let a = values(&public, &p);
            let b = values(&compiled, &c);
            for (&x, &y) in a.iter().zip(&b) {
                assert!(x.is_finite() && y.is_finite() && (x - y).abs() <= 0.05 + 0.03 * x.abs());
            }
        }
    }
}

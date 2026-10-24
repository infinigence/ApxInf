//! Native MiniCPM5 forward and request state. No Python model provider.
//! Arithmetic/schedule adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin.
use super::{
    config::{attention_width, Config, PrefillChunk, PrefillPlan, Variant},
    math,
    weights::Weights,
};
use crate::{
    LlmInput, LlmTrait, TextCompilationScope, TextPreparationState, TextPreparationStatus,
};
use apxinf_core::{Backend, Device, Error, Result, Tensor};
use apxinf_loader::ModelConfig;
use apxinf_mlx::fusions::PackedResidualRmsNorm;
use apxinf_mlx::{Array, Compiled, MlxBackend, MlxDType};
use std::{collections::HashMap, rc::Rc};

struct LocalMath {
    norm: Compiled,
    rope: Compiled,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DecodeAttention {
    Bucketed,
    Exact,
}

struct PreparedRequest {
    prefill: PrefillPlan,
    output: usize,
    positions: HashMap<usize, Array>,
}

pub struct MiniCpm5 {
    pub(crate) backend: MlxBackend,
    pub(crate) weights: Rc<Weights>,
    config: Config,
    variant: Variant,
    prefill_chunk: Option<PrefillChunk>,
    epsilon: Array,
    cos: Array,
    sin: Array,
    local: Option<LocalMath>,
    swiglu: Compiled,
    step: Option<Compiled>,
    attention: DecodeAttention,
    prepared: Option<PreparedRequest>,
    states: Vec<Array>,
    capacity: usize,
    pub(crate) offset: usize,
    pub(crate) last_taps: Option<Array>,
    failure: Option<String>,
}

impl MiniCpm5 {
    pub(crate) fn supports_dspark(&self) -> bool {
        self.variant == Variant::DSpark
    }

    pub fn new(
        config: Config,
        weights: Weights,
        backend: MlxBackend,
        variant: Variant,
    ) -> Result<Self> {
        config.validate()?;
        let prefill_chunk = PrefillChunk::from_env()?;
        let stream = backend.stream();
        let epsilon = Array::scalar(stream, config.rms_norm_eps, MlxDType::F32)?;
        let (cos, sin) = math::tables(stream, 256)?;
        let local = if variant == Variant::Public {
            None
        } else {
            let eps = epsilon.clone();
            let native_norm = variant == Variant::Compiled;
            let rms_norm_eps = config.rms_norm_eps;
            Some(LocalMath {
                norm: Compiled::with_shapeless(stream, 1, true, move |a| {
                    let normalized = if native_norm {
                        a[0].fast_rms_norm(&a[1], rms_norm_eps)?
                    } else {
                        math::norm(&a[0], &a[1], &eps)?
                    };
                    Ok(vec![normalized])
                })?,
                rope: Compiled::new(stream, 1, |a| Ok(vec![math::rotate(&a[0], &a[1], &a[2])?]))?,
            })
        };
        let swiglu =
            Compiled::with_shapeless(stream, 1, true, |a| Ok(vec![math::swiglu(&a[0], &a[1])?]))?;
        let attention = if variant == Variant::Compiled
            && std::env::var("APXINF_MINICPM5_EXACT_ATTN").as_deref() == Ok("1")
        {
            DecodeAttention::Exact
        } else {
            DecodeAttention::Bucketed
        };
        Ok(Self {
            backend,
            weights: Rc::new(weights),
            config,
            variant,
            prefill_chunk,
            epsilon,
            cos,
            sin,
            local,
            swiglu,
            step: None,
            attention,
            prepared: None,
            states: Vec::new(),
            capacity: 0,
            offset: 0,
            last_taps: None,
            failure: None,
        })
    }

    /// Prepare bounded KV capacity and all admitted decode graph profiles.
    /// Exact attention evaluates endpoint profiles and traces interior profiles.
    /// Bucketed attention evaluates every prepared profile.
    /// All changing tokens, positions and K/V are explicit compiled arguments.
    pub fn prepare(&mut self, prompt: usize, output: usize) -> Result<()> {
        match self.prepare_inner(prompt, output) {
            Ok(()) => Ok(()),
            Err(error) => {
                // No partially built or unsuccessfully warmed plan is ready.
                // The next request must reset and complete preparation again.
                self.prepared = None;
                self.failure = Some(error.to_string());
                Err(error)
            }
        }
    }

    fn prepare_inner(&mut self, prompt: usize, output: usize) -> Result<()> {
        let capacity = self.config.request_capacity(self.variant, prompt, output)?;
        if self.capacity == capacity
            && self
                .prepared
                .as_ref()
                .is_some_and(|p| p.prefill.prompt == prompt && p.output == output)
            && self.failure.is_none()
            && !self.states.is_empty()
        {
            return Ok(());
        }
        if self.offset != 0 {
            return Err(Error::Contract(
                "reset before changing MiniCPM request preparation",
            ));
        }
        if self.attention == DecodeAttention::Exact {
            self.step = None;
            self.prepared = None;
        }
        if self.capacity != capacity || self.states.is_empty() {
            self.states.clear();
            self.step = None;
            self.prepared = None;
            self.capacity = 0;
            let stream = self.backend.stream();
            let (cos, sin) = math::tables(stream, capacity)?;
            self.cos = cos;
            self.sin = sin;
            let mut states = Vec::with_capacity(84);
            for _ in 0..84 {
                states.push(Array::zeros(
                    stream,
                    &[1, 2, capacity, 128],
                    MlxDType::BF16,
                )?);
            }
            stream.eval(&states)?;
            self.states = states;
            self.capacity = capacity;
        }
        if self.step.is_none() && self.variant != Variant::Public {
            self.step = Some(self.compile_step()?);
        }
        // Warm shape-dependent RoPE traces used by prefill. Norm and SwiGLU
        // are shapeless; changed request values are never captured.
        let prefill = PrefillPlan::new(
            prompt,
            self.prefill_chunk.unwrap_or(PrefillChunk::Tokens512),
        )?;
        let lengths = prefill.required_rows();
        for &rows in &lengths {
            if let Some(local) = &self.local {
                let stream = self.backend.stream();
                let zeros = Array::zeros(stream, &[1, rows, 2048], MlxDType::BF16)?;
                local.norm.prepare(&[zeros, self.weights.norm.clone()])?;
                for heads in [2, 16] {
                    let zeros = Array::zeros(stream, &[1, heads, rows, 128], MlxDType::BF16)?;
                    local.rope.prepare(&[
                        zeros,
                        self.cos.slice_axis(2, 0, rows)?,
                        self.sin.slice_axis(2, 0, rows)?,
                    ])?;
                }
            }
        }
        // MLX-LM's public activation is compiled too. Its warmup is independent
        // of the optional norm/RoPE seams and covers eager prompt/decode shapes.
        for rows in lengths.into_iter().chain(std::iter::once(1)) {
            let activation = Array::zeros(self.backend.stream(), &[1, rows, 6144], MlxDType::BF16)?;
            self.swiglu.prepare(&[activation.clone(), activation])?;
        }
        let mut positions = self
            .prepared
            .take()
            .map(|p| p.positions)
            .unwrap_or_default();
        if let Some(step) = &self.step {
            // Shape-specialized DSpark verification warms all admitted row counts.
            let max_rows = if self.variant == Variant::DSpark {
                8
            } else {
                1
            };
            let (first, last, stride) = match self.attention {
                DecodeAttention::Exact => (prompt, prompt + output.saturating_sub(1), 1),
                DecodeAttention::Bucketed => (
                    attention_width(self.variant, prompt, capacity),
                    capacity,
                    256,
                ),
            };
            let mut width = first;
            loop {
                if !positions.contains_key(&width) {
                    let key_positions =
                        Array::arange(self.backend.stream(), 0., width as f32, 1., MlxDType::I32)?;
                    for rows in 1..=max_rows {
                        let tokens =
                            Array::from_i32(self.backend.stream(), &[1, rows], &vec![0; rows])?;
                        let position = if self.attention == DecodeAttention::Exact {
                            width - 1
                        } else {
                            0
                        };
                        let args =
                            self.step_inputs(tokens, position, rows, &self.states, &key_positions)?;
                        if self.attention == DecodeAttention::Exact
                            && width != first
                            && width != last
                        {
                            step.prepare_trace(&args)?;
                        } else {
                            step.prepare(&args)?;
                        }
                    }
                    positions.insert(width, key_positions);
                }
                if width == last {
                    break;
                }
                width = (width + stride).min(last);
            }
        }
        self.failure = None;
        self.prepared = Some(PreparedRequest {
            prefill,
            output,
            positions,
        });
        Ok(())
    }

    fn norm(&self, x: &Array, weight: &Array) -> Result<Array> {
        match &self.local {
            Some(local) => Ok(local.norm.call(&[x.clone(), weight.clone()])?.remove(0)),
            None => math::norm(x, weight, &self.epsilon),
        }
    }

    fn rotate(&self, x: &Array, cos: &Array, sin: &Array) -> Result<Array> {
        match &self.local {
            Some(local) => Ok(local
                .rope
                .call(&[x.clone(), cos.clone(), sin.clone()])?
                .remove(0)),
            None => math::rotate(x, cos, sin),
        }
    }

    fn step_inputs(
        &self,
        tokens: Array,
        position: usize,
        rows: usize,
        states: &[Array],
        positions: &Array,
    ) -> Result<Vec<Array>> {
        let mut args = vec![
            tokens,
            Array::from_i32(self.backend.stream(), &[1], &[position as i32])?,
            self.cos.slice_axis(2, position, position + rows)?,
            self.sin.slice_axis(2, position, position + rows)?,
            positions.clone(),
        ];
        args.extend(states.iter().cloned());
        Ok(args)
    }

    fn compile_step(&self) -> Result<Compiled> {
        let stream = self.backend.stream();
        let packed = Rc::new(PackedResidualRmsNorm::new(
            stream,
            self.config.rms_norm_eps,
        )?);
        let weights = self.weights.clone();
        let eps = self.epsilon.clone();
        let tapped = self.variant == Variant::DSpark;
        let exact = self.attention == DecodeAttention::Exact;
        Compiled::new(stream, 85 + usize::from(tapped), move |a| {
            let rows = a[0].shape()[1];
            let width = a[4].shape()[0];
            let mask = if exact && rows == 1 {
                None
            } else {
                Some(math::causal_mask(&a[4], &a[1], rows)?)
            };
            let mut x = weights.embedding.take(&a[0], 0)?;
            let mut delta: Option<Array> = None;
            let mut states = Vec::with_capacity(84);
            let mut taps = Vec::new();
            let pack_rows = |x: &Array, delta: &Array, weight: &Array| -> Result<(Array, Array)> {
                let mut residuals = Vec::with_capacity(rows);
                let mut norms = Vec::with_capacity(rows);
                for row in 0..rows {
                    let (r, n) = packed.call(
                        &x.slice_axis(1, row, row + 1)?,
                        &delta.slice_axis(1, row, row + 1)?,
                        weight,
                    )?;
                    residuals.push(r);
                    norms.push(n);
                }
                Ok((
                    Array::concat(&residuals.iter().collect::<Vec<_>>(), 1)?,
                    Array::concat(&norms.iter().collect::<Vec<_>>(), 1)?,
                ))
            };
            for (i, w) in weights.layers.iter().enumerate() {
                let n = if let Some(d) = &delta {
                    let (residual, normalized) = pack_rows(&x, d, &w.input_norm)?;
                    x = residual;
                    normalized
                } else {
                    math::norm(&x, &w.input_norm, &eps)?
                };
                let project = |v: &Array, w: &Array| math::project(v, w, tapped);
                let q = project(&n, &w.q)?
                    .reshape(&[1, rows, 16, 128])?
                    .transpose(&[0, 2, 1, 3])?;
                let k = project(&n, &w.k)?
                    .reshape(&[1, rows, 2, 128])?
                    .transpose(&[0, 2, 1, 3])?;
                let v = project(&n, &w.v)?
                    .reshape(&[1, rows, 2, 128])?
                    .transpose(&[0, 2, 1, 3])?;
                let q = math::rotate(&q, &a[2], &a[3])?;
                let k = math::rotate(&k, &a[2], &a[3])?;
                let keys = a[5 + 2 * i].slice_update(&k, &a[1], &[2])?;
                let values = a[6 + 2 * i].slice_update(&v, &a[1], &[2])?;
                let attention = q
                    .sdpa(
                        &keys.slice_axis(2, 0, width)?,
                        &values.slice_axis(2, 0, width)?,
                        128f32.powf(-0.5),
                        false,
                        mask.as_ref(),
                    )?
                    .transpose(&[0, 2, 1, 3])?
                    .reshape(&[1, rows, 2048])?;
                let attention_delta = project(&attention, &w.o)?;
                let (h, n) = pack_rows(&x, &attention_delta, &w.post_norm)?;
                x = h;
                delta = Some(project(
                    &math::swiglu(&project(&n, &w.gate)?, &project(&n, &w.up)?)?,
                    &w.down,
                )?);
                states.extend([keys, values]);
                if tapped && [1, 10, 20, 30, 39].contains(&i) {
                    taps.push(pack_rows(&x, delta.as_ref().unwrap(), &w.input_norm)?.0);
                }
            }
            let (_, normalized) = pack_rows(&x, delta.as_ref().unwrap(), &weights.norm)?;
            let logits = math::project(&normalized, &weights.head, tapped)?;
            let mut output = vec![logits];
            output.extend(states);
            if tapped {
                output.push(Array::concat(&taps.iter().collect::<Vec<_>>(), 2)?);
            }
            Ok(output)
        })
    }

    fn eager(
        &self,
        tokens: &Array,
        position: usize,
        last_only: bool,
        previous: &[Array],
    ) -> Result<Vec<Array>> {
        let rows = tokens.shape()[1];
        let end = position + rows;
        let index = Array::from_i32(self.backend.stream(), &[1], &[position as i32])?;
        let cos = self.cos.slice_axis(2, position, end)?;
        let sin = self.sin.slice_axis(2, position, end)?;
        let mask = if self.variant == Variant::Compiled {
            None
        } else {
            let k_positions =
                Array::arange(self.backend.stream(), 0., end as f32, 1., MlxDType::I32)?
                    .reshape(&[1, end])?;
            let q_positions = Array::arange(
                self.backend.stream(),
                position as f32,
                end as f32,
                1.,
                MlxDType::I32,
            )?
            .reshape(&[rows, 1])?;
            let mask = k_positions
                .less_equal(&q_positions)?
                .reshape(&[1, 1, rows, end])?;
            Some(mask)
        };
        let mut x = self.weights.embedding.take(tokens, 0)?;
        let mut states = Vec::with_capacity(84);
        let mut taps = Vec::new();
        for (i, w) in self.weights.layers.iter().enumerate() {
            let n = self.norm(&x, &w.input_norm)?;
            let q = n
                .matmul(&w.q)?
                .reshape(&[1, rows, 16, 128])?
                .transpose(&[0, 2, 1, 3])?;
            let k = n
                .matmul(&w.k)?
                .reshape(&[1, rows, 2, 128])?
                .transpose(&[0, 2, 1, 3])?;
            let v = n
                .matmul(&w.v)?
                .reshape(&[1, rows, 2, 128])?
                .transpose(&[0, 2, 1, 3])?;
            let q = self.rotate(&q, &cos, &sin)?;
            let k = self.rotate(&k, &cos, &sin)?;
            let keys = previous[2 * i].slice_update(&k, &index, &[2])?;
            let values = previous[2 * i + 1].slice_update(&v, &index, &[2])?;
            let attention = q
                .sdpa(
                    &keys.slice_axis(2, 0, end)?,
                    &values.slice_axis(2, 0, end)?,
                    128f32.powf(-0.5),
                    self.variant == Variant::Compiled && rows > 1,
                    mask.as_ref(),
                )?
                .transpose(&[0, 2, 1, 3])?
                .reshape(&[1, rows, 2048])?;
            x = x.add(&attention.matmul(&w.o)?)?;
            let n = self.norm(&x, &w.post_norm)?;
            let gate = n.matmul(&w.gate)?;
            let up = n.matmul(&w.up)?;
            let act = self.swiglu.call(&[gate, up])?.remove(0);
            x = x.add(&act.matmul(&w.down)?)?;
            states.extend([keys, values]);
            if self.variant == Variant::DSpark && [1, 10, 20, 30, 39].contains(&i) {
                taps.push(x.clone());
            }
        }
        let normalized = self.norm(&x, &self.weights.norm)?;
        let head_input = if last_only {
            normalized.slice_axis(1, rows - 1, rows)?
        } else {
            normalized
        };
        let mut out = vec![head_input.matmul(&self.weights.head)?];
        out.extend(states);
        if self.variant == Variant::DSpark {
            out.push(Array::concat(&taps.iter().collect::<Vec<_>>(), 2)?);
        }
        Ok(out)
    }

    pub(crate) fn run_tokens(
        &mut self,
        ids: &[u32],
        position: usize,
        last_only: bool,
    ) -> Result<Array> {
        if let Some(failure) = &self.failure {
            return Err(Error::Other(format!(
                "MiniCPM state invalid: {failure}; reset and prepare again"
            )));
        }
        if ids.is_empty()
            || ids.iter().any(|&t| t >= self.config.vocab_size as u32)
            || position != self.offset
            || position
                .checked_add(ids.len())
                .is_none_or(|n| n > self.capacity)
        {
            return Err(Error::Contract(
                "invalid MiniCPM tokens, position, or context",
            ));
        }
        if self.states.is_empty() || self.prepared.is_none() {
            return Err(Error::Contract("prepare MiniCPM before inference"));
        }
        let prefill = self.prepared.as_ref().unwrap().prefill;
        let admitted = if position < prefill.prompt {
            prefill.admits(position, ids.len())
        } else {
            ids.len() == 1 || (self.variant == Variant::DSpark && ids.len() <= 8)
        };
        if !admitted {
            return Err(Error::Contract(
                "MiniCPM input shape is outside its prepared prompt/decode profile",
            ));
        }
        let result = (|| {
            let tokens = Array::from_i32(
                self.backend.stream(),
                &[1, ids.len()],
                &ids.iter().map(|&v| v as i32).collect::<Vec<_>>(),
            )?;
            let output = self.run_graph(tokens, position, last_only)?;
            self.backend.stream().eval(&output)?;
            Ok::<_, Error>(self.install_output(output, position + ids.len()))
        })();
        if let Err(error) = &result {
            self.failure = Some(error.to_string());
        }
        result
    }

    fn run_graph(&mut self, tokens: Array, position: usize, last_only: bool) -> Result<Vec<Array>> {
        let rows = tokens.numel();
        let compiled =
            position > 0 && (rows == 1 || (self.variant == Variant::DSpark && rows <= 8));
        let positions = if self.step.is_some() && compiled {
            let width = match self.attention {
                DecodeAttention::Exact => position + rows,
                DecodeAttention::Bucketed => {
                    attention_width(self.variant, position + rows, self.capacity)
                }
            };
            Some(
                self.prepared
                    .as_ref()
                    .and_then(|p| p.positions.get(&width))
                    .ok_or(Error::Contract("MiniCPM attention width was not prepared"))?
                    .clone(),
            )
        } else {
            None
        };
        let previous = std::mem::take(&mut self.states);
        let output = match (&self.step, compiled) {
            (Some(step), true) => step.call(&self.step_inputs(
                tokens,
                position,
                rows,
                &previous,
                positions.as_ref().unwrap(),
            )?)?,
            _ => self.eager(&tokens, position, last_only, &previous)?,
        };
        drop(previous);
        Ok(output)
    }

    fn install_output(&mut self, mut output: Vec<Array>, end: usize) -> Array {
        let logits = output.remove(0);
        self.last_taps = if self.variant == Variant::DSpark {
            output.pop()
        } else {
            None
        };
        self.states = output;
        self.offset = end;
        logits
    }

    pub(crate) fn trim_to(&mut self, position: usize) -> Result<()> {
        if position > self.offset {
            return Err(Error::Contract("MiniCPM rollback cannot advance state"));
        }
        self.offset = position;
        self.last_taps = None;
        Ok(())
    }
}

impl crate::mlx_decode::DeviceDecode for MiniCpm5 {
    fn stream(&self) -> &apxinf_mlx::Stream {
        self.backend.stream()
    }
    fn decode(&mut self, token: &Array) -> Result<Array> {
        if self.failure.is_some() || self.offset >= self.capacity || self.prepared.is_none() {
            return Err(Error::Contract(
                "MiniCPM decode requires valid prepared capacity",
            ));
        }
        let result = self.run_graph(
            token.cast(MlxDType::I32)?.reshape(&[1, 1])?,
            self.offset,
            false,
        );
        match result {
            Ok(output) => Ok(self.install_output(output, self.offset + 1)),
            Err(error) => {
                self.failure = Some(error.to_string());
                Err(error)
            }
        }
    }
    fn finish_decode(&mut self, outcome: crate::mlx_decode::DecodeFinish) -> Result<()> {
        if let Err(error) = self.backend.stream().synchronize() {
            self.failure = Some(error.to_string());
            return Err(error);
        }
        match outcome {
            crate::mlx_decode::DecodeFinish::PublishedPosition(position)
                if self.failure.is_none()
                    && position <= self.offset
                    && self.offset - position <= 1 =>
            {
                self.trim_to(position)
            }
            crate::mlx_decode::DecodeFinish::Invalidated => {
                self.failure
                    .get_or_insert_with(|| "greedy request failed or was abandoned".into());
                Ok(())
            }
            _ => {
                self.failure = Some("invalid greedy rollback position".into());
                Err(Error::Contract("invalid MiniCPM greedy rollback position"))
            }
        }
    }
}

impl LlmTrait for MiniCpm5 {
    fn token_sequence<'a>(
        &'a mut self,
        first: Tensor,
        sampler: Box<dyn apxinf_core::TokenSampler>,
        params: &apxinf_core::TokenSamplingParams,
        prompt_end: u32,
    ) -> Result<Box<dyn crate::llm_trait::TokenSequence + 'a>> {
        if self.variant == Variant::Compiled && crate::mlx_decode::enabled(params) {
            let first = self.backend.array(&first)?;
            Ok(Box::new(crate::mlx_decode::GreedySequence::new(
                self, first, prompt_end,
            )?))
        } else {
            Ok(Box::new(crate::llm_trait::SyncTokenSequence::new(
                self, first, sampler, prompt_end,
            )))
        }
    }

    fn preparation_status(&self) -> TextPreparationStatus {
        let ready = self.failure.is_none() && self.prepared.is_some() && !self.states.is_empty();
        let mut scopes = Vec::new();
        let mut lengths = Vec::new();
        let max_rows = if self.variant == Variant::DSpark {
            8
        } else {
            1
        };
        if ready {
            scopes.push(TextCompilationScope::LocalSubgraphs);
            if self.step.is_some() {
                scopes.push(TextCompilationScope::DecodeStep);
            }
            lengths.extend(1..=max_rows);
            lengths.extend(self.prepared.as_ref().unwrap().prefill.required_rows());
            lengths.sort_unstable();
            lengths.dedup();
        }
        TextPreparationStatus {
            state: if self.failure.is_some() {
                TextPreparationState::Invalidated
            } else if ready {
                TextPreparationState::Ready
            } else {
                TextPreparationState::Unprepared
            },
            implementation: "apxinf/minicpm5/mlx-0.32.3",
            variant: Some(match self.variant {
                Variant::Public => "bf16-public",
                Variant::Compiled => "bf16-compiled",
                Variant::DSpark => "dspark-target",
            }),
            compiled_scopes: scopes,
            prepared_prompt_tokens: self
                .prepared
                .as_ref()
                .filter(|_| ready)
                .map(|p| p.prefill.prompt),
            prepared_sequence_lengths: lengths,
            kv_capacity: (self.capacity > 0).then_some(self.capacity),
            max_decode_rows: ready.then_some(max_rows),
            error: self.failure.clone(),
        }
    }

    fn load(_: ModelConfig, _: HashMap<String, Tensor>, _: Device) -> Result<Self> {
        Err(Error::Contract("MiniCPM5 needs checkpoint-specific config; load through AutoModel with model_name minicpm5"))
    }
    fn forward(&mut self, ids: &[u32], start_pos: u32) -> Result<Tensor> {
        let logits = self.run_tokens(ids, start_pos as usize, false)?;
        self.backend
            .from_array(logits.reshape(&[ids.len(), self.vocab_size()])?)
    }
    fn backend(&self) -> &dyn Backend {
        &self.backend
    }
    fn reset(&mut self) {
        self.offset = 0;
        self.last_taps = None;
        self.failure = self
            .backend
            .stream()
            .synchronize()
            .err()
            .map(|e| e.to_string());
    }
    fn prewarm_decode(&mut self, prompt: usize, output: usize) {
        if let Err(e) = self.prepare(prompt, output) {
            self.failure = Some(e.to_string());
        }
    }
    fn vocab_size(&self) -> usize {
        self.config.vocab_size
    }
    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        if input.image.is_some() {
            return Err(Error::Contract("MiniCPM5 is text-only"));
        }
        if self.offset != 0
            || self.prepared.as_ref().map(|p| p.prefill.prompt) != Some(input.token_ids.len())
            || input.token_ids.is_empty()
            || input
                .token_ids
                .iter()
                .any(|&id| id as usize >= self.config.vocab_size)
        {
            return Err(Error::Contract(
                "MiniCPM prefill requires the complete prepared prompt and valid token IDs",
            ));
        }
        let plan = self.prepared.as_ref().unwrap().prefill;
        let mut logits = None;
        for range in plan.chunks() {
            logits = Some(self.run_tokens(&input.token_ids[range], self.offset, true)?);
        }
        let logits = logits.unwrap();
        self.backend
            .from_array(logits.reshape(&[1, self.vocab_size()])?)
    }
}

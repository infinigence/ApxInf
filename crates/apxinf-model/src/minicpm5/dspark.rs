//! Native official MiniCPM5 DSpark: five BF16 draft layers, seven proposals,
//! causal target verification and exact consumed-prefix rollback.
//!
//! Adapted from EngineTailor (MIT), Copyright 2026 Haiyan Qin, and mlx-dspark
//! revision d6042f38f0aa7dd3e03a2a406111d725056d45a3 (MIT), Copyright 2026 erahim3.
//! The full notices are retained in the repository's third-party notices.
use super::{model::MiniCpm5, weights::Weights, CONTEXT_CAPACITY};
use crate::{
    GeneratedToken, GenerationOutput, GenerationProfile, GenerationRequest, LlmInput, LlmTrait,
    TextCompilationScope, TextPreparationState, TextPreparationStatus,
};
use apxinf_core::{Backend, DType, Device, Error, Result, Tensor, TokenSelection};
use apxinf_loader::ModelConfig;
use apxinf_mlx::{Array, Compiled, MlxDType, Stream};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs::File, io::Read, path::Path, rc::Rc};

const VOCAB: usize = 130_560;
const BLOCK: usize = 7;
const MASK_TOKEN: i32 = 75_982;
const LAYERS: usize = 5;
const EPSILON: f32 = 1e-6;
const CONFIG_SHA256: &str = "bfbcab77ce2b466928deeb23109e7ff7738639c499d45f2c15941743b475d14b";
const WEIGHTS_SHA256: &str = "ae9ff4a8c944e2f88f266cc9452f6b8908a6d2bfce57cd4cf12cfb5cb979bc97";
const WEIGHTS_BYTES: u64 = 647_558_522;

struct DraftLayer {
    input_norm: Array,
    post_norm: Array,
    q_norm: Array,
    k_norm: Array,
    q: Array,
    k: Array,
    v: Array,
    o: Array,
    gate: Array,
    up: Array,
    down: Array,
}

struct DraftWeights {
    fc: Array,
    hidden_norm: Array,
    norm: Array,
    markov_embedding: Array,
    markov_head: Array,
    // Official tensors are validated and retained even though this admitted
    // schedule never confidence-parks or truncates the seven-slot draft.
    _confidence_weight: Array,
    _confidence_bias: Array,
    layers: Vec<DraftLayer>,
}

fn sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut chunk = vec![0; 8 * 1024 * 1024];
    loop {
        let count = file.read(&mut chunk)?;
        if count == 0 {
            break;
        }
        hash.update(&chunk[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn tensor_shapes() -> Vec<(String, Vec<usize>)> {
    let mut shapes = vec![
        ("fc.weight".into(), vec![2048, 10240]),
        ("hidden_norm.weight".into(), vec![2048]),
        ("norm.weight".into(), vec![2048]),
        ("confidence_head.proj.weight".into(), vec![1, 2304]),
        ("confidence_head.proj.bias".into(), vec![1]),
        ("markov_head.markov_w1.weight".into(), vec![VOCAB, 256]),
        ("markov_head.markov_w2.weight".into(), vec![VOCAB, 256]),
    ];
    for i in 0..LAYERS {
        for (name, shape) in [
            ("input_layernorm.weight", vec![2048]),
            ("post_attention_layernorm.weight", vec![2048]),
            ("self_attn.q_norm.weight", vec![128]),
            ("self_attn.k_norm.weight", vec![128]),
            ("self_attn.q_proj.weight", vec![2048, 2048]),
            ("self_attn.k_proj.weight", vec![256, 2048]),
            ("self_attn.v_proj.weight", vec![256, 2048]),
            ("self_attn.o_proj.weight", vec![2048, 2048]),
            ("mlp.gate_proj.weight", vec![6144, 2048]),
            ("mlp.up_proj.weight", vec![6144, 2048]),
            ("mlp.down_proj.weight", vec![2048, 6144]),
        ] {
            shapes.push((format!("layers.{i}.{name}"), shape));
        }
    }
    shapes
}

impl DraftWeights {
    fn load(stream: &Stream, snapshot: &Path) -> Result<Self> {
        if sha256(&snapshot.join("config.json"))? != CONFIG_SHA256 {
            return Err(Error::Contract(
                "DSpark requires the exact official config revision 114a20fd",
            ));
        }
        let weights = snapshot.join("model.safetensors");
        if weights.metadata()?.len() != WEIGHTS_BYTES || sha256(&weights)? != WEIGHTS_SHA256 {
            return Err(Error::Contract(
                "DSpark requires the exact official BF16 weights revision 114a20fd",
            ));
        }
        let (mut map, _) =
            apxinf_loader::safetensors::load_native(&weights).map_err(Error::Other)?;
        let shapes = tensor_shapes();
        if map.len() != shapes.len() {
            return Err(Error::Contract(
                "DSpark requires exactly 62 official tensors",
            ));
        }
        // Validate the entire inventory before allocating device parameters.
        for (name, shape) in &shapes {
            let tensor = map
                .get(name)
                .ok_or_else(|| Error::Other(format!("missing DSpark tensor {name}")))?;
            if tensor.dtype() != DType::BF16 || tensor.shape().dims() != shape.as_slice() {
                return Err(Error::Other(format!(
                    "DSpark tensor {name} must be BF16 {shape:?}"
                )));
            }
        }
        let mut arrays = HashMap::with_capacity(shapes.len());
        for (name, shape) in shapes {
            let tensor = map
                .remove(&name)
                .ok_or(Error::Contract("DSpark validated tensor disappeared"))?;
            let array = Array::from_bytes(
                stream,
                &shape,
                MlxDType::BF16,
                tensor
                    .storage()
                    .as_cpu()
                    .ok_or(Error::Contract("DSpark checkpoint tensor must be on host"))?,
            )?;
            array.eval()?;
            arrays.insert(name, array);
        }
        let mut take = |name: &str, linear: bool| -> Result<Array> {
            let a = arrays
                .remove(name)
                .ok_or_else(|| Error::Other(format!("missing DSpark device weight {name}")))?;
            // Keep the checkpoint's physical [out,in] layout as a private view.
            // A contiguous transpose can select a different reduction kernel.
            if linear {
                a.transpose(&[1, 0])
            } else {
                Ok(a)
            }
        };
        let fc = take("fc.weight", true)?;
        let hidden_norm = take("hidden_norm.weight", false)?;
        let norm = take("norm.weight", false)?;
        let markov_embedding = take("markov_head.markov_w1.weight", false)?;
        let markov_head = take("markov_head.markov_w2.weight", true)?;
        let confidence_weight = take("confidence_head.proj.weight", true)?;
        let confidence_bias = take("confidence_head.proj.bias", false)?;
        let mut layers = Vec::with_capacity(LAYERS);
        for i in 0..LAYERS {
            let mut w = |name: &str, linear| take(&format!("layers.{i}.{name}.weight"), linear);
            layers.push(DraftLayer {
                input_norm: w("input_layernorm", false)?,
                post_norm: w("post_attention_layernorm", false)?,
                q_norm: w("self_attn.q_norm", false)?,
                k_norm: w("self_attn.k_norm", false)?,
                q: w("self_attn.q_proj", true)?,
                k: w("self_attn.k_proj", true)?,
                v: w("self_attn.v_proj", true)?,
                o: w("self_attn.o_proj", true)?,
                gate: w("mlp.gate_proj", true)?,
                up: w("mlp.up_proj", true)?,
                down: w("mlp.down_proj", true)?,
            });
        }
        if !arrays.is_empty() {
            return Err(Error::Contract("unconsumed DSpark tensors"));
        }
        Ok(Self {
            fc,
            hidden_norm,
            norm,
            markov_embedding,
            markov_head,
            _confidence_weight: confidence_weight,
            _confidence_bias: confidence_bias,
            layers,
        })
    }
}

/// Only generated token decisions cross the device boundary. Hidden taps,
/// projection logits, norms and all target/draft KV remain device-resident.
fn token_decisions(array: &Array, expected: usize) -> Result<Vec<u32>> {
    if array.numel() != expected || expected > BLOCK + 1 {
        return Err(Error::Contract(
            "DSpark only reads a bounded token decision vector",
        ));
    }
    let bytes = array.cast(MlxDType::U32)?.to_bytes()?;
    let tokens = bytes
        .chunks_exact(4)
        .map(|b| u32::from_ne_bytes([b[0], b[1], b[2], b[3]]))
        .collect::<Vec<_>>();
    if tokens.iter().any(|&t| t as usize >= VOCAB) {
        return Err(Error::Contract("DSpark decision outside target vocabulary"));
    }
    Ok(tokens)
}

/// One device expression for the complete drafter and sequential Markov head.
/// args = [pending target taps, seven block ids, draft offset, K0,V0,...,K4,V4].
/// K/V outputs still contain seven tentative block positions. The state owner
/// commits only the newly supplied target context by advancing its offset by n.
fn draft_step(
    weights: &DraftWeights,
    target: &Weights,
    silu: &Compiled,
    positions: &Array,
    args: &[Array],
) -> Result<Vec<Array>> {
    if args.len() != 3 + 2 * LAYERS {
        return Err(Error::Contract("DSpark draft IO count"));
    }
    let n = args[0].shape()[1];
    let stream = args[0].stream();
    let context = args[0]
        .matmul(&weights.fc)?
        .fast_rms_norm(&weights.hidden_norm, EPSILON)?;
    let mut h = target.embedding.take(&args[1], 0)?;
    let offset = &args[2];
    let block_offset = offset.add(&Array::from_i32(stream, &[1], &[n as i32])?)?;
    let last_valid = block_offset.add(&Array::from_i32(stream, &[1], &[(BLOCK - 1) as i32])?)?;
    let mask = positions
        .less_equal(&last_valid)?
        .reshape(&[1, 1, 1, positions.numel()])?;
    let mut states = Vec::with_capacity(2 * LAYERS);
    for (i, w) in weights.layers.iter().enumerate() {
        // These are stock MLX nn.RMSNorm/RoPE semantics. MiniCPM target's
        // cast-before-affine norm and precomputed BF16 RoPE are different.
        let normalized = h.fast_rms_norm(&w.input_norm, EPSILON)?;
        let q = normalized
            .matmul(&w.q)?
            .reshape(&[1, BLOCK, 16, 128])?
            .fast_rms_norm(&w.q_norm, EPSILON)?
            .transpose(&[0, 2, 1, 3])?
            .rope(&block_offset, 128, false, 5_000_000.0, 1.0)?;
        let kv = |x: &Array, rows: usize, pos: &Array| -> Result<(Array, Array)> {
            let k = x
                .matmul(&w.k)?
                .reshape(&[1, rows, 2, 128])?
                .fast_rms_norm(&w.k_norm, EPSILON)?
                .transpose(&[0, 2, 1, 3])?
                .rope(pos, 128, false, 5_000_000.0, 1.0)?;
            let v = x
                .matmul(&w.v)?
                .reshape(&[1, rows, 2, 128])?
                .transpose(&[0, 2, 1, 3])?;
            Ok((k, v))
        };
        let (kc, vc) = kv(&context, n, offset)?;
        let (kb, vb) = kv(&normalized, BLOCK, &block_offset)?;
        let k = args[3 + 2 * i].slice_update(&Array::concat(&[&kc, &kb], 2)?, offset, &[2])?;
        let v = args[4 + 2 * i].slice_update(&Array::concat(&[&vc, &vb], 2)?, offset, &[2])?;
        // The block is bidirectional; only unused fixed-capacity keys are masked.
        let attention = q
            .sdpa(&k, &v, (128.0_f32).powf(-0.5), false, Some(&mask))?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, BLOCK, 2048])?
            .matmul(&w.o)?;
        h = h.add(&attention)?;
        let normalized = h.fast_rms_norm(&w.post_norm, EPSILON)?;
        let gate = normalized.matmul(&w.gate)?;
        let up = normalized.matmul(&w.up)?;
        // Public mlx.nn.silu is locally compiled even in the eager reference.
        // This keeps its fused activation result before the eager up-product;
        // under whole-step tracing MLX expands the nested callable itself.
        let activated = silu.call(&[gate])?.remove(0);
        h = h.add(&activated.mul(&up)?.matmul(&w.down)?)?;
        states.extend([k, v]);
    }
    let logits = h
        .fast_rms_norm(&weights.norm, EPSILON)?
        .matmul(&target.head)?;
    let mut previous = args[1].slice_axis(1, 0, 1)?.reshape(&[1])?;
    let mut proposals = Vec::with_capacity(BLOCK);
    for i in 0..BLOCK {
        let bias = weights
            .markov_embedding
            .take(&previous, 0)?
            .matmul(&weights.markov_head)?;
        previous = logits
            .slice_axis(1, i, i + 1)?
            .reshape(&[1, VOCAB])?
            .add(&bias)?
            .argmax(-1)?;
        proposals.push(previous.clone());
    }
    let mut outputs = vec![Array::concat(&proposals.iter().collect::<Vec<_>>(), 0)?];
    outputs.extend(states);
    Ok(outputs)
}

struct BlockCommit {
    accepted: usize,
    tokens: Vec<u32>,
}

fn select_block(
    proposals: &[u32],
    predictions: &[u32],
    remaining: usize,
    eos: &[u32],
) -> Result<BlockCommit> {
    if proposals.len() != BLOCK
        || predictions.is_empty()
        || predictions.len() > BLOCK + 1
        || remaining == 0
    {
        return Err(Error::Contract("invalid DSpark verification block"));
    }
    let accepted = proposals
        .iter()
        .zip(predictions)
        .take(predictions.len() - 1)
        .take_while(|(proposal, prediction)| proposal == prediction)
        .count();
    let mut tokens = proposals[..accepted].to_vec();
    tokens.push(predictions[accepted]);
    tokens.truncate(remaining);
    if let Some(index) = tokens.iter().position(|token| eos.contains(token)) {
        tokens.truncate(index + 1);
    }
    Ok(BlockCommit { accepted, tokens })
}

#[derive(Clone, Debug, Default)]
pub struct DSparkStats {
    pub proposal_calls: usize,
    pub proposed_tokens: usize,
    pub verify_calls: usize,
    pub accepted_proposals: usize,
    pub emitted_tokens: usize,
    pub target_consumed: usize,
    pub draft_consumed: usize,
    pub pending_context_tokens: usize,
}

/// Explicitly selected official BF16 DSpark. No lookup proposals, confidence
/// parking, quantized draft, sampling approximation or no-draft fallback.
pub struct DSpark {
    target: MiniCpm5,
    weights: Rc<DraftWeights>,
    silu: Rc<Compiled>,
    draft_kv: Vec<Array>,
    draft_offset: usize,
    pending_context: Option<Array>,
    positions: Option<Array>,
    compiled: Option<Compiled>,
    prepared: Option<(usize, usize)>,
    use_compiled: bool,
    failure: Option<String>,
    pub last_stats: DSparkStats,
}

impl DSpark {
    pub fn load(target: MiniCpm5, snapshot: &Path) -> Result<Self> {
        if !target.supports_dspark() {
            return Err(Error::Contract(
                "DSpark requires a target prepared for hidden taps and causal verification",
            ));
        }
        let weights = Rc::new(DraftWeights::load(target.backend.stream(), snapshot)?);
        let silu = Rc::new(Compiled::with_shapeless(
            target.backend.stream(),
            1,
            true,
            |a| Ok(vec![a[0].mul(&a[0].sigmoid()?)?]),
        )?);
        Ok(Self {
            target,
            weights,
            silu,
            draft_kv: Vec::new(),
            draft_offset: 0,
            pending_context: None,
            positions: None,
            compiled: None,
            prepared: None,
            use_compiled: true,
            failure: None,
            last_stats: DSparkStats::default(),
        })
    }

    /// Diagnostic eager execution uses the same explicit-state mathematics.
    /// Compiled execution is the maintained default; changes require a reset.
    pub fn set_draft_compiled(&mut self, enabled: bool) -> Result<()> {
        if self.target.offset != 0 || self.draft_offset != 0 {
            return Err(Error::Contract(
                "reset DSpark before changing execution mode",
            ));
        }
        self.use_compiled = enabled;
        Ok(())
    }

    pub fn prepare(&mut self, prompt: usize, output: usize) -> Result<()> {
        match self.prepare_inner(prompt, output) {
            Ok(()) => Ok(()),
            Err(error) => {
                self.prepared = None;
                self.failure = Some(error.to_string());
                Err(error)
            }
        }
    }

    fn prepare_inner(&mut self, prompt: usize, output: usize) -> Result<()> {
        let total = prompt
            .checked_add(output)
            .ok_or(Error::Contract("DSpark context overflow"))?;
        if prompt == 0 || output == 0 || output > 256 || total > CONTEXT_CAPACITY {
            return Err(Error::Contract(
                "DSpark supports B1, 1..256 output tokens and at most 4096 positions",
            ));
        }
        if self.target.offset != 0 || self.draft_offset != 0 {
            return Err(Error::Contract("reset DSpark before preparing a request"));
        }
        if let Some(error) = &self.failure {
            return Err(Error::Other(error.clone()));
        }
        // A verification block can consume seven more rows than its output
        // budget. These speculative rows must fit even at a capacity boundary.
        self.target
            .prepare(prompt, (output + BLOCK).min(CONTEXT_CAPACITY - prompt))?;
        let capacity = (total + BLOCK).div_ceil(256) * 256;
        if self.prepared == Some((prompt, capacity)) {
            return Ok(());
        }
        let stream = self.target.backend.stream();
        self.silu
            .prepare(&[Array::zeros(stream, &[1, BLOCK, 6144], MlxDType::BF16)?])?;
        let states = (0..2 * LAYERS)
            .map(|_| Array::zeros(stream, &[1, 2, capacity, 128], MlxDType::BF16))
            .collect::<Result<Vec<_>>>()?;
        stream.eval(&states)?;
        let positions = Array::arange(stream, 0.0, capacity as f32, 1.0, MlxDType::I32)?;
        let weights = self.weights.clone();
        let target = self.target.weights.clone();
        let silu = self.silu.clone();
        let grid = positions.clone();
        let compiled = Compiled::new(stream, 1 + 2 * LAYERS, move |args| {
            draft_step(&weights, &target, &silu, &grid, args)
        })?;
        let mut admitted = (1..=BLOCK + 1).collect::<Vec<_>>();
        if !admitted.contains(&prompt) {
            admitted.push(prompt);
        }
        for rows in admitted {
            let mut ids = vec![MASK_TOKEN; BLOCK];
            ids[0] = 0;
            let mut args = vec![
                Array::zeros(stream, &[1, rows, 10240], MlxDType::BF16)?,
                Array::from_i32(stream, &[1, BLOCK], &ids)?,
                Array::from_i32(stream, &[1], &[0])?,
            ];
            args.extend(states.iter().cloned());
            compiled.prepare(&args)?;
        }
        self.draft_kv = states;
        self.positions = Some(positions);
        self.compiled = Some(compiled);
        self.prepared = Some((prompt, capacity));
        Ok(())
    }

    fn propose(&mut self, pending: u32) -> Result<Vec<u32>> {
        let context = self
            .pending_context
            .as_ref()
            .ok_or(Error::Contract("DSpark missing pending target taps"))?;
        let (prompt, capacity) = self
            .prepared
            .ok_or(Error::Contract("prepare DSpark before proposing"))?;
        let shape = context.shape();
        if shape.len() != 3
            || shape[0] != 1
            || shape[2] != 10240
            || context.dtype() != MlxDType::BF16
        {
            return Err(Error::Contract(
                "DSpark requires BF16 target taps [1,n,10240]",
            ));
        }
        let rows = shape[1];
        if rows == 0
            || (rows != prompt && rows > BLOCK + 1)
            || (self.draft_offset == 0 && rows != prompt)
            || (self.draft_offset > 0 && rows > BLOCK + 1)
            || self.draft_offset + rows + BLOCK > capacity
            || pending as usize >= VOCAB
        {
            return Err(Error::Contract(
                "DSpark pending context is outside its prepared profile",
            ));
        }
        let mut ids = vec![MASK_TOKEN; BLOCK];
        ids[0] = pending as i32;
        let stream = self.target.backend.stream();
        let mut args = vec![
            context.clone(),
            Array::from_i32(stream, &[1, BLOCK], &ids)?,
            Array::from_i32(stream, &[1], &[self.draft_offset as i32])?,
        ];
        args.extend(self.draft_kv.iter().cloned());
        let mut outputs = if self.use_compiled {
            self.compiled
                .as_ref()
                .ok_or(Error::Contract("DSpark compiled step missing"))?
                .call_and_eval(&args)?
        } else {
            let outputs = draft_step(
                &self.weights,
                &self.target.weights,
                &self.silu,
                self.positions
                    .as_ref()
                    .ok_or(Error::Contract("DSpark position grid missing"))?,
                &args,
            )?;
            stream.eval(&outputs)?;
            outputs
        };
        let proposals = token_decisions(&outputs.remove(0), BLOCK)?;
        // Functional array updates become live only after every output has
        // evaluated. The next call overwrites this round's seven draft slots.
        self.draft_kv = outputs;
        self.draft_offset += rows;
        Ok(proposals)
    }

    fn generate_inner(
        &mut self,
        request: GenerationRequest<'_>,
        on_token: &mut dyn FnMut(GeneratedToken),
    ) -> Result<GenerationOutput> {
        if let Some(error) = &self.failure {
            return Err(Error::Other(error.clone()));
        }
        if request.input.image.is_some() {
            return Err(Error::Contract("DSpark is text-only"));
        }
        let options = request.options.resolve()?;
        let ids = request.input.token_ids;
        let penalties = &options.sampling.penalties;
        if !matches!(options.sampling.selection, TokenSelection::Greedy)
            || penalties.repetition != 1.0
            || penalties.frequency != 0.0
            || penalties.presence != 0.0
            || options.sampling.return_logprob
        {
            return Err(Error::Contract(
                "DSpark supports greedy generation with neutral penalties and no logprob",
            ));
        }
        if ids.is_empty()
            || ids
                .iter()
                .chain(options.eos_token_ids.iter())
                .any(|&id| id as usize >= VOCAB)
            || options.max_new_tokens > 256
            || ids
                .len()
                .checked_add(options.max_new_tokens)
                .is_none_or(|n| n > CONTEXT_CAPACITY)
        {
            return Err(Error::Contract(
                "unsupported DSpark B1 token/context/output bounds",
            ));
        }
        self.reset();
        if let Some(error) = &self.failure {
            return Err(Error::Other(error.clone()));
        }
        if options.max_new_tokens == 0 {
            let mut profile = GenerationProfile::new();
            profile.finalize(ids.len(), 0);
            return Ok(GenerationOutput {
                tokens: vec![],
                profile,
            });
        }
        self.prepare(ids.len(), options.max_new_tokens)?;
        // Preparation is intentionally outside inference TTFT/TPOT.
        let mut profile = GenerationProfile::new();
        let first = self
            .target
            .run_tokens(ids, 0, true)?
            .argmax(-1)?
            .to_u32_scalar()?;
        self.pending_context = Some(
            self.target
                .last_taps
                .as_ref()
                .ok_or(Error::Contract(
                    "DSpark target must capture five hidden-layer taps",
                ))?
                .clone(),
        );
        let first = GeneratedToken {
            token_id: first,
            logprob: None,
        };
        let mut output = vec![first];
        profile.record_first_token();
        on_token(first);
        while output.len() < options.max_new_tokens
            && !options
                .eos_token_ids
                .contains(&output.last().unwrap().token_id)
        {
            let pending = output.last().unwrap().token_id;
            let proposals = self.propose(pending)?;
            self.last_stats.proposal_calls += 1;
            self.last_stats.proposed_tokens += BLOCK;
            let before = self.target.offset;
            let width = (BLOCK + 1).min(CONTEXT_CAPACITY - before);
            if width == 0 {
                return Err(Error::Contract("DSpark target context exhausted"));
            }
            let mut verify = vec![pending];
            verify.extend_from_slice(&proposals[..width - 1]);
            let predictions = token_decisions(
                &self.target.run_tokens(&verify, before, false)?.argmax(-1)?,
                width,
            )?;
            self.last_stats.verify_calls += 1;
            let commit = select_block(
                &proposals,
                &predictions,
                options.max_new_tokens - output.len(),
                &options.eos_token_ids,
            )?;
            self.last_stats.accepted_proposals += commit.accepted;
            // Existing pending plus all new outputs except the final pending
            // consume exactly this many verification input rows, including
            // when EOS or output budget truncates a fully accepted block.
            let consumed = commit.tokens.len();
            let taps = self
                .target
                .last_taps
                .as_ref()
                .ok_or(Error::Contract("DSpark verification taps missing"))?
                .slice_axis(1, 0, consumed)?;
            self.target.trim_to(before + consumed)?;
            self.pending_context = Some(taps);
            for token_id in commit.tokens {
                let token = GeneratedToken {
                    token_id,
                    logprob: None,
                };
                output.push(token);
                on_token(token);
            }
        }
        self.last_stats.emitted_tokens = output.len();
        self.last_stats.target_consumed = self.target.offset;
        self.last_stats.draft_consumed = self.draft_offset;
        self.last_stats.pending_context_tokens =
            self.pending_context.as_ref().map_or(0, |a| a.shape()[1]);
        profile.finalize(ids.len(), output.len());
        Ok(GenerationOutput {
            tokens: output,
            profile,
        })
    }

    fn generate(
        &mut self,
        request: GenerationRequest<'_>,
        on_token: &mut dyn FnMut(GeneratedToken),
    ) -> Result<GenerationOutput> {
        match self.generate_inner(request, on_token) {
            Ok(output) => Ok(output),
            Err(error) => {
                self.failure = Some(error.to_string());
                Err(error)
            }
        }
    }
}

impl LlmTrait for DSpark {
    fn load(_: ModelConfig, _: HashMap<String, Tensor>, _: Device) -> Result<Self> {
        Err(Error::Contract(
            "load DSpark with AutoModel and the official draft snapshot asset",
        ))
    }
    fn forward(&mut self, ids: &[u32], start_pos: u32) -> Result<Tensor> {
        if let Some(error) = &self.failure {
            return Err(Error::Other(error.clone()));
        }
        // Teacher forcing remains target-only, with the public full-logit shape.
        self.pending_context = None;
        self.draft_offset = 0;
        self.target.forward(ids, start_pos)
    }
    fn backend(&self) -> &dyn Backend {
        &self.target.backend
    }
    fn preparation_status(&self) -> TextPreparationStatus {
        let target = self.target.preparation_status();
        let error = self.failure.clone().or(target.error.clone());
        let ready = error.is_none()
            && target.state == TextPreparationState::Ready
            && self
                .prepared
                .is_some_and(|(prompt, _)| target.prepared_prompt_tokens == Some(prompt))
            && self.draft_kv.len() == 2 * LAYERS
            && self.positions.is_some()
            && self.compiled.is_some();
        let mut status = TextPreparationStatus {
            state: if error.is_some() || target.state == TextPreparationState::Invalidated {
                TextPreparationState::Invalidated
            } else if ready {
                TextPreparationState::Ready
            } else {
                TextPreparationState::Unprepared
            },
            implementation: "minicpm5_mlx_dspark",
            variant: Some(if self.use_compiled {
                "dspark"
            } else {
                "dspark-eager-draft"
            }),
            error,
            ..TextPreparationStatus::default()
        };
        if ready {
            let (prompt, draft_capacity) = self.prepared.unwrap();
            status.compiled_scopes = target.compiled_scopes;
            if self.use_compiled {
                status
                    .compiled_scopes
                    .push(TextCompilationScope::DraftProposal);
            }
            status.prepared_prompt_tokens = Some(prompt);
            status.prepared_sequence_lengths = target.prepared_sequence_lengths;
            // The target's usable capacity bounds the combined schedule; the
            // draft additionally reserves seven tentative proposal positions.
            status.kv_capacity = target.kv_capacity.map(|n| n.min(draft_capacity));
            status.max_decode_rows = target.max_decode_rows.map(|n| n.min(BLOCK + 1));
        }
        status
    }
    fn reset(&mut self) {
        self.target.reset();
        self.draft_offset = 0;
        self.pending_context = None;
        self.last_stats = DSparkStats::default();
        self.failure = self
            .target
            .backend
            .stream()
            .synchronize()
            .err()
            .map(|e| e.to_string());
    }
    fn prewarm_decode(&mut self, prompt: usize, output: usize) {
        if let Err(error) = self.prepare(prompt, output) {
            self.failure = Some(error.to_string());
        }
    }
    fn vocab_size(&self) -> usize {
        VOCAB
    }
    fn prefill(&mut self, input: LlmInput<'_>) -> Result<Tensor> {
        if input.image.is_some() {
            return Err(Error::Contract("DSpark is text-only"));
        }
        self.forward(input.token_ids, 0)
    }
    fn generate_streaming_with_options_dyn(
        &mut self,
        request: GenerationRequest<'_>,
        on_token: &mut dyn FnMut(GeneratedToken),
    ) -> Result<GenerationOutput> {
        self.generate(request, on_token)
    }
    fn generate_streaming_with_options(
        &mut self,
        request: GenerationRequest<'_>,
        mut on_token: impl FnMut(GeneratedToken),
    ) -> Result<GenerationOutput> {
        self.generate(request, &mut on_token)
    }
    fn generate_streaming_dyn(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        on_token: &mut dyn FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, GenerationProfile)> {
        let options = crate::GenerationOptions::greedy(max_new_tokens, eos_token_id);
        let output = self.generate(
            GenerationRequest {
                input,
                options: &options,
            },
            &mut |token| on_token(token.token_id),
        )?;
        Ok((output.token_ids(), output.profile))
    }
    fn generate_streaming(
        &mut self,
        input: LlmInput<'_>,
        max_new_tokens: usize,
        mut on_token: impl FnMut(u32),
        eos_token_id: Option<u32>,
    ) -> Result<(Vec<u32>, GenerationProfile)> {
        self.generate_streaming_dyn(input, max_new_tokens, &mut on_token, eos_token_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(serde::Deserialize)]
    struct ReplayInputs {
        max_new_tokens: usize,
        eos_token_ids: Vec<u32>,
        cases: Vec<ReplayCase>,
    }

    #[derive(serde::Deserialize)]
    struct ReplayCase {
        id: String,
        prompt_token_ids: Vec<u32>,
        max_new_tokens: Option<usize>,
    }

    /// Run this ignored test under the host's shared Metal measurement lock.
    /// All checkpoint paths, real prompts and stopping settings are external.
    /// APXINF_DSPARK_CASE_LIMIT defaults to two and must admit two distinct
    /// prompts; increasing it exercises the remainder of the frozen suite.
    #[test]
    #[ignore = "requires official target/draft checkpoints, real replay inputs and the shared Metal lock"]
    fn real_checkpoint_draft_compilation_and_reset() -> Result<()> {
        use super::super::config::{Config, Variant};
        use crate::GenerationOptions;
        use std::{env, path::PathBuf};

        let path = |name| {
            env::var_os(name)
                .map(PathBuf::from)
                .ok_or_else(|| Error::Other(format!("set {name} for this ignored test")))
        };
        let checkpoint = path("APXINF_MINICPM_CHECKPOINT")?;
        let draft = path("APXINF_DSPARK_CHECKPOINT")?;
        let replay = path("APXINF_DSPARK_REPLAY_INPUT")?;
        let inputs: ReplayInputs = serde_json::from_slice(&std::fs::read(&replay)?)
            .map_err(|e| Error::Other(format!("DSpark real replay inputs: {e}")))?;
        let limit = env::var("APXINF_DSPARK_CASE_LIMIT")
            .ok()
            .map(|value| value.parse::<usize>())
            .transpose()
            .map_err(|e| Error::Other(format!("DSpark case limit: {e}")))?
            .unwrap_or(2);
        if limit < 2 || limit > inputs.cases.len() {
            return Err(Error::Contract("DSpark test needs at least two real cases"));
        }
        let cases = &inputs.cases[..limit];
        if cases[0].prompt_token_ids == cases[1].prompt_token_ids {
            return Err(Error::Contract(
                "DSpark test needs two different real prompts",
            ));
        }
        let config = Config::from_json(&std::fs::read_to_string(checkpoint.join("config.json"))?)?;
        let (map, _) =
            apxinf_loader::safetensors::load_native_path(&checkpoint).map_err(Error::Other)?;
        let backend = apxinf_mlx::MlxBackend::new(0)?;
        let weights = Weights::load(&config, backend.stream(), map)?;
        let target = MiniCpm5::new(config, weights, backend, Variant::DSpark)?;
        let mut model = DSpark::load(target, &draft)?;
        println!(
            "DSPARK_TEST_INPUT {}",
            serde_json::json!({
                "replay_sha256": sha256(&replay)?,
                "target_config_sha256": sha256(&checkpoint.join("config.json"))?,
                "draft_config_sha256": CONFIG_SHA256,
                "draft_weights_sha256": WEIGHTS_SHA256,
                "case_limit": limit,
            })
        );
        // Repeated A, changed B, then A again exercise stale token, hidden-tap
        // and KV capture. Both modes share this same model/compiled instance.
        let order = [0, 0, 1, 0].into_iter().chain(2..limit).collect::<Vec<_>>();
        let mut eager = Vec::new();
        for compiled in [false, true] {
            let mut records = Vec::new();
            for (round, &index) in order.iter().enumerate() {
                let case = &cases[index];
                let budget = case.max_new_tokens.unwrap_or(inputs.max_new_tokens);
                if budget < 2 {
                    return Err(Error::Contract("DSpark test must exercise draft proposals"));
                }
                model.reset();
                assert_eq!(model.target.offset, 0);
                assert_eq!(model.draft_offset, 0);
                assert!(model.pending_context.is_none());
                model.set_draft_compiled(compiled)?;
                model.prepare(case.prompt_token_ids.len(), budget)?;
                let preparation = model.preparation_status();
                assert_eq!(preparation.state, TextPreparationState::Ready);
                assert_eq!(
                    preparation
                        .compiled_scopes
                        .contains(&TextCompilationScope::DraftProposal),
                    compiled,
                );
                let mut options = GenerationOptions::greedy(budget, None);
                options.eos_token_ids = Some(inputs.eos_token_ids.clone());
                let mut streamed = Vec::new();
                let output = model.generate_streaming_with_options_dyn(
                    GenerationRequest {
                        input: LlmInput::text(&case.prompt_token_ids),
                        options: &options,
                    },
                    &mut |token| streamed.push(token.token_id),
                )?;
                let ids = output.token_ids();
                let finish = if ids
                    .last()
                    .is_some_and(|id| inputs.eos_token_ids.contains(id))
                {
                    "eos"
                } else if ids.len() == budget {
                    "length"
                } else {
                    "unexpected"
                };
                let stats = &model.last_stats;
                println!(
                    "DSPARK_TEST_OUTPUT {}",
                    serde_json::json!({
                        "compiled": compiled, "round": round, "id": case.id,
                        "prompt_tokens": case.prompt_token_ids.len(), "max_new_tokens": budget,
                        "token_ids": ids, "finish_reason": finish,
                        "preparation": preparation,
                        "stats": {
                            "proposal_calls": stats.proposal_calls,
                            "proposed_tokens": stats.proposed_tokens,
                            "verify_calls": stats.verify_calls,
                            "accepted_proposals": stats.accepted_proposals,
                            "emitted_tokens": stats.emitted_tokens,
                            "target_consumed": stats.target_consumed,
                            "draft_consumed": stats.draft_consumed,
                            "pending_context_tokens": stats.pending_context_tokens,
                        },
                    })
                );
                assert_eq!(streamed, ids, "callback mismatch: {}", case.id);
                assert_ne!(finish, "unexpected", "incomplete generation: {}", case.id);
                assert!(stats.proposal_calls > 0, "draft never ran: {}", case.id);
                assert_eq!(stats.proposed_tokens, stats.proposal_calls * BLOCK);
                assert_eq!(stats.verify_calls, stats.proposal_calls);
                assert!(stats.accepted_proposals <= stats.proposed_tokens);
                assert_eq!(stats.emitted_tokens, ids.len());
                assert_eq!(
                    stats.target_consumed,
                    case.prompt_token_ids.len() + ids.len() - 1
                );
                assert_eq!(
                    stats.draft_consumed + stats.pending_context_tokens,
                    stats.target_consumed
                );
                records.push((ids, finish));
            }
            assert_eq!(records[0], records[1], "same prompt after reset changed");
            assert_eq!(
                records[0], records[3],
                "changed prompt polluted later request"
            );
            if compiled {
                assert_eq!(
                    records, eager,
                    "draft eager/compiled outputs or stopping differ"
                );
            } else {
                eager = records;
            }
        }
        Ok(())
    }

    #[test]
    fn official_inventory_is_complete() {
        let shapes = tensor_shapes();
        assert_eq!(shapes.len(), 62);
        assert_eq!(
            shapes
                .iter()
                .map(|(_, shape)| shape.iter().product::<usize>())
                .sum::<usize>(),
            323_776_001
        );
        let keys = shapes
            .iter()
            .map(|(key, _)| key)
            .collect::<std::collections::HashSet<_>>();
        assert_eq!(keys.len(), 62);
    }
    #[test]
    fn rejection_is_contiguous_and_keeps_replacement() {
        let block = select_block(
            &[10, 11, 12, 13, 14, 15, 16],
            &[10, 99, 12, 13, 14, 15, 16, 17],
            32,
            &[],
        )
        .unwrap();
        assert_eq!(block.accepted, 1);
        assert_eq!(block.tokens, [10, 99]);
        let block = select_block(&[10, 11, 12, 13, 14, 15, 16], &[99], 32, &[]).unwrap();
        assert_eq!(block.accepted, 0);
        assert_eq!(block.tokens, [99]);
    }
    #[test]
    fn all_accepted_adds_bonus_but_eos_and_budget_bound_commit() {
        let proposed = [10, 11, 12, 13, 14, 15, 16];
        let predicted = [10, 11, 12, 13, 14, 15, 16, 17];
        let block = select_block(&proposed, &predicted, 32, &[]).unwrap();
        assert_eq!(block.accepted, 7);
        assert_eq!(block.tokens, predicted);
        assert_eq!(
            select_block(&proposed, &predicted, 3, &[]).unwrap().tokens,
            [10, 11, 12]
        );
        assert_eq!(
            select_block(&proposed, &predicted, 32, &[11])
                .unwrap()
                .tokens,
            [10, 11]
        );
        assert!(select_block(&proposed, &[], 1, &[]).is_err());
        assert!(select_block(&proposed, &predicted, 0, &[]).is_err());
    }
}

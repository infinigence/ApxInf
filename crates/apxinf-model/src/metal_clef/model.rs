//! Clef Flash forward pass: Qwen3.5 hybrid backbone in the stock or the C14
//! composition, and the BF16 joint schema head.
//!
//! The operator sequence and every rounding point follow the stock reference
//! (MLX-VLM 0.7.4 backbone, clef_mlx.py head, MLX 0.32.3), including its
//! compiled functions. The gated-delta recurrence is a Metal kernel with the
//! MLX-VLM kernel's arithmetic. C14 (EngineTailor layer-stack frontier)
//! replaces the MLP, the gated-delta projections, convolution and recurrence,
//! and the attention projections with its own kernels. One request is one lazy
//! graph evaluated once; no state survives between requests.
use super::config::Config;
use super::encode::{Encoded, EncodedQuestion};
use super::weights::{
    Attention, Backbone, DecoderLayer, EvidenceLayer, GatedDelta, Head, Linear, Mha, Mixer, Mlp,
    Norm, Q4,
};
use apxinf_core::Result;
use apxinf_mlx::fusions::{
    padded_rows, CausalDepthwiseConv, F16GateSwiglu, F16Gemm, GatedDeltaLayout,
    GatedDeltaRecurrence, HalfSplitRope, PadRowsF16, Q4DecodeArithmetic, Q4RowDecodeF16,
    Q4TransposeDecode, SplitF16Gemm,
};
use apxinf_mlx::{Array, Compiled, MlxDType, Stream};

const EPS: f32 = 1e-6;
const HEAD_EPS: f32 = 1e-5;

struct Kernels {
    recurrence: GatedDeltaRecurrence,
    rope: HalfSplitRope,
    /// MLX-VLM `_compute_g_beta`, compiled shapeless as there: the fused
    /// kernel rounds exp/logaddexp differently from eager operations on some
    /// devices (one decay ulp per layer on the M1 Pro).
    g_beta: Compiled,
}

/// Backbone composition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Composition {
    /// MLX-VLM: Q4 matmuls, MLX convolution, the four-value recurrence.
    Reference,
    /// EngineTailor C14: gate GEMM with SwiGLU epilogue, wm1/wn2 split
    /// projections for the gated delta and attention q/o, one-pass k/v, the
    /// causal convolution kernel and the eight-lane recurrence. Its GDN
    /// reduction order differs from MLX-VLM's.
    C14,
}

/// Kernels only the C14 composition uses (EngineTailor `c14.py` names).
struct C14Kernels {
    pad: PadRowsF16,
    /// `g18`: projection weights decoded in BF16, transposed to FP16.
    project_decode: Q4TransposeDecode,
    /// `r06`: MLP down weight decoded in FP16, transposed.
    down_decode: Q4TransposeDecode,
    /// `s22`: MLP gate and up weights decoded row-major.
    row_decode: Q4RowDecodeF16,
    /// `a13`: gate GEMM with the SwiGLU epilogue.
    gate_swiglu: F16GateSwiglu,
    /// `s28`: split projections (gated delta, attention q and o).
    split: SplitF16Gemm,
    /// `s13`: attention k and v in one pass.
    single: F16Gemm,
    /// `s31`: gated-delta convolution.
    conv: CausalDepthwiseConv,
    /// `s12`: eight value rows per SIMD group.
    recurrence: GatedDeltaRecurrence,
}

impl C14Kernels {
    fn new(stream: &Stream) -> Result<Self> {
        Ok(Self {
            pad: PadRowsF16::new(stream)?,
            project_decode: Q4TransposeDecode::new(stream, Q4DecodeArithmetic::Bf16)?,
            down_decode: Q4TransposeDecode::new(stream, Q4DecodeArithmetic::F16)?,
            row_decode: Q4RowDecodeF16::new(stream)?,
            gate_swiglu: F16GateSwiglu::new(stream)?,
            split: SplitF16Gemm::new(stream)?,
            single: F16Gemm::new(stream)?,
            conv: CausalDepthwiseConv::new(stream)?,
            recurrence: GatedDeltaRecurrence::with_layout(stream, GatedDeltaLayout::EightLanes)?,
        })
    }

    /// `_g20_linear` with the wm1/wn2 GEMM: pad to 64 rows, decode each half
    /// of the Q4 weight, two GEMMs writing one BF16 output.
    fn split_linear(&self, x: &Array, q: &Q4) -> Result<Array> {
        let tokens = x.shape()[1];
        let half = q.rows() / 2;
        let h = self.pad.call(x)?;
        let low = self.split.low(
            &h,
            tokens,
            &self
                .project_decode
                .call(&q.weight, &q.scales, &q.biases, 0, half)?,
        )?;
        // Built after the low half, listed first, as in C14's dispatch order.
        let high = self
            .project_decode
            .call(&q.weight, &q.scales, &q.biases, half, half)?;
        self.split.high(&high, &low, &h, tokens)
    }

    /// `_single_linear`: one full-width decode and one GEMM.
    fn single_linear(&self, x: &Array, q: &Q4) -> Result<Array> {
        let weight = self
            .project_decode
            .call(&q.weight, &q.scales, &q.biases, 0, q.rows())?;
        self.single.call(&self.pad.call(x)?, x.shape()[1], &weight)
    }

    /// `create_mlp`: the row-decoded up weight through an MLX GEMM, the gate
    /// GEMM with its SwiGLU epilogue, and the transposed down decode.
    fn mlp(&self, h: &Array, m: &Mlp) -> Result<Array> {
        let tokens = h.shape()[1];
        let x = if padded_rows(tokens) == tokens {
            h.cast(MlxDType::F16)?
        } else {
            self.pad.call(h)?
        };
        let decode = |q: &Q4| self.row_decode.call(&q.weight, &q.scales, &q.biases);
        let up = x.matmul(&decode(&m.up)?.transpose(&[1, 0])?)?;
        let activation = self.gate_swiglu.call(&x, &decode(&m.gate)?, &up)?;
        let down = self.down_decode.call(
            &m.down.weight,
            &m.down.scales,
            &m.down.biases,
            0,
            m.down.rows(),
        )?;
        activation
            .matmul(&down)?
            .slice_axis(1, 0, tokens)?
            .cast(MlxDType::BF16)
    }
}

/// `compute_g`: `exp(-exp(A_log) * softplus(a + dt_bias))`; beta: `sigmoid(b)`.
fn g_beta(args: &[Array]) -> Result<Vec<Array>> {
    let [a_log, a, b, dt_bias] = args else {
        unreachable!("g_beta takes four inputs")
    };
    let shifted = a.add(dt_bias)?;
    let softplus = shifted.logaddexp(&scalar(&shifted, 0.0)?)?;
    let decay = a_log
        .cast(MlxDType::F32)?
        .exp()?
        .neg()?
        .mul(&softplus)?
        .exp()?;
    Ok(vec![decay, b.sigmoid()?])
}

pub struct Network {
    config: Config,
    stream: Stream,
    backbone: Backbone,
    head: Head,
    kernels: Kernels,
    c14: Option<C14Kernels>,
    /// MRoPE inverse frequencies, computed on the device as MLX-VLM does.
    inv_freq: Array,
}

/// Python scalars combine with arrays in the array's dtype.
fn scalar(like: &Array, value: f64) -> Result<Array> {
    Array::scalar(like.stream(), value as f32, like.dtype())
}

fn silu(x: &Array) -> Result<Array> {
    x.mul(&x.sigmoid()?)
}

/// `nn.gelu`: `x * (1 + erf(x / sqrt(2))) / 2`.
fn gelu(x: &Array) -> Result<Array> {
    let inner = x.div(&scalar(x, std::f64::consts::SQRT_2)?)?.erf()?;
    x.mul(&scalar(x, 1.0)?.add(&inner)?)?.div(&scalar(x, 2.0)?)
}

fn qmm(x: &Array, q: &Q4) -> Result<Array> {
    x.quantized_matmul(&q.weight, &q.scales, &q.biases, true, 64, 4)
}

/// `nn.Linear` without bias: `x @ W.T`.
fn project(x: &Array, weight: &Array) -> Result<Array> {
    x.matmul(&weight.transpose(&[1, 0])?)
}

/// `nn.Linear` with bias: `addmm(b, x, W.T)`.
fn linear(x: &Array, l: &Linear) -> Result<Array> {
    l.bias.addmm(x, &l.weight.transpose(&[1, 0])?, 1.0, 1.0)
}

fn layer_norm(x: &Array, n: &Norm) -> Result<Array> {
    x.fast_layer_norm(Some(&n.weight), Some(&n.bias), HEAD_EPS)
}

/// `mx.stack(rows)` along a new leading axis.
fn stack(rows: &[Array]) -> Result<Array> {
    let expanded = rows
        .iter()
        .map(|r| {
            let mut shape = vec![1];
            shape.extend_from_slice(r.shape());
            r.reshape(&shape)
        })
        .collect::<Result<Vec<_>>>()?;
    Array::concat(&expanded.iter().collect::<Vec<_>>(), 0)
}

fn row(x: &Array, i: usize) -> Result<Array> {
    x.slice_axis(0, i, i + 1)?.reshape(&x.shape()[1..])
}

fn span_mean(x: &Array, (start, end): (usize, usize)) -> Result<Array> {
    x.slice_axis(0, start, end)?.mean(&[0], false)
}

/// `x / maximum(norm(x, axis=-1, keepdims=True), 1e-12)`.
fn l2_normalize(x: &Array) -> Result<Array> {
    let norm = x.norm(&[-1], true)?;
    x.div(&norm.maximum(&scalar(&norm, 1e-12)?)?)
}

impl Network {
    pub fn new(
        config: Config,
        stream: Stream,
        backbone: Backbone,
        head: Head,
        composition: Composition,
    ) -> Result<Self> {
        let kernels = Kernels {
            recurrence: GatedDeltaRecurrence::new(&stream)?,
            rope: HalfSplitRope::new(&stream, config.rotary_dim())?,
            g_beta: Compiled::with_shapeless(&stream, 2, true, g_beta)?,
        };
        let c14 = match composition {
            Composition::C14 => Some(C14Kernels::new(&stream)?),
            Composition::Reference => None,
        };
        // compute_inv_freq: 1 / (base ** (arange(0, dim, 2) / dim)) in FP32.
        let dim = config.rotary_dim();
        let exponents = Array::arange(&stream, 0.0, dim as f32, 2.0, MlxDType::F32)?
            .div(&Array::scalar(&stream, dim as f32, MlxDType::F32)?)?;
        let base = Array::scalar(
            &stream,
            config.text.rope_parameters.rope_theta,
            MlxDType::F32,
        )?;
        let inv_freq = Array::scalar(&stream, 1.0, MlxDType::F32)?.div(&base.pow(&exponents)?)?;
        stream.eval(&[inv_freq.clone()])?;
        Ok(Self {
            config,
            stream,
            backbone,
            head,
            kernels,
            c14,
            inv_freq,
        })
    }

    pub fn stream(&self) -> &Stream {
        &self.stream
    }

    /// Per-question logits, lazily; the caller evaluates them once.
    pub fn logits(&self, encoded: &Encoded) -> Result<Vec<Array>> {
        let tokens = encoded.input_ids.len();
        let ids = Array::from_i32(
            &self.stream,
            &[tokens],
            &encoded
                .input_ids
                .iter()
                .map(|&id| id as i32)
                .collect::<Vec<_>>(),
        )?;
        let hidden = self.backbone_forward(&ids.reshape(&[1, tokens])?, tokens)?;
        self.head_forward(
            &hidden.reshape(&[tokens, self.config.text.hidden_size])?,
            &ids,
            &encoded.questions,
        )
    }

    fn backbone_forward(&self, ids: &Array, tokens: usize) -> Result<Array> {
        let e = &self.backbone.embedding;
        let mut h = e.weight.take(ids, 0)?.dequantize(
            &e.scales.take(ids, 0)?,
            &e.biases.take(ids, 0)?,
            64,
            4,
        )?;
        let positions = Array::arange(&self.stream, 0.0, tokens as f32, 1.0, MlxDType::I32)?
            .reshape(&[1, tokens])?;
        for layer in &self.backbone.layers {
            let x = h.fast_rms_norm(&layer.input_norm, EPS)?;
            let r = match &layer.mixer {
                Mixer::GatedDelta(g) => self.gated_delta(&x, g, tokens)?,
                Mixer::Attention(a) => self.attention(&x, a, &positions, tokens)?,
            };
            let h1 = h.add(&r)?;
            let m = self.mlp(&h1.fast_rms_norm(&layer.post_norm, EPS)?, &layer.mlp)?;
            h = h1.add(&m)?;
        }
        h.fast_rms_norm(&self.backbone.norm, EPS)
    }

    /// `Qwen3_5MLP`: `down(swiglu(gate(x), up(x)))` on Q4, or C14's MLP.
    fn mlp(&self, h: &Array, m: &Mlp) -> Result<Array> {
        match &self.c14 {
            Some(c) => c.mlp(h, m),
            None => qmm(&silu(&qmm(h, &m.gate)?)?.mul(&qmm(h, &m.up)?)?, &m.down),
        }
    }

    /// A gated-delta `in_proj_qkv`, `in_proj_z` or `out_proj`, or attention
    /// q/o: C14 splits them in two halves.
    fn split_projection(&self, x: &Array, q: &Q4) -> Result<Array> {
        match &self.c14 {
            Some(c) => c.split_linear(x, q),
            None => qmm(x, q),
        }
    }

    /// Attention k/v: one pass in C14.
    fn single_projection(&self, x: &Array, q: &Q4) -> Result<Array> {
        match &self.c14 {
            Some(c) => c.single_linear(x, q),
            None => qmm(x, q),
        }
    }

    fn gated_delta(&self, x: &Array, g: &GatedDelta, tokens: usize) -> Result<Array> {
        let t = &self.config.text;
        let (hk, hv) = (t.linear_num_key_heads, t.linear_num_value_heads);
        let (dk, dv) = (t.linear_key_head_dim, t.linear_value_head_dim);
        let (key_dim, value_dim) = (hk * dk, hv * dv);
        let conv_dim = 2 * key_dim + value_dim;
        let mixed = self.split_projection(x, &g.qkv)?;
        let z = self
            .split_projection(x, &g.z)?
            .reshape(&[1, tokens, hv, dv])?;
        let (b, a) = (qmm(x, &g.b)?, qmm(x, &g.a)?);
        let conv = match &self.c14 {
            Some(c) => silu(&c.conv.call(&mixed, &g.conv)?)?,
            None => {
                let state = Array::zeros(
                    &self.stream,
                    &[1, t.linear_conv_kernel_dim - 1, conv_dim],
                    MlxDType::BF16,
                )?;
                silu(&Array::concat(&[&state, &mixed], 1)?.conv1d(&g.conv, 1, 0, 1, conv_dim)?)?
            }
        };
        let q = conv
            .slice_axis(2, 0, key_dim)?
            .reshape(&[1, tokens, hk, dk])?;
        let k = conv
            .slice_axis(2, key_dim, 2 * key_dim)?
            .reshape(&[1, tokens, hk, dk])?;
        let v = conv
            .slice_axis(2, 2 * key_dim, conv_dim)?
            .reshape(&[1, tokens, hv, dv])?;
        let inv_scale = (dk as f64).powf(-0.5);
        let q = q
            .fast_rms_norm_unweighted(1e-6)?
            .mul(&scalar(&q, inv_scale * inv_scale)?)?;
        let k = k
            .fast_rms_norm_unweighted(1e-6)?
            .mul(&scalar(&k, inv_scale)?)?;
        let [decay, beta] = <[Array; 2]>::try_from(self.kernels.g_beta.call(&[
            g.a_log.clone(),
            a,
            b,
            g.dt_bias.clone(),
        ])?)
        .unwrap();
        let state = Array::zeros(&self.stream, &[1, hv, dv, dk], MlxDType::F32)?;
        let recurrence = match &self.c14 {
            Some(c) => &c.recurrence,
            None => &self.kernels.recurrence,
        };
        let (y, _) = recurrence.call(&q, &k, &v, &decay, &beta, &state)?;
        // Gated RMSNorm: (silu(z) in FP32) * norm(y) in FP32, rounded to BF16.
        let normed = y.fast_rms_norm(&g.norm, EPS)?;
        let gated = silu(&z.cast(MlxDType::F32)?)?
            .mul(&normed.cast(MlxDType::F32)?)?
            .cast(MlxDType::BF16)?;
        self.split_projection(&gated.reshape(&[1, tokens, value_dim])?, &g.out)
    }

    fn attention(
        &self,
        x: &Array,
        a: &Attention,
        positions: &Array,
        tokens: usize,
    ) -> Result<Array> {
        let t = &self.config.text;
        let (heads, kv, d) = (t.num_attention_heads, t.num_key_value_heads, t.head_dim);
        let projected = self
            .split_projection(x, &a.q)?
            .reshape(&[1, tokens, heads, 2 * d])?;
        let (keys, values) = (
            self.single_projection(x, &a.k)?,
            self.single_projection(x, &a.v)?,
        );
        let queries = projected.slice_axis(3, 0, d)?;
        let gate = projected
            .slice_axis(3, d, 2 * d)?
            .reshape(&[1, tokens, heads * d])?;
        let queries = queries
            .fast_rms_norm(&a.q_norm, EPS)?
            .transpose(&[0, 2, 1, 3])?;
        let keys = keys
            .reshape(&[1, tokens, kv, d])?
            .fast_rms_norm(&a.k_norm, EPS)?
            .transpose(&[0, 2, 1, 3])?;
        let values = values
            .reshape(&[1, tokens, kv, d])?
            .transpose(&[0, 2, 1, 3])?;
        let rope = &self.kernels.rope;
        let queries = rope.call(&queries, positions, &self.inv_freq)?;
        let keys = rope.call(&keys, positions, &self.inv_freq)?;
        let out = queries
            .sdpa(&keys, &values, (d as f32).powf(-0.5), tokens > 1, None)?
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[1, tokens, heads * d])?;
        self.split_projection(&out.mul(&gate.sigmoid()?)?, &a.o)
    }

    /// `torch.nn.MultiheadAttention`, batch first, packed input projection.
    fn mha(&self, m: &Mha, q: &Array, k: &Array, v: &Array) -> Result<Array> {
        let width = self.config.head.width;
        let heads = self.config.head.heads;
        let head_dim = width / heads;
        let part = |x: &Array, i: usize| -> Result<Array> {
            let w = m.in_weight.slice_axis(0, i * width, (i + 1) * width)?;
            let b = m.in_bias.slice_axis(0, i * width, (i + 1) * width)?;
            let [batch, length, _] = *x.shape() else {
                unreachable!("head attention operands are rank 3")
            };
            project(x, &w)?
                .add(&b)?
                .reshape(&[batch, length, heads, head_dim])?
                .transpose(&[0, 2, 1, 3])
        };
        let (batch, length) = (q.shape()[0], q.shape()[1]);
        let o = part(q, 0)?.sdpa(
            &part(k, 1)?,
            &part(v, 2)?,
            (head_dim as f32).powf(-0.5),
            false,
            None,
        )?;
        let o = o
            .transpose(&[0, 2, 1, 3])?
            .reshape(&[batch, length, width])?;
        linear(
            &o,
            &Linear {
                weight: m.out_weight.clone(),
                bias: m.out_bias.clone(),
            },
        )
    }

    fn evidence(&self, l: &EvidenceLayer, queries: &Array, memory: &Array) -> Result<Array> {
        let m = layer_norm(memory, &l.memory_norm)?;
        let queries = queries.add(&self.mha(
            &l.attention,
            &layer_norm(queries, &l.query_norm)?,
            &m,
            &m,
        )?)?;
        let ff = linear(
            &gelu(&linear(
                &layer_norm(&queries, &l.feedforward_norm)?,
                &l.fc1,
            )?)?,
            &l.fc2,
        )?;
        queries.add(&ff)
    }

    /// `torch.nn.TransformerDecoderLayer(norm_first=True, activation="gelu")`.
    fn decoder(&self, l: &DecoderLayer, x: &Array, memory: &Array) -> Result<Array> {
        let h = layer_norm(x, &l.norm1)?;
        let x = x.add(&self.mha(&l.self_attn, &h, &h, &h)?)?;
        let x = x.add(&self.mha(&l.cross_attn, &layer_norm(&x, &l.norm2)?, memory, memory)?)?;
        let ff = linear(
            &gelu(&linear(&layer_norm(&x, &l.norm3)?, &l.linear1)?)?,
            &l.linear2,
        )?;
        x.add(&ff)
    }

    /// `JointSchemaHead.__call__` for one record: hidden `[L, H]`.
    fn head_forward(
        &self,
        hidden: &Array,
        ids: &Array,
        questions: &[EncodedQuestion],
    ) -> Result<Vec<Array>> {
        let hd = &self.head;
        let lm = &self.backbone.lm_head;
        let lexical = |span: (usize, usize)| -> Result<Array> {
            let rows = ids.slice_axis(0, span.0, span.1)?;
            lm.weight
                .take(&rows, 0)?
                .dequantize(
                    &lm.scales.take(&rows, 0)?,
                    &lm.biases.take(&rows, 0)?,
                    64,
                    4,
                )?
                .mean(&[0], false)
        };
        let h = layer_norm(hidden, &hd.hidden_norm)?;
        let length = h.shape()[0];
        let memory =
            project(&h, &hd.memory_projection)?.reshape(&[1, length, self.config.head.width])?;
        let global = row(&h, length - 1)?;
        let question_vectors = stack(
            &questions
                .iter()
                .map(|q| span_mean(&h, q.span))
                .collect::<Result<Vec<_>>>()?,
        )?;
        let kinds: Vec<i32> = questions.iter().map(|q| q.kind as i32).collect();
        let kinds = Array::from_i32(&self.stream, &[kinds.len()], &kinds)?;

        let mut lexicals = Vec::with_capacity(questions.len());
        let mut option_queries = Vec::with_capacity(questions.len());
        for (i, q) in questions.iter().enumerate() {
            let context = stack(
                &q.option_spans
                    .iter()
                    .map(|&s| span_mean(&h, s))
                    .collect::<Result<Vec<_>>>()?,
            )?;
            let lex = stack(
                &q.option_spans
                    .iter()
                    .map(|&s| lexical(s))
                    .collect::<Result<Vec<_>>>()?,
            )?;
            let question = project(&row(&question_vectors, i)?, &hd.option_question_projection)?;
            let width = question.shape()[0];
            option_queries.push(
                project(&context, &hd.option_context_projection)?
                    .add(&project(&lex, &hd.option_lexical_projection)?)?
                    .add(&question.reshape(&[1, width])?)?,
            );
            lexicals.push(lex);
        }
        let mut routed = Array::concat(&option_queries.iter().collect::<Vec<_>>(), 0)?;
        let total = routed.shape()[0];
        let width = self.config.head.width;
        routed = routed.reshape(&[1, total, width])?;
        for layer in &hd.evidence {
            routed = self.evidence(layer, &routed, &memory)?;
        }
        let routed = routed.reshape(&[total, width])?;
        let mut split = Vec::with_capacity(questions.len());
        let mut start = 0;
        for q in questions {
            split.push(routed.slice_axis(0, start, start + q.option_spans.len())?);
            start += q.option_spans.len();
        }

        let base = project(&question_vectors, &hd.question_projection)?;
        let mut summaries = Vec::with_capacity(questions.len());
        for (i, options) in split.iter().enumerate() {
            let field = row(&base, i)?;
            let scores = options
                .matmul(&field)?
                .div(&scalar(options, (width as f64).sqrt())?)?;
            let w = scores.softmax_with(0, false)?;
            let n = w.shape()[0];
            summaries.push(w.reshape(&[n, 1])?.mul(options)?.sum(&[0], false)?);
        }
        let global_projected = project(&global, &hd.global_projection)?;
        let fields = base
            .add(&layer_norm(&stack(&summaries)?, &hd.option_summary_norm)?)?
            .add(&global_projected.reshape(&[1, width])?)?
            .add(&hd.type_embedding.take(&kinds, 0)?)?;
        let count = questions.len();
        let mut fields = fields.reshape(&[1, count, width])?;
        for layer in &hd.layers {
            fields = self.decoder(layer, &fields, &memory)?;
        }
        let fields = layer_norm(&fields.reshape(&[count, width])?, &hd.field_norm)?;

        let clamp = |s: &Array| -> Result<Array> { s.minimum(&scalar(s, 100f64.ln())?)?.exp() };
        let prior_scale = clamp(&hd.prior_logit_scale)?;
        let joint_scale = clamp(&hd.joint_logit_scale)?;
        let gate = hd.residual_gate.sigmoid()?;
        let mut logits = Vec::with_capacity(count);
        for i in 0..count {
            let field = row(&fields, i)?;
            let anchor = l2_normalize(&row(&question_vectors, i)?.add(&global)?)?;
            let prior = prior_scale.mul(&l2_normalize(&lexicals[i])?.matmul(&anchor)?)?;
            let options = layer_norm(&split[i], &hd.option_norm)?;
            let rf = field.reshape(&[1, width])?.broadcast_to(options.shape())?;
            let denominator = rf.norm(&[-1], false)?.mul(&options.norm(&[-1], false)?)?;
            let cosine = rf
                .mul(&options)?
                .sum(&[-1], false)?
                .div(&denominator.maximum(&scalar(&denominator, 1e-8)?)?)?;
            let features = Array::concat(
                &[&rf, &options, &rf.mul(&options)?, &rf.sub(&options)?.abs()?],
                1,
            )?;
            let n = options.shape()[0];
            let residual =
                linear(&gelu(&linear(&features, &hd.scorer1)?)?, &hd.scorer2)?.reshape(&[n])?;
            logits.push(prior.add(&gate.mul(&joint_scale.mul(&cosine)?.add(&residual)?)?)?);
        }
        Ok(logits)
    }
}

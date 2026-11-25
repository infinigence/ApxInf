//! BF16 composition of the MolmoAct2 vision encoder, pooling connector and VLM prefill.
//!
//! Rounding follows the official `bf16 + autocast` execution wherever the existing L3 operators
//! can express it; each place where it cannot is listed next to the call instead of being hidden:
//! * ViT LayerNorm: computed in FP32 from BF16 input and rounded once to BF16 (official: FP32
//!   output consumed by an autocast linear, i.e. rounded once at the cast). The remaining FP32
//!   evaluation-order difference to the official operator is not zero.
//! * Residual adds, GELU, SiLU and the gating product each round to BF16 separately, as in torch.
//! * Weighted RMSNorm (hidden and per-head Q/K): the official module rounds `x * rsqrt` to BF16
//!   and then `w * that` again (two roundings). Default = composed from the unit-weight `norm`
//!   and an elementwise product; `official_norm_rounding = false` selects the single-rounding L3
//!   norm (and the fused norm+RoPE operator) instead.
//! * RoPE: the official BF16 path rounds `q*cos`, `rotate_half(q)*sin` and their sum separately.
//!   Default = swap-gather + two elementwise products + add on the official BF16 cos/sin
//!   operands (all three roundings reproduced). The fused norm+RoPE operator, which keeps the
//!   rotation in FP32 and rounds once, is used only in the `official_norm_rounding = false` mode.
//! * Patch normalization: the official GPU path (`images.float() / 255.0 * 2.0 - 1.0`, then BF16)
//!   multiplies by the FP32 reciprocal of 255. The shared `RgbToPatchesReciprocal` gather semantic
//!   expresses exactly that (`((float(v) * float(1/255)) * 2) - 1` in FP32, one BF16 rounding); the
//!   plain `RgbToPatches` (true FP32 division) differs from it only at byte value 127.
use crate::cuda_molmoact2::backend::ops::{
    concat_rows, elementwise_activation, elementwise_add, elementwise_mul, embedding_gather, gemm, gemm_bias,
    AttentionNumerics, ElementwiseActivation, GemmArgs, GemmBiasArgs, GemmPolicy, WeightVersion,
};
use crate::cuda_molmoact2::backend::{
    attention, gather, norm, rms_norm_apply_precomputed, AttentionArgs, AttentionMask, AttentionPolicy,
    CudaBuffer, CudaContext, GatherArgs, GatherPatchGeometry, GatherSemantic, NormArgs,
};
use crate::cuda_molmoact2::config::MolmoAct2Config;
use crate::cuda_molmoact2::geometry::{
    half_swap_ids, plan_prefill_segments, rope_operands_per_head, rope_table, two_view_layout, AttentionSegment,
    PoolingMaps, SegmentKind, TokenLayout, NUM_VIEWS,
};
use crate::cuda_molmoact2::weights::{upload, Bf16Weights, LayerNorm, Linear};
use apxinf_core::{DType, Error, Result, Shape, Tensor};
use std::ops::Range;
use std::rc::Rc;

/// Which attention stream a per-head Q/K operation works on.
#[derive(Clone, Copy)]
enum Qk {
    Query = 0,
    Key = 1,
}

/// Per-head operands of the official BF16 Q/K path for one head count.
struct QkOperands {
    heads: usize,
    /// `[tokens * heads]` zeros: broadcast ids for the per-head norm weight.
    zero_ids: Tensor,
    /// `[tokens * heads, head_dim]` BF16 `[cos, cos]` rows.
    cos: Tensor,
    /// `[tokens * heads, head_dim]` BF16 `[-sin, +sin]` rows (the sign of `rotate_half` folded in).
    sin: Tensor,
    /// `[tokens * heads * 2]` ids swapping the two halves of each row (see `geometry::half_swap_ids`).
    swap_ids: Tensor,
}

/// Request-lifetime inputs derived from the token ids (the "prepared profile").
pub(in crate::cuda_molmoact2) struct Prepared {
    pub tokens: usize,
    pub layout: TokenLayout,
    pub maps: PoolingMaps,
    pub plan: Vec<AttentionSegment>,
    token_ids: Tensor,
    /// `[tokens]` zeros: gather ids that broadcast a one-row table to every position.
    zero_ids: Tensor,
    /// For each position: row of `[connector rows; 0]` to add (the last row is all zeros).
    merge_ids: Tensor,
    /// `[rows*4]` key/value gather ids for the pooling attention.
    key_ids: Tensor,
    /// `[rows, padded_patches]` BF16 averaging matrix for the masked-mean query.
    mean_matrix: Tensor,
    /// Interleave ids turning `[h_a; h_b; zeros]` rows into `[patch, 2*hidden]` feature rows.
    interleave_ids: Tensor,
    /// `[tokens, 2, head_dim/2]` FP32 official BF16 RoPE table (used by the fused operator mode).
    rope_table: Tensor,
    /// Official-rounding Q/K operands, indexed by [`Qk`].
    qk: [QkOperands; 2],
    /// `[max(pad_rows, 1), vit_hidden]` zero rows that pad the stacked ViT features to the GEMM width.
    zero_vit_rows: Tensor,
    /// `[1, text_hidden]` zero row: the "no image feature" entry of the merge table.
    zero_text_row: Tensor,
    /// Some positions are invalid (padding): their attention output rows are never produced by a
    /// segment, so they are zeroed instead of left undefined (see `prefill`).
    has_padding: bool,
}

/// Host-side result of analysing one request's prompt tokens ([`Bf16Blocks::analyze`]).
///
/// *Structure* (image token layout, prefill attention segments, number of visible context rows) fixes
/// every kernel launch and tensor shape of a prepared profile; *contents* (the token ids and the visible
/// row indices) only feed gather tables and can be rewritten in place. A prepared profile accepts a new
/// request exactly when [`HostPlan::same_structure`] holds.
#[derive(Clone)]
pub(in crate::cuda_molmoact2) struct HostPlan {
    pub ids: Vec<i32>,
    pub rows: Vec<i32>,
    valid: Vec<bool>,
    layout: TokenLayout,
    maps: PoolingMaps,
    plan: Vec<AttentionSegment>,
}

impl HostPlan {
    pub fn tokens(&self) -> usize {
        self.ids.len()
    }

    pub fn same_structure(&self, other: &HostPlan) -> bool {
        self.valid == other.valid
            && self.layout == other.layout
            && self.maps == other.maps
            && self.plan == other.plan
            && self.rows.len() == other.rows.len()
    }
}

/// One VLM layer's attention cache entry as the action expert consumes it: `[tokens, kv_heads,
/// head_dim]` BF16, keys after Q/K-norm and RoPE.
pub struct LayerKv {
    pub key: Tensor,
    pub value: Tensor,
}

pub(in crate::cuda_molmoact2) struct Bf16Blocks {
    ctx: Rc<CudaContext>,
    weights: Rc<Bf16Weights>,
    cfg: MolmoAct2Config,
    gemm: GemmPolicy,
    attention: AttentionPolicy,
    numerics: AttentionNumerics,
    /// Reproduce the official two-step weighted RMSNorm and BF16 rotation roundings (see module doc).
    official_norm_rounding: bool,
    /// All-ones weights for the unit-weight RMSNorm of the two-step composition (hidden / head width).
    ones_hidden: Tensor,
    ones_head: Tensor,
    /// Unit weights of the action expert's non-affine norms (hidden / head width).
    ones_expert_hidden: Tensor,
    ones_expert_head: Tensor,
}

mod expert;
pub(in crate::cuda_molmoact2) use expert::{ContextRows, ExpertCache, ExpertContext};

fn io<T>(r: std::result::Result<T, String>) -> Result<T> {
    r.map_err(Error::Cuda)
}

fn dims(d: &[usize]) -> Shape {
    Shape::new(d.to_vec())
}

impl Bf16Blocks {
    pub fn new(
        ctx: Rc<CudaContext>,
        weights: Rc<Bf16Weights>,
        cfg: MolmoAct2Config,
        gemm: GemmPolicy,
        attention: AttentionPolicy,
        numerics: AttentionNumerics,
        official_norm_rounding: bool,
    ) -> Result<Self> {
        let ones = |n: usize| upload(&Tensor::from_bf16(vec![n], &vec![half::bf16::ONE; n])?, ctx.device_id());
        let (ones_hidden, ones_head) = (ones(cfg.text.hidden)?, ones(cfg.text.head_dim)?);
        let (ones_expert_hidden, ones_expert_head) = (ones(cfg.expert.hidden)?, ones(cfg.expert.head_dim)?);
        Ok(Self {
            ctx,
            weights,
            cfg,
            gemm,
            attention,
            numerics,
            official_norm_rounding,
            ones_hidden,
            ones_head,
            ones_expert_hidden,
            ones_expert_head,
        })
    }

    fn alloc(&self, d: &[usize], dtype: DType) -> Result<Tensor> {
        self.ctx.allocate_output(dims(d), dtype)
    }

    fn bf(&self, d: &[usize]) -> Result<Tensor> {
        self.alloc(d, DType::BF16)
    }

    // ---------------------------------------------------------------- prepared inputs

    /// Host-only analysis of the prompt tokens: the structure a prepared profile is built for (image
    /// layout, prefill segments, number of visible context rows) and the token-dependent contents
    /// (ids, visible row indices). No device work.
    pub fn analyze(&self, token_ids: &[u32], valid: &[bool]) -> Result<HostPlan> {
        let (layout, maps) = two_view_layout(&self.cfg, token_ids, valid)?;
        let plan = plan_prefill_segments(&layout.token_type, valid)?;
        if token_ids.len() > self.cfg.text.max_positions {
            return Err(Error::Other("MolmoAct2 input: sequence exceeds max_position_embeddings".into()));
        }
        let vocab = (self.cfg.text.vocab + self.cfg.text.additional_vocab) as u32;
        if token_ids.iter().any(|&i| i >= vocab) {
            return Err(Error::Other("MolmoAct2 input: token id outside the embedding table".into()));
        }
        Ok(HostPlan {
            ids: token_ids.iter().map(|&i| i as i32).collect(),
            valid: valid.to_vec(),
            rows: self.visible_rows(token_ids, valid)?,
            layout,
            maps,
            plan,
        })
    }

    pub fn prepare_host(&self, host: &HostPlan) -> Result<Prepared> {
        let device = self.ctx.device_id();
        let HostPlan { ids, valid, layout, maps, plan, .. } = host.clone();
        let tokens = ids.len();
        let rows = maps.rows;
        let zero_row = rows as i32;
        let mut merge = vec![zero_row; tokens];
        for (k, &p) in layout.patch_positions.iter().enumerate() {
            merge[p] = k as i32;
        }
        let key_ids: Vec<i32> = maps.key_indices()?;
        let patches = maps.total_patches();
        let padded = patches.div_ceil(8) * 8;
        let mean: Vec<half::bf16> = maps.mean_matrix(padded).into_iter().map(half::bf16::from_f32).collect();
        // Table rows: [layer a: patches][layer b: patches][zeros: padded - patches].
        let (a0, b0, z0) = (0i32, patches as i32, 2 * patches as i32);
        let mut inter = Vec::with_capacity(2 * padded);
        for p in 0..padded {
            if p < patches {
                inter.extend([a0 + p as i32, b0 + p as i32]);
            } else {
                let z = z0 + (p - patches) as i32;
                inter.extend([z, z]);
            }
        }
        let table = rope_table(&self.weights.inv_freq, tokens);
        let zeros = |rows: usize, cols: usize| upload(&Tensor::zeros(dims(&[rows, cols]), DType::BF16), device);
        let width = self.cfg.text.head_dim;
        let qk_operands = |heads: usize| -> Result<QkOperands> {
            let rows = tokens * heads;
            let (cos, sin) = rope_operands_per_head(&table, tokens, heads, width / 2);
            Ok(QkOperands {
                heads,
                zero_ids: upload(&Tensor::from_i32(vec![rows], &vec![0i32; rows])?, device)?,
                cos: upload(&Tensor::from_bf16(vec![rows, width], &cos)?, device)?,
                sin: upload(&Tensor::from_bf16(vec![rows, width], &sin)?, device)?,
                swap_ids: upload(&Tensor::from_i32(vec![rows * 2], &half_swap_ids(rows))?, device)?,
            })
        };
        Ok(Prepared {
            tokens,
            layout,
            maps,
            plan,
            token_ids: upload(&Tensor::from_i32(vec![tokens], &ids)?, device)?,
            zero_ids: upload(&Tensor::from_i32(vec![tokens], &vec![0i32; tokens])?, device)?,
            merge_ids: upload(&Tensor::from_i32(vec![tokens], &merge)?, device)?,
            key_ids: upload(&Tensor::from_i32(vec![key_ids.len()], &key_ids)?, device)?,
            mean_matrix: upload(&Tensor::from_bf16(vec![rows, padded], &mean)?, device)?,
            interleave_ids: upload(&Tensor::from_i32(vec![inter.len()], &inter)?, device)?,
            rope_table: upload(&Tensor::from_f32(vec![tokens, 2, self.cfg.text.head_dim / 2], &table)?, device)?,
            qk: [qk_operands(self.cfg.text.heads)?, qk_operands(self.cfg.text.kv_heads)?],
            zero_vit_rows: zeros((padded - patches).max(1), self.cfg.vit.hidden)?,
            zero_text_row: zeros(1, self.cfg.text.hidden)?,
            has_padding: valid.iter().any(|&v| !v),
        })
    }

    /// Rewrite the token-dependent contents of a prepared profile in place (token ids and visible context
    /// rows). Fails before touching the device when the request's structure differs from the one the
    /// profile was built for; the caller then has to prepare again.
    pub fn update_inputs(&self, prep: &Prepared, rows: &ContextRows, host: &HostPlan) -> Result<()> {
        let same = prep.tokens == host.tokens()
            && prep.layout == host.layout
            && prep.plan == host.plan
            && prep.has_padding == host.valid.iter().any(|&v| !v)
            && rows.count == host.rows.len();
        if !same {
            return Err(Error::Other(
                "MolmoAct2: the prompt tokens do not have the image layout / context size this plan was prepared for".into(),
            ));
        }
        let bytes = |v: &[i32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_ne_bytes()).collect() };
        io(io(CudaBuffer::from_tensor(&prep.token_ids))?.copy_from_host_on(&bytes(&host.ids), self.ctx.stream()))?;
        self.write_context_rows(rows, &host.rows)
    }

    // ---------------------------------------------------------------- small helpers

    fn linear(&self, x: &Tensor, l: &Linear) -> Result<Tensor> {
        let m = x.shape().dims()[0];
        let n = l.weight.shape().dims()[1];
        let mut out = self.bf(&[m, n])?;
        {
            let mut g = GemmArgs::new(x, &l.weight, &mut out);
            g.policy = self.gemm.clone();
            g.weight_version = Some(WeightVersion::new(1));
            match &l.bias {
                Some(bias) => gemm_bias(&self.ctx, GemmBiasArgs { gemm: g, bias })?,
                None => gemm(&self.ctx, g)?,
            }
        }
        Ok(out)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let out = self.bf(a.shape().dims())?;
        elementwise_add(&self.ctx, a, b, &out)?;
        Ok(out)
    }

    fn activate(&self, x: &Tensor, act: ElementwiseActivation) -> Result<Tensor> {
        let out = self.bf(x.shape().dims())?;
        elementwise_activation(&self.ctx, x, &out, act)?;
        Ok(out)
    }

    fn layer_norm(&self, x: &Tensor, n: &LayerNorm, eps: f32) -> Result<Tensor> {
        let out = self.bf(x.shape().dims())?;
        norm(&self.ctx, NormArgs::layer(x, &n.weight, &n.bias, &out, eps))?;
        Ok(out)
    }

    fn rms(&self, x: &Tensor, w: &Tensor, eps: f32, prep: &Prepared) -> Result<Tensor> {
        self.rms_with(x, w, eps, prep, self.official_norm_rounding)
    }

    /// Hidden-width weighted RMSNorm with the official two-step rounding (`official`) or the L3 single rounding.
    fn rms_with(&self, x: &Tensor, w: &Tensor, eps: f32, prep: &Prepared, official: bool) -> Result<Tensor> {
        self.rms_rows(x, w, eps, &prep.zero_ids, official)
    }

    fn ones_for(&self, cols: usize) -> Result<&Tensor> {
        if cols == self.cfg.text.hidden {
            Ok(&self.ones_hidden)
        } else if cols == self.cfg.text.head_dim {
            Ok(&self.ones_head)
        } else if cols == self.cfg.expert.hidden {
            Ok(&self.ones_expert_hidden)
        } else if cols == self.cfg.expert.head_dim {
            Ok(&self.ones_expert_head)
        } else {
            Err(Error::Other(format!("no unit norm weight of width {cols}")))
        }
    }

    /// Weighted RMSNorm of `x [rows, cols]`. Official module: `bf16(x * rsqrt)` first, then `w * that`
    /// rounded again. The two roundings are composed from existing operators: unit-weight RMSNorm,
    /// then an elementwise product with the weight broadcast to every row (`bcast_ids` = `rows` zeros
    /// selecting the single row of the one-row weight table). With `official == false`, the L3 operator
    /// rounds `x * rsqrt * w` once.
    fn rms_rows(&self, x: &Tensor, w: &Tensor, eps: f32, bcast_ids: &Tensor, official: bool) -> Result<Tensor> {
        if !official {
            let out = self.bf(x.shape().dims())?;
            norm(&self.ctx, NormArgs::rms(x, w, &out, eps))?;
            return Ok(out);
        }
        let rows = x.shape().dims()[0];
        let cols = w.shape().dims()[0];
        let unit = self.bf(x.shape().dims())?;
        norm(&self.ctx, NormArgs::rms(x, self.ones_for(cols)?, &unit, eps))?;
        let table = w.reshape(dims(&[1, cols]))?;
        let wide = self.gather_rows(&table, bcast_ids, rows)?;
        let out = self.bf(x.shape().dims())?;
        elementwise_mul(&self.ctx, &wide, &unit, &out)?;
        Ok(out)
    }

    /// Per-head Q/K RMSNorm of `x [tokens, heads, head_dim]`; returns `[tokens * heads, head_dim]`.
    fn qk_norm(&self, x: &Tensor, w: &Tensor, which: Qk, prep: &Prepared, official: bool) -> Result<Tensor> {
        let ops = &prep.qk[which as usize];
        let rows = prep.tokens * ops.heads;
        let x2 = x.reshape(dims(&[rows, self.cfg.text.head_dim]))?;
        self.rms_rows(&x2, w, self.cfg.text.eps, &ops.zero_ids, official)
    }

    /// Official BF16 rotary embedding of normalized rows `n [tokens * heads, head_dim]`:
    /// `bf16(bf16(n * cos) + bf16(rotate_half(n) * sin))`, with `rotate_half(n) * sin` evaluated as
    /// `swap_halves(n) * [-sin, +sin]` (bit-identical, see `geometry::rope_operands_per_head`).
    /// Returns `[tokens, heads, head_dim]`.
    fn qk_rope(&self, n: &Tensor, which: Qk, prep: &Prepared) -> Result<Tensor> {
        let ops = &prep.qk[which as usize];
        let t = &self.cfg.text;
        let rows = prep.tokens * ops.heads;
        let halves = n.reshape(dims(&[rows * 2, t.head_dim / 2]))?;
        let swapped = self.gather_rows(&halves, &ops.swap_ids, rows * 2)?.reshape(dims(&[rows, t.head_dim]))?;
        let (p1, p2) = (self.bf(&[rows, t.head_dim])?, self.bf(&[rows, t.head_dim])?);
        elementwise_mul(&self.ctx, n, &ops.cos, &p1)?;
        elementwise_mul(&self.ctx, &swapped, &ops.sin, &p2)?;
        self.add(&p1, &p2)?.reshape(dims(&[prep.tokens, ops.heads, t.head_dim]))
    }

    /// Per-head Q/K norm followed by RoPE: official rounding by default, the fused L3 operator
    /// (single rounding for the norm, FP32 rotation) in the `official_norm_rounding = false` mode.
    fn qk_norm_rope(&self, x: &Tensor, w: &Tensor, which: Qk, prep: &Prepared) -> Result<Tensor> {
        if self.official_norm_rounding {
            let n = self.qk_norm(x, w, which, prep, true)?;
            return self.qk_rope(&n, which, prep);
        }
        let out = self.bf(x.shape().dims())?;
        rms_norm_apply_precomputed(&self.ctx, x, w, &prep.rope_table, self.cfg.text.eps, &out)?;
        Ok(out)
    }

    fn gather_rows(&self, table: &Tensor, ids: &Tensor, rows: usize) -> Result<Tensor> {
        let cols = table.shape().dims()[1];
        let out = self.bf(&[rows, cols])?;
        embedding_gather(&self.ctx, table, ids, &out)?;
        Ok(out)
    }

    /// Rows `[start, start+n)` of `t` (any rank >= 2, leading dim = rows) as a view with `shape`.
    fn rows_view(&self, t: &Tensor, start: usize, n: usize, shape: &[usize]) -> Result<Tensor> {
        let d = t.shape().dims();
        let row_bytes = t.size_in_bytes() / d[0];
        let buf = io(CudaBuffer::from_tensor(t))?;
        io(io(buf.view(start * row_bytes, n * row_bytes))?.as_tensor(dims(shape), t.dtype()))
    }

    /// Concatenate row ranges of `t` into one tensor of shape `[total, tail...]`; a single range
    /// is returned as a view without copying.
    fn gather_ranges(&self, t: &Tensor, ranges: &[Range<usize>]) -> Result<Tensor> {
        let d = t.shape().dims();
        let tail = &d[1..];
        let shape_for = |n: usize| -> Vec<usize> { std::iter::once(n).chain(tail.iter().copied()).collect() };
        if let [r] = ranges {
            return self.rows_view(t, r.start, r.len(), &shape_for(r.len()));
        }
        let total: usize = ranges.iter().map(|r| r.len()).sum();
        let out = self.alloc(&shape_for(total), t.dtype())?;
        let row_bytes = t.size_in_bytes() / d[0];
        let src = io(CudaBuffer::from_tensor(t))?;
        let dst = io(CudaBuffer::from_tensor(&out))?;
        let mut at = 0usize;
        for r in ranges {
            let n = r.len() * row_bytes;
            let s = io(src.view(r.start * row_bytes, n))?;
            let dview = io(dst.view(at, n))?;
            io(dview.copy_from_device_async(&s, n, self.ctx.stream()))?;
            at += n;
        }
        Ok(out)
    }

    fn dense_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, out: &mut Tensor, causal: bool) -> Result<()> {
        let mut args = AttentionArgs::new(q, k, v, out);
        if causal {
            args.mask = AttentionMask::Causal;
        }
        args.numerics = self.numerics;
        args.policy = self.attention.clone();
        attention(&self.ctx, args)
    }

    // ---------------------------------------------------------------- vision encoder

    /// uint8 RGB `[views, size, size, 3]` -> normalized BF16 patch rows `[views * patches, 3 * p * p]`,
    /// emitted per patch in `(c, dy, dx)` order (the checkpoint's patch weight is permuted to match).
    fn patchify(&self, rgb: &CudaBuffer) -> Result<Tensor> {
        let v = &self.cfg.vit;
        let side = self.cfg.patch_side();
        let (rows, patch_dim) = (NUM_VIEWS * side * side, v.patch_size * v.patch_size * 3);
        let mut patches = self.bf(&[rows, patch_dim])?;
        // The official GPU normalization multiplies by the FP32 reciprocal of 255 (see the module note).
        let mut args = GatherArgs::rgb_to_patches(
            rgb,
            &mut patches,
            GatherPatchGeometry { views: NUM_VIEWS, image_size: v.image_size, patch_size: v.patch_size, nhwc: true },
        );
        args.semantic = GatherSemantic::RgbToPatchesReciprocal;
        gather(&self.ctx, args)?;
        Ok(patches)
    }

    /// Returns the two selected ViT block outputs `[views * patches, hidden]` in concatenation order.
    fn vision(&self, rgb: &CudaBuffer) -> Result<Vec<Tensor>> {
        let patches = self.patchify(rgb)?;
        let v = &self.cfg.vit;
        let w = &self.weights;
        let side = self.cfg.patch_side();
        let rows = NUM_VIEWS * side * side;
        // Patch embedding rounds the linear output (with bias) to BF16, then the positional
        // embedding is added with a second rounding, as torch does.
        let projected = self.linear(&patches, &w.patch_embedding)?;
        let mut x = self.bf(&[rows, v.hidden])?;
        {
            let mut g = GatherArgs::new(GatherSemantic::BiasPosition, &projected, &mut x);
            g.position = Some(&w.positional_embedding);
            g.tokens_per_view = side * side;
            gather(&self.ctx, g)?;
        }
        let mut selected: Vec<Option<Tensor>> = vec![None; self.cfg.adapter.vit_layers.len()];
        for (i, blk) in w.vit.iter().enumerate() {
            let n1 = self.layer_norm(&x, &blk.attention_norm, v.eps)?;
            let q = self.linear(&n1, &blk.wq)?;
            let k = self.linear(&n1, &blk.wk)?;
            let vv = self.linear(&n1, &blk.wv)?;
            let shape = [NUM_VIEWS, side * side, v.heads, v.head_dim];
            let (q, k, vv) = (q.reshape(dims(&shape))?, k.reshape(dims(&shape))?, vv.reshape(dims(&shape))?);
            let mut a = self.bf(&shape)?;
            self.dense_attention(&q, &k, &vv, &mut a, false)?;
            let o = self.linear(&a.reshape(dims(&[rows, v.hidden]))?, &blk.wo)?;
            let x1 = self.add(&x, &o)?;
            let n2 = self.layer_norm(&x1, &blk.ffn_norm, v.eps)?;
            let h = self.linear(&n2, &blk.w1)?;
            let h = self.activate(&h, ElementwiseActivation::GeluTanh)?;
            let m = self.linear(&h, &blk.w2)?;
            x = self.add(&x1, &m)?;
            for (slot, &layer) in self.cfg.adapter.vit_layers.iter().enumerate() {
                if layer == i {
                    selected[slot] = Some(x.clone());
                }
            }
        }
        selected
            .into_iter()
            .map(|s| s.ok_or_else(|| Error::Other("selected ViT layer was not executed".into())))
            .collect()
    }

    // ---------------------------------------------------------------- pooling + connector

    fn connector(&self, selected: &[Tensor], prep: &Prepared) -> Result<Tensor> {
        let (v, a) = (&self.cfg.vit, &self.cfg.adapter);
        let w = &self.weights;
        let patches = prep.maps.total_patches();
        let padded = patches.div_ceil(8) * 8;
        let heads = a.heads;
        // `[h_a; h_b; zero rows]` -> interleave -> `[padded, 2*hidden]` feature rows (a|b per patch).
        let stacked = concat_rows(&self.ctx, &concat_rows(&self.ctx, &selected[0], &selected[1])?, &prep.zero_vit_rows)?;
        let feats = self
            .gather_rows(&stacked, &prep.interleave_ids, 2 * padded)?
            .reshape(dims(&[padded, 2 * v.hidden]))?;

        // Masked-mean query: averaging matrix @ features (FP32 accumulate, one BF16 rounding).
        let mut query = self.bf(&[prep.maps.rows, 2 * v.hidden])?;
        {
            let mut g = GemmArgs::new(&prep.mean_matrix, &feats, &mut query);
            g.policy = self.gemm.clone();
            gemm(&self.ctx, g)?;
        }
        let rows = prep.maps.rows;
        let q = self.linear(&query, &w.pooling.wq)?;
        // K/V are projected once per patch row, then gathered per window; invalid window cells
        // repeat the valid ones so each row presents exactly its valid key set (see PoolingMaps).
        let k_all = self.linear(&feats, &w.pooling.wk)?;
        let v_all = self.linear(&feats, &w.pooling.wv)?;
        let k = self.gather_rows(&k_all, &prep.key_ids, rows * 4)?;
        let vv = self.gather_rows(&v_all, &prep.key_ids, rows * 4)?;
        let q4 = q.reshape(dims(&[rows, 1, heads, a.head_dim]))?;
        let k4 = k.reshape(dims(&[rows, 4, heads, a.head_dim]))?;
        let v4 = vv.reshape(dims(&[rows, 4, heads, a.head_dim]))?;
        let mut att = self.bf(&[rows, 1, heads, a.head_dim])?;
        self.dense_attention(&q4, &k4, &v4, &mut att, false)?;
        let pooled = self.linear(&att.reshape(dims(&[rows, a.hidden]))?, &w.pooling.wo)?;
        // Projector: w2(silu(w1 x) * w3 x), each step rounded to BF16 like torch.
        let g = self.activate(&self.linear(&pooled, &w.projector.w1)?, ElementwiseActivation::Silu)?;
        let u = self.linear(&pooled, &w.projector.w3)?;
        let gu = self.bf(g.shape().dims())?;
        elementwise_mul(&self.ctx, &g, &u, &gu)?;
        self.linear(&gu, &w.projector.w2)
    }

    // ---------------------------------------------------------------- embeddings

    fn embed(&self, connector: &Tensor, prep: &Prepared) -> Result<Tensor> {
        let tokens = prep.tokens;
        let wte = self.gather_rows(&self.weights.wte, &prep.token_ids, tokens)?;
        let table = concat_rows(&self.ctx, connector, &prep.zero_text_row)?;
        let image = self.gather_rows(&table, &prep.merge_ids, tokens)?;
        // `x[is_patch] += image_features`: one BF16 add; rows without an image feature add zero.
        self.add(&wte, &image)
    }

    // ---------------------------------------------------------------- VLM prefill

    fn prefill_attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, prep: &Prepared, out: &Tensor) -> Result<()> {
        let t = &self.cfg.text;
        for seg in &prep.plan {
            let tq = seg.queries.len();
            let qv = self.rows_view(q, seg.queries.start, tq, &[1, tq, t.heads, t.head_dim])?;
            let ks = self.gather_ranges(k, &seg.key_ranges)?;
            let vs = self.gather_ranges(v, &seg.key_ranges)?;
            let tk = ks.shape().dims()[0];
            let ks = ks.reshape(dims(&[1, tk, t.kv_heads, t.head_dim]))?;
            let vs = vs.reshape(dims(&[1, tk, t.kv_heads, t.head_dim]))?;
            let mut o = self.rows_view(out, seg.queries.start, tq, &[1, tq, t.heads, t.head_dim])?;
            self.dense_attention(&qv, &ks, &vs, &mut o, seg.kind == SegmentKind::Causal)?;
        }
        Ok(())
    }

    fn prefill(&self, mut x: Tensor, prep: &Prepared) -> Result<Vec<LayerKv>> {
        let t = &self.cfg.text;
        let tokens = prep.tokens;
        let mut kv = Vec::with_capacity(t.layers);
        for blk in &self.weights.blocks {
            let h = self.rms(&x, &blk.attn_norm, t.eps, prep)?;
            let q = self.linear(&h, &blk.q)?.reshape(dims(&[tokens, t.heads, t.head_dim]))?;
            let k = self.linear(&h, &blk.k)?.reshape(dims(&[tokens, t.kv_heads, t.head_dim]))?;
            let v = self.linear(&h, &blk.v)?.reshape(dims(&[tokens, t.kv_heads, t.head_dim]))?;
            // Per-head Q/K RMSNorm (weighted) and half-split RoPE of the official BF16 table.
            let q_out = self.qk_norm_rope(&q, &blk.q_norm, Qk::Query, prep)?;
            let k_out = self.qk_norm_rope(&k, &blk.k_norm, Qk::Key, prep)?;
            let attn = self.bf(&[tokens, t.heads, t.head_dim])?;
            if prep.has_padding {
                // Invalid positions are not queries of any segment; define their attention output as
                // zero (finite) rather than leaving device memory undefined. The official masked
                // attention gives such rows an arbitrary/NaN-prone value; they are masked out of the
                // action expert's context either way.
                io(io(CudaBuffer::from_tensor(&attn))?.zero_async(self.ctx.stream()))?;
            }
            self.prefill_attention(&q_out, &k_out, &v, prep, &attn)?;
            let o = self.linear(&attn.reshape(dims(&[tokens, t.heads * t.head_dim]))?, &blk.attn_out)?;
            let x1 = self.add(&x, &o)?;
            let n = self.rms(&x1, &blk.ff_norm, t.eps, prep)?;
            let gate = self.activate(&self.linear(&n, &blk.gate)?, ElementwiseActivation::Silu)?;
            let up = self.linear(&n, &blk.up)?;
            let gu = self.bf(gate.shape().dims())?;
            elementwise_mul(&self.ctx, &gate, &up, &gu)?;
            let m = self.linear(&gu, &blk.down)?;
            x = self.add(&x1, &m)?;
            kv.push(LayerKv { key: k_out, value: v });
        }
        // The final norm of the language model is not applied: only the K/V cache is consumed.
        Ok(kv)
    }

    /// Vision encoder -> pooling connector -> embedding merge -> 36-layer prefill.
    pub fn encode(&self, rgb: &CudaBuffer, prep: &Prepared) -> Result<Vec<LayerKv>> {
        let selected = self.vision(rgb)?;
        let connector = self.connector(&selected, prep)?;
        let x = self.embed(&connector, prep)?;
        self.prefill(x, prep)
    }
}

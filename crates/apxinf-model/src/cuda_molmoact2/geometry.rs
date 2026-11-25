//! Pure host-side geometry for the single-crop, two-view MolmoAct2 profile: token types, image
//! runs, pooling index maps, the VLM prefill visibility plan and the RoPE table. Everything here
//! is a function of the token ids / checkpoint constants, never of a fixture's layout, and is
//! checked against a dense reference mask in the tests below.
use super::config::{ImageTokens, MolmoAct2Config};
use apxinf_core::{Error, Result};
use half::bf16;
use std::ops::Range;

fn bad(what: impl std::fmt::Display) -> Error {
    Error::Other(format!("MolmoAct2 input: {what}"))
}

/// Number of camera views this profile accepts.
pub const NUM_VIEWS: usize = 2;

/// Token-derived layout of one request.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenLayout {
    /// `token_type_ids`: true for every image-type token.
    pub token_type: Vec<bool>,
    /// Maximal runs of contiguous valid image-type tokens, in order.
    pub image_runs: Vec<Range<usize>>,
    /// Positions of `image_patch` tokens, in order; the k-th one receives pooled row k.
    pub patch_positions: Vec<usize>,
}

impl TokenLayout {
    pub fn from_tokens(
        tokens: &ImageTokens,
        ids: &[u32],
        valid: &[bool],
        expected_runs: usize,
        patches_per_run: usize,
    ) -> Result<Self> {
        if ids.is_empty() || ids.len() != valid.len() {
            return Err(bad("token ids and valid mask must be non-empty and equally long"));
        }
        let token_type: Vec<bool> = ids.iter().map(|&i| tokens.is_image_token(i)).collect();
        let mut image_runs = Vec::new();
        let mut start = None;
        for (pos, (&img, &ok)) in token_type.iter().zip(valid).enumerate() {
            match (img && ok, start) {
                (true, None) => start = Some(pos),
                (false, Some(s)) => {
                    image_runs.push(s..pos);
                    start = None;
                }
                _ => {}
            }
        }
        if let Some(s) = start {
            image_runs.push(s..ids.len());
        }
        let patch_positions: Vec<usize> = ids
            .iter()
            .enumerate()
            .filter(|&(p, &i)| i == tokens.patch && valid[p])
            .map(|(p, _)| p)
            .collect();
        if image_runs.len() != expected_runs {
            return Err(bad(format!(
                "expected {expected_runs} image runs, found {}",
                image_runs.len()
            )));
        }
        for run in &image_runs {
            let n = patch_positions.iter().filter(|p| run.contains(p)).count();
            if n != patches_per_run {
                return Err(bad(format!(
                    "image run {run:?} has {n} patch tokens, expected {patches_per_run}"
                )));
            }
        }
        if patch_positions.len() != expected_runs * patches_per_run {
            return Err(bad("image patch tokens outside the image runs"));
        }
        Ok(Self {
            token_type,
            image_runs,
            patch_positions,
        })
    }
}

/// Pooling index maps for `views` single-crop images of `side x side` patches and a 2x2 window.
/// Mirrors the processor: pad the patch grid at the bottom/right with -1 to an even size, then
/// group each 2x2 window as `(dh dw)`; image `k` indexes into the concatenated crop axis.
#[derive(Clone, Debug, PartialEq)]
pub struct PoolingMaps {
    pub views: usize,
    pub patches_per_view: usize,
    /// `[rows, 4]` row-major; -1 marks a padded (invalid) window cell.
    pub idx: Vec<i32>,
    pub rows: usize,
}

impl PoolingMaps {
    pub fn new(views: usize, side: usize, pool: (usize, usize)) -> Self {
        let (ph, pw) = pool;
        let (gh, gw) = (side.div_ceil(ph), side.div_ceil(pw));
        // Padding is split as (pad // 2, (pad + 1) // 2); for the odd side of this profile that
        // puts the single padded row/column at the bottom/right.
        let (pad_h, pad_w) = (gh * ph - side, gw * pw - side);
        let (top, left) = (pad_h / 2, pad_w / 2);
        let mut idx = Vec::with_capacity(views * gh * gw * ph * pw);
        for view in 0..views {
            for gy in 0..gh {
                for gx in 0..gw {
                    for dh in 0..ph {
                        for dw in 0..pw {
                            let (r, c) = ((gy * ph + dh) as i64 - top as i64, (gx * pw + dw) as i64 - left as i64);
                            let valid = r >= 0 && c >= 0 && (r as usize) < side && (c as usize) < side;
                            idx.push(if valid {
                                (view * side * side) as i32 + (r as usize * side + c as usize) as i32
                            } else {
                                -1
                            });
                        }
                    }
                }
            }
        }
        Self {
            views,
            patches_per_view: side * side,
            rows: views * gh * gw,
            idx,
        }
    }

    pub fn total_patches(&self) -> usize {
        self.views * self.patches_per_view
    }

    fn row(&self, r: usize) -> &[i32] {
        &self.idx[r * 4..r * 4 + 4]
    }

    /// Number of valid cells per row (the masked-mean denominator).
    #[cfg(test)]
    pub fn valid_counts(&self) -> Vec<usize> {
        (0..self.rows)
            .map(|r| self.row(r).iter().filter(|&&i| i >= 0).count())
            .collect()
    }

    /// Averaging matrix `[rows, padded_cols]` for the masked-mean query: row `r` holds `1/n` at
    /// its `n` valid patch columns and 0 elsewhere. 1, 1/2 and 1/4 are exact in BF16, so
    /// `matrix @ features` accumulates the valid features in FP32 and rounds once, which for these
    /// power-of-two counts equals the official `bf16(sum) / n`.
    pub fn mean_matrix(&self, padded_cols: usize) -> Vec<f32> {
        assert!(padded_cols >= self.total_patches());
        let mut m = vec![0f32; self.rows * padded_cols];
        for r in 0..self.rows {
            let n = self.row(r).iter().filter(|&&i| i >= 0).count();
            for &i in self.row(r).iter().filter(|&&i| i >= 0) {
                m[r * padded_cols + i as usize] = 1.0 / n as f32;
            }
        }
        m
    }

    /// Gather ids `[rows * 4]` (row-major `(row, slot)`) for the attention keys/values. Invalid
    /// slots repeat the row's valid keys cyclically, so every row presents exactly its valid key
    /// set, each key repeated `4 / n` times (n in {1, 2, 4}); softmax over a uniformly repeated
    /// set equals softmax over the set. Returns an error for any other valid count.
    pub fn key_indices(&self) -> Result<Vec<i32>> {
        let mut out = Vec::with_capacity(self.rows * 4);
        for r in 0..self.rows {
            let valid: Vec<i32> = self.row(r).iter().copied().filter(|&i| i >= 0).collect();
            if !matches!(valid.len(), 1 | 2 | 4) {
                return Err(bad(format!("pooling row {r} has {} valid cells; only 1, 2 or 4 are supported", valid.len())));
            }
            for slot in 0..4 {
                out.push(valid[slot % valid.len()]);
            }
        }
        Ok(out)
    }
}

/// How one group of query rows attends during VLM prefill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentKind {
    /// Keys are ordered by position and the queries are the last `query_len` keys: standard causal.
    Causal,
    /// Every query sees every key.
    Full,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AttentionSegment {
    pub queries: Range<usize>,
    pub kind: SegmentKind,
    /// Key rows (original positions) as ascending contiguous ranges; the attention keys are
    /// their concatenation.
    pub key_ranges: Vec<Range<usize>>,
}

fn ranges_of(select: impl Fn(usize) -> bool, len: usize, within: Range<usize>) -> Vec<Range<usize>> {
    let mut out = Vec::new();
    let mut start = None;
    for p in within.start..within.end.min(len) {
        match (select(p), start) {
            (true, None) => start = Some(p),
            (false, Some(s)) => {
                out.push(s..p);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push(s..within.end.min(len));
    }
    out
}

/// Prefill visibility: `allow(q, k) = valid[k] && (k <= q || (image[q] && image[k]))`.
///
/// Queries are split into maximal runs of contiguous valid positions with the same token type.
/// All queries of one run share a key set, so each run is one dense attention call:
/// * a text run `[s, t]` is causal over the valid positions `<= t` (the run is the tail of that key list);
/// * an image run `[s, e]` sees every valid position `<= e` plus every valid image position after
///   it, with no mask, because the visible set does not depend on the query inside the run.
pub fn plan_prefill_segments(token_type: &[bool], valid: &[bool]) -> Result<Vec<AttentionSegment>> {
    let n = token_type.len();
    if n == 0 || valid.len() != n {
        return Err(bad("token types and valid mask must be non-empty and equally long"));
    }
    let mut segments = Vec::new();
    let mut p = 0;
    while p < n {
        if !valid[p] {
            p += 1;
            continue;
        }
        let kind_image = token_type[p];
        let mut e = p + 1;
        while e < n && valid[e] && token_type[e] == kind_image {
            e += 1;
        }
        let queries = p..e;
        let segment = if kind_image {
            let mut key_ranges = ranges_of(|i| valid[i], n, 0..e);
            key_ranges.extend(ranges_of(|i| valid[i] && token_type[i], n, e..n));
            AttentionSegment { queries, kind: SegmentKind::Full, key_ranges }
        } else {
            AttentionSegment {
                key_ranges: ranges_of(|i| valid[i], n, 0..e),
                queries,
                kind: SegmentKind::Causal,
            }
        };
        segments.push(segment);
        p = e;
    }
    Ok(segments)
}

/// Dense reference mask `[n, n]`: exactly the official `valid & (causal | image x image)`.
#[cfg(test)]
pub fn dense_allowed_mask(token_type: &[bool], valid: &[bool]) -> Vec<bool> {
    let n = token_type.len();
    let mut m = vec![false; n * n];
    for q in 0..n {
        for k in 0..n {
            m[q * n + k] = valid[k] && (k <= q || (token_type[q] && token_type[k]));
        }
    }
    m
}

/// What the segment plan lets each valid query see, expanded to the dense form.
#[cfg(test)]
pub fn expand_plan(plan: &[AttentionSegment], n: usize) -> Vec<bool> {
    let mut m = vec![false; n * n];
    for seg in plan {
        let keys: Vec<usize> = seg.key_ranges.iter().flat_map(|r| r.clone()).collect();
        for q in seg.queries.clone() {
            match seg.kind {
                SegmentKind::Full => keys.iter().for_each(|&k| m[q * n + k] = true),
                SegmentKind::Causal => {
                    // The queries are the last `len` keys, query i aligned to key `keys.len()-len+i`.
                    let aligned = keys.len() - seg.queries.len() + (q - seg.queries.start);
                    keys[..=aligned].iter().for_each(|&k| m[q * n + k] = true);
                }
            }
        }
    }
    m
}

/// RoPE cos/sin table for positions `0..positions`, laid out `[positions, 2, head_dim/2]` (cos
/// then sin, FP32 carrying BF16-rounded values), reproducing the official BF16 cache: FP32
/// `position * inv_freq`, sin/cos, rounded to BF16. The trig runs in f64 and is rounded to f32
/// first, which agrees bitwise with the official CPU/BF16 cache.
pub fn rope_table(inv_freq: &[f32], positions: usize) -> Vec<f32> {
    let half = inv_freq.len();
    let mut out = vec![0f32; positions * 2 * half];
    for p in 0..positions {
        for (j, &f) in inv_freq.iter().enumerate() {
            let angle = (p as f32 * f) as f64;
            let round = |v: f64| bf16::from_f32(v as f32).to_f32();
            out[p * 2 * half + j] = round(angle.cos());
            out[p * 2 * half + half + j] = round(angle.sin());
        }
    }
    out
}

/// Rows of the VLM context the action expert may attend (official `_get_encoder_attention_mask`
/// for `action_mode == "both"`): valid tokens, minus every `eos_token_id` position, minus each
/// `[action_start, matching action_end]` span (an unmatched start masks to the end).
pub fn encoder_attention_mask(cfg: &MolmoAct2Config, ids: &[u32], valid: &[bool]) -> Vec<bool> {
    let mut mask: Vec<bool> = ids.iter().zip(valid).map(|(&i, &v)| v && i != cfg.eos_token_id).collect();
    let starts: Vec<usize> = (0..ids.len()).filter(|&p| ids[p] == cfg.action_start_token_id).collect();
    let ends: Vec<usize> = (0..ids.len()).filter(|&p| ids[p] == cfg.action_end_token_id).collect();
    let mut e = 0;
    for s in starts {
        while e < ends.len() && ends[e] < s {
            e += 1;
        }
        if e >= ends.len() {
            mask[s..].iter_mut().for_each(|m| *m = false);
            break;
        }
        mask[s..=ends[e]].iter_mut().for_each(|m| *m = false);
        e += 1;
    }
    mask
}

/// Operands of the official BF16 `apply_rotary_pos_emb(q) = q * cos + rotate_half(q) * sin`, expanded
/// to one row per `(position, head)` so they can be multiplied elementwise with `[positions * heads,
/// 2 * half]` activations by existing operators. `table` is [`rope_table`]'s `[positions, 2, half]`
/// layout (BF16-exact values). Returns `(cos_full, sin_signed)`:
/// * `cos_full = [cos, cos]` (the official cache concatenates the frequencies with themselves);
/// * `sin_signed = [-sin, +sin]`: `rotate_half(x) = [-x2, x1]`, so `rotate_half(x) * sin` equals
///   `swap_halves(x) * sin_signed` bit for bit (a sign flip never changes the rounded product).
pub fn rope_operands_per_head(table: &[f32], positions: usize, heads: usize, half: usize) -> (Vec<bf16>, Vec<bf16>) {
    assert_eq!(table.len(), positions * 2 * half);
    let width = 2 * half;
    let mut cos = Vec::with_capacity(positions * heads * width);
    let mut sin = Vec::with_capacity(positions * heads * width);
    for p in 0..positions {
        let (c, s) = (&table[p * width..p * width + half], &table[p * width + half..(p + 1) * width]);
        for _ in 0..heads {
            cos.extend(c.iter().chain(c).map(|&v| bf16::from_f32(v)));
            sin.extend(s.iter().map(|&v| bf16::from_f32(-v)).chain(s.iter().map(|&v| bf16::from_f32(v))));
        }
    }
    (cos, sin)
}

/// Gather ids that swap the two halves of every row when the `[rows, 2 * half]` activation is viewed
/// as a `[rows * 2, half]` table: output row `2r` is input row `2r + 1` and vice versa.
pub fn half_swap_ids(rows: usize) -> Vec<i32> {
    (0..rows * 2).map(|i| (i ^ 1) as i32).collect()
}

/// Convenience: the layout for the fixed two-view profile of a config.
pub fn two_view_layout(cfg: &MolmoAct2Config, ids: &[u32], valid: &[bool]) -> Result<(TokenLayout, PoolingMaps)> {
    let side = cfg.patch_side();
    let maps = PoolingMaps::new(NUM_VIEWS, side, cfg.pooling);
    let layout = TokenLayout::from_tokens(&cfg.image_tokens, ids, valid, NUM_VIEWS, maps.rows / NUM_VIEWS)?;
    Ok((layout, maps))
}

/// Official `SinusoidalTimeEmbedding` of the flow times `t_i = i / steps` (the timesteps are FP32
/// tensors holding `float32(i / steps)`), `[steps, dim]` as `[sin | cos]`, cast to the BF16 dtype of the
/// first time-embedding linear. The product `t * freq` is an FP32 multiply as in torch; `exp`, `sin`
/// and `cos` are evaluated in f64 and rounded to f32 (the device functions are within an ulp of that).
/// These are checkpoint-independent constants of the flow schedule, prepared once on the host.
pub fn timestep_embedding(steps: usize, dim: usize) -> Vec<bf16> {
    let half = dim / 2;
    // `arange(half) * (-ln(10000) / max(half - 1, 1))`: the Python scalar multiplies an FP32 tensor.
    let scale = (-(10000f64.ln()) / half.saturating_sub(1).max(1) as f64) as f32;
    let freq: Vec<f32> = (0..half).map(|j| ((j as f32 * scale) as f64).exp() as f32).collect();
    let mut out = Vec::with_capacity(steps * dim);
    for i in 0..steps {
        let t = (i as f64 / steps as f64) as f32;
        let args: Vec<f32> = freq.iter().map(|&f| t * f).collect();
        out.extend(args.iter().map(|&a| bf16::from_f32((a as f64).sin() as f32)));
        out.extend(args.iter().map(|&a| bf16::from_f32((a as f64).cos() as f32)));
        if dim % 2 == 1 {
            out.push(bf16::ZERO);
        }
    }
    out
}

/// `inv_freq` of the expert's self-attention RoPE (`ActionExpertRotaryEmbedding.build_cache`):
/// `1 / (10000 ** (arange(half) / half))` in FP32.
pub fn expert_inv_freq(head_dim: usize) -> Vec<f32> {
    let half = head_dim / 2;
    (0..half)
        .map(|j| {
            let exponent = j as f32 / half.max(1) as f32;
            1.0f32 / (10000f64.powf(exponent as f64) as f32)
        })
        .collect()
}

/// Row ids that expand the block modulation `[steps * 9, hidden]` (row `step * 9 + stage * 3 + kind`,
/// kinds shift / scale / gate) into `[steps * 3 * rows, hidden]` tables whose row
/// `(step * 3 + stage) * rows + r` repeats the vector for every action row `r`.
pub fn block_modulation_ids(steps: usize, rows: usize, kind: usize) -> Vec<i32> {
    assert!(kind < 3);
    let mut ids = Vec::with_capacity(steps * 3 * rows);
    for step in 0..steps {
        for stage in 0..3 {
            ids.extend(std::iter::repeat((step * 9 + stage * 3 + kind) as i32).take(rows));
        }
    }
    ids
}

/// Same for the final layer's `[steps * 2, hidden]` (row `step * 2 + part`, parts shift / scale):
/// `[steps * rows, hidden]` with row `step * rows + r`.
pub fn final_modulation_ids(steps: usize, rows: usize, part: usize) -> Vec<i32> {
    assert!(part < 2);
    (0..steps).flat_map(|step| std::iter::repeat((step * 2 + part) as i32).take(rows)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *state >> 33
    }

    #[test]
    fn plan_reproduces_dense_mask_for_random_layouts_including_padding() {
        let mut s = 7u64;
        for trial in 0..400 {
            let n = 1 + (lcg(&mut s) % 40) as usize;
            // Bursty types so multi-token image runs and several runs appear.
            let mut ty = Vec::new();
            let mut cur = lcg(&mut s) % 2 == 0;
            while ty.len() < n {
                for _ in 0..1 + lcg(&mut s) % 7 {
                    if ty.len() < n {
                        ty.push(cur);
                    }
                }
                cur = !cur;
            }
            let valid: Vec<bool> = (0..n)
                .map(|_| trial % 3 == 0 || lcg(&mut s) % 6 != 0)
                .collect();
            let plan = plan_prefill_segments(&ty, &valid).unwrap();
            let dense = dense_allowed_mask(&ty, &valid);
            let got = expand_plan(&plan, n);
            for q in 0..n {
                if !valid[q] {
                    continue; // outputs of invalid query rows are never consumed
                }
                assert_eq!(&got[q * n..(q + 1) * n], &dense[q * n..(q + 1) * n], "trial {trial} q {q} ty {ty:?} valid {valid:?}");
            }
        }
    }

    #[test]
    fn plan_for_two_image_blocks_keeps_text_between_blocks_causal() {
        // text(2) img(3) text(2) img(2) text(2)
        let ty = [false, false, true, true, true, false, false, true, true, false, false];
        let valid = vec![true; ty.len()];
        let plan = plan_prefill_segments(&ty, &valid).unwrap();
        assert_eq!(plan.len(), 5);
        // First image run sees everything up to its end plus the whole second run, but not the
        // text between the runs.
        assert_eq!(plan[1].kind, SegmentKind::Full);
        assert_eq!(plan[1].key_ranges, vec![0..5, 7..9]);
        // The text between the runs may not see the second run.
        assert_eq!(plan[2].kind, SegmentKind::Causal);
        assert_eq!(plan[2].key_ranges, vec![0..7]);
        assert_eq!(plan[3].key_ranges, vec![0..9]);
        // The whole prefix is NOT bidirectional: the first text token cannot see the images.
        let dense = dense_allowed_mask(&ty, &valid);
        assert!(!dense[0 * 11 + 2]);
    }

    #[test]
    fn pooling_maps_match_the_processor_construction() {
        let m = PoolingMaps::new(2, 27, (2, 2));
        assert_eq!(m.rows, 392);
        let counts = m.valid_counts();
        let hist = |n| counts.iter().filter(|&&c| c == n).count();
        assert_eq!((hist(4), hist(2), hist(1)), (338, 52, 2));
        // First window of view 0 is the top-left 2x2 patches; the last window of view 0 has one
        // valid cell: the bottom-right patch (26, 26) of the view.
        assert_eq!(&m.idx[0..4], &[0, 1, 27, 28]);
        let last0 = 195 * 4;
        assert_eq!(&m.idx[last0..last0 + 4], &[26 * 27 + 26, -1, -1, -1]);
        // View 1 indexes the second crop.
        assert_eq!(m.idx[196 * 4], 729);
        // Key map repeats valid cells cyclically and never points at padding.
        let keys = m.key_indices().unwrap();
        assert!(keys.iter().all(|&i| i >= 0));
        assert_eq!(&keys[last0..last0 + 4], &[26 * 27 + 26; 4]);
        // The averaging matrix weights each valid cell by 1/n.
        let a = m.mean_matrix(1464);
        assert_eq!(a[195 * 1464 + 26 * 27 + 26], 1.0);
        assert_eq!(a[1], 0.25);
    }

    #[test]
    fn encoder_mask_hides_eos_and_action_spans() {
        let cfg = MolmoAct2Config::from_json(&super::super::config::tests::libero_json()).unwrap();
        let ids = [151645, 5, 6, 151932, 7, 151933, 8, 151645, 151932, 9];
        let valid = [true; 10];
        let m = encoder_attention_mask(&cfg, &ids, &valid);
        assert_eq!(m, vec![false, true, true, false, false, false, true, false, false, false]);
    }

    #[test]
    fn swap_and_signed_sin_reproduce_rotate_half_bit_for_bit() {
        // Textbook BF16 rotation: bf16( bf16(x*cos) + bf16(rotate_half(x)*sin) ), rotate_half = [-x2, x1].
        let half = 8usize;
        let inv: Vec<f32> = (0..half).map(|j| 1.0 / 10f32.powf(j as f32 / 3.0)).collect();
        let (positions, heads) = (5usize, 3usize);
        let table = rope_table(&inv, positions);
        let (cos_full, sin_signed) = rope_operands_per_head(&table, positions, heads, half);
        let width = 2 * half;
        let mut s = 99u64;
        let x: Vec<bf16> = (0..positions * heads * width)
            .map(|_| bf16::from_f32((lcg(&mut s) % 2001) as f32 / 100.0 - 10.0))
            .collect();
        let r = |v: f32| bf16::from_f32(v).to_f32();
        let ids = half_swap_ids(positions * heads);
        for row in 0..positions * heads {
            let p = row / heads;
            for j in 0..width {
                let (cos, sin) = (table[p * width + (j % half)], table[p * width + half + (j % half)]);
                let xv = x[row * width + j].to_f32();
                let rotated = if j < half { -x[row * width + j + half].to_f32() } else { x[row * width + j - half].to_f32() };
                let want = r(r(xv * cos) + r(rotated * sin));
                // Candidate: swapped operand (table row ids) times the folded-sign sin.
                let swapped = x[ids[row * 2 + j / half] as usize * half + j % half].to_f32();
                let got = r(r(xv * cos_full[row * width + j].to_f32()) + r(swapped * sin_signed[row * width + j].to_f32()));
                assert_eq!(got.to_bits(), want.to_bits(), "row {row} col {j}");
            }
        }
        assert_eq!(half_swap_ids(2), vec![1, 0, 3, 2]);
    }

    #[test]
    fn rope_table_values_are_bf16_exact() {
        let inv: Vec<f32> = (0..64).map(|j| (1.0f64 / 5.0e6f64.powf(2.0 * j as f64 / 128.0)) as f32).collect();
        let t = rope_table(&inv, 5);
        assert_eq!(t.len(), 5 * 128);
        assert!(t.iter().all(|&v| bf16::from_f32(v).to_f32() == v));
        assert_eq!(t[0], 1.0); // cos(0)
        assert_eq!(t[64], 0.0); // sin(0)
    }

    #[test]
    fn timestep_embedding_follows_the_official_sinusoid() {
        let e = timestep_embedding(10, 256);
        assert_eq!(e.len(), 10 * 256);
        // t = 0: sin(0) = 0 for the first half, cos(0) = 1 for the second.
        assert!(e[..128].iter().all(|v| v.to_f32() == 0.0));
        assert!(e[128..256].iter().all(|v| v.to_f32() == 1.0));
        // t = 0.1, frequency index 0 (freq = 1): sin(0.1), cos(0.1) rounded to BF16.
        let row1 = &e[256..512];
        assert_eq!(row1[0], bf16::from_f32(0.1f32.sin()));
        assert_eq!(row1[128], bf16::from_f32(0.1f32.cos()));
        // The highest frequency is 10000^-1 at index 127: sin(t * 1e-4).
        assert_eq!(row1[127], bf16::from_f32(((0.1f32 * 1e-4f32) as f64).sin() as f32));
        // Flow time grows forward: row 9 uses t = 0.9, not a reversed schedule.
        assert_eq!(e[9 * 256], bf16::from_f32(0.9f32.sin()));
    }

    #[test]
    fn expert_rope_inverse_frequencies_are_the_default_10000_geometric_series() {
        let inv = expert_inv_freq(96);
        assert_eq!(inv.len(), 48);
        assert_eq!(inv[0], 1.0);
        let last = 1.0f64 / 10000f64.powf(47.0 / 48.0);
        assert!(((inv[47] as f64) - last).abs() < 1e-6 * last);
    }

    #[test]
    fn modulation_tables_repeat_each_step_stage_vector_for_every_action_row() {
        let ids = block_modulation_ids(2, 3, 1);
        assert_eq!(ids.len(), 2 * 3 * 3);
        // step 0 stage 0..2 -> scale chunks 1, 4, 7; step 1 -> 10, 13, 16; each repeated for 3 rows.
        assert_eq!(ids, vec![1, 1, 1, 4, 4, 4, 7, 7, 7, 10, 10, 10, 13, 13, 13, 16, 16, 16]);
        assert_eq!(block_modulation_ids(1, 1, 0), vec![0, 3, 6]);
        assert_eq!(block_modulation_ids(1, 1, 2), vec![2, 5, 8]);
        assert_eq!(final_modulation_ids(2, 2, 0), vec![0, 0, 2, 2]);
        assert_eq!(final_modulation_ids(2, 2, 1), vec![1, 1, 3, 3]);
    }
}

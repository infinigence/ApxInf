//! Gated delta-rule recurrence over a whole prompt, four value rows per SIMD
//! group (EngineTailor g20) or eight (C14). Both adapt the MLX-VLM 0.7.4
//! kernel (MIT); see NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

fn check_index_extents(
    length: usize,
    key_heads: usize,
    value_heads: usize,
    key_width: usize,
    value_width: usize,
) -> Result<()> {
    for (dimensions, what) in [
        (
            [length, key_heads, key_width],
            "gated delta q/k index exceeds u32",
        ),
        (
            [length, value_heads, 1],
            "gated delta g/beta index exceeds u32",
        ),
        (
            [length, value_heads, value_width],
            "gated delta v/y index exceeds u32",
        ),
        (
            [value_heads, value_width, key_width],
            "gated delta state index exceeds u32",
        ),
    ] {
        dimensions
            .into_iter()
            .try_fold(1usize, |extent, dimension| extent.checked_mul(dimension))
            .and_then(|extent| extent.checked_sub(1))
            .filter(|&last| last <= u32::MAX as usize)
            .ok_or(Error::Contract(what))?;
    }
    Ok(())
}

/// For each value head and time step `t`, with `S` the `[Dv, Dk]` state:
/// `S = g_t S`, `S += k_t (beta_t (v_t - S k_t))^T`, `y_t = S q_t`.
/// Reductions over `Dk` use the selected layout's FP32 order.
///
/// Inputs: BF16 `q`, `k` `[1, L, Hk, Dk]`, `v` `[1, L, Hv, Dv]`, `beta`
/// `[1, L, Hv]`; FP32 `g` `[1, L, Hv]` and `state` `[1, Hv, Dv, Dk]`.
/// Returns BF16 `y` `[1, L, Hv, Dv]` and the FP32 final state.
/// Both layouts require nonempty flattened extents with a last element index
/// representable as `u32`, including q/k offsets before packed-word division.
pub struct GatedDeltaRecurrence {
    kernel: MetalKernel,
    layout: GatedDeltaLayout,
}

/// How a SIMD group splits the key reduction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatedDeltaLayout {
    /// Four value rows per group; each of 32 lanes holds `Dk / 32` keys and
    /// the reduction is one `simd_sum`, as in MLX-VLM.
    FourValues,
    /// Eight value rows per group, each lane holding two rows of `Dk / 8`
    /// keys, reduced over eight lanes by `simd_shuffle_xor`, with the next
    /// step's operands loaded ahead (C14 `s12`). Requires `Dv` a multiple of 32.
    EightLanes,
}

impl GatedDeltaRecurrence {
    pub fn new(stream: &Stream) -> Result<Self> {
        Self::with_layout(stream, GatedDeltaLayout::FourValues)
    }

    pub fn with_layout(stream: &Stream, layout: GatedDeltaLayout) -> Result<Self> {
        let (name, source) = match layout {
            GatedDeltaLayout::FourValues => (
                "apxinf_gated_delta_four_values",
                include_str!("../native/gated_delta_four_values.metal"),
            ),
            GatedDeltaLayout::EightLanes => (
                "apxinf_gated_delta_eight_lanes",
                include_str!("../native/gated_delta_eight_lanes.metal"),
            ),
        };
        Ok(Self {
            kernel: MetalKernel::new(
                stream,
                name,
                &["q", "k", "v", "g", "beta", "state", "length"],
                &["y", "next_state"],
                source,
                "",
            )?,
            layout,
        })
    }

    pub fn call(
        &self,
        q: &Array,
        k: &Array,
        v: &Array,
        g: &Array,
        beta: &Array,
        state: &Array,
    ) -> Result<(Array, Array)> {
        let [1, length, hk, dk] = *q.shape() else {
            return Err(Error::Contract("gated delta q must be [1, L, Hk, Dk]"));
        };
        let [1, vl, hv, dv] = *v.shape() else {
            return Err(Error::Contract("gated delta v must be [1, L, Hv, Dv]"));
        };
        if length == 0
            || k.shape() != q.shape()
            || vl != length
            || g.shape() != [1, length, hv]
            || beta.shape() != g.shape()
            || state.shape() != [1, hv, dv, dk]
            || [q, k, v, beta].iter().any(|a| a.dtype() != MlxDType::BF16)
            || [g, state].iter().any(|a| a.dtype() != MlxDType::F32)
            || dk == 0
            || dk % 32 != 0
            || dv == 0
            || dv % self.value_rows_per_threadgroup() != 0
            || hk == 0
            || hv % hk != 0
            || [length, dk, dv, hk, hv]
                .iter()
                .any(|&n| i32::try_from(n).is_err())
        {
            return Err(Error::Contract(
                "gated delta requires BF16 q/k/v/beta, FP32 g/state, one batch, \
                 Dk a multiple of 32, Dv a multiple of 16 (32 for eight lanes) \
                 and Hv a multiple of Hk",
            ));
        }
        check_index_extents(length, hk, hv, dk, dv)?;
        let steps = Array::from_i32(q.stream(), &[], &[length as i32])?;
        let rows = self.value_rows_per_group();
        // SAFETY: grid z enumerates the Hv value heads, y the Dv / rows value
        // groups and x the 32 lanes. Four values: each lane holds Dk / 32 state
        // columns and lane 0 writes the group's four y values. Eight lanes:
        // lane `l` holds Dk / 8 columns of rows `2 (l / 8)` and `2 (l / 8) + 1`,
        // reads keys and queries as BF16 pairs from an even offset (Dk / 8 is
        // even), and lanes with `l % 8 == 0` write their two y values; the
        // lookahead reads step `min(t + 1, L - 1)`. Every q/k/v/g/beta read is
        // at `(t * H + head) * D + i` with t < L, head < H and i < D from the
        // guarded shapes, and `hv / (Hv / Hk)` is a valid key head because Hv
        // is a multiple of Hk. Each state element is written by one lane.
        // `length` is a scalar equal to L.
        let mut out = unsafe {
            self.kernel.call(
                &[
                    q.clone(),
                    k.clone(),
                    v.clone(),
                    g.clone(),
                    beta.clone(),
                    state.clone(),
                    steps,
                ],
                &[(v.shape(), MlxDType::BF16), (state.shape(), MlxDType::F32)],
                [32, dv / rows, hv],
                [32, 4, 1],
                &[
                    ("InT", TemplateArg::DType(MlxDType::BF16)),
                    ("Dk", TemplateArg::Int(dk as i32)),
                    ("Dv", TemplateArg::Int(dv as i32)),
                    ("Hk", TemplateArg::Int(hk as i32)),
                    ("Hv", TemplateArg::Int(hv as i32)),
                ],
            )?
        };
        let state = out.pop().unwrap();
        Ok((out.pop().unwrap(), state))
    }

    fn value_rows_per_group(&self) -> usize {
        match self.layout {
            GatedDeltaLayout::FourValues => 4,
            GatedDeltaLayout::EightLanes => 8,
        }
    }

    /// Four SIMD groups per threadgroup along `Dv`.
    fn value_rows_per_threadgroup(&self) -> usize {
        4 * self.value_rows_per_group()
    }
}

#[cfg(test)]
mod tests {
    use super::check_index_extents;
    use apxinf_core::Error;

    #[test]
    fn rejects_each_wrapping_address_family() {
        for (dimensions, message) in [
            (
                [65_537, 1, 1, 65_536, 32],
                "gated delta q/k index exceeds u32",
            ),
            (
                [65_537, 1, 65_536, 32, 32],
                "gated delta g/beta index exceeds u32",
            ),
            (
                [65_537, 1, 1, 32, 65_536],
                "gated delta v/y index exceeds u32",
            ),
            (
                [1, 1, 65_537, 32, 2_048],
                "gated delta state index exceeds u32",
            ),
        ] {
            let [length, key_heads, value_heads, key_width, value_width] = dimensions;
            assert!(matches!(
                check_index_extents(length, key_heads, value_heads, key_width, value_width),
                Err(Error::Contract(actual)) if actual == message
            ));
        }
    }

    #[test]
    fn accepts_inclusive_last_u32_index() {
        assert!(check_index_extents(65_536, 1, 1, 65_536, 32).is_ok());
        assert!(check_index_extents(65_536, 1, 1, 32, 65_536).is_ok());
        assert!(check_index_extents(1, 1, 65_536, 32, 2_048).is_ok());
        assert!(check_index_extents(65_536, 1, 65_536, 1, 1).is_ok());
    }

    #[test]
    fn rejects_empty_and_host_overflow_extents() {
        for dimensions in [
            [0, 1, 1, 32, 32],
            [1, 0, 1, 32, 32],
            [1, 1, 0, 32, 32],
            [1, 1, 1, 0, 32],
            [1, 1, 1, 32, 0],
            [usize::MAX, 2, 2, 32, 32],
            [1, 1, usize::MAX, 32, 32],
        ] {
            let [length, key_heads, value_heads, key_width, value_width] = dimensions;
            assert!(
                check_index_extents(length, key_heads, value_heads, key_width, value_width)
                    .is_err()
            );
        }
    }

    #[test]
    fn accepts_small_and_clef_profiles() {
        for length in [1, 5, 37, 4_096] {
            assert!(check_index_extents(length, 2, 4, 32, 16).is_ok());
            assert!(check_index_extents(length, 2, 4, 32, 32).is_ok());
            assert!(check_index_extents(length, 16, 32, 128, 128).is_ok());
        }
    }
}

//! Gated delta-rule recurrence over a whole prompt, four value rows per SIMD
//! group (EngineTailor g20) or eight (C14). Both adapt the MLX-VLM 0.7.4
//! kernel (MIT); see NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream, TemplateArg};
use apxinf_core::{Error, Result};

/// For each value head and time step `t`, with `S` the `[Dv, Dk]` state:
/// `S = g_t S`, `S += k_t (beta_t (v_t - S k_t))^T`, `y_t = S q_t`.
/// Reductions over `Dk` use the selected layout's FP32 order.
///
/// Inputs: BF16 `q`, `k` `[1, L, Hk, Dk]`, `v` `[1, L, Hv, Dv]`, `beta`
/// `[1, L, Hv]`; FP32 `g` `[1, L, Hv]` and `state` `[1, Hv, Dv, Dk]`.
/// Returns BF16 `y` `[1, L, Hv, Dv]` and the FP32 final state.
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

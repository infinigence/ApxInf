//! BF16 SwiGLU MLP for one or two rows at hidden width 1024 and inner width
//! 3072, with FP32 accumulation. Adapted from EngineTailor (MIT); see NOTICE.
use crate::{Array, MetalKernel, MlxDType, Stream};
use apxinf_core::{Error, Result};

const HIDDEN: usize = 1024;
const INNER: usize = 3072;

/// `down(silu(gate(x)) * up(x))` for `x` `[1, M, 1024]`, `M` 1 or 2. Gate and
/// up accumulate in FP32 and round to BF16; SiLU runs in FP32 with a precise
/// exponential, rounds to BF16 and multiplies `up` in BF16. One row uses
/// row4-packed weights for all three projections; two rows run the fused
/// gate/up kernel and an MLX matmul for `down`. Reduction order differs from
/// MLX matmul, so results can differ from the composed operators in the last
/// bits.
pub struct SwigluMlpGemv {
    gate_up_m1: MetalKernel,
    down_m1: MetalKernel,
    gate_up_m2: MetalKernel,
}

/// Original and row4-packed weights of one MLP.
pub struct SwigluWeights {
    gate: Array,
    up: Array,
    down: Array,
    /// `[768, 1024, 8]`: four gate rows then the same four up rows per k.
    packed_gate_up: Array,
    /// `[256, 3072, 4]`: four down rows per k.
    packed_down: Array,
}

impl SwigluMlpGemv {
    pub fn new(stream: &Stream) -> Result<Self> {
        let header = include_str!("../native/swiglu_activation.h");
        Ok(Self {
            gate_up_m1: MetalKernel::new(
                stream,
                "apxinf_swiglu_gate_up_m1_row4_packed",
                &["packed_gate_up", "x"],
                &["activation"],
                include_str!("../native/swiglu_gate_up_m1_packed.metal"),
                header,
            )?,
            down_m1: MetalKernel::new(
                stream,
                "apxinf_swiglu_down_m1_row4_packed",
                &["packed_down", "activation"],
                &["output"],
                include_str!("../native/swiglu_down_m1_packed.metal"),
                "",
            )?,
            gate_up_m2: MetalKernel::new(
                stream,
                "apxinf_swiglu_gate_up_m2",
                &["w_gate", "w_up", "x"],
                &["activation"],
                include_str!("../native/swiglu_gate_up_m2.metal"),
                header,
            )?,
        })
    }

    /// Check BF16 gate/up `[3072, 1024]` and down `[1024, 3072]` and build the
    /// packed copies (evaluated; about 19 MB per MLP).
    pub fn prepare(&self, gate: &Array, up: &Array, down: &Array) -> Result<SwigluWeights> {
        if [gate, up, down].iter().any(|w| w.dtype() != MlxDType::BF16)
            || gate.shape() != [INNER, HIDDEN]
            || up.shape() != [INNER, HIDDEN]
            || down.shape() != [HIDDEN, INNER]
        {
            return Err(Error::Contract(
                "SwiGLU GEMV requires BF16 gate/up [3072, 1024] and down [1024, 3072]",
            ));
        }
        let row4 =
            |w: &Array, rows: usize, k: usize| w.reshape(&[rows / 4, 4, k])?.transpose(&[0, 2, 1]);
        let packed_gate_up =
            Array::concat(&[&row4(gate, INNER, HIDDEN)?, &row4(up, INNER, HIDDEN)?], 2)?;
        let packed_down = row4(down, HIDDEN, INNER)?.contiguous()?;
        gate.stream()
            .eval(&[packed_gate_up.clone(), packed_down.clone()])?;
        Ok(SwigluWeights {
            gate: gate.clone(),
            up: up.clone(),
            down: down.clone(),
            packed_gate_up,
            packed_down,
        })
    }

    pub fn call(&self, x: &Array, w: &SwigluWeights) -> Result<Array> {
        let rows = match *x.shape() {
            [1, rows @ (1 | 2), HIDDEN] if x.dtype() == MlxDType::BF16 => rows,
            _ => {
                return Err(Error::Contract(
                    "SwiGLU GEMV input must be BF16 [1, 1 or 2, 1024]",
                ))
            }
        };
        let activation = [1, rows, INNER];
        if rows == 2 {
            // SAFETY: inputs are row-contiguous. Each of the 768 groups of
            // four simdgroups owns rows `4 * group.y + simdgroup < 3072`; lanes
            // read vec4 elements `< 1024 / 4` of their gate/up row and `< 2 *
            // 1024 / 4` of x. Lane 0 alone writes `row` and `3072 + row` of the
            // [1, 2, 3072] activation; rows do not overlap.
            let out = unsafe {
                self.gate_up_m2.call(
                    &[w.gate.clone(), w.up.clone(), x.clone()],
                    &[(&activation, MlxDType::BF16)],
                    [32, INNER, 1],
                    [32, 4, 1],
                    &[],
                )?
            };
            return out[0].matmul(&w.down.transpose(&[1, 0])?);
        }
        // SAFETY: inputs are row-contiguous. Each of 192 groups of four
        // simdgroups owns row quad `4 * group.x + simdgroup < 768`; lanes read
        // packed vec4 elements `(quad * 1024 + k) * 2 (+1)` with k < 1024,
        // inside [768, 1024, 8], and x[k]. Lane 0 alone writes activation rows
        // `4 * quad .. 4 * quad + 3 < 3072`; quads do not overlap.
        let act = unsafe {
            self.gate_up_m1.call(
                &[w.packed_gate_up.clone(), x.clone()],
                &[(&activation, MlxDType::BF16)],
                [2 * INNER, 1, 4],
                [32, 1, 4],
                &[],
            )?
        };
        // SAFETY: as above for 64 groups and row quads `< 256` of [256, 3072, 4]
        // with k < 3072 into the [1, 1, 3072] activation; lane 0 writes output
        // rows `4 * quad .. 4 * quad + 3 < 1024`.
        let out = unsafe {
            self.down_m1.call(
                &[w.packed_down.clone(), act[0].clone()],
                &[(&[1, 1, HIDDEN], MlxDType::BF16)],
                [2 * HIDDEN, 1, 4],
                [32, 1, 4],
                &[],
            )?
        };
        Ok(out.into_iter().next().unwrap())
    }
}

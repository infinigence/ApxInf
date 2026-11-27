//! Affine-quantized decode copy of a projection (`[out, in]` packing, one-row lane).
use crate::Array;
use apxinf_core::Result;

/// Group size of the affine decode copies.
pub const GROUP: i32 = 64;

#[derive(Clone)]
pub struct Quantized {
    packed: Array,
    scales: Array,
    biases: Array,
    bits: i32,
    out: usize,
    input: usize,
}

impl Quantized {
    /// Builds the decode copy from the checkpoint-layout `[out, in]` weight.
    pub fn new(hf_weight: &Array, bits: i32) -> Result<Self> {
        let shape = hf_weight.shape().to_vec();
        let [packed, scales, biases] = hf_weight.quantize(GROUP, bits)?;
        Ok(Self {
            packed,
            scales,
            biases,
            bits,
            out: shape[0],
            input: shape[1],
        })
    }

    /// `[rows, in]` BF16/BF32 input → `[rows, out]`, one row per call.
    pub fn project(&self, x: &Array) -> Result<Array> {
        if x.shape().last() != Some(&self.input) {
            return Err(apxinf_core::Error::Contract(
                "quantized projection input width mismatch".into(),
            ));
        }
        x.quantized_matmul(&self.packed, &self.scales, &self.biases, true, GROUP, self.bits)
    }

    pub fn bits(&self) -> i32 {
        self.bits
    }

    pub fn arrays(&self) -> Vec<Array> {
        vec![self.packed.clone(), self.scales.clone(), self.biases.clone()]
    }
}

//! Model-neutral token sampling and device random-number contracts.
//!
//! The categorical sampler is shared by autoregressive text, vision-language,
//! and future action-token models. Continuous VLA model logic remains outside
//! this module; the only continuous primitive exposed here is a reusable
//! standard-normal generator for model-owned latent buffers.

use crate::{Error, Result, Tensor};

/// How the next categorical token is selected.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenSelection {
    /// Select the maximum adjusted logit. Ties resolve to the lowest token ID.
    Greedy,
    /// Sample from temperature-scaled logits after top-k and top-p filtering.
    Random {
        temperature: f32,
        top_k: Option<usize>,
        top_p: f32,
    },
}

impl Default for TokenSelection {
    fn default() -> Self {
        Self::Greedy
    }
}

/// History-dependent logit penalties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenPenalties {
    /// Multiplicative/divisive repetition penalty. `1.0` disables it.
    pub repetition: f32,
    /// Subtract `frequency * occurrence_count` from seen tokens.
    pub frequency: f32,
    /// Subtract `presence` once from every seen token.
    pub presence: f32,
}

impl Default for TokenPenalties {
    fn default() -> Self {
        Self {
            repetition: 1.0,
            frequency: 0.0,
            presence: 0.0,
        }
    }
}

/// Complete categorical sampling policy for one sequence.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TokenSamplingParams {
    pub selection: TokenSelection,
    pub penalties: TokenPenalties,
    /// Return the selected token's log-probability after filtering.
    pub return_logprob: bool,
}

impl TokenSamplingParams {
    pub fn greedy() -> Self {
        Self::default()
    }

    pub fn validate(&self, vocab_size: usize) -> Result<()> {
        if vocab_size == 0 || vocab_size > u32::MAX as usize {
            return Err(Error::Other(format!(
                "token sampling requires vocabulary size in 1..={}, got {vocab_size}",
                u32::MAX
            )));
        }
        let penalties = self.penalties;
        if !penalties.repetition.is_finite() || penalties.repetition <= 0.0 {
            return Err(Error::Other(
                "repetition penalty must be finite and greater than zero".into(),
            ));
        }
        if !penalties.frequency.is_finite() || !penalties.presence.is_finite() {
            return Err(Error::Other(
                "frequency and presence penalties must be finite".into(),
            ));
        }
        if let TokenSelection::Random {
            temperature,
            top_k,
            top_p,
        } = self.selection
        {
            if !temperature.is_finite() || temperature <= 0.0 {
                return Err(Error::Other(
                    "sampling temperature must be finite and greater than zero".into(),
                ));
            }
            if !top_p.is_finite() || !(0.0 < top_p && top_p <= 1.0) {
                return Err(Error::Other(
                    "top-p must be finite and in the interval (0, 1]".into(),
                ));
            }
            if let Some(top_k) = top_k {
                if top_k == 0 || top_k > vocab_size {
                    return Err(Error::Other(format!(
                        "top-k must be in 1..={vocab_size}, got {top_k}"
                    )));
                }
            }
        }
        Ok(())
    }
}

/// Counter-based random stream identity.
///
/// A random value is a pure function of this key and an element index. This
/// makes results independent of batch order and avoids process-global RNG
/// state. `sequence` identifies a logical request; `draw` identifies a step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RngKey {
    pub seed: u64,
    pub sequence: u64,
    pub draw: u64,
}

impl RngKey {
    pub const fn new(seed: u64, sequence: u64, draw: u64) -> Self {
        Self {
            seed,
            sequence,
            draw,
        }
    }

    pub fn advance(&mut self) -> Result<()> {
        self.draw = self
            .draw
            .checked_add(1)
            .ok_or_else(|| Error::Other("sampling RNG draw counter overflow".into()))?;
        Ok(())
    }
}

/// Fixed allocation contract for a categorical sampler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenSamplingSpec {
    pub vocab_size: usize,
    /// Maximum prompt plus generated-token history retained by the sampler.
    pub max_sequence_len: usize,
}

impl TokenSamplingSpec {
    pub fn validate(self) -> Result<()> {
        if self.vocab_size == 0 || self.vocab_size > u32::MAX as usize {
            return Err(Error::Other(format!(
                "token sampling requires vocabulary size in 1..={}, got {}",
                u32::MAX,
                self.vocab_size
            )));
        }
        if self.max_sequence_len == 0 {
            return Err(Error::Other(
                "token sampling max_sequence_len must be greater than zero".into(),
            ));
        }
        Ok(())
    }
}

/// Per-request sampler initialization.
#[derive(Clone, Copy, Debug)]
pub struct TokenSamplingInit<'a> {
    pub prompt_token_ids: &'a [u32],
    pub params: &'a TokenSamplingParams,
    pub rng: RngKey,
}

/// A contiguous vocabulary row inside a model logits tensor.
#[derive(Clone, Copy, Debug)]
pub struct NextTokenLogits<'a> {
    tensor: &'a Tensor,
    row: usize,
    vocab_size: usize,
}

impl<'a> NextTokenLogits<'a> {
    /// Select an explicit row from a tensor whose final dimension is vocab.
    pub fn row(tensor: &'a Tensor, row: usize, vocab_size: usize) -> Result<Self> {
        let actual_vocab = tensor.shape().dims().last().copied().ok_or_else(|| {
            Error::Other("next-token logits must have at least one dimension".into())
        })?;
        if actual_vocab != vocab_size {
            return Err(Error::ShapeMismatch {
                expected: format!("[..., {vocab_size}]"),
                got: tensor.shape().to_string(),
            });
        }
        let rows = tensor.numel() / vocab_size;
        if row >= rows {
            return Err(Error::Other(format!(
                "next-token logits row {row} is out of range for {rows} rows"
            )));
        }
        Ok(Self {
            tensor,
            row,
            vocab_size,
        })
    }

    /// Select the final row, which is the next-token distribution after
    /// prefill or a single-token decode.
    pub fn last(tensor: &'a Tensor, vocab_size: usize) -> Result<Self> {
        let actual_vocab = tensor.shape().dims().last().copied().ok_or_else(|| {
            Error::Other("next-token logits must have at least one dimension".into())
        })?;
        if actual_vocab != vocab_size {
            return Err(Error::ShapeMismatch {
                expected: format!("[..., {vocab_size}]"),
                got: tensor.shape().to_string(),
            });
        }
        let rows = tensor.numel() / vocab_size;
        if rows == 0 {
            return Err(Error::Other("next-token logits contain no rows".into()));
        }
        Self::row(tensor, rows - 1, vocab_size)
    }

    pub fn tensor(self) -> &'a Tensor {
        self.tensor
    }

    pub fn row_index(self) -> usize {
        self.row
    }

    pub fn row_offset(self) -> usize {
        self.row * self.vocab_size
    }

    pub fn vocab_size(self) -> usize {
        self.vocab_size
    }
}

/// Result of one token-sampling step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TokenSample {
    pub token_id: u32,
    pub logprob: Option<f32>,
}

/// Stateful categorical sampler. Implementations own history counts, RNG
/// state, output buffers, and backend-specific fixed workspaces.
pub trait TokenSampler {
    fn spec(&self) -> TokenSamplingSpec;
    fn begin(&mut self, init: TokenSamplingInit<'_>) -> Result<()>;
    fn sample(&mut self, logits: NextTokenLogits<'_>) -> Result<TokenSample>;
}

/// Reusable standard-normal generator bound to one stable output tensor.
///
/// CUDA implementations fill the tensor in place on their compute stream;
/// CPU implementations mutate owned CPU storage. The returned tensor remains
/// valid for the generator lifetime and is overwritten by the next call.
pub trait NormalGenerator {
    fn output(&self) -> &Tensor;
    fn generate(&mut self, rng: RngKey) -> Result<&Tensor>;
}

/// Sampling facilities implemented by every compute backend.
pub trait SamplingBackend {
    fn create_token_sampler(&self, spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>>;

    /// Bind a reusable standard-normal generator to `output`. Implementations
    /// validate that the tensor belongs to this backend and has a supported
    /// floating-point dtype.
    fn create_normal_generator(&self, output: Tensor) -> Result<Box<dyn NormalGenerator>>;
}

/// Deterministic Philox4x32-10 primitive used by CPU and CUDA samplers.
pub fn philox4x32_10(mut counter: [u32; 4], mut key: [u32; 2]) -> [u32; 4] {
    const M0: u32 = 0xd251_1f53;
    const M1: u32 = 0xcd9e_8d57;
    const W0: u32 = 0x9e37_79b9;
    const W1: u32 = 0xbb67_ae85;
    for _ in 0..10 {
        let p0 = u64::from(M0) * u64::from(counter[0]);
        let p1 = u64::from(M1) * u64::from(counter[2]);
        counter = [
            (p1 >> 32) as u32 ^ counter[1] ^ key[0],
            p1 as u32,
            (p0 >> 32) as u32 ^ counter[3] ^ key[1],
            p0 as u32,
        ];
        key[0] = key[0].wrapping_add(W0);
        key[1] = key[1].wrapping_add(W1);
    }
    counter
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn rng_words(rng: RngKey, element_group: u64) -> [u32; 4] {
    let stream = splitmix64(rng.sequence ^ rng.draw.rotate_left(32));
    philox4x32_10(
        [
            element_group as u32,
            (element_group >> 32) as u32,
            stream as u32,
            (stream >> 32) as u32,
        ],
        [rng.seed as u32, (rng.seed >> 32) as u32],
    )
}

fn unit_open(word: u32) -> f32 {
    // Use 23 random mantissa bits. Clamp zero to the smallest selected step,
    // keeping the result strictly inside (0, 1) without rounding 2^32-1 to 1.
    let value = f32::from_bits(0x3f80_0000 | (word >> 9)) - 1.0;
    value.max(f32::from_bits(0x3380_0000))
}

pub fn uniform_f32(rng: RngKey, element: u64) -> f32 {
    let words = rng_words(rng, element / 4);
    unit_open(words[(element % 4) as usize])
}

/// Generate deterministic standard-normal f32 values with Box-Muller.
pub fn standard_normal_f32(count: usize, rng: RngKey) -> Vec<f32> {
    let mut output = Vec::with_capacity(count);
    for pair in 0..count.div_ceil(2) {
        let words = rng_words(rng, pair as u64);
        let u1 = unit_open(words[0]);
        let u2 = unit_open(words[1]);
        let radius = (-2.0 * u1.ln()).sqrt();
        let angle = std::f32::consts::TAU * u2;
        output.push(radius * angle.cos());
        if output.len() < count {
            output.push(radius * angle.sin());
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_matches_random123_zero_vector() {
        assert_eq!(
            philox4x32_10([0; 4], [0; 2]),
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
    }

    #[test]
    fn next_token_logits_selects_last_row() {
        let tensor = Tensor::from_f32(vec![2, 3], &[9.0, 8.0, 7.0, 1.0, 3.0, 2.0]).unwrap();
        let logits = NextTokenLogits::last(&tensor, 3).unwrap();
        assert_eq!(logits.row_index(), 1);
        assert_eq!(logits.row_offset(), 3);
        assert!(NextTokenLogits::last(&tensor, 4).is_err());
    }

}

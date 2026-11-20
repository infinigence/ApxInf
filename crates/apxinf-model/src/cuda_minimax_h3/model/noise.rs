//! Seed-compatible initial noise. ComfyUI (`comfy.sample.prepare_noise`) and the diffusers pipeline draw the starting
//! latent with `torch.manual_seed(seed); torch.randn(shape)` on the CPU, so a request seed reproduces the composition the
//! reference implementations produce only if this module draws the same numbers: a 32-bit Mersenne Twister with torch's
//! seeding, 24-bit uniforms, and torch's Box-Muller in blocks of 16 (`normal_fill` in ATen's `DistributionTemplates.h`).
//! torch's AVX2 build evaluates log/cos/sin with its own vector approximations; those agree with libm to about 1e-7 relative.

const N: usize = 624;
const M: usize = 397;

pub(crate) struct Mt19937 {
    state: [u32; N],
    index: usize,
}

impl Mt19937 {
    /// `at::mt19937(seed)`: only the low 32 bits of the seed are used.
    pub fn new(seed: u64) -> Self {
        let mut state = [0u32; N];
        state[0] = seed as u32;
        for i in 1..N {
            state[i] = 1_812_433_253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: N }
    }

    fn refill(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % N] & 0x7fff_ffff);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= 0x9908_b0df;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.refill();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `uniform_real_distribution<float>(0, 1)`: the low 24 bits of one draw, scaled by 2^-24.
    fn uniform(&mut self) -> f32 {
        (self.next_u32() & ((1 << 24) - 1)) as f32 * (1.0 / (1u32 << 24) as f32)
    }
}

/// `normal_fill_16`: Box-Muller over 8 pairs `(data[j], data[j + 8])`, in place.
fn normal_fill_16(data: &mut [f32]) {
    for j in 0..8 {
        let u1 = 1.0 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        data[j] = radius * theta.cos();
        data[j + 8] = radius * theta.sin();
    }
}

/// A torch CPU generator seeded with `manual_seed(seed)`. ComfyUI draws the noise of every tensor of a nested latent in order from
/// one generator (`comfy.sample.prepare_noise`), so the audio noise is the continuation of the video noise's stream, not a fresh
/// draw from the same seed.
pub(crate) struct Generator {
    rng: Mt19937,
}

impl Generator {
    pub fn new(seed: u64) -> Self {
        Self {
            rng: Mt19937::new(seed),
        }
    }

    /// `torch.randn(n, generator=self)` for a contiguous float32 tensor of `n >= 16` elements. Consumes `n` draws, or `n + 16`
    /// when `n` is not a multiple of 16 (torch recomputes the last 16 values from fresh draws).
    pub fn randn(&mut self, n: usize) -> Vec<f32> {
        assert!(
            n >= 16,
            "torch uses a different (scalar) path below 16 elements"
        );
        let rng = &mut self.rng;
        let mut data: Vec<f32> = (0..n).map(|_| rng.uniform()).collect();
        for i in (0..n - 15).step_by(16) {
            normal_fill_16(&mut data[i..i + 16]);
        }
        if n % 16 != 0 {
            let tail = &mut data[n - 16..];
            for value in tail.iter_mut() {
                *value = rng.uniform();
            }
            normal_fill_16(tail);
        }
        data
    }
}

/// `torch.randn(n, generator=torch.Generator().manual_seed(seed))` for a contiguous float32 tensor of `n >= 16` elements.
#[cfg(test)]
pub(crate) fn randn(seed: u64, n: usize) -> Vec<f32> {
    Generator::new(seed).randn(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mersenne_twister_matches_the_reference_sequence() {
        // mt19937 seeded with 5489 (the C++ default) produces 3499211612 first and 10000th value 4123659995
        let mut rng = Mt19937::new(5489);
        assert_eq!(rng.next_u32(), 3_499_211_612);
        let mut last = 0;
        for _ in 1..10_000 {
            last = rng.next_u32();
        }
        assert_eq!(last, 4_123_659_995);
        // seeding uses the low 32 bits only
        assert_eq!(
            Mt19937::new(42).next_u32(),
            Mt19937::new(42 + (7 << 32)).next_u32()
        );
    }

    #[test]
    fn randn_is_standard_normal_and_seed_deterministic() {
        let a = randn(42, 98_304);
        assert_eq!(a, randn(42, 98_304));
        assert_ne!(a, randn(43, 98_304));
        let mean = a.iter().map(|v| *v as f64).sum::<f64>() / a.len() as f64;
        let var = a.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / a.len() as f64;
        assert!(
            mean.abs() < 0.02 && (var - 1.0).abs() < 0.03,
            "mean {mean} var {var}"
        );
        // a length that is not a multiple of 16 recomputes the last block
        let b = randn(42, 100);
        assert_eq!(b.len(), 100);
        assert_eq!(
            b[..84],
            randn(42, 84 + 16)[..84],
            "values before the recomputed tail are unchanged"
        );
        assert_ne!(b[84..], randn(42, 100 + 12)[84..100]);
    }

    #[test]
    fn a_generator_continues_its_stream_across_draws() {
        // two draws of one generator are not two fresh draws: the second starts where the first stopped
        let mut generator = Generator::new(7);
        let (first, second) = (generator.randn(64), generator.randn(32));
        assert_eq!(first, randn(7, 64));
        assert_ne!(second, randn(7, 32));
        // 64 draws were consumed, so the second tensor equals the tail of one 96-element draw
        assert_eq!(second, randn(7, 96)[64..]);
        // a length that is not a multiple of 16 consumes 16 extra draws
        let mut generator = Generator::new(7);
        let _ = generator.randn(20);
        let continued = generator.randn(16);
        let mut reference = Mt19937::new(7);
        for _ in 0..36 {
            reference.next_u32();
        }
        let mut expected: Vec<f32> = (0..16).map(|_| reference.uniform()).collect();
        normal_fill_16(&mut expected);
        assert_eq!(continued, expected);
    }
}

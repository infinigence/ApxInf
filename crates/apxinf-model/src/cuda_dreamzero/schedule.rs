//! Fixed flow UniPC order-two bh2 schedule, computed before device execution.
use apxinf_core::{Error, Result};

#[derive(Clone, Copy, Debug)]
pub(super) struct Update {
    pub ratio: f32,
    pub weight: f32,
    pub divisor: Option<f32>,
    pub rho_old: f32,
    pub rho_current: f32,
}
#[derive(Clone, Copy, Debug)]
pub(super) struct Step {
    pub sigma: f32,
    pub timestep: i64,
    pub corrector: Option<Update>,
    pub predictor: Update,
}
#[derive(Clone, Debug)]
pub(super) struct Schedule {
    pub steps: Vec<Step>,
}

fn lambda(sigma: f32) -> f32 {
    (1. - sigma).ln() - sigma.ln()
}
fn bf16(x: f32) -> f32 {
    half::bf16::from_f32(x).to_f32()
}

impl Schedule {
    pub fn standard() -> Result<Self> {
        // Official constructor stores sigma_max in FP32, then NumPy builds
        // and shifts the inference linspace in FP64 before casting sigmas.
        let maximum = f64::from(1f32 - 0.001f32);
        let mut sigmas = Vec::with_capacity(17);
        let mut times = Vec::with_capacity(16);
        for i in 0..16 {
            let x = maximum + i as f64 * (-maximum / 16.);
            let shifted = 5. * x / (1. + 4. * x);
            sigmas.push(shifted as f32);
            times.push((shifted * 1000.) as i64);
        }
        sigmas.push(0.);
        Self::from_sigmas(&sigmas, &times)
    }

    /// Network code is independent of the two modality schedules. Admission
    /// of an additional trained profile is the loader's responsibility.
    pub fn from_sigmas(sigmas: &[f32], timesteps: &[i64]) -> Result<Self> {
        if timesteps.len() < 2
            || sigmas.len() != timesteps.len() + 1
            || sigmas
                .iter()
                .any(|s| !s.is_finite() || !(0. ..1.).contains(s))
            || sigmas.windows(2).any(|s| s[0] <= s[1])
            || timesteps.iter().any(|t| *t < 0 || *t > 1000)
        {
            return Err(Error::Other(
                "invalid DreamZero flow sigma/time schedule".into(),
            ));
        }
        let mut steps = Vec::with_capacity(timesteps.len());
        for i in 0..timesteps.len() {
            let transition = |target: usize, source: usize| {
                let h = lambda(sigmas[target]) - lambda(sigmas[source]);
                let phi = (-h).exp_m1();
                (
                    h,
                    phi,
                    Update {
                        ratio: sigmas[target] / sigmas[source],
                        weight: (1. - sigmas[target]) * phi,
                        divisor: None,
                        rho_old: 0.,
                        rho_current: 0.,
                    },
                )
            };
            let corrector = if i == 0 {
                None
            } else {
                let (h, phi, mut c) = transition(i, i - 1);
                if i == 1 {
                    c.rho_current = 0.5;
                } else {
                    let rk = (lambda(sigmas[i - 2]) - lambda(sigmas[i - 1])) / h;
                    let h_phi_k = phi / (-h) - 1.;
                    let b0 = h_phi_k / phi;
                    let b1 = (h_phi_k / (-h) - 0.5) * 2. / phi;
                    let old = (b1 - b0) / (rk - 1.);
                    c.divisor = Some(rk);
                    c.rho_old = bf16(old);
                    c.rho_current = bf16(b0 - old);
                }
                Some(c)
            };
            let (h, _, mut predictor) = transition(i + 1, i);
            if i > 0 && i + 1 < timesteps.len() {
                predictor.divisor = Some((lambda(sigmas[i - 1]) - lambda(sigmas[i])) / h);
                predictor.rho_old = 0.5;
            }
            for u in corrector.iter().chain(std::iter::once(&predictor)) {
                if !u.ratio.is_finite()
                    || !u.weight.is_finite()
                    || !u.rho_old.is_finite()
                    || !u.rho_current.is_finite()
                    || u.divisor.is_some_and(|d| !d.is_finite() || d == 0.)
                {
                    return Err(Error::Other(
                        "nonfinite DreamZero UniPC coefficients".into(),
                    ));
                }
            }
            steps.push(Step {
                sigma: sigmas[i],
                timestep: timesteps[i],
                corrector,
                predictor,
            });
        }
        Ok(Self { steps })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn standard_schedule_uses_warmup_and_lower_order_final() {
        let s = Schedule::standard().unwrap();
        assert_eq!(s.steps.len(), 16);
        assert_eq!(s.steps[0].timestep, 999);
        assert!(s.steps[0].corrector.is_none());
        assert!(s.steps[0].predictor.divisor.is_none());
        assert!(s.steps[1].corrector.unwrap().divisor.is_none());
        assert!(s.steps[2].corrector.unwrap().divisor.is_some());
        assert!(s.steps[15].predictor.divisor.is_none());
        assert_eq!(s.steps[15].predictor.ratio, 0.);
        assert_eq!(s.steps[15].predictor.weight, -1.);
        assert!(s.steps.windows(2).all(|x| x[0].sigma > x[1].sigma));
    }
    #[test]
    fn invalid_sigma_schedules_fail_before_device_work() {
        for sigmas in [
            [1., 0.5, 0.],
            [0.9, 0.9, 0.],
            [0.9, f32::NAN, 0.],
            [0.9, 0., 0.],
        ] {
            assert!(Schedule::from_sigmas(&sigmas, &[900, 500]).is_err());
        }
    }
}

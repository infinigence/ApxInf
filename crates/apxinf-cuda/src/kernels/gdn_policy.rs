//! Launch policy for the gated-delta-net kernels.
//!
//! These constants are not hardware facts, so they do not belong in
//! [`CudaDeviceCaps`], whose contract is that it carries hardware only. They
//! are also not portable, so they do not belong hard-coded in the CUDA
//! adapter: every one of them was swept on a board and every one of them came
//! out different on the next board. Orin wants a chunk-state tile of 8 and a
//! chunk-gemm tile of 32; Thor wants 4 and 4, and its chunk-state curve is not
//! merely shifted but steeper -- 16 is 8% off the optimum on Orin and 26% off
//! here.
//!
//! So they live here: one table, chosen from the device, with a per-field
//! environment override so a new board can be re-swept without a rebuild. The
//! adapter does what it is told and asks the driver nothing.
//!
//! Re-sweep after changing any of these kernels. The optimum has moved once
//! already for the chunk-state block width, and it will move again.

use crate::device_caps::{CudaArchFamily, CudaDeviceCaps};

/// Which form of a kernel that has a tensor-core implementation to run.
///
/// The split forms carry an fp32 operand as two or three BF16 terms so that
/// every partial product is exact; see the kernels for what each costs against
/// an fp64 reference.
pub mod wmma {
    /// The scalar fp32 kernel.
    pub const OFF: i32 = 0;
    /// One BF16 pass: the fp32 operand is rounded.
    pub const LOSSY: i32 = 1;
    /// Two BF16 terms, about 2^-16 relative.
    pub const SPLIT2: i32 = 2;
    /// Three BF16 terms, about 2^-24, which is fp32's own resolution.
    pub const SPLIT3: i32 = 3;
}

/// Launch constants handed to the GDN adapters. Plain `i32` fields in a
/// `#[repr(C)]` struct, so the same declaration serves both sides of the ABI.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnLaunchPolicy {
    /// Cells accumulated per thread in the chunk-state kernel.
    pub chunk_state_tile: i32,
    /// Block width for the chunk-state kernel.
    pub chunk_state_threads: i32,
    /// Which form of the chunk-state scan to run; see [`wmma`].
    pub chunk_state_wmma: i32,
    /// Columns accumulated per thread in the chunk-gemm kernel.
    pub chunk_gemm_tile: i32,
    /// Which form of the chunk GEMM to run; see [`wmma`].
    pub chunk_gemm_wmma: i32,
    /// Which form of the raw-attention term to run; see [`wmma`].
    pub attn_raw_wmma: i32,
    /// Threads per output column in the decode recurrence.
    pub recurrent_split: i32,
}

impl GdnLaunchPolicy {
    /// The measured defaults for `caps`, then any environment overrides.
    pub fn for_device(caps: &CudaDeviceCaps) -> Self {
        let mut policy = Self::defaults_for(caps.arch_family);
        policy.apply_overrides();
        policy
    }

    /// Defaults by architecture family, each swept on a board of that family.
    ///
    /// sm80 family (Orin sm_87, RTX 4090 sm_89): the values #72 measured, with
    /// the tensor-core forms off -- those kernels need an SM100-family tensor
    /// core to be worth their extra passes.
    ///
    /// sm100 family (Thor sm_110): chunk-state tile 4 rather than 8 and
    /// chunk-gemm tile 4 rather than 32, both re-swept here; the chunk-state
    /// scan on tensor cores, which is 21.9% of the fixed cost at an operator
    /// error of 1.422947e-3 against the scalar form's 1.418808e-3; and four
    /// threads per output column in the decode recurrence, which takes it from
    /// 39.383 to 37.373 ms/token.
    ///
    /// The other two tensor-core forms are built and measured but default off:
    /// each is worth under 2% and each moves the end-to-end probe. See their
    /// kernels for the numbers.
    pub fn defaults_for(family: CudaArchFamily) -> Self {
        match family {
            CudaArchFamily::Sm100 => Self {
                chunk_state_tile: 4,
                chunk_state_threads: 1024,
                chunk_state_wmma: wmma::SPLIT2,
                chunk_gemm_tile: 4,
                chunk_gemm_wmma: wmma::OFF,
                attn_raw_wmma: wmma::OFF,
                recurrent_split: 4,
            },
            CudaArchFamily::Sm80 | CudaArchFamily::Other(_) => Self {
                chunk_state_tile: 8,
                chunk_state_threads: 1024,
                chunk_state_wmma: wmma::OFF,
                chunk_gemm_tile: 32,
                chunk_gemm_wmma: wmma::OFF,
                attn_raw_wmma: wmma::OFF,
                recurrent_split: 1,
            },
        }
    }

    fn apply_overrides(&mut self) {
        override_from("APXINF_GDN_CHUNK_TILE", &mut self.chunk_state_tile, &[1, 2, 4, 8, 16]);
        override_from(
            "APXINF_GDN_CHUNK_STATE_THREADS",
            &mut self.chunk_state_threads,
            &[128, 256, 512, 1024],
        );
        override_wmma("APXINF_GDN_CHUNK_STATE_WMMA", &mut self.chunk_state_wmma);
        override_from(
            "APXINF_GDN_CHUNK_GEMM_TILE",
            &mut self.chunk_gemm_tile,
            &[1, 2, 4, 8, 16, 32],
        );
        override_wmma("APXINF_GDN_CHUNK_GEMM_WMMA", &mut self.chunk_gemm_wmma);
        override_wmma("APXINF_GDN_ATTN_RAW_WMMA", &mut self.attn_raw_wmma);
        override_from("APXINF_GDN_RECURRENT_SPLIT", &mut self.recurrent_split, &[1, 2, 4]);
    }
}

/// Accept only values the kernels are instantiated for; anything else is
/// ignored rather than passed through to a launch that would fail.
fn override_from(name: &str, slot: &mut i32, allowed: &[i32]) {
    if let Some(value) = std::env::var(name).ok().and_then(|v| v.trim().parse::<i32>().ok()) {
        if allowed.contains(&value) {
            *slot = value;
        }
    }
}

/// `0`/`off`/`false` for the scalar kernel, `lossy` for one BF16 pass, or the
/// number of BF16 terms.
fn override_wmma(name: &str, slot: &mut i32) {
    let Ok(raw) = std::env::var(name) else { return };
    let value = raw.trim();
    *slot = match value {
        "0" | "off" | "false" => wmma::OFF,
        "lossy" | "1" => wmma::LOSSY,
        "2" => wmma::SPLIT2,
        "3" | "on" | "true" => wmma::SPLIT3,
        _ => return,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_families_get_their_own_sweeps() {
        let thor = GdnLaunchPolicy::defaults_for(CudaArchFamily::Sm100);
        let orin = GdnLaunchPolicy::defaults_for(CudaArchFamily::Sm80);
        assert_eq!((thor.chunk_state_tile, thor.chunk_gemm_tile), (4, 4));
        assert_eq!((orin.chunk_state_tile, orin.chunk_gemm_tile), (8, 32));
        assert_eq!(thor.recurrent_split, 4);
        assert_eq!(orin.recurrent_split, 1);
        assert_eq!(thor.chunk_state_wmma, wmma::SPLIT2);
        assert_eq!(orin.chunk_state_wmma, wmma::OFF);
    }

    #[test]
    fn an_unknown_family_gets_the_conservative_table() {
        assert_eq!(
            GdnLaunchPolicy::defaults_for(CudaArchFamily::Other(75)),
            GdnLaunchPolicy::defaults_for(CudaArchFamily::Sm80)
        );
    }

    #[test]
    fn overrides_reject_values_the_kernels_are_not_instantiated_for() {
        let mut tile = 4;
        override_from("APXINF_GDN_TEST_UNSET", &mut tile, &[1, 2, 4]);
        assert_eq!(tile, 4);
        let mut mode = wmma::OFF;
        override_wmma("APXINF_GDN_TEST_UNSET", &mut mode);
        assert_eq!(mode, wmma::OFF);
    }
}

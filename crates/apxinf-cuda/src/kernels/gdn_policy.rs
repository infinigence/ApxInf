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
pub mod wmma {
    /// The scalar fp32 kernel.
    pub const OFF: i32 = 0;
    /// One BF16 pass: the fp32 operand is rounded.
    ///
    /// On the prefill path every operand reaching the chunk-state scan was
    /// already written through `__float2bfloat16` by the kernel that produced
    /// it, so this pass is exact there rather than merely close.
    pub const LOSSY: i32 = 1;
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
    /// Threads per output column in the decode recurrence.
    pub recurrent_split: i32,
    /// How many blocks share one head's chunk-state scan, splitting the value
    /// dimension between them.
    ///
    /// The scan is sequential over chunks and its only parallelism is the head
    /// count, so the launch is one block per head -- 32 at the shipped shape.
    /// That is a full GPU on a 16-SM Orin and a quarter of one on a 128-SM
    /// 4090, which ran this kernel at 5.47 ms a layer, 131 ms a scene, on 25%
    /// of the device. Nothing in the kernel crosses the value dimension: the
    /// decay multiplies rows of the state, every accumulation sums over the
    /// key dimension, and each output column reads only its own column of the
    /// state. Splitting it is therefore the same arithmetic in the same order
    /// on a subset of the columns, and the result is bit-identical rather than
    /// merely close.
    pub chunk_state_v_split: i32,
}

impl GdnLaunchPolicy {
    /// The measured defaults for `caps`, then any environment overrides.
    pub fn for_device(caps: &CudaDeviceCaps) -> Self {
        let mut policy = Self::defaults_for(caps.arch_family);
        policy.chunk_state_v_split = chunk_state_v_split_for(caps.multiprocessor_count);
        policy.apply_overrides();
        policy
    }

    /// Defaults by architecture family, each swept on a board of that family.
    ///
    /// sm80 family (Orin sm_87, RTX 4090 sm_89): re-swept after the GDN
    /// kernels went to column tiles under a register cap, which moved both of
    /// them. The chunk-gemm width moved furthest, because fusing its two
    /// products doubled the accumulators and 32 of them no longer fit the cap:
    /// tile 4 is 5.785 s/scene, 8 is 5.764, 16 is 5.774 and 32 is 6.328 -- not
    /// a mild regression but a 10% one. The chunk-state block went the other
    /// way, from 1024 back to 512 (5.7645 against 5.7787 over four interleaved
    /// pairs), because the column tile needs fewer registers per thread and the
    /// cap already bought back the residency the wider block was there to
    /// provide. The tensor-core form stays off: that kernel needs an
    /// SM100-family tensor core to be worth its extra pass.
    ///
    /// sm100 family (Thor sm_110): chunk-state tile 4 rather than 8 and
    /// chunk-gemm tile 4 rather than 32, both re-swept here; the chunk-state
    /// scan on tensor cores, which is 21.9% of the fixed cost; and four
    /// threads per output column in the decode recurrence, which takes it from
    /// 39.383 to 37.373 ms/token.
    ///
    /// The chunk-state scan runs one BF16 pass. That pass is exact here rather
    /// than merely close: on the prefill path q, k, the chunk transition and
    /// the key-decay product are each written through `__float2bfloat16` by the
    /// kernel that produces them, so no operand carries a low term for a second
    /// pass to accumulate. `chunk_state_split_is_dead_when_operands_are_bf16`
    /// holds the producers to that. Worth 27% of the scan.
    pub fn defaults_for(family: CudaArchFamily) -> Self {
        match family {
            CudaArchFamily::Sm100 => Self {
                chunk_state_tile: 4,
                chunk_state_threads: 1024,
                chunk_state_wmma: wmma::LOSSY,
                chunk_gemm_tile: 4,
                recurrent_split: 4,
                chunk_state_v_split: 1,
            },
            CudaArchFamily::Sm80 | CudaArchFamily::Other(_) => Self {
                chunk_state_tile: 8,
                chunk_state_threads: 512,
                chunk_state_wmma: wmma::OFF,
                chunk_gemm_tile: 8,
                recurrent_split: 1,
                chunk_state_v_split: 1,
            },
        }
    }

    fn apply_overrides(&mut self) {
        override_from(
            "APXINF_GDN_CHUNK_TILE",
            &mut self.chunk_state_tile,
            &[1, 2, 4, 8, 16],
        );
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
        override_from(
            "APXINF_GDN_RECURRENT_SPLIT",
            &mut self.recurrent_split,
            &[1, 2, 4],
        );
        override_from(
            "APXINF_GDN_CHUNK_STATE_V_SPLIT",
            &mut self.chunk_state_v_split,
            &[1, 2, 4, 8],
        );
    }
}

/// How many blocks to give one head's chunk-state scan, from the width of the
/// device.
///
/// The scan launches one block per head and the shipped model has 32 of them,
/// so a device with more multiprocessors than that is idle in proportion. The
/// split is chosen to reach the multiprocessor count and no further: past it
/// the extra blocks only replicate the key-side reads without adding a wave.
/// It is capped at 4 because the value dimension is 128 and the block still
/// wants enough columns to keep its accumulator tiles whole.
fn chunk_state_v_split_for(multiprocessor_count: u32) -> i32 {
    // 32 heads is the shipped shape; the adapter clamps the split to whatever
    // the real head count and value width allow.
    const HEADS: u32 = 32;
    if multiprocessor_count >= HEADS * 4 {
        4
    } else if multiprocessor_count >= HEADS * 2 {
        2
    } else {
        1
    }
}

/// Accept only values the kernels are instantiated for; anything else is
/// ignored rather than passed through to a launch that would fail.
fn override_from(name: &str, slot: &mut i32, allowed: &[i32]) {
    if let Some(value) = std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<i32>().ok())
    {
        if allowed.contains(&value) {
            *slot = value;
        }
    }
}

/// `0`/`off`/`false` for the scalar kernel, `1`/`lossy`/`on`/`true` for the
/// one-pass BF16 kernel.
fn override_wmma(name: &str, slot: &mut i32) {
    let Ok(raw) = std::env::var(name) else { return };
    let value = raw.trim();
    *slot = match value {
        "0" | "off" | "false" => wmma::OFF,
        "1" | "lossy" | "on" | "true" => wmma::LOSSY,
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
        assert_eq!((orin.chunk_state_tile, orin.chunk_gemm_tile), (8, 8));
        assert_eq!(
            (thor.chunk_state_threads, orin.chunk_state_threads),
            (1024, 512)
        );
        assert_eq!(thor.recurrent_split, 4);
        assert_eq!(orin.recurrent_split, 1);
        assert_eq!(thor.chunk_state_wmma, wmma::LOSSY);
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

    #[test]
    fn value_split_follows_the_device_width() {
        // 16 SMs (Orin) and 20 (Thor) are narrower than the 32 heads the scan
        // already launches, so they keep one block per head.
        assert_eq!(chunk_state_v_split_for(16), 1);
        assert_eq!(chunk_state_v_split_for(20), 1);
        // 128 SMs (RTX 4090) fit four blocks per head.
        assert_eq!(chunk_state_v_split_for(128), 4);
        assert_eq!(chunk_state_v_split_for(64), 2);
        assert_eq!(chunk_state_v_split_for(63), 1);
    }
}

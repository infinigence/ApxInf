//! Immutable MUSA device facts used by kernel hardware dispatch.
//!
//! Mirrors `apxinf-cuda`'s `device_caps`, with `sm` replaced by `mp` — MUSA
//! names a device target `mp_<major><minor>`, so M1000's compute capability
//! 2.2 is `mp_22`.

use std::fmt;

use crate::ffi;

/// Moore Threads architecture families which share ApxInf kernel
/// implementations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MusaArchFamily {
    /// `mp_22` — M1000. BF16 MMA uses a `32x32x16` atom; dispatch uses
    /// native muBLAS/muDNN. There is no TMA, cluster, or tcgen05.
    Mp22,
    /// `mp_31` — S5000/S4000 family, which adds TME and SQMMA.
    Mp31,
    Other(u32),
}

impl MusaArchFamily {
    /// Classify a compute-capability number.
    ///
    /// `Other` is not a fallback for "unknown hardware" — it is the arm a
    /// generation lands in before this backend has measured what it can run, and
    /// the capability questions below answer `false` for it on purpose: an
    /// unmeasured generation has to be told apart from a supported one, and
    /// answering `true` would hide exactly that.
    pub const fn from_mp(mp: u32) -> Self {
        match mp {
            22 => Self::Mp22,
            31 => Self::Mp31,
            other => Self::Other(other),
        }
    }

    /// Whether the vision tower's D72 MHA has a muDNN flash kernel here.
    ///
    /// Both generations do, and the answer is a property of the family rather
    /// than of a call, so it is asked of the family. Spelling
    /// `mp == 22 || mp == 31` at the dispatch site would be a list a third
    /// generation falls off the end of: the kernel stays correct, just slow, and
    /// nothing says so.
    pub const fn has_vision_flash_mha(self) -> bool {
        matches!(self, Self::Mp22 | Self::Mp31)
    }
}

impl fmt::Display for MusaArchFamily {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Mp22 => formatter.write_str("mp22-family"),
            Self::Mp31 => formatter.write_str("mp31-family"),
            Self::Other(mp) => write!(formatter, "mp{mp}"),
        }
    }
}

/// Hardware capabilities queried once when a [`crate::MusaContext`] is made.
///
/// This type deliberately contains hardware facts only. Model names,
/// precision policy, and selected tactics belong to higher or lower layers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MusaDeviceCaps {
    pub device_name: String,
    pub compute_major: u32,
    pub compute_minor: u32,
    /// The generation as one number: `compute_major * 10 + compute_minor`, so
    /// M1000's 2.2 is 22.
    ///
    /// **A fingerprint, not an arch name.** [`Self::offload_arch`] is the string
    /// the build side and `mcc` use, and it joins the two components rather than
    /// doing arithmetic on them — `mp_22`, and `mp_310` if a minor ever reaches
    /// two digits, where this field would say 40. The two encodings agreed only
    /// while every minor stayed below 10, which is why `context.rs`'s
    /// build/device check compares `offload_arch()` strings and never this.
    ///
    /// What reads this field is bookkeeping that wants a number:
    /// [`MusaArchFamily::from_mp`], the tuning fingerprint, and the tactic
    /// directory name.
    pub mp: u32,
    pub multiprocessor_count: u32,
    pub arch_family: MusaArchFamily,
    /// The hardware warp width, which on M1000 is 128.
    ///
    /// This is *not* the shuffle group width: MUSA hardcodes the `warpSize`
    /// macro to 32 and every `__shfl_*`/`__ballot_*`/`__syncwarp` intrinsic
    /// works on 32-lane groups (verified on an M1000). Carried here so
    /// occupancy and barrier reasoning can use the real width without any
    /// kernel mistakenly deriving its shuffle grouping from it.
    pub hardware_warp_size: u32,
}

impl MusaDeviceCaps {
    pub fn query(device_id: usize) -> Result<Self, String> {
        let device = i32::try_from(device_id)
            .map_err(|_| format!("MUSA device id {device_id} does not fit in i32"))?;
        let compute_major =
            query_attribute(device, ffi::musa::attribute::COMPUTE_CAPABILITY_MAJOR)?;
        let compute_minor =
            query_attribute(device, ffi::musa::attribute::COMPUTE_CAPABILITY_MINOR)?;
        let multiprocessor_count =
            query_attribute(device, ffi::musa::attribute::MULTIPROCESSOR_COUNT)?;
        let mp = compute_major
            .checked_mul(10)
            .and_then(|major| major.checked_add(compute_minor))
            .ok_or_else(|| "MUSA compute capability overflow".to_string())?;

        let mut name = [0 as std::ffi::c_char; 256];
        // SAFETY: the driver is initialised by the runtime on first use; the
        // buffers are live for the whole call.
        unsafe {
            let mut driver_device = 0;
            let status = ffi::musa::muDeviceGet(&mut driver_device, device);
            if status != ffi::musa::MUDA_SUCCESS {
                return Err(ffi::driver_error(status));
            }
            let status =
                ffi::musa::muDeviceGetName(name.as_mut_ptr(), name.len() as i32, driver_device);
            if status != ffi::musa::MUDA_SUCCESS {
                return Err(ffi::driver_error(status));
            }
        }
        let device_name = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        Ok(Self {
            device_name,
            compute_major,
            compute_minor,
            mp,
            multiprocessor_count,
            arch_family: MusaArchFamily::from_mp(mp),
            hardware_warp_size: query_attribute(device, ffi::musa::attribute::WARP_SIZE)?,
        })
    }

    /// `mp_<major><minor>` — the value `mcc --offload-arch` takes.
    ///
    /// Joined from the two components, not printed from [`Self::mp`], so this
    /// agrees with `build_support::musa_arch::mcc_arch_for`, which is fed the
    /// driver's `"3.1"` and concatenates it the same way. Joining the components
    /// rather than printing `mp` is what keeps the two equal for any minor at or
    /// above 10, where the other spelling would report a build/device mismatch
    /// for a device that matches.
    pub fn offload_arch(&self) -> String {
        format!("mp_{}{}", self.compute_major, self.compute_minor)
    }
}

fn query_attribute(device: i32, attribute: i32) -> Result<u32, String> {
    let mut value = 0i32;
    let status = unsafe { ffi::musa::musaDeviceGetAttribute(&mut value, attribute, device) };
    if status != ffi::musa::MUSA_SUCCESS {
        return Err(ffi::runtime_error(status));
    }
    u32::try_from(value).map_err(|_| format!("MUSA attribute {attribute} is negative ({value})"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn caps(major: u32, minor: u32) -> MusaDeviceCaps {
        let mp = major * 10 + minor;
        MusaDeviceCaps {
            device_name: format!("test device {major}.{minor}"),
            compute_major: major,
            compute_minor: minor,
            mp,
            multiprocessor_count: 1,
            arch_family: MusaArchFamily::from_mp(mp),
            hardware_warp_size: 32,
        }
    }

    /// The two encodings of a generation agree everywhere they are compared.
    ///
    /// `offload_arch` is what `context.rs` compares against `MUSA_ARCH`, and
    /// `MUSA_ARCH` is produced by the build script's own concatenation. This is
    /// the property that used to hold by luck below minor 10 and nowhere above
    /// it.
    #[test]
    fn the_arch_string_is_the_components_joined() {
        assert_eq!(caps(2, 2).offload_arch(), "mp_22");
        assert_eq!(caps(3, 1).offload_arch(), "mp_31");
        // The case the arithmetic form cannot express, and the reason the two
        // are kept apart: 3.10 is `mp_310` to the build script and 40 here.
        assert_eq!(caps(3, 10).offload_arch(), "mp_310");
        assert_eq!(caps(3, 10).mp, 40);
    }

    /// `mp` still classifies the generations this backend knows.
    ///
    /// It is the fingerprint the tuning database is keyed on, so it has to keep
    /// answering for the two generations that exist — and `Other` for the rest
    /// rather than the nearest neighbour.
    #[test]
    fn the_fingerprint_classifies_the_known_generations() {
        assert_eq!(caps(2, 2).arch_family, MusaArchFamily::Mp22);
        assert_eq!(caps(3, 1).arch_family, MusaArchFamily::Mp31);
        assert_eq!(caps(3, 10).arch_family, MusaArchFamily::Other(40));
    }
}

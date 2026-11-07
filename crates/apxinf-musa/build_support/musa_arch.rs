//! MUSA offload-architecture selection.
//!
//! MUSA names device targets `mp_<major><minor>` for `mcc --offload-arch`
//! (M1000 reports compute capability 2.2, so `mp_22`). The value is taken from
//! `APXINF_MUSA_ARCH` when set, otherwise detected from the visible device via
//! `musaInfo`, and finally validated by compiling a one-line `.mu` — the same
//! shape as the CUDA build's `select_cuda_arch`, minus the multi-arch lists
//! that MUSA kernels have no use for yet.

use std::ffi::OsString;
use std::path::Path;
use std::process::Command;

use crate::mcc;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MusaArch {
    /// Value passed to `mcc --offload-arch`, e.g. `mp_22`.
    pub mcc_arch: String,
    /// How the value was chosen, for the build log.
    pub source: String,
}

/// `mp_22` for compute capability 2.2.
fn mcc_arch_for(compute_capability: &str) -> Option<String> {
    let (major, minor) = compute_capability.split_once('.')?;
    let major: u32 = major.trim().parse().ok()?;
    let minor: u32 = minor.trim().parse().ok()?;
    Some(format!("mp_{major}{minor}"))
}

/// Accept `mp_22`, `2.2` or `22` and normalise to `mp_22`.
fn normalize_explicit(value: &str) -> Option<String> {
    let value = value.trim();
    if let Some(rest) = value.strip_prefix("mp_") {
        return (!rest.is_empty()).then(|| value.to_string());
    }
    if value.contains('.') {
        return mcc_arch_for(value);
    }
    // `chars().all(...)` answers `true` for the empty string, which would spell
    // the target `mp_` — a name `mcc` has never heard of (`unknown target CPU`),
    // reached through `APXINF_MUSA_ARCH=`. Blank is handled before this by
    // `classify_explicit`; this is the belt to that pair of braces.
    (!value.is_empty() && value.chars().all(|c| c.is_ascii_digit())).then(|| format!("mp_{value}"))
}

/// What `APXINF_MUSA_ARCH` was set to, as a decision rather than a string.
///
/// Three states, because `Option` here was two facts wearing one type:
/// `explicit.and_then(normalize_explicit)` could not tell "unset" from "set to
/// something this build cannot read", so `APXINF_MUSA_ARCH=mp31` — a typo the
/// documentation invites by naming the values in prose — discarded the whole
/// explicit branch and fell through to detection or the default. The build log
/// then read `detected from musaInfo`, and the mismatch surfaced at run time in
/// `MusaContext::new`, a long way from the setting that caused it.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ArchOverride {
    /// Unset, or set to nothing but whitespace. `APXINF_MUSA_ARCH=` is how a
    /// shell or a CI template spells "not set", and it is treated as one.
    Unset,
    /// A name `mcc --offload-arch` takes.
    Named(String),
    /// Set to something that is not a name, and not blank either.
    Unusable(String),
}

fn classify_explicit(value: Option<&str>) -> ArchOverride {
    let Some(raw) = value else {
        return ArchOverride::Unset;
    };
    if raw.trim().is_empty() {
        return ArchOverride::Unset;
    }
    match normalize_explicit(raw) {
        Some(arch) => ArchOverride::Named(arch),
        None => ArchOverride::Unusable(raw.trim().to_string()),
    }
}

/// The compute capability `musaInfo` printed, or why it did not print one.
///
/// Pure over `(exit status, stdout)` so it can be tested without the toolkit,
/// and so the **diagnostic is not thrown away**: with no visible device
/// `musaInfo` exits 1 and prints `error: 'no MUSA-capable device is
/// detected'(100)` on *stdout* — in colour, unconditionally (measured:
/// `NO_COLOR` and `TERM=dumb` do not turn it off), which is why the escape
/// sequences come off before the line is used.
fn parse_musa_info(success: bool, stdout: &str) -> Result<String, String> {
    if let Some(arch) = first_compute_capability(stdout) {
        return Ok(arch);
    }
    if success {
        return Err("musaInfo printed no parsable `compute capability` line".to_string());
    }
    let line = stdout
        .lines()
        .map(strip_ansi)
        .find(|line| line.trim_start().starts_with("error:"))
        .unwrap_or_else(|| "musaInfo exited non-zero without a diagnostic".to_string());
    Err(line.trim().to_string())
}

/// Device 0's `compute capability` line, as an `mcc` target.
fn first_compute_capability(text: &str) -> Option<String> {
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        if key.trim().eq_ignore_ascii_case("compute capability") {
            return mcc_arch_for(value.trim());
        }
    }
    None
}

/// Drop CSI escape sequences (`ESC [ … <letter>`), the only kind `musaInfo`
/// emits: it colours its `error:` lines whether or not anything is looking.
fn strip_ansi(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut characters = line.chars();
    while let Some(character) = characters.next() {
        if character != '\u{1b}' {
            out.push(character);
            continue;
        }
        if characters.next() != Some('[') {
            continue;
        }
        for parameter in characters.by_ref() {
            if parameter.is_ascii_alphabetic() {
                break;
            }
        }
    }
    out
}

/// Detect the compute capability of device 0 from `musaInfo`, or say why not.
fn detect_from_musa_info(musa_info: &Path) -> Result<String, String> {
    let output = Command::new(musa_info).output().map_err(|error| {
        format!("musaInfo could not be run ({}): {error}", musa_info.display())
    })?;
    parse_musa_info(
        output.status.success(),
        &String::from_utf8_lossy(&output.stdout),
    )
}

/// Compile a trivial translation unit to check that `mcc` accepts the arch name.
///
/// The probe's source has no MUSA code of its own and `mcc` supplies the include
/// that reaches this host's glibc, so a failure without the shim says nothing
/// about the arch name; that is what the retry is for. Without it the probe
/// could only ever answer "unverified".
fn validate_mcc_arch(mcc_path: &Path, arch: &str, host_math_shim: Option<&Path>) -> Result<(), String> {
    let scratch = mcc::Scratch::new("arch");
    let source = scratch.source("probe.mu", "int main() { return 0; }\n")?;
    let object = scratch.object("probe");
    let extra = [OsString::from(format!("--offload-arch={arch}"))];

    let diagnostic = match mcc::compile(mcc_path, &extra, None, &source, &object)? {
        Ok(()) => return Ok(()),
        Err(diagnostic) => diagnostic,
    };
    match host_math_shim {
        Some(shim) => match mcc::compile(mcc_path, &extra, Some(shim), &source, &object)? {
            Ok(()) => Ok(()),
            Err(retry) => Err(mcc::first_line(&retry).to_owned()),
        },
        None => Err(mcc::first_line(&diagnostic).to_owned()),
    }
    .map_err(|reason| {
        format!(
            "could not confirm --offload-arch={arch} with a test compile, so the name is \
             unverified rather than known bad — {reason}"
        )
    })
}

/// The selected target plus a non-fatal complaint, if any.
///
/// The `validate_mcc_arch` probe is advisory and must not be able to disable the
/// backend. Its failure mode is shared with every other reason a `-x musa` probe
/// cannot compile, so treating it as fatal would trade the wheel's MUSA backend
/// for a question the real kernel compile answers better.
///
/// An **unreadable `APXINF_MUSA_ARCH` is a different matter**, and
/// [`select_musa_arch`] refuses it rather than falling through: the probe's
/// failure says nothing about the arch name, while a name this build cannot
/// parse says everything about the intent behind it — honoring that intent by
/// guessing another generation is how a build machine with no visible device
/// produces a wheel for the wrong one.
pub struct ArchChoice {
    arch: MusaArch,
    warning: Option<String>,
}

impl ArchChoice {
    pub fn target(&self) -> MusaArch {
        self.arch.clone()
    }

    pub fn warning(&self) -> Option<&str> {
        self.warning.as_deref()
    }
}

/// The target to build for: what was asked for, or what this host is.
///
/// `Err` only when `APXINF_MUSA_ARCH` is set to something that is not a MUSA
/// target — see [`ArchChoice`]. Everything else degrades to a warning, including
/// a probe that found no device at all: a build machine with no MUSA card is a
/// legitimate place to produce a wheel, and the run-time check in
/// `MusaContext::new` is what refuses the result by name if it is aimed at the
/// wrong generation.
pub fn select_musa_arch(
    explicit: Option<&str>,
    mcc: &Path,
    host_math_shim: Option<&Path>,
) -> Result<ArchChoice, String> {
    // MUSA's own tooling sits next to the compiler.
    let musa_info = mcc.with_file_name("musaInfo");

    let (mcc_arch, source) = match classify_explicit(explicit) {
        ArchOverride::Unusable(value) => {
            return Err(format!(
                "APXINF_MUSA_ARCH is set to {value:?}, which is not a MUSA target. Accepted \
                 forms are `mp_31`, `3.1` or `31`; the names `mcc --offload-arch` takes are \
                 mp_10, mp_21, mp_22, mp_31, mp_32, mp_42 and mp_43. Unset the variable to \
                 detect the visible device with musaInfo instead."
            ));
        }
        ArchOverride::Named(arch) => (arch, "explicit".to_string()),
        ArchOverride::Unset => match detect_from_musa_info(&musa_info) {
            Ok(arch) => (arch, "detected from musaInfo".to_string()),
            Err(reason) => (
                "mp_22".to_string(),
                format!(
                    "default (no arch from musaInfo: {reason}). This binary is built for mp_22 \
                     and a context on another generation is refused by name — set \
                     APXINF_MUSA_ARCH to build for one"
                ),
            ),
        },
    };

    Ok(ArchChoice {
        arch: MusaArch {
            mcc_arch: mcc_arch.clone(),
            source,
        },
        warning: validate_mcc_arch(mcc, &mcc_arch, host_math_shim).err(),
    })
}

/// The toolkit version, for the kernel build id.
pub fn musa_version(musa_path: &Path) -> Option<String> {
    for file in ["version.json", "musa_version"] {
        if let Ok(contents) = std::fs::read_to_string(musa_path.join(file)) {
            return Some(contents.trim().to_string());
        }
    }
    let output = Command::new(musa_path.join("bin/musa_toolkits_version"))
        .output()
        .ok()?;
    Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

/// Just the toolkit's `major.minor` version, for a decision rather than a hash.
///
/// `musa_version` above returns the whole `version.json` — it is a kernel
/// build-id input, where the entire file is the point. This picks the number
/// out of it. `version.json` is generated by the toolkit and its shape is not
/// ours to assume, so a file that does not parse reports `None` and the caller
/// decides what an unknown toolkit means; it must not guess a version.
pub fn musa_version_number(musa_path: &Path) -> Option<(u32, u32)> {
    // `{"musa_toolkits": {"version": "5.2.0", ...}, ...}`. Anchoring on the
    // section first keeps a later `"version"` key from being read as the
    // toolkit's.
    let contents = std::fs::read_to_string(musa_path.join("version.json"))
        .or_else(|_| std::fs::read_to_string(musa_path.join("musa_version")))
        .ok()?;
    let after_section = contents
        .find("\"musa_toolkits\"")
        .map_or(contents.as_str(), |at| &contents[at..]);

    for needle in ["\"version\"", "version"] {
        let Some(at) = after_section.find(needle) else {
            continue;
        };
        let rest = &after_section[at + needle.len()..];
        let Some(open) = rest.find('"') else { continue };
        let Some(close) = rest[open + 1..].find('"') else {
            continue;
        };
        let text = &rest[open + 1..open + 1 + close];
        let mut parts = text.trim().split('.');
        let major = parts.next().and_then(|value| value.parse().ok());
        let minor = parts.next().and_then(|value| value.parse().ok());
        if let (Some(major), Some(minor)) = (major, minor) {
            return Some((major, minor));
        }
    }
    None
}

/// The generation the D256 attention assembly in `kernels/asm/` is written for.
///
/// The three `.asm` files are MP22 ISA and `llvm-mc` rejects their scheduling
/// annotations on any other generation, so a build for another target has no
/// image to load and does not try to assemble one.
pub const ASSEMBLY_ARCH: &str = "mp_22";

/// The generation the vendored FMHA in `kernels/mutlass_fmha/` was rendered and
/// measured on.
pub const VENDOR_FMHA_ARCH: &str = "mp_31";

/// Which D256 attention kernel a build runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum D256Route {
    /// The MP22 assembly, loaded through the driver API by
    /// `adapters/mubin_fa_adapter.mu`.
    Assembly,
    /// The vendored mp_31 FMHA, launched by `adapters/mutlass_fa_adapter.mu`.
    Mutlass,
    /// Neither is available to this build, so the operator reports itself as a
    /// kernel gap instead of running something adjacent.
    Unsupported,
}

impl D256Route {
    /// The name `build.rs` publishes for this route.
    ///
    /// A string rather than a `cfg` because the library has to *report* which
    /// route it took as well as branch on it, and a boolean `cfg` cannot say
    /// "unsupported" — only "not this one, and not that one either". The
    /// library asserts at compile time that the name is one it knows, so the
    /// third possibility cannot become a silent fallback.
    pub const fn as_str(self) -> &'static str {
        match self {
            D256Route::Assembly => "assembly",
            D256Route::Mutlass => "mutlass",
            D256Route::Unsupported => "unsupported",
        }
    }
}

/// The one place the D256 routing rule is written.
///
/// `build.rs` asks it what to compile **and** what to publish, and
/// `kernels/attention.rs` asks what it published. That last step is what makes
/// this sentence true: the library used to re-derive the rule from the same
/// inputs, and the two copies were held in step by `tests/arch_gate.rs`. The
/// failure mode of a drift is not a wrong message: `Mutlass` is matched by
/// string, so it would simply stop matching, and mp_31's only attention would
/// become a gap with nothing to point at.
///
/// `vendor_fmha_built` is separate from the arch because the arch is necessary
/// and not sufficient — the frozen kernels are also gated on the toolkit they
/// were measured with, which `build.rs` decides.
pub fn d256_route(arch: &str, vendor_fmha_built: bool) -> D256Route {
    if arch == ASSEMBLY_ARCH {
        D256Route::Assembly
    } else if arch == VENDOR_FMHA_ARCH && vendor_fmha_built {
        D256Route::Mutlass
    } else {
        D256Route::Unsupported
    }
}

/// The parsing half of this file, tested through `tests/arch_gate.rs`, which
/// includes it by path (`mod build_side`). None of it needs a toolkit or a
/// device, which is the point: the build script cannot be unit-tested, but the
/// decisions it makes can.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unreadable_apxinf_musa_arch_is_refused_rather_than_fallen_through() {
        assert_eq!(classify_explicit(None), ArchOverride::Unset);
        assert_eq!(classify_explicit(Some("")), ArchOverride::Unset);
        assert_eq!(classify_explicit(Some("   ")), ArchOverride::Unset);

        for named in ["mp_22", "2.2", "22", " mp_31 "] {
            assert!(
                matches!(classify_explicit(Some(named)), ArchOverride::Named(_)),
                "{named} should be readable"
            );
        }
        // `MP_22` is in here because `mcc` is case-sensitive too: measured,
        // `--offload-arch=MP_22` answers `unknown target CPU 'MP_22'`. A value
        // mcc would reject must not be quietly replaced by a detected one.
        for bad in ["mp31", "MP_22", "mp-22", "sm_87", "mp_", "garbage"] {
            assert!(
                matches!(classify_explicit(Some(bad)), ArchOverride::Unusable(_)),
                "{bad} is not a name and must not fall through to detection"
            );
        }
    }

    #[test]
    fn the_empty_string_never_spells_a_target() {
        // `chars().all(...)` is true for the empty string; that is how `mp_`
        // used to come out of `APXINF_MUSA_ARCH=`.
        for value in ["", " ", "mp_", "\t"] {
            assert_ne!(normalize_explicit(value).as_deref(), Some("mp_"), "{value:?}");
        }
    }

    /// The message `musaInfo` prints when no device is visible is carried, not
    /// swallowed. Measured on MUSA 5.2: exit 1, the text on stdout, in colour.
    #[test]
    fn a_failed_probe_carries_musainfo_s_own_diagnostic() {
        let captured = "compiler: mcc\n\x1b[31merror: 'no MUSA-capable device is detected'(100) at \
                        /go/build/musa_toolkit.cpp:130\x1b[0m\n\x1b[31merror: API returned error \
                        code.\n";
        let reason = parse_musa_info(false, captured).unwrap_err();
        assert!(
            reason.contains("no MUSA-capable device is detected"),
            "{reason}"
        );
        assert!(
            !reason.contains('\u{1b}'),
            "the escape sequences must come off: {reason:?}"
        );

        // A probe that succeeds without a capability line is still not a target.
        assert!(parse_musa_info(true, "compiler: mcc\n").is_err());
        assert_eq!(
            parse_musa_info(true, "compute capability:              3.1\n").unwrap(),
            "mp_31"
        );
    }
}


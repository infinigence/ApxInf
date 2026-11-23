//! The MUSA context: a device binding, its hardware facts, its stream, and
//! the runtime-owned tuning state.

use std::sync::{Arc, RwLock};

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::device_caps::{MusaArchFamily, MusaDeviceCaps};
use crate::ffi;
use crate::stream::MusaStream;
use crate::tuning::TuningSession;

/// Identify the mapped vendor binaries a persisted tuning record depends on.
///
/// Best-effort by construction. This is diagnostic evidence attached to a
/// tactic record, so it must never be able to stop a context from being
/// created: an unreadable `/proc/self/maps`, or a mapped `.so` whose path no
/// longer resolves — an in-place SDK upgrade unlinks the old file while the
/// mapping survives — is a normal event, not a backend failure.
///
/// Either condition returns an **empty** map rather than a partial one. Empty
/// reads as "cannot be validated" downstream, which drops fingerprint-bound
/// records and falls back to muBLAS; a partial map would look complete, compare
/// equal to no stored fingerprint, and silently invalidate a database that is
/// in fact fine.
fn library_fingerprints() -> std::collections::BTreeMap<String, String> {
    let mut result = std::collections::BTreeMap::new();
    let Ok(maps) = std::fs::read_to_string("/proc/self/maps") else {
        return result;
    };
    for line in maps.lines() {
        let Some(path) = line.split_whitespace().nth(5) else {
            continue;
        };
        let Some(name) = std::path::Path::new(path)
            .file_name()
            .and_then(|v| v.to_str())
        else {
            continue;
        };
        if !["libmublas", "libmudnn", "libmusart", "libmusa.so"]
            .iter()
            .any(|prefix| name.starts_with(prefix))
            || result.contains_key(name)
        {
            continue;
        }
        let Ok(file) = std::fs::metadata(path) else {
            return std::collections::BTreeMap::new();
        };
        let Ok(modified) = file.modified() else {
            return std::collections::BTreeMap::new();
        };
        let Ok(modified) = modified.duration_since(std::time::UNIX_EPOCH) else {
            return std::collections::BTreeMap::new();
        };
        // SDK plugins are up to a gigabyte and have no ELF build IDs. Their
        // versioned filename, size and nanosecond mtime invalidate normal SDK
        // updates without hashing several gigabytes during context creation.
        result.insert(
            name.to_owned(),
            format!("file-v1:{}:{}", file.len(), modified.as_nanos()),
        );
    }
    result
}

/// MUSA libraries whose versions constrain persisted tuning results.
///
/// `mublas` exposes no version query, so only the pieces that can be asked
/// are recorded; a persisted database that would need a mublas version is not
/// something this backend can key on yet.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct MusaLibraryVersions {
    /// `musaRuntimeGetVersion`, in the toolkit's own encoding.
    pub musa: i32,
    /// `mublasLtGetVersion`, in its own encoding.
    pub mublas_lt: usize,
    /// Actual mapped vendor file versions, lengths and modification times.
    pub binaries: std::collections::BTreeMap<String, String>,
}

impl MusaLibraryVersions {
    fn query() -> Self {
        let mut musa = 0;
        // SAFETY: a plain out-parameter; a failure leaves it untouched.
        let musa = match unsafe { ffi::musa::musaRuntimeGetVersion(&mut musa) } {
            ffi::musa::MUSA_SUCCESS => musa,
            _ => 0,
        };
        // SAFETY: takes no arguments and returns a value.
        let mublas_lt = unsafe { ffi::musa::mublasLtGetVersion() };
        Self {
            musa,
            mublas_lt,
            binaries: library_fingerprints(),
        }
    }
}

pub struct MusaContext {
    pub(crate) asm_weights: std::sync::Mutex<crate::kernels::gemm::assembly::PreparedWeights>,
    device_id: usize,
    caps: MusaDeviceCaps,
    stream: MusaStream,
    library_versions: MusaLibraryVersions,
    tuning: RwLock<Arc<TuningSession>>,
}

/// Say so when this device's generation is one no kernel here was measured on.
///
/// [`MusaArchFamily::Other`] is not "unknown hardware": it is a generation that
/// exists and that this backend has not been qualified against. Its capability
/// questions answer `false` — `has_vision_flash_mha` is the one with a
/// consequence — and that is both the conservative answer and a **silent** one.
/// The vision attention takes the slower route, every status stays `SUCCESS`,
/// and nothing in the output distinguishes it from a device that simply has no
/// flash kernel. Centralising the list in `MusaArchFamily` moved the decision
/// from the dispatch site to one place; it did not make it louder, which is what
/// this does.
///
/// A warning rather than an error, because there is nothing to refuse: the
/// backend runs, and every operator with no kernel for this generation already
/// names itself when it is reached. What is left unstated without this is the
/// other half — that an operator which *does* run on an unmeasured generation
/// may be running a route chosen for a different one.
///
/// Once per process: a machine can hold several cards, and one line naming the
/// generation is the message; one per card is noise.
fn warn_unmeasured_generation(caps: &MusaDeviceCaps) {
    if !matches!(caps.arch_family, MusaArchFamily::Other(_)) {
        return;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        eprintln!(
            "[apxinf] MUSA device {} is {}, a generation this backend has not been measured \
             on. It runs, but every capability question is answered conservatively rather than \
             from measurement, so a kernel that exists for this generation may be unused \
             rather than absent — see doc/musa-backend.md.",
            caps.device_name, caps.arch_family
        );
    });
}

/// Make `device_id` the device this thread's next runtime call acts on.
///
/// Context creation and stream replacement both say so rather than inheriting
/// whichever device was left current: `musaStreamCreate` has no device
/// argument, so a replacement stream made under someone else's binding would
/// silently move the whole context to another card.
fn set_device(device_id: usize) -> Result<()> {
    let device = i32::try_from(device_id)
        .map_err(|_| Error::Other(format!("MUSA device id {device_id} does not fit in i32")))?;
    unsafe { ffi::check_runtime(ffi::musa::musaSetDevice(device)) }.map_err(Error::Other)
}

impl MusaContext {
    pub fn new(device_id: usize) -> Result<Self> {
        // A standalone Rust process has not had torch_musa initialize the
        // driver. Initialize explicitly before the runtime selects a device.
        let initialized = unsafe { ffi::musa::muInit(0) };
        if initialized != ffi::musa::MUDA_SUCCESS {
            return Err(Error::Other(format!(
                "MUSA driver initialization failed: {initialized}"
            )));
        }
        set_device(device_id)?;
        let caps = MusaDeviceCaps::query(device_id).map_err(Error::Other)?;
        // This binary's kernels were compiled for one generation, and a machine
        // can hold two — M1000 and S5000 are both MUSA devices and are expected
        // to coexist. Launching one generation's image on another is not a
        // degraded run: it is a driver-level failure with no attribution, or a
        // plausible wrong answer. So the mismatch is named here, where the
        // device is known, rather than left to whatever the launch reports.
        //
        // `unknown` is the value `build.rs` writes when it found no toolkit at
        // all. That build has no kernels to mismatch, and the check must not
        // turn a toolkit-less environment into a context-creation failure.
        let built = crate::MUSA_ARCH;
        let actual = caps.offload_arch();
        if built != "unknown" && built != actual {
            return Err(Error::Other(format!(
                "MUSA build/device mismatch: this binary's kernels were built for {built} \
                 (APXINF_MUSA_ARCH at build time) but device {device_id} ({}) is {actual}. \
                 Rebuild with APXINF_MUSA_ARCH={actual}, or select a {built} device",
                caps.device_name
            )));
        }
        warn_unmeasured_generation(&caps);
        Ok(Self {
            asm_weights: Default::default(),
            device_id,
            caps,
            stream: MusaStream::new()?,
            library_versions: MusaLibraryVersions::query(),
            tuning: RwLock::new(Arc::new(TuningSession::inference(Default::default()))),
        })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn caps(&self) -> &MusaDeviceCaps {
        &self.caps
    }

    pub fn arch_family(&self) -> MusaArchFamily {
        self.caps.arch_family
    }

    pub fn stream(&self) -> &MusaStream {
        &self.stream
    }

    /// Continue on a fresh stream after the runtime poisoned this one.
    ///
    /// One caller: [`crate::graph::capture`], and only when leaving capture
    /// failed — an invalidated capture leaves its stream in `INVALIDATED`,
    /// where every later launch is refused and `musaDeviceSynchronize` fails
    /// for as long as it exists. See [`MusaStream::replace`].
    pub fn replace_stream(&self) -> Result<()> {
        set_device(self.device_id)?;
        self.stream.replace()
    }

    pub fn library_versions(&self) -> &MusaLibraryVersions {
        &self.library_versions
    }

    /// Runtime-owned tuning state.
    ///
    /// Cloning the `Arc` is limited to prepare paths. Panics on a poisoned lock
    /// rather than returning a `Result`: `install_tuning` replaces the whole
    /// `Arc` in one assignment, so the only way to poison this is a panic
    /// somewhere else entirely, and every caller here would have to invent a
    /// recovery it does not have. [`crate::tuning::TuningSession`] carries the
    /// reasoning for the one lock whose contents are not a single value.
    pub fn tuning(&self) -> Arc<TuningSession> {
        self.tuning
            .read()
            .expect("MUSA tuning session lock is poisoned")
            .clone()
    }

    /// Install the mode and immutable tactic snapshot before model prepare.
    pub fn install_tuning(&self, session: TuningSession) -> std::result::Result<(), String> {
        *self
            .tuning
            .write()
            .map_err(|_| "MUSA tuning session lock is poisoned".to_string())? = Arc::new(session);
        Ok(())
    }

    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()
    }

    /// Allocate an operator output on this context's device.
    ///
    /// Inside an [`crate::workspace::ExecutionSession`] the allocation is a
    /// deterministic slice of the session arena; outside one it is a driver
    /// allocation on this stream. Operators call this instead of allocating for
    /// themselves, which is what lets one model body run eager, under capture
    /// and under replay unchanged.
    pub fn allocate_output(&self, shape: Shape, dtype: DType) -> Result<Tensor> {
        let bytes = shape
            .numel()
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::Other("MUSA output byte size overflow".into()))?;
        crate::workspace::output_buffer(self, bytes)?
            .as_tensor(shape, dtype)
            .map_err(crate::error::musa_error)
    }
}

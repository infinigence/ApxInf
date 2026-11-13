//! Row-major BF16 GEMM, through hipBLAS or hipBLASLt.
//!
//! Neither library wins everywhere. Measured over GR00T N1.7's 26 GEMM shapes
//! on gfx1151, hipBLASLt is 1.11x-3.65x faster on the 13 shapes below the
//! roofline ridge point and 0.61x-0.95x *slower* on the 5 above it, and the
//! shapes within +/-15% of the ridge go either way with no rule separating
//! them: `128x2048x1536` and `128x4096x4096` have near-identical arithmetic
//! intensity and opposite verdicts. So the provider is chosen per shape from a
//! measured table, and hipBLAS -- the faster default on more than a fifth of
//! the shapes -- is what an unlisted shape gets.
//!
//! See `devlocal/gr00t-hip-port/reports/perf/gemm-provider-comparison.md` for
//! the measurement and `doc/hip-gr00t.md` for the recipe format.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use std::ffi::c_void;

use apxinf_core::{Error, Result, Tensor};

use super::elementwise::{
    Activation, EPILOGUE_BIAS, EPILOGUE_BIAS_GELU_TANH, EPILOGUE_BIAS_RELU, EPILOGUE_NONE,
};
use super::{bf16_ptr, int, rows_cols};
use crate::ffi::{self, check, DTYPE_BF16};
use crate::runtime::HipContext;

/// `(M, K, N) -> call count`, collected when `APXINF_HIP_GEMM_SHAPES` is set.
///
/// Choosing a provider or a tactic per shape needs to know which shapes a
/// model actually issues and how often -- a model's GEMMs are not one shape,
/// and the rare ones are not worth tuning. Off by default: the lock would sit
/// on the dispatch path of every call.
static SHAPES: OnceLock<Option<Mutex<BTreeMap<(usize, usize, usize), u64>>>> = OnceLock::new();

fn recorder() -> Option<&'static Mutex<BTreeMap<(usize, usize, usize), u64>>> {
    SHAPES
        .get_or_init(|| {
            std::env::var_os("APXINF_HIP_GEMM_SHAPES").map(|_| Mutex::new(BTreeMap::new()))
        })
        .as_ref()
}

/// Every `(M, K, N)` issued since the last [`reset_recorded_shapes`], with
/// call counts. Empty unless `APXINF_HIP_GEMM_SHAPES` is set.
pub fn recorded_shapes() -> Vec<((usize, usize, usize), u64)> {
    recorder().map_or_else(Vec::new, |cell| {
        cell.lock()
            .map(|map| map.iter().map(|(k, v)| (*k, *v)).collect())
            .unwrap_or_default()
    })
}

pub fn reset_recorded_shapes() {
    if let Some(cell) = recorder() {
        if let Ok(mut map) = cell.lock() {
            map.clear();
        }
    }
}

/// Below this measured ratio a shape stays on hipBLAS. Run-to-run spread on
/// this part is a few percent, so routing a 1.01x "win" would be routing
/// noise, and every switch costs a warm-up query and a cached descriptor set.
const ROUTE_MARGIN: f64 = 1.05;

/// Which library a GEMM is issued through.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Provider {
    HipBlas,
    HipBlasLt,
}

/// How `gemm` picks a provider, resolved once from the environment.
///
/// - unset or `hipblas` — hipBLAS for everything. The default: a backend that
///   changes which kernels it runs because an environment variable happens to
///   be set is not one anyone can reason about.
/// - `lt` — hipBLASLt for everything it can serve. For A/B measurement; the
///   measurement says this is a net loss on the compute-bound shapes.
/// - `auto` — the recipe named by `APXINF_HIP_GEMM_TACTICS`, or, with no
///   recipe, `M <= 64 -> hipBLASLt`. That one rule captures 84% of the
///   available saving, because the tall-skinny shapes are where hipBLAS leaves
///   the most bandwidth unused.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    HipBlasOnly,
    LtOnly,
    Auto,
}

/// A measured provider choice, and optionally the tactic that was measured.
///
/// The tactic is the serialized `hipblasLtMatmulAlgo_t` -- opaque here on
/// purpose, since its size comes from the C++ side. Without one, hipBLASLt's
/// own heuristic picks, which measurement puts at 70% of the achievable gain.
/// `(M, K, N, epilogue)`.
///
/// The epilogue is part of the key because it changes the answer, not just
/// the work: `41x1536x6144` measures 1.04x *slower* on hipBLASLt as a plain
/// GEMM and faster with the bias and GELU fused in, because the fused form
/// also deletes a kernel. Keying on shape alone would route the fused call by
/// the unfused measurement and lose most of what fusing is worth.
type Key = (usize, usize, usize, i32);

struct Recipe {
    providers: BTreeMap<Key, Provider>,
    tactics: BTreeMap<Key, Vec<u8>>,
}

struct Routing {
    mode: Mode,
    recipe: Recipe,
    /// Measure both providers on a shape the recipe does not cover, and write
    /// the winner back. Opt-in, like the CUDA side's `LoadOptions.autotune`:
    /// tuning perturbs the request that triggers it and writes to disk, and
    /// neither should happen because a model happened to be loaded.
    autotune: bool,
    /// Where a tuned entry is written. `None` disables persistence, which
    /// keeps a read-only deployment from failing on a miss.
    recipe_path: Option<PathBuf>,
}

/// Shapes measured in this process, with the winner.
///
/// Consulted only after the loaded recipe misses *and* autotune is on, so the
/// shipped path -- recipe hit, or no autotune -- never takes this lock.
static TUNED: OnceLock<Mutex<BTreeMap<Key, (Provider, Vec<u8>)>>> = OnceLock::new();

fn tuned() -> &'static Mutex<BTreeMap<Key, (Provider, Vec<u8>)>> {
    TUNED.get_or_init(|| Mutex::new(BTreeMap::new()))
}

static ROUTING: OnceLock<Routing> = OnceLock::new();

/// The hipBLASLt handle whose tactics have been registered, or 0.
///
/// `apxinf_hip_blaslt_set_tactic` discards the handle's cached descriptors for
/// that shape, so calling it per GEMM would rebuild the plan every time. One
/// atomic load per call is the price of doing it once; with the single context
/// this backend creates, the slow path runs exactly once.
static TACTICS_REGISTERED: AtomicUsize = AtomicUsize::new(0);

/// GEMMs issued through each provider, as `(hipBLAS, hipBLASLt)`.
///
/// This exists because the fallback is silent by design: a null hipBLASLt
/// handle, a shape with no usable algorithm, or a recipe that routes nothing
/// all end with the work on hipBLAS and the right answer. Without a count, an
/// A/B that secretly never left the default path is indistinguishable from one
/// that found no speedup -- and a test asserting "hipBLASLt still matches the
/// reference" passes most convincingly when hipBLASLt never ran.
///
/// Only touched when a provider other than hipBLAS is in play, so the default
/// path keeps the atomics out of it entirely.
static DISPATCHES: [AtomicUsize; 2] = [AtomicUsize::new(0), AtomicUsize::new(0)];

/// `(hipBLAS, hipBLASLt)` dispatch counts since [`reset_dispatch_counts`].
/// Both zero while the default provider is in force.
pub fn dispatch_counts() -> (usize, usize) {
    (
        DISPATCHES[0].load(Ordering::Relaxed),
        DISPATCHES[1].load(Ordering::Relaxed),
    )
}

pub fn reset_dispatch_counts() {
    DISPATCHES[0].store(0, Ordering::Relaxed);
    DISPATCHES[1].store(0, Ordering::Relaxed);
}

fn routing() -> &'static Routing {
    ROUTING.get_or_init(|| {
        let mode = match std::env::var("APXINF_HIP_GEMM_PROVIDER")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "lt" | "hipblaslt" => Mode::LtOnly,
            "auto" => Mode::Auto,
            // Resolved below, once it is known whether a recipe was found.
            "" | "hipblas" => Mode::HipBlasOnly,
            other => {
                eprintln!(
                    "apxinf-hip: APXINF_HIP_GEMM_PROVIDER={other} is not hipblas, lt or auto; \
                     using hipblas"
                );
                Mode::HipBlasOnly
            }
        };
        let autotune = std::env::var("APXINF_HIP_GEMM_AUTOTUNE")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let explicit = std::env::var_os("APXINF_HIP_GEMM_TACTICS").map(PathBuf::from);
        let recipe_path = explicit.clone().or_else(default_recipe_path);
        let recipe = match &recipe_path {
            Some(path) => load_recipe(path),
            None => Recipe {
                providers: BTreeMap::new(),
                tactics: BTreeMap::new(),
            },
        };
        // Autotuning without a destination measures on every process start and
        // keeps the result to itself, which is worse than not tuning.
        if autotune && recipe_path.is_none() {
            eprintln!(
                "apxinf-hip: APXINF_HIP_GEMM_AUTOTUNE is set but no recipe path could be \
                 resolved; tuning results would be discarded, so autotune is off"
            );
        }
        // A recipe that exists was measured on this device, under these
        // library versions, by someone who asked for it: the path carries the
        // architecture and both versions, the header is checked against them,
        // and every tactic is trial-launched before it is trusted. Honouring
        // it without a further switch is what makes autotune worth running --
        // tune once, and later loads are faster with no flag to remember.
        //
        // Nothing changes on a machine where no one has tuned: no file, no
        // recipe, hipBLAS everywhere, exactly as before.
        let mode = match mode {
            Mode::HipBlasOnly if autotune || !recipe.providers.is_empty() => Mode::Auto,
            other => other,
        };
        if !recipe.providers.is_empty() {
            if let Some(path) = &recipe_path {
                eprintln!(
                    "apxinf-hip: using {} GEMM provider choices from {}",
                    recipe.providers.len(),
                    path.display()
                );
            }
        }
        Routing {
            mode,
            recipe,
            autotune: autotune && recipe_path.is_some(),
            recipe_path,
        }
    })
}

/// The ROCm and hipBLASLt runtime versions, read once per process.
///
/// `apxinf_hip_library_versions` creates a temporary hipBLASLt handle to read
/// the library version, and hipBLASLt's initialization kernels run on the
/// legacy default stream -- which fails fatally while a capture is open on
/// any stream in the process (error 906, and hipblaslt answers by exiting the
/// process). The read therefore happens under the capture lock, and exactly
/// once, so no later read can land in a capture window. `None` when the
/// versions could not be read, or when the calling thread is itself
/// capturing: callers already treat an unreadable version as "no recipe path
/// / no stamp", which is the right degradation inside a capture anyway.
fn library_versions() -> Option<(i32, i32)> {
    static VERSIONS: OnceLock<Option<(i32, i32)>> = OnceLock::new();
    *VERSIONS.get_or_init(|| {
        let (mut rocm, mut blaslt) = (0i32, 0i32);
        let _exclusive = crate::graph::allocation_exclusive().ok()?;
        if unsafe { ffi::apxinf_hip_library_versions(&mut rocm, &mut blaslt) } != 0 {
            return None;
        }
        Some((rocm, blaslt))
    })
}

/// `configs/tuning/amd/<arch>/rocm<version>-hipblaslt<version>/gemm.recipe`,
/// relative to the working directory.
///
/// The library versions are in the path because a recorded tactic does not
/// survive either of them changing -- hipBLASLt's header says a solution index
/// cannot be used across library versions. Putting them in the directory name
/// rather than inside the file means an upgraded host simply misses and
/// re-tunes, instead of reading a file that no longer means what it says. This
/// mirrors `configs/tuning/nvidia/<family>/cuda<..>-cublas<..>/`.
fn default_recipe_path() -> Option<PathBuf> {
    let (rocm, blaslt) = library_versions()?;
    let arch = env!("APXINF_HIP_ARCH");
    if arch == "none" {
        return None;
    }
    Some(
        Path::new("configs/tuning/amd")
            .join(arch)
            .join(format!("rocm{}-hipblaslt{blaslt}", rocm_version(rocm)))
            .join("gemm.recipe"),
    )
}

/// `70125424` -> `7.1`.
///
/// HIP packs the runtime version as `(major * 100 + minor) * 100000 + patch`.
/// The patch is dropped: it moves on every point release and does not change
/// which tactics are valid, so including it would scatter recipes across
/// directories that all mean the same thing. hipBLASLt's version is kept whole
/// because that one does invalidate them.
fn rocm_version(packed: i32) -> String {
    let major = packed / 10_000_000;
    let minor = (packed / 100_000) % 100;
    format!("{major}.{minor}")
}

/// Parse `APXINF_HIP_GEMM_TACTICS`, reporting problems rather than failing.
///
/// A malformed or stale recipe must degrade to hipBLAS, not abort a model
/// load: the recipe is a performance input, and the backend is correct without
/// it. Every rejection is printed once, because a silently ignored recipe
/// looks exactly like a recipe that did not help.
fn load_recipe(path: &Path) -> Recipe {
    let mut recipe = Recipe {
        providers: BTreeMap::new(),
        tactics: BTreeMap::new(),
    };
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) => {
            // A missing default path is the normal state before anything has
            // been tuned, and must not be noise on every load.
            if error.kind() != std::io::ErrorKind::NotFound {
                eprintln!("apxinf-hip: cannot read {}: {error}", path.display());
            }
            return recipe;
        }
    };
    // The header states which device and libraries the tactics were measured
    // under. A mismatch is rejected whole: the per-shape trial launch would
    // catch a stale tactic one at a time, but reading the header costs nothing
    // and says so once instead of silently degrading on every shape.
    if let Some(reason) = header_mismatch(&text) {
        eprintln!("apxinf-hip: ignoring {}: {reason}", path.display());
        return recipe;
    }
    let tactic_bytes = unsafe { ffi::apxinf_hip_blaslt_tactic_bytes() };
    let mut rejected = 0usize;
    for (number, line) in text.lines().enumerate() {
        let line = line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        // dtype M K N provider [tactic_hex]
        // dtype M K N epilogue provider [tactic]
        let parsed = (|| -> Option<(Key, Provider, Option<Vec<u8>>)> {
            if fields.len() < 6 || fields[0] != "bf16" {
                return None;
            }
            let key = (
                fields[1].parse().ok()?,
                fields[2].parse().ok()?,
                fields[3].parse().ok()?,
                match fields[4] {
                    "none" => EPILOGUE_NONE,
                    "bias" => EPILOGUE_BIAS,
                    "bias_gelu" => EPILOGUE_BIAS_GELU_TANH,
                    "bias_relu" => EPILOGUE_BIAS_RELU,
                    _ => return None,
                },
            );
            let provider = match fields[5] {
                "lt" | "hipblaslt" => Provider::HipBlasLt,
                "hipblas" => Provider::HipBlas,
                _ => return None,
            };
            let tactic = match fields.get(6) {
                None => None,
                Some(hex) => Some(decode_hex(hex)?),
            };
            Some((key, provider, tactic))
        })();
        match parsed {
            Some((key, provider, tactic)) => {
                recipe.providers.insert(key, provider);
                if let Some(blob) = tactic {
                    if blob.len() as i32 == tactic_bytes && tactic_bytes > 0 {
                        recipe.tactics.insert(key, blob);
                    } else {
                        eprintln!(
                            "apxinf-hip: {}:{} tactic is {} bytes, this hipBLASLt wants {}; \
                             using its heuristic for {}x{}x{}",
                            path.display(),
                            number + 1,
                            blob.len(),
                            tactic_bytes,
                            key.0,
                            key.1,
                            key.2
                        );
                    }
                }
            }
            None => rejected += 1,
        }
    }
    if rejected > 0 {
        eprintln!(
            "apxinf-hip: ignored {rejected} malformed line(s) in {}",
            path.display()
        );
    }
    recipe
}

/// `Some(reason)` when the recipe was not measured on what is running now.
fn header_mismatch(text: &str) -> Option<String> {
    let (rocm, blaslt) = library_versions().unwrap_or((0, 0));
    let want = format!(
        "{} rocm{} hipblaslt{blaslt}",
        env!("APXINF_HIP_ARCH"),
        rocm_version(rocm)
    );
    let found = text.lines().find_map(|line| {
        line.trim()
            .strip_prefix("# measured-on ")
            .map(str::to_owned)
    });
    match found {
        // An older hand-written recipe has no stamp. Its tactics are still
        // validated one at a time by the trial launch, so it is accepted with
        // a warning rather than discarded.
        None => {
            eprintln!(
                "apxinf-hip: recipe has no `# measured-on` line; its tactics will be \
                 checked one at a time instead"
            );
            None
        }
        Some(stamp) if stamp.trim() == want => None,
        Some(stamp) => Some(format!("measured on `{}`, running `{want}`", stamp.trim())),
    }
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn decode_hex(text: &str) -> Option<Vec<u8>> {
    if text.len() % 2 != 0 {
        return None;
    }
    (0..text.len() / 2)
        .map(|i| u8::from_str_radix(&text[i * 2..i * 2 + 2], 16).ok())
        .collect()
}

/// Hand every recorded tactic to the handle, once per handle.
fn register_tactics(ctx: &HipContext, recipe: &Recipe) {
    let handle = ctx.blaslt();
    if handle.is_null() || recipe.tactics.is_empty() {
        return;
    }
    let address = handle as usize;
    if TACTICS_REGISTERED.load(Ordering::Relaxed) == address {
        return;
    }
    for (&(m, k, n, epilogue), blob) in &recipe.tactics {
        // Shapes too large for the C ABI's `int` are skipped here and
        // rejected by `gemm` itself, which is where the error belongs.
        let (Ok(m), Ok(n), Ok(k)) = (int(m), int(n), int(k)) else {
            continue;
        };
        let status = unsafe {
            ffi::apxinf_hip_blaslt_set_tactic(
                handle,
                DTYPE_BF16,
                m,
                n,
                k,
                epilogue,
                blob.as_ptr(),
                blob.len() as i32,
            )
        };
        if status != 0 {
            eprintln!("apxinf-hip: hipBLASLt rejected the tactic for {m}x{k}x{n}");
        }
    }
    TACTICS_REGISTERED.store(address, Ordering::Relaxed);
}

/// Measure both providers for a shape the recipe does not cover, and persist
/// the winner.
///
/// Runs at most once per shape per process. Returns the chosen provider; on
/// any failure it returns `None` and the caller uses the untuned rule, because
/// a tuning problem must not become an inference problem.
///
/// `a` and `b` are the caller's own inputs, read but never written: the
/// measurement allocates its own output and its own weight ring on the C++
/// side. The request that triggers tuning therefore gets the right answer, a
/// few hundred milliseconds late.
fn tune_shape(
    ctx: &HipContext,
    key: Key,
    a: *const c_void,
    b: *const c_void,
    bias: *const c_void,
) -> Option<Provider> {
    let (m, k, n, epilogue) = key;
    let mut table = tuned().lock().ok()?;
    if let Some((provider, _)) = table.get(&key) {
        return Some(*provider);
    }

    let tactic_bytes = unsafe { ffi::apxinf_hip_blaslt_tactic_bytes() };
    if tactic_bytes <= 0 {
        return None;
    }
    let mut blob = vec![0u8; tactic_bytes as usize];
    let (mut blas_ms, mut lt_ms, mut worst) = (-1.0f64, -1.0f64, 0.0f64);
    let status = unsafe {
        ffi::apxinf_hip_gemm_tune(
            ctx.blaslt(),
            ctx.blas(),
            DTYPE_BF16,
            int(m).ok()?,
            int(n).ok()?,
            int(k).ok()?,
            a,
            b,
            epilogue,
            bias,
            &mut blas_ms,
            &mut lt_ms,
            blob.as_mut_ptr(),
            tactic_bytes,
            &mut worst,
        )
    };
    if status != 0 || blas_ms <= 0.0 {
        eprintln!(
            "apxinf-hip: could not tune {m}x{k}x{n}{}; leaving it on hipBLAS",
            epilogue_suffix(epilogue)
        );
        table.insert(key, (Provider::HipBlas, Vec::new()));
        return Some(Provider::HipBlas);
    }

    // The same margin the offline probe uses. Below it the two providers are
    // indistinguishable given run-to-run spread, and switching still costs a
    // warm-up query and a cached descriptor set.
    let ratio = if lt_ms > 0.0 { blas_ms / lt_ms } else { 0.0 };
    let entry = if ratio >= ROUTE_MARGIN {
        (Provider::HipBlasLt, blob)
    } else {
        (Provider::HipBlas, Vec::new())
    };
    eprintln!(
        "apxinf-hip: tuned {m}x{k}x{n}{}: hipBLAS {blas_ms:.4} ms, hipBLASLt {} -> {} \
         (worst candidate drift {worst:.1e})",
        epilogue_suffix(epilogue),
        if lt_ms > 0.0 {
            format!("{lt_ms:.4} ms, {ratio:.2}x")
        } else {
            "none".into()
        },
        if entry.0 == Provider::HipBlasLt {
            "hipBLASLt"
        } else {
            "hipBLAS"
        },
    );
    // Hand the tactic to the handle here, while the blob is in hand and the
    // lock is already held. Doing it from the dispatch path instead would mean
    // taking this lock on every GEMM, for the one call in a thousand that has
    // something new to register.
    if let (Provider::HipBlasLt, Ok(im), Ok(i_n), Ok(ik)) = (entry.0, int(m), int(n), int(k)) {
        unsafe {
            ffi::apxinf_hip_blaslt_set_tactic(
                ctx.blaslt(),
                DTYPE_BF16,
                im,
                i_n,
                ik,
                epilogue,
                entry.1.as_ptr(),
                entry.1.len() as i32,
            );
        }
    }
    table.insert(key, entry.clone());

    let routing = routing();
    if let Some(path) = routing.recipe_path.as_deref() {
        if let Err(error) = persist(path, &routing.recipe, &table) {
            eprintln!(
                "apxinf-hip: tuned {m}x{k}x{n} but could not write {}: {error}",
                path.display()
            );
        }
    }
    Some(entry.0)
}

/// Write the whole tuned table through a temporary file and a rename.
///
/// Rewritten after every shape rather than once at the end: a tuning pass runs
/// inside a real inference and can be interrupted, and a partial recipe is
/// still worth keeping. The rename is what makes an interrupted write leave
/// the previous file intact rather than a truncated one.
/// `""`, or ` +bias_gelu`, for messages.
fn epilogue_suffix(epilogue: i32) -> &'static str {
    match epilogue {
        EPILOGUE_BIAS => " +bias",
        EPILOGUE_BIAS_GELU_TANH => " +bias_gelu",
        EPILOGUE_BIAS_RELU => " +bias_relu",
        _ => "",
    }
}

fn epilogue_name(epilogue: i32) -> &'static str {
    match epilogue {
        EPILOGUE_BIAS => "bias",
        EPILOGUE_BIAS_GELU_TANH => "bias_gelu",
        EPILOGUE_BIAS_RELU => "bias_relu",
        _ => "none",
    }
}

/// Write the recipe: everything that was loaded, plus everything tuned here.
///
/// The merge is the point. A process tunes only the keys it missed, so writing
/// the tuned table alone discards every entry that came from disk -- a machine
/// with a complete recipe that autotunes one new shape loses the rest and
/// drifts back toward hipBLAS, which presents as "autotune made it slower".
/// This project wrote 5 rows over a 31-row recipe exactly once before the
/// merge existed.
///
/// Rewritten after every shape rather than once at the end: a tuning pass runs
/// inside a real inference and can be interrupted, and a partial recipe is
/// still worth keeping. The rename is what makes an interrupted write leave
/// the previous file intact rather than a truncated one.
fn persist(
    path: &Path,
    loaded: &Recipe,
    tuned: &BTreeMap<Key, (Provider, Vec<u8>)>,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let (rocm, blaslt) = library_versions().unwrap_or((0, 0));
    let mut text = String::from("# apxinf-hip gemm recipe v1\n");
    text.push_str(&format!(
        "# measured-on {} rocm{} hipblaslt{blaslt}\n",
        env!("APXINF_HIP_ARCH"),
        rocm_version(rocm)
    ));
    text.push_str(
        "#\n\
         # Written by APXINF_HIP_GEMM_AUTOTUNE. One line per GEMM shape: the provider\n\
         # that measured faster, and for hipBLASLt the serialized algorithm it won with.\n\
         # Delete this file to re-tune; it is a cache, not a source.\n\
         #\n\
         # The epilogue column is part of the key: a shape can measure faster on\n\
         # hipBLAS as a plain GEMM and faster on hipBLASLt with the bias and\n\
         # activation fused in, because the fused form also deletes a kernel.\n\
         #\n\
         # dtype M K N epilogue provider [tactic]\n",
    );
    // Loaded first, then tuned, so a key measured in this process replaces the
    // one read from disk rather than being dropped by it.
    let mut merged: BTreeMap<Key, (Provider, Vec<u8>)> = loaded
        .providers
        .iter()
        .map(|(&key, &provider)| {
            let blob = loaded.tactics.get(&key).cloned().unwrap_or_default();
            (key, (provider, blob))
        })
        .collect();
    merged.extend(tuned.iter().map(|(&k, v)| (k, v.clone())));

    for (&(m, k, n, epilogue), (provider, blob)) in &merged {
        let epi = epilogue_name(epilogue);
        match provider {
            Provider::HipBlasLt if !blob.is_empty() => {
                text.push_str(&format!("bf16 {m} {k} {n} {epi} lt {}\n", encode_hex(blob)))
            }
            // No tactic recorded: hipBLASLt still serves the shape, by its own
            // heuristic. Measurement puts that at 78% of the achievable gain,
            // which is worth keeping over a silent demotion to hipBLAS.
            Provider::HipBlasLt => text.push_str(&format!("bf16 {m} {k} {n} {epi} lt\n")),
            Provider::HipBlas => text.push_str(&format!("bf16 {m} {k} {n} {epi} hipblas\n")),
        }
    }
    // Same directory as the destination, so the rename stays within one
    // filesystem and is therefore atomic.
    let temporary = path.with_extension(format!("recipe.{}.tmp", std::process::id()));
    std::fs::write(&temporary, text)?;
    std::fs::rename(&temporary, path)
}

fn provider_for(routing: &Routing, key: Key) -> Provider {
    let m = key.0;
    match routing.mode {
        Mode::HipBlasOnly => Provider::HipBlas,
        Mode::LtOnly => Provider::HipBlasLt,
        Mode::Auto => match routing.recipe.providers.get(&key) {
            Some(provider) => *provider,
            // A miss against a recipe that covers other shapes means this
            // shape was never measured here -- a different model, or one
            // whose shapes changed. Unmeasured switches lose 1.6x as readily
            // as they win it, so an unlisted shape stays on hipBLAS.
            None if !routing.recipe.providers.is_empty() => Provider::HipBlas,
            // With no recipe at all there is nothing to be consistent with,
            // and the one rule worth guessing is the one the measurement
            // supports: the tall-skinny shapes are where hipBLAS leaves the
            // most bandwidth unused.
            None => {
                if m <= 64 {
                    Provider::HipBlasLt
                } else {
                    Provider::HipBlas
                }
            }
        },
    }
}

/// Which provider each recorded shape resolves to, for reports and tests.
/// Which provider each recorded shape resolves to, for reports and tests.
pub fn routed_shapes() -> Vec<((usize, usize, usize, i32), Provider)> {
    let routing = routing();
    let mut keys: BTreeSet<Key> = routing.recipe.providers.keys().copied().collect();
    keys.extend(
        recorded_shapes()
            .into_iter()
            .map(|((m, k, n), _)| (m, k, n, EPILOGUE_NONE)),
    );
    keys.into_iter()
        .map(|key| (key, provider_for(routing, key)))
        .collect()
}

/// `out[M, N] = a[M, K] @ b[K, N]` in BF16 with F32 accumulation. Leading
/// dimensions of `a` and `out` fold into M; `b` is a 2-D weight.
pub fn gemm(ctx: &HipContext, a: &Tensor, b: &Tensor, out: &Tensor) -> Result<()> {
    dispatch(ctx, a, b, out, None, Activation::None).map(|_| ())
}

/// `out[M, N] = activation(a[M, K] @ b[K, N] + bias[N])`, fused where the
/// provider can do it.
///
/// Prefer this over `gemm` followed by `bias_activation` wherever the model's
/// shape is a GEMM with a bias: hipBLASLt applies both inside the GEMM
/// epilogue, which removes a kernel launch, removes a round trip of the output
/// through memory, and -- because the epilogue works on the F32 accumulator --
/// removes two BF16 roundings that the separate path performs.
///
/// Falls back to exactly those two calls when the provider cannot fuse: when
/// hipBLASLt is unavailable, when the recipe routes this shape to hipBLAS
/// (whose `hipblasGemmEx` has no epilogue), or when the activation has no
/// epilogue. `Activation::GeluErf` is the case that matters there -- the
/// epilogue implements the tanh approximation, and substituting it for the
/// exact form would be a silent change of model.
pub fn gemm_bias_activation(
    ctx: &HipContext,
    a: &Tensor,
    b: &Tensor,
    bias: Option<&Tensor>,
    activation: Activation,
    out: &Tensor,
) -> Result<()> {
    let Some(bias) = bias else {
        // No bias means no epilogue to fuse into; the activation, if any,
        // still needs its own pass.
        gemm(ctx, a, b, out)?;
        return if matches!(activation, Activation::None) {
            Ok(())
        } else {
            super::bias_activation(ctx, out, None, activation, out)
        };
    };
    if dispatch(ctx, a, b, out, Some(bias), activation)? {
        return Ok(());
    }
    // In place: `bias_activation` reads and writes the same index, so the
    // product can be rewritten where it lies rather than through a second
    // buffer. That also keeps the fused and unfused paths using the same
    // arena allocation, so a caller does not have to know which one ran.
    super::bias_activation(ctx, out, Some(bias), activation, out)
}

/// Issue the GEMM, returning whether a requested epilogue was applied.
///
/// `Ok(false)` means the product is in `out` but the bias and activation are
/// not, and the caller must run them separately. That is a normal outcome, not
/// an error: the provider for a shape is a measured choice and hipBLAS has no
/// epilogue.
fn dispatch(
    ctx: &HipContext,
    a: &Tensor,
    b: &Tensor,
    out: &Tensor,
    bias: Option<&Tensor>,
    activation: Activation,
) -> Result<bool> {
    let (m, k) = rows_cols(a)?;
    let weight = b.shape().dims();
    let (n_out_rows, n) = rows_cols(out)?;
    if weight.len() != 2 || weight[0] != k || weight[1] != n || n_out_rows != m {
        return Err(Error::Other(format!(
            "apxinf-hip gemm: {} @ {} -> {} is not [M, K] @ [K, N] -> [M, N]",
            a.shape(),
            b.shape(),
            out.shape()
        )));
    }
    if let Some(bias) = bias {
        if bias.shape().numel() != n {
            return Err(Error::Other(format!(
                "apxinf-hip gemm: bias {} does not match the {n} output features",
                bias.shape()
            )));
        }
    }
    if let Some(cell) = recorder() {
        if let Ok(mut map) = cell.lock() {
            *map.entry((m, k, n)).or_insert(0) += 1;
        }
    }
    let (pa, pb, pc) = (bf16_ptr(ctx, a)?, bf16_ptr(ctx, b)?, bf16_ptr(ctx, out)?);
    let (im, i_n, ik) = (int(m)?, int(n)?, int(k)?);

    // An epilogue needs both a bias to fuse and an activation the library
    // implements; without either, this is the plain shape key.
    let wanted = match (bias, activation.epilogue()) {
        (Some(_), Some(code)) => code,
        _ => EPILOGUE_NONE,
    };
    let bias_ptr = match (bias, wanted) {
        (Some(bias), e) if e != EPILOGUE_NONE => bf16_ptr(ctx, bias)? as *const c_void,
        _ => std::ptr::null(),
    };

    let routing = routing();
    let resolve = |key: Key, bias_ptr: *const c_void| -> Provider {
        if routing.autotune
            && !ctx.blaslt().is_null()
            && !routing.recipe.providers.contains_key(&key)
        {
            tune_shape(ctx, key, pa, pb, bias_ptr).unwrap_or_else(|| provider_for(routing, key))
        } else {
            provider_for(routing, key)
        }
    };

    // Only hipBLASLt has an epilogue, so a fused key that measured faster on
    // hipBLAS means "do not fuse this one" -- it does not mean the GEMM
    // itself belongs on hipBLAS. Falling back has to re-resolve against the
    // plain key, or a shape that fuses badly but runs 1.8x faster on
    // hipBLASLt as a plain GEMM would quietly lose that too.
    let (epilogue, bias_ptr, provider) = match wanted {
        EPILOGUE_NONE => {
            let key = (m, k, n, EPILOGUE_NONE);
            (
                EPILOGUE_NONE,
                std::ptr::null(),
                resolve(key, std::ptr::null()),
            )
        }
        code => match resolve((m, k, n, code), bias_ptr) {
            Provider::HipBlasLt => (code, bias_ptr, Provider::HipBlasLt),
            Provider::HipBlas => {
                let key = (m, k, n, EPILOGUE_NONE);
                (
                    EPILOGUE_NONE,
                    std::ptr::null(),
                    resolve(key, std::ptr::null()),
                )
            }
        },
    };
    let key: Key = (m, k, n, epilogue);
    let _ = key;

    if routing.mode != Mode::HipBlasOnly
        && provider == Provider::HipBlasLt
        && !ctx.blaslt().is_null()
    {
        register_tactics(ctx, &routing.recipe);
        // `kInvalidArgument` means this key has no usable hipBLASLt algorithm,
        // not that the call was wrong, so it falls through to hipBLAS rather
        // than failing the inference. Any other non-zero status is a real
        // hipBLASLt failure and is reported.
        let status = unsafe {
            ffi::apxinf_hip_gemm_lt(
                ctx.blaslt(),
                DTYPE_BF16,
                im,
                i_n,
                ik,
                pa,
                pb,
                pc,
                epilogue,
                bias_ptr,
            )
        };
        if status == 0 {
            DISPATCHES[1].fetch_add(1, Ordering::Relaxed);
            return Ok(epilogue != EPILOGUE_NONE);
        }
        if status != ffi::INVALID_ARGUMENT {
            check("gemm (hipBLASLt)", status)?;
        }
    }
    if routing.mode != Mode::HipBlasOnly {
        DISPATCHES[0].fetch_add(1, Ordering::Relaxed);
    }

    check("gemm", unsafe {
        ffi::apxinf_hip_gemm(ctx.blas(), DTYPE_BF16, im, i_n, ik, pa, pb, pc)
    })?;
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `persist` writes, `load_recipe` must read back unchanged.
    ///
    /// These two are the only places the recipe format exists, and a drift
    /// between them degrades silently: a recipe that parses to nothing looks
    /// exactly like a machine that was never tuned. The round trip is cheap to
    /// assert and is the one thing that cannot be noticed by running the
    /// model, because the model is still correct either way -- only slower.
    #[test]
    fn a_written_recipe_reads_back_with_the_same_choices() {
        let directory =
            std::env::temp_dir().join(format!("apxinf-hip-recipe-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("gemm.recipe");

        let mut written: BTreeMap<Key, (Provider, Vec<u8>)> = BTreeMap::new();
        written.insert(
            (41, 6144, 1536, EPILOGUE_NONE),
            (Provider::HipBlasLt, vec![0x52, 0x15, 0x00, 0xff]),
        );
        // The same shape with a fused epilogue is a different key and may hold
        // a different provider -- the case that makes the epilogue part of the
        // key rather than a flag beside it.
        written.insert(
            (41, 6144, 1536, EPILOGUE_BIAS_GELU_TANH),
            (Provider::HipBlas, Vec::new()),
        );
        written.insert(
            (512, 1024, 3072, EPILOGUE_NONE),
            (Provider::HipBlas, Vec::new()),
        );
        // A single-byte tactic: shorter than any real algorithm, so it must be
        // rejected as a tactic while the provider choice on that line survives.
        written.insert(
            (1, 132, 1024, EPILOGUE_BIAS),
            (Provider::HipBlasLt, vec![0x01]),
        );
        persist(&path, &empty_recipe(), &written).unwrap();

        let read = load_recipe(&path);
        assert_eq!(
            read.providers.get(&(41, 6144, 1536, EPILOGUE_NONE)),
            Some(&Provider::HipBlasLt)
        );
        assert_eq!(
            read.providers
                .get(&(41, 6144, 1536, EPILOGUE_BIAS_GELU_TANH)),
            Some(&Provider::HipBlas),
            "the two epilogues of one shape must not collapse into a single entry"
        );
        assert_eq!(
            read.providers.get(&(512, 1024, 3072, EPILOGUE_NONE)),
            Some(&Provider::HipBlas)
        );
        assert_eq!(read.providers.len(), 4);

        let expected_bytes = unsafe { ffi::apxinf_hip_blaslt_tactic_bytes() };
        if expected_bytes == 4 {
            assert_eq!(
                read.tactics
                    .get(&(41, 6144, 1536, EPILOGUE_NONE))
                    .map(Vec::as_slice),
                Some([0x52u8, 0x15, 0x00, 0xff].as_slice())
            );
        }
        // Whatever the real size is, a one-byte blob is never it, and a
        // hipBLAS row has nothing to carry.
        assert!(!read.tactics.contains_key(&(1, 132, 1024, EPILOGUE_BIAS)));
        assert!(!read.tactics.contains_key(&(512, 1024, 3072, EPILOGUE_NONE)));

        std::fs::remove_dir_all(&directory).ok();
    }

    /// A recipe written before the epilogue column must be refused, not
    /// misread.
    ///
    /// Its five-field lines would otherwise parse with the provider name
    /// landing in the epilogue column, which fails the match and drops every
    /// line -- the same outcome, but reported as "malformed" rather than as a
    /// version difference. Either way nothing is silently misrouted, and that
    /// is what this pins.
    #[test]
    fn a_recipe_without_the_epilogue_column_is_not_misread() {
        let directory =
            std::env::temp_dir().join(format!("apxinf-hip-old-recipe-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("gemm.recipe");
        std::fs::write(
            &path,
            "# apxinf-hip gemm recipe v1\n\
             bf16 41 6144 1536 lt 52150000\n\
             bf16 512 1024 3072 hipblas\n",
        )
        .unwrap();
        let read = load_recipe(&path);
        assert!(
            read.providers.is_empty(),
            "a pre-epilogue recipe parsed to {} entries; one of them routes a shape by a \
             measurement that did not include the epilogue",
            read.providers.len()
        );
        std::fs::remove_dir_all(&directory).ok();
    }

    fn empty_recipe() -> Recipe {
        Recipe {
            providers: BTreeMap::new(),
            tactics: BTreeMap::new(),
        }
    }

    /// Tuning one key must not delete the rest of the recipe.
    ///
    /// A process only tunes what it missed, so persisting the tuned table on
    /// its own throws away everything that came from disk. The failure is
    /// silent -- the model stays correct, the lost shapes just fall back to
    /// hipBLAS -- and it presents as autotune making things slower, which is
    /// the last place anyone looks. This fixes the shape of that bug in place.
    #[test]
    fn tuning_one_key_preserves_the_rest_of_the_recipe() {
        let directory =
            std::env::temp_dir().join(format!("apxinf-hip-merge-{}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let path = directory.join("gemm.recipe");

        // Thirty-one keys on disk, one of them tuned again in this process.
        let mut on_disk = Recipe {
            providers: BTreeMap::new(),
            tactics: BTreeMap::new(),
        };
        for i in 0..31 {
            let key = (41 + i, 1536, 1536, EPILOGUE_NONE);
            on_disk.providers.insert(key, Provider::HipBlas);
        }
        let retuned = (41, 1536, 1536, EPILOGUE_NONE);
        let mut tuned: BTreeMap<Key, (Provider, Vec<u8>)> = BTreeMap::new();
        tuned.insert(retuned, (Provider::HipBlasLt, Vec::new()));

        persist(&path, &on_disk, &tuned).unwrap();
        let read = load_recipe(&path);

        assert_eq!(
            read.providers.len(),
            31,
            "writing after tuning 1 key left {} of 31 entries",
            read.providers.len()
        );
        assert_eq!(
            read.providers.get(&retuned),
            Some(&Provider::HipBlasLt),
            "the freshly tuned key must win over the one read from disk"
        );
        assert_eq!(
            read.providers.get(&(60, 1536, 1536, EPILOGUE_NONE)),
            Some(&Provider::HipBlas)
        );
        std::fs::remove_dir_all(&directory).ok();
    }

    #[test]
    fn hex_round_trips_and_rejects_malformed_input() {
        assert_eq!(
            decode_hex(&encode_hex(&[0, 1, 0x7f, 0xff])),
            Some(vec![0, 1, 0x7f, 0xff])
        );
        assert_eq!(decode_hex("abc"), None, "odd length");
        assert_eq!(decode_hex("zz"), None, "not hex");
        assert_eq!(decode_hex(""), Some(Vec::new()));
    }

    /// `70125424` is what ROCm 7.1 reports; the patch must not reach the path.
    #[test]
    fn the_rocm_version_in_a_recipe_path_drops_the_patch() {
        assert_eq!(rocm_version(70125424), "7.1");
        assert_eq!(rocm_version(60342134), "6.3");
    }
}

//! Environment variables read by the current CUDA and MUSA backends.
//!
//! CUDA uses `apxinf-cuda-new`: both its Rust runtime and native C++/CUDA
//! implementation read configuration. MUSA retains an independent runtime and
//! PI0.5 model port. The legacy `apxinf-cuda` tree has been removed, so its
//! allocation and attention switches are no longer CUDA counterparts.
//!
//! This table checks literal environment reads in those source trees, including
//! diagnostic reads in inline tests but excluding dedicated test directories.
//! It records differences without requiring the backends to share an API.
//! A declared counterpart must exist on the other backend; the table cannot
//! establish that two switches have identical defaults or numerical effects.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// One `APXINF_*` variable read by one or both backends.
struct Knob {
    name: &'static str,
    /// The same knob on the other backend, when there is one. `None` means this
    /// backend has a knob the other does not, which the test enforces rather
    /// than assumes: the counterpart named here has to be read by the other
    /// crate.
    counterpart: Option<&'static str>,
    /// One sentence a reader can act on.
    note: &'static str,
}

const KNOBS: &[Knob] = &[
    // Current CUDA runtime (Rust and native C++/CUDA).
    Knob {
        name: "APXINF_DISABLE_FA2_SPLITKV",
        counterpart: None,
        note: "disable CUDA FlashAttention-2 split-KV candidates; MUSA has no equivalent switch",
    },
    Knob {
        name: "APXINF_FA2_DECODE_SPLITS",
        counterpart: None,
        note: "select one or five splits for the fixed CUDA FA2 decode kernel",
    },
    Knob {
        name: "APXINF_FP8_GEMV_SHARED",
        counterpart: None,
        note: "enable the shared-memory CUDA FP8 GEMV path for supported shapes with =1",
    },
    Knob {
        name: "APXINF_FP8_NATIVE_PAIR",
        counterpart: None,
        note: "enable native paired CUDA FP8 quantization for aligned inputs with =1",
    },
    Knob {
        name: "APXINF_GDN_CONV_PAIR",
        counterpart: None,
        note: "enable paired-channel CUDA GDN prefill convolution with =1",
    },
    Knob {
        name: "APXINF_GDN_PREFILL_TILE",
        counterpart: None,
        note: "set the positive token tile size for CUDA GDN prefill convolution",
    },
    Knob {
        name: "APXINF_GDN_PREFILL_TILED",
        counterpart: None,
        note: "select tiled CUDA GDN prefill implementations; no MUSA GDN port exists",
    },
    Knob {
        name: "APXINF_GDN_PREPARE_PARALLEL",
        counterpart: None,
        note: "select the CUDA GDN prefill preparation implementation; no MUSA equivalent",
    },
    Knob {
        name: "APXINF_GDN_SMEM_VARIANT",
        counterpart: None,
        note: "select the CUDA GDN shared-memory precision variant; the default is eight",
    },
    Knob {
        name: "APXINF_GEMM_CUBLASLT_NVFP4",
        counterpart: None,
        note: "admit native cuBLASLt NVFP4 GEMM candidates on supported toolkits with =1",
    },
    Knob {
        name: "APXINF_GEMM_FP8_F32_PROJECTION",
        counterpart: None,
        note: "force CUDA FP8 projection outputs to F32 for numerical comparison",
    },
    Knob {
        name: "APXINF_GEMM_REPORT",
        counterpart: None,
        note: "report CUDA GEMM choices and weight layout transitions on stderr",
    },
    Knob {
        name: "APXINF_GR00T_BF16_LEGACY_QK_MROPE_THREADS",
        counterpart: None,
        note: "restore legacy GR00T QK MRoPE thread geometry on CUDA SM87",
    },
    Knob {
        name: "APXINF_GR00T_BF16_LEGACY_VISION_POINTWISE_PACK",
        counterpart: None,
        note: "disable the CUDA SM87 GR00T vectorized vision pointwise packing path",
    },
    Knob {
        name: "APXINF_GR00T_FP8_LEGACY_STATIC_QUANT",
        counterpart: None,
        note: "disable the packed CUDA SM110 GR00T static FP8 quantization path",
    },
    Knob {
        name: "APXINF_GR00T_LEGACY_QK_MROPE_THREADS",
        counterpart: None,
        note: "restore legacy GR00T QK MRoPE thread geometry on CUDA SM110",
    },
    Knob {
        name: "APXINF_GR00T_W8A8_LEGACY_VISION_POINTWISE_PACK",
        counterpart: None,
        note: "disable the CUDA SM87 GR00T vectorized vision pointwise packing path",
    },
    Knob {
        name: "APXINF_NVFP4_VECTOR_QUANT",
        counterpart: None,
        note: "override CUDA NVFP4 vector quantization selection for supported input shapes",
    },
    Knob {
        name: "APXINF_PREFILL_SDPA_LEGACY",
        counterpart: None,
        note: "keep the legacy CUDA prefill SDPA route instead of the native attention fast path",
    },
    Knob {
        name: "APXINF_QWEN38_NVFP4_DP4A",
        counterpart: None,
        note: "enable the CUDA NVFP4 DP4A GEMV path for aligned inputs with =1",
    },
    Knob {
        name: "APXINF_QWEN38_VECTOR_ELEMENTWISE",
        counterpart: None,
        note: "enable vectorized CUDA elementwise and quantization paths for supported inputs with =1",
    },
    // MUSA backend and its independent PI0.5 port.
    Knob {
        name: "APXINF_MUSA_POISON_ALLOC",
        counterpart: None,
        note: "fill MUSA allocations with 0xFF to expose reads before writes; cuda-new no longer reads APXINF_CUDA_POISON_ALLOC",
    },
    Knob {
        name: "APXINF_MUSA_ALLOC_CACHE",
        counterpart: None,
        note: "opt in to the MUSA stream-ordered free list; unset or =0 disables it; cuda-new has no matching environment switch",
    },
    Knob {
        name: "APXINF_MUSA_ALLOC_CACHE_MB",
        counterpart: None,
        note: "cap bytes retained by the MUSA allocation cache in MiB, default 4096; cuda-new has no matching environment switch",
    },
    Knob {
        name: "APXINF_MUSA_BF16_MHA",
        counterpart: None,
        note: "=legacy selects the MUSA legacy vision MHA; cuda-new no longer reads APXINF_MHA_NAIVE",
    },
    Knob {
        name: "APXINF_MUSA_FA_REPLAY_DIR",
        counterpart: None,
        note: "directory a captured MATE reference is replayed from, for \
               `mqa_bf16_reproduces_the_mate_reference`. The equivalent CUDA evidence \
               is a golden bundle rather than a knob",
    },
    Knob {
        name: "APXINF_MUSA_OUTPUT_FILL",
        counterpart: None,
        note: "=poison fills MUSA operator outputs outside graph capture to detect unwritten reads; the default leaves outputs dirty",
    },
    Knob {
        name: "APXINF_MUSA_WORKSPACE_SLACK",
        counterpart: None,
        note: "pad every arena allocation, so an out-of-bounds write lands in unused \
               memory instead of the next allocation. A diagnostic for a suspected \
               kernel overrun",
    },
    Knob {
        name: "APXINF_INT8_EQUALIZATION",
        counterpart: None,
        note: "per-channel equalization table applied at INT8 pack time; CUDA's W8A8 \
               route does not fold one through",
    },
    Knob {
        name: "APXINF_INT8_DYNAMIC_EXCLUDE",
        counterpart: None,
        note: "comma-separated projection sites the INT8 route must leave in BF16, so a \
               shape the tiles cannot serve is a configuration choice rather than a failure",
    },
    Knob {
        name: "APXINF_PI05_INT8_STATS",
        counterpart: None,
        note: "record per-row activation levels for the equalization table. Reads \
               activations back to the host, so it must stay off while timing or capturing",
    },
    Knob {
        name: "APXINF_PI05_INT8_CHANNELS",
        counterpart: None,
        note: "where `APXINF_PI05_INT8_STATS` writes its per-channel maxima. Same cost: \
               a host round trip per recorded row",
    },
    Knob {
        name: "APXINF_PI05_TEST_CHECKPOINT",
        counterpart: None,
        note: "checkpoint path for the ignored real-weight device test; test-only, and \
               never read on a production path",
    },
];

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("the crate lives inside the workspace")
}

/// Literal `APXINF_*` arguments of Rust and native environment reads.
fn names_in_source(text: &str, calls: &[&str], into: &mut BTreeSet<String>) {
    for call in calls {
        for (index, _) in text.match_indices(call) {
            // Computed names and function references are not literal reads.
            let rest = text[index + call.len()..].trim_start();
            let Some(rest) = rest.strip_prefix('(') else {
                continue;
            };
            let Some(rest) = rest.trim_start().strip_prefix('"') else {
                continue;
            };
            let Some(close) = rest.find('"') else {
                continue;
            };
            let name = &rest[..close];
            if name.starts_with("APXINF_") {
                into.insert(name.to_string());
            }
        }
    }
}

/// Read both Rust and native implementation sources, failing on missing trees.
fn environment_names(root: &Path, into: &mut BTreeSet<String>) {
    let entries = std::fs::read_dir(root)
        .unwrap_or_else(|error| panic!("cannot scan {}: {error}", root.display()));
    for entry in entries {
        let entry = entry.expect("read source directory entry");
        let path = entry.path();
        if path.is_dir() {
            // Integration tests read nothing a user configures.
            if path.file_name().is_some_and(|name| name == "tests") {
                continue;
            }
            environment_names(&path, into);
            continue;
        }
        let calls: &[&str] = match path.extension().and_then(|extension| extension.to_str()) {
            Some("rs") => &["env::var", "env::var_os"],
            Some("c" | "cc" | "cpp" | "h" | "hpp" | "cu" | "cuh") => &["getenv"],
            _ => continue,
        };
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
        names_in_source(&text, calls, into);
    }
}

/// The names a source tree actually reads.
fn read_names(source_dir: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    environment_names(&workspace_root().join(source_dir), &mut names);
    names
}

/// Current CUDA implementation, MUSA implementation and MUSA model port.
fn backend_knobs() -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
    let mut cuda = read_names("crates/apxinf-cuda-new/src");
    cuda.extend(read_names("crates/apxinf-cuda-new/native"));
    (
        cuda,
        read_names("crates/apxinf-musa/src"),
        read_names("crates/apxinf-model/src/musa_pi05"),
    )
}

#[test]
fn the_table_is_the_knob_surface_and_the_knob_surface_is_the_table() {
    let (cuda, musa, model) = backend_knobs();

    let declared: BTreeSet<&str> = KNOBS.iter().map(|knob| knob.name).collect();
    let read: BTreeSet<&str> = cuda
        .iter()
        .chain(musa.iter())
        .chain(model.iter())
        .map(String::as_str)
        .collect();

    let undocumented: Vec<&&str> = read.difference(&declared).collect();
    assert!(
        undocumented.is_empty(),
        "these environment variables are read by a GPU backend and are not in KNOBS: \
         {undocumented:?}.\n\
         Add each one with the counterpart it has on the other backend, or `None` and a \
         sentence saying why there is none. That sentence is the whole point — a knob a \
         user can set on one backend and not the other is a defect this table exists to \
         make visible."
    );

    let stale: Vec<&&str> = declared.difference(&read).collect();
    assert!(
        stale.is_empty(),
        "KNOBS lists {stale:?} but nothing reads them any more. Removing a knob means \
         removing its row, or the table stops describing the code."
    );
}

/// The `note` field is read by people, not by this file, so it needs a property
/// a test can hold it to: every knob says something.
///
/// A row that is only a name and an arrow is a row that passes the checks above
/// and tells the reader nothing — which is the failure mode of a table that
/// exists to be read. Only the length is checked, deliberately: whether a
/// sentence is any *good* is not something a test can say, and a test that tried
/// would be a spell-check on prose.
#[test]
fn every_knob_says_what_it_does() {
    for knob in KNOBS {
        assert!(
            knob.note.split_whitespace().count() >= 5,
            "`{}` has no note worth the name: {:?}",
            knob.name,
            knob.note
        );
    }
}

#[test]
fn every_declared_counterpart_is_read_by_the_other_backend() {
    let (cuda, musa, model) = backend_knobs();

    let mut broken = Vec::new();
    for knob in KNOBS {
        let Some(counterpart) = knob.counterpart else {
            continue;
        };
        // The named counterpart has to exist on the other backend.
        // Both rows of a pair name each other, so a one-sided edit
        // fails on the row that stayed behind.
        if !cuda.contains(knob.name) && !musa.contains(knob.name) && !model.contains(knob.name) {
            continue; // the first assertion already reported this
        }
        let counterpart_exists = (cuda.contains(knob.name)
            && (musa.contains(counterpart) || model.contains(counterpart)))
            || ((musa.contains(knob.name) || model.contains(knob.name))
                && cuda.contains(counterpart));
        if !counterpart_exists {
            broken.push(format!("{} -> {counterpart}", knob.name));
        }
    }

    assert!(
        broken.is_empty(),
        "these rows name a counterpart that no backend reads any more: {broken:?}. \
         Either the other side dropped the knob — in which case this side's users are \
         setting something that does nothing there — or it was renamed."
    );
}

#[test]
fn a_pair_names_the_same_knob_from_both_sides() {
    // The table's shape is what lets the check stay honest: a row that claims a
    // counterpart asserts the pairing, and the pairing has to be symmetric or
    // one of the two rows is a lie.
    let mut asymmetric = Vec::new();
    for knob in KNOBS {
        let Some(counterpart) = knob.counterpart else {
            continue;
        };
        let Some(other) = KNOBS.iter().find(|other| other.name == counterpart) else {
            asymmetric.push(format!("{} -> {counterpart} (no such row)", knob.name));
            continue;
        };
        if other.counterpart != Some(knob.name) {
            asymmetric.push(format!(
                "{} -> {counterpart}, but that row points at {:?}",
                knob.name, other.counterpart
            ));
        }
    }

    assert!(
        asymmetric.is_empty(),
        "these pairings are one-sided: {asymmetric:?}"
    );
}

/// A guard on the scanner itself: if it silently stopped finding names, the two
/// tests above would pass by finding nothing and comparing nothing.
#[test]
fn the_scanner_finds_both_backends() {
    for (tree, representative) in [
        ("crates/apxinf-cuda-new/src", "APXINF_GEMM_REPORT"),
        (
            "crates/apxinf-cuda-new/native",
            "APXINF_DISABLE_FA2_SPLITKV",
        ),
        ("crates/apxinf-musa/src", "APXINF_MUSA_ALLOC_CACHE"),
        (
            "crates/apxinf-model/src/musa_pi05",
            "APXINF_INT8_EQUALIZATION",
        ),
    ] {
        assert!(
            read_names(tree).contains(representative),
            "{tree} did not yield {representative}"
        );
    }
}

#[test]
fn scanner_reads_rust_and_native_literal_arguments() {
    let source = r#"
        std::env::var("APXINF_RUST");
        std::env::var_os ("APXINF_RUST_OS");
        std::getenv("APXINF_CPP");
        getenv ( "APXINF_C" );
        std::env::var(computed_name);
        getenv(computed_name);
        std::env::var("UNRELATED");
    "#;
    let mut names = BTreeSet::new();
    names_in_source(source, &["env::var", "env::var_os", "getenv"], &mut names);
    assert_eq!(
        names,
        ["APXINF_RUST", "APXINF_RUST_OS", "APXINF_CPP", "APXINF_C"]
            .into_iter()
            .map(String::from)
            .collect()
    );
}

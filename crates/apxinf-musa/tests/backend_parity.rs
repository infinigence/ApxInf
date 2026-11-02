//! Every environment variable the GPU backends read, and what each has on the
//! other side.
//!
//! The backends are ports of one another and carry parallel infrastructure, and
//! an earlier version of this file was going to compare those files line by
//! line. It cannot: `buffer.rs` differs on 250 of its 393 code lines and most of
//! that is legitimate — a port is allowed to be better in places. A divergence
//! budget over those numbers would be stale on the day it was written and would
//! be "updated" rather than read.
//!
//! What *is* bounded and worth locking down is the **user-facing knob surface**.
//! A variable that works on one backend and not the other, or that means the
//! same thing under two spellings, is a defect a user meets and the repository
//! does not currently say anything about. The table below is the documentation
//! as well as the check, and it is read from three trees: the CUDA backend, the
//! MUSA backend, and the MUSA model port (`apxinf-model/src/musa_pi05`), whose
//! INT8 knobs are user-facing even though they live above the backend crate.
//!
//! It does not catch everything. `APXINF_CUDA_ALLOC_CACHE=0` and
//! `APXINF_MUSA_ALLOC_CACHE=0` disagreed about *polarity* while agreeing about
//! their names, and this table declared them counterparts while that was true.
//! What the table does is make the pairing visible at the one moment a person is
//! looking: when they add or move a knob.

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
    // ── The three the two backends genuinely share ─────────────────────────
    Knob {
        name: "APXINF_CUDA_POISON_ALLOC",
        counterpart: Some("APXINF_MUSA_POISON_ALLOC"),
        note: "fill every allocation with 0xFF so a read-before-write shows up in the result",
    },
    Knob {
        name: "APXINF_CUDA_ALLOC_CACHE",
        counterpart: Some("APXINF_MUSA_ALLOC_CACHE"),
        note: "opt in to the stream-ordered free list. On CUDA the cache is on and `=0` \
               turns it off; on MUSA it is off and any other value turns it on — the \
               defaults differ because only CUDA's has the measurement behind it, and \
               `=0` means \"off\" on both",
    },
    Knob {
        name: "APXINF_CUDA_ALLOC_CACHE_MB",
        counterpart: Some("APXINF_MUSA_ALLOC_CACHE_MB"),
        note: "cap on bytes the free list retains, default 4096",
    },
    // ── The same knob under two names ──────────────────────────────────────
    Knob {
        name: "APXINF_MHA_NAIVE",
        counterpart: Some("APXINF_MUSA_BF16_MHA"),
        note: "force the legacy vision MHA. **The spellings differ and so does the \
               polarity**: CUDA reads its presence, MUSA needs `=legacy`. Declared here \
               rather than renamed, because renaming either is a breaking change for \
               whoever scripts it — but a reader who knows one should know the other",
    },
    // ── MUSA's half of the three shared knobs ──────────────────────────────
    Knob {
        name: "APXINF_MUSA_POISON_ALLOC",
        counterpart: Some("APXINF_CUDA_POISON_ALLOC"),
        note: "see the CUDA row: the same diagnostic, the same `0xFF`",
    },
    Knob {
        name: "APXINF_MUSA_ALLOC_CACHE",
        counterpart: Some("APXINF_CUDA_ALLOC_CACHE"),
        note: "see the CUDA row. Off by default here, on by default there, and `=0` \
               turns it off on both",
    },
    Knob {
        name: "APXINF_MUSA_ALLOC_CACHE_MB",
        counterpart: Some("APXINF_CUDA_ALLOC_CACHE_MB"),
        note: "see the CUDA row: the same cap, the same default",
    },
    // ── CUDA-only: knobs whose subject the MUSA port has not reached ────────
    Knob {
        name: "APXINF_ATTN_COMPOSED_DECODE",
        counterpart: None,
        note: "CUDA's composed-decode attention path; no MUSA equivalent",
    },
    Knob {
        name: "APXINF_ATTN_COMPOSED_PREFILL",
        counterpart: None,
        note: "CUDA's composed-prefill attention path; no MUSA equivalent",
    },
    Knob {
        name: "APXINF_DISABLE_FA2_SPLITKV",
        counterpart: None,
        note: "CUDA's FlashAttention-2 split-KV switch. MUSA's D256 attention splits \
               inside the vendored assembly and has no counterpart",
    },
    Knob {
        name: "APXINF_GEMM_DUMP_SHAPES",
        counterpart: None,
        note: "CUDA GEMM shape logging; MUSA logs through MTTX instead (doc/pi05-mttx.md)",
    },
    Knob {
        name: "APXINF_PREFILL_SDPA_LEGACY",
        counterpart: None,
        note: "CUDA prefill SDPA selection; no MUSA equivalent",
    },
    Knob {
        name: "APXINF_VISION_FA2",
        counterpart: None,
        note: "CUDA vision FlashAttention-2 selection; MUSA's is \
               `APXINF_MUSA_BF16_MHA`",
    },
    Knob {
        name: "APXINF_VISION_SCORES_FP32",
        counterpart: None,
        note: "CUDA vision score-dtype switch. The measurement it belongs to is \
               CUDA's; the MUSA port has no counterpart and should not grow one \
               without its own",
    },
    Knob {
        name: "APXINF_VISION_SDPA_LEGACY",
        counterpart: None,
        note: "CUDA vision SDPA selection; no MUSA equivalent",
    },
    // ── MUSA-only: knobs written for this port's diagnostics ───────────────
    Knob {
        name: "APXINF_MUSA_BF16_MHA",
        counterpart: Some("APXINF_MHA_NAIVE"),
        note: "`=legacy` forces the legacy vision MHA; see the pairing above",
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
        note: "`=poison` fills every operator output on the no-graph path to prove \
               nothing reads it unwritten. The default is dirty, measured on M1000; CUDA \
               clears unconditionally and has no switch",
    },
    Knob {
        name: "APXINF_MUSA_WORKSPACE_SLACK",
        counterpart: None,
        note: "pad every arena allocation, so an out-of-bounds write lands in unused \
               memory instead of the next allocation. A diagnostic for a suspected \
               kernel overrun",
    },
    // ── The MUSA model port's INT8 knobs ───────────────────────────────────
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

/// Every `APXINF_*` name `env::var`/`env::var_os` is called with under `root`.
fn environment_names(root: &Path, into: &mut BTreeSet<String>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            // Integration tests read nothing a user configures.
            if path.file_name().is_some_and(|name| name == "tests") {
                continue;
            }
            environment_names(&path, into);
            continue;
        }
        if path.extension().is_none_or(|extension| extension != "rs") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (index, _) in text.match_indices("env::var") {
            // `env::var("NAME")` or `env::var_os("NAME")`; anything else (a
            // computed name) is skipped rather than guessed at.
            let rest = &text[index..];
            let Some(open) = rest.find('(') else { continue };
            let rest = &rest[open + 1..];
            let Some(rest) = rest.trim_start().strip_prefix('"') else {
                continue;
            };
            let Some(close) = rest.find('"') else { continue };
            let name = &rest[..close];
            if name.starts_with("APXINF_") {
                into.insert(name.to_string());
            }
        }
    }
}

/// The names a source tree actually reads.
fn read_names(source_dir: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    environment_names(&workspace_root().join(source_dir), &mut names);
    names
}

/// The three trees whose `APXINF_*` reads this table covers.
fn backend_knobs() -> (BTreeSet<String>, BTreeSet<String>, BTreeSet<String>) {
    (
        read_names("crates/apxinf-cuda/src"),
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
        // Whichever side this row is on, the named counterpart has to exist in
        // the union. Both rows of a pair name each other, so a one-sided edit
        // fails on the row that stayed behind.
        if !cuda.contains(knob.name) && !musa.contains(knob.name) && !model.contains(knob.name) {
            continue; // the first assertion already reported this
        }
        if !cuda.contains(counterpart) && !musa.contains(counterpart) && !model.contains(counterpart)
        {
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
    let (cuda, musa, model) = backend_knobs();
    assert!(
        cuda.len() >= 10,
        "the CUDA knob scan found {} names; the reader is broken, not the crate",
        cuda.len()
    );
    assert!(
        musa.len() >= 5,
        "the MUSA knob scan found {} names; the reader is broken, not the crate",
        musa.len()
    );
    assert!(
        model.len() >= 4,
        "the MUSA model scan found {} names; the reader is broken, not the crate",
        model.len()
    );
}

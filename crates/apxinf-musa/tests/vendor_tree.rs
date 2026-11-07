//! The vendored MATE/MUTLASS tree's own guard.
//!
//! A vendored tree is the one part of a crate whose contents are not checked by
//! the compiler against anything the repo owns. `kernels/mutlass_fmha/README.md`
//! says the tree is re-vendored by copying three directories from a wheel and
//! auditing with `diff -r` — and that audit is the only thing standing between a
//! copy and a silent omission.
//!
//! It has already failed once. `bbc254e` added seven `builders/` headers that a
//! global `**/build*/` ignore rule had dropped from the tree before it was ever
//! committed; they were missing for two commits, and nothing said so, because a
//! missing header is not a missing file until something tries to `#include` it
//! and the symptom lands somewhere else entirely.
//!
//! What can be checked from inside a checkout is exactly that failure mode:
//! **every file in the tree must be one git would actually track.** The check
//! cannot verify the tree against the render it came from — the render lives in
//! a MATE wheel this repository does not carry — and it does not pretend to.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The tree, relative to this crate.
const VENDOR_TREE: &str = "kernels/mutlass_fmha";

fn tree_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(VENDOR_TREE)
}

/// Every file under `root`, in a stable order.
fn files_under(root: &Path, into: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files_under(&path, into);
        } else if path.is_file() {
            into.push(path);
        }
    }
}

#[test]
fn every_vendored_file_would_be_tracked_by_git() {
    let root = tree_root();
    assert!(
        root.is_dir(),
        "the vendored tree is missing at {} — re-vendor it per kernels/mutlass_fmha/README.md",
        root.display()
    );

    let mut files = Vec::new();
    files_under(&root, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "the vendored tree at {} holds no files",
        root.display()
    );

    // Asked in one call rather than one per file: `git check-ignore` prints the
    // paths it considers ignored, so an empty stdout is the pass.
    let mut command = Command::new("git");
    command.arg("check-ignore").arg("--");
    for file in &files {
        command.arg(file);
    }
    let Ok(output) = command.output() else {
        // No git, or no checkout to ask: the property is about this repository,
        // and there is nothing here to assert it against. Skipped loudly.
        eprintln!("skipping: `git` is unavailable, so the ignore rules cannot be read");
        return;
    };
    if output.status.code() == Some(1) {
        // 1 is "nothing matched", which is the answer we want.
        return;
    }
    assert!(
        output.status.success(),
        "git check-ignore failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let ignored = String::from_utf8_lossy(&output.stdout);
    panic!(
        "{} of {} file(s) under {VENDOR_TREE} are matched by a gitignore rule, so they are \
         in the working tree but not in the repository — this is how `bbc254e` lost seven \
         headers. Either narrow the ignore rule or force-add the files:\n{ignored}",
        ignored.lines().count(),
        files.len()
    );
}

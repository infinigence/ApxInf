//! Shared nvcc compile scheduling for the CUDA build scripts: an on-disk
//! object cache keyed by command line + dependency list, and a bounded worker
//! pool so independent translation units compile concurrently instead of one
//! at a time. Both `apxinf-cuda` and `apxinf-cuda-new` include this file with
//! `#[path]`, the same way they share `aot/link.rs`.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The full argv of an nvcc invocation, in a form that changes whenever any
/// flag, include path, define or architecture changes.
pub fn describe_command(command: &Command) -> String {
    let mut text = command.get_program().to_string_lossy().into_owned();
    for argument in command.get_args() {
        text.push('\u{1f}');
        text.push_str(&argument.to_string_lossy());
    }
    text
}

/// True when `object` can be reused: it exists, the recorded command line is
/// identical, a dependency list exists, and every file in that list is older
/// than the object. Anything missing or unreadable means "rebuild" -- the
/// cache never decides to skip on incomplete information.
pub fn object_is_current(
    object: impl AsRef<Path>,
    deps: impl AsRef<Path>,
    stamp: impl AsRef<Path>,
    cmdline: &str,
) -> bool {
    let (object, deps, stamp) = (object.as_ref(), deps.as_ref(), stamp.as_ref());
    let Ok(object_time) = std::fs::metadata(object).and_then(|m| m.modified()) else {
        return false;
    };
    if std::fs::read_to_string(stamp).ok().as_deref() != Some(cmdline) {
        return false;
    }
    let Ok(rule) = std::fs::read_to_string(deps) else {
        return false;
    };
    // A make rule: "target: dep dep \\\n dep ...". Drop everything up to the
    // first unescaped colon, then split on unescaped whitespace.
    let Some((_, prerequisites)) = rule.split_once(':') else {
        return false;
    };
    let mut any = false;
    for prerequisite in prerequisites.split_whitespace() {
        if prerequisite == "\\" {
            continue;
        }
        any = true;
        let Ok(time) = std::fs::metadata(prerequisite).and_then(|m| m.modified()) else {
            return false;
        };
        if time > object_time {
            return false;
        }
    }
    any
}

/// One pending nvcc invocation. The stamp is written only after nvcc
/// succeeded, so an interrupted or failed compile cannot leave a stamp that
/// hides a stale object from the next build.
pub struct CompileJob {
    pub command: Command,
    pub action: String,
    pub stamp: PathBuf,
    pub cmdline: String,
}

/// Concurrent nvcc process count: `APXINF_NVCC_PARALLELISM` when set,
/// otherwise one less than the available cores so the host stays responsive.
/// Each nvcc already serializes its own cicc/ptxas stages; translation units
/// are independent of each other.
pub fn parallelism() -> usize {
    std::env::var("APXINF_NVCC_PARALLELISM")
        .ok()
        .and_then(|value| value.trim().parse().ok())
        .unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|cores| cores.get().saturating_sub(1))
                .unwrap_or(1)
        })
        .max(1)
}

/// Run every job across `workers` threads, failing the build on the first
/// error. A worker panic (nvcc missing or a failed compile) propagates when
/// the scope joins, after in-flight compiles finish.
pub fn run_parallel(jobs: Vec<CompileJob>, workers: usize) {
    if jobs.is_empty() {
        return;
    }
    let workers = workers.clamp(1, jobs.len());
    let queue = std::sync::Mutex::new(jobs.into_iter());
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let Some(mut job) = queue.lock().unwrap().next() else {
                    break;
                };
                let status = job
                    .command
                    .status()
                    .unwrap_or_else(|error| panic!("{}: {error}", job.action));
                assert!(status.success(), "{} failed with {status}", job.action);
                std::fs::write(&job.stamp, &job.cmdline)
                    .unwrap_or_else(|error| panic!("write {}: {error}", job.stamp.display()));
            });
        }
    });
}

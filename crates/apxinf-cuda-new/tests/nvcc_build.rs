#![allow(dead_code)]

#[path = "../build_support/nvcc_build.rs"]
mod nvcc_build;

use std::path::{Path, PathBuf};
use std::process::Command;

use nvcc_build::{describe_command, object_is_current, run_parallel, CompileJob};

struct Unit {
    root: PathBuf,
    header: PathBuf,
    object: PathBuf,
    deps: PathBuf,
    stamp: PathBuf,
}

impl Unit {
    fn new(name: &str) -> Self {
        let root =
            std::env::temp_dir().join(format!("apxinf-nvcc-build-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&root).unwrap();
        let header = root.join("unit.h");
        std::fs::write(&header, "header").unwrap();
        Self {
            header,
            object: root.join("unit.o"),
            deps: root.join("unit.d"),
            stamp: root.join("unit.cmdline"),
            root,
        }
    }

    /// A stand-in for nvcc: writes `content` to the object and a dependency
    /// rule naming the header, then exits with `code`.
    fn compiler(&self, content: &str, code: i32) -> Command {
        let mut command = Command::new("sh");
        command.arg("-c").arg(format!(
            "printf '%s' '{content}' > '{object}' && printf 'unit.o: %s\\n' '{header}' > '{deps}'; exit {code}",
            object = self.object.display(),
            header = self.header.display(),
            deps = self.deps.display(),
        ));
        command
    }

    fn job(&self, command: Command) -> (CompileJob, String) {
        let cmdline = describe_command(&command);
        let job = CompileJob {
            command,
            action: "compile unit".to_string(),
            stamp: self.stamp.clone(),
            cmdline: cmdline.clone(),
        };
        (job, cmdline)
    }

    fn is_current(&self, cmdline: &str) -> bool {
        object_is_current(&self.object, &self.deps, &self.stamp, cmdline)
    }
}

impl Drop for Unit {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap()
}

#[test]
fn successful_compile_publishes_a_reusable_object() {
    let unit = Unit::new("success");
    let (job, cmdline) = unit.job(unit.compiler("first", 0));
    assert!(!unit.is_current(&cmdline));
    run_parallel(vec![job], 1);
    assert!(unit.is_current(&cmdline));
    assert_eq!(read(&unit.object), "first");

    let other = describe_command(&unit.compiler("second", 0));
    assert!(!unit.is_current(&other));
}

#[test]
fn failed_recompile_invalidates_the_previous_stamp() {
    let unit = Unit::new("failure");
    let (job, cmdline) = unit.job(unit.compiler("good", 0));
    run_parallel(vec![job], 1);
    assert!(unit.is_current(&cmdline));

    // The same command line fails after overwriting the object, as an
    // interrupted nvcc does. The partial object is newer than every
    // dependency, so only the missing stamp keeps it from being reused.
    let mut failing = unit.compiler("partial", 1);
    let (mut job, _) = unit.job(unit.compiler("good", 0));
    std::mem::swap(&mut job.command, &mut failing);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_parallel(vec![job], 1);
    }));
    assert!(result.is_err());
    assert_eq!(read(&unit.object), "partial");
    assert!(!unit.stamp.exists());
    assert!(!unit.is_current(&cmdline));
}

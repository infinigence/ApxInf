//! CPU-only regression: rustc --edition=2021 --test tests/nvcc_cache.rs -o cache-test
//! NVCC_CACHE_TEST_ROOT must name task-local scratch space; no CUDA is executed.
#[path = "../build_support/nvcc_cache.rs"]
mod nvcc_cache;

use std::fs::{self, FileTimes};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;

const FAKE: &str = r##"#!/usr/bin/python3
import os, pathlib, signal, sys
args = sys.argv[1:]
def arg(name):
    return pathlib.Path(args[args.index(name) + 1])
def escape(path):
    return str(path).replace('\\', '\\\\').replace(' ', '\\ ').replace('#', '\\#').replace('$', '$$').replace(':', '\\:')
with open(os.environ['FAKE_COUNT'], 'a') as log:
    log.write('compile\n')
source, output = arg('-c'), arg('-o')
header = pathlib.Path(os.environ['FAKE_HEADER'])
if '-DFAKE_SHADOW' in args:
    header = next(pathlib.Path(a[2:]) / 'shadow.h' for a in args if a.startswith('-I') and (pathlib.Path(a[2:]) / 'shadow.h').is_file())
mode = next((a.split('=', 1)[1] for a in args if a.startswith('--fake-mode=')), 'ok')
if mode == 'no-object':
    sys.exit(0)
output.write_bytes(b'partial')
if mode == 'fail':
    sys.exit(7)
if mode == 'signal':
    os.kill(os.getpid(), signal.SIGKILL)
if mode == 'kill-parent':
    os.kill(os.getppid(), signal.SIGKILL)
    sys.exit(0)
content = source.read_bytes() + b'|' + header.read_bytes()
content += b'|optional-present' if (source.parent / 'optional').exists() else b'|optional-absent'
output.write_bytes(content)
if '-MF' in args and mode != 'no-deps':
    text = escape(output) + ': ' + escape(source) + ' \\\n  ' + escape(header) + '\n'
    if mode == 'bad-deps':
        text = 'not a make rule'
    arg('-MF').write_text(text)
if mode == 'changed-during-compile':
    header.write_bytes(b'changed while compiling')
"##;

struct Fixture {
    root: PathBuf,
    compiler: PathBuf,
    source: PathBuf,
    header: PathBuf,
    object: PathBuf,
    count: PathBuf,
}
impl Fixture {
    fn new(name: &str) -> Self {
        let base = PathBuf::from(
            std::env::var_os("NVCC_CACHE_TEST_ROOT")
                .or_else(|| option_env!("CARGO_TARGET_TMPDIR").map(Into::into))
                .expect("set NVCC_CACHE_TEST_ROOT for standalone rustc tests"),
        );
        let root = base.join(format!("{}-{name} space#dollar$back\\", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let this = Self::at(root);
        for dir in ["bin", "input", "out"] {
            fs::create_dir_all(this.root.join(dir)).unwrap();
        }
        fs::write(this.root.join("bin/g++"), "#!/bin/sh\nprintf '#include <...> search starts here:\n %s\nEnd of search list.\n' \"$FAKE_SYSTEM_INCLUDE\" >&2\n").unwrap();
        fs::set_permissions(this.root.join("bin/g++"), fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(&this.compiler, FAKE).unwrap();
        fs::set_permissions(&this.compiler, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(&this.source, b"source0").unwrap();
        fs::write(&this.header, b"header0").unwrap();
        this
    }
    fn at(root: PathBuf) -> Self {
        Self {
            compiler: root.join("bin/nvcc tool"),
            source: root.join("input/source file.cu"),
            header: root.join("input/header #$\\:.h"),
            object: root.join("out/object.o"),
            count: root.join("count"),
            root,
        }
    }
    fn command(&self) -> Command {
        let mut command = Command::new(&self.compiler);
        command
            .arg("-c")
            .arg(&self.source)
            .args(["-O3", "-gencode=arch=compute_87,code=sm_87"])
            .env("PATH", self.root.join("bin"))
            .env("FAKE_SYSTEM_INCLUDE", self.root.join("input"))
            .env("FAKE_COUNT", &self.count)
            .env("FAKE_HEADER", &self.header)
            .env_remove("NVCC_APPEND_FLAGS")
            .env_remove("NVCC_PREPEND_FLAGS");
        command
    }
    fn run(&self) -> bool {
        nvcc_cache::Cache::default()
            .compile(&self.command(), &self.source, &self.object)
            .unwrap()
    }
    fn count(&self) -> usize {
        fs::read_to_string(&self.count).unwrap().lines().count()
    }
    fn stamp(&self) -> PathBuf {
        self.object.with_extension("o.cache")
    }
}

fn replace_preserving_mtime(path: &std::path::Path, bytes: &[u8]) {
    let time = fs::metadata(path).unwrap().modified().unwrap();
    fs::write(path, bytes).unwrap();
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .unwrap()
        .set_times(FileTimes::new().set_modified(time))
        .unwrap();
    assert_eq!(fs::metadata(path).unwrap().modified().unwrap(), time);
}

#[test]
fn exact_repeat_hits_with_escaped_paths() {
    let f = Fixture::new("repeat");
    assert!(!f.run());
    assert!(f.run());
    assert_eq!(f.count(), 1);
}

#[test]
fn source_header_and_object_content_invalidate_even_with_preserved_mtime() {
    let f = Fixture::new("content");
    assert!(!f.run());
    for path in [&f.source, &f.header, &f.object] {
        replace_preserving_mtime(path, b"newdata");
        assert!(!f.run());
        assert!(f.run());
    }
    assert_eq!(f.count(), 4);
}

#[test]
fn full_command_environment_and_compiler_identity_invalidate() {
    let f = Fixture::new("signature");
    assert!(!f.run());
    let mut command = f.command();
    command.args(["-DFLAG=1", "-gencode=arch=compute_80,code=sm_80"]);
    assert!(!nvcc_cache::Cache::default()
        .compile(&command, &f.source, &f.object)
        .unwrap());
    assert!(nvcc_cache::Cache::default()
        .compile(&command, &f.source, &f.object)
        .unwrap());
    command.env("NVCC_CCBIN", f.root.join("bin/g++"));
    assert!(!nvcc_cache::Cache::default()
        .compile(&command, &f.source, &f.object)
        .unwrap());
    assert!(nvcc_cache::Cache::default()
        .compile(&command, &f.source, &f.object)
        .unwrap());
    replace_preserving_mtime(
        &f.compiler,
        format!("{FAKE}\n# modified compiler\n").as_bytes(),
    );
    assert!(!nvcc_cache::Cache::default()
        .compile(&command, &f.source, &f.object)
        .unwrap());
    assert_eq!(f.count(), 4);
}

#[test]
fn reconnect_and_jobserver_environment_changes_still_hit() {
    let f = Fixture::new("session-env");
    assert!(!f.run());
    let mut command = f.command();
    for (key, value) in [
        ("SSH_CONNECTION", "a different session"),
        ("SSH_TTY", "/dev/pts/99"),
        ("SHLVL", "9"),
        ("_", "/different/cargo"),
        ("CARGO_MAKEFLAGS", "--jobserver-auth=41,42"),
    ] {
        command.env(key, value);
    }
    assert!(nvcc_cache::Cache::default()
        .compile(&command, &f.source, &f.object)
        .unwrap());
    assert_eq!(f.count(), 1);
}

#[test]
fn toolkit_components_and_host_compiler_changes_invalidate() {
    let f = Fixture::new("tools");
    fs::create_dir_all(f.root.join("nvvm/libdevice")).unwrap();
    fs::create_dir_all(f.root.join("nvvm/bin")).unwrap();
    let files = [
        "bin/nvcc.profile",
        "bin/ptxas",
        "bin/cudafe++",
        "bin/fatbinary",
        "bin/nvlink",
        "nvvm/bin/cicc",
        "nvvm/libdevice/libdevice.10.bc",
    ];
    for path in files {
        fs::write(f.root.join(path), b"tool0").unwrap();
    }
    assert!(!f.run());
    for path in files {
        replace_preserving_mtime(&f.root.join(path), b"tool1");
        assert!(!f.run());
        assert!(f.run());
    }
    let host = f.root.join("bin/g++");
    let mut bytes = fs::read(&host).unwrap();
    bytes.extend(b"\n# changed host compiler\n");
    replace_preserving_mtime(&host, &bytes);
    assert!(!f.run());
    assert!(f.run());
    assert_eq!(f.count(), 9);
}

#[test]
fn new_shadowing_and_optional_headers_invalidate() {
    let f = Fixture::new("includes");
    let early = f.root.join("early");
    let late = f.root.join("late");
    fs::create_dir_all(&early).unwrap();
    fs::create_dir_all(&late).unwrap();
    fs::write(late.join("shadow.h"), b"late").unwrap();
    let mut command = f.command();
    command
        .arg("-DFAKE_SHADOW")
        .arg(format!("-I{}", early.display()))
        .arg(format!("-I{}", late.display()));
    let run = || {
        nvcc_cache::Cache::default()
            .compile(&command, &f.source, &f.object)
            .unwrap()
    };
    assert!(!run());
    assert!(run());
    let late_object = fs::read(&f.object).unwrap();
    fs::write(early.join("shadow.h"), b"early").unwrap();
    assert!(!run());
    assert_ne!(fs::read(&f.object).unwrap(), late_object);
    assert!(run());
    let old_object = fs::read(&f.object).unwrap();
    fs::write(
        f.source.parent().unwrap().join("optional"),
        b"extensionless",
    )
    .unwrap();
    assert!(!run());
    assert_ne!(fs::read(&f.object).unwrap(), old_object);
    assert!(run());
    assert_eq!(f.count(), 3);
}

#[test]
fn include_symlinks_and_cycles_do_not_disable_cache() {
    let f = Fixture::new("symlinks");
    let external = f.root.join("external");
    fs::create_dir_all(&external).unwrap();
    std::os::unix::fs::symlink(&external, f.source.parent().unwrap().join("linked")).unwrap();
    std::os::unix::fs::symlink(f.source.parent().unwrap(), external.join("cycle")).unwrap();
    assert!(!f.run());
    assert!(f.run());
    fs::write(external.join("new-header"), b"new").unwrap();
    assert!(!f.run());
    assert!(f.run());
    assert_eq!(f.count(), 2);
}

#[test]
fn missing_or_corrupt_state_rebuilds() {
    let f = Fixture::new("missing");
    assert!(!f.run());
    fs::remove_file(&f.object).unwrap();
    assert!(!f.run());
    fs::remove_file(f.object.with_extension("o.d")).unwrap();
    assert!(!f.run());
    fs::write(f.object.with_extension("o.d"), b"broken deps").unwrap();
    assert!(!f.run());
    fs::write(f.stamp(), b"invalid stamp").unwrap();
    assert!(!f.run());
    assert!(f.run());
    assert_eq!(f.count(), 5);
}

#[test]
fn missing_dependency_fails_and_cannot_hit() {
    let f = Fixture::new("missing-header");
    assert!(!f.run());
    fs::remove_file(&f.header).unwrap();
    assert!(nvcc_cache::Cache::default()
        .compile(&f.command(), &f.source, &f.object)
        .is_err());
    assert!(!f.stamp().exists());
    fs::write(&f.header, b"header0").unwrap();
    assert!(!f.run());
    assert!(f.run());
    assert_eq!(f.count(), 3);
}

#[test]
fn nvcc_injection_bypasses_cache() {
    let f = Fixture::new("injection");
    assert!(!f.run());
    for name in ["NVCC_APPEND_FLAGS", "NVCC_PREPEND_FLAGS"] {
        let mut command = f.command();
        command.env(name, "-DFLAG=1");
        for _ in 0..2 {
            assert!(!nvcc_cache::Cache::default()
                .compile(&command, &f.source, &f.object)
                .unwrap());
            assert!(!f.stamp().exists());
        }
    }
    assert!(!f.run());
    assert!(f.run());
    assert_eq!(f.count(), 6);
}

#[test]
fn failure_or_signal_preserves_old_object_and_invalidates_stamp() {
    let f = Fixture::new("failure");
    assert!(!f.run());
    let original = fs::read(&f.object).unwrap();
    for mode in ["fail", "signal", "no-object"] {
        let mut command = f.command();
        command.arg(format!("--fake-mode={mode}"));
        assert!(nvcc_cache::Cache::default()
            .compile(&command, &f.source, &f.object)
            .is_err());
        assert_eq!(fs::read(&f.object).unwrap(), original);
        assert!(!f.stamp().exists());
        assert!(!f.run());
        assert!(f.run());
    }
    assert_eq!(f.count(), 7);
}

#[test]
fn unusable_deps_and_racing_header_are_uncacheable() {
    let f = Fixture::new("uncacheable");
    assert!(!f.run());
    for mode in ["no-deps", "bad-deps", "changed-during-compile"] {
        let mut command = f.command();
        command.arg(format!("--fake-mode={mode}"));
        assert!(!nvcc_cache::Cache::default()
            .compile(&command, &f.source, &f.object)
            .unwrap());
        assert!(!f.stamp().exists());
    }
    assert_eq!(f.count(), 4);
}

#[test]
fn interrupted_driver() {
    let Some(root) = std::env::var_os("NVCC_CACHE_KILL_ROOT") else {
        return;
    };
    let f = Fixture::at(PathBuf::from(root));
    let mut command = f.command();
    command.arg("--fake-mode=kill-parent");
    nvcc_cache::Cache::default()
        .compile(&command, &f.source, &f.object)
        .unwrap();
    panic!("fake compiler should have killed its cache parent");
}

#[test]
fn killed_cache_process_leaves_orphan_unusable() {
    let f = Fixture::new("kill-parent");
    assert!(!f.run());
    let original = fs::read(&f.object).unwrap();
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "interrupted_driver", "--nocapture"])
        .env("NVCC_CACHE_KILL_ROOT", &f.root)
        .status()
        .unwrap();
    assert!(!status.success());
    assert!(!f.stamp().exists());
    assert_eq!(fs::read(&f.object).unwrap(), original);
    let orphans = fs::read_dir(f.root.join("out"))
        .unwrap()
        .flatten()
        .filter(|entry| {
            entry
                .file_name()
                .to_string_lossy()
                .starts_with(".nvcc-tmp-")
        })
        .count();
    assert_eq!(orphans, 1);
    assert!(!f.run());
    assert!(f.run());
    assert_eq!(f.count(), 3);
}

//! Asking `mcc` a question by compiling something.
//!
//! Two things in this build probe the toolchain: whether it needs the bundled
//! host `<bits/math-vector.h>` shim, and whether it accepts the selected
//! architecture name. Both compile a translation unit with `-x musa`, and both
//! have to append the shim when it is available, because a MUSA translation unit
//! reaches this host's glibc either way. So the invocation and the shim's
//! location live here, and each asker keeps only its own question.
//!
//! Every probe stops at the object file. Linking would fold in a second,
//! unrelated way to fail, and the question is always about the compile.

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Where the bundled shim's headers live.
pub fn shim_dir(manifest_dir: &Path) -> PathBuf {
    manifest_dir.join("build_support/host_math_shim")
}

/// The include directory to hand to `-isystem`, or `None` when this crate ships
/// no shim.
///
/// The path is fixed rather than resolved, so anything that probes `mcc` can use
/// it before the build has decided whether the real compile needs it.
pub fn shim(manifest_dir: &Path) -> Option<PathBuf> {
    let directory = shim_dir(manifest_dir);
    directory
        .join("bits/math-vector.h")
        .is_file()
        .then_some(directory)
}

/// A scratch directory holding probe sources, removed on drop.
pub struct Scratch {
    directory: PathBuf,
}

impl Scratch {
    pub fn new(label: &str) -> Self {
        let directory =
            std::env::temp_dir().join(format!("apxinf-{label}-probe-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&directory);
        Self { directory }
    }

    /// Write a probe source and return the path to it.
    pub fn source(&self, name: &str, contents: &str) -> Result<PathBuf, String> {
        let path = self.directory.join(name);
        std::fs::write(&path, contents)
            .map_err(|error| format!("cannot write the {name} probe: {error}"))?;
        Ok(path)
    }

    pub fn object(&self, name: &str) -> PathBuf {
        self.directory.join(format!("{name}.o"))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

/// Compile `source` to `object`, reporting the compiler's stderr on failure.
///
/// `extra` lands between `-x musa` and the source, which is where each caller's
/// distinguishing arguments belong. `shim` is appended last, so it can only add
/// a header search path — never displace one the caller asked for.
pub fn compile(
    mcc: &Path,
    extra: &[OsString],
    shim: Option<&Path>,
    source: &Path,
    object: &Path,
) -> Result<Result<(), String>, String> {
    let mut command = Command::new(mcc);
    command
        .arg("-x")
        .arg("musa")
        .args(extra)
        .arg("-c")
        .arg(source)
        .arg("-o")
        .arg(object);
    if let Some(shim) = shim {
        command.arg("-isystem").arg(shim);
    }
    let output = command
        .output()
        .map_err(|error| format!("cannot run {}: {error}", mcc.display()))?;
    if output.status.success() {
        Ok(Ok(()))
    } else {
        Ok(Err(String::from_utf8_lossy(&output.stderr).into_owned()))
    }
}

/// The first `error:` line, since a header cascade buries the cause under the
/// include chain that reached it.
pub fn first_line(text: &str) -> &str {
    text.lines()
        .find(|line| line.contains("error"))
        .or_else(|| text.lines().next())
        .unwrap_or("(no output)")
        .trim()
}

//! Content-based object reuse using NVCC's own complete dependency list.
//! A missing/invalid dependency list always falls back to compilation.
use std::{
    collections::HashMap,
    ffi::OsStr,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

fn hash(bytes: &[u8]) -> u128 {
    let mut value = 0x6c62272e07bb014262b821756295c58du128;
    for byte in bytes {
        value ^= u128::from(*byte);
        value = value.wrapping_mul(0x0000000001000000000000000000013b);
    }
    value
}

/// Parse Make escaping, including spaces and backslash-newline continuations.
fn dependencies(text: &str) -> Option<Vec<PathBuf>> {
    let (_, body) = text.split_once(':')?;
    let mut paths = Vec::new();
    let mut token = String::new();
    let mut chars = body.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            let next = chars.next()?;
            if next != '\n' {
                token.push(next);
            }
        } else if ch.is_whitespace() {
            if !token.is_empty() {
                paths.push(PathBuf::from(std::mem::take(&mut token)));
            }
        } else {
            token.push(ch);
        }
    }
    if !token.is_empty() {
        paths.push(PathBuf::from(token));
    }
    paths.sort();
    paths.dedup();
    if paths.is_empty() {
        None
    } else {
        Some(paths)
    }
}

pub struct NvccCache {
    identity: Vec<u8>,
    files: HashMap<PathBuf, u128>,
}
impl NvccCache {
    pub fn new(nvcc: &str) -> Self {
        let mut identity = b"apxinf-nvcc-object-cache-v2\0".to_vec();
        for compiler in [nvcc, "g++"] {
            identity.extend_from_slice(compiler.as_bytes());
            if let Ok(output) = Command::new(compiler).arg("--version").output() {
                identity.extend_from_slice(&output.stdout);
                identity.extend_from_slice(&output.stderr);
            }
        }
        for name in [
            "PATH",
            "CPATH",
            "CPLUS_INCLUDE_PATH",
            "C_INCLUDE_PATH",
            "LIBRARY_PATH",
            "COMPILER_PATH",
            "GCC_EXEC_PREFIX",
            "CUDAHOSTCXX",
            "NVCC_CCBIN",
            "CC",
            "CXX",
            "NVCC_PREPEND_FLAGS",
            "NVCC_APPEND_FLAGS",
            "SOURCE_DATE_EPOCH",
        ] {
            println!("cargo:rerun-if-env-changed={name}");
            identity.extend_from_slice(name.as_bytes());
            identity.push(u8::from(std::env::var_os(name).is_some()));
            if let Some(value) = std::env::var_os(name) {
                identity.extend_from_slice(value.as_encoded_bytes());
            }
            identity.push(0);
        }
        Self {
            identity,
            files: HashMap::new(),
        }
    }

    fn fingerprint(&mut self, command: &Command) -> Option<u128> {
        let mut dependency_command = Command::new(command.get_program());
        if let Some(directory) = command.get_current_dir() {
            dependency_command.current_dir(directory);
        }
        for (name, value) in command.get_envs() {
            if let Some(value) = value {
                dependency_command.env(name, value);
            } else {
                dependency_command.env_remove(name);
            }
        }
        let mut args = command.get_args();
        while let Some(arg) = args.next() {
            if arg == OsStr::new("-o") {
                args.next();
            } else if arg != OsStr::new("-c") {
                dependency_command.arg(arg);
            }
        }
        dependency_command.args(["-M", "-MT", "apxinf_object"]);
        let output = dependency_command.output().ok()?;
        if !output.status.success() {
            return None;
        }
        let paths = dependencies(std::str::from_utf8(&output.stdout).ok()?)?;
        let mut signature = self.identity.clone();
        if let Some(directory) = command.get_current_dir() {
            signature.extend_from_slice(directory.as_os_str().as_encoded_bytes());
        }
        signature.push(0);
        for (name, value) in command.get_envs() {
            signature.extend_from_slice(name.as_encoded_bytes());
            signature.push(0);
            signature.push(u8::from(value.is_some()));
            if let Some(value) = value {
                signature.extend_from_slice(value.as_encoded_bytes());
            }
            signature.push(0);
        }
        signature.extend_from_slice(command.get_program().as_encoded_bytes());
        signature.push(0);
        for arg in command.get_args() {
            signature.extend_from_slice(arg.as_encoded_bytes());
            signature.push(0);
        }
        for path in paths {
            let path = if path.is_relative() {
                command
                    .get_current_dir()
                    .map(Path::to_path_buf)
                    .unwrap_or(std::env::current_dir().ok()?)
                    .join(path)
            } else {
                path
            };
            println!("cargo:rerun-if-changed={}", path.display());
            let content = if let Some(value) = self.files.get(&path) {
                *value
            } else {
                let value = hash(&fs::read(&path).ok()?);
                self.files.insert(path.clone(), value);
                value
            };
            signature.extend_from_slice(path.as_os_str().as_encoded_bytes());
            signature.push(0);
            signature.extend_from_slice(&content.to_le_bytes());
        }
        Some(hash(&signature))
    }

    pub fn compile(&mut self, command: &mut Command, object: &Path) -> io::Result<()> {
        let fingerprint = self.fingerprint(command);
        let stamp = object.with_extension("nvcc-cache");
        if let (Some(key), Ok(record), Ok(bytes)) =
            (fingerprint, fs::read_to_string(&stamp), fs::read(object))
        {
            if record == format!("{key:032x} {:032x}\n", hash(&bytes)) {
                return Ok(());
            }
        }
        match fs::remove_file(&stamp) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
        if !command.status()?.success() {
            return Err(io::Error::other("nvcc compilation failed"));
        }
        if let Some(key) = fingerprint {
            let record = format!("{key:032x} {:032x}\n", hash(&fs::read(object)?));
            fs::write(&stamp, record)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_continuations_and_spaces() {
        assert_eq!(
            dependencies("target: /a.cu \\\n /some\\ path/a.h /a.cu\n"),
            Some(vec![
                PathBuf::from("/a.cu"),
                PathBuf::from("/some path/a.h")
            ])
        );
    }
    #[test]
    fn rejects_missing_dependencies() {
        assert_eq!(dependencies("target:"), None);
        assert_eq!(dependencies("invalid"), None);
    }
    #[test]
    fn changes_are_visible_in_content_hash() {
        assert_ne!(hash(b"header-v1"), hash(b"header-v2"));
    }
    #[test]
    #[cfg(unix)]
    fn reuses_only_valid_objects_and_rebuilds_after_failure() {
        use std::os::unix::fs::PermissionsExt;
        let directory = std::env::temp_dir().join(format!(
            "apxinf-cache-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&directory).unwrap();
        let compiler = directory.join("fake-nvcc");
        fs::write(
            &compiler,
            r#"#!/bin/sh
task_dir=$(dirname "$0")
for arg in "$@"; do
  if [ "$arg" = --version ]; then echo fake-compiler-v1; exit 0; fi
  if [ "$arg" = -M ]; then
    if [ -f "$task_dir/no-deps" ]; then exit 1; fi
    printf 'object: %s/source.cu %s/header.h\n' "$task_dir" "$task_dir"
    exit 0
  fi
done
printf x >> "$task_dir/count"
if [ -f "$task_dir/fail" ]; then exit 1; fi
while [ "$#" -gt 0 ]; do
  if [ "$1" = -o ]; then shift; printf object > "$1"; exit 0; fi
  shift
done
exit 1
"#,
        )
        .unwrap();
        fs::set_permissions(&compiler, fs::Permissions::from_mode(0o755)).unwrap();
        fs::write(directory.join("source.cu"), "source").unwrap();
        fs::write(directory.join("header.h"), "header-v1").unwrap();
        let object = directory.join("kernel.o");
        let run = |flag: &str| {
            let mut command = Command::new(&compiler);
            command
                .arg("-c")
                .arg(directory.join("source.cu"))
                .arg("-o")
                .arg(&object)
                .arg(flag);
            NvccCache::new(compiler.to_str().unwrap()).compile(&mut command, &object)
        };
        let count = || fs::read(directory.join("count")).unwrap().len();
        run("-O3").unwrap();
        run("-O3").unwrap();
        assert_eq!(count(), 1);
        fs::write(directory.join("header.h"), "header-v2").unwrap();
        run("-O3").unwrap();
        assert_eq!(count(), 2);
        run("-O2").unwrap();
        assert_eq!(count(), 3);
        fs::write(&object, "corrupt").unwrap();
        run("-O2").unwrap();
        assert_eq!(count(), 4);
        fs::write(directory.join("fail"), "").unwrap();
        assert!(run("-O1").is_err());
        assert!(!object.with_extension("nvcc-cache").exists());
        fs::remove_file(directory.join("fail")).unwrap();
        run("-O1").unwrap();
        assert_eq!(count(), 6);
        fs::write(directory.join("no-deps"), "").unwrap();
        run("-O1").unwrap();
        run("-O1").unwrap();
        assert_eq!(count(), 8);
        fs::remove_dir_all(directory).unwrap();
    }
}

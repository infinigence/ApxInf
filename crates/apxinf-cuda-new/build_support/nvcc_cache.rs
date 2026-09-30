//! Local, per-object nvcc cache. Cargo serializes builds in a single OUT_DIR.
//! The caller supplies all compilation arguments except output/dependency paths.
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

const INJECTION_ENV: [&str; 2] = ["NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS"];
pub const COMPILE_ENV: &[&str] = &[
    "PATH",
    "NVCC_CCBIN",
    "CUDA_HOME",
    "CUDA_PATH",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "COMPILER_PATH",
    "GCC_EXEC_PREFIX",
    "LIBRARY_PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "SOURCE_DATE_EPOCH",
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
];

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

// Length-framed FNV-1a/128, also used by the existing native build fingerprints.
struct Digest(u128);
impl Digest {
    fn new() -> Self {
        Self(0x6c62272e07bb014262b821756295c58d)
    }
    fn bytes(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 = (self.0 ^ u128::from(*byte)).wrapping_mul(0x1000000000000000000013b);
        }
    }
    fn field(&mut self, bytes: &[u8]) {
        self.bytes(&(bytes.len() as u64).to_le_bytes());
        self.bytes(bytes);
    }
    fn os(&mut self, value: &OsStr) {
        self.field(value.as_encoded_bytes());
    }
    fn file(&mut self, path: &Path) -> io::Result<()> {
        let mut file = fs::File::open(path)?;
        self.bytes(&file.metadata()?.len().to_le_bytes());
        let mut buffer = [0; 65536];
        loop {
            let count = file.read(&mut buffer)?;
            if count == 0 {
                return Ok(());
            }
            self.bytes(&buffer[..count]);
        }
    }
}

// nvcc -MD emits one Make rule, with escaped paths and continuation lines.
// Unknown syntax is deliberately uncacheable, never interpreted as no deps.
fn dependencies(bytes: &[u8], cwd: &Path) -> io::Result<Vec<PathBuf>> {
    let text = std::str::from_utf8(bytes).map_err(|_| invalid("non-UTF8 depfile"))?;
    let text = text.replace("\\\r\n", " ").replace("\\\n", " ");
    let mut chars = text.chars().peekable();
    let mut target = false;
    let mut token = String::new();
    let mut paths = Vec::new();
    while let Some(character) = chars.next() {
        match character {
            '\\' => {
                let next = chars
                    .next()
                    .ok_or_else(|| invalid("trailing depfile escape"))?;
                if !next.is_whitespace() && !"\\#$:".contains(next) {
                    return Err(invalid("unsupported depfile escape"));
                }
                token.push(next);
            }
            ':' if !target => {
                target = true;
                token.clear();
            }
            ':' => return Err(invalid("multiple depfile rules")),
            '$' if chars.next() == Some('$') => token.push('$'),
            '$' => return Err(invalid("unsupported depfile variable")),
            '#' => return Err(invalid("unsupported depfile comment")),
            c if c.is_whitespace() => {
                if target && !token.is_empty() {
                    paths.push(cwd.join(std::mem::take(&mut token)));
                }
            }
            c => token.push(c),
        }
    }
    if target && !token.is_empty() {
        paths.push(cwd.join(token));
    }
    if !target || paths.is_empty() {
        return Err(invalid("missing depfile prerequisites"));
    }
    Ok(paths)
}

// Commands from build.rs inherit the environment (do not use env_clear()).
fn environment(command: &Command) -> BTreeMap<OsString, OsString> {
    let mut result: BTreeMap<_, _> = std::env::vars_os().collect();
    for (key, value) in command.get_envs() {
        if let Some(value) = value {
            result.insert(key.to_owned(), value.to_owned());
        } else {
            result.remove(key);
        }
    }
    result
}

fn signature(
    command: &Command,
    source: &Path,
    object: &Path,
    deps: &Path,
    context: u128,
) -> io::Result<String> {
    let cwd = std::env::current_dir()?.join(command.get_current_dir().unwrap_or(Path::new("")));
    let mut hash = Digest::new();
    hash.field(include_bytes!("nvcc_cache.rs"));
    hash.os(cwd.as_os_str());
    hash.field(&context.to_le_bytes());
    hash.os(command.get_program());
    for arg in command.get_args() {
        hash.os(arg);
    }
    // Normalize only the helper-owned temporary output/dependency paths.
    for arg in [
        OsStr::new("-o"),
        object.as_os_str(),
        OsStr::new("-MD"),
        OsStr::new("-MF"),
        deps.as_os_str(),
    ] {
        hash.os(arg);
    }
    let dep_bytes = fs::read(deps)?;
    hash.field(&dep_bytes);
    hash.file(&cwd.join(source))?;
    for path in dependencies(&dep_bytes, &cwd)? {
        track(&path, object.parent().unwrap());
        hash.os(path.as_os_str());
        hash.file(&path)?;
    }
    if fs::metadata(object)?.len() == 0 {
        return Err(invalid("empty nvcc object"));
    }
    hash.file(object)?;
    Ok(format!("{:032x}\n", hash.0))
}

fn track(path: &Path, out: &Path) {
    // Generated files are owned by this build; watch their source inputs instead.
    if !path.starts_with(out) {
        // Cargo treats a watched nonexistent file as perpetually dirty. Watch
        // its existing ancestor to notice creation instead.
        let existing = path.ancestors().find(|path| path.exists()).unwrap_or(path);
        println!("cargo:rerun-if-changed={}", existing.display());
    }
}

fn resolve(name: &OsStr, env: &BTreeMap<OsString, OsString>, cwd: &Path) -> io::Result<PathBuf> {
    let path = Path::new(name);
    if path.components().count() > 1 || path.is_absolute() {
        return fs::canonicalize(cwd.join(path));
    }
    std::env::split_paths(
        env.get(OsStr::new("PATH"))
            .map(OsString::as_os_str)
            .unwrap_or_default(),
    )
    .map(|dir| cwd.join(dir).join(path))
    .find(|path| path.is_file())
    .ok_or_else(|| invalid("host compiler not found in PATH"))
    .and_then(fs::canonicalize)
}

fn identity(hash: &mut Digest, path: &Path, out: &Path) -> io::Result<()> {
    hash.os(path.as_os_str());
    track(path, out);
    if path.is_file() {
        let real = fs::canonicalize(path)?;
        hash.os(real.as_os_str());
        track(&real, out);
        hash.file(&real)?;
    } else {
        hash.field(b"absent");
    }
    Ok(())
}

fn directory_entries(
    hash: &mut Digest,
    root: &Path,
    out: &Path,
    object: &Path,
    seen: &mut BTreeSet<PathBuf>,
) -> io::Result<()> {
    hash.os(root.as_os_str());
    track(root, out);
    if !root.is_dir() {
        hash.field(b"absent");
        return Ok(());
    }
    let real = fs::canonicalize(root)?;
    track(&real, out);
    if !seen.insert(real) {
        hash.field(b"already visited");
        return Ok(());
    }
    let mut entries = fs::read_dir(root)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if root == out
            && (name.starts_with(".nvcc-tmp-")
                || name == "libapxinf_gemm_native.a"
                || (name.starts_with("gemm-")
                    && [".o", ".o.d", ".o.cache"].iter().any(|s| name.ends_with(s)))
                || path == object
                || path == object.with_extension("o.d")
                || path == object.with_extension("o.cache"))
        {
            continue;
        }
        hash.os(entry.file_name().as_os_str());
        let kind = entry.file_type()?;
        hash.field(&[
            kind.is_dir() as u8,
            kind.is_file() as u8,
            kind.is_symlink() as u8,
        ]);
        if kind.is_symlink() {
            let target = fs::read_link(&path)?;
            hash.os(target.as_os_str());
            hash.field(&[path.exists() as u8, path.is_dir() as u8]);
            track(&root.join(target), out);
            if path.is_dir() {
                directory_entries(hash, &path, out, object, seen)?;
            }
        } else if kind.is_dir() {
            directory_entries(hash, &path, out, object, seen)?;
        }
    }
    Ok(())
}

/// Toolchains and include directory entries are immutable during one Cargo build.
/// A new Cache is constructed for every build-script process, including resumes.
#[derive(Default)]
pub struct Cache {
    toolchains: BTreeMap<u128, (u128, Vec<PathBuf>)>,
    trees: BTreeMap<PathBuf, u128>,
}
impl Cache {
    fn context(&mut self, command: &Command, source: &Path, object: &Path) -> io::Result<u128> {
        let env = environment(command);
        let cwd = std::env::current_dir()?.join(command.get_current_dir().unwrap_or(Path::new("")));
        let out = object
            .parent()
            .ok_or_else(|| invalid("object without parent"))?;
        let mut hash = Digest::new();
        hash.os(command.get_program());
        hash.os(cwd.as_os_str());
        for key in COMPILE_ENV {
            hash.os(OsStr::new(key));
            if let Some(value) = env.get(OsStr::new(key)) {
                hash.field(b"set");
                hash.os(value);
            } else {
                hash.field(b"unset");
            }
        }
        // build.rs does not use options files or -ccbin arguments. Those can
        // introduce untracked toolchain configuration, so fail closed if added.
        if command.get_args().any(|a| {
            let a = a.to_string_lossy();
            ["-ccbin", "--compiler-bindir", "-optf", "--options-file"]
                .iter()
                .any(|flag| a == *flag || a.starts_with(&format!("{flag}=")))
        }) {
            return Err(invalid(
                "unsupported external compiler/options-file selection",
            ));
        }
        let key = hash.0;
        if !self.toolchains.contains_key(&key) {
            let nvcc = resolve(command.get_program(), &env, &cwd)?;
            let toolkit = nvcc
                .parent()
                .and_then(Path::parent)
                .ok_or_else(|| invalid("nvcc toolkit path"))?;
            identity(&mut hash, &nvcc, out)?;
            for relative in [
                "bin/nvcc.profile",
                "bin/ptxas",
                "bin/cudafe++",
                "bin/fatbinary",
                "bin/nvlink",
                "nvvm/bin/cicc",
            ] {
                identity(&mut hash, &toolkit.join(relative), out)?;
            }
            let libdevice = toolkit.join("nvvm/libdevice");
            track(&libdevice, out);
            if libdevice.is_dir() {
                let mut files = fs::read_dir(&libdevice)?.collect::<Result<Vec<_>, _>>()?;
                files.sort_by_key(|entry| entry.file_name());
                for file in files {
                    identity(&mut hash, &file.path(), out)?;
                }
            }
            for name in ["gcc", "g++", "cc", "c++", "as", "ld"] {
                if let Ok(path) = resolve(OsStr::new(name), &env, &cwd) {
                    identity(&mut hash, &path, out)?;
                }
            }
            let ccbin = env.get(OsStr::new("NVCC_CCBIN")).filter(|v| !v.is_empty());
            let host = match ccbin {
                Some(path) if cwd.join(path).is_dir() => {
                    resolve(cwd.join(path).join("g++").as_os_str(), &env, &cwd)?
                }
                Some(path) => resolve(path, &env, &cwd)?,
                None => resolve(OsStr::new("g++"), &env, &cwd)?,
            };
            identity(&mut hash, &host, out)?;
            for program in ["cc1", "cc1plus", "as", "ld", "collect2"] {
                let query = Command::new(&host)
                    .arg(format!("-print-prog-name={program}"))
                    .env_clear()
                    .envs(&env)
                    .current_dir(&cwd)
                    .output()?;
                if query.status.success() {
                    let name = String::from_utf8_lossy(&query.stdout);
                    if let Ok(path) = resolve(OsStr::new(name.trim()), &env, &cwd) {
                        identity(&mut hash, &path, out)?;
                    }
                }
            }
            let query = Command::new(&host)
                .args(["-E", "-x", "c++", "-v", "-"])
                .env_clear()
                .envs(&env)
                .current_dir(&cwd)
                .stdin(Stdio::null())
                .output()?;
            if !query.status.success() {
                return Err(invalid("cannot discover host compiler include directories"));
            }
            let stderr = String::from_utf8_lossy(&query.stderr);
            let mut in_list = false;
            let mut implicit = Vec::new();
            for line in stderr.lines() {
                if line.contains("search starts here:") {
                    in_list = true;
                    continue;
                }
                if line.contains("End of search list.") {
                    in_list = false;
                }
                if in_list && line.starts_with(' ') {
                    implicit.push(cwd.join(line.trim()));
                }
            }
            if implicit.is_empty() {
                return Err(invalid("empty host compiler include search list"));
            }
            self.toolchains.insert(key, (hash.0, implicit));
        }
        let (toolchain, implicit) = &self.toolchains[&key];
        let mut roots = implicit.clone();
        hash = Digest(*toolchain);
        roots.push(
            cwd.join(source)
                .parent()
                .ok_or_else(|| invalid("source parent"))?
                .to_owned(),
        );
        let mut args = command.get_args();
        while let Some(arg) = args.next() {
            let arg = arg.to_string_lossy();
            if ["-I", "-isystem", "-iquote"].contains(&arg.as_ref()) {
                roots.push(cwd.join(args.next().ok_or_else(|| invalid("missing include path"))?));
            } else if let Some(path) = arg.strip_prefix("-I") {
                roots.push(cwd.join(path));
            }
        }
        for key in ["CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH"] {
            if let Some(value) = env.get(OsStr::new(key)) {
                roots.extend(std::env::split_paths(value).map(|path| cwd.join(path)));
            }
        }
        for root in roots {
            let entry = self.trees.entry(root.clone());
            let value = match entry {
                std::collections::btree_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::btree_map::Entry::Vacant(entry) => {
                    let mut tree = Digest::new();
                    directory_entries(&mut tree, &root, out, object, &mut BTreeSet::new())?;
                    *entry.insert(tree.0)
                }
            };
            hash.field(&value.to_le_bytes());
        }
        Ok(hash.0)
    }
}

struct Scratch(PathBuf);
impl Scratch {
    fn new(parent: &Path) -> io::Result<Self> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let path = parent.join(format!(
                ".nvcc-tmp-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Returns true only for a verified cache hit. Missing/invalid state rebuilds.
impl Cache {
    pub fn compile(&mut self, command: &Command, source: &Path, object: &Path) -> io::Result<bool> {
        let deps = object.with_extension("o.d");
        let stamp = object.with_extension("o.cache");
        let env = environment(command);
        // These can override -o/-MF or inject options files. Preserve their original
        // semantics by running uncached with the final output path.
        let injected = INJECTION_ENV
            .iter()
            .any(|key| env.get(OsStr::new(key)).is_some_and(|v| !v.is_empty()));
        let context = if injected {
            Err(invalid("NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS is set"))
        } else {
            self.context(command, source, object)
        };
        if let Ok(context) = context {
            if let (Ok(saved), Ok(current)) = (
                fs::read_to_string(&stamp),
                signature(command, source, object, &deps, context),
            ) {
                if saved == current {
                    return Ok(true);
                }
            }
        }
        match fs::remove_file(&stamp) {
            Ok(()) => (),
            Err(error) if error.kind() == io::ErrorKind::NotFound => (),
            Err(error) => return Err(error),
        }
        let scratch = Scratch::new(
            object
                .parent()
                .ok_or_else(|| invalid("object without parent"))?,
        )?;
        let staged_object = scratch.0.join("object.o");
        let staged_deps = scratch.0.join("object.d");
        let mut run = Command::new(command.get_program());
        run.args(command.get_args()).envs(&env);
        if let Some(cwd) = command.get_current_dir() {
            run.current_dir(cwd);
        }
        for (key, value) in command.get_envs() {
            if value.is_none() {
                run.env_remove(key);
            }
        }
        run.arg("-o")
            .arg(if injected { object } else { &staged_object });
        if !injected {
            run.arg("-MD").arg("-MF").arg(&staged_deps);
        }
        let started = SystemTime::now();
        let status = run.status()?;
        if !status.success() {
            return Err(io::Error::other(format!(
                "compile {} failed with {status}",
                source.display()
            )));
        }
        if injected {
            println!(
                "cargo:warning=nvcc cache bypass: NVCC_PREPEND_FLAGS/NVCC_APPEND_FLAGS is set"
            );
            return Ok(false);
        }
        if fs::metadata(&staged_object)?.len() == 0 {
            return Err(invalid("nvcc produced an empty object"));
        }
        let cwd = std::env::current_dir()?.join(command.get_current_dir().unwrap_or(Path::new("")));
        let stable_deps = fs::read(&staged_deps)
            .and_then(|bytes| dependencies(&bytes, &cwd))
            .and_then(|mut paths| {
                paths.push(cwd.join(source));
                paths.into_iter().try_for_each(|path| {
                    if fs::metadata(path)?.modified()? > started {
                        Err(invalid("dependency changed during compilation"))
                    } else {
                        Ok(())
                    }
                })
            });
        fs::rename(&staged_object, object)?;
        // An absent/malformed depfile still permits this build, but never a hit.
        if staged_deps.is_file() {
            fs::rename(&staged_deps, &deps)?;
        }
        match stable_deps
            .and(context)
            .and_then(|context| signature(command, source, object, &deps, context))
        {
            Ok(value) => {
                let staged_stamp = scratch.0.join("stamp");
                fs::write(&staged_stamp, value)?;
                fs::rename(staged_stamp, stamp)?; // Publish last: an interrupted build cannot hit.
            }
            Err(error) => println!(
                "cargo:warning=nvcc cache bypass for {}: {error}",
                source.display()
            ),
        }
        Ok(false)
    }
}

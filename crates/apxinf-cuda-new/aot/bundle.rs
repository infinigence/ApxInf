//! Read and verify native AOT artifacts without executing Python or CUDA.
use serde_json::{json, Value};
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub fn digest(path: &Path) -> Result<String> {
    // Use the host SHA-256 utility so the standalone build tool needs no
    // additional Rust dependencies beyond the crate's existing serde_json.
    let mut command = if cfg!(target_os = "macos") {
        let mut cmd = Command::new("shasum");
        cmd.args(["-a", "256"]);
        cmd
    } else {
        Command::new("sha256sum")
    };
    let output = command.arg(path).output()?;
    if !output.status.success() {
        return Err(format!("hash failed: {}", path.display()).into());
    }
    let text = String::from_utf8(output.stdout)?;
    let hash = text.split_whitespace().next().ok_or("missing digest")?;
    if hash.len() != 64 || !hash.bytes().all(|c| c.is_ascii_hexdigit()) {
        return Err("invalid SHA256 utility output".into());
    }
    Ok(hash.to_owned())
}

fn string<'a>(value: &'a Value, field: &str) -> Result<&'a str> {
    value[field]
        .as_str()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| format!("missing {field}").into())
}

/// Compare a bundle with the reviewed recipe, including the exporter that
/// determines compiler specialization. Rehashing an edited exporter cannot
/// silently bless a different kernel under the same tensor contract.
pub fn verify_recipe(kernel: &Value, recipe: &Value) -> Result<()> {
    for field in ["symbol", "contract", "specialization"] {
        if kernel[field].is_null() || kernel[field] != recipe[field] {
            return Err(format!("AOT {field} mismatch for {}", recipe["id"]).into());
        }
    }
    if string(&kernel["source"], "exporter_sha256")? != string(recipe, "exporter_sha256")? {
        return Err(format!("AOT exporter mismatch for {}", recipe["id"]).into());
    }
    for field in ["project", "revision", "license"] {
        if kernel["source"][field] != recipe["source"][field] {
            return Err(format!("AOT source {field} mismatch for {}", recipe["id"]).into());
        }
    }
    Ok(())
}

pub fn verify(manifest: &Path, target: &str, sm: &str, cuda: &str) -> Result<Value> {
    let manifest = manifest.canonicalize()?;
    let data: Value = serde_json::from_slice(&fs::read(&manifest)?)?;
    if data["schema"] != 1 {
        return Err("unsupported AOT schema".into());
    }
    if target != "aarch64-unknown-linux-gnu" || !matches!(sm, "sm_110" | "sm_110a") {
        return Err("AOT currently supports Linux AArch64 SM110".into());
    }
    for (field, expected) in [
        ("target", target),
        ("sm", "sm_110"),
        ("cuda", cuda),
        ("cutlass_dsl", "4.7.0"),
    ] {
        if data[field].as_str() != Some(expected) {
            return Err(format!("AOT {field}: expected {expected}, got {}", data[field]).into());
        }
    }
    let mut inputs = vec![manifest.clone()];
    let mut hashes = Vec::new();
    let mut artifact = |entry: &Value, kind: &str| -> Result<PathBuf> {
        let path = manifest
            .parent()
            .unwrap()
            .join(string(entry, "path")?)
            .canonicalize()?;
        if !path.is_file() {
            return Err(format!("not a file: {}", path.display()).into());
        }
        let hash = digest(&path)?;
        if entry["sha256"].as_str() != Some(hash.as_str()) {
            return Err(format!("AOT {kind} checksum mismatch: {}", path.display()).into());
        }
        let bytes = fs::read(&path)?;
        if kind == "object"
            && (bytes.get(..6) != Some(b"\x7fELF\x02\x01")
                || bytes.get(16..20) != Some(b"\x01\x00\xb7\x00"))
        {
            return Err(format!("not an AArch64 ELF64 relocatable: {}", path.display()).into());
        }
        if kind == "runtime" && !bytes.starts_with(b"!<arch>\n") {
            return Err("AOT runtime is not a static archive".into());
        }
        inputs.push(path.clone());
        hashes.push(json!([kind, hash]));
        Ok(path)
    };
    let kernels = data["kernels"]
        .as_array()
        .filter(|v| !v.is_empty())
        .ok_or("no AOT kernels")?;
    let mut ids = BTreeSet::new();
    let mut symbols = BTreeSet::new();
    let mut resolved = Vec::new();
    let mut contracts = Vec::new();
    for kernel in kernels {
        let id = string(kernel, "id")?;
        let symbol = string(kernel, "symbol")?;
        if !ids.insert(id)
            || !symbols.insert(symbol)
            || !symbol
                .bytes()
                .enumerate()
                .all(|(i, c)| c == b'_' || c.is_ascii_alphabetic() || (i > 0 && c.is_ascii_digit()))
        {
            return Err("duplicate kernel identity or invalid symbol".into());
        }
        for field in ["source", "contract", "specialization"] {
            if kernel[field].as_object().is_none_or(|v| v.is_empty()) {
                return Err(format!("{id}: missing {field}").into());
            }
        }
        for field in ["project", "revision", "license", "exporter_sha256"] {
            string(&kernel["source"], field)?;
        }
        let exporter = string(&kernel["source"], "exporter_sha256")?;
        if exporter.len() != 64 || !exporter.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(format!("{id}: invalid exporter SHA256").into());
        }
        let object = artifact(&kernel["object"], "object")?;
        let header = artifact(&kernel["header"], "header")?;
        if !fs::read_to_string(&header)?.contains(&format!("{symbol}_Kernel_Module_t")) {
            return Err(format!("{id}: header ABI mismatch").into());
        }
        resolved.push(json!({"id":id,"symbol":symbol,"object":object,"header":header}));
        let mut contract = kernel.clone();
        contract.as_object_mut().unwrap().remove("object");
        contract.as_object_mut().unwrap().remove("header");
        contracts.push(contract);
    }
    let runtime = artifact(&data["runtime"], "runtime")?;
    // The semantic identity excludes storage paths; serialized bytes are used
    // directly in the enclosing kernel build fingerprint.
    let fingerprint =
        serde_json::to_string(&json!({"schema":1,"target":target,"sm":sm,"cuda":cuda,
        "cutlass_dsl":data["cutlass_dsl"],"artifacts":hashes,"kernels":contracts}))?;
    Ok(json!({"kernels":resolved,"runtime":runtime,"inputs":inputs,"fingerprint":fingerprint}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    fn fixture() -> (PathBuf, Value) {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../devlocal/qwen-drive-performance/aot-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&root).unwrap();
        let mut object = vec![0; 20];
        object[..6].copy_from_slice(b"\x7fELF\x02\x01");
        object[16..20].copy_from_slice(b"\x01\x00\xb7\x00");
        fs::write(root.join("kernel.o"), object).unwrap();
        fs::write(
            root.join("kernel.h"),
            "typedef struct {} test_kernel_Kernel_Module_t;",
        )
        .unwrap();
        fs::write(root.join("runtime.a"), b"!<arch>\n").unwrap();
        let artifact = |name: &str| json!({"path":name,"sha256":digest(&root.join(name)).unwrap()});
        let data = json!({"schema":1,"target":"aarch64-unknown-linux-gnu","sm":"sm_110",
            "cuda":"13.2","cutlass_dsl":"4.7.0","runtime":artifact("runtime.a"),
            "kernels":[{"id":"test","symbol":"test_kernel","contract":{"shape":[10,256]},
                "specialization":{"shape":[10,256]},
                "source":{"project":"test","revision":"pinned","license":"Apache-2.0","exporter_sha256":"a".repeat(64)},
                "object":artifact("kernel.o"),"header":artifact("kernel.h")}]});
        (root, data)
    }
    fn check(root: &Path, data: &Value) -> Result<Value> {
        let path = root.join("manifest.json");
        fs::write(&path, serde_json::to_vec(data)?)?;
        verify(&path, "aarch64-unknown-linux-gnu", "sm_110", "13.2")
    }
    #[test]
    fn edited_exporter_and_missing_specialization_are_rejected() {
        let (root, mut data) = fixture();
        let mut recipe = data["kernels"][0].clone();
        recipe["exporter_sha256"] = recipe["source"]["exporter_sha256"].clone();
        assert!(verify_recipe(&data["kernels"][0], &recipe).is_ok());
        data["kernels"][0]["source"]["exporter_sha256"] = json!("b".repeat(64));
        assert!(verify_recipe(&data["kernels"][0], &recipe).is_err());
        data["kernels"][0]
            .as_object_mut()
            .unwrap()
            .remove("specialization");
        assert!(check(&root, &data).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn architecture_specific_sm110_accepts_the_same_bundle() {
        let (root, data) = fixture();
        check(&root, &data).unwrap();
        assert!(verify(
            &root.join("manifest.json"),
            "aarch64-unknown-linux-gnu",
            "sm_110a",
            "13.2"
        )
        .is_ok());
    }
    #[test]
    fn identity_tracks_contract_and_all_inputs() {
        let (root, mut data) = fixture();
        let first = check(&root, &data).unwrap();
        assert_eq!(first["inputs"].as_array().unwrap().len(), 4);
        data["kernels"][0]["contract"]["shape"][0] = json!(11);
        assert_ne!(
            first["fingerprint"],
            check(&root, &data).unwrap()["fingerprint"]
        );
    }
    #[test]
    fn damaged_and_missing_artifacts_are_rejected() {
        let (root, data) = fixture();
        fs::write(root.join("kernel.o"), b"damaged").unwrap();
        assert!(check(&root, &data)
            .unwrap_err()
            .to_string()
            .contains("checksum"));
        let (root, data) = fixture();
        fs::remove_file(root.join("runtime.a")).unwrap();
        assert!(check(&root, &data).is_err());
    }
    #[test]
    fn wrong_target_and_toolchain_are_rejected() {
        let (root, data) = fixture();
        for (field, value) in [
            ("target", json!("x86_64-unknown-linux-gnu")),
            ("sm", json!("sm_100")),
            ("cuda", json!("13.0")),
            ("cutlass_dsl", json!("4.6.0")),
            ("schema", json!(2)),
        ] {
            let mut changed = data.clone();
            changed[field] = value;
            assert!(check(&root, &changed).is_err());
        }
    }
    #[test]
    fn object_architecture_is_checked_after_hash_validation() {
        let (root, mut data) = fixture();
        let path = root.join("kernel.o");
        let mut bytes = fs::read(&path).unwrap();
        bytes[18..20].copy_from_slice(b"\x3e\x00");
        fs::write(&path, bytes).unwrap();
        data["kernels"][0]["object"]["sha256"] = json!(digest(&path).unwrap());
        assert!(check(&root, &data)
            .unwrap_err()
            .to_string()
            .contains("AArch64"));
    }
    #[test]
    fn duplicate_identity_and_wrong_header_are_rejected() {
        let (root, mut data) = fixture();
        let mut duplicate = data.clone();
        duplicate["kernels"]
            .as_array_mut()
            .unwrap()
            .push(data["kernels"][0].clone());
        assert!(check(&root, &duplicate)
            .unwrap_err()
            .to_string()
            .contains("duplicate"));
        data["kernels"][0]["symbol"] = json!("other");
        assert!(check(&root, &data).unwrap_err().to_string().contains("ABI"));
    }
}

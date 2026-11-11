use crate::{
    contracts,
    supervisor::{file_digest, Config, WorkerRuntime},
};
use serde_json::{json, Value};
use std::{collections::BTreeSet, path::Path};

fn verify_file(record: &Value, selected: &Path, preserve_launch_path: bool) -> Result<(), String> {
    let resolved = std::fs::canonicalize(selected).map_err(|error| error.to_string())?;
    let recorded = Path::new(record["path"].as_str().ok_or("Missing Clef file path.")?);
    if recorded
        != if preserve_launch_path {
            selected
        } else {
            &resolved
        }
    {
        return Err(format!(
            "The Clef file path differs: {}",
            selected.display()
        ));
    }
    let (_, digest) = file_digest(&resolved)?;
    if record["sha256"] != digest
        || std::fs::canonicalize(selected).map_err(|error| error.to_string())? != resolved
    {
        return Err(format!(
            "The Clef file digest differs: {}",
            selected.display()
        ));
    }
    Ok(())
}

pub(crate) fn verify_manifest(manifest: &Value, config: &Config) -> Result<(), String> {
    config.validate_memory()?;
    if manifest["artifacts"] != contracts::clef::approved_artifacts() {
        return Err("The Clef artifact manifest differs from the approved checkpoint.".into());
    }
    let WorkerRuntime::PythonClef { loader } = &config.runtime else {
        return Err("Clef requires the selected decision provider.".into());
    };
    if manifest["execution"]
        != json!({"provider":"python-clef","precision":"backbone-4bit-head-bf16","backend":"lm","max_context":config.max_context,"max_output_tokens":0,"idle_cache_limit_bytes":0,"memory_limit_bytes":config.memory_budget,"truncate":false,"kv_reuse":false,"worker_lifetime":"one-attempt"})
    {
        return Err("The Clef execution identity differs from its launch configuration.".into());
    }
    verify_file(&manifest["loader"], loader, false)?;
    if manifest["loader"]["sha256"] != contracts::clef::LOADER_SHA256 {
        return Err("The Clef loader digest is not approved.".into());
    }
    verify_file(&manifest["python"], &config.python, true)?;
    for name in contracts::clef::SOURCE_NAMES {
        verify_file(
            &manifest["sources"][name],
            &config.worker.with_file_name(name),
            false,
        )?;
    }
    let bin = std::fs::canonicalize(
        config
            .python
            .parent()
            .ok_or("Missing Python parent path.")?,
    )
    .map_err(|error| error.to_string())?;
    let environment = bin.parent().ok_or("Missing Python environment path.")?;
    if bin.file_name().is_none_or(|name| name != "bin") || !environment.join("pyvenv.cfg").is_file()
    {
        return Err("The Clef Python path must select a virtual environment.".into());
    }
    let packages = environment.join("lib/python3.12/site-packages");
    for (name, relative) in [
        ("mlx_core", "mlx/core.cpython-312-darwin.so"),
        ("mlx_lm", "mlx_lm/__init__.py"),
        ("transformers", "transformers/__init__.py"),
        ("numpy", "numpy/__init__.py"),
        ("pillow", "PIL/__init__.py"),
        ("tokenizers", "tokenizers/__init__.py"),
    ] {
        let path = packages.join(relative);
        let resolved = std::fs::canonicalize(&path).map_err(|error| error.to_string())?;
        if !resolved.starts_with(environment) {
            return Err("A Clef runtime module leaves the selected environment.".into());
        }
        verify_file(&manifest["runtime_files"]["modules"][name], &path, false)?;
    }
    let library = packages.join("mlx/lib/libmlx.dylib");
    if !std::fs::canonicalize(&library)
        .map_err(|error| error.to_string())?
        .starts_with(environment)
    {
        return Err("The Clef MLX library leaves the selected environment.".into());
    }
    verify_file(&manifest["runtime_files"]["libraries"][0], &library, false)?;
    verify_artifacts(manifest, &config.model)
}

fn verify_artifacts(manifest: &Value, model: &Path) -> Result<(), String> {
    let mut names = BTreeSet::new();
    for entry in std::fs::read_dir(model).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let kind = entry.file_type().map_err(|error| error.to_string())?;
        if !kind.is_file() {
            continue;
        }
        let path = entry.path();
        if path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| ["json", "safetensors", "jinja"].contains(&extension))
        {
            names.insert(
                entry
                    .file_name()
                    .into_string()
                    .map_err(|_| "Non-UTF-8 Clef artifact filename.")?,
            );
        }
    }
    for required in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "joint_head_config.json",
        "joint_head.safetensors",
        "model.safetensors.index.json",
    ] {
        if !names.contains(required) {
            return Err(format!("Missing required Clef artifact: {required}"));
        }
    }
    let artifacts = manifest["artifacts"]
        .as_array()
        .ok_or("Missing Clef artifact list.")?;
    if artifacts.len() != names.len() {
        return Err("The Clef artifact set differs from local files.".into());
    }
    for (name, record) in names.iter().zip(artifacts) {
        let path = model.join(name);
        let (size, digest) = file_digest(&path)?;
        if record["path"] != *name
            || record["size_bytes"] != size
            || record["sha256"] != digest
            || std::fs::symlink_metadata(&path)
                .map_err(|error| error.to_string())?
                .file_type()
                .is_symlink()
        {
            return Err(format!("The Clef artifact identity differs: {name}"));
        }
    }
    let index = contracts::parse_document(
        &std::fs::read(model.join("model.safetensors.index.json"))
            .map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;
    let weight_map = index["weight_map"]
        .as_object()
        .ok_or("Missing Clef weight map.")?;
    if weight_map.is_empty()
        || weight_map.values().any(|value| {
            value
                .as_str()
                .is_none_or(|name| !names.contains(name) || !name.ends_with(".safetensors"))
        })
    {
        return Err("The Clef weight index names an unavailable shard.".into());
    }
    Ok(())
}

pub(crate) fn verify_prepared(preparation: &Value, ready: &Value) -> Result<(), String> {
    let record = crate::decision::validate_record(&preparation["record"])
        .map_err(|error| error.to_string())?;
    crate::decision::validate_encoding(
        &preparation["encoding"],
        &record,
        ready["capabilities"]["max_context"]
            .as_u64()
            .ok_or("Missing Clef context limit.")? as usize,
        ready["capabilities"]["vocab_size"]
            .as_u64()
            .ok_or("Missing Clef vocabulary size.")?,
    )
    .map_err(|error| error.to_string())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clef_file_identity_checks_path_content_and_launch_symlink() {
        let root =
            std::env::temp_dir().join(format!("apxinf-clef-identity-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(root.join("bin")).unwrap();
        let file = root.join("interpreter");
        std::fs::write(&file, b"original identity fixture").unwrap();
        let (_, digest) = file_digest(&file).unwrap();
        let resolved = std::fs::canonicalize(&file).unwrap();
        let record = json!({"path":resolved,"sha256":digest});
        verify_file(&record, &file, false).unwrap();
        let mut wrong = record.clone();
        wrong["path"] = json!(root.join("other"));
        assert!(verify_file(&wrong, &file, false).is_err());
        #[cfg(unix)]
        {
            let link = root.join("bin/python");
            std::os::unix::fs::symlink(&file, &link).unwrap();
            let launch = std::fs::canonicalize(link.parent().unwrap())
                .unwrap()
                .join("python");
            let linked = json!({"path":launch,"sha256":digest});
            verify_file(&linked, &launch, true).unwrap();
            assert!(verify_file(&record, &launch, true).is_err());
        }
        std::fs::write(&file, b"changed identity fixture").unwrap();
        assert!(verify_file(&record, &file, false).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}

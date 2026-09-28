//! Build-time AOT bundle validation and link inputs. No runtime configuration.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "bundle.rs"]
mod bundle;

pub struct Kernel {
    pub object: PathBuf,
    pub header: PathBuf,
}

pub struct Bundle {
    pub kernels: BTreeMap<String, Kernel>,
    pub runtime: PathBuf,
    pub fingerprint: String,
}

impl Bundle {
    pub fn load(crate_root: &Path, manifest: &Path, target: &str, sm: &str, nvcc: &Path) -> Self {
        let version = Command::new(nvcc)
            .arg("--version")
            .output()
            .expect("query CUDA compiler version for AOT bundle");
        assert!(version.status.success(), "nvcc --version failed");
        let version = String::from_utf8(version.stdout).expect("nvcc version is UTF-8");
        let cuda = version
            .split("release ")
            .nth(1)
            .and_then(|part| part.split(',').next())
            .expect("nvcc release version");
        println!(
            "cargo:rerun-if-changed={}",
            crate_root.join("aot/bundle.rs").display()
        );
        println!("cargo:rerun-if-changed={}", manifest.display());
        let result = bundle::verify(manifest, target, sm, cuda)
            .unwrap_or_else(|error| panic!("AOT bundle rejected: {error}"));
        // A valid checksum does not establish compatibility with our adapters.
        // Require the exact exported symbols and tensor contracts they consume.
        let expected: serde_json::Value =
            serde_json::from_str(include_str!("manifest.json")).expect("AOT recipes");
        let actual: serde_json::Value =
            serde_json::from_slice(&std::fs::read(manifest).expect("read AOT manifest"))
                .expect("parse AOT manifest");
        for recipe in expected["kernels"].as_array().unwrap() {
            let kernel = actual["kernels"]
                .as_array()
                .unwrap()
                .iter()
                .find(|kernel| kernel["id"] == recipe["id"])
                .unwrap_or_else(|| panic!("required AOT kernel {} is missing", recipe["id"]));
            bundle::verify_recipe(kernel, recipe)
                .unwrap_or_else(|error| panic!("AOT bundle rejected: {error}"));
            let exporter = crate_root
                .join("aot/exporters")
                .join(recipe["exporter"].as_str().unwrap());
            println!("cargo:rerun-if-changed={}", exporter.display());
            assert_eq!(
                bundle::digest(&exporter).expect("hash committed exporter"),
                recipe["exporter_sha256"].as_str().expect("pinned exporter"),
                "AOT exporter differs from reviewed recipe"
            );
        }
        for path in result["inputs"].as_array().expect("AOT input paths") {
            println!(
                "cargo:rerun-if-changed={}",
                path.as_str().expect("AOT input path")
            );
        }
        let kernels = result["kernels"]
            .as_array()
            .expect("AOT kernels")
            .iter()
            .map(|kernel| {
                (
                    kernel["id"].as_str().unwrap().to_owned(),
                    Kernel {
                        object: PathBuf::from(kernel["object"].as_str().unwrap()),
                        header: PathBuf::from(kernel["header"].as_str().unwrap()),
                    },
                )
            })
            .collect();
        Self {
            kernels,
            runtime: PathBuf::from(result["runtime"].as_str().unwrap()),
            fingerprint: result["fingerprint"].as_str().unwrap().to_owned(),
        }
    }

    pub fn kernel(&self, id: &str) -> &Kernel {
        self.kernels
            .get(id)
            .unwrap_or_else(|| panic!("required AOT kernel {id} is missing"))
    }

    pub fn link_runtime(&self, out: &Path) {
        // Give the archive a deterministic link name without changing its source.
        let staged = out.join("libapxinf_cute_runtime.a");
        std::fs::copy(&self.runtime, &staged).expect("stage verified CuTe runtime archive");
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=apxinf_cute_runtime");
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=dl");
        println!("cargo:rustc-link-lib=pthread");
    }
}

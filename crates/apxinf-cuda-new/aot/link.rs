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
    /// Load every kernel this crate reviewed from the artifact manifest.
    ///
    /// The required set is the full recipe list in
    /// `<crate_root>/aot/manifest.json`, so adding a reviewed recipe cannot
    /// silently stop linking its object.
    ///
    /// Returns `None` when the bundle carries none of them. One bundle may
    /// serve several model crates, so a bundle exported for another crate's
    /// recipes is "no bundle" here rather than an error; a bundle holding only
    /// some of them is still rejected by `load_required`.
    // `link.rs` is shared by two build scripts; only apxinf-cuda wants the
    // whole-recipe form, apxinf-cuda-new requests a single kernel by id.
    #[allow(dead_code)]
    pub fn load(
        crate_root: &Path,
        manifest: &Path,
        target: &str,
        sm: &str,
        nvcc: &Path,
    ) -> Option<Self> {
        let aot_root = crate_root.join("aot");
        let recipes = aot_root.join("manifest.json");
        let reviewed: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&recipes).expect("read reviewed AOT recipes"))
                .expect("parse reviewed AOT recipes");
        let ids = reviewed["kernels"]
            .as_array()
            .expect("reviewed AOT kernels must be an array")
            .iter()
            .map(|kernel| {
                kernel["id"]
                    .as_str()
                    .expect("reviewed AOT kernel id")
                    .to_owned()
            })
            .collect::<Vec<_>>();
        let required = ids.iter().map(String::as_str).collect::<Vec<_>>();
        Self::load_required(&aot_root, &recipes, manifest, &required, target, sm, nvcc)
    }

    /// Load the requested kernels from a shared artifact manifest.
    ///
    /// One AOT manifest may carry operators for several model crates.  If none
    /// of `required` is present this consumer simply has no AOT bundle; if a
    /// subset is present, the incomplete bundle is rejected.  Only requested
    /// objects are returned and linked.
    pub fn load_required(
        aot_root: &Path,
        recipes: &Path,
        manifest: &Path,
        required: &[&str],
        target: &str,
        sm: &str,
        nvcc: &Path,
    ) -> Option<Self> {
        assert!(!required.is_empty(), "AOT required kernel list is empty");
        println!("cargo:rerun-if-changed={}", manifest.display());
        let actual: serde_json::Value =
            serde_json::from_slice(&std::fs::read(manifest).expect("read AOT manifest"))
                .expect("parse AOT manifest");
        let actual_kernels = actual["kernels"]
            .as_array()
            .expect("AOT manifest kernels must be an array");
        let present = required
            .iter()
            .filter(|id| actual_kernels.iter().any(|kernel| kernel["id"] == **id))
            .count();
        if present == 0 {
            return None;
        }
        assert_eq!(
            present,
            required.len(),
            "AOT bundle contains only {present}/{} required kernels: {}",
            required.len(),
            required.join(", ")
        );
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
            aot_root.join("bundle.rs").display()
        );
        println!("cargo:rerun-if-changed={}", recipes.display());
        let result = bundle::verify(manifest, target, sm, cuda)
            .unwrap_or_else(|error| panic!("AOT bundle rejected: {error}"));
        // A valid checksum does not establish compatibility with our adapters.
        // Require the exact exported symbols and tensor contracts they consume.
        let expected: serde_json::Value =
            serde_json::from_slice(&std::fs::read(recipes).expect("read reviewed AOT recipes"))
                .expect("parse reviewed AOT recipes");
        for id in required {
            let recipe = expected["kernels"]
                .as_array()
                .expect("reviewed AOT kernels must be an array")
                .iter()
                .find(|kernel| kernel["id"] == *id)
                .unwrap_or_else(|| panic!("required AOT recipe {id} is missing"));
            let kernel = actual["kernels"]
                .as_array()
                .unwrap()
                .iter()
                .find(|kernel| kernel["id"] == recipe["id"])
                .unwrap_or_else(|| panic!("required AOT kernel {} is missing", recipe["id"]));
            bundle::verify_recipe(kernel, recipe)
                .unwrap_or_else(|error| panic!("AOT bundle rejected: {error}"));
            let exporter = aot_root
                .join("exporters")
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
            .filter(|kernel| required.iter().any(|id| kernel["id"].as_str() == Some(*id)))
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
        Some(Self {
            kernels,
            runtime: PathBuf::from(result["runtime"].as_str().unwrap()),
            fingerprint: result["fingerprint"].as_str().unwrap().to_owned(),
        })
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

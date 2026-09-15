use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const DEVICE_PROBE_SOURCE: &str = r#"
#include <cuda_runtime_api.h>
#include <cstdio>
int main() {
    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDevice failed: %s\n", cudaGetErrorString(status));
        return 2;
    }
    cudaDeviceProp properties{};
    status = cudaGetDeviceProperties(&properties, device);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceProperties(%d) failed: %s\n",
                     device, cudaGetErrorString(status));
        return 3;
    }
    std::printf("APXINF_CUDA_DEVICE %d %d %d %s\n", device,
                properties.major, properties.minor, properties.name);
    return 0;
}
"#;

const ARCH_CHECK_SOURCE: &str = r#"
__global__ void apxinf_cuda_arch_check() {}
"#;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ComputeCapability {
    major: u32,
    minor: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArchSource {
    Explicit,
    Detected { device: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchTarget {
    pub nvcc_arch: String,
    pub cutlass_arch: String,
}

impl ArchTarget {
    pub fn sm(&self) -> u32 {
        arch_sm(&self.nvcc_arch).expect("validated CUDA architecture")
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchSelection {
    pub targets: Vec<ArchTarget>,
    pub source: ArchSource,
}

pub const DEVICE_FEATURE_NATIVE_FP8: u64 = 1 << 0;
pub const DEVICE_FEATURE_CUTLASS_SM100: u64 = 1 << 1;

pub fn is_cutlass_sm100_family(arch: &str) -> bool {
    matches!(
        arch,
        "sm_100" | "sm_100a" | "sm_101" | "sm_101a" | "sm_110" | "sm_110a" | "sm_120" | "sm_120a"
    )
}

pub fn has_native_fp8(arch: &str) -> bool {
    matches!(
        arch,
        "sm_89"
            | "sm_90"
            | "sm_90a"
            | "sm_100"
            | "sm_100a"
            | "sm_101"
            | "sm_101a"
            | "sm_110"
            | "sm_110a"
            | "sm_120"
            | "sm_120a"
    )
}

pub fn target_features(target: &ArchTarget) -> u64 {
    let mut features = 0;
    if has_native_fp8(&target.nvcc_arch) {
        features |= DEVICE_FEATURE_NATIVE_FP8;
    }
    if is_cutlass_sm100_family(&target.cutlass_arch) {
        features |= DEVICE_FEATURE_CUTLASS_SM100;
    }
    features
}

#[cfg(test)]
fn supports_target(targets: &[ArchTarget], sm: u32, required_features: u64) -> bool {
    targets.iter().any(|target| {
        target.sm() == sm && target_features(target) & required_features == required_features
    })
}

pub fn select_cuda_arch(
    explicit_nvcc_arch: Option<String>,
    explicit_cutlass_arch: Option<String>,
    host: &str,
    target: &str,
    nvcc: &Path,
    out_dir: &Path,
) -> Result<ArchSelection, String> {
    let (nvcc_arches, source) = if let Some(arches) = explicit_nvcc_arch {
        (parse_arch_list(&arches)?, ArchSource::Explicit)
    } else {
        if host != target {
            return Err(format!(
                "cannot auto-detect the target GPU while cross-compiling ({host} -> {target})"
            ));
        }
        let (device, capability) = detect_compute_capability(nvcc, out_dir)?;
        (
            vec![capability_to_arch(capability)],
            ArchSource::Detected { device },
        )
    };

    let cutlass_arches = match explicit_cutlass_arch {
        Some(arches) => {
            let parsed = parse_arch_list(&arches)?;
            if parsed.len() != nvcc_arches.len() {
                return Err(format!(
                    "APXINF_CUDA_ARCH_CUTLASS has {} target(s), but APXINF_CUDA_ARCH has {}; provide one corresponding CUTLASS target per CUDA target",
                    parsed.len(), nvcc_arches.len()
                ));
            }
            parsed
        }
        None => nvcc_arches
            .iter()
            .map(|arch| cutlass_arch_for(arch))
            .collect(),
    };

    let targets = nvcc_arches
        .into_iter()
        .zip(cutlass_arches)
        .map(|(nvcc_arch, cutlass_arch)| {
            if arch_sm(&nvcc_arch) != arch_sm(&cutlass_arch) {
                return Err(format!(
                    "CUDA target {nvcc_arch} and CUTLASS target {cutlass_arch} describe different compute capabilities"
                ));
            }
            Ok(ArchTarget { nvcc_arch, cutlass_arch })
        })
        .collect::<Result<Vec<_>, String>>()?;

    for arch in targets
        .iter()
        .flat_map(|target| [&target.nvcc_arch, &target.cutlass_arch])
    {
        validate_nvcc_arch(nvcc, out_dir, arch)?;
    }
    Ok(ArchSelection { targets, source })
}

pub fn parse_arch_list(value: &str) -> Result<Vec<String>, String> {
    let mut unique = BTreeMap::new();
    for item in value.split(|character: char| {
        character == ',' || character == ';' || character.is_ascii_whitespace()
    }) {
        if item.is_empty() {
            continue;
        }
        let arch = validate_arch_name(item)?.to_owned();
        let sm = arch_sm(&arch).expect("validated architecture");
        if let Some(previous) = unique.insert(sm, arch.clone()) {
            if previous != arch {
                return Err(format!(
                    "duplicate compute capability sm_{sm} specified as both {previous} and {arch}"
                ));
            }
        }
    }
    if unique.is_empty() {
        return Err("CUDA architecture list is empty".to_owned());
    }
    Ok(unique.into_values().collect())
}

pub fn gencode_args(arches: impl IntoIterator<Item = String>) -> Vec<String> {
    arches
        .into_iter()
        .flat_map(|arch| {
            let compute = arch.replacen("sm_", "compute_", 1);
            [
                "--generate-code".to_owned(),
                format!("arch={compute},code={arch}"),
            ]
        })
        .collect()
}

fn validate_arch_name(arch: &str) -> Result<&str, String> {
    let Some(suffix) = arch.strip_prefix("sm_") else {
        return Err(format!(
            "invalid CUDA architecture {arch:?}; expected a value such as sm_87, sm_101, or sm_110"
        ));
    };
    let digits = suffix.strip_suffix('a').unwrap_or(suffix);
    if digits.len() < 2 || !digits.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(format!(
            "invalid CUDA architecture {arch:?}; expected a value such as sm_87, sm_101, or sm_110"
        ));
    }
    Ok(arch)
}

fn arch_sm(arch: &str) -> Result<u32, String> {
    validate_arch_name(arch)?;
    arch.trim_start_matches("sm_")
        .trim_end_matches('a')
        .parse()
        .map_err(|error| format!("invalid CUDA architecture {arch:?}: {error}"))
}

fn cutlass_arch_for(nvcc_arch: &str) -> String {
    if is_cutlass_sm100_family(nvcc_arch) && !nvcc_arch.ends_with('a') {
        format!("{nvcc_arch}a")
    } else {
        nvcc_arch.to_owned()
    }
}

fn capability_to_arch(capability: ComputeCapability) -> String {
    format!("sm_{}{}", capability.major, capability.minor)
}

fn detect_compute_capability(
    nvcc: &Path,
    out_dir: &Path,
) -> Result<(usize, ComputeCapability), String> {
    let source = out_dir.join("apxinf_cuda_device_probe.cu");
    let executable = executable_path(out_dir, "apxinf_cuda_device_probe");
    std::fs::write(&source, DEVICE_PROBE_SOURCE)
        .map_err(|error| format!("write CUDA device probe {}: {error}", source.display()))?;
    let compile = Command::new(nvcc)
        .args(["-std=c++17", "-O0"])
        .arg(&source)
        .arg("-o")
        .arg(&executable)
        .output()
        .map_err(|error| format!("run {} for CUDA device probe: {error}", nvcc.display()))?;
    ensure_success("compile CUDA device probe", &compile)?;
    let probe = Command::new(&executable)
        .output()
        .map_err(|error| format!("run CUDA device probe {}: {error}", executable.display()))?;
    ensure_success("run CUDA device probe", &probe)?;
    parse_probe_output(&String::from_utf8_lossy(&probe.stdout))
}

fn parse_probe_output(output: &str) -> Result<(usize, ComputeCapability), String> {
    for line in output.lines() {
        let mut fields = line.split_whitespace();
        if fields.next() != Some("APXINF_CUDA_DEVICE") {
            continue;
        }
        let device = fields
            .next()
            .ok_or_else(|| format!("malformed CUDA device probe output: {line:?}"))?
            .parse::<usize>()
            .map_err(|error| format!("invalid CUDA device ordinal: {error}"))?;
        let major = fields
            .next()
            .ok_or_else(|| format!("missing compute capability for CUDA device {device}"))?
            .parse::<u32>()
            .map_err(|error| {
                format!("invalid compute capability for CUDA device {device}: {error}")
            })?;
        let minor = fields
            .next()
            .ok_or_else(|| format!("missing compute capability for CUDA device {device}"))?
            .parse::<u32>()
            .map_err(|error| {
                format!("invalid compute capability for CUDA device {device}: {error}")
            })?;
        return Ok((device, ComputeCapability { major, minor }));
    }
    Err(format!(
        "CUDA device probe produced no current-device record; stdout was {output:?}"
    ))
}

fn validate_nvcc_arch(nvcc: &Path, out_dir: &Path, arch: &str) -> Result<(), String> {
    let source = out_dir.join("apxinf_cuda_arch_check.cu");
    let object = out_dir.join(format!("apxinf_cuda_arch_check_{arch}.o"));
    std::fs::write(&source, ARCH_CHECK_SOURCE).map_err(|error| {
        format!(
            "write NVCC architecture check {}: {error}",
            source.display()
        )
    })?;
    let output = Command::new(nvcc)
        .arg("-c")
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .arg(format!("-arch={arch}"))
        .output()
        .map_err(|error| format!("run {} to validate {arch}: {error}", nvcc.display()))?;
    ensure_success(&format!("validate CUDA architecture {arch}"), &output)
}

fn ensure_success(action: &str, output: &Output) -> Result<(), String> {
    if output.status.success() {
        return Ok(());
    }
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
    Err(format!(
        "{action} failed with {}\nstdout:\n{}\nstderr:\n{}",
        output.status,
        if stdout.is_empty() {
            "<empty>"
        } else {
            &stdout
        },
        if stderr.is_empty() {
            "<empty>"
        } else {
            &stderr
        }
    ))
}

fn executable_path(out_dir: &Path, stem: &str) -> PathBuf {
    out_dir.join(format!("{stem}{}", std::env::consts::EXE_SUFFIX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_current_compute_capability() {
        let parsed = parse_probe_output("APXINF_CUDA_DEVICE 1 11 0 NVIDIA Thor\n").unwrap();
        assert_eq!(
            parsed,
            (
                1,
                ComputeCapability {
                    major: 11,
                    minor: 0
                }
            )
        );
        assert_eq!(capability_to_arch(parsed.1), "sm_110");
    }

    #[test]
    fn parses_and_canonicalizes_architecture_lists() {
        assert_eq!(
            parse_arch_list("sm_110, sm_87;sm_110").unwrap(),
            vec!["sm_87", "sm_110"]
        );
        assert!(parse_arch_list(" , ; ").is_err());
        assert!(parse_arch_list("sm_110,sm_110a").is_err());
    }

    #[test]
    fn emits_one_gencode_pair_per_architecture() {
        assert_eq!(
            gencode_args(["sm_87".to_owned(), "sm_110a".to_owned()]),
            [
                "--generate-code",
                "arch=compute_87,code=sm_87",
                "--generate-code",
                "arch=compute_110a,code=sm_110a"
            ]
        );
    }

    #[test]
    fn runtime_filter_requires_an_exact_compiled_target_and_capabilities() {
        let targets = vec![
            ArchTarget {
                nvcc_arch: "sm_87".to_owned(),
                cutlass_arch: "sm_87".to_owned(),
            },
            ArchTarget {
                nvcc_arch: "sm_110".to_owned(),
                cutlass_arch: "sm_110a".to_owned(),
            },
        ];
        assert!(supports_target(&targets, 87, 0));
        assert!(!supports_target(&targets, 87, DEVICE_FEATURE_NATIVE_FP8));
        assert!(supports_target(
            &targets,
            110,
            DEVICE_FEATURE_NATIVE_FP8 | DEVICE_FEATURE_CUTLASS_SM100
        ));
        assert!(!supports_target(&targets, 120, 0));
    }

    #[test]
    fn maps_known_blackwell_targets_to_arch_specific_cutlass() {
        assert_eq!(cutlass_arch_for("sm_101"), "sm_101a");
        assert_eq!(cutlass_arch_for("sm_110"), "sm_110a");
        assert_eq!(cutlass_arch_for("sm_87"), "sm_87");
        assert_eq!(cutlass_arch_for("sm_110a"), "sm_110a");
    }

    #[test]
    fn validates_architecture_names() {
        for valid in ["sm_87", "sm_101", "sm_110a"] {
            assert_eq!(validate_arch_name(valid).unwrap(), valid);
        }
        for invalid in ["", "87", "compute_87", "sm_", "sm_xx", "sm_87aa"] {
            assert!(validate_arch_name(invalid).is_err(), "{invalid}");
        }
    }

    #[test]
    fn cross_compilation_never_probes_the_host_gpu() {
        let error = select_cuda_arch(
            None,
            None,
            "x86_64-unknown-linux-gnu",
            "aarch64-unknown-linux-gnu",
            Path::new("nvcc-must-not-run"),
            Path::new("unused-out-dir"),
        )
        .unwrap_err();
        assert!(error.contains("cross-compiling"), "{error}");
    }
}

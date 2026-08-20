use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const DEVICE_PROBE_SOURCE: &str = r#"
#include <cuda_runtime_api.h>
#include <cstdio>

int main() {
    int device_count = 0;
    cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        std::fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(status));
        return 2;
    }
    if (device_count == 0) {
        std::fprintf(stderr, "CUDA reported zero visible devices\n");
        return 3;
    }

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp properties{};
        status = cudaGetDeviceProperties(&properties, device);
        if (status != cudaSuccess) {
            std::fprintf(
                stderr,
                "cudaGetDeviceProperties(%d) failed: %s\n",
                device,
                cudaGetErrorString(status));
            return 4;
        }
        std::printf(
            "APXINF_CUDA_DEVICE %d %d %d %s\n",
            device,
            properties.major,
            properties.minor,
            properties.name);
    }
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
    Detected { device_count: usize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchSelection {
    pub nvcc_arch: String,
    pub cutlass_arch: String,
    pub source: ArchSource,
}

pub fn is_cutlass_sm100_family(arch: &str) -> bool {
    matches!(
        arch,
        "sm_100" | "sm_100a" | "sm_101" | "sm_101a" | "sm_110" | "sm_110a" | "sm_120" | "sm_120a"
    )
}

pub fn select_cuda_arch(
    explicit_nvcc_arch: Option<String>,
    explicit_cutlass_arch: Option<String>,
    host: &str,
    target: &str,
    nvcc: &Path,
    out_dir: &Path,
) -> Result<ArchSelection, String> {
    let (nvcc_arch, source) = if let Some(arch) = explicit_nvcc_arch {
        (validate_arch_name(&arch)?.to_owned(), ArchSource::Explicit)
    } else {
        if host != target {
            return Err(format!(
                "cannot auto-detect the target GPU while cross-compiling ({host} -> {target})"
            ));
        }
        let capabilities = detect_compute_capabilities(nvcc, out_dir)?;
        let device_count = capabilities.len();
        (
            select_uniform_arch(&capabilities)?,
            ArchSource::Detected { device_count },
        )
    };

    let cutlass_arch = match explicit_cutlass_arch {
        Some(arch) => validate_arch_name(&arch)?.to_owned(),
        None => cutlass_arch_for(&nvcc_arch),
    };

    validate_nvcc_arch(nvcc, out_dir, &nvcc_arch)?;
    if cutlass_arch != nvcc_arch {
        validate_nvcc_arch(nvcc, out_dir, &cutlass_arch)?;
    }

    Ok(ArchSelection {
        nvcc_arch,
        cutlass_arch,
        source,
    })
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

fn cutlass_arch_for(nvcc_arch: &str) -> String {
    if is_cutlass_sm100_family(nvcc_arch) && !nvcc_arch.ends_with('a') {
        format!("{nvcc_arch}a")
    } else {
        nvcc_arch.to_owned()
    }
}

fn select_uniform_arch(capabilities: &[ComputeCapability]) -> Result<String, String> {
    if capabilities.is_empty() {
        return Err("CUDA reported zero visible devices".to_owned());
    }

    let unique: BTreeSet<_> = capabilities.iter().copied().collect();
    if unique.len() != 1 {
        let architectures = unique
            .into_iter()
            .map(capability_to_arch)
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "visible CUDA devices have different architectures ({architectures})"
        ));
    }

    Ok(capability_to_arch(*unique.iter().next().unwrap()))
}

fn capability_to_arch(capability: ComputeCapability) -> String {
    format!("sm_{}{}", capability.major, capability.minor)
}

fn detect_compute_capabilities(
    nvcc: &Path,
    out_dir: &Path,
) -> Result<Vec<ComputeCapability>, String> {
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

fn parse_probe_output(output: &str) -> Result<Vec<ComputeCapability>, String> {
    let mut capabilities = Vec::new();
    for line in output.lines() {
        let mut fields = line.split_whitespace();
        if fields.next() != Some("APXINF_CUDA_DEVICE") {
            continue;
        }
        let device = fields
            .next()
            .ok_or_else(|| format!("malformed CUDA device probe output: {line:?}"))?;
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
        capabilities.push(ComputeCapability { major, minor });
    }

    if capabilities.is_empty() {
        Err(format!(
            "CUDA device probe produced no device records; stdout was {output:?}"
        ))
    } else {
        Ok(capabilities)
    }
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
            stdout.as_str()
        },
        if stderr.is_empty() {
            "<empty>"
        } else {
            stderr.as_str()
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
    fn parses_compute_capabilities() {
        let parsed = parse_probe_output(
            "APXINF_CUDA_DEVICE 0 11 0 NVIDIA Thor\nAPXINF_CUDA_DEVICE 1 11 0 NVIDIA Thor\n",
        )
        .unwrap();
        assert_eq!(
            parsed,
            vec![
                ComputeCapability { major: 11, minor: 0 },
                ComputeCapability { major: 11, minor: 0 }
            ]
        );
        assert_eq!(select_uniform_arch(&parsed).unwrap(), "sm_110");
    }

    #[test]
    fn rejects_mixed_visible_architectures() {
        let capabilities = [
            ComputeCapability { major: 8, minor: 7 },
            ComputeCapability { major: 11, minor: 0 },
        ];
        let error = select_uniform_arch(&capabilities).unwrap_err();
        assert!(error.contains("sm_87, sm_110"), "{error}");
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

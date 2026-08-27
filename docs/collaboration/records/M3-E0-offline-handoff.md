# M3-E0 Offline Preparation Handoff

Date: 2026-08-23 (Asia/Shanghai)
Owner: member3
Scope: CPU/offline preparation only; no server benchmark, CUDA kernel run, or GPU1 oracle replay.

## Result

This handoff is development preparation evidence, not a performance result.
The benchmark portion is **inconclusive** because member1 has not supplied the
required `BASE_GOOD` commit, real serve command, feature-off/on flags, health /
correctness / reliability artifact, or GPU0 replay conditions. No latency,
TPOT, goodput, CV, peak VRAM, or end-to-end speedup is reported.

The local machine has an NVIDIA GeForce RTX 4080 Laptop GPU, not an RTX 4090.
Any GPU observation below is environment/development evidence only and cannot be
used for official scoring or an accepted official candidate.

## Machine

```text
Host: LAPTOP-H4O2JO2M
OS: Microsoft Windows 10.0.26200 (x64)
Git: 2.46.2.windows.1
Rust: rustc 1.98.0 (88d9e12ae 2026-08-18)
Cargo: cargo 1.98.0 (797e8a9bc 2026-08-05)
Python: 3.12.13 (Codex bundled interpreter)
GPU UUID: GPU-d09782a4-e31c-7858-228f-4e5c15d8d6b7
GPU model: NVIDIA GeForce RTX 4080 Laptop GPU
Driver: 592.00
nvidia-smi CUDA runtime report: 13.1
CUDA toolkit / nvcc: absent (`nvcc` was not found; CUDA_PATH and CUDA_HOME unset)
APXINF_CUDA_ARCH: not set; no CUDA build or kernel experiment was run
evidence_scope: development
```

The GPU was only queried with `nvidia-smi`; no model server or benchmark was
started. Desktop processes were using the GPU during the probe, so the memory
snapshot is not a benchmark measurement.

## Source And Payload Boundary

```text
BASE commit / scaffolding: 6c0f7ffc7f8900ed4ea931add801798135ecfad0
Working branch: exp/w4-gemv
Remote tracking branch: origin/exp/w4-gemv
Rollback SHA: 6c0f7ffc7f8900ed4ea931add801798135ecfad0
```

The committed metadata fixtures were read. No model payload, safetensors shard,
tokenizer bundle, or model weight was downloaded or opened. The inventory uses
only `fixtures/qwen35-metadata/config.json` and
`fixtures/qwen35-metadata/model.safetensors.index.json`.

## Commands And Exit Codes

All commands below were run from the repository root. The raw combined
stdout/stderr for each command is preserved outside Git at:

```text
C:\Users\WTM的~1\AppData\Local\Temp\apxinf-m3-e0-handoff\
```

Each `.log` file contains combined stdout/stderr and each `.exitcode` file
contains the recorded process exit code.

| Step | Command | Exit |
| --- | --- | ---: |
| 01 | Windows OS/hardware probes (`Get-CimInstance` plus runtime fallback) | 0 (CIM printed access denied; fallback values are recorded above) |
| 02 | `rustc --version; cargo --version; python.exe --version; git --version` | 0 |
| 03 | `cargo +stable fmt --all -- --check` | 1 (existing repository formatting drift; no mass-format) |
| 04 | `cargo +stable check --workspace --locked` | 0 |
| 05 | `python.exe benchmarks/qwen38_4090/evaluation/test.py check` | 0 (`assignment checks passed`) |
| 06 | `python.exe -m compileall -q benchmarks scripts` | 0 |
| 07 | `python.exe scripts/campaign/shape_inventory.py --config fixtures/qwen35-metadata/config.json --index fixtures/qwen35-metadata/model.safetensors.index.json` | 0 |
| 08 | `python.exe scripts/campaign/validate_experiment.py benchmarks/campaign/manifests/w4-gemv-baseline.json --mode template --json` | 0 (`valid: true`) |
| 09 | `python.exe scripts/campaign/validate_experiment.py benchmarks/campaign/manifests/w4-gemv-baseline.json --mode ready --json` | 1 (expected: placeholders and missing replay evidence) |
| 10a | `nvidia-smi --query-gpu=uuid,name,driver_version,memory.total --format=csv,noheader` | 0 |
| 10b | `nvidia-smi` | 0 |
| 10c | `nvcc --version` | N/A: PowerShell command resolution failed because `nvcc` is not installed; CUDA variables were empty |
| 11 | `git status --short --branch; git diff --check; git rev-parse HEAD; git diff --name-only ... -- benchmarks/qwen38_4090/evaluation` | 0 |

Step 10 is an environment probe only. It must not be interpreted as a CUDA
toolkit validation or benchmark run.

## Shape Inventory Summary

Artifact: `benchmarks/campaign/manifests/qwen38-shape-inventory.json`

```text
model revision: 63768c10df38c0395e12ef49edac1bd539eaeeea
tensor count: 2396
shard count: 5
packed W4 weight count: 399
layer count: 64 (48 linear_attention, 16 full_attention)
quantization: pack-quantized, 4-bit, group size 32, asymmetric
metadata total_size: 21017689808 bytes (19.574249 GiB)
derived GDN recurrent state: 150994944 bytes/request
derived full-attention KV: 65536 bytes/token/request
```

The index has tensor names and shard mapping but no SafeTensors per-tensor
headers, so category byte totals are intentionally not guessed.

Fixture hashes:

```text
config.json: FECE2915D4C8AD4C10877622F04EA5E01CD3AE38768CE5C1EDB700DD1DE290F6
model.safetensors.index.json: 82B1BF79F5B61333E83DA17EC3BF89C9F178E29395A14C6B3CE3BBC474E1EAD8
contract-v1.json: 520349B1279C3BF999A6848B296C23D20CDAEAB7420934E9196C90018BAC7433
```

## Manifest Validation

The template validator passed with no errors. Ready-mode validation correctly
failed because the template has no real `BASE_GOOD` / candidate commit, input
manifest or contract hash, server commands, GPU0 replay metadata, raw artifact,
correctness/reliability/recovery evidence, or measured latency series. The
template was not edited to fabricate any of these values.

Manifest hash:

```text
benchmarks/campaign/manifests/w4-gemv-baseline.json
F4FA1ED6624653495D12D609B2B059EB9B7396B962BE75CEBFB5AFFE56D02A95
```

## Artifact Hashes

```text
benchmarks/campaign/manifests/qwen38-shape-inventory.json
5172C37A641809E10C14B56FC71B07D077821CD091DA4CF8FCE904E3A7B960E1

benchmarks/campaign/manifests/w4-gemv-baseline.json
F4FA1ED6624653495D12D609B2B059EB9B7396B962BE75CEBFB5AFFE56D02A95

raw stdout/stderr directory (not committed):
C:\Users\WTM的~1\AppData\Local\Temp\apxinf-m3-e0-handoff\
SHA256SUMS.txt SHA256:
A1A88CB7618FF29370C996EBC1FD653AA93EB06E2DE0EA1698CE53D6E455CA0E
```

The raw logs are local evidence and are not claimed as official benchmark
artifacts. No `.ncu-rep`, `.nsys-rep`, evaluation run, or model payload was
created.

## Boundary And Next Dependency

Nothing in this handoff requests `/tmp/apxinf-gpu-job.lock`, starts a checkpoint
server, or claims an isolated kernel result. Before W4 GEMV A/B, member1 must
provide the complete `BASE_GOOD` SHA, real serve command, feature-off/on flags,
model revision, contract SHA256, health/correctness/reliability artifact, and
GPU0 replay conditions. Only then may the manifest be filled and a paired
warmup-1 / measured-5 experiment be run. A non-4090 or non-GPU0 run remains
`evidence_scope=development`; any missing raw artifact, full command, or SHA256
keeps the result `inconclusive`.

# GEMM tuning databases

Each hardware compatibility domain owns one shared `tactics.json` and one
diagnostic `tuning_report.json`. Records are keyed by the physical GEMM
contract, not by model, layer, or executor name.

```text
configs/tuning/<vendor>/<device-family>-sm<version>/cuda<major.minor>-cublas<major.minor>/
├── tactics.json
├── tuning_report.json
└── <exact-key-hash>.recipe
```

At runtime an exact key wins over a bucket key. Missing records use the safe
provider default in inference mode. With autotuning explicitly enabled, a real
request validates and benchmarks provider candidates, atomically merges the
exact winner into the hardware database, and records candidate measurements in
the report.

Legacy CUDA can also resolve older hardware-only directories when their records
are library-compatible. `kernel_build_id` and the full device name never reject
the whole legacy database. Provider
`implementation_version` (plus the relevant CUDA/cuBLAS compatibility for that
provider) controls record-local invalidation.

## cuda-new recipes

cuda-new GEMM and Attention default to the hardware/toolkit directory above,
independently of whether `tactics.json` exists. Each `.recipe` contains one exact
operator key and its selected implementation/configuration, not model weights or
a whole-model plan. Different models can share a record only when their exact
operator keys match. Build/device/library/shape/policy compatibility is validated
by the key; legacy JSON records are not imported.

PI05 `LoadOptions.autotune` and `pi05_bench --autotune` enable benchmarking on a
recipe miss during preparation or first execution, then persist the winner.
Without autotune, existing recipes are reused and misses use a safe fallback;
creating a cache directory alone does not tune anything. Explicit `cache_dir`
overrides the default; an empty string disables disk persistence. Paths are
relative to the process working directory unless explicitly absolute. Generated
recipes are ignored by Git, and writes require a writable directory.

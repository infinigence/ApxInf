# GEMM tuning databases

Each hardware compatibility domain owns one shared `tactics.json` and one
diagnostic `tuning_report.json`. Records are keyed by the physical GEMM
contract, not by model, layer, or executor name.

```text
configs/tuning/<vendor>/<device-family>-sm<version>/
├── tactics.json
└── tuning_report.json
```

At runtime an exact key wins over a bucket key. Missing records use the safe
provider default in inference mode. With autotuning explicitly enabled, a real
request validates and benchmarks provider candidates, atomically merges the
exact winner into the hardware database, and records candidate measurements in
the report.

`kernel_build_id`, device name, and library versions are not part of the
directory name and never reject the whole database. Provider
`implementation_version` (plus the relevant CUDA/cuBLAS compatibility for that
provider) controls record-local invalidation.

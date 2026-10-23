# Benchmark and memory acceptance

The tools distinguish planned, executed, generation-observed, and reached cases.
Historical stage 02 counts were 12/12/12/7. Stage 03 counts were 4/4/4/0, and stage 04 counts were 3/3/3/3.
Pressure warnings stopped stage 05 at 5/0/0/0. Stages 06 and 07 did not run. Stage 08 lacked a frozen estimator.
The production collector could not start through the default pressure gate. It produced no real collector measurements.
The evidence keeps admission_approved=false and full_memory_domain_established=false.
Prefill matrix requests completed, but the output consistency gate across prefill settings failed.
Those measurements do not support a recommended performance configuration.
CPU tests check tool behavior. These checks do not establish benchmark measurements or memory calibration.

## Evidence scope

Historical observations come from the pinned fork tree `d82ae3b`. They are not new measurements from this split.
Later checks must state their commit and configuration separately.

Original records remain in the fork history:

- [memory-calibration-20261009.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/memory-calibration-20261009.md).
- [memory-evidence-20261010.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/memory-evidence-20261010.md).
- [memory-evidence-plan-20261010.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/memory-evidence-plan-20261010.md).
- [metal-results-20261009.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/metal-results-20261009.md).
- [observability-load-20261009.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/observability-load-20261009.md).
- [qwen35-memory-geometry-20261010.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/qwen35-memory-geometry-20261010.md).

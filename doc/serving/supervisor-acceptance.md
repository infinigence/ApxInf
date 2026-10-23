# Supervisor acceptance

CPU fixtures exercise real worker processes, HTTP connections, signals, queues, deadlines, cancellation, pressure gates, and resource settlement.
The public deadline can expire before worker cleanup ends. Reservations remain until release or process reaping.
Worker faults make readiness return 503 while health remains available. This does not prove automatic recovery or request replay.
CPU checks do not establish model correctness or a device memory bound.

## Evidence scope

Historical observations come from the pinned fork tree `d82ae3b`. They are not new measurements from this split.
Later checks must state their commit and configuration separately.

Original records remain in the fork history:

- [diagnostics-coverage-20261010.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/diagnostics-coverage-20261010.md).
- [generation-deadline-20261009.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/generation-deadline-20261009.md).
- [host-pressure-memory-20261009.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/host-pressure-memory-20261009.md).
- [lifecycle-hardening-20261009.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/lifecycle-hardening-20261009.md).
- [review-fixes-20261010.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/review-fixes-20261010.md).
- [review-fixes-20261011.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/review-fixes-20261011.md).
- [serving-update-20261023.md](https://github.com/qhy991/ApxInf/blob/d82ae3b06631ea1285f22f7e83b03722db4f6eee/doc/serving/serving-update-20261023.md).

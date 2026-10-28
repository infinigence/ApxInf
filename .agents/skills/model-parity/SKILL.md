---
name: model-parity
description: Compare model outputs across builds or against a reference using recorded tensors and executable checks. Use for refactor parity, precision gaps, effective-shape coverage or persisted tuning regressions; single-operator tests use l3-reference-testing.
---

# Model parity

Use the checkout's `just parity` tool and
[model comparison guide](../../../doc/guides/model-parity.md). The guide defines
the capture API, contract and evidence limits. Start with the requested claim:
output comparison, shape coverage, or tuning persistence; execute only its checks.

1. Before measuring, declare cases, full output shapes/domain and byte identity
   or numerical thresholds. For a reference comparison, qualify the adapter on
   the historical matching fixture first. Preserve the original fixture and
   input-generation sequence. Record actual consumed tensors, including noise,
   at a common boundary in both implementations.
2. Capture actual grids and execution mode at the consumer, full outputs, and
   the loaded native artifact. Reuse a reviewed adapter when available; inspect
   a new adapter against the real caller. Configuration flags alone cannot
   supply runtime observations. Keep public-path acceptance in the existing
   [acceptance plan](../../../doc/guides/agent-acceptance.md).
3. Run the relevant executable checks. Numerical comparison, shape coverage,
   source binding and new tuning keys are separate results. A missing build
   receipt leaves provenance UNVERIFIED even when output bytes match. Read the
   JSON status and command exit code; a completed process is not a passed check.
4. Diagnose the first failing boundary: consumed inputs → preprocessing →
   vision → hidden/KV → per-step velocity → full actions. Capture that boundary
   on both sides without changing production arithmetic to fit the adapter.
   For precision gaps, inspect casts and measure a reference-vs-reference noise
   floor before attributing error to rounding.
5. Rerun affected checks after source or adapter edits. Report per-check results
   and unresolved evidence. The tool checks supplied observations; it does not
   certify GPU execution, Graph replay, performance or the adapter's correctness.

For an offline source audit, give the source diagnosis and concrete next
measurement. Existing timestamps, cache eligibility and object hashes can be
useful evidence without establishing which source bytes were compiled. Keep
that identity unresolved until a fresh build and loaded-artifact match exist.

# Python frontend agent rules

Three public layers: L1 native `ModelRunner` binding (from `apxinf-py`), L2
policy (`apxinf.policies`), L3 OpenPI-compatible websocket serving
(`apxinf.serving`). Pick the outermost layer that serves the task.

- `apxinf` must stay importable without a GPU and without the native
  extension: built-in tokenization and model loading import `apxinf_py`
  lazily.
- Policies return unnormalized-domain action chunks; L1 returns
  normalized-domain actions. Keep the `L2 minus unnormalize == L1`
  invariant checkable.
- New family: add `apxinf/policies/impls/<name>.py` decorated with
  `@register_policy("<name>")`, re-export in `impls/__init__.py`,
  update `AutoPolicy` docs. `AutoPolicy` picks it up.
- Tests run with `just test-py` (CPU); GPU-gated tests must skip cleanly
  without a device or checkpoint.

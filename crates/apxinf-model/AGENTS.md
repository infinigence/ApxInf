# apxinf-model agent rules

All model architecture lives here; the backend crates never reach into model
code, and model code never calls into a specific backend implementation.

- Responsibility split per family: forward computation `model/`, execution
  ownership `model_runner/`, device representations `weights/`. New VLA code
  follows the split; single-implementation families may collapse it.
- Families are self-contained: no family imports another family's private
  model, weights, runner, graph, or backbone — enforced by
  `scripts/check_model_family_boundaries.sh` (`just boundaries`).
- Adding a family: use the `model-port-workflow` skill; register in
  `registry.rs` / `builtin.rs`; read `doc/guides/adding-a-new-model.md`.
- `cuda-new` and `pi05-only` features are deprecated aliases of `cuda`;
  gate new CUDA code on `cuda` only.

# GR00T N1.7 on AMD HIP

GR00T N1.7 written directly against HIP (`apxinf_hip::native`), not through
`dyn Backend`. A self-contained port in `crates/apxinf-model/src/hip_gr00t/`:
it owns its config, checkpoint reader, device weights, vision tower, text
backbone, DiT action head, runner and loader, and imports no other model
module. Agreement with the CUDA family comes from both passing the same
accuracy fixtures, not from shared code.

Registered as `Gr00tN1d7-hip` (aliases `gr00t-hip`, `gr00t_n1_7-hip`);
`AutoModel` reaches it through `config.json:model_type` plus the `-hip`
device suffix. BF16 only — gfx1151 has no FP8 tensor cores; quantization
requests are refused at load time with the hardware reason.

## Layout

Mirrors PI0.5's organization; `scripts/check_hip_gr00t_module_boundaries.py`
enforces the direction with the same rule set as `check_pi05_module_boundaries.py`.

| Directory | Owns |
| --- | --- |
| `model/` | Forward math: `HipGr00tModel<B: Blocks>`, `ModelVariant` dispatch, the public stage methods, and `blocks/` (vision tower, Qwen3-VL text stack, VLLN + VL self-attention and the DiT action head, fused q/k/v). Reports audited `WorkspaceRequirements` without allocating the workspace. |
| `model_runner/` | Request validation, the plan cache, embodiment selection, and `prepare.rs`: allocation from the model's requirements, warm-up, execution-policy-driven graph capture and replay. |
| `weights/` | Checkpoint mapping, host tree, BF16 device tree, q/k/v packing (`APXINF_GR00T_QKV`), and `EmbodimentSource` — the seven category tensor groups that outlive upload. |
| `load.rs`, `config.rs`, `assets.rs`, `math.rs` | Registry entry and loader, config validation, asset resolution, CPU-only time-embedding/geometry math. |

VLLN and VL self-attention are action-head components: their weights were
always action-head weights, and their compute lives in
`model/blocks/action_head.rs` (the reference's `backbone_features_post_adapter`).

## Use it

```bash
# benchmark through the public runner (AutoModel path, graph replay)
cargo run -p apxinf-model --features hip --release --example hip_gr00t_bench -- \
    <checkpoint> --fixture <fixture-dir> [--stages] [--output <json>]

# stage dump for comparison against a reference
cargo run -p apxinf-model --features hip --release --example hip_gr00t_stage_dump -- \
    <checkpoint> --fixture <fixture-dir> --out <dir>

# acceptance (needs checkpoint + fixtures + reference, skips without them)
APXINF_GR00T_CHECKPOINT=<dir> APXINF_GR00T_FIXTURE=<dir> \
APXINF_GR00T_REFERENCE=<npy> \
cargo test -p apxinf-model --features hip --test hip_gr00t_acceptance -- --test-threads=1
```

Python: `AutoPolicy` → GR00T policy → `apxinf.ModelRunner` with a
`Device.Hip` device; the loader is the same registered `Gr00tN1d7-hip` entry.

## Comparing against the NVIDIA reference

The gate is `doc/gr00t-n1.7.md`'s: BF16 cosine ≥ `0.999`, relative L2 ≤
`0.05` on the final action chunk, plus the worst single action step. Measured
on the frozen LIBERO task-0 fixture (gfx1151, ROCm 7.1): final cosine
`0.9999976`, relative L2 `0.0005`, worst per-step cosine `0.99999`. Stage
comparison (see [hip-pi05.md](hip-pi05.md) for the same workflow):
vision stages and `backbone_features_raw` pass a 0.999 gate; the
post-adapter stage carries the port's characteristic BF16-vs-NVIDIA
divergence (cosine ≈ 0.997) that the end-to-end gate covers — the same
divergence the pre-restructure port showed.

Acceptance loads through `AutoModel` (detection from `config.json`, the
registered factory, the registry-opened backend), so the tested path is the
path a policy gets.

## Known limits

- AlternateVLDiT only; a checkpoint that disables it is refused at load.
- `select_layer` and DiT depth come from the checkpoint, not config defaults.
- One prompt shape per plan; tokens are refilled per request
  (`PreparedBackbone::refill_tokens`).
- Captured graphs are device-resident from public input upload to public
  output transfer; host round-trips inside a capture are unfinished work.

## Reproducing the acceptance fixtures

The fixture is an `apxinf.gr00t-n1.7.preprocessed-fixture.v1` directory: the
official processor's output tensors (BF16 `pixel_values`, `token_ids`,
`attention_mask`, `image_grid_thw`, `state`, `noise`) plus a manifest naming
the task, embodiment id, simulator seed and noise seed. Regenerate it by
running the official Isaac-GR00T processor on a LIBERO task-0 observation and
serializing its tensors with the manifest fields above (two 256×256 views,
`embodiment_id=2` for `libero_sim`).

The reference is the same fixture through upstream Isaac-GR00T on an NVIDIA
GPU (Jetson AGX Orin, sdpa attention), dumped as
`gr00t-task0-nvidia-reference.npz` plus the final action `.npy` the test
reads. The generating script, environment, revision and file hashes are
recorded in the reference directory's `PROVENANCE.md`, which travels with the
fixtures rather than the repository; regenerate with the same
`gr00t_n1d7_reference_dump.py` procedure the CUDA GR00T acceptance used.

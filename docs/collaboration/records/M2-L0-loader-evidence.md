# M2-L0 Loader Manifest and Synthetic W4 Evidence

Date: 2026-08-23
Owner: member2 / protocol-oracle
Branch: `feat/oracle-loader`
Loader commits: `8e5c1cc496c96a2423caa82a99128bb278e8aa65`,
`a5cf465c159cfc6b860c43301d38b0292896ba51`, and final hardening
`a52f7ef79546f460b17c932cc658dc01215703cb` (manifest roles, inventory bridge,
shard symlink/extra rejection, and production sharded parity).
Rollback order: revert `a52f7ef`, then `a5cf465`, then `8e5c1cc`

## Scope and local boundary

This evidence covers immutable loader metadata, header-only SafeTensors
inventory, exact W4 direction validation, and synthetic pack/dequant fixtures.
The local run did not download a checkpoint, map tensor payloads, expand a full
BF16 model, execute an 8K/16K oracle, install `transformers`, vLLM, or
`huggingface_hub`, or use a GPU. The manifest-only oracle smoke read only the
already-present `config.json` and `generation_config.json` metadata files.

No evaluator, scorer, core forward, public/hidden answer, `Cargo.toml`,
`Cargo.lock`, or `src/main.rs` change is part of the loader commit. The shared
worktree's pre-existing `crates/apxinf-loader/src/gguf.rs` modification was not
staged or committed by member2.

## Required W4 inventory

| Tensor | Required shape | Dtype | Axis/group | Result |
| --- | --- | --- | --- | --- |
| `k_proj.weight_packed` | `[1024,640]` | I32 | K packed | PASS |
| `k_proj.weight_scale` | `[1024,160]` | BF16 | K group-32 | PASS |
| `k_proj.weight_zero_point` | `[128,160]` | I32 | N group-32 | PASS |
| `down_proj.weight_packed` | `[5120,2176]` | I32 | K packed | PASS |
| `down_proj.weight_scale` | `[5120,544]` | BF16 | K group-32 | PASS |
| `down_proj.weight_zero_point` | `[640,544]` | I32 | N group-32 | PASS |

The production shapes are metadata constants in tests; no full tensor buffer is
allocated. `validate_qwen35_w4_inventory` rejects missing tensors, swapped N/K
direction, wrong shape, dtype, or group size with tensor-specific errors.

## Synthetic fixture coverage

- nibble pack/unpack: lengths 7, 8, 9, and 35;
- tail words: unused high nibbles remain zero;
- extreme values: weight nibbles 0 and 15;
- invalid values: weight nibble 16 and zero-point nibble 16 rejected;
- group boundary: K positions 31, 32, and 33 use the correct group-32 scale and
  zero-point;
- direction negative: `k_proj.weight_zero_point` changed from N to K is
  rejected;
- header-only inventory: a SafeTensors file truncated immediately after its
  JSON header still yields sorted shape/dtype metadata, proving payload bytes
  are not read;
- sharded inventory: unsafe paths, shard symlinks, missing entries, duplicate
  names, wrong shard assignment, and unindexed tensor names are rejected by
  both header-only and production sharded readers;
- header inventory bridge: `build_qwen35_w4_layer_manifest()` normalizes one
  layer prefix, attaches packed/scale/zero-point roles and N/K/group metadata,
  then runs the same frozen identity/W4 gate.

Synthetic fixture source SHA256 (`crates/apxinf-loader/src/w4.rs`):
`ff07de2815400baa9c3e6ba267de807bffb979b2ec6da608a8a6ae1a6a94b5f6`.
Historical fixture bundle identity retained from the initial evidence run:
`6a390cb8604c90b9daee0da6e3b81d67c134ea5d40e3b8b96d50bcfdf6c27992`.

## Raw local commands and results

```text
CARGO_TARGET_DIR=<fresh /tmp target> cargo test -p apxinf-loader --locked -- --nocapture
result: PASS; 26 passed, 0 failed; doc tests 0 failed

python3 -m unittest tools.oracle.test_generate_golden -v
result: PASS; 15 passed, 0 failed

python3 -m py_compile tools/oracle/generate_golden.py tools/oracle/test_generate_golden.py
result: PASS

cargo fmt --all -- --check
result: PASS

git diff --check
result: PASS at evidence preparation time
```

Observed warnings were pre-existing dead-code warnings in `apxinf-core` and
`gguf.rs`; no loader test failed. The 26 passing tests include the role/axis
bridge and production sharded-reader parity checks in addition to the original
header-only and synthetic W4 cases.

Final loader source hashes used for review:

- schema identity: `apxinf.loader-manifest.v1`, model revision
  `63768c10df38c0395e12ef49edac1bd539eaeeea`, model vocab `248320`;
- W4 fixture source (`crates/apxinf-loader/src/w4.rs`):
  `ff07de2815400baa9c3e6ba267de807bffb979b2ec6da608a8a6ae1a6a94b5f6`.

## Interface for member1

- `read_tensor_manifest(path)` returns a sorted shape/dtype inventory after
  reading only the SafeTensors header.
- `read_sharded_tensor_manifest(index_path)` enforces exact safe shard paths,
  unique tensor names, index ownership, and no unindexed extras.
- `LoaderManifest::validate()` freezes model vocab at `248320` and rejects
  invalid identity, duplicate names, zero dimensions, and zero group size.
- `validate_qwen35_w4_inventory()` first validates manifest identity and is the
  single fail-closed W4 admission point.

Member1 remains responsible for production runtime consumption, real
checkpoint execution, GPU memory measurement, and final integration. The five
reliability booleans remain global scorer eligibility gates; this local loader
evidence does not set or bypass them.

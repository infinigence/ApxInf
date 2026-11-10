# M2-L0/O0 Loader and Oracle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement an immutable loader manifest, synthetic W4 directionality fixture, and portable oracle job generator for member1's GPU1 run.

**Architecture:** Header-only loader code owns inventory validation. Oracle code writes canonical job/schema bundles locally and delegates real checkpoint execution to an explicit runner; it never creates fake golden values.

**Tech Stack:** Rust 2021, serde/serde_json, existing loader dependencies, Python 3 standard library, Cargo and unittest.

**Execution status (2026-08-23):** Tasks 1-6 were implemented in
`8e5c1cc496c96a2423caa82a99128bb278e8aa65`,
`7e26b944c06e4953480b445b3746e854aa32d527`, `a5cf465`, and final hardening
`a52f7ef79546f460b17c932cc658dc01215703cb`. Task 7 local evidence and the
GPU1 handoff are complete in the accompanying records; real checkpoint
execution, GPU UUID observation, VRAM measurement, and golden artifact hashes
remain append-only member1 server work.

---

### Task 1: Loader Manifest

**Files:** Create `crates/apxinf-loader/src/manifest.rs`; modify `crates/apxinf-loader/src/lib.rs`; test `manifest.rs`.

- [ ] Write failing tests for `ManifestDType { I32, BF16, F16, F32, Other(String) }`, `PackAxis { N, K }`, `TensorManifest { name, shape, dtype, pack_axis, group_size }`, and `LoaderManifest { schema, revision, vocab_size, tensors }`. Require nonempty revision, vocab `248320`, unique names, nonzero dimensions, and a stable JSON round trip.
- [ ] Run `cargo test -p apxinf-loader manifest::tests -- --nocapture`; expect compilation to fail because the module is missing.
- [ ] Implement derived serde types, `LoaderManifest::validate() -> Result<(), ManifestError>`, and `tensor(&self, name: &str) -> Option<&TensorManifest>`. Reject duplicate names, empty dimensions, empty revisions, and mismatched caller-provided vocab.
- [ ] Run `cargo test -p apxinf-loader manifest::tests -- --nocapture`; expect PASS.
- [ ] Commit with `git add crates/apxinf-loader/src/lib.rs crates/apxinf-loader/src/manifest.rs && git commit -m "feat(loader): add immutable checkpoint manifest"`.

### Task 2: SafeTensors Header Inventory

**Files:** Modify `crates/apxinf-loader/src/safetensors.rs` and `manifest.rs`; test `safetensors.rs`.

- [ ] Write failing tiny-header tests for `read_tensor_manifest(path)` and `read_sharded_tensor_manifest(index_path)`. Require I32/BF16 shape/dtype output without payload loading, rejection of unsafe shard paths and duplicates, and detection of missing indexed tensors.
- [ ] Run `cargo test -p apxinf-loader safetensors::tests::manifest -- --nocapture`; expect RED.
- [ ] Implement parsing that stops before `Tensor::from_raw`, maps I32/BF16/F16/F32 to `ManifestDType`, and sorts inventory by name.
- [ ] Run `cargo test -p apxinf-loader safetensors::tests -- --nocapture`; expect PASS.
- [ ] Commit with `git add crates/apxinf-loader/src/safetensors.rs crates/apxinf-loader/src/manifest.rs && git commit -m "feat(loader): parse checkpoint tensor inventory"`.

### Task 3: Required W4 Inventory

**Files:** Create `crates/apxinf-loader/src/w4.rs`; modify `lib.rs`; test `w4.rs`.

- [ ] Write failing `validate_qwen35_w4_inventory` tests for `k_proj.weight_packed [1024,640] I32 K`, `k_proj.weight_scale [1024,160] BF16 K group-32`, `k_proj.weight_zero_point [128,160] I32 N group-32`, `down_proj.weight_packed [5120,2176] I32 K`, `down_proj.weight_scale [5120,544] BF16 K group-32`, and `down_proj.weight_zero_point [640,544] I32 N group-32`. Include swapped-axis, wrong group, dtype, and one-off dimension negatives.
- [ ] Run `cargo test -p apxinf-loader w4::tests::inventory -- --nocapture`; expect RED.
- [ ] Implement exact shape/dtype/axis/group checks with errors naming the tensor and expected versus actual value. Do not infer zero-point packing from the weight packing axis.
- [ ] Run `cargo test -p apxinf-loader w4::tests::inventory -- --nocapture`; expect PASS.
- [ ] Commit with `git add crates/apxinf-loader/src/lib.rs crates/apxinf-loader/src/w4.rs && git commit -m "feat(loader): validate qwen35 W4 directions"`.

### Task 4: Synthetic W4 Reference Fixture

**Files:** Modify and test `crates/apxinf-loader/src/w4.rs`.

- [ ] Write failing tests for `pack_nibbles`, `unpack_nibbles`, and `dequantize_group`; cover nibble values 0/15, lengths 7/8/9, K positions 31/32/33, a 35-value tail, nibble 16 rejection, and a directed N/K indexing failure.
- [ ] Run `cargo test -p apxinf-loader w4::tests::synthetic -- --nocapture`; expect RED.
- [ ] Implement eight little-endian nibbles per `u32`, zero-filled final word, logical-length truncation, F32 `(value-zero_point)*scale`, and 0..15 input validation.
- [ ] Run `cargo test -p apxinf-loader w4::tests -- --nocapture`; expect PASS.
- [ ] Commit with `git add crates/apxinf-loader/src/w4.rs && git commit -m "test(loader): add synthetic W4 direction fixture"`.

### Task 5: Manifest-Only Oracle CLI

**Files:** Create `tools/oracle/README.md`, `tools/oracle/generate_golden.py`, and `tools/oracle/test_generate_golden.py`.

- [ ] Write failing unittest cases using a temporary `config.json` and `generation_config.json`. Test `build_job(model_dir, revision, layers, stages, input_manifest)` and `write_manifest_bundle(job, output_dir)` for explicit flags, sorted unique layers, stages `embedding,layer_hidden,gdn_state,kv_state,logits,tokens`, vocab 248320, EOS `[248046,248044]`, input/output token schemas, generation parameters, dtype/shape/file/SHA256 schemas.
- [ ] Run `python3 -m unittest tools.oracle.test_generate_golden -v`; expect import failure.
- [ ] Implement standard-library argparse/json/hashlib/pathlib generation. Require `--model-dir`, `--output-dir`, `--revision`, and `--layers` or `--stages`; write `input-manifest.json`, `selection.json`, `golden-schema.json`, `generation.json`, and `artifact-manifest.json` as canonical sorted JSON. Default input is `[1,2,3,4,5,6,7,8]`; write no numeric golden output without a runner.
- [ ] Run `python3 -m unittest tools.oracle.test_generate_golden -v`; expect PASS.
- [ ] Commit with `git add tools/oracle && git commit -m "feat(oracle): add portable golden job generator"`.

### Task 6: Runner Validation and Artifact Hashes

**Files:** Modify `tools/oracle/generate_golden.py` and `tools/oracle/test_generate_golden.py`.

- [ ] Write failing runner tests with a temporary executable reading `APXINF_ORACLE_JOB_MANIFEST`, writing declared artifacts, and negative cases for missing/extra file, invalid metadata, and nonzero exit.
- [ ] Run `python3 -m unittest tools.oracle.test_generate_golden.RunnerTests -v`; expect RED.
- [ ] Implement required `--runner`, repeatable `--runner-arg`, no-shell subprocess execution, environment variables `APXINF_ORACLE_JOB_MANIFEST` and `APXINF_ORACLE_OUTPUT_DIR`, declared-file-set validation, and SHA256 writes only after complete validation.
- [ ] Run `python3 -m unittest tools.oracle.test_generate_golden -v`; expect PASS.
- [ ] Commit with `git add tools/oracle/generate_golden.py tools/oracle/test_generate_golden.py && git commit -m "feat(oracle): validate runner artifacts and hashes"`.

### Task 7: Evidence and GPU1 Handoff

**Files:** Create `docs/collaboration/records/M2-O0-oracle-handoff.md` and `docs/collaboration/records/M2-L0-loader-evidence.md`.

- [ ] Populate every `oracle-handoff.md` template field with generator SHA, schema, revision, input hash, selection, exact GPU1 P0 command, artifact root, expected files, and rollback. Mark real GPU UUID, VRAM, and golden hashes as pending member1 server execution.
- [ ] Record the six inventory entries, tail/extreme/group/axis test results, local environment, commands, and no real checkpoint/BF16 expansion.
- [ ] Run `cargo fmt --all -- --check`, `cargo check --workspace --locked`, `cargo test -p apxinf-loader --locked`, `python3 -m unittest tools.oracle.test_generate_golden -v`, `python3 benchmarks/qwen38_4090/evaluation/test.py check`, `git diff --check`, and `git diff --exit-code HEAD -- benchmarks/qwen38_4090/evaluation`; expect all relevant checks PASS and no evaluator diff.
- [ ] Commit with `git add docs/collaboration/records/M2-O0-oracle-handoff.md docs/collaboration/records/M2-L0-loader-evidence.md && git commit -m "docs(oracle): add loader evidence and GPU1 handoff"`.

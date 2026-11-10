# M2-O0 Oracle Handoff Record

```text
TASK_ID: M2-O0
GENERATOR_SHA: commit a52f7ef79546f460b17c932cc658dc01215703cb; source SHA256 41254c3e9c1b284ff8e8b1bfdf9e2b0f25733c5bf95c716995a141f887cad1bb
GENERATOR_ENTRYPOINT: tools/oracle/generate_golden.py
MODEL_REPO: cyankiwi/Qwen3.8-27B-AWQ-INT4
MODEL_REVISION: 63768c10df38c0395e12ef49edac1bd539eaeeea
CONTRACT_SHA256: 520349b1279c3bf999a6848b296c23d20cdaeab7420934e9196c90018bac7433
INPUT_MANIFEST_SHA256: d9a6eb390e7d4da20d6296a27e4048071e0b48296c84d49c9dce81c1452fbcc4
LAYER_OR_STAGE_SELECTION: layers [0,3,31,32,60,63]; stages [embedding,gdn_state,kv_state,layer_hidden,logits,tokens]; max_new_tokens 128
SYNTHETIC_FIXTURE_SHA256: crates/apxinf-loader/src/w4.rs source SHA256 ff07de2815400baa9c3e6ba267de807bffb979b2ec6da608a8a6ae1a6a94b5f6
EXPECTED_ARTIFACT_SCHEMA: apxinf.oracle-golden-schema.v1 plus artifact-report generation {completion_tokens,stop_reason}; local metadata-only schema SHA256 4225aa76119262c87a75af6ac5626a21e56a572a26ed52ec4645c9e79fcae35f
APPROVED_EXPORT_METHOD: server-only; any minimal golden export requires member1 approval and project artifact channel registration
APPROVED_EXPORT_FILES: control manifests/schema/hash; real golden export list pending member1 server execution and approval
APPROVED_EXPORT_SHA256: input-manifest d9a6eb390e7d4da20d6296a27e4048071e0b48296c84d49c9dce81c1452fbcc4; selection a6ee71934a7881e8168322e496c9cb3456bb14e92032db220fc89ad61ff87dff; generation 89bae0d244a3bf4a5206ca6ba6553096f5371b96353233d43196a2834baef543; golden-schema 4225aa76119262c87a75af6ac5626a21e56a572a26ed52ec4645c9e79fcae35f; artifact-manifest 775afab258c4b77d6ee5b4e3070384b8fb8d6046e8d7eb114221fae547fcc4e7; job-manifest 6a6284230a42a299b1a4a7f9cb37aa71b0ef0425b7ee120b8e709cfc633ae9b4; artifact-identity 24dfbc8e0d741906b67c19055fd82d4e2d6229b55434bb542d18622730638018; pending real artifact hashes
SERVER_QUEUE_PRIORITY: P0-oracle
TARGET_GPU_UUID: GPU-343bc895-b011-22fa-4449-97207aa2bdec
REPLAY_COMMAND: exact command block below; checkpoint runner executable is a member1 integration prerequisite and the command fails closed if absent
EXPECTED_ARTIFACT_DIR: /mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/a52f7ef79546f460b17c932cc658dc01215703cb
EXPECTED_MAX_VRAM_MIB: pending member1 GPU1 measurement; no local estimate is asserted as evidence
STOP_CONDITION: any lock/UUID/revision/config/EOS mismatch, runner absence/nonzero exit, missing/extra/symlink artifact, control-manifest mutation, schema/dtype/shape/hash/token failure, OOM, Xid, NaN, or fallback
ROLLBACK_OR_DISABLE_COMMAND: omit --runner to remain manifest-only; discard the incomplete artifact directory; code rollback order is a52f7ef, ae2fb4e, a5cf465, 7e26b94, then 8e5c1cc
```

## Exact GPU1 replay command

Member1 first installs or integrates the checkpoint-specific executable at the
path below. M2-O0 intentionally does not provide that real model runner.

```bash
cd /mnt/chuangxin/team2/ApxInf
exec 9>/tmp/apxinf-gpu-job.lock
flock -n 9 || { echo "another ApxInf GPU job is running" >&2; exit 2; }
export CUDA_VISIBLE_DEVICES=GPU-343bc895-b011-22fa-4449-97207aa2bdec
export APXINF_GPU_LABEL=GPU1
test -x /mnt/chuangxin/team2/ApxInf/tools/oracle/qwen35_checkpoint_runner || {
  echo "member1 checkpoint runner is not installed" >&2
  exit 3
}
python3 tools/oracle/generate_golden.py \
  --model-dir /mnt/chuangxin/team2/models/Qwen3.8-27B-AWQ-INT4 \
  --output-dir /mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/a52f7ef79546f460b17c932cc658dc01215703cb \
  --revision 63768c10df38c0395e12ef49edac1bd539eaeeea \
  --input-manifest tools/oracle/manifests/m2-o0-short.json \
  --layers 0,3,31,32,60,63 \
  --stages embedding,layer_hidden,gdn_state,kv_state,logits,tokens \
  --max-new-tokens 128 \
  --runner /mnt/chuangxin/team2/ApxInf/tools/oracle/qwen35_checkpoint_runner
```

Layer roles in this selection are early/middle/late representatives:

- GDN / linear attention: 0, 32, 60;
- full attention: 3, 31, 63.

The generator uses the checkpoint's 64-entry `layer_types` map and will emit
GDN state only for the first set and KV state only for the second set.

## Expected declared artifact files

```text
embedding.f32.bin
layer-000-gdn-state.f32.bin
layer-000-hidden.f32.bin
layer-003-hidden.f32.bin
layer-003-kv-key.f32.bin
layer-003-kv-value.f32.bin
layer-031-hidden.f32.bin
layer-031-kv-key.f32.bin
layer-031-kv-value.f32.bin
layer-032-gdn-state.f32.bin
layer-032-hidden.f32.bin
layer-060-gdn-state.f32.bin
layer-060-hidden.f32.bin
layer-063-hidden.f32.bin
layer-063-kv-key.f32.bin
layer-063-kv-value.f32.bin
logits.f32.bin
output-tokens.json
artifact-report.json
```

The metadata-only local bundle had artifact identity
`24dfbc8e0d741906b67c19055fd82d4e2d6229b55434bb542d18622730638018`,
config SHA256
`fece2915d4c8ad4c10877622f04ea5e01cd3ae38768ce5c1edb700dd1de290f6`,
and generation-config SHA256
`e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e`.
The control-file SHA256 values are recorded in `APPROVED_EXPORT_SHA256` above;
they are metadata-only identities, not real checkpoint artifact results.
Member1 must compare these identities before running. A mismatch stops the job;
it is not silently accepted as a new oracle.

## Member1 append-only server results

```text
QUEUE_ID: P0-oracle-20260823T095000Z
STARTED_AT_UTC: 2026-08-23T09:42:28Z
ENDED_AT_UTC: 2026-08-23T09:46:34Z
RUNNER_COMMIT_SHA: 46182a1167570e7595b3e658b02fb8acadac9f7a
ACTUAL_GPU_UUID: GPU-343bc895-b011-22fa-4449-97207aa2bdec
DRIVER_CUDA: NVIDIA driver 580.82.07; CUDA toolkit 12.8
PEAK_VRAM_MIB: 20942 (two-second nvidia-smi sampling)
COMMAND_OUTPUT_PATH: /mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/46182a1167570e7595b3e658b02fb8acadac9f7a/P0-oracle-20260823T095000Z.command.log
RAW_ARTIFACT_FILE_LIST: artifact-manifest.json status complete; 18 declared payload files plus artifact-report.json under artifacts/
RAW_ARTIFACT_SHA256: artifact-manifest bbdf28c9fffe2e89fbf83a6a8d06aafd71d2a292a64a7444f07d3cf234c6cb75; artifact-report e8d0e98f7d80edd089262c387e89e98cc8e93e9a4a365c0baab83712708a9f4d; full list in manifest.sha256
OUTPUT: 128 completion tokens; stop_reason budget; model token range validation passed
APPROVED_MINIMAL_EXPORT: none exported; raw golden remains server-only pending an explicit remote-consumption request
INCIDENT_OR_SUCCESS_STATUS: success; generator exit 0, GPU1 returned to 1 MiB, no compute process remained
ROLLBACK_POINT: 097fe7252becef338da25585b2be571e98f6e8d9; omit --runner for manifest-only mode
```

Server artifact directory:

```text
/mnt/chuangxin/team2/artifacts/apxinf/oracle/63768c10df38c0395e12ef49edac1bd539eaeeea/46182a1167570e7595b3e658b02fb8acadac9f7a/
```

The original full-model decompression failure is preserved at
`097fe7252becef338da25585b2be571e98f6e8d9.incident-first-forward-full-decompress-20260823T091804Z/`.
Later runner/schema failures are preserved under their own commit-qualified incident directories and were not reused as golden evidence.

No local result in this record is a real checkpoint correctness, GPU memory,
hidden-case, reliability, or eligibility claim. The five reliability booleans
remain global scorer gates and are not modified by this handoff.

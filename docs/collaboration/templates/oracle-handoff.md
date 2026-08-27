# Oracle Handoff Record

Use this record when member2 hands an oracle generator to member1 for the one-time server
execution. The generator and this record may be committed; model weights, complete BF16
expansions, hidden cases and raw private answers must remain outside Git.

```text
TASK_ID: M2-O0
GENERATOR_SHA:
GENERATOR_ENTRYPOINT: tools/oracle/generate_golden.py
MODEL_REPO:
MODEL_REVISION:
CONTRACT_SHA256:
INPUT_MANIFEST_SHA256:
LAYER_OR_STAGE_SELECTION:
SYNTHETIC_FIXTURE_SHA256:
EXPECTED_ARTIFACT_SCHEMA:
APPROVED_EXPORT_METHOD: server-only | project-artifact-channel | other-approved-channel
APPROVED_EXPORT_FILES:
APPROVED_EXPORT_SHA256:
SERVER_QUEUE_PRIORITY: P0-oracle
TARGET_GPU_UUID: GPU-343bc895-b011-22fa-4449-97207aa2bdec
REPLAY_COMMAND:
EXPECTED_ARTIFACT_DIR:
EXPECTED_MAX_VRAM_MIB:
STOP_CONDITION:
ROLLBACK_OR_DISABLE_COMMAND:
```

Member1 appends the actual queue id, start/end timestamps, command output, peak VRAM, artifact
file list and SHA256 after the server run. A failed or partial oracle run is recorded as an
incident; it is not silently replaced by a local Qwen3Next result.

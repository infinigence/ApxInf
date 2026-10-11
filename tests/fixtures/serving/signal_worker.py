"""Original CPU process for CLI signal and loading cancellation checks."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument("--worker-epoch")
parser.add_argument("--model")
parser.add_argument("--max-context", type=int)
parser.add_argument("--max-output-tokens", type=int)
parser.add_argument("--prefill-step-size", type=int)
parser.add_argument("--output-batch-tokens", type=int)
parser.add_argument("--memory-limit-bytes", type=int)
args, _ = parser.parse_known_args()
model = Path(args.model)
root = model.parent
(root / "worker.pid").write_text(str(os.getpid()))
mode = (root / "mode").read_text()
if mode == "loading":
    time.sleep(60)
    sys.exit(3)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(domain, value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(domain.encode() + b"\0" + data.encode()).hexdigest()


capabilities = {
    "text": True, "image": False, "greedy": True, "sampling": False,
    "token_events": True, "cancel_granularity": "token",
    "max_active_sequences": 1, "max_context": args.max_context,
    "max_output_tokens": args.max_output_tokens,
    "exact_append": False, "state_schema": "none",
}
manifest = {
    "artifacts": [],
    "adapter_sha256": digest(__file__),
    "python_sha256": digest(sys.executable),
    "contracts_sha256": digest(Path(__file__).with_name("contracts.py")),
    "runtime_pins_sha256": digest(root / "scripts/apxinf_mlx_generate.py"),
    "execution": {
        "provider": "mlx-lm", "precision": "bundle",
        "prefill_step_size": args.prefill_step_size,
        "output_batch_tokens": args.output_batch_tokens,
        "memory_limit_bytes": args.memory_limit_bytes,
    },
}
ready = {
    "protocol": "apxinf-worker/2.0", "kind": "ready",
    "worker_epoch": args.worker_epoch,
    "model_revision": identity("apxinf-model-identity-v1", manifest),
    "capability_revision": identity("apxinf-capability-v1", capabilities),
    "model_manifest": manifest, "capabilities": capabilities,
    "limits": {"max_frame_bytes": 1048576, "max_depth": 16, "max_commands": 10000},
    "model_path": str(model), "eos_token_ids": [2], "vocab_size": 8,
    "runtime": {"python": "cpu-fixture", "mlx": "absent", "mlx_lm": "absent"},
    "memory": {"active_bytes": 0, "peak_bytes": 0, "cache_bytes": 0},
}
if mode == "invalid_ready":
    ready["kind"] = "invalid_ready"
print(json.dumps(ready), flush=True)
for line in sys.stdin:
    command = json.loads(line)
    if command["kind"] != "prepare_input":
        raise RuntimeError("The fixture only supports token counting.")
    (root / "preparing").write_text(command["command_id"])
    while not (root / "release").exists():
        time.sleep(0.01)
    tokens = [3, 4]
    data = b"apxinf-token-prefix-v1\0" + len(tokens).to_bytes(8, "little")
    data += b"".join(token.to_bytes(4, "little") for token in tokens)
    print(json.dumps({
        "protocol": "apxinf-worker/2.0", "kind": "prepared_input",
        "worker_epoch": args.worker_epoch, "command_id": command["command_id"],
        "model_revision": ready["model_revision"], "token_ids": tokens,
        "effective_input_tokens": len(tokens),
        "prompt_digest": hashlib.sha256(data).hexdigest(),
    }), flush=True)

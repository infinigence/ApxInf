#!/usr/bin/env python3
"""Generate a trajectory reference for the public token-trajectory suite using
the torch CPU reference model. Output matches run_evaluation.py's
trajectory-reference schema so `test.py run` can consume it via
run_evaluation.py --trajectory-reference.

Usage: qwen38_trajectory_reference.py <cases.jsonl> <out.json>
"""
import hashlib, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen38_torch_reference import load_patched

TRAJECTORY_SCHEMA = "apxinf.qwen38_27b.trajectory_reference.v1"
MODEL_DIR = "/mnt/chuangxin/team3/work/model/qwen"
REPO = "cyankiwi/Qwen3.8-27B-AWQ-INT4"
REVISION = "63768c10df38c0395e12ef49edac1bd539eaeeea"

def canonical_json(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")

def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def main():
    cases_file, out_file = sys.argv[1], sys.argv[2]
    cases = [json.loads(line) for line in open(cases_file)]
    model = load_patched()
    out = {"schema": TRAJECTORY_SCHEMA, "model_repo_id": REPO,
           "model_revision": REVISION,
           "tokenizer_config_sha256": sha256_bytes(
               open(os.path.join(MODEL_DIR, "tokenizer_config.json"), "rb").read()),
           "cases": {}}
    for case in cases:
        if case["id"] not in ("text-perf-1024", "text-perf-8192"):
            continue
        ids = case["input_ids"]
        with torch.no_grad():
            gen = model.generate(torch.tensor([ids], dtype=torch.long),
                                 max_new_tokens=128, do_sample=False,
                                 pad_token_id=None, eos_token_id=None)
        output = gen[0].tolist()[len(ids):]
        assert len(output) == 128, f"{case['id']}: expected 128 tokens, got {len(output)}"
        out["cases"][case["id"]] = {
            "input_ids_sha256": sha256_bytes(canonical_json(ids)),
            "output_ids": output,
            "output_ids_sha256": sha256_bytes(canonical_json(output)),
        }
        json.dump(out, open(out_file, "w"), indent=2, sort_keys=True)
        print(f"{case['id']}: 128 tokens done", flush=True)
    print("wrote", out_file)

if __name__ == "__main__":
    main()

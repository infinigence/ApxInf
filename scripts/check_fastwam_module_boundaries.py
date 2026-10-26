#!/usr/bin/env python3
"""Check Fast-WAM model/blocks, checkpoint and execution ownership.

This is a structural guard, not a substitute for reviewing numerical semantics.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / "crates/apxinf-model/src/fastwam"


def source(path):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)


def check(root):
    errors = []
    execution_types = r"\b(model_runner|ModelRunner|PreparedInference|ExecutionSession|PreparedPhase|ExecutionPolicy|VlaRequest|InferenceSpec)\b"
    for name in ("model/mod.rs", "model/blocks/bf16/mod.rs"):
        if not (root / name).is_file():
            errors.append(f"{name}: missing model computation owner")
    # All new computation files must live under model, not escape a fixed list.
    allowed_root = {"mod.rs", "backend.rs", "config.rs", "load.rs", "model_runner.rs", "weights.rs"}
    for path in root.glob("*.rs"):
        if path.name not in allowed_root:
            errors.append(f"{path.name}: computation belongs under model/blocks/bf16")
    for path in (root / "model").rglob("*.rs"):
        text = source(path).split("#[cfg(test)]")[0]
        if re.search(execution_types, text):
            errors.append(f"{path.relative_to(root)}: computation depends on execution ownership")
    for path in [root / "weights.rs", *sorted((root / "weights").glob("*.rs"))]:
        text = source(path)
        if re.search(execution_types, text) or re.search(r"\b(model|blocks)\s*::", text):
            errors.append(f"{path.relative_to(root)}: weights depend on computation or runner")
        # Inherent methods implemented elsewhere evade import-only checks.
        if re.search(r"\.\s*(forward|forward_conv|context)\s*\(", text):
            errors.append(f"{path.relative_to(root)}: weights invoke model computation")
    backend = source(root / "backend.rs")
    # Keep this seam deliberately small: imports and one host upload adapter.
    if re.search(r"\bops\s*::|\ballocate_output\s*\(|\bimpl\b", backend):
        errors.append("backend.rs: operator composition belongs in model/blocks/bf16")
    if re.findall(r"\bfn\s+(\w+)", backend) != ["upload"]:
        errors.append("backend.rs: only the upload transfer adapter may define a function")
    load = source(root / "load.rs")
    if re.search(r"\bblocks\s*::", load):
        errors.append("load.rs: concrete block construction belongs in model")
    for path in root.rglob("*.rs"):
        text = source(path)
        if re.search(r"\b(unsafe|apxinf_cuda)\b|::ffi\b", text):
            errors.append(f"{path.relative_to(root)}: family must use the safe cuda-new seam")
    return errors


def main():
    errors = check(ROOT)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Fast-WAM module dependency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

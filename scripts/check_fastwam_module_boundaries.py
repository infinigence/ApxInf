#!/usr/bin/env python3
"""Check Fast-WAM model/blocks, checkpoint and execution ownership.

This is a structural guard, not a substitute for reviewing numerical semantics.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / "crates/apxinf-model/src/cuda_fastwam"


def source(path):
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)


def production_source(path):
    text = source(path)
    # Inline test modules may occur before more production code in bf16.rs.
    while match := re.search(r"#\[cfg\(test\)\]\s*mod\s+\w+\s*\{", text):
        depth = 1
        end = match.end()
        while depth and end < len(text):
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        text = text[:match.start()] + text[end:]
    return text


def check(root):
    errors = []
    execution_types = r"\b(model_runner|FastWamModelRunner|ModelRunner|PreparedInference|ExecutionSession|PreparedPhase|ExecutionPolicy|VlaRequest|InferenceSpec)\b"
    for name in ("model/mod.rs", "model/blocks/bf16.rs", "weights/host.rs", "weights/bf16.rs", "model_runner/runner.rs", "model_runner/prepare.rs"):
        if not (root / name).is_file():
            errors.append(f"{name}: missing model computation owner")
    for directory in ("weights/bf16", "model/blocks/bf16"):
        if (root / directory).exists():
            errors.append(f"{directory}: BF16 implementation stays in one file")
    # All new computation files must live under model, not escape a fixed list.
    allowed_root = {"mod.rs", "backend.rs", "config.rs", "load.rs"}
    for path in root.glob("*.rs"):
        if path.name not in allowed_root:
            errors.append(f"{path.name}: computation belongs under model/blocks/bf16.rs")
    for path in (root / "model").rglob("*.rs"):
        text = production_source(path)
        if re.search(execution_types, text):
            errors.append(f"{path.relative_to(root)}: computation depends on execution ownership")
    model = production_source(root / "model/mod.rs")
    if re.search(r"\b(Expert|TextEncoder|Vae|DType|Shape)\b|\ballocate_output\s*\(|\bops::[a-z]\w*\s*\(", model):
        errors.append("model/mod.rs: precision and subnetwork details belong in Bf16Blocks")
    weight_types = set()
    for path in (root / "weights").rglob("*.rs"):
        weight_types.update(re.findall(r"\bstruct\s+(\w+Weights)\b", source(path)))
    for path in (root / "model").rglob("*.rs"):
        text = production_source(path)
        if re.search(r"\bimpl\s+(?:" + "|".join(sorted(weight_types)) + r")\b", text):
            errors.append(f"{path.relative_to(root)}: computation adds inherent methods to weight data")
        if re.search(r"\bStore\b|\b(?:Bf16Weights|\w+Weights)::load\s*\(", text):
            errors.append(f"{path.relative_to(root)}: checkpoint loading belongs in weights and load")
        if re.search(r"weights::(?:cast|cast_weight)\b", text):
            errors.append(f"{path.relative_to(root)}: activation conversion belongs in Blocks")
    for path in sorted((root / "weights").rglob("*.rs")):
        text = source(path)
        if re.search(execution_types, text) or re.search(r"\b(model|blocks)\s*::", text):
            errors.append(f"{path.relative_to(root)}: weights depend on computation or runner")
        if path.name != "host.rs" and re.search(r"\bimpl\s+Store\b", text):
            errors.append(f"{path.relative_to(root)}: device preparation must consume the host store explicitly")
        if re.search(r"\bfn\s+(?:forward|forward_conv|context|cast)\s*\(", text):
            errors.append(f"{path.relative_to(root)}: activation computation belongs in Blocks")
        # Inherent methods implemented elsewhere evade import-only checks.
        if re.search(r"\.\s*(forward|forward_conv|context)\s*\(", text):
            errors.append(f"{path.relative_to(root)}: weights invoke model computation")
    host = source(root / "weights/host.rs")
    if re.search(r"\b(backend|ops|Context|CudaBuffer)\b", host):
        errors.append("weights/host.rs: host checkpoint storage depends on CUDA execution")
    backend = source(root / "backend.rs")
    # Keep this seam deliberately small: imports and one host upload adapter.
    if re.search(r"\bops\s*::|\ballocate_output\s*\(|\bimpl\b", backend):
        errors.append("backend.rs: operator composition belongs in model/blocks/bf16.rs")
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

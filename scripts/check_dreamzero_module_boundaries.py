#!/usr/bin/env python3
"""Enforce DreamZero ownership; numerical and device checks remain independent."""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / "crates/apxinf-model/src/cuda_dreamzero"


def source(path):
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
    while True:
        match = re.search(r"#\[cfg\(test\)\]\s*mod\s+\w+\s*\{", text)
        if match is None:
            break
        end, depth = match.end(), 1
        while depth and end < len(text):
            depth += (text[end] == "{") - (text[end] == "}")
            end += 1
        text = text[:match.start()] + text[end:]
    return text


def check(root):
    errors = []
    required = {
        "mod.rs", "backend.rs", "config.rs", "schedule.rs", "load.rs",
        "model/mod.rs", "model/blocks/mod.rs", "model/blocks/bf16.rs",
        "model_runner/mod.rs", "model_runner/runner.rs", "model_runner/prepare.rs",
        "model_runner/acceptance.rs", "weights/mod.rs", "weights/host.rs", "weights/bf16.rs",
    }
    actual = {str(p.relative_to(root)) for p in root.rglob("*.rs")}
    if actual != required:
        return [f"family layout: missing={sorted(required-actual)}, unexpected={sorted(actual-required)}"]
    execution = r"\b(VlaRequest|VlaRuntime|InferenceSpec|ExecutionPolicy|PreparationStatus|PreparedInference|CapturedGraph|ExecutionSession|model_runner)\b"
    for path in root.rglob("*.rs"):
        name, text = str(path.relative_to(root)), source(path)
        if name.startswith(("model/", "weights/")) and re.search(execution, text):
            errors.append(f"{name}: computation/weights depend on execution ownership")
        if name.startswith("weights/") and re.search(r"\b(model|blocks|model_runner)\s*::|\b(DreamZeroModel|DreamZeroModelRunner)\b", text):
            errors.append(f"{name}: weights depend on computation/runner")
        if name.startswith("model_runner/") and name != "model_runner/acceptance.rs" and re.search(r"\b(weights|blocks|dit|dit_block|clip|vae|unipc)\s*::", text):
            errors.append(f"{name}: runner must use the model interface")
        if path.name != "backend.rs" and re.search(r"\bapxinf_cuda(?:_new)?\s*::", text):
            errors.append(f"{name}: import safe CUDA through backend.rs")
        if re.search(r'\bunsafe\b|\bextern\s+"C"', text):
            errors.append(f"{name}: device FFI belongs in the backend")
    if re.search(r"\b(fn|impl|struct|enum)\b", source(root / "backend.rs")):
        errors.append("backend.rs: only safe API reexports belong here")
    if re.search(r"\b(backend|Context|CudaBuffer|upload|transfers)\b", source(root / "weights/host.rs")):
        errors.append("weights/host.rs: host validation cannot own CUDA representation")
    if re.search(r"\b(DType|Shape|DitWorkspace|VisionWorkspace|JointWorkspace)\b|\bops\s*::", source(root / "model/mod.rs")):
        errors.append("model/mod.rs: physical representations/operators belong in Bf16Blocks")
    if re.search(r"DreamZeroModelRunner\s*\{", source(root / "load.rs")):
        errors.append("load.rs: construct runner through its constructor")
    return errors


if __name__ == "__main__":
    errors = check(ROOT)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        sys.exit(1)
    print("DreamZero module dependency checks passed")

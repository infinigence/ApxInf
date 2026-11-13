#!/usr/bin/env python3
"""Check the GR00T HIP port's internal dependency direction and isolation.

Rust privacy already enforces field access. This guards what it cannot, with
the PI0.5 rule set adapted to this family's directories:

1. **Directory rules** (as `check_pi05_module_boundaries.py`):
   - `model/` owns forward math only: no runner, request or execution-policy
     types, and no HIP graph machinery.
   - `weights/` maps checkpoints and uploads tensors: it must not name model
     or runner types.
   - `model_runner/` validates requests and executes plans: it must not name
     the concrete blocks, match on the variant enum, or dig into `weights::`
     internals -- the family root re-exports what it needs.
   - `load.rs` builds the runner through its constructor, never its fields.
2. **Extra strictness this port keeps on top**: the host checkpoint reader and
   the pure-CPU modules stay backend-free, and model code reaches HIP through
   `apxinf_hip::native` only -- raw FFI is forbidden.
3. **Family isolation.** The port is a private module: no cross-family
   imports, no root wildcard.
4. **The activation-variant audit.** GR00T uses two GELU variants at four
   positions, three of them config-driven. Every hardcoded variant must stay
   covered by `validate_activations`, because a mismatch is a silent ~3e-4
   drift that an end-to-end cosine gate does not catch.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / "crates/apxinf-model/src/hip_gr00t"

# The PI0.5 rule set, adapted to this family's names.
RULES = {
    "model": r"\b(model_runner|HipGr00tRunner|HipGr00tPrepared|ExecutionPolicy|VlaRequest|InferenceSpec|HipGraph)\b|apxinf_hip::graph\b",
    "weights": r"\b(HipGr00tModel|ModelVariant|HipGr00tRunner|HipGr00tPrepared)\b|\b(model_runner|model)\s*::|::(?:model_runner|model)\b",
    "model_runner": r"\bBf16Blocks\b|ModelVariant\s*::|\bweights\s*::",
}

# Modules that must stay free of device types: the host checkpoint reader and
# the pure-CPU geometry / math helpers (testable without a GPU).
BACKEND_FREE = {
    "weights/host.rs": r"\b(HipContext|HipArena|native::)\b",
    "math.rs": r"\b(HipContext|HipArena|native::)\b",
    "model/token_groups.rs": r"\b(HipContext|HipArena|native::)\b",
    "model/blocks/dit_source.rs": r"\b(HipContext|HipArena|native::)\b",
}

# Every file: the port is a private module and owns its own copies.
FORBIDDEN_EVERYWHERE = {
    r"\bcrate::gr00t\b": "imports the CUDA GR00T family; this port copies, never shares",
    r"\bcrate::(pi05|hip_pi05|qwen3vl|qwen_drive|pi0fast|smolvla|d3|walloss|llama)\b": "imports another model family",
    r"\bapxinf_hip::ffi\b|\bextern\s+\"C\"": "raw FFI in model code; go through apxinf_hip::native",
    r"\buse\s+crate::hip_gr00t::\*": "root wildcard hides dependencies",
}


def source(path):
    # Ignore prose; this guard checks explicit Rust dependencies, not an AST.
    return re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)


def main():
    violations = []
    paths = sorted(ROOT.rglob("*.rs"))
    if not paths:
        print(f"{ROOT} has no Rust sources", file=sys.stderr)
        return 1

    for path in paths:
        code = source(path)
        relative = str(path.relative_to(ROOT))
        for pattern, reason in FORBIDDEN_EVERYWHERE.items():
            if re.search(pattern, code):
                violations.append(f"{relative}: {reason}")
        for module, forbidden in {**RULES, **BACKEND_FREE}.items():
            if relative == module or relative.startswith(module + "/"):
                if hit := re.search(forbidden, code):
                    violations.append(
                        f"{relative}: forbidden dependency on {hit.group(0)} "
                        f"inverts the {module} module's direction"
                    )

    for module in {**RULES, **BACKEND_FREE}:
        if not (ROOT / module).is_file() and not (ROOT / module).is_dir():
            violations.append(f"{module}: expected module source is missing")

    # load.rs builds the runner through its constructor, never its fields.
    load = source(ROOT / "load.rs")
    if re.search(r"HipGr00tRunner\s*\{|\.prepared\b", load):
        violations.append(
            "load.rs: construct the runner through its interface, not its fields"
        )

    # Every hardcoded activation must be covered by the load-time audit.
    variants = set()
    for path in paths:
        variants.update(re.findall(r"Activation::(Gelu\w+)", source(path)))
    if variants and not variants <= {"GeluTanh", "GeluErf"}:
        violations.append(f"unaudited GELU variants in use: {sorted(variants)}")
    # Whether the guard actually *guards* is behaviour, not structure: a
    # regex cannot tell reading a config field from comparing it, nor a
    # function definition from a call to it. Attempting that here produced a
    # check that passed on a gutted guard, so it is a unit test's job --
    # `weights::host`'s checkpoint tests drive `from_map` and assert the
    # rejection. What is checked here is only that the function still exists
    # to be tested.
    guard = source(ROOT / "weights/host.rs")
    if not re.search(r"fn validate_activations\b", guard):
        violations.append(
            "weights/host.rs: validate_activations is gone; a config-driven activation "
            "would then be computed as whatever the code hardcodes"
        )

    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("GR00T HIP port module dependency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

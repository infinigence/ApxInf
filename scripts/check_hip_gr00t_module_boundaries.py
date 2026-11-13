#!/usr/bin/env python3
"""Check the GR00T HIP port's internal dependency direction and isolation.

Rust privacy already enforces field access. This guards the three things it
cannot:

1. **Dependency direction.** `weights` must not reach back into the forward
   path, and the forward modules must not depend on the runner's request types.
   The architecture doc's rule is that a runner never re-derives model math and
   weights never depend on model/runner types.
2. **Backend boundary.** Model code reaches HIP through `apxinf_hip::native`;
   raw FFI in model code is forbidden.
3. **The activation-variant audit.** GR00T uses two GELU variants at four
   positions, three of them config-driven. Every hardcoded variant must stay
   covered by `checkpoint::validate_activations`, because a mismatch is a
   silent ~3e-4 drift that an end-to-end cosine gate does not catch.
"""
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / "crates/apxinf-model/src/hip_gr00t"

# Per module, the dependencies that would invert the intended direction.
RULES = {
    "weights/bf16.rs": r"\b(HipGr00tRunner|HipGr00tPrepared|HipGr00tModel|VlaRequest|InferenceSpec|PreparedVision|PreparedBackbone|DitConditioning)\b",
    "weights/host.rs": r"\b(HipGr00tRunner|HipGr00tPrepared|HipGr00tModel|VlaRequest|InferenceSpec|PreparedVision|PreparedBackbone|DitConditioning|HipContext|HipArena|native::)\b",
    "model/blocks/vision.rs": r"\b(HipGr00tRunner|HipGr00tPrepared|HipGr00tModel|VlaRequest|InferenceSpec)\b",
    "model/blocks/backbone.rs": r"\b(HipGr00tRunner|HipGr00tPrepared|HipGr00tModel|VlaRequest|InferenceSpec)\b",
    "model/blocks/action_head.rs": r"\b(HipGr00tRunner|HipGr00tPrepared|HipGr00tModel|VlaRequest|InferenceSpec)\b",
    "math.rs": r"\b(HipContext|HipArena|native::)\b",
    "model/blocks/dit_source.rs": r"\b(HipContext|HipArena|native::)\b",
    "model/token_groups.rs": r"\b(HipContext|HipArena|native::)\b",
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
        relative = path.relative_to(ROOT)
        for pattern, reason in FORBIDDEN_EVERYWHERE.items():
            if re.search(pattern, code):
                violations.append(f"{relative}: {reason}")
        forbidden = RULES.get(str(relative))
        if forbidden and (hit := re.search(forbidden, code)):
            violations.append(
                f"{relative}: forbidden dependency on {hit.group(0)} inverts the module direction"
            )

    for module in RULES:
        if not (ROOT / module).is_file():
            violations.append(f"{module}: expected module source is missing")

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
    # `checkpoint::tests::a_checkpoint_asking_for_a_different_activation_is_refused`
    # drives `from_map` and asserts the rejection. What is checked here is only
    # that the function still exists to be tested.
    guard = source(ROOT / "weights/host.rs")
    if not re.search(r"fn validate_activations\b", guard):
        violations.append(
            "checkpoint/mod.rs: validate_activations is gone; a config-driven activation "
            "would then be computed as whatever the code hardcodes"
        )

    if violations:
        print("\n".join(violations), file=sys.stderr)
        return 1
    print("GR00T HIP port module dependency checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The structural guard must reject the ownership mistakes found in PR #149."""
import importlib.util
from pathlib import Path
import shutil

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "fastwam_boundaries", REPO / "scripts/check_fastwam_module_boundaries.py"
)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_current_fastwam_ownership():
    assert guard.check(guard.ROOT) == []


@pytest.mark.parametrize(
    "path, code, expected",
    [
        ("backend.rs", "fn projection() { ops::gemm(); }", "operator composition"),
        ("weights/dit.rs", "fn prepare() { time.forward(); }", "weights invoke"),
        ("text.rs", "fn forward() {}", "belongs under model/blocks/bf16"),
        ("model/blocks/bf16/extra.rs", "use crate::vla::VlaRequest;", "execution ownership"),
        ("load.rs", "use super::model::blocks::bf16;", "concrete block construction"),
    ],
)
def test_guard_rejects_ownership_regressions(tmp_path, path, code, expected):
    root = tmp_path / "fastwam"
    shutil.copytree(guard.ROOT, root)
    with (root / path).open("a") as output:
        output.write("\n" + code + "\n")
    assert any(expected in error for error in guard.check(root))

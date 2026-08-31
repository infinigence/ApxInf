"""Startup preflight: does the checkpoint on disk match the preset serving it?

The delivered configuration that motivated :mod:`apxinf.robots.preflight` was a
Unitree G1 checkpoint (16-DoF, three cameras, delta joint actions) served with a
LIBERO ``norm_stats.json`` (8-dim state, 7-dim action) and ``--action-dim 7``.
Every layer accepted it. These tests pin the checks that refuse it, and — just as
importantly — pin that a *correct* checkpoint produces no fatal finding, so the
preflight cannot become a thing operators route around with
``--skip-preflight``.

Runs offline against synthesised checkpoint directories; no CUDA, no weights.
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "python" / "apxinf"))

from apxinf.robots.preflight import (  # noqa: E402
    FAIL,
    INFO,
    WARN,
    check_checkpoint,
    format_findings,
)
from apxinf.robots.presets import get_robot_preset  # noqa: E402

G1_DIM = 16


def _stats(width: int, *, quantiles: bool = True) -> dict:
    entry = {"mean": [0.0] * width, "std": [1.0] * width}
    if quantiles:
        entry["q01"] = [-1.0] * width
        entry["q99"] = [1.0] * width
    return entry


class CheckpointFixture:
    """A checkpoint directory with exactly the files the preflight reads."""

    def __init__(self, case: unittest.TestCase, **norm_stats) -> None:
        self.path = pathlib.Path(tempfile.mkdtemp())
        case.addCleanup(self._cleanup)
        if norm_stats:
            self.write_norm_stats(**norm_stats)

    def _cleanup(self) -> None:
        import shutil

        shutil.rmtree(self.path, ignore_errors=True)

    def write_norm_stats(self, *, nested: bool = True, **entries) -> "CheckpointFixture":
        document = {"norm_stats": entries} if nested else entries
        (self.path / "norm_stats.json").write_text(json.dumps(document))
        return self

    def write_tokenizer(self, name: str = "paligemma_tokenizer.model") -> "CheckpointFixture":
        (self.path / name).write_bytes(b"not a real sentencepiece model")
        return self


def _levels(findings, check_prefix: str) -> list:
    return [f.level for f in findings if f.check.startswith(check_prefix)]


class NormStatsWidthTest(unittest.TestCase):
    """A1: the shipped statistics have to be *this robot's* statistics."""

    def test_libero_stats_under_the_g1_preset_are_fatal(self) -> None:
        """The exact delivered combination: G1 preset, LIBERO norm_stats."""
        ckpt = CheckpointFixture(
            self, actions=_stats(7), state=_stats(8)
        ).write_tokenizer()

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        fatal = [f for f in findings if f.level == FAIL]
        self.assertEqual(len(fatal), 2, format_findings(findings))
        widths = {f.check for f in fatal}
        self.assertEqual(
            widths, {"norm_stats['actions'] width", "norm_stats['state'] width"}
        )
        # The remedy has to name the number the operator should be looking for;
        # "width mismatch" alone sends them to the wrong file.
        self.assertIn("16", fatal[0].detail + fatal[0].remedy)

    def test_matching_widths_pass(self) -> None:
        ckpt = CheckpointFixture(
            self, actions=_stats(G1_DIM), state=_stats(G1_DIM)
        ).write_tokenizer()

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        self.assertEqual([f for f in findings if f.level == FAIL], [], format_findings(findings))
        self.assertEqual([f for f in findings if f.level == WARN], [])

    def test_libero_preset_checks_its_own_asymmetric_widths(self) -> None:
        """LIBERO is 8-dim state / 7-dim action; the two are checked separately.

        A single "robot width" field would have to pick one and skip the other,
        so this pins that the preset carries both.
        """
        preset = get_robot_preset("franka_libero")
        self.assertEqual((preset.state_dim, preset.action_width), (8, 7))

        # LIBERO's default is discrete_state=False, so only "actions" is checked;
        # turn state injection on and the 8-dim entry has to be there and correct.
        ckpt = CheckpointFixture(self, actions=_stats(7), state=_stats(16)).write_tokenizer()
        off = check_checkpoint(ckpt.path, "franka_libero")
        self.assertEqual([f for f in off if f.level == FAIL], [], format_findings(off))

        on = check_checkpoint(ckpt.path, "franka_libero", discrete_state=True)
        self.assertEqual(_levels(on, "norm_stats['state'] width"), [FAIL])

    def test_flat_norm_stats_layout_is_read_too(self) -> None:
        """openpi writes ``{"norm_stats": {...}}``; some exports are flat."""
        ckpt = CheckpointFixture(self)
        ckpt.write_norm_stats(
            nested=False, actions=_stats(G1_DIM), state=_stats(G1_DIM)
        ).write_tokenizer()

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        self.assertEqual([f for f in findings if f.level == FAIL], [], format_findings(findings))

    def test_missing_quantiles_are_fatal_for_pi05(self) -> None:
        """pi05 always unnormalizes with q01/q99 regardless of any config field."""
        ckpt = CheckpointFixture(
            self,
            actions=_stats(G1_DIM, quantiles=False),
            state=_stats(G1_DIM),
        ).write_tokenizer()

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        self.assertEqual(_levels(findings, "norm_stats['actions'] quantiles"), [FAIL])

    def test_missing_state_entry_is_fatal_only_when_state_is_used(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(G1_DIM)).write_tokenizer()

        used = check_checkpoint(ckpt.path, "unitree_g1")
        self.assertEqual(_levels(used, "norm_stats['state']"), [FAIL])

        dropped = check_checkpoint(ckpt.path, "unitree_g1", discrete_state=False)
        self.assertEqual(_levels(dropped, "norm_stats['state']"), [])

    def test_missing_file_is_fatal(self) -> None:
        ckpt = CheckpointFixture(self)
        ckpt.write_tokenizer()

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        self.assertEqual(_levels(findings, "norm_stats.json"), [FAIL])


class TokenizerTest(unittest.TestCase):
    """A14: openpi downloads its tokenizer at runtime, so exports lack one."""

    def test_absent_tokenizer_is_fatal(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(G1_DIM), state=_stats(G1_DIM))

        findings = check_checkpoint(ckpt.path, "unitree_g1")

        self.assertEqual(_levels(findings, "tokenizer"), [FAIL])

    def test_either_accepted_filename_satisfies_the_check(self) -> None:
        for name in ("tokenizer.model", "paligemma_tokenizer.model"):
            with self.subTest(name=name):
                ckpt = CheckpointFixture(
                    self, actions=_stats(G1_DIM), state=_stats(G1_DIM)
                ).write_tokenizer(name)
                findings = check_checkpoint(ckpt.path, "unitree_g1")
                self.assertEqual(_levels(findings, "tokenizer"), [INFO])

    def test_explicit_path_outranks_the_checkpoint_and_is_itself_checked(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(G1_DIM), state=_stats(G1_DIM))
        elsewhere = ckpt.path / "elsewhere.model"

        missing = check_checkpoint(ckpt.path, "unitree_g1", tokenizer_path=elsewhere)
        self.assertEqual(_levels(missing, "tokenizer"), [FAIL])

        elsewhere.write_bytes(b"x")
        present = check_checkpoint(ckpt.path, "unitree_g1", tokenizer_path=elsewhere)
        self.assertEqual(_levels(present, "tokenizer"), [INFO])


class OverrideTest(unittest.TestCase):
    """The flags that turn a correct checkpoint into a wrong deployment."""

    def setUp(self) -> None:
        self.ckpt = CheckpointFixture(
            self, actions=_stats(G1_DIM), state=_stats(G1_DIM)
        ).write_tokenizer()

    def test_action_dim_override_that_contradicts_the_robot_warns(self) -> None:
        """``--action-dim 7`` on a 16-DoF robot: the other half of the delivery."""
        findings = check_checkpoint(self.ckpt.path, "unitree_g1", action_dim=7)

        self.assertEqual(_levels(findings, "action_dim"), [WARN])

    def test_action_dim_matching_the_robot_is_silent(self) -> None:
        findings = check_checkpoint(self.ckpt.path, "unitree_g1", action_dim=G1_DIM)

        self.assertEqual(_levels(findings, "action_dim"), [])

    def test_turning_state_off_on_a_joint_space_robot_warns(self) -> None:
        findings = check_checkpoint(self.ckpt.path, "unitree_g1", discrete_state=False)

        warned = [f for f in findings if f.check == "discrete_state"]
        self.assertEqual([f.level for f in warned], [WARN])
        # State is *dropped*, not left continuous — the remedy has to say so,
        # because "no discrete state" reads like a precision choice.
        self.assertIn("dropped", warned[0].remedy)

    def test_reordered_camera_keys_warn_even_at_the_right_count(self) -> None:
        preset = get_robot_preset("unitree_g1")
        swapped = (preset.image_keys[1], preset.image_keys[0], preset.image_keys[2])

        findings = check_checkpoint(self.ckpt.path, "unitree_g1", image_keys=swapped)

        self.assertEqual(_levels(findings, "cameras"), [WARN])

    def test_fewer_cameras_than_the_preset_warns(self) -> None:
        preset = get_robot_preset("unitree_g1")

        findings = check_checkpoint(self.ckpt.path, "unitree_g1", image_keys=preset.image_keys[:2])

        self.assertEqual(_levels(findings, "cameras"), [WARN])


class ReportTest(unittest.TestCase):
    def test_findings_are_sorted_most_severe_first(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(7), state=_stats(8))

        findings = check_checkpoint(ckpt.path, "unitree_g1", action_dim=7)

        levels = [f.level for f in findings]
        self.assertEqual(levels, sorted(levels, key={FAIL: 0, WARN: 1, INFO: 2}.get))

    def test_quiet_rendering_drops_info_but_keeps_problems(self) -> None:
        ckpt = CheckpointFixture(self, actions=_stats(7), state=_stats(G1_DIM))

        findings = check_checkpoint(ckpt.path, "unitree_g1")
        quiet = format_findings(findings, include_info=False)

        self.assertNotIn("[INFO", quiet)
        self.assertIn("[FAIL", quiet)


if __name__ == "__main__":
    unittest.main()

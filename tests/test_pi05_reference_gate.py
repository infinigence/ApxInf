import copy
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gate", ROOT / "scripts/check_pi05_reference_gate.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class ReferenceGateTests(unittest.TestCase):
    def setUp(self):
        self.policy = json.loads(gate.POLICY.read_text())

    def test_policy_and_same_or_higher_thresholds(self):
        gate.validate(self.policy, gate.MANIFEST.read_bytes())
        gate.ratchet(self.policy, self.policy)
        higher = copy.deepcopy(self.policy)
        case = next(iter(higher["lanes"]["thor/bf16"]["cases"].values()))
        case["min_cosine"] += 0.000001
        gate.ratchet(higher, self.policy)

    def test_lowering_deleting_demoting_and_substitution_fail(self):
        for change in ("lower", "delete", "demote", "corpus", "graph"):
            with self.subTest(change=change):
                changed = copy.deepcopy(self.policy)
                lane = changed["lanes"]["thor/bf16"]
                name = next(iter(lane["cases"]))
                if change == "lower":
                    lane["cases"][name]["min_cosine"] -= 0.000001
                elif change == "delete":
                    del lane["cases"][name]
                elif change == "demote":
                    lane["required"] = False
                elif change == "corpus":
                    changed["manifest_sha256"] = "0" * 64
                else:
                    lane["eager_graph_max_abs"] = 0.01
                with self.assertRaises(ValueError):
                    gate.ratchet(changed, self.policy)

    def report(self, lane="thor/bf16"):
        metric = {"cosine": 1., "relative_l2": 0., "max_abs": 0.}
        return {"manifest_sha256": self.policy["manifest_sha256"],
                "variant": lane.split("/")[1], "mode": "measurement_only",
                "cases": [{"name": name, "eager_vs_cpu": dict(metric),
                           "graph_vs_cpu": dict(metric), "eager_vs_graph": dict(metric),
                           "repeat_bitwise_equal": True}
                          for name in self.policy["lanes"][lane]["cases"]]}

    def test_each_case_and_both_modes_are_gated(self):
        for mode in ("eager_vs_cpu", "graph_vs_cpu"):
            report = self.report()
            self.assertEqual(gate.check_report(self.policy, "thor/bf16", report), [])
            report["cases"][1][mode]["cosine"] = 0.99
            self.assertTrue(gate.check_report(self.policy, "thor/bf16", report))

    def test_diagnostic_error_metrics_do_not_override_cosine(self):
        report = self.report()
        report["cases"][0]["graph_vs_cpu"].update(relative_l2=0.5, max_abs=2.)
        self.assertEqual(gate.check_report(self.policy, "thor/bf16", report), [])

    def test_incomplete_nonfinite_and_wrong_identity_fail(self):
        for change in ("missing", "duplicate", "nan", "identity", "variant"):
            report = self.report()
            if change == "missing":
                report["cases"].pop()
            elif change == "duplicate":
                report["cases"][1] = report["cases"][0]
            elif change == "nan":
                report["cases"][0]["graph_vs_cpu"]["cosine"] = float("nan")
            elif change == "variant":
                report["variant"] = "int8_dynamic"
            else:
                report["manifest_sha256"] = "0" * 64
            with self.subTest(change=change), self.assertRaises(ValueError):
                gate.check_report(self.policy, "thor/bf16", report)

    def test_graph_replay_and_consistency_remain_required(self):
        report = self.report()
        report["cases"][0]["repeat_bitwise_equal"] = False
        self.assertTrue(gate.check_report(self.policy, "thor/bf16", report))
        report = self.report()
        report["cases"][0]["eager_vs_graph"]["max_abs"] = 0.001
        self.assertTrue(gate.check_report(self.policy, "thor/bf16", report))

    def test_int8_failure_is_still_recorded(self):
        report = self.report("orin/int8_dynamic")
        report["cases"][0]["graph_vs_cpu"]["cosine"] = -1.
        self.assertTrue(gate.check_report(self.policy, "orin/int8_dynamic", report))
        self.assertFalse(self.policy["lanes"]["orin/int8_dynamic"]["required"])


if __name__ == "__main__":
    unittest.main()

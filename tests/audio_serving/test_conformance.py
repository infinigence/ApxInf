import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/apxinf/apxinf/serving"))
from audio_contracts import AttemptKey, AttemptState, parse_frame
from audio_wav import inspect


class Conformance(unittest.TestCase):
    def test_shared_original_corpus(self):
        corpus = json.loads((ROOT / "tests/fixtures/audio-serving/cases.json").read_text())
        frames = {}
        for case in corpus["frames"]:
            with self.subTest(case=case["name"]):
                try:
                    raw = case["raw"] if "raw" in case else json.dumps(case["frame"], ensure_ascii=False)
                    frame = parse_frame(raw.encode())
                    valid = True
                    frames[case["name"]] = frame
                except (ValueError, TypeError, UnicodeError):
                    valid = False
                self.assertEqual(valid, case["valid"])
        for case in corpus["wav"]:
            with self.subTest(case=case["name"]):
                try:
                    inspect(bytes.fromhex(case["hex"]), case["asr"])
                    valid = True
                except ValueError:
                    valid = False
                self.assertEqual(valid, case["valid"])
        for case in corpus["states"]:
            with self.subTest(case=case["name"]):
                state = AttemptState(AttemptKey("22222222-2222-4222-8222-222222222222", 1), "33333333-3333-4333-8333-333333333333")
                try:
                    for name in case["events"]:
                        state.event(frames[name].message)
                    valid = True
                except ValueError:
                    valid = False
                self.assertEqual(valid, case["valid"])


if __name__ == "__main__":
    unittest.main()

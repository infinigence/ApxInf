import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/apxinf/apxinf/serving"))
from audio_contracts import AttemptKey, AttemptState, is_text, parse_frame
from audio_wav import inspect


class Conformance(unittest.TestCase):
    def test_shared_original_corpus(self):
        corpus = json.loads((ROOT / "tests/fixtures/audio-serving/cases.json").read_text())
        frames = {}
        for case in corpus["texts"]:
            with self.subTest(case=case["name"]):
                self.assertEqual(is_text(case["text"]), case["valid"])
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
        for case in corpus["encodings"]:
            with self.subTest(case=case["name"]):
                if "hex" in case:
                    raw = bytes.fromhex(case["hex"])
                else:
                    source = next(c["raw"] for c in corpus["frames"] if c["name"] == case["frame"])
                    encoding = {"utf8":"utf-8", "utf16le":"utf-16le", "utf32le":"utf-32le"}[case["encoding"]]
                    raw = source.encode(encoding)
                    if case["bom"]:
                        raw = {"utf8":b"\xef\xbb\xbf", "utf16le":b"\xff\xfe", "utf32le":b"\xff\xfe\x00\x00"}[case["encoding"]] + raw
                try:
                    parse_frame(raw)
                    valid = True
                except (ValueError, UnicodeError):
                    valid = False
                self.assertEqual(valid, case["valid"])


if __name__ == "__main__":
    unittest.main()

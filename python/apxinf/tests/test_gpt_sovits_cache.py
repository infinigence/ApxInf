"""Cache correctness and invalidation through the public TTS policy."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from apxinf import GptSovitsPolicy


class Segmentation:
    """Stands in for the frontend's TTS_infer_pack/text_segmentation_method.py."""

    METHODS = ("cut0", "cut1", "cut2", "cut3", "cut4", "cut5")

    def get_method_names(self):
        return list(self.METHODS)

    def get_method(self, name):
        if name not in self.METHODS:
            raise ValueError(name)
        return lambda text: text


class Runner:
    sample_rate = 48000
    device = "cpu"
    is_prepared = False

    def configure_cache(self, **options):
        self.options = options

    def clear_cache(self, **options):
        pass

    def cache_info(self):
        return {"supported": True, **getattr(self, "options", {})}

    def synthesize(self, *values, **options):
        samples = np.concatenate([value.reshape(-1) for value in values]).astype(np.float32)
        return {"samples": samples, "valid_length": len(samples), "sample_rate": 48000}


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.audio = Path(self.directory.name) / "reference.wav"
        self.audio.write_bytes(b"audio-a")
        self.calls = {"text": 0, "audio": 0}
        def clean(text, language, version):
            self.calls["text"] += 1
            return list(text), [1] * len(text), text
        self.policy = GptSovitsPolicy(
            Runner(),
            frontend={
                "segmentation": Segmentation(),
                "clean_text": clean,
                "cleaned_text_to_sequence": lambda phones, version: [ord(x) for x in phones],
                "tokenizer": lambda text, **kwargs: {"input_ids": [0, *map(ord, text), 0]},
                "torchaudio": object(),
                "mel_spectrogram_torch": object(),
                "spectrogram_torch": object(),
            },
        )
        def audio(path):
            self.calls["audio"] += 1
            value = float(sum(path.read_bytes()))
            return tuple(np.full(shape, value, dtype=np.float32)
                         for shape in [(16,), (4, 100), (4, 704)])
        self.policy._compute_reference_inputs = audio
        self.observation = {
            "text": "甲乙", "reference_text": "声音",
            "reference_audio": self.audio, "seed": 1234,
        }

    def test_cache_changes_work_but_preserves_pcm(self):
        self.policy.configure_cache(prompt=False, text=False)
        expected = self.policy.infer(self.observation)["samples"]
        self.policy.configure_cache(prompt=True, text=True)
        first = self.policy.infer(self.observation)["samples"]
        second = self.policy.infer(self.observation)["samples"]
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, expected)
        info = self.policy.cache_info()["frontend"]
        self.assertEqual(info["reference_audio_hits"], 1)
        self.assertEqual(info["reference_text_hits"], 1)
        self.assertEqual(info["text_hits"], 1)
        self.assertEqual(info["reference_audio_entries"], 1)

    def test_prompt_reuses_reference_for_new_target_with_text_disabled(self):
        self.policy.configure_cache(prompt=True, text=False)
        self.policy.infer(self.observation)
        self.policy.infer({**self.observation, "text": "丙丁"})
        self.assertEqual(self.calls, {"text": 3, "audio": 1})
        info = self.policy.cache_info()["frontend"]
        self.assertEqual(info["text_entries"], 0)
        self.assertEqual(info["reference_text_hits"], 1)

    def test_lru_eviction_language_version_and_frontend_invalidation(self):
        self.policy.configure_cache(max_text_entries=2)
        for text in ["甲", "乙", "甲", "丙", "乙"]:
            self.policy._text_inputs(text)
        self.assertEqual(self.calls["text"], 4)
        self.assertEqual(self.policy.cache_info()["frontend"]["text_entries"], 2)
        self.policy.language = "en"
        self.policy._text_inputs("乙")
        self.policy.text_version = "v3"
        self.policy._text_inputs("乙")
        self.policy.frontend["tokenizer"] = lambda text, **kw: {"input_ids": [0, *map(ord, text), 0]}
        self.policy._text_inputs("乙")
        self.assertEqual(self.calls["text"], 7)

    def test_same_path_rewrite_and_reference_switch_invalidate(self):
        first = self.policy._reference_inputs(self.audio)[0].copy()
        self.audio.write_bytes(b"audio-b")
        second = self.policy._reference_inputs(self.audio)[0].copy()
        self.assertFalse(np.array_equal(first, second))
        other = self.audio.with_name("other.wav")
        other.write_bytes(b"audio-c")
        self.policy._reference_inputs(other)
        self.policy._reference_inputs(self.audio)
        self.assertEqual(self.calls["audio"], 4)
        self.assertEqual(self.policy.cache_info()["frontend"]["reference_audio_entries"], 1)

    def test_cached_values_have_immutable_storage(self):
        for values in [
            self.policy._text_inputs("甲"),
            self.policy._reference_inputs(self.audio),
        ]:
            for value in values:
                with self.assertRaises(ValueError):
                    value.flat[0] = 0
                with self.assertRaises(ValueError):
                    value.setflags(write=True)

    def test_clear_disable_zero_capacity_and_close(self):
        self.policy.infer(self.observation)
        self.policy.clear_cache(reset_stats=False)
        info = self.policy.cache_info()["frontend"]
        self.assertEqual(info["host_bytes"], 0)
        self.assertEqual(info["reference_audio_misses"], 1)
        self.policy.clear_cache()
        self.assertEqual(self.policy.cache_info()["frontend"]["reference_audio_misses"], 0)
        self.policy.configure_cache(prompt=False, text=False, max_text_entries=0)
        self.policy.infer(self.observation)
        self.policy.infer(self.observation)
        self.assertEqual(self.policy.cache_info()["frontend"]["host_bytes"], 0)
        self.assertFalse(self.policy.model_runner.options["prompt"])
        with self.assertRaises(ValueError):
            self.policy.configure_cache(max_text_entries=-1)
        self.policy.close()
        self.assertIsNone(self.policy.model_runner)


if __name__ == "__main__":
    unittest.main()

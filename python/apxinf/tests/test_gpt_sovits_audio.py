"""Independent reference tests for the optional GPT-SoVITS audio frontend."""

import numpy as np
import unittest

from apxinf.policies.impls._gpt_sovits_audio import resample_kaiser_best


class AudioReferenceTests(unittest.TestCase):
    def test_kaiser_best_matches_independent_resampy(self):
        try:
            import torch
            import resampy
        except ImportError as exc:
            self.skipTest(str(exc))
        for source_rate in [8000, 16000, 32000, 44100, 48000]:
            with self.subTest(source_rate=source_rate):
                t = np.arange(4097, dtype=np.float32) / source_rate
                wave = (
                    0.2 * np.sin(t * (2 * np.pi * 997))
                    + 0.05 * np.cos(t * (2 * np.pi * 317))
                ).astype(np.float32)
                expected = resampy.resample(wave, source_rate, 16000, filter="kaiser_best")
                audio = torch.from_numpy(np.stack([wave, -wave]))
                cache = {}
                actual = resample_kaiser_best(audio, source_rate, 16000, cache)
                self.assertEqual(actual.shape, (2, expected.size))
                np.testing.assert_allclose(actual[0].numpy(), expected, atol=1e-4, rtol=1e-4)
                np.testing.assert_allclose(actual[1].numpy(), -expected, atol=1e-4, rtol=1e-4)
                self.assertTrue(torch.equal(resample_kaiser_best(audio, source_rate, 16000, cache), actual))
                if source_rate == 16000:
                    self.assertNotEqual(actual.data_ptr(), audio.data_ptr())
                else:
                    self.assertEqual(len(cache), 1)


if __name__ == "__main__":
    unittest.main()

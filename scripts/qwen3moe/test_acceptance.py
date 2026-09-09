"""Failure-path tests for the numerical gate; run with Python + NumPy."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np


class NumericalGate(unittest.TestCase):
    def check_dump(self, mutation=None, expected=0):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logits = np.array([[10, 3, 2, 1, 0, -1], [2, 10, 3, 1, 0, -1]], dtype=np.float32)
            np.savez(root / 'ref.npz', token_ids=[12], greedy_tokens=[0, 1], step_logits=logits)
            meta = dict(logits='out.bin', rows=2, vocab_size=6, token_ids=[12],
                        greedy_tokens=[0, 1], reference_greedy_tokens=[0, 1], teacher_forced=True)
            if mutation:
                logits, meta = mutation(logits, meta)
            logits.astype('<f4').tofile(root / 'out.bin')
            (root / 'out.json').write_text(json.dumps(meta))
            result = subprocess.run([sys.executable, str(Path(__file__).with_name('compare_reference.py')),
                    '--reference', str(root / 'ref.npz'), '--apxinf', str(root / 'out.json'), '--strict'], capture_output=True)
            self.assertEqual(result.returncode == 0, expected == 0, result.stdout + result.stderr)

    def test_complete(self):
        self.check_dump()

    def test_truncated(self):
        self.check_dump(lambda x, m: (x[:1], dict(m, rows=1, greedy_tokens=[0])), 1)

    def test_nonfinite(self):
        def mutate(x, m):
            x[0, 3] = np.nan
            return x, m
        self.check_dump(mutate, 1)

    def test_wrong_forcing(self):
        self.check_dump(lambda x, m: (x, dict(m, reference_greedy_tokens=[0, 2])), 1)

    def test_error_even_when_argmax_matches(self):
        self.check_dump(lambda x, m: (x + 2, m), 1)

    def test_free_running(self):
        self.check_dump(lambda x, m: (x, dict(m, teacher_forced=False)), 1)


if __name__ == '__main__':
    unittest.main()

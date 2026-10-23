from __future__ import annotations

import copy
import math
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/apxinf/apxinf/serving"))
import clef_contracts as c
from clef_runtime import ClefRuntime, ClefError
from test_serving_clef_contracts import CORPUS, expand


def external_encoding(document):
    return SimpleNamespace(input_ids=tuple(document["input_ids"]), media=None, questions=tuple(
        SimpleNamespace(**{**q, "question_span": tuple(q["question_span"]),
                            "option_spans": tuple(tuple(s) for s in q["option_spans"]),
                            "option_ids": tuple(q["option_ids"])}) for q in document["questions"]))


class Vector:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)

    def astype(self, dtype):
        if dtype != "float32":
            raise AssertionError("logits must narrow to float32")
        return self

    def tolist(self):
        return list(self.values)


class FakeMX:
    float32 = "float32"

    def softmax(self, vector):
        values = [math.exp(x - max(vector.values)) for x in vector.values]
        return Vector([value / sum(values) for value in values])

    def eval(self, value):
        pass

    def synchronize(self):
        pass

    def get_peak_memory(self):
        return 123


class ClefRuntimeTests(unittest.TestCase):
    def runtime(self):
        record = c.validate_record(expand(CORPUS["objects"]["record"]))
        document = expand(CORPUS["objects"]["encoding"])
        encoding = c.validate_encoding(document, record)
        runtime = ClefRuntime.__new__(ClefRuntime)
        runtime.owner = threading.get_ident()
        runtime.max_context = 4096
        runtime.vocab_size = 248320
        runtime.used = False
        runtime.mx = FakeMX()
        calls = []

        class ContextTooLong(ValueError):
            pass

        def encode(tokenizer, value, **kwargs):
            self.assertEqual(value, record.document())
            self.assertEqual(kwargs, dict(max_length=4096, max_state_tokens=None, processor=None, truncate=False))
            calls.append("encode")
            return external_encoding(document)

        def logits(value, **kwargs):
            self.assertEqual(value, record.document())
            self.assertEqual(kwargs, dict(max_length=4096, max_state_tokens=None, truncate=False))
            calls.append("forward")
            return external_encoding(document), [Vector([0, 1]), Vector([0, 1, 2]), Vector([2, 0])]

        runtime.dependency = SimpleNamespace(encode_record=encode, ContextTooLong=ContextTooLong)
        runtime.model = SimpleNamespace(tokenizer=object(), processor=None, logits=logits)
        return runtime, record, encoding, document, calls

    def test_single_forward_and_full_precision_probabilities(self):
        runtime, record, encoding, _, calls = self.runtime()
        probabilities, metrics = runtime.decide(record, encoding)
        self.assertEqual(calls, ["encode", "forward"])
        c.validate_probabilities(probabilities, encoding)
        self.assertEqual(metrics["peak_device_bytes"], 123)
        self.assertNotEqual(probabilities[0]["values"][0], round(probabilities[0]["values"][0], 4))
        with self.assertRaises(ClefError):
            runtime.decide(record, encoding)

    def test_changed_pre_forward_encoding_never_calls_forward(self):
        for change in (lambda d: d["input_ids"].__setitem__(0, 99),
                       lambda d: d["questions"][0].__setitem__("question_span", [0, 2]),
                       lambda d: d["questions"][0].__setitem__("option_ids", ["red", "blue"])):
            runtime, record, encoding, document, calls = self.runtime()
            change(document)
            with self.assertRaises((ClefError, c.ContractError)):
                runtime.decide(record, encoding)
            self.assertNotIn("forward", calls)

    def test_changed_returned_encoding_discards_logits(self):
        runtime, record, encoding, document, _ = self.runtime()
        original = runtime.model.logits

        def changed(*args, **kwargs):
            _, logits = original(*args, **kwargs)
            mutated = copy.deepcopy(document)
            mutated["input_ids"][0] = 99
            return external_encoding(mutated), logits

        runtime.model.logits = changed
        with self.assertRaises(ClefError):
            runtime.decide(record, encoding)

    def test_context_overflow_never_calls_forward(self):
        runtime, record, _, _, calls = self.runtime()

        def overflow(*args, **kwargs):
            raise runtime.dependency.ContextTooLong()

        runtime.dependency.encode_record = overflow
        with self.assertRaises(ClefError) as failure:
            runtime.encode(record)
        self.assertEqual(failure.exception.code, "context_limit")
        self.assertEqual(calls, [])

    def test_complete_vectors_required(self):
        for replacement in ([Vector([0, 1])], [Vector([0, 1]), Vector([0, 1]), Vector([0, 1])],
                            [Vector([float("nan"), 0]), Vector([0, 1, 2]), Vector([0, 1])]):
            runtime, record, encoding, document, _ = self.runtime()
            runtime.model.logits = lambda *args, **kwargs: (external_encoding(document), replacement)
            with self.assertRaises((ClefError, c.ContractError)):
                runtime.decide(record, encoding)

    def test_owner_thread_required(self):
        runtime, record, _, _, _ = self.runtime()
        errors = []

        def run():
            try:
                runtime.encode(record)
            except ClefError as error:
                errors.append(error.code)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        self.assertEqual(errors, ["internal_error"])


if __name__ == "__main__":
    unittest.main()

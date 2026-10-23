from __future__ import annotations

import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python/apxinf/apxinf/serving"))
import contracts
import clef_contracts as c
from clef_worker import ClefWorker
from clef_identity import ClefError
from test_serving_clef_contracts import CORPUS, expand


def fake_runtime():
    source = lambda path: {"path": path, "sha256": "a" * 64}
    manifest = {"checkpoint": c.CHECKPOINT, "checkpoint_revision": c.CHECKPOINT_REVISION,
                "adapter_revision": c.ADAPTER_REVISION, "platform": "macos-metal",
                "artifacts": c.approved_artifacts(),
                "loader": {"path": "/local/clef_mlx.py", "sha256": c.LOADER_SHA256}, "runtime": c.RUNTIME,
                "sources": {name: source("/adapter/" + name) for name in c.SOURCE_FILES},
                "python": source("/env/bin/python"), "runtime_files": {
                    "modules": {name: source("/env/" + name + ".py") for name in
                                ("mlx_core", "mlx_lm", "transformers", "numpy", "pillow", "tokenizers")},
                    "libraries": [source("/env/libmlx.dylib")]}, "execution": c.execution(4096, 9000000000)}
    record = c.validate_record(expand(CORPUS["objects"]["record"]))
    encoding = c.validate_encoding(expand(CORPUS["objects"]["encoding"]), record)
    return SimpleNamespace(manifest=manifest, model_path="/model", capabilities=c.capabilities(248320, 4096),
        model_revision=contracts.canonical_identity_digest("apxinf-model-identity-v1", manifest),
        capability_revision=contracts.canonical_identity_digest("apxinf-capability-v1", c.capabilities(248320, 4096)),
        memory=lambda: dict(active_bytes=1, peak_bytes=1, cache_bytes=0), encode=lambda _: encoding,
        decide=lambda *_: (expand(CORPUS["objects"]["probabilities"]), dict(elapsed_ms=1, peak_device_bytes=2)))


class ClefWorkerTests(unittest.TestCase):
    def worker(self):
        output = []
        runtime = fake_runtime()
        worker = ClefWorker(runtime, CORPUS["frames"]["prepare"]["worker_epoch"],
                            lambda raw: output.append(contracts.decode_frame(raw)))
        prepare = expand(CORPUS["frames"]["prepare"])
        prepare.update(model_revision=runtime.model_revision, capability_revision=runtime.capability_revision)
        return worker, prepare, output

    def submit(self, worker, prepare, output):
        self.assertIsNone(worker.handle(prepare))
        submitted = expand(CORPUS["frames"]["submit"])
        submitted.update(model_revision=worker.runtime.model_revision,
                         capability_revision=worker.runtime.capability_revision, preparation=output[-1]["preparation"])
        return submitted

    def test_ready_and_successful_single_attempt_stream(self):
        worker, prepare, output = self.worker()
        worker.ready()
        submit = self.submit(worker, prepare, output)
        seen = []
        original = worker.runtime.decide

        def decide(*args):
            seen.append(output[-1]["kind"])
            return original(*args)

        worker.runtime.decide = decide
        self.assertEqual(worker.handle(submit), 0)
        self.assertEqual(seen, ["accepted"])
        self.assertEqual([event["kind"] for event in output], ["ready", "prepared_decision", "accepted", "decision_result", "drained"])
        tracker = c.DecisionTracker()
        for frame in [output[1], submit, *output[2:]]:
            tracker.observe(frame)
        self.assertFalse(tracker.public_success)
        self.assertTrue(tracker.reaped(0, True))
        with self.assertRaises(ClefError):
            worker.handle(submit)

    def test_context_overflow_finishes_without_acceptance(self):
        worker, prepare, output = self.worker()

        def overflow(_):
            raise ClefError("context_limit", "The record exceeds the context limit.")

        worker.runtime.encode = overflow
        self.assertEqual(worker.handle(prepare), 0)
        self.assertEqual([frame["kind"] for frame in output], ["failure", "drained"])
        self.assertEqual(output[0]["event_seq"], 0)

    def test_forward_exception_never_emits_success(self):
        worker, prepare, output = self.worker()
        submit = self.submit(worker, prepare, output)

        def failed(*_):
            raise RuntimeError("device failure")

        worker.runtime.decide = failed
        self.assertEqual(worker.handle(submit), 1)
        self.assertEqual([frame["kind"] for frame in output], ["prepared_decision", "accepted", "failure", "drained"])
        self.assertEqual(output[-2]["event_seq"], 1)

    def test_checked_preparation_is_owned_independently(self):
        worker, prepare, output = self.worker()
        submit = self.submit(worker, prepare, output)
        submit["preparation"]["record"]["state"] = "changed"
        with self.assertRaises(c.ContractError):
            worker.handle(submit)
        self.assertEqual(len(output), 1)

    def test_shutdown_and_incomplete_input(self):
        worker, _, output = self.worker()
        shutdown = expand(CORPUS["frames"]["shutdown"])
        self.assertEqual(worker.run(io.BytesIO(contracts.encode_frame(shutdown))), 0)
        self.assertEqual([frame["kind"] for frame in output], ["ready", "drained"])
        self.assertEqual(output[-1]["reason"], "shutdown")
        worker, _, output = self.worker()
        self.assertEqual(worker.run(io.BytesIO(b"{}")), 1)
        self.assertEqual([frame["kind"] for frame in output], ["ready"])


if __name__ == "__main__":
    unittest.main()

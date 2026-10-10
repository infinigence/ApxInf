"""Original lifecycle tests for the serial text worker. No model is loaded."""

from __future__ import annotations

import importlib.util
import codecs
import contextlib
import gc
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
import weakref
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
MODULE_DIR = ROOT / "python/apxinf/apxinf/serving"
sys.path.insert(0, str(MODULE_DIR))
import contracts
import text_worker as worker_module


EPOCH = "11111111-1111-4111-8111-111111111111"
REQUEST = "22222222-2222-4222-8222-222222222222"
LEASE = "33333333-3333-4333-8333-333333333333"


class FakeRuntime:
    def __init__(self, tokens=(10, 11, 2)):
        self.owner = threading.get_ident()
        self.max_context = 64
        self.vocab_size = 128
        self.eos_token_ids = [2]
        self.model_path = "/fake/model"
        self.runtime = dict(worker_module.PINNED_RUNTIME)
        self.manifest = {"model": "original-test-model", "runtime": self.runtime}
        self.tokens = list(tokens)
        self.consumed_position = 0
        self.settled = 0
        self.generated = 0
        self.prepared = None
        self.final_tail = ""
        self.on_progress = None

    def check_owner(self):
        assert threading.get_ident() == self.owner

    def memory(self):
        self.check_owner()
        return {"active_bytes": 1024, "peak_bytes": 2048, "cache_bytes": 0}

    def prepare(self, command):
        self.check_owner()
        self.prepared = command
        return [3, 4]

    def generate(self, command, progress):
        self.check_owner()
        self.generated += 1
        for consumed in range(len(command["token_ids"]) + 1):
            self.consumed_position = consumed
            if self.on_progress:
                self.on_progress(consumed)
            progress(consumed)
        for token in self.tokens[:command["max_tokens"]]:
            self.consumed_position += 1
            yield token

    def decode(self, token, eos=False):
        self.check_owner()
        return "" if eos else str(token)

    def finish_text(self):
        self.check_owner()
        return self.final_tail

    def settle(self):
        self.check_owner()
        self.settled += 1


class BufferedDecodeRuntime(FakeRuntime):
    """Use a real incremental UTF-8 decoder with original failure boundaries."""

    def __init__(self, tokens=(10, 11), model_failure=None, decode_failure=None):
        super().__init__(tokens)
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.model_failure = model_failure
        self.decode_failure = decode_failure
        self.finalized = 0

    def generate(self, command, progress):
        yield from super().generate(command, progress)
        if self.model_failure is not None:
            raise self.model_failure

    def decode(self, token, eos=False):
        self.check_owner()
        text = self.decoder.decode({10: b"A", 11: b"\xe4", 12: b"\xb8\xad", 13: b"B"}[token])
        if token == self.decode_failure:
            raise RuntimeError("The original decoder probe failed after mutation.")
        return text

    def finish_text(self):
        self.check_owner()
        self.finalized += 1
        return self.decoder.decode(b"", final=True)


class MemoryBackend:
    """Record stream operations without importing MLX or creating arrays."""

    def __init__(self):
        self.current = "load"
        self.events = []
        self.fail_stream = None
        self.fail_reset = False
        self.on_clear = None

    def default_device(self):
        return "test-device"

    def default_stream(self, device):
        return self.current

    def synchronize(self, stream=None):
        self.events.append(("sync", stream))
        if stream == self.fail_stream:
            raise RuntimeError("Original simulated stream failure.")

    def set_memory_limit(self, limit):
        self.events.append(("limit", limit))

    def reset_peak_memory(self):
        if self.fail_reset:
            raise RuntimeError("Original simulated reset failure.")
        self.events.append(("reset_peak",))

    def clear_cache(self):
        self.events.append(("clear_cache",))
        if self.on_clear:
            self.on_clear()

    def array(self, values):
        return list(values)

    def get_active_memory(self):
        return 512

    def get_cache_memory(self):
        return 32

    def get_peak_memory(self):
        return 768


class MemoryPayload:
    pass


class PositionedCache:
    def __init__(self):
        self.offset = 0
        self.nbytes = 64
        self.payload = MemoryPayload()


class RecurrentCache:
    nbytes = 128

    def size(self):
        raise AssertionError("A recurrent cache does not report a token offset.")


class RuntimeMemoryTests(unittest.TestCase):
    def setUp(self):
        self.backend = MemoryBackend()
        self.cache_factory = lambda model: [PositionedCache(), RecurrentCache()]
        self.detokenizer = types.SimpleNamespace(reset=lambda: None)
        tokenizer = types.SimpleNamespace(detokenizer=self.detokenizer, eos_token_ids={2})
        config = {"vocab_size": 128, "max_position_embeddings": 64}

        def load(*args, **kwargs):
            self.backend.events.append(("load",))
            return object(), tokenizer, config

        # Exercise the actual constructor while replacing only its external boundaries.
        mlx = types.ModuleType("mlx")
        mlx.core = self.backend
        mlx_lm = types.ModuleType("mlx_lm")
        mlx_lm.load = load
        generation = types.ModuleType("mlx_lm.generate")
        generation.generate_step = self.generate_step
        cache_module = types.ModuleType("mlx_lm.models.cache")
        cache_module.make_prompt_cache = lambda model: self.cache_factory(model)
        helper = types.SimpleNamespace(_pinned_toolchain_versions=lambda: {},
                                       _parse_json=lambda data, name: json.loads(data))
        spec = types.SimpleNamespace(loader=types.SimpleNamespace(exec_module=lambda module: None))
        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            (Path(directory) / "config.json").write_text("{}")
            stack.enter_context(patch.dict(sys.modules, {
                "mlx": mlx, "mlx.core": self.backend, "mlx_lm": mlx_lm,
                "mlx_lm.generate": generation, "mlx_lm.models.cache": cache_module,
            }))
            stack.enter_context(patch.dict(worker_module.os.environ))
            stack.enter_context(patch.object(worker_module.platform, "python_version", return_value="3.14.3"))
            stack.enter_context(patch.object(worker_module.importlib.metadata, "version",
                                            side_effect={"mlx": "0.32.1", "mlx-lm": "0.31.3"}.__getitem__))
            stack.enter_context(patch.object(worker_module.importlib.util, "spec_from_file_location", return_value=spec))
            stack.enter_context(patch.object(worker_module.importlib.util, "module_from_spec", return_value=helper))
            stack.enter_context(patch.object(worker_module, "model_manifest", return_value={}))
            stack.enter_context(patch.object(worker_module, "_file_digest", return_value="original-test-digest"))
            self.runtime = worker_module.MLXRuntime(directory, 64, 16)
        self.command = {"token_ids": [3, 4], "max_tokens": 1}

    def generate_step(self, prompt, model, *, max_tokens, prompt_cache,
                      prompt_progress_callback, **kwargs):
        previous = self.backend.current
        try:
            self.backend.current = "generation"
            prompt_progress_callback(0, len(prompt))
            prompt_cache[0].offset = len(prompt) + bool(max_tokens)
            # The final progress callback runs after the generation stream context exits.
            self.backend.current = previous
            prompt_progress_callback(len(prompt), len(prompt))
            for index in range(max_tokens):
                prompt_cache[0].offset = len(prompt) + index + 1
                yield 10 + index, None
        finally:
            self.backend.current = previous
            self.backend.events.append(("generator_closed",))

    def test_loading_and_inspection_keep_the_original_default_stream(self):
        self.assertEqual(self.backend.events[-2:], [("load",), ("sync", "load")])
        self.backend.current = "unrelated"
        self.backend.events.clear()
        observation = self.runtime.inspect_memory()
        self.assertEqual(self.backend.events, [("sync", "load")])
        self.assertEqual(observation, {
            "allocator": {"active_bytes": 512, "cache_bytes": 32, "peak_bytes": 768},
            "peak_epoch": 0, "runtime_consumed_position": 0,
            "cache_payload_bytes": 0, "layers": [],
        })

    def test_lazy_iterator_does_not_reset_peak_or_create_state(self):
        iterator = self.runtime.generate(self.command, lambda consumed: None)
        iterator.close()
        self.assertEqual(self.runtime.inspect_memory()["peak_epoch"], 0)
        self.assertIsNone(self.runtime._cache)
        self.assertNotIn(("reset_peak",), self.backend.events)
        self.backend.fail_reset = True
        with self.assertRaisesRegex(RuntimeError, "reset failure"):
            next(self.runtime.generate(self.command, lambda consumed: None))
        self.assertEqual(self.runtime.inspect_memory()["peak_epoch"], 0)

    def test_progress_stream_capture_survives_final_callback_and_iterator_close(self):
        self.backend.current = "request"
        observations = []
        iterator = self.runtime.generate(self.command,
                                         lambda consumed: observations.append(self.runtime.inspect_memory()))
        self.assertEqual(next(iterator), 10)
        iterator.close()
        self.assertEqual(self.backend.current, "request")
        self.assertEqual(observations[-1]["runtime_consumed_position"], 2)
        self.assertEqual(observations[-1]["layers"][0]["offset"], 3)
        self.assertEqual(observations[-1]["layers"][1]["offset"], None)
        self.assertEqual(observations[-1]["cache_payload_bytes"], 192)
        self.backend.events.clear()
        self.backend.current = "unrelated"
        observation = self.runtime.inspect_memory()
        self.assertEqual(self.backend.events, [("sync", "generation"), ("sync", "request"), ("sync", "load")])
        self.assertEqual(observation["runtime_consumed_position"], 3)
        self.assertEqual(observation["peak_epoch"], 1)

    def test_settlement_synchronizes_before_release_and_cache_clear(self):
        self.backend.current = "request"
        iterator = self.runtime.generate(self.command, lambda consumed: None)
        next(iterator)
        iterator.close()
        self.backend.current = "unrelated"
        self.backend.events.clear()

        def after_release():
            self.assertIsNone(self.runtime._cache)
            self.assertIsNone(self.runtime._detokenizer)
        self.backend.on_clear = after_release
        self.runtime.settle()
        streams = [("sync", "generation"), ("sync", "request"), ("sync", "load")]
        self.assertEqual(self.backend.events, streams + [("clear_cache",)] + streams)
        self.assertEqual(self.runtime.inspect_memory()["cache_payload_bytes"], 0)

    def test_sync_failure_keeps_state_and_does_not_report_settlement(self):
        self.backend.current = "request"
        iterator = self.runtime.generate(self.command, lambda consumed: None)
        next(iterator)
        iterator.close()
        cache = self.runtime._cache
        self.backend.events.clear()
        for stream in ["generation", "request", "load"]:
            self.backend.fail_stream = stream
            for operation in [self.runtime.inspect_memory, self.runtime.settle]:
                with self.assertRaisesRegex(RuntimeError, "stream failure"):
                    operation()
                self.assertIs(self.runtime._cache, cache)
                self.assertIs(self.runtime._detokenizer, self.detokenizer)
        self.assertNotIn(("clear_cache",), self.backend.events)
        self.backend.fail_stream = None
        self.runtime.settle()
        self.assertIsNone(self.runtime._cache)

    def test_second_sync_failure_does_not_complete_settlement(self):
        self.backend.current = "request"
        iterator = self.runtime.generate(self.command, lambda consumed: None)
        next(iterator)
        iterator.close()
        self.backend.events.clear()

        def fail_after_clear():
            self.backend.fail_stream = "generation"
        self.backend.on_clear = fail_after_clear
        with self.assertRaisesRegex(RuntimeError, "stream failure"):
            self.runtime.settle()
        self.assertEqual(self.backend.events, [
            ("sync", "generation"), ("sync", "request"), ("sync", "load"),
            ("clear_cache",), ("sync", "generation"),
        ])
        self.assertIsNone(self.runtime._cache)
        self.assertIsNone(self.runtime._detokenizer)
        self.assertEqual(self.runtime._generation_stream, "generation")
        self.assertEqual(self.runtime._request_stream, "request")
        with self.assertRaisesRegex(RuntimeError, "stream failure"):
            self.runtime.inspect_memory()

    def test_prefill_stop_before_first_yield_keeps_its_captured_stream(self):
        def stop(consumed):
            raise worker_module.GenerationStopped()

        iterator = self.runtime.generate(self.command, stop)
        with self.assertRaises(worker_module.GenerationStopped):
            next(iterator)
        iterator.close()
        self.assertEqual(self.backend.current, "load")
        self.backend.events.clear()
        observation = self.runtime.inspect_memory()
        self.assertEqual(self.backend.events, [("sync", "generation"), ("sync", "load")])
        self.assertEqual(observation["peak_epoch"], 1)
        self.assertEqual(observation["runtime_consumed_position"], 0)
        self.runtime.settle()

    def test_started_zero_yield_generator_has_one_peak_epoch(self):
        command = dict(self.command, max_tokens=0)
        self.assertEqual(list(self.runtime.generate(command, lambda consumed: None)), [])
        observation = self.runtime.inspect_memory()
        self.assertEqual(observation["peak_epoch"], 1)
        self.assertEqual(observation["runtime_consumed_position"], 2)
        self.assertEqual(observation["layers"][0]["offset"], 2)
        self.runtime.settle()
        self.assertEqual(list(self.runtime.generate(command, lambda consumed: None)), [])
        self.assertEqual(self.runtime.inspect_memory()["peak_epoch"], 2)
        self.runtime.settle()

    def test_unknown_metadata_stays_null_and_snapshots_do_not_retain_arrays(self):
        class UnsupportedCache:
            @property
            def nbytes(self):
                raise NotImplementedError("No byte observation.")

        class ArrayMetadata:
            offset = MemoryPayload()
            nbytes = True

        self.cache_factory = lambda model: [PositionedCache(), UnsupportedCache(), ArrayMetadata()]
        iterator = self.runtime.generate(self.command, lambda consumed: None)
        next(iterator)
        iterator.close()
        payload = weakref.ref(self.runtime._cache[0].payload)
        observation = self.runtime.inspect_memory()
        self.assertIsNone(observation["cache_payload_bytes"])
        self.assertEqual(observation["layers"][1], {"type": "UnsupportedCache", "offset": None, "nbytes": None})
        self.assertEqual(observation["layers"][2], {"type": "ArrayMetadata", "offset": None, "nbytes": None})
        json.dumps(observation, allow_nan=False)
        self.runtime.settle()
        gc.collect()
        self.assertIsNone(payload())

    def test_observation_and_settlement_require_the_execution_owner(self):
        errors = []

        def other_thread():
            for operation in [self.runtime.inspect_memory, self.runtime.settle]:
                try:
                    operation()
                except RuntimeError as error:
                    errors.append(str(error))
        self.backend.events.clear()
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 2)
        self.assertTrue(all("execution owner" in error for error in errors))
        self.assertEqual(self.backend.events, [])


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.runtime = FakeRuntime()

        def emit(frame, critical=False):
            contracts.validate_frame(frame)
            self.events.append(frame)
        self.worker = worker_module.TextWorker(self.runtime, EPOCH, emit, max_output_tokens=16)

    def command(self, kind, index=1, **fields):
        return {"protocol": contracts.PROTOCOL, "kind": kind, "worker_epoch": EPOCH,
                "command_id": f"44444444-4444-4444-8444-{index:012d}", **fields}

    def submit(self, **fields):
        defaults = {"request_id": REQUEST, "attempt": 1,
                    "model_revision": self.worker.model_revision,
                    "capability_revision": self.worker.capability_revision,
                    "token_ids": [3, 4], "max_tokens": 8, "remaining_timeout_ms": 10000,
                    "eos_token_ids": [2], "capacity_lease_ids": [LEASE]}
        defaults.update(fields)
        return self.command("submit", **defaults)

    def check_transcript(self, submit):
        tracker = contracts.AttemptTracker(submit)
        for event in self.events:
            if event["kind"] in contracts.ATTEMPT_EVENTS:
                tracker.observe(event)
        self.assertTrue(tracker.done)

    def test_ready_reports_real_profile_and_matching_identity(self):
        ready = self.worker.ready()
        contracts.validate_frame(ready)
        self.assertFalse(ready["capabilities"]["exact_append"])
        self.assertEqual(ready["capabilities"]["max_active_sequences"], 1)

    def test_success_streams_before_terminal_and_settles_after_terminal(self):
        submit = self.submit()
        self.worker.handle(submit)
        self.check_transcript(submit)
        self.assertEqual([frame["kind"] for frame in self.events][-2:], ["terminal", "resources_released"])
        self.assertEqual([token for frame in self.events if frame["kind"] == "tokens"
                          for token in frame["token_ids"]], [10, 11, 2])
        self.assertEqual(self.events[-2]["cause"], "eos")
        self.assertEqual(self.events[-2]["state_result"]["consumed_position"], 5)
        self.assertEqual(self.runtime.settled, 1)

    def test_zero_output_does_not_evaluate_model(self):
        submit = self.submit(max_tokens=0)
        self.worker.handle(submit)
        self.check_transcript(submit)
        self.assertEqual(self.runtime.generated, 0)
        self.assertEqual(self.events[-2]["metrics"]["ttft_ns"], 0)

    def test_context_rejection_returns_assigned_lease_without_model_mutation(self):
        submit = self.submit(token_ids=[3] * 60)
        self.worker.handle(submit)
        self.check_transcript(submit)
        self.assertEqual(self.events[0]["kind"], "rejected")
        self.assertEqual(self.events[-2]["cause"], "context_limit")
        self.assertEqual(self.runtime.generated, 0)
        self.assertEqual(self.events[-1]["released_lease_ids"], [LEASE])

    def test_control_reader_can_cancel_at_prefill_boundary(self):
        cancel = self.command("cancel_request", 2, request_id=REQUEST, attempt=1, reason="user_cancel")

        def arrive(consumed):
            if consumed == 1:
                thread = threading.Thread(target=self.worker.read_commands,
                                          args=(io.BytesIO(contracts.encode_frame(cancel)),))
                thread.start()
                thread.join(1)
                self.assertFalse(thread.is_alive())
        self.runtime.on_progress = arrive
        submit = self.submit()
        self.worker.handle(submit)
        self.check_transcript(submit)
        self.assertFalse(any(event["kind"] == "tokens" for event in self.events))
        self.assertEqual(self.events[-2]["status"], "cancelled")
        self.assertEqual(self.events[-2]["state_result"]["consumed_position"], 1)
        self.assertEqual(self.runtime.settled, 1)

    def test_deadline_stops_before_first_output(self):
        self.runtime.on_progress = lambda consumed: time.sleep(0.003)
        submit = self.submit(remaining_timeout_ms=1)
        self.worker.handle(submit)
        self.check_transcript(submit)
        self.assertEqual(self.events[-2]["status"], "expired")
        self.assertEqual(self.events[-2]["cause"], "deadline_exceeded")

    def test_duplicate_submit_does_not_generate_twice(self):
        submit = self.submit()
        self.worker.handle(submit)
        count = len(self.events)
        duplicate = dict(submit, command_id=self.command("query_stats", 9)["command_id"])
        self.worker.handle(duplicate)
        self.assertEqual(len(self.events), count + 1)
        self.assertEqual(self.events[-1]["status"], "already_registered")
        self.assertEqual(self.runtime.generated, 1)

    def test_late_cancel_cannot_change_terminal_result(self):
        self.worker.handle(self.submit())
        self.worker.handle(self.command("cancel_request", 2, request_id=REQUEST, attempt=1, reason="user_cancel"))
        self.assertEqual(self.events[-1]["status"], "already_terminal")
        self.assertEqual(sum(event["kind"] == "terminal" for event in self.events), 1)

    def test_same_cancel_command_after_terminal_returns_already_terminal(self):
        cancel = self.command("cancel_request", 2, request_id=REQUEST, attempt=1, reason="user_cancel")
        self.worker.controls.put(cancel)
        submit = self.submit()
        self.worker.handle(submit)
        self.check_transcript(submit)
        self.worker.handle(cancel)
        self.assertEqual(self.events[-1]["kind"], "command_result")
        self.assertEqual(self.events[-1]["status"], "already_terminal")
        self.assertEqual(sum(frame["kind"] == "terminal" for frame in self.events), 1)
        self.assertEqual(sum(frame["kind"] == "resources_released" for frame in self.events), 1)

    def test_settlement_failure_does_not_return_capacity(self):
        self.runtime.settle = lambda: (_ for _ in ()).throw(RuntimeError("test device failure"))
        with self.assertRaises(worker_module.ChannelFailure):
            self.worker.handle(self.submit())
        self.assertEqual(self.events[-1]["kind"], "terminal")
        self.assertFalse(any(event["kind"] == "resources_released" for event in self.events))

    def test_prepare_passes_tools_and_template_options_without_rewriting(self):
        command = self.command("prepare_input", model_revision=self.worker.model_revision,
                               messages=[{"role": "user", "content": "原样输入"}],
                               tools=[{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
                               template_options={"enable_thinking": False})
        self.worker.handle(command)
        self.assertEqual(self.runtime.prepared, command)
        self.assertEqual(self.events[-1]["kind"], "prepared_input")
        self.assertEqual(self.events[-1]["prompt_digest"], contracts.token_prefix_digest([3, 4]))

    def test_output_batching_preserves_first_token_and_token_ranges(self):
        self.worker.output_batch_tokens = 3
        self.runtime.tokens = [10, 11, 12, 13]
        self.runtime.final_tail = "尾"
        submit = self.submit(max_tokens=4)
        self.worker.handle(submit)
        self.check_transcript(submit)
        chunks = [event for event in self.events if event["kind"] == "tokens"]
        self.assertEqual([chunk["token_ids"] for chunk in chunks], [[10], [11, 12, 13]])
        self.assertEqual(self.events[-2]["text_delta"], "尾")

    def test_model_failure_reports_buffered_tokens_before_decoder_tail(self):
        for failure in (RuntimeError("The original model probe failed."),
                        worker_module.WorkerFailure("internal_error", "Model execution failed.")):
            with self.subTest(failure=type(failure).__name__):
                self.setUp()
                runtime = BufferedDecodeRuntime(model_failure=failure)
                self.worker.runtime = runtime
                self.worker.output_batch_tokens = 4
                submit = self.submit()
                self.worker.handle(submit)
                self.check_transcript(submit)
                chunks = [event for event in self.events if event["kind"] == "tokens"]
                self.assertEqual([chunk["token_ids"] for chunk in chunks], [[10], [11]])
                terminal = self.events[-2]
                self.assertEqual(terminal["status"], "failed")
                self.assertEqual(terminal["usage"]["output_tokens"], 2)
                self.assertEqual("".join(chunk["text_delta"] for chunk in chunks)
                                 + terminal["text_delta"], "A\ufffd")
                self.assertEqual(runtime.finalized, 1)
                self.assertEqual(runtime.settled, 1)

    def test_decoder_failure_excludes_failed_token_and_does_not_finalize_mutated_decoder(self):
        for failed_token, expected_tokens in ((10, []), (12, [10, 11])):
            with self.subTest(failed_token=failed_token):
                self.setUp()
                runtime = BufferedDecodeRuntime(tokens=(10, 11, 12), decode_failure=failed_token)
                self.worker.runtime = runtime
                self.worker.output_batch_tokens = 4
                submit = self.submit()
                self.worker.handle(submit)
                self.check_transcript(submit)
                chunks = [event for event in self.events if event["kind"] == "tokens"]
                self.assertEqual([token for chunk in chunks for token in chunk["token_ids"]], expected_tokens)
                self.assertEqual("".join(chunk["text_delta"] for chunk in chunks), "A" if expected_tokens else "")
                terminal = self.events[-2]
                self.assertEqual(terminal["status"], "failed")
                self.assertEqual(terminal["usage"]["output_tokens"], len(expected_tokens))
                self.assertEqual(terminal["text_delta"], "")
                self.assertEqual(runtime.finalized, 0)
                self.assertEqual(runtime.settled, 1)

    def test_buffered_output_rejection_preserves_failure_and_suppresses_decoder_tail(self):
        for failure in ("model", "decoder", "none"):
            with self.subTest(failure=failure):
                self.setUp()
                runtime = BufferedDecodeRuntime(
                    tokens=(10, 11) if failure == "model" else (10, 11, 12, 13),
                    model_failure=RuntimeError("The original model probe failed.") if failure == "model" else None,
                    decode_failure=12 if failure == "decoder" else None,
                )
                self.worker.runtime = runtime
                self.worker.output_batch_tokens = 4
                original_emit = self.worker.emit
                rejected = []

                def emit(frame, critical=False):
                    if frame["kind"] == "tokens" and frame["output_index"] > 0:
                        rejected.append(frame)
                        raise worker_module.WorkerFailure("slow_consumer", "The output queue is full.")
                    original_emit(frame, critical=critical)

                self.worker.emit = emit
                submit = self.submit(max_tokens=4)
                self.worker.handle(submit)
                self.check_transcript(submit)
                self.assertEqual(len(rejected), 1)
                terminal = self.events[-2]
                self.assertEqual(terminal["status"], "cancelled" if failure == "none" else "failed")
                self.assertEqual(terminal["cause"], "slow_consumer" if failure == "none" else "internal_error")
                self.assertEqual(terminal["usage"]["output_tokens"], 1)
                self.assertEqual(terminal["text_delta"], "")
                self.assertEqual(runtime.finalized, 0)
                self.assertEqual(runtime.settled, 1)

    def test_failure_flush_channel_loss_does_not_report_resource_release(self):
        runtime = BufferedDecodeRuntime(model_failure=RuntimeError("The original model probe failed."))
        self.worker.runtime = runtime
        self.worker.output_batch_tokens = 4
        original_emit = self.worker.emit

        def emit(frame, critical=False):
            if frame["kind"] == "tokens" and frame["output_index"] > 0:
                raise worker_module.ChannelFailure("The output pipe closed.")
            original_emit(frame, critical=critical)

        self.worker.emit = emit
        with self.assertRaises(worker_module.ChannelFailure):
            self.worker.handle(self.submit())
        self.assertFalse(any(event["kind"] in {"terminal", "resources_released"} for event in self.events))
        self.assertEqual(runtime.finalized, 0)
        self.assertEqual(runtime.settled, 0)

    def test_reader_rejects_incomplete_frame(self):
        self.worker.read_commands(io.BytesIO(contracts.encode_frame(self.submit()).rstrip(b"\n")))
        self.assertIsNotNone(self.worker.input_fault)
        self.assertTrue(self.worker.commands.empty())

    def test_reader_stops_after_valid_shutdown(self):
        shutdown = self.command("shutdown")
        first = contracts.encode_frame(shutdown)
        source = io.BytesIO(first + contracts.encode_frame(self.command("query_stats", 2)))
        self.worker.read_commands(source)
        self.assertEqual(source.tell(), len(first))
        self.assertEqual(self.worker.controls.get_nowait(), shutdown)
        self.assertTrue(self.worker.controls.empty())
        self.assertTrue(self.worker.input_done.is_set())
        self.assertIsNone(self.worker.input_fault)

    def test_shutdown_during_prefill_does_not_cancel_accepted_work(self):
        shutdown = self.command("shutdown", 2)

        def arrive(consumed):
            if consumed == 1:
                reader = threading.Thread(target=self.worker.read_commands,
                                          args=(io.BytesIO(contracts.encode_frame(shutdown)),))
                reader.start()
                reader.join(1)
                self.assertFalse(reader.is_alive())

        self.runtime.on_progress = arrive
        submit = self.submit()
        self.worker.handle(submit)
        self.check_transcript(submit)
        self.assertTrue(self.worker.shutdown_requested)
        self.assertTrue(self.worker.input_done.is_set())
        self.assertEqual(self.events[-2]["status"], "completed")
        self.assertEqual(self.events[-2]["cause"], "eos")
        self.assertEqual(self.runtime.settled, 1)

    def test_invalid_request_payload_settles_valid_envelope(self):
        invalid = self.submit(max_tokens=True)
        self.worker.read_commands(io.BytesIO(json.dumps(invalid).encode() + b"\n"))
        self.assertIsNone(self.worker.input_fault)
        self.worker.handle(self.worker.commands.get_nowait())
        self.assertEqual([frame["kind"] for frame in self.events],
                         ["rejected", "terminal", "resources_released"])
        self.assertEqual(self.runtime.generated, 0)
        self.assertEqual(self.events[-1]["released_lease_ids"], [LEASE])

    def test_stop_generation_preserves_completion_cause(self):
        original_emit = self.worker.emit

        def emit(frame, critical=False):
            original_emit(frame, critical=critical)
            if frame["kind"] == "tokens" and frame["output_index"] == 0:
                self.worker.controls.put(self.command(
                    "stop_generation", 2, request_id=REQUEST, attempt=1,
                    cause="stop_sequence", output_token_count=1, text_byte_cutoff=1))
        self.worker.emit = emit
        submit = self.submit()
        self.worker.handle(submit)
        self.check_transcript(submit)
        self.assertEqual(self.events[-2]["status"], "completed")
        self.assertEqual(self.events[-2]["cause"], "stop_sequence")
        self.assertEqual(self.events[-2]["usage"]["output_tokens"], 1)

    def test_slow_output_cancels_without_false_text_tail(self):
        original_emit = self.worker.emit

        def emit(frame, critical=False):
            if frame["kind"] == "tokens":
                raise worker_module.WorkerFailure("slow_consumer", "The output queue is full.")
            original_emit(frame, critical=critical)
        self.worker.emit = emit
        self.runtime.final_tail = "must not leak unreported text"
        submit = self.submit()
        self.worker.handle(submit)
        self.check_transcript(submit)
        self.assertEqual(self.events[-2]["cause"], "slow_consumer")
        self.assertEqual(self.events[-2]["text_delta"], "")
        self.assertEqual(self.events[-2]["usage"]["output_tokens"], 0)

    def test_completed_records_do_not_retain_prompt(self):
        self.worker.handle(self.submit())
        attempt = self.worker.attempts[(REQUEST, 1)]
        self.assertEqual(set(attempt.command), {"request_id", "attempt"})
        record = next(iter(self.worker.command_records.values()))
        self.assertEqual(len(record[0]), 32)

    def test_outbox_reserves_room_for_terminal_frames(self):
        release = threading.Event()
        entered = threading.Event()

        def sink(payload):
            entered.set()
            release.wait(1)
        outbox = worker_module.ByteOutbox(sink, limit=worker_module.MAX_FRAME_BYTES + 1024)
        frame = self.worker.frame("tokens", request_id=REQUEST, attempt=1,
                                  event_seq=0, output_index=0, token_ids=[10], text_delta="x" * 500)
        try:
            outbox.put(frame)
            self.assertTrue(entered.wait(1))
            with self.assertRaises(worker_module.WorkerFailure):
                outbox.put(frame)
            outbox.put(self.worker.frame("drained"), critical=True)
        finally:
            release.set()
            outbox.close()

    def test_manifest_binds_hf_symlink_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bundle = root / "bundle"
            bundle.mkdir()
            (root / "blob").write_bytes(b"original model bytes")
            (bundle / "weights.bin").symlink_to(root / "blob")
            manifest = worker_module.model_manifest(bundle, dict(worker_module.PINNED_RUNTIME), {"profile": "test"})
            self.assertEqual(manifest["artifacts"][0]["path"], "weights.bin")
            self.assertEqual(manifest["artifacts"][0]["size_bytes"], 20)


if __name__ == "__main__":
    unittest.main()

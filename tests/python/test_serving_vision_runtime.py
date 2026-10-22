import copy
import gc
import json
from pathlib import Path
import threading
import unittest

import numpy as np

from test_serving_vision_preparation import MEDIA, Processor
from vision_preparation import PreparationIdentity, VisionPreparation
from vision_runtime import VisionRuntime
from text_worker import ChannelFailure, GenerationStopped, WorkerFailure
from vision_worker import VisionWorker
import contracts


class Device:
    def __init__(self):
        self.array_calls = 0
        self.synchronized = []

    def default_device(self):
        return 'gpu'

    def default_stream(self, device):
        return 'request'

    def reset_peak_memory(self):
        pass

    def array(self, value):
        self.array_calls += 1
        return np.array(value)

    def synchronize(self, stream):
        self.synchronized.append(stream)


class State:
    fenced = False

    def begin(self):
        pass

    def restore(self):
        pass

    def fence(self):
        self.fenced = True


class VisionRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.processor = Processor()
        selected = PreparationIdentity('cd000000-0000-4000-8000-000000000001', 'b' * 64, 'c' * 64)
        self.runtime = VisionRuntime.__new__(VisionRuntime)
        self.runtime.owner = threading.get_ident()
        self.runtime._request_state = State()
        self.runtime.processor = self.processor
        self.runtime.mx = Device()
        self.runtime.model = object()
        self.runtime.consumed_position = None
        self.runtime._generation_stream = 'generation'
        self.runtime._request_stream = None
        self.runtime._load_stream = 'load'
        self.runtime._decode_stream_class = lambda **options: object()
        self.runtime._preparation = VisionPreparation(self.processor, 'a' * 64, selected)
        command = dict(worker_epoch=selected.worker_epoch, model_revision=selected.model_revision,
                       command_id='cd000000-0000-4000-8000-000000000002', tools=[],
                       messages=[dict(role='user', content=[dict(type='image', media=copy.deepcopy(MEDIA))])])
        prepared = self.runtime.prepare(command)
        self.submit = dict(**selected.fields(), token_ids=prepared['token_ids'], replay=prepared['replay'],
                           max_tokens=2, sampling=dict(mode='random', temperature=0.7, top_p=0.9, seed=(1 << 53) - 1))
        self.library_closed = False
        self.controls = None

        def generate(ids, model, pixels, mask, **options):
            self.controls = options
            try:
                yield 4, None
                yield 248044, None
            finally:
                self.library_closed = True

        self.runtime.generate_step = generate

    def test_replay_precedes_transfer_and_preserves_selection_and_metadata(self):
        observed = []

        def progress(consumed):
            self.assertEqual(self.runtime.mx.array_calls, 0)
            observed.append(consumed)

        self.assertEqual(list(self.runtime.generate(self.submit, progress)), [4, 248044])
        self.assertEqual(observed, [0])
        self.assertEqual(self.controls['seed'], (1 << 53) - 1)
        self.assertEqual(self.controls['temperature'], 0.7)
        self.assertEqual(self.controls['top_k'], 100)
        self.assertEqual(self.controls['repetition_penalty'], 1.03)
        for field in ('mm_token_type_ids', 'tgt_sizes', 'image_bound', 'num_patches_per_image', 'downsample_mode'):
            self.assertIn(field, self.controls)
        self.assertTrue(self.library_closed)
        self.assertIsNone(self.runtime.consumed_position)
        self.assertEqual(self.runtime.mx.synchronized, ['generation', 'request', 'load'])

    def test_changed_arrays_never_reach_device_or_library(self):
        self.processor.changed_mask = True
        with self.assertRaises(WorkerFailure) as failure:
            list(self.runtime.generate(self.submit, lambda consumed: self.fail('Invalid replay reached progress')))
        self.assertEqual(failure.exception.code, 'invalid_request')
        self.assertEqual(self.runtime.mx.array_calls, 0)
        self.assertIsNone(self.controls)
        self.assertGreater(self.runtime.peak_host_array_bytes, 0)

    def test_close_during_output_closes_library_and_synchronizes(self):
        generator = self.runtime.generate(self.submit, lambda consumed: None)
        self.assertEqual(next(generator), 4)
        generator.close()
        self.controls = None
        gc.collect()
        self.assertTrue(self.library_closed)
        self.assertEqual(self.runtime.mx.synchronized, ['generation', 'request', 'load'])
        self.assertFalse(any(reference() is not None for reference in self.processor.array_refs))

    def test_cancel_before_transfer_releases_host_arrays(self):
        def cancel(consumed):
            raise GenerationStopped()

        with self.assertRaises(GenerationStopped):
            list(self.runtime.generate(self.submit, cancel))
        gc.collect()
        self.assertEqual(self.runtime.mx.array_calls, 0)
        self.assertFalse(any(reference() is not None for reference in self.processor.array_refs))
        self.assertEqual(self.runtime.mx.synchronized, ['generation', 'request', 'load'])

    def test_library_failure_before_yield_synchronizes(self):
        def fail(*args, **kwargs):
            raise RuntimeError('Original dependency failure')

        self.runtime.generate_step = fail
        with self.assertRaisesRegex(RuntimeError, 'Original dependency failure'):
            list(self.runtime.generate(self.submit, lambda consumed: None))
        gc.collect()
        self.assertEqual(self.runtime.mx.synchronized, ['generation', 'request', 'load'])
        self.assertFalse(any(reference() is not None for reference in self.processor.array_refs))

    def test_synchronization_failure_fences_all_runtime_entry_points(self):
        def fail(stream):
            raise RuntimeError('Uncertain stream completion')

        self.runtime.mx.synchronize = fail
        with self.assertRaisesRegex(RuntimeError, 'Uncertain stream completion'):
            self.runtime.settle()
        self.assertTrue(self.runtime._request_state.fenced)
        with self.assertRaises(RuntimeError):
            self.runtime.prepare({})
        with self.assertRaises(RuntimeError):
            next(self.runtime.generate(self.submit, lambda consumed: None))

    def test_restore_failure_fences_runtime(self):
        def fail():
            raise RuntimeError('Unqualified model state')

        self.runtime._request_state.restore = fail
        with self.assertRaisesRegex(RuntimeError, 'Unqualified model state'):
            self.runtime.settle()
        self.assertTrue(self.runtime._request_state.fenced)

    def test_one_shot_generator_cleanup_failure_never_releases_or_reuses_worker(self):
        corpus = json.loads((Path(__file__).resolve().parents[2]
                             / 'tests/fixtures/serving/minicpm-vision-v0.1.json').read_text())
        for boundary in ('exhaustion', 'before_first_yield'):
            with self.subTest(boundary=boundary):
                self.setUp()
                runtime = self.runtime
                runtime.max_context = 16384
                runtime.vocab_size = 248057
                runtime.eos_token_ids = [2]
                runtime.model_path = '/models/vision-fixture'
                runtime.runtime = copy.deepcopy(corpus['templates']['ready']['runtime'])
                runtime.manifest = copy.deepcopy(corpus['templates']['ready']['model_manifest'])
                runtime._preparation = None
                runtime.decode = lambda token, eos=False: ''
                runtime.finish_text = lambda: ''
                runtime.memory = lambda: dict(active_bytes=10, peak_bytes=20, cache_bytes=0)
                runtime.mx.clear_cache = lambda: None
                calls = []

                def synchronize(stream):
                    calls.append(stream)
                    if len(calls) == 1:
                        raise RuntimeError('One-shot stream uncertainty')

                def fail_before_yield(*args, **kwargs):
                    raise RuntimeError('Dependency failed before its first yield')

                runtime.mx.synchronize = synchronize
                if boundary == 'before_first_yield':
                    runtime.generate_step = fail_before_yield
                events = []

                def emit(frame, **options):
                    contracts.validate_frame(frame)
                    events.append(copy.deepcopy(frame))

                worker = VisionWorker(runtime, corpus['templates']['ready']['worker_epoch'],
                                      emit, max_output_tokens=8192)
                prepare = copy.deepcopy(corpus['templates']['prepare'])
                prepare['model_revision'] = worker.model_revision
                worker.handle(prepare)
                prepared = events[-1]
                self.assertEqual(prepared['kind'], 'prepared_input')
                submit = copy.deepcopy(corpus['templates']['submit'])
                submit.update(model_revision=worker.model_revision,
                              capability_revision=worker.capability_revision,
                              token_ids=prepared['token_ids'], replay=prepared['replay'])
                with self.assertRaises(ChannelFailure):
                    worker.handle(submit)
                self.assertTrue(runtime._request_state.fenced)
                self.assertEqual([event['kind'] for event in events].count('terminal'), 1)
                self.assertNotIn('resources_released', [event['kind'] for event in events])
                with self.assertRaises(RuntimeError):
                    runtime._check_owner()


if __name__ == '__main__':
    unittest.main()

import copy
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python/apxinf/apxinf/serving'))
import contracts
from text_worker import ChannelFailure, InvalidCommand
from vision_worker import VisionWorker

CORPUS = json.loads((ROOT / 'tests/fixtures/serving/minicpm-vision-v0.1.json').read_text())


class Runtime:
    max_context = 16384
    vocab_size = 248057
    eos_token_ids = [2]
    consumed_position = None
    model_path = '/models/vision-fixture'

    def __init__(self):
        self.runtime = copy.deepcopy(CORPUS['templates']['ready']['runtime'])
        self.manifest = copy.deepcopy(CORPUS['templates']['ready']['model_manifest'])
        self.peak_host_array_bytes = 0
        self.generations = 0
        self.order = []
        self.before_progress = lambda: None

    def activate(self, identity):
        self.identity = identity

    def prepare(self, command):
        result = copy.deepcopy(CORPUS['templates']['prepared'])
        replay = result['replay']
        replay.update(self.identity.fields(), preparation_id=command['command_id'])
        return {key: result[key] for key in ('token_ids', 'effective_input_tokens', 'prompt_digest', 'replay')}

    def generate(self, command, progress):
        self.generations += 1
        self.peak_host_array_bytes = command['replay']['host_array_bytes']
        try:
            self.before_progress()
            progress(0)
            yield 4
            yield 2
        finally:
            self.order.append('generator_closed')

    def decode(self, token, eos=False):
        return '' if eos else 'red'

    def finish_text(self):
        return ''

    def memory(self):
        return dict(active_bytes=10, peak_bytes=20, cache_bytes=0)

    def settle(self):
        self.order.append('settled')


class VisionWorkerTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Runtime()
        self.events = []

        def emit(frame, **options):
            contracts.validate_frame(frame)
            if frame['kind'] == 'resources_released':
                self.runtime.order.append('released')
            self.events.append(copy.deepcopy(frame))

        self.worker = VisionWorker(self.runtime, CORPUS['templates']['ready']['worker_epoch'],
                                   emit, max_output_tokens=8192)

    def prepare_and_submit(self, max_tokens=2):
        command = copy.deepcopy(CORPUS['templates']['prepare'])
        command['model_revision'] = self.worker.model_revision
        self.worker.handle(command)
        prepared = self.events[-1]
        self.assertEqual(prepared['kind'], 'prepared_input')
        submit = copy.deepcopy(CORPUS['templates']['submit'])
        submit.update(model_revision=self.worker.model_revision,
                      capability_revision=self.worker.capability_revision, max_tokens=max_tokens,
                      token_ids=prepared['token_ids'], replay=prepared['replay'])
        return submit

    def check_stream(self, submit):
        tracker = contracts.AttemptTracker(submit)
        for frame in self.events:
            if frame['kind'] in contracts.ATTEMPT_EVENTS:
                tracker.observe(frame)
        self.assertTrue(tracker.done)
        return next(frame for frame in self.events if frame['kind'] == 'terminal')

    def test_ready_and_stream_keep_media_and_settlement_accounting(self):
        contracts.validate_frame(self.worker.ready())
        submit = self.prepare_and_submit()
        self.worker.handle(submit)
        terminal = self.check_stream(submit)
        self.assertEqual(terminal['metrics']['source_media_bytes'], 69)
        self.assertEqual(terminal['metrics']['peak_host_array_bytes'], 2376)
        self.assertIsNone(terminal['state_result']['consumed_position'])
        self.assertEqual(terminal['usage']['output_tokens'], 2)
        self.assertEqual(self.runtime.order, ['generator_closed', 'settled', 'released'])
        self.worker.handle(submit)
        self.assertEqual(self.runtime.generations, 1)

    def test_zero_output_never_generates_or_reuses_prior_metrics(self):
        self.runtime.peak_host_array_bytes = 12345
        submit = self.prepare_and_submit(0)
        self.worker.handle(submit)
        terminal = self.check_stream(submit)
        self.assertEqual(self.runtime.generations, 0)
        self.assertEqual(terminal['metrics']['source_media_bytes'], 69)
        self.assertEqual(terminal['metrics']['peak_host_array_bytes'], 0)
        self.assertIsNone(terminal['state_result']['consumed_position'])
        self.assertEqual(self.runtime.order, ['released'])

    def test_cancel_before_first_device_operation_settles(self):
        submit = self.prepare_and_submit()
        cancel = copy.deepcopy(CORPUS['templates']['cancel'])
        self.runtime.before_progress = lambda: self.worker.controls.put_nowait(cancel)
        self.worker.handle(submit)
        terminal = self.check_stream(submit)
        self.assertEqual(terminal['status'], 'cancelled')
        self.assertEqual(terminal['usage']['output_tokens'], 0)
        self.assertEqual(self.runtime.order, ['generator_closed', 'settled', 'released'])

    def test_invalid_replay_still_has_one_terminal_and_release(self):
        command = copy.deepcopy(CORPUS['templates']['submit'])
        command['replay'] = None
        self.worker.handle(InvalidCommand(command))
        kinds = [frame['kind'] for frame in self.events]
        self.assertEqual(kinds, ['rejected', 'terminal', 'resources_released'])
        self.assertEqual(self.events[1]['metrics']['source_media_bytes'], 0)
        self.assertEqual(self.runtime.generations, 0)

    def test_reader_selects_the_vision_protocol(self):
        command = copy.deepcopy(CORPUS['templates']['prepare'])
        command['model_revision'] = self.worker.model_revision
        self.worker.read_commands(BytesIO(contracts.encode_frame(command)))
        self.assertIsNone(self.worker.input_fault)
        self.assertEqual(self.worker.commands.get_nowait(), command)

    def test_uncertain_settlement_never_returns_capacity(self):
        submit = self.prepare_and_submit()

        def fail():
            raise RuntimeError('Uncertain model request state')

        self.runtime.settle = fail
        with self.assertRaises(ChannelFailure):
            self.worker.handle(submit)
        kinds = [frame['kind'] for frame in self.events]
        self.assertEqual(kinds.count('terminal'), 1)
        self.assertNotIn('resources_released', kinds)


if __name__ == '__main__':
    unittest.main()

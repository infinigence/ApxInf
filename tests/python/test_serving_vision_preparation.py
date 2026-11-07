import copy
import gc
import hashlib
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import weakref

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python/apxinf/apxinf/serving'))
import vision_preparation as preparation
from vision_preparation import PreparationError, PreparationIdentity, VisionPreparation, typed_output

MEDIA = json.loads((ROOT / 'tests/fixtures/serving/owned-media-v0.1.json').read_text())['images']['rgb']


class Processor:
    def __init__(self):
        self.tokenizer = self
        self.template_calls = 0
        self.changed_mask = False
        self.array_refs = []

    def apply_chat_template(self, messages, **options):
        self.template_calls += 1
        return 'Original prompt'

    def __call__(self, text, images, **options):
        tokens = [10]
        bounds = []
        pixels = []
        for image in images:
            start = len(tokens)
            tokens.extend([248056, 248056, 20])
            bounds.append([start, start + 2])
            pixels.append(np.full((3, 14, 56), image.getpixel((0, 0))[0], dtype=np.float32))
        ids = np.array([tokens], dtype=np.int32)
        mask = np.ones_like(ids)
        if self.changed_mask:
            mask[0, 0] = 0
        values = dict(input_ids=ids, attention_mask=mask, mm_token_type_ids=np.zeros_like(ids),
                      pixel_values=[pixels], image_bound=[np.array(bounds, dtype=np.int32).reshape(-1, 2)],
                      tgt_sizes=[np.array([[2, 2]] * len(images), dtype=np.int32).reshape(-1, 2)],
                      downsample_mode='16x', num_patches_per_image=[[1] * len(images)])
        self.array_refs = [weakref.ref(ids), weakref.ref(mask)] + [weakref.ref(item) for item in pixels]
        return values


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.processor = Processor()
        self.identity = PreparationIdentity('cd000000-0000-4000-8000-000000000001', 'b' * 64, 'c' * 64)
        self.adapter = VisionPreparation(self.processor, 'a' * 64, self.identity)
        self.command = dict(worker_epoch=self.identity.worker_epoch, model_revision=self.identity.model_revision,
                            command_id='cd000000-0000-4000-8000-000000000002', tools=[],
                            messages=[dict(role='user', content=[dict(type='text', text='Inspect.'),
                                      dict(type='image', media=copy.deepcopy(MEDIA))])])

    def submission(self):
        result = self.adapter.prepare(self.command)
        return dict(**self.identity.fields(), token_ids=result['token_ids'], replay=result['replay'])

    def test_replay_keeps_template_once_and_disposes_arrays(self):
        before = copy.deepcopy(self.command)
        submit = self.submission()
        self.assertEqual(self.command, before)
        gc.collect()
        self.assertFalse(any(reference() is not None for reference in self.processor.array_refs))
        with self.adapter.replay(submit) as host:
            self.assertEqual(host.token_ids, submit['token_ids'])
            references = list(self.processor.array_refs)
        self.assertIsNone(host.values)
        gc.collect()
        self.assertFalse(any(reference() is not None for reference in references))
        self.assertEqual(self.processor.template_calls, 1)

    def test_all_processor_arrays_affect_replay(self):
        submit = self.submission()
        self.processor.changed_mask = True
        with self.assertRaises(PreparationError):
            with self.adapter.replay(submit):
                self.fail('Changed attention mask reached generation')

    def test_context_limit_fails_before_replay(self):
        adapter = VisionPreparation(self.processor, 'a' * 64, self.identity, max_context=3)
        with self.assertRaises(PreparationError) as failure:
            adapter.prepare(self.command)
        self.assertEqual(failure.exception.code, 'context_limit')

    def test_epoch_processor_and_host_count_changes_fail(self):
        submit = self.submission()
        for key, value in [('worker_epoch', 'cd000000-0000-4000-8000-000000000099'),
                           ('model_revision', 'd' * 64), ('capability_revision', 'd' * 64),
                           ('processor_revision', 'd' * 64), ('host_array_bytes', 0)]:
            changed = copy.deepcopy(submit)
            changed['replay'][key] = value
            with self.subTest(key), self.assertRaises(PreparationError):
                with self.adapter.replay(changed):
                    self.fail('Changed preparation reached generation')

    def test_repeated_images_retain_distinct_bindings(self):
        self.command['messages'][0]['content'].extend([copy.deepcopy(self.command['messages'][0]['content'][1])] * 3)
        submit = self.submission()
        self.assertEqual(len(submit['replay']['media']), 4)
        self.assertEqual([row['source_index'] for row in submit['replay']['image_bindings']], [0, 1, 2, 3])
        self.assertEqual(sum(row['byte_count'] for row in submit['replay']['media']), 276)
        self.command['messages'][0]['content'].append(copy.deepcopy(self.command['messages'][0]['content'][1]))
        with self.assertRaises(ValueError):
            self.adapter.prepare(self.command)

    def test_shared_storage_counts_once_and_bytes_are_little_endian(self):
        values = np.array([1, 2, 3, 4], dtype='>i4')
        represented, size = typed_output(dict(full=values, view=values[1:]))
        self.assertEqual(size, 16)
        self.assertEqual(represented['items']['full']['dtype'], '<i4')
        expected = hashlib.sha256(np.array([1, 2, 3, 4], dtype='<i4').tobytes()).hexdigest()
        self.assertEqual(represented['items']['full']['sha256'], expected)

    def test_unsupported_storage_nonfinite_and_budget_fail(self):
        for values in [np.frombuffer(bytearray(16), dtype=np.int32), np.array([np.nan]),
                       np.zeros((2, 3), dtype=np.float32).T, object()]:
            with self.subTest(type(values).__name__), self.assertRaises(PreparationError):
                typed_output(values)
        with patch.object(preparation, 'MAX_HOST_ARRAY_BYTES', 3), self.assertRaises(PreparationError):
            typed_output(np.zeros(1, dtype=np.int32))


if __name__ == '__main__':
    unittest.main()

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'python/apxinf/apxinf/serving'))
from vision_runtime import _RequestField, _ResidentRequestState
from text_worker import WorkerFailure

try:
    import mlx.core as mx
    import mlx.nn as nn
except ImportError:
    mx = nn = None


@unittest.skipIf(mx is None, 'MLX is unavailable for public module-state checks.')
class VisionStateTests(unittest.TestCase):
    def setUp(self):
        previous_device = mx.default_device()
        mx.set_default_device(mx.cpu)
        self.addCleanup(mx.set_default_device, previous_device)
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.language = self.model.language_model
        self.language.weight = mx.array([[1, 2]], dtype=mx.float32)
        self.language._position_ids = None
        self.language._rope_deltas = None
        self.model.convolution = nn.Module()
        self.convolution = self.model.convolution
        self.convolution.conv1d = nn.Module()
        self.convolution.conv1d.weight = mx.zeros((6144, 4, 1), dtype=mx.bfloat16, stream=mx.cpu)
        self.model.rotary = nn.Module()
        self.rotary = self.model.rotary
        self.rotary._compiled_apply = {}
        fields = (_RequestField(self.language, '_position_ids', 'positions'),
                  _RequestField(self.language, '_rope_deltas', 'deltas'),
                  _RequestField(self.convolution, '_qwen3_5_decode_conv_weight', 'convolution'),
                  _RequestField(self.rotary, '_compiled_apply', 'rotary'))
        self.owner = _ResidentRequestState(self.model, mx, 16, fields)

    def positions(self):
        self.language._position_ids = mx.zeros((3, 1, 4), dtype=mx.int32, stream=mx.cpu)
        self.language._rope_deltas = mx.zeros((1, 1), dtype=mx.int32, stream=mx.cpu)

    def cache(self):
        self.convolution._qwen3_5_decode_conv_weight = (
            id(self.convolution.conv1d.weight), mx.ones((4, 6144), dtype=mx.float32, stream=mx.cpu))

    def test_repeated_restore_preserves_resident_arrays_and_readable_baselines(self):
        language_weight = self.language.weight
        convolution_weight = self.convolution.conv1d.weight
        compiled_cache = self.rotary._compiled_apply
        for _ in range(3):
            self.owner.begin()
            self.positions()
            self.cache()
            mx.synchronize(mx.cpu)
            self.owner.restore()
            self.assertIsNone(self.language._position_ids)
            self.assertIsNone(self.language._rope_deltas)
            self.assertNotIn('_position_ids', self.language.state)
            self.assertNotIn('_rope_deltas', self.language.state)
            self.assertFalse(hasattr(self.convolution, '_qwen3_5_decode_conv_weight'))
            self.assertIs(self.language.weight, language_weight)
            self.assertIs(self.convolution.conv1d.weight, convolution_weight)
            self.assertIs(self.rotary._compiled_apply, compiled_cache)
            self.assertEqual(compiled_cache, {})
            self.assertEqual(self.language.weight.tolist(), [[1, 2]])
            self.owner.restore()

    def test_partial_installation_and_cancel_before_mutation_restore(self):
        self.owner.begin()
        self.owner.restore()
        self.owner.begin()
        self.language._position_ids = mx.zeros((3, 1, 1), dtype=mx.int32, stream=mx.cpu)
        mx.synchronize(mx.cpu)
        self.owner.restore()
        self.assertIsNone(self.language._position_ids)
        self.assertIsNone(self.language._rope_deltas)

    def test_unknown_array_fences_before_any_restoration(self):
        self.owner.begin()
        self.positions()
        self.language.unqualified_array = mx.zeros((1,), stream=mx.cpu)
        mx.synchronize(mx.cpu)
        with self.assertRaises(RuntimeError):
            self.owner.restore()
        self.assertTrue(self.owner.fenced)
        self.assertIsNotNone(self.language._position_ids)
        with self.assertRaises(RuntimeError):
            self.owner.begin()

    def test_replaced_weight_fences(self):
        self.owner.begin()
        self.language.weight = mx.array([[1, 2]], dtype=mx.float32)
        mx.synchronize(mx.cpu)
        with self.assertRaises(RuntimeError):
            self.owner.restore()
        self.assertTrue(self.owner.fenced)

    def test_wrong_position_dtype_and_shape_fence(self):
        for value in (mx.zeros((3, 1, 4), dtype=mx.float32, stream=mx.cpu),
                      mx.zeros((3, 1, 17), dtype=mx.int32, stream=mx.cpu)):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                self.setUp()
                self.owner.begin()
                self.language._position_ids = value
                mx.synchronize(mx.cpu)
                with self.assertRaises(RuntimeError):
                    self.owner.restore()
                self.assertTrue(self.owner.fenced)

    def test_invalid_derived_cache_fences(self):
        self.owner.begin()
        self.convolution._qwen3_5_decode_conv_weight = (
            0, mx.zeros((4, 6144), dtype=mx.float32, stream=mx.cpu))
        mx.synchronize(mx.cpu)
        with self.assertRaises(RuntimeError):
            self.owner.restore()
        self.assertTrue(self.owner.fenced)

    def test_preexisting_request_array_refuses_baseline(self):
        self.positions()
        mx.synchronize(mx.cpu)
        with self.assertRaises(RuntimeError):
            _ResidentRequestState(self.model, mx, 16, self.owner.fields)

    def test_shadowed_none_refuses_baseline(self):
        self.language.state['_position_ids'] = None
        with self.assertRaises(RuntimeError):
            _ResidentRequestState(self.model, mx, 16, self.owner.fields)

    def test_qualification_refuses_unexpected_model_class(self):
        with self.assertRaises(WorkerFailure) as failure:
            _ResidentRequestState.qualify(self.model, mx, 16)
        self.assertEqual(failure.exception.code, 'model_unavailable')

    def test_overlapping_begin_permanently_fences(self):
        self.owner.begin()
        with self.assertRaises(RuntimeError):
            self.owner.begin()
        self.assertTrue(self.owner.fenced)
        with self.assertRaises(RuntimeError):
            self.owner.restore()

    def test_replaced_rotary_dictionary_fences(self):
        self.owner.begin()
        self.rotary._compiled_apply = {}
        with self.assertRaises(RuntimeError):
            self.owner.restore()
        self.assertTrue(self.owner.fenced)

    def test_unsupported_rotary_entry_fences_before_cleanup(self):
        for key, value in ((2, mx.compile(lambda item: item + 1)),
                           (3, mx.zeros((1,), stream=mx.cpu)),
                           (3, lambda item: item)):
            with self.subTest(key=key, value_type=type(value).__name__):
                self.setUp()
                self.owner.begin()
                self.positions()
                self.rotary._compiled_apply[key] = value
                with self.assertRaises(RuntimeError):
                    self.owner.restore()
                self.assertTrue(self.owner.fenced)
                self.assertIsNotNone(self.language._position_ids)

    def test_nonempty_rotary_startup_refuses_baseline(self):
        self.rotary._compiled_apply[3] = mx.compile(lambda value: value + 1)
        with self.assertRaises(RuntimeError):
            _ResidentRequestState(self.model, mx, 16, self.owner.fields)

    def test_pinned_rotary_compilation_cache_clears_without_replacing_dictionary(self):
        function = mx.compile(lambda value: value + 1)
        if type(function).__module__ != 'mlx.core' or type(function).__name__ != '_gc_func':
            self.skipTest('The pinned MLX compilation wrapper is unavailable.')
        cache = self.rotary._compiled_apply
        for _ in range(2):
            self.owner.begin()
            cache[3] = function
            self.owner.restore()
            self.assertIs(self.rotary._compiled_apply, cache)
            self.assertEqual(cache, {})


if __name__ == '__main__':
    unittest.main()

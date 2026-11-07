from dataclasses import dataclass
import math
from pathlib import Path
import platform
import struct
import sys
import threading
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python/apxinf/apxinf/serving"))
import audio_provider
from audio_contracts import AsrInput, AttemptKey, MediaGrant, TtsInput
from audio_wav import encode_pcm16, inspect


class Samples:
    def __init__(self, values, shape=None):
        self.values = values
        self.shape = shape or (len(values),)

    def astype(self, dtype):
        return Samples([int(v) if dtype == "<i2" else float(v) for v in self.values])

    def __truediv__(self, other):
        return Samples([v / other for v in self.values])

    def __mul__(self, other):
        return Samples([v * other for v in self.values])

    def tobytes(self):
        return struct.pack("<" + "h" * len(self.values), *self.values)


class PublicArrayApi:
    float32 = "float32"
    def __init__(self):
        self.copies = 0

    def frombuffer(self, audio, dtype, count, offset):
        assert dtype == "<i2"
        return Samples(list(struct.unpack_from("<" + "h" * count, audio, offset)))

    def asarray(self, audio, dtype):
        self.copies += 1
        return Samples(audio.values, audio.shape)

    def isfinite(self, samples):
        return types.SimpleNamespace(all=lambda: all(math.isfinite(v) for v in samples.values))

    def clip(self, samples, low, high):
        return Samples([max(low, min(high, v)) for v in samples.values])


@dataclass
class SttOutput:
    text: str


@dataclass
class GenerationResult:
    audio: Samples
    sample_rate: int


class PublicModel:
    def __init__(self, output):
        self.output = output
        self.calls = []
        self.tokenizer_error = None
        self.language_codes = ("en", "fr")
        self.is_multilingual = True
        self.tokenizer_calls = []

    def get_tokenizer(self, **kwargs):
        self.tokenizer_calls.append(kwargs)
        if self.tokenizer_error:
            raise self.tokenizer_error
        return types.SimpleNamespace(all_language_codes=self.language_codes)

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


class ProviderAdapter(unittest.TestCase):
    def adapter(self, model, family="whisper", languages=None, frontend_factory=None):
        PublicModel.__module__ = "mlx_audio.stt.models.whisper" if family == "whisper" else "mlx_audio.tts.models.kokoro"
        arrays = PublicArrayApi()
        device = types.SimpleNamespace(synchronize=lambda: None, clear_cache=lambda: None)
        loads = []
        loader = types.SimpleNamespace(load_model=lambda path, **kwargs: (loads.append((path,kwargs)),model)[1])
        modules = {"numpy":arrays, "mlx.core":device, "mlx_audio.stt.utils":loader, "mlx_audio.tts.utils":loader,
            "misaki.espeak":types.SimpleNamespace(EspeakFallback=lambda british: object()),
            "misaki.en":types.SimpleNamespace(G2P=frontend_factory or (lambda **kwargs: object()))}
        manifest = {"family":family,"model_dir":"/original/local/model","voices":{"heart":"/original/local/heart.safetensors"},
            "languages":languages or ["en"]}
        with patch.object(audio_provider,"verify_local_profile") as verify, patch.object(audio_provider.sys,"addaudithook"), patch.object(audio_provider.importlib,"import_module",side_effect=modules.__getitem__):
            adapter = audio_provider.MlxAudioProvider(manifest)
        verify.assert_called_once_with(manifest)
        self.assertEqual(loads,[(Path("/original/local/model"),{"lazy":False,"strict":True})])
        return adapter, arrays

    def test_missing_whisper_processor_fails_before_readiness(self):
        model = PublicModel(SttOutput("unreachable"))
        model.tokenizer_error = ValueError("Processor not found")
        with self.assertRaisesRegex(ValueError, "Processor not found"):
            self.adapter(model)

    def test_kokoro_frontend_failure_prevents_readiness(self):
        def unavailable_frontend(**kwargs):
            raise ValueError("English tokenizer cannot load")
        with self.assertRaisesRegex(ValueError, "English tokenizer cannot load"):
            self.adapter(PublicModel(iter([])), "kokoro", frontend_factory=unavailable_frontend)

    def test_whisper_declared_languages_require_model_support(self):
        model = PublicModel(SttOutput("unreachable"))
        with self.assertRaisesRegex(ValueError, "language"):
            self.adapter(model, languages=["en", "unsupported"])
        model.is_multilingual = False
        with self.assertRaisesRegex(ValueError, "language"):
            self.adapter(model, languages=["fr"])

    def test_whisper_preflights_each_declared_language(self):
        model = PublicModel(SttOutput("unreachable"))
        self.adapter(model, languages=["en", "fr"])
        self.assertEqual(model.tokenizer_calls, [
            {"language":"en", "task":"transcribe"},
            {"language":"fr", "task":"transcribe"}])

    def test_whisper_public_dataclass_result_and_exact_arguments(self):
        model = PublicModel(SttOutput("An original transcript."))
        adapter,_ = self.adapter(model)
        grant = MediaGrant("unused","unused",AttemptKey("unused",1),48,"unused",2,16000)
        audio = encode_pcm16(struct.pack("<hh",-32768,16384),16000)
        result = adapter.transcribe(AsrInput(grant,"en"),audio,threading.Event())
        self.assertEqual(result,"An original transcript.")
        call = model.calls[0]
        self.assertEqual(set(call),{"audio","language","stream","temperature","return_timestamps"})
        self.assertEqual(call["audio"].values,[-1.0,0.5])
        self.assertEqual({k:v for k,v in call.items() if k != "audio"},{"language":"en","stream":False,"temperature":0,"return_timestamps":False})

    def test_kokoro_local_voice_and_complete_pcm(self):
        model = PublicModel(iter([GenerationResult(Samples([-1.0,0.5,1.0]),24000),GenerationResult(Samples([0.0]),24000)]))
        adapter,_ = self.adapter(model,"kokoro")
        audio = adapter.synthesize(TtsInput("A fixed sentence.","heart","en","unused"),threading.Event())
        self.assertEqual(inspect(audio,False).frames,4)
        self.assertEqual(struct.unpack_from("<4h",audio,44),(-32768,16384,32767,0))
        self.assertEqual(model.calls,[{"text":"A fixed sentence.","voice":"/original/local/heart.safetensors","speed":1.0,"lang_code":"a"}])

    def test_shape_and_output_ceiling_precede_host_copy(self):
        for shape in ((1,1),(1440001,)):
            with self.subTest(shape=shape):
                model = PublicModel(iter([GenerationResult(Samples([0.0],shape),24000)]))
                adapter,arrays = self.adapter(model,"kokoro")
                with self.assertRaises(ValueError):
                    adapter.synthesize(TtsInput("x","heart","en","unused"),threading.Event())
                self.assertEqual(arrays.copies,0)

    def test_nonfinite_and_rate_change_fail(self):
        for outputs in ([GenerationResult(Samples([float("nan")]),24000)],
            [GenerationResult(Samples([0.0]),24000),GenerationResult(Samples([0.0]),16000)]):
            adapter,_ = self.adapter(PublicModel(iter(outputs)),"kokoro")
            with self.assertRaises(ValueError):
                adapter.synthesize(TtsInput("x","heart","en","unused"),threading.Event())

    def test_extra_package_cannot_override_core_pins(self):
        for name in ("mlx","mlx-audio","MLX_Audio","mlx.audio","mlx__audio","mlx..audio","MLX---AUDIO"):
            manifest = {"runtime":{"python":platform.python_version(),"mlx":"0.32.4","mlx_audio":"0.6.1",
                "mlx_audio_commit":audio_provider.AUDIO_COMMIT,"packages":{name:"wrong"}}}
            with self.subTest(package=name), self.assertRaisesRegex(ValueError,"override"):
                audio_provider.verify_local_profile(manifest)

    def test_cancellation_suppresses_provider_output(self):
        adapter,_ = self.adapter(PublicModel(SttOutput("late")))
        cancelled = threading.Event()
        cancelled.set()
        grant = MediaGrant("unused","unused",AttemptKey("unused",1),46,"unused",1,16000)
        self.assertIsNone(adapter.transcribe(AsrInput(grant,"en"),encode_pcm16(b"\x00\x00",16000),cancelled))


if __name__ == "__main__":
    unittest.main()

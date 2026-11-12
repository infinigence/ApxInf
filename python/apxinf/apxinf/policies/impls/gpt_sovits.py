"""GPT-SoVITS V5 policy: text plus a reference utterance to 48 kHz PCM."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import operator
from collections import OrderedDict
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from ..registry import register_policy
from ._gpt_sovits_audio import resample_kaiser_best

__all__ = ["GptSovitsPolicy"]

_TEXT_CACHE_BYTES = 8 * 1024 * 1024
_REFERENCE_CACHE_BYTES = 64 * 1024 * 1024


def _immutable_inputs(values):
    # A bytes owner prevents both writes and re-enabling the NumPy write flag.
    return tuple(np.frombuffer(value.tobytes(), dtype=value.dtype).reshape(value.shape)
                 for value in values)


def _sha256_file(path):
    # hashlib.file_digest needs Python 3.11; the package supports 3.9.
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _frontend_root(checkpoint: Path, explicit: Optional[Path]) -> Optional[Path]:
    candidates = [
        explicit,
        checkpoint / "frontend",
        Path(os.environ["GPT_SOVITS_ROOT"]) if os.environ.get("GPT_SOVITS_ROOT") else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_dir():
            return candidate.resolve()
    return None


def _load_frontend(checkpoint: Path, root: Optional[Path]):
    root = _frontend_root(checkpoint, root)
    if root is not None:
        for entry in (root, root / "GPT_SoVITS"):
            value = str(entry)
            if value not in sys.path:
                sys.path.insert(0, value)

    try:
        torch = importlib.import_module("torch")
        torchaudio = importlib.import_module("torchaudio")
        cleaner = importlib.import_module("text.cleaner")
        text_module = importlib.import_module("text")
        mel_processing = importlib.import_module("module.mel_processing")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise ImportError(
            "GptSovitsPolicy requires the GPT-SoVITS text frontend plus torch, "
            "torchaudio and transformers. Put the frontend in CHECKPOINT/frontend, "
            "pass frontend_root=, or set GPT_SOVITS_ROOT."
        ) from exc

    tokenizer_dir = checkpoint / "tokenizer"
    if not tokenizer_dir.is_dir() and root is not None:
        tokenizer_dir = (
            root
            / "GPT_SoVITS"
            / "pretrained_models"
            / "chinese-roberta-wwm-ext-large"
        )
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(
            "RoBERTa tokenizer not found. Expected CHECKPOINT/tokenizer or "
            "GPT_SOVITS_ROOT/GPT_SoVITS/pretrained_models/"
            "chinese-roberta-wwm-ext-large."
        )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        str(tokenizer_dir), local_files_only=True
    )
    # The official API defaults to cut5. Load the same segmentation module the
    # PyTorch pipeline uses instead of re-implementing its punctuation rules.
    segmentation = None
    if root is not None:
        candidate = (
            root / "GPT_SoVITS" / "TTS_infer_pack" / "text_segmentation_method.py"
        )
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location(
                "apxinf_gpt_sovits_text_segmentation", candidate
            )
            segmentation = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(segmentation)
    if segmentation is None:
        try:
            segmentation = importlib.import_module(
                "TTS_infer_pack.text_segmentation_method"
            )
        except ImportError as exc:
            raise ImportError(
                "GptSovitsPolicy requires TTS_infer_pack/text_segmentation_method.py "
                "from the GPT-SoVITS frontend"
            ) from exc
    return {
        "torch": torch,
        "torchaudio": torchaudio,
        "clean_text": cleaner.clean_text,
        "cleaned_text_to_sequence": text_module.cleaned_text_to_sequence,
        "mel_spectrogram_torch": mel_processing.mel_spectrogram_torch,
        "spectrogram_torch": mel_processing.spectrogram_torch,
        "tokenizer": tokenizer,
        "segmentation": segmentation,
        "root": root,
    }


@register_policy("gpt_sovits")
class GptSovitsPolicy:
    """Synthesize speech from target text, reference text and reference audio."""

    def __init__(
        self,
        model_runner,
        *,
        frontend: Mapping[str, Any],
        language: str = "zh",
        text_version: str = "v2",
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.model_runner = model_runner
        self.frontend = dict(frontend)
        # Match the reference STFT/mel device: CPU FFT and CUDA FFT can round
        # differently before the native FP16 input conversion.
        self._frontend_device = str(getattr(model_runner, "device", "cpu"))
        self._resamplers = {}
        self._resampy_filters = {}
        self.language = language.removeprefix("all_")
        self.text_version = text_version
        self._prompt_cache_enabled = True
        self._text_cache_enabled = True
        self._max_text_entries = 32
        self._text_cache = OrderedDict()
        self._reference_text_cache = None
        self._reference_audio_cache = None
        self._cache_stats = {
            "text_hits": 0, "text_misses": 0,
            "reference_text_hits": 0, "reference_text_misses": 0,
            "reference_audio_hits": 0, "reference_audio_misses": 0,
        }
        self.metadata = {
            "model_type": "gpt_sovits",
            "sample_rate": int(model_runner.sample_rate),
            "language": self.language,
            "text_version": self.text_version,
            **(dict(metadata) if metadata else {}),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        device: str = "cuda:0",
        precision: str = "auto",
        autotune: bool = False,
        frontend_root=None,
        language: str = "zh",
        text_version: str = "v2",
        model_runner=None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "GptSovitsPolicy":
        model_dir = Path(model_dir).resolve()
        frontend = _load_frontend(
            model_dir, None if frontend_root is None else Path(frontend_root)
        )
        if model_runner is None:
            from apxinf_py import SpeechRunner

            model_runner = SpeechRunner.from_pretrained(
                model_dir,
                device=device,
                precision=precision,
                model_name="gpt_sovits",
                autotune=autotune,
            )
        return cls(
            model_runner,
            frontend=frontend,
            language=language,
            text_version=text_version,
            metadata=metadata,
        )

    def configure_cache(
        self, *, prompt: bool = True, text: bool = True, max_text_entries: int = 32
    ) -> None:
        """Configure bounded per-instance frontend and native conditioning caches."""
        max_text_entries = operator.index(max_text_entries)
        if max_text_entries < 0:
            raise ValueError("max_text_entries must be non-negative")
        prompt, text = bool(prompt), bool(text)
        configure = getattr(self.model_runner, "configure_cache", None)
        if configure is not None:
            configure(prompt=prompt, text=text, max_text_entries=max_text_entries)
        if prompt != self._prompt_cache_enabled:
            self._reference_text_cache = None
            self._reference_audio_cache = None
        if text != self._text_cache_enabled:
            self._text_cache.clear()
        self._prompt_cache_enabled, self._text_cache_enabled = prompt, text
        self._max_text_entries = max_text_entries
        while len(self._text_cache) > max_text_entries:
            self._text_cache.popitem(last=False)

    def clear_cache(self, *, reset_stats: bool = True) -> None:
        """Release frontend and native conditioning; retain prepared graphs."""
        clear = getattr(self.model_runner, "clear_cache", None)
        if clear is not None:
            clear(reset_stats=reset_stats)
        self._text_cache.clear()
        self._reference_text_cache = None
        self._reference_audio_cache = None
        if reset_stats:
            for key in self._cache_stats:
                self._cache_stats[key] = 0

    def cache_info(self) -> Mapping[str, Any]:
        """Return separate frontend/native hit counts and retained byte counts."""
        audio = self._reference_audio_cache
        reference_text = self._reference_text_cache
        native_info = getattr(self.model_runner, "cache_info", None)
        return {
            "frontend": {
                "prompt_enabled": self._prompt_cache_enabled,
                "text_enabled": self._text_cache_enabled,
                "max_text_entries": self._max_text_entries,
                "text_entries": len(self._text_cache),
                "reference_text_entries": int(reference_text is not None),
                "reference_audio_entries": int(audio is not None),
                "host_bytes": (
                    sum(entry[1] for entry in self._text_cache.values())
                    + (reference_text[2] if reference_text else 0)
                    + (audio[3] if audio else 0)
                ),
                "text_budget_bytes": _TEXT_CACHE_BYTES,
                "reference_budget_bytes": _REFERENCE_CACHE_BYTES,
                "reference_audio_sha256": audio[2] if audio else None,
                **self._cache_stats,
            },
            "native": dict(native_info()) if native_info is not None else {"supported": False},
        }

    def _text_key(self, text):
        return (
            str(text).strip(), self.language, self.text_version,
            str(self.frontend.get("root")),
            id(self.frontend["clean_text"]),
            id(self.frontend["cleaned_text_to_sequence"]),
            id(self.frontend["tokenizer"]),
        )

    def _text_inputs(self, text: str, *, reference: bool = False):
        key = self._text_key(text)
        enabled = self._prompt_cache_enabled if reference else self._text_cache_enabled
        kind = "reference_text" if reference else "text"
        if enabled:
            if reference:
                entry = self._reference_text_cache
                if entry is not None and entry[0] == key:
                    self._cache_stats[kind + "_hits"] += 1
                    return entry[1]
            elif key in self._text_cache:
                entry = self._text_cache.pop(key)
                self._text_cache[key] = entry
                self._cache_stats[kind + "_hits"] += 1
                return entry[0]
        self._cache_stats[kind + "_misses"] += 1
        values = self._compute_text_inputs(text)
        if enabled:
            size = sum(value.nbytes for value in values) + len(key[0].encode("utf-8"))
            if size <= _TEXT_CACHE_BYTES:
                values = _immutable_inputs(values)
                if reference:
                    self._reference_text_cache = (key, values, size)
                elif self._max_text_entries:
                    self._text_cache[key] = (values, size)
                    while (
                        len(self._text_cache) > self._max_text_entries
                        or sum(entry[1] for entry in self._text_cache.values()) > _TEXT_CACHE_BYTES
                    ):
                        self._text_cache.popitem(last=False)
        return values

    def _compute_text_inputs(self, text: str):
        text = str(text).strip()
        if not text:
            raise ValueError("text and reference_text must be non-empty")
        previous_cwd = Path.cwd()
        try:
            frontend_root = self.frontend.get("root")
            if frontend_root is not None:
                os.chdir(frontend_root)
            phones, word2ph, normalized = self.frontend["clean_text"](
                text, self.language, self.text_version
            )
            phones = self.frontend["cleaned_text_to_sequence"](
                phones, self.text_version
            )
        finally:
            os.chdir(previous_cwd)
        token_ids = self.frontend["tokenizer"](
            normalized, return_tensors=None
        )["input_ids"]
        if len(token_ids) - 2 != len(word2ph):
            raise ValueError(
                f"tokenizer produced {len(token_ids)} IDs for "
                f"{len(word2ph)} word2ph entries"
            )
        if sum(int(value) for value in word2ph) != len(phones):
            raise ValueError("word2ph does not expand to the phone sequence")
        return (
            np.ascontiguousarray(phones, dtype=np.uint32),
            np.ascontiguousarray(token_ids, dtype=np.uint32),
            np.ascontiguousarray(word2ph, dtype=np.uint32),
        )

    def _reference_inputs(self, path):
        path = Path(path).resolve(strict=True)
        identity = _file_identity(path)
        key = (
            str(path), identity, self._frontend_device,
            id(self.frontend["torchaudio"]),
            id(self.frontend["mel_spectrogram_torch"]),
            id(self.frontend["spectrogram_torch"]),
        )
        digest = None
        if self._prompt_cache_enabled:
            # Some filesystems retain identical timestamps for rapid same-size
            # overwrites. Verify contents on hits, too; stat alone is insufficient.
            digest = _sha256_file(path)
        entry = self._reference_audio_cache
        if (self._prompt_cache_enabled and entry is not None
                and entry[0] == key and entry[2] == digest):
            self._cache_stats["reference_audio_hits"] += 1
            return entry[1]
        self._cache_stats["reference_audio_misses"] += 1
        # Drop a stale entry before computing; exceptions must not retain it.
        self._reference_audio_cache = None
        values = self._compute_reference_inputs(path)
        if self._prompt_cache_enabled:
            size = sum(value.nbytes for value in values)
            if size <= _REFERENCE_CACHE_BYTES:
                final_digest = _sha256_file(path)
                if _file_identity(path) == identity and final_digest == digest:
                    values = _immutable_inputs(values)
                    self._reference_audio_cache = (key, values, digest, size)
        return values

    def _compute_reference_inputs(self, path):
        torch = self.frontend["torch"]
        torchaudio = self.frontend["torchaudio"]
        waveform, source_rate = torchaudio.load(str(Path(path)))
        waveform = waveform.to(
            device=self._frontend_device, dtype=torch.float32
        )
        if waveform.ndim != 2 or waveform.shape[1] == 0:
            raise ValueError("reference_audio must contain audio samples")
        waveform = waveform.mean(dim=0, keepdim=True)

        def resample(audio, source, target):
            key = (int(source), int(target))
            if key not in self._resamplers:
                self._resamplers[key] = torchaudio.transforms.Resample(
                    source, target
                ).to(waveform.device)
            return self._resamplers[key](audio)
        audio_16k = (
            waveform
            if source_rate == 16000
            else resample_kaiser_best(
                waveform, source_rate, 16000, self._resampy_filters
            )
        )
        duration = audio_16k.shape[-1] / 16000.0
        if duration < 3.0 or duration > 10.0:
            raise ValueError(
                f"reference_audio must be 3-10 seconds, got {duration:.3f}s"
            )
        # GPT-SoVITS appends 0.3 seconds measured at the 32 kHz acoustic rate
        # before HuBERT, which is 9,600 samples after the 16 kHz resample.
        hubert_audio = torch.cat(
            [
                audio_16k.reshape(-1),
                torch.zeros(9600, dtype=torch.float32, device=waveform.device),
            ]
        )

        audio_32k = (
            waveform
            if source_rate == 32000
            else resample(waveform, source_rate, 32000)
        )
        # The reference normalizes peaks only on the spectrogram branch.
        # Its mel branch uses the original waveform.
        spec_audio = audio_32k
        peak = spec_audio.abs().max()
        if peak > 1:
            spec_audio = spec_audio / min(2.0, float(peak))

        mel = self.frontend["mel_spectrogram_torch"](
            audio_32k,
            n_fft=1280,
            win_size=1280,
            hop_size=320,
            num_mels=100,
            sampling_rate=32000,
            fmin=0,
            fmax=None,
            center=False,
        )
        mel = (mel - (-12.0)) / 14.0 * 2.0 - 1.0
        # V5 keeps the final 500 prompt frames; the full spectrogram remains
        # available to the reference encoder and native alignment logic.
        mel = mel[:, :, -500:]
        spec = self.frontend["spectrogram_torch"](
            spec_audio,
            2048,
            32000,
            640,
            2048,
            center=False,
        )
        mel = mel[0].transpose(0, 1).contiguous().cpu().numpy()
        spec = spec[0, :704].transpose(0, 1).contiguous().cpu().numpy()
        return (
            np.ascontiguousarray(
                hubert_audio.to(dtype=torch.float16).float().cpu().numpy(),
                dtype=np.float32,
            ),
            np.ascontiguousarray(mel, dtype=np.float32),
            np.ascontiguousarray(spec, dtype=np.float32),
        )

    def prepare(
        self,
        observation: Mapping[str, Any],
        *,
        target_frames: int,
        sample_steps: Optional[int] = None,
    ) -> Mapping[str, int]:
        """Capture frontend, acoustic and vocoder graphs for one fixed profile.

        CFM captures initialization and a reusable solver step with fixed buffers.
        target_frames is four times the semantic token count produced by GPT.
        A later request whose generated length differs is rejected explicitly.
        """
        try:
            text = observation["text"]
            reference_text = observation["reference_text"]
            reference_audio = observation["reference_audio"]
        except KeyError as exc:
            raise KeyError(
                "GptSovitsPolicy.prepare requires text, reference_text and "
                "reference_audio"
            ) from exc

        target_frames = int(target_frames)
        if target_frames <= 0 or target_frames % 4:
            raise ValueError("target_frames must be a positive multiple of 4")
        steps = int(
            observation.get("sample_steps", 4)
            if sample_steps is None
            else sample_steps
        )
        if steps <= 0:
            raise ValueError("sample_steps must be positive")

        # The captured profile is fixed to one fragment's phone count, so it
        # must be prepared on exactly what infer() will synthesize.
        split_method = str(observation.get("text_split_method", "cut5"))
        fragments = self._split_text(text, split_method)
        if len(fragments) != 1:
            raise ValueError(
                f"prepare() captures a single fragment, but text_split_method "
                f"{split_method!r} splits this text into {len(fragments)}; "
                "use 'cut0' or prepare each fragment separately"
            )
        phone_ids, bert_ids, _word2ph = self._text_inputs(fragments[0])
        ref_phone_ids, ref_bert_ids, _ref_word2ph = self._text_inputs(
            reference_text, reference=True
        )
        ref_audio, ref_mel, ref_spec = self._reference_inputs(reference_audio)
        self.model_runner.prepare(
            len(phone_ids),
            len(bert_ids),
            len(ref_phone_ids),
            len(ref_bert_ids),
            len(ref_audio),
            ref_mel.shape[0],
            ref_spec.shape[0],
            target_frames,
            steps,
        )
        return {
            "phone_count": len(phone_ids),
            "bert_token_count": len(bert_ids),
            "reference_phone_count": len(ref_phone_ids),
            "reference_bert_token_count": len(ref_bert_ids),
            "reference_audio_samples": len(ref_audio),
            "reference_frames": int(ref_mel.shape[0]),
            "reference_spec_frames": int(ref_spec.shape[0]),
            "target_frames": target_frames,
            "sample_steps": steps,
        }

    @property
    def is_prepared(self) -> bool:
        return bool(self.model_runner.is_prepared)

    def clear_prepared(self) -> None:
        self.model_runner.clear_prepared()

    def _split_text(self, text: str, method: str):
        """Split target text with the official GPT-SoVITS method.

        `cut0` is the no-split path. Every other method returns one fragment per
        line; the official API then synthesizes each line independently and
        joins it with `fragment_interval` silence.
        """
        segmenter = self.frontend.get("segmentation")
        if segmenter is None:
            raise RuntimeError("GPT-SoVITS text segmentation frontend is not loaded")
        try:
            split = segmenter.get_method(method)
        except ValueError as exc:
            available = ", ".join(segmenter.get_method_names())
            raise ValueError(
                f"unknown text_split_method {method!r}; available: {available}"
            ) from exc
        result = split(text)
        if method == "cut0":
            return [result] if result.strip() else [text]
        fragments = [line for line in result.split("\n") if line.strip()]
        return fragments or [text]

    def infer(self, observation: Mapping[str, Any], *, noise=None):
        if noise is not None:
            raise ValueError("GptSovitsPolicy does not accept a noise array")
        try:
            text = observation["text"]
            reference_text = observation["reference_text"]
            reference_audio = observation["reference_audio"]
        except KeyError as exc:
            raise KeyError(
                "GptSovitsPolicy requires text, reference_text and reference_audio"
            ) from exc

        started = time.perf_counter()
        split_method = str(observation.get("text_split_method", "cut5"))
        speed_factor = float(observation.get("speed_factor", 1.0))
        fragment_interval = float(observation.get("fragment_interval", 0.3))
        if not np.isfinite(speed_factor) or speed_factor <= 0.0:
            raise ValueError("speed_factor must be finite and positive")
        if not np.isfinite(fragment_interval) or fragment_interval < 0.0:
            raise ValueError("fragment_interval must be finite and non-negative")
        fragments = self._split_text(text, split_method)
        base_seed = int(observation.get("seed", 0))
        if base_seed < 0 or base_seed + len(fragments) - 1 >= 1 << 64:
            raise ValueError("seed must be a non-negative 64-bit integer")
        ref_phone_ids, ref_bert_ids, ref_word2ph = self._text_inputs(
            reference_text, reference=True
        )
        ref_audio, ref_mel, ref_spec = self._reference_inputs(reference_audio)
        prepared = time.perf_counter()

        common = {
            "top_k": int(observation.get("top_k", 5)),
            "top_p": float(observation.get("top_p", 1.0)),
            "temperature": float(observation.get("temperature", 1.0)),
            "repetition_penalty": float(
                observation.get("repetition_penalty", 1.35)
            ),
            "sample_steps": int(observation.get("sample_steps", 4)),
            "cfg_rate": float(observation.get("cfg_rate", 0.0)),
            "max_seconds": float(observation.get("max_seconds", 30.0)),
            "speed_factor": speed_factor,
        }
        # The official pipeline seeds once and lets the generator advance while
        # it synthesizes fragments. The native runner seeds each call, so use a
        # deterministic continuation instead of restarting every fragment from
        # the same noise.
        # Like the official pipeline, the gap is sized at the 32 kHz acoustic
        # rate even though the vocoder emits 48 kHz, so it plays 2/3 as long.
        gap = int(32000 * fragment_interval)
        pieces = []
        detail = []
        sample_rate = None
        model_started = time.perf_counter()
        for index, fragment in enumerate(fragments):
            phone_ids, bert_ids, word2ph = self._text_inputs(fragment)
            result = self.model_runner.synthesize(
                phone_ids,
                bert_ids,
                word2ph,
                ref_phone_ids,
                ref_bert_ids,
                ref_word2ph,
                ref_audio,
                ref_mel,
                ref_spec,
                seed=base_seed + index,
                **common,
            )
            audio = np.asarray(result["samples"], dtype=np.float32)
            valid_length = int(result["valid_length"])
            audio = np.ascontiguousarray(audio[:valid_length])
            peak = float(np.abs(audio).max()) if audio.size else 0.0
            if peak > 1.0:
                audio = (audio / peak).astype(np.float32, copy=False)
            if sample_rate is None:
                sample_rate = int(result["sample_rate"])
            pieces.append(audio)
            if gap:
                pieces.append(np.zeros(gap, dtype=np.float32))
            detail.append(
                {
                    "text": fragment,
                    "samples": int(audio.size),
                    "duration": float(audio.size) / float(sample_rate),
                }
            )
        model_finished = time.perf_counter()
        samples = (
            np.concatenate(pieces).astype(np.float32, copy=False)
            if pieces
            else np.zeros(0, dtype=np.float32)
        )
        finished = time.perf_counter()
        return {
            "samples": samples,
            "sample_rate": int(sample_rate or 0),
            "valid_length": int(samples.size),
            "fragments": detail,
            "timing": {
                "preprocess_ms": (prepared - started) * 1000.0,
                "model_ms": (model_finished - prepared) * 1000.0,
                "total_ms": (finished - started) * 1000.0,
            },
        }

    def close(self) -> None:
        self.clear_cache()
        self._resamplers.clear()
        self._resampy_filters.clear()
        self.model_runner = None

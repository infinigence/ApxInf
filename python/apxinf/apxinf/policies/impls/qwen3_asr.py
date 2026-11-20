"""qwen3-asr policy: 16 kHz mono waveform to transcribed text.

The native runtime (``apxinf_py.AsrRunner``) owns every neural computation
from the waveform upload to token ids — including the chat-template prompt
(prefix + audio pad tokens + language suffix), which it builds internally.
This policy owns only the tokenizer: text decoding of the generated ids.
Unlike whisper, the decoder prefix is NOT constructed here; a non-empty
``prompt_token_ids`` is rejected by the runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Mapping, Optional

import numpy as np

from ..registry import register_policy

__all__ = ["Qwen3AsrPolicy"]


def _load_tokenizer(tokenizer_dir: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment guard
        raise ImportError(
            "Qwen3AsrPolicy requires transformers (AutoTokenizer) for "
            "text decoding"
        ) from exc
    return AutoTokenizer.from_pretrained(str(tokenizer_dir))


@register_policy("qwen3_asr")
class Qwen3AsrPolicy:
    """Transcribe speech with qwen3-asr-1.7B (v1: Chinese, transcribe)."""

    def __init__(
        self,
        model_runner,
        *,
        tokenizer,
        language: str = "Chinese",
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.language = str(language)
        self.sample_rate = int(model_runner.sample_rate)
        self.max_audio_samples = int(model_runner.max_audio_samples)
        self._languages = set(model_runner.languages)
        if self.language not in self._languages:
            raise ValueError(
                f"language {self.language!r} is not supported; choose one of "
                f"{sorted(self._languages)}"
            )
        self.metadata = {
            "model_type": "qwen3_asr",
            "sample_rate": self.sample_rate,
            "language": self.language,
            "task": "transcribe",
            **(dict(metadata) if metadata else {}),
        }

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        device: str = "cuda:0",
        autotune: bool = False,
        tuning_dir=None,
        language: str = "Chinese",
        tokenizer_dir=None,
        model_runner=None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "Qwen3AsrPolicy":
        model_dir = Path(model_dir).resolve()
        tokenizer_dir = (
            model_dir
            if tokenizer_dir is None
            else Path(tokenizer_dir).resolve()
        )
        if model_runner is None:
            from apxinf_py import AsrRunner

            runner_kwargs = {"autotune": autotune}
            if tuning_dir is not None:
                runner_kwargs["tuning_dir"] = str(Path(tuning_dir).resolve())
            model_runner = AsrRunner.load(model_dir, device=device, **runner_kwargs)
        tokenizer = _load_tokenizer(tokenizer_dir)
        return cls(
            model_runner,
            tokenizer=tokenizer,
            language=language,
            metadata=metadata,
        )

    def transcribe(
        self,
        waveform,
        *,
        language: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> Mapping[str, Any]:
        """Transcribe one 16 kHz mono float32 waveform (<= 30 s).

        Returns ``{"text": str, "token_ids": list[int],
        "stopped_reason": str, "valid_tokens": int}``.
        """
        values = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if values.size < 1:
            raise ValueError("waveform must be non-empty")
        if values.size > self.max_audio_samples:
            raise ValueError(
                f"waveform has {values.size} samples; qwen3-asr accepts at "
                f"most {self.max_audio_samples} "
                f"({self.max_audio_samples / self.sample_rate:.0f} s at "
                f"{self.sample_rate} Hz)"
            )
        language = str(language or self.language)
        if language not in self._languages:
            raise ValueError(
                f"language {language!r} is not supported; choose one of "
                f"{sorted(self._languages)}"
            )
        # The runtime builds the full prompt (chat template + audio pads +
        # language suffix) internally; no prefix is passed here.
        result = self.model_runner.transcribe(
            values, language=language, task="transcribe",
            max_new_tokens=int(max_new_tokens),
        )
        token_ids = [int(token) for token in result["token_ids"]]
        # Special tokens (<|asr_text|>, <|im_end|>, ...) are metadata, not
        # content — the runtime's EOS ids stay in token_ids for parity work.
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True).strip()
        return {
            "text": text,
            "token_ids": token_ids,
            "stopped_reason": result["stopped_reason"],
            "valid_tokens": int(result["valid_tokens"]),
        }

    # Alias matching the other policies' call verb.
    infer = transcribe

    def __repr__(self) -> str:
        return (
            f"Qwen3AsrPolicy(device={self.model_runner.device}, "
            f"language={self.language!r})"
        )

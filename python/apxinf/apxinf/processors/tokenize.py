"""Prompt construction + tokenization step.

Wraps a SentencePiece model exactly as the OpenPI-derived reference did: the
prompt is cleaned (``strip`` + ``_``/newline -> space), encoded with a BOS, and
a trailing newline token is appended; the result is validated to
``1..=max_token_len`` tokens.

State injection (proprioception) is a **reserved** capability, matching the Rust
``pi05_prompt`` / ``discretize_state`` path but **off by default** so behavior is
identical to the current serving link. When ``discrete_state=True`` the prompt
becomes ``"Task: {task}, State: {s0 s1 ...};\nAction: "`` with the state
discretized to ``0..=255`` — see :func:`discretize_state`. sentencepiece is
imported lazily so the rest of the processor library stays importable without it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import ProcessorStep

__all__ = ["PromptTokenizer", "SyntheticTokenizer", "discretize_state", "build_prompt"]


def discretize_state(state: Sequence[float]) -> np.ndarray:
    """Discretize a normalized state to ``uint8`` bins in ``0..=255``.

    Matches NumPy ``digitize(state, linspace(-1, 1, 257)[:-1]) - 1`` with
    saturation, i.e. the Rust ``discretize_state``:
    ``clamp(floor((v + 1) * 128), 0, 255)``.
    """
    values = np.asarray(state, dtype=np.float64)
    bins = np.floor((values + 1.0) * 128.0)
    return np.clip(bins, 0, 255).astype(np.uint8)


def _clean_task(prompt: str) -> str:
    return prompt.strip().replace("_", " ").replace("\n", " ")


def build_prompt(
    prompt: str,
    state: Optional[Sequence[float]] = None,
    discrete_state: bool = False,
) -> str:
    """Build the text fed to the tokenizer.

    With ``discrete_state=False`` (default) the cleaned task text is returned
    (the trailing newline token is added separately during encoding, matching
    the reference). With ``discrete_state=True`` the discretized state is spliced
    into the ``Task: ... , State: ...;\\nAction: `` template (aligned with Rust
    ``pi05_prompt``).
    """
    task = _clean_task(prompt)
    if not discrete_state:
        return task
    if state is None:
        raise ValueError("PromptTokenizer: discrete_state=True requires a state array")
    tokens = " ".join(str(int(v)) for v in discretize_state(state))
    return f"Task: {task}, State: {tokens};\nAction: "


class PromptTokenizer(ProcessorStep):
    """Turn a prompt string (and optionally a state) into ``uint32`` token ids.

    Parameters
    ----------
    model_path:
        Path to the SentencePiece ``.model`` file.
    max_token_len:
        Upper bound on the token count (default 200, the pi05 contract).
    discrete_state:
        Reserved. When ``True``, splice a discretized state into the prompt;
        the caller must then pass ``state`` to :meth:`__call__`. Default ``False``.
    """

    PARAMS = ("max_token_len", "discrete_state")

    def __init__(self, model_path, max_token_len: int = 200, discrete_state: bool = False):
        import sentencepiece  # lazy: keeps the rest of the library importable without it

        self.model_path = str(model_path)
        self.max_token_len = int(max_token_len)
        self.discrete_state = bool(discrete_state)
        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=self.model_path)

    def __call__(self, prompt: str, state: Optional[Sequence[float]] = None) -> np.ndarray:
        if not isinstance(prompt, str):
            raise TypeError(f"PromptTokenizer: prompt must be a string, got {type(prompt)!r}")
        text = build_prompt(prompt, state=state, discrete_state=self.discrete_state)
        tokens = self._tokenizer.encode(text, add_bos=True)
        if not self.discrete_state:
            # Reference appends a standalone newline token after the task text.
            tokens = list(tokens) + list(self._tokenizer.encode("\n"))
        if not 0 < len(tokens) <= self.max_token_len:
            raise ValueError(
                f"PromptTokenizer: token count must be in 1..={self.max_token_len}, "
                f"got {len(tokens)}"
            )
        return np.asarray(tokens, dtype=np.uint32)


class SyntheticTokenizer(ProcessorStep):
    """Checkpoint-free stand-in for :class:`PromptTokenizer`.

    Emits a fixed ``token_count``-length id vector for *any* prompt, so the
    language prefix runs the intended sequence length with **no SentencePiece
    model on disk**. The ids are a deterministic small ramp kept well inside any
    pi05 vocabulary; their content is irrelevant to latency (they only index the
    token-embedding rows). This is for L2/L3 latency benchmarking with random
    weights, never for real numerics.
    """

    PARAMS = ("token_count", "max_token_len")
    discrete_state = False  # never injects state; matches PromptTokenizer's attr

    def __init__(self, token_count: int, max_token_len: int = 200):
        self.token_count = int(token_count)
        self.max_token_len = int(max_token_len)
        if not 0 < self.token_count <= self.max_token_len:
            raise ValueError(
                f"SyntheticTokenizer: token_count must be in 1..={self.max_token_len}, "
                f"got {self.token_count}"
            )

    def __call__(self, prompt: str, state: Optional[Sequence[float]] = None) -> np.ndarray:
        # Content-free: a deterministic ramp in 1..=256, safe for any pi05 vocab.
        return (np.arange(self.token_count, dtype=np.uint32) % np.uint32(256)) + np.uint32(1)

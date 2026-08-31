"""Prompt construction + tokenization step.

Wraps a SentencePiece model exactly as the OpenPI-derived reference did: the
prompt is cleaned (``strip`` + ``_``/newline -> space), encoded with a BOS, and
a trailing newline token is appended; the result is validated to
``1..=max_token_len`` tokens.

State injection (proprioception) is a **reserved** capability, matching the Rust
``pi05_prompt`` / ``discretize_state`` path but **off by default** so behavior is
identical to the current serving link. When ``discrete_state=True`` the prompt
becomes ``"Task: {task}, State: {s0 s1 ...};\nAction: "`` with the state
discretized to ``-1..=255`` — see :func:`discretize_state`. sentencepiece is
imported lazily so the rest of the processor library stays importable without it.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .base import ProcessorStep

__all__ = ["PromptTokenizer", "SyntheticTokenizer", "discretize_state", "build_prompt"]


#: The 256 bin edges openpi discretizes against, ``linspace(-1, 1, 257)[:-1]``.
#: Every edge is ``-1 + i/128``, exact in binary floating point, so a comparison
#: against one is exact — which is the whole reason :func:`discretize_state`
#: compares instead of computing an index.
_BIN_EDGES = np.linspace(-1.0, 1.0, 256 + 1)[:-1]


def discretize_state(state: Sequence[float]) -> np.ndarray:
    """Discretize a normalized state into the pi05 prompt's integer bins.

    Reproduces NumPy ``digitize(state, linspace(-1, 1, 257)[:-1]) - 1``
    *exactly*, including its **signed underflow bin**: ``digitize`` returns ``0``
    for ``v < -1``, so the bin is ``-1``. openpi writes that ``-1`` verbatim into
    the prompt (``models/tokenizer.py``), so it is a value the model saw during
    training and must not be clamped away — hence ``int16``, not ``uint8``.
    Values ``>= 1`` do saturate, to ``255``.

    Normalized state *does* leave ``[-1, 1]`` on real hardware, because q01/q99
    come from the training split; on the customer's own G1 validation set 7 of
    3200 elements underflow. Clamping them to ``0`` changes the prompt string,
    hence the token ids, hence the whole rollout.

    Implemented as a search over the edges rather than the arithmetic
    ``floor((v + 1) * 128)``, because that expression is not equal to
    ``digitize`` for every input. Adding ``1.0`` to a value just under an edge in
    ``[-1, -0.5)`` moves it into a coarser binade and the sum rounds *up* onto
    the edge, putting it one bin too high::

        v = nextafter(-0.4921875, -inf)     # one ulp below bin edge 65
        v + 1.0 == 0.5078125                # exactly, by round-half-to-even
        floor((v + 1.0) * 128) == 65        # wrong
        digitize(v, edges) - 1  == 64       # right

    ``searchsorted(..., side="right")`` is what ``digitize`` calls for increasing
    bins, so this is the same comparison openpi makes, with no arithmetic on
    ``v`` to round.
    """
    values = np.asarray(state, dtype=np.float64)
    bins = np.searchsorted(_BIN_EDGES, values, side="right") - 1
    return bins.astype(np.int16)


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

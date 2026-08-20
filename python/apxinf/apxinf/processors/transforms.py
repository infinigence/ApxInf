"""dict→dict processor steps for a policy's pre/post :class:`Pipeline`.

openpi composes a policy as ``inputs transforms → model → outputs transforms``,
where each transform is a ``dict -> dict`` function over a shared data dict. This
module provides that transform layer for pi05, reusing the existing single-value
steps (:class:`~apxinf.processors.resize.ResizeWithPad`,
:class:`~apxinf.processors.tokenize.PromptTokenizer`, etc.) unchanged — each class
here is a thin ``ProcessorStep`` that reads a few keys from the data dict,
delegates to a wrapped natural-signature step, and writes its output key back.

Because a policy's pre/post chain is just a :class:`~apxinf.processors.base.Pipeline`
whose flowing value is the data dict, these steps compose, reorder, and override
through the same ``Pipeline`` machinery as the image sub-chain.

**Data-dict key contract**

* pre (input) chain — reads ``observation`` / ``prompt``; writes ``rgb`` (uint8
  NHWC), ``token_ids`` (uint32), ``noise``.
* post (output) chain — reads ``normalized_actions``; writes ``trimmed`` then
  ``actions`` (unnormalized float32).

Each step returns the *same* dict, updated in place — a pre/post chain is a linear
single-owner flow, so there is no aliasing hazard, and later steps see earlier
steps' keys.
"""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

from .base import ProcessorStep

__all__ = ["ImageStack", "Tokenize", "SampleNoise", "Trim", "Unnormalize"]

# Canonical data-dict keys (the inter-step contract).
OBSERVATION = "observation"
PROMPT = "prompt"
RGB = "rgb"
TOKEN_IDS = "token_ids"
NOISE = "noise"
NORMALIZED = "normalized_actions"
TRIMMED = "trimmed"
ACTIONS = "actions"


def _require(data: Mapping[str, Any], key: str, step: str) -> Any:
    try:
        return data[key]
    except KeyError:
        raise KeyError(
            f"{step}: missing data key {key!r}; present keys: {sorted(data)}"
        ) from None


class ImageStack(ProcessorStep):
    """Stack per-view images into one uint8 NHWC array under ``rgb``.

    Runs the single-value ``image_pipeline`` (parse + resize) on each configured
    camera key and stacks the **real** views — one row per ``image_keys`` entry,
    in order. No slot padding: the model runs the exact shape it is handed, so
    the caller must supply precisely the cameras the checkpoint expects (absent
    cameras are simply not sent, never zero-filled).
    """

    def __init__(
        self,
        image_pipeline,
        image_keys: Sequence[str],
        image_size: int,
        *,
        observation_key: str = OBSERVATION,
    ):
        self.image_pipeline = image_pipeline
        self.image_keys = tuple(image_keys)
        self.image_size = int(image_size)
        self.observation_key = observation_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        observation = _require(data, self.observation_key, "ImageStack")
        views = [self.image_pipeline(observation[key]) for key in self.image_keys]
        data[RGB] = np.ascontiguousarray(np.stack(views), dtype=np.uint8)
        return data


class Tokenize(ProcessorStep):
    """Tokenize the prompt (optionally injecting discretized state) into ``token_ids``.

    Mirrors the policy's old state routing: when the tokenizer runs in
    ``discrete_state`` mode and a ``state_normalizer`` is set, the raw state is
    first mapped to ``[-1, 1]`` before discretization; otherwise state is dropped.
    """

    def __init__(
        self,
        tokenizer,
        state_normalizer=None,
        state_key: str = "observation/state",
        *,
        observation_key: str = OBSERVATION,
        prompt_key: str = PROMPT,
    ):
        self.tokenizer = tokenizer
        self.state_normalizer = state_normalizer
        self.state_key = state_key
        self.observation_key = observation_key
        self.prompt_key = prompt_key

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        prompt = _require(data, self.prompt_key, "Tokenize")
        if getattr(self.tokenizer, "discrete_state", False):
            observation = _require(data, self.observation_key, "Tokenize")
            state = observation.get(self.state_key)
            if self.state_normalizer is not None and state is not None:
                state = self.state_normalizer(np.asarray(state, dtype=np.float32))
            data[TOKEN_IDS] = self.tokenizer(prompt, state=state)
        else:
            data[TOKEN_IDS] = self.tokenizer(prompt)
        return data


class SampleNoise(ProcessorStep):
    """Draw the flow-matching prior noise into ``noise``."""

    def __init__(self, noise):
        self.noise = noise

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        data[NOISE] = self.noise()
        return data


class Trim(ProcessorStep):
    """Trim the model's normalized action to the deployable width, under ``trimmed``."""

    PARAMS = ("action_dim",)

    def __init__(self, action_dim: int):
        self.action_dim = int(action_dim)

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        normalized = _require(data, NORMALIZED, "Trim")
        data[TRIMMED] = np.ascontiguousarray(normalized[:, : self.action_dim])
        return data


class Unnormalize(ProcessorStep):
    """Unnormalize ``trimmed`` (or ``normalized_actions``) into ``actions``.

    Delegates to a wrapped :class:`~apxinf.processors.normalize.Unnormalizer`. Reads
    ``trimmed`` when present (the usual post chain ``Trim -> Unnormalize``), else
    falls back to ``normalized_actions`` so the step is usable standalone.
    """

    def __init__(self, unnormalizer):
        self.unnormalizer = unnormalizer

    def __call__(self, data: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        array = data.get(TRIMMED)
        if array is None:
            array = _require(data, NORMALIZED, "Unnormalize")
        data[ACTIONS] = self.unnormalizer(array)
        return data

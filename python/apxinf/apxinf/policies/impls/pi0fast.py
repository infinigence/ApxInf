"""pi0_fast L2 policy: raw observation dict + prompt -> unnormalized action chunk.

π0-FAST is ApxInf's first **token** VLA. Its bare model is autoregressive: the
runtime's ``infer_action_tokens_rgb`` returns raw FAST action-token ids — ending
at the ``|`` terminator, which is where this layer would truncate them anyway,
so the decode never runs the ~90% of steps whose tokens are discarded. Turning those ids back into a deployable chunk — PaliGemma token-space
bookkeeping, FAST BPE detokenization, the orthonormal DCT, and the checkpoint's
action unnormalization — is this module's job, exactly as it is LeRobot's
``detokenize_actions`` on the reference side. L1 stays token-level so a caller
can inspect the ids; L2 never lets them out unconverted.

Prompt assembly is the reference's::

    Task: {cleaned task}, State: {discretized state};

from the *cleaned* task text and the checkpoint-normalized state, padded to
``max_state_dim`` and discretized into the same 256 bins openpi uses. The
PaliGemma BOS is prepended by the tokenizer and appended again after the prompt
by the runtime, so the prefix is ``[images, language, BOS]`` as in the reference.

**No torch, no transformers.** Both tokenizers are loaded through ``tokenizers``
— the engine behind HF's fast tokenizers — straight from the assets the
checkpoint names: PaliGemma (``google/paligemma-3b-pt-224`` by default) for the
prompt and the action id space, and the FAST action tokenizer
(``action_tokenizer_name``) for the BPE detokenization. The normalization
statistics are read out of the checkpoint's ``policy_*_processor_*.safetensors``
state files. That keeps this policy loadable in the same numpy/scipy
environment as PI0.5, with no model-side torch anywhere on the path.

**State is required, not optional.** π0-FAST conditions on proprioception
through the prompt text, so there is no "drop state" variant: the wire state
must be ``state_dim`` wide, where ``state_dim`` is the width of the checkpoint's
own statistics (8 for LIBERO's ``eef_pos + eef_axis_angle + gripper_qpos``), and
it is published in ``metadata`` so a caller can build the right vector.

This module registers ``Pi0FastPolicy`` under ``model_type="pi0_fast"`` (the
checkpoint's ``config.json`` discriminator) and its ``pi0fast`` spelling so
:class:`~apxinf.policies.auto.AutoPolicy` can dispatch to it.
"""

from __future__ import annotations

import json
import os
import pathlib
import struct
import time
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from ...processors import ImageStack, ParseImage, Pipeline, TorchResizeWithPad
from ...processors.tokenize import discretize_state
from ...processors.transforms import OBSERVATION, PROMPT, RGB, lookup_key
from ..base import VIEW_SLOTS
from ..registry import register_policy

__all__ = ["Pi0FastPolicy"]

#: ``prompt`` is openpi's protocol-level name for the instruction field, so it
#: keeps a default here; the camera and state keys are dataset vocabulary and
#: have none (see ``apxinf.conventions``).
_PROMPT_KEY = "prompt"

#: The reference's own defaults. The checkpoint's ``config.json`` normally
#: carries all of them; the fallbacks only keep a hand-made config loadable.
_DEFAULT_TEXT_TOKENIZER = "google/paligemma-3b-pt-224"
_DEFAULT_ACTION_HORIZON = 10
_DEFAULT_ACTION_DIM = 7
_DEFAULT_FAST_SKIP_TOKENS = 128
_DEFAULT_MAX_ACTION_TOKENS = 256
_DEFAULT_MAX_STATE_DIM = 32
_DEFAULT_TOKENIZER_MAX_LENGTH = 200

#: LeRobot's ``NormalizerProcessorStep`` epsilon: ``(x - mean) / (std + eps)``.
_MEAN_STD_EPS = 1e-8


# --- checkpoint assets --------------------------------------------------------


def _hf_cache_matches(repo_id: str, filename: str) -> list:
    """Local Hugging Face cache snapshots holding ``filename``, newest first.

    A deliberate no-network lookup: the assets a checkpoint names are ordinary
    hub repos, and a machine that already ran the reference has them cached. An
    absent cache is reported by the caller's "tried" list, never by a download.
    """
    home = os.environ.get("HF_HOME")
    root = pathlib.Path(home) if home else pathlib.Path.home() / ".cache" / "huggingface"
    snapshots = root / "hub" / f"models--{repo_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return []
    matches = [path for path in snapshots.glob(f"*/{filename}") if path.is_file()]
    return sorted(matches, key=lambda path: (path.stat().st_mtime, str(path)), reverse=True)


def _resolve_asset(spec: str, filename: str, *, env: str, what: str) -> pathlib.Path:
    """Resolve a checkpoint-named tokenizer asset to a local file.

    Order: an explicit path (file or directory), the ``env`` override, the
    checkpoint directory, then the local hub cache for ``spec`` read as a repo
    id. Nothing here touches the network — a miss lists every path tried.
    """
    tried = []
    for candidate in (spec, os.environ.get(env)):
        if not candidate:
            continue
        path = pathlib.Path(candidate)
        resolved = path / filename if path.is_dir() else path
        tried.append(resolved)
        if resolved.is_file():
            return resolved
    cached = _hf_cache_matches(spec, filename)
    tried.extend(cached)
    if cached:
        return cached[0]
    rendered = "\n".join(f"  {path}" for path in tried)
    raise FileNotFoundError(
        f"Pi0FastPolicy: could not find the FAST prompt {what} {filename!r} for "
        f"{spec!r}. Tried:\n{rendered}\n"
        f"Pass an explicit path, or set {env} to a directory holding {filename} "
        f"(the reference checkpoint's hub copy works: {spec})."
    )


def _read_safetensors_f32(path: pathlib.Path, names: Sequence[str]) -> dict:
    """Read named ``float32`` tensors straight out of a ``safetensors`` file.

    The checkpoint's normalization state files are a documented LeRobot layout,
    and a policy that already avoids torch should not pull in a tensor stack just
    to read two small vectors. The format is an 8-byte little-endian header
    length, a JSON header, then a flat payload addressed by byte offsets.
    """
    with pathlib.Path(path).open("rb") as stream:
        header_size = struct.unpack("<Q", stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        payload = 8 + header_size
        tensors = {}
        for name in names:
            entry = header.get(name)
            if entry is None:
                raise KeyError(
                    f"{pathlib.Path(path).name} has no tensor {name!r}; "
                    f"present: {sorted(key for key in header if key != '__metadata__')}"
                )
            if entry["dtype"] != "F32":
                raise ValueError(
                    f"{pathlib.Path(path).name}:{name} is {entry['dtype']}, this reader "
                    "only handles float32 (the LeRobot normalizer state files are F32)"
                )
            start, end = entry["data_offsets"]
            stream.seek(payload + start)
            values = np.frombuffer(stream.read(end - start), dtype="<f4")
            tensors[name] = values.reshape(entry["shape"]).astype(np.float32, copy=True)
        return tensors


def _normalizer_state_file(model_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Locate one ``policy_*_processor_*.safetensors`` state file by its stem."""
    matches = sorted(model_dir.glob(f"{name}*.safetensors"))
    if not matches:
        raise FileNotFoundError(
            f"Pi0FastPolicy: {model_dir} has no {name}*.safetensors; a LeRobot "
            "π0-FAST checkpoint ships its normalization statistics as "
            "policy_preprocessor_step_2_normalizer_processor.safetensors and "
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
        )
    return matches[0]


def _preprocessor_tokenizer_names(model_dir: pathlib.Path) -> dict:
    """The tokenizer names the checkpoint's preprocessor pipeline declares."""
    path = model_dir / "policy_preprocessor.json"
    if not path.is_file():
        return {}
    document = json.loads(path.read_text())
    names = {}
    for step in document.get("steps", []):
        registry = step.get("registry_name")
        config = step.get("config", {})
        if registry == "tokenizer_processor":
            names["text"] = config.get("tokenizer_name")
        elif registry == "action_tokenizer_processor":
            names["action"] = config.get("action_tokenizer_name")
    return names


# --- tokenizers ---------------------------------------------------------------


class _PaligemmaTokenizer:
    """PaliGemma as a plain id map: prompt ids in, action id space out.

    LeRobot round-trips generated ids through token *strings* only to strip the
    ``Action: `` prefix and everything after the ``|`` terminator. Both are
    single-token sequences here, so the same edit is exact in id space — and it
    is the same edit: the reference's ``convert_ids_to_tokens`` /
    ``convert_tokens_to_ids`` pair is the identity on every id the model can
    emit.
    """

    def __init__(self, tokenizer_json: pathlib.Path, *, max_length: int):
        from tokenizers import Tokenizer  # lazy: keeps the module import offline

        self._backend = Tokenizer.from_file(str(tokenizer_json))
        self.max_length = int(max_length)
        # HF's ``PreTrainedTokenizerFast.vocab_size`` — the base vocabulary the
        # action-id offset is computed against, *not* ``tokenizers``' own
        # added-token-inclusive count (PaliGemma carries ``<image>`` at 257152, so
        # the two differ by exactly that one PaliGemma-only token).
        self.vocab_size = int(self._backend.get_vocab_size(with_added_tokens=False))
        self.bos_token_id = self._require("<bos>")
        self.pipe_token_id = self._require("|")
        self.action_prefix_ids = self.encode("Action: ")

    def _require(self, token: str) -> int:
        token_id = self._backend.token_to_id(token)
        if token_id is None:
            raise ValueError(
                f"Pi0FastPolicy: {token!r} is not a token of the PaliGemma "
                f"tokenizer loaded here (vocab {self.vocab_size}); pass the "
                "checkpoint's own tokenizer via tokenizer_path"
            )
        return int(token_id)

    def encode(self, text: str) -> list:
        """Ids for ``text`` with no special tokens — the prompt's raw tokens."""
        return list(self._backend.encode(text, add_special_tokens=False).ids)

    def prompt_ids(self, text: str) -> list:
        """``[BOS] + text + [BOS]``, the sequence the runtime embeds.

        The reference tokenizes the task with ``padding="max_length"`` and
        ``truncation=True`` — so the budget applies to the BOS-prefixed text — and
        then concatenates a second, *trailing* BOS (``sample_actions_fast_kv_cache``:
        ``tokens_in = cat([tokens, bos_token])``) that triggers the autoregressive
        decode. The runtime embeds exactly the ids it is handed and appends nothing
        (``bf16_runtime::infer``: "token_count is the prompt length, which already
        includes the BOS token the reference appends"), so the trailing BOS is
        added here. Right-hand padding is dropped: only the valid prefix is sent.

        The text budget is one token short of ``max_length`` so the whole sequence,
        trailing BOS included, stays inside the checkpoint's ``tokenizer_max_length``
        — the same budget the binding enforces and the one the runtime's sequence
        reservation (``patches + max_token_len + 1``) is sized for.
        """
        prompt = ([self.bos_token_id] + self.encode(text))[: self.max_length - 1]
        return prompt + [self.bos_token_id]


class _FastActionTokenizer:
    """The FAST BPE detokenizer, from the checkpoint's ``action_tokenizer_name``."""

    def __init__(self, root: pathlib.Path):
        from tokenizers import Tokenizer

        self._backend = Tokenizer.from_file(str(root / "tokenizer.json"))
        config = json.loads((root / "processor_config.json").read_text())
        self.min_token = int(config["min_token"])
        self.scale = float(config["scale"])

    def decode(self, token_ids: Sequence[int]) -> str:
        return self._backend.decode(list(token_ids))


def detokenize_action_tokens(
    token_ids,
    *,
    tokenizer: _PaligemmaTokenizer,
    fast_tokenizer: _FastActionTokenizer,
    fast_skip_tokens: int,
    action_horizon: int,
    action_dim: int,
) -> np.ndarray:
    """Raw PaliGemma action tokens -> ``[action_horizon, action_dim]`` actions.

    The reference's ``detokenize_actions`` + ``decode_actions_with_fast``: drop
    the preamble, drop everything after the ``|`` terminator, invert the id
    offset into FAST's own vocabulary, BPE-decode into DCT coefficients, then
    take the orthonormal inverse DCT. Relaxed decoding — truncate or zero-pad to
    ``action_horizon * action_dim`` — is what makes a short or long token stream
    still produce a full chunk.
    """
    sequence = [int(token) for token in token_ids]
    if tokenizer.pipe_token_id in sequence:
        sequence = sequence[: sequence.index(tokenizer.pipe_token_id)]

    prefix = tokenizer.action_prefix_ids
    index = 0
    while index <= len(sequence) - len(prefix):
        if sequence[index : index + len(prefix)] == prefix:
            sequence = sequence[:index] + sequence[index + len(prefix) :]
        else:
            index += 1

    action_ids = [
        tokenizer.vocab_size - 1 - int(fast_skip_tokens) - token for token in sequence
    ]
    text = fast_tokenizer.decode(action_ids)
    coefficients = np.asarray([ord(character) for character in text], dtype=np.float64)
    coefficients = coefficients + fast_tokenizer.min_token

    expected = int(action_horizon) * int(action_dim)
    if coefficients.size < expected:
        coefficients = np.pad(coefficients, (0, expected - coefficients.size))
    else:
        coefficients = coefficients[:expected]
    coefficients = coefficients.reshape(int(action_horizon), int(action_dim))

    from scipy.fft import idct  # lazy: only a detokenizing policy needs scipy

    return idct(coefficients / fast_tokenizer.scale, axis=0, norm="ortho")


def _default_image_keys(num_views: int) -> tuple:
    """Name the cameras a checkpoint consumes when the caller names none.

    The model's own :data:`~apxinf.policies.base.VIEW_SLOTS` vocabulary, first
    ``num_views`` entries — the same slots π0-FAST was trained on.
    """
    if num_views <= len(VIEW_SLOTS):
        return tuple(VIEW_SLOTS[:num_views])
    extra = [f"view_{index}_rgb" for index in range(len(VIEW_SLOTS), num_views)]
    return tuple(VIEW_SLOTS) + tuple(extra)


@register_policy("pi0_fast")
@register_policy("pi0fast")
class Pi0FastPolicy:
    """Token-VLA policy: raw observation dict -> unnormalized action chunk."""

    def __init__(
        self,
        model,
        *,
        tokenizer: _PaligemmaTokenizer,
        fast_tokenizer: _FastActionTokenizer,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        action_mean: np.ndarray,
        action_std: np.ndarray,
        image_pipeline: Pipeline,
        image_keys: Sequence[str],
        state_key: Optional[str],
        prompt_key: str,
        action_dim: int,
        action_horizon: int,
        fast_skip_tokens: int,
        max_state_dim: int,
        max_action_tokens: int,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.fast_tokenizer = fast_tokenizer
        self.state_mean = np.asarray(state_mean, dtype=np.float32)
        self.state_std = np.asarray(state_std, dtype=np.float32)
        self.action_mean = np.asarray(action_mean, dtype=np.float32)[:action_dim]
        self.action_std = np.asarray(action_std, dtype=np.float32)[:action_dim]
        self.input_pipeline = image_pipeline
        self.image_keys = tuple(image_keys)
        self.state_key = state_key
        self.prompt_key = prompt_key
        self.fast_skip_tokens = int(fast_skip_tokens)
        self.max_state_dim = int(max_state_dim)
        self.max_action_tokens = int(max_action_tokens)
        self._action_dim = int(action_dim)
        self._action_horizon = int(action_horizon)

        if self.state_mean.shape != self.state_std.shape:
            raise ValueError(
                f"Pi0FastPolicy: state statistics disagree, mean {self.state_mean.shape} "
                f"vs std {self.state_std.shape}"
            )
        if self.action_mean.shape != self.action_std.shape:
            raise ValueError(
                f"Pi0FastPolicy: action statistics disagree, mean {self.action_mean.shape} "
                f"vs std {self.action_std.shape}"
            )
        if self.state_mean.size > self.max_state_dim:
            raise ValueError(
                f"Pi0FastPolicy: state statistics are {self.state_mean.size}-wide but "
                f"max_state_dim is {self.max_state_dim}"
            )

        self._extra_metadata = dict(metadata) if metadata else {}
        self.metadata = {**self._derived_metadata(), **self._extra_metadata}

    def _derived_metadata(self) -> dict:
        return {
            "model_type": "pi0_fast",
            "action_horizon": self._action_horizon,
            "action_dim": self._action_dim,
            "state_dim": int(self.state_mean.size),
            "max_state_dim": self.max_state_dim,
            "image_size": [self.model.image_size, self.model.image_size],
            "num_views": self.model.num_views,
            "image_keys": list(self.image_keys),
            "state_key": self.state_key,
            "prompt_key": self.prompt_key,
            "discrete_state": True,
            "action_space": "fast_tokens",
            "max_action_tokens": self.max_action_tokens,
            "fast_skip_tokens": self.fast_skip_tokens,
            "input_pipeline": self.input_pipeline.names,
        }

    # --- construction ------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        model=None,
        checkpoint=None,
        device: str = "cuda:0",
        precision: str = "auto",
        action_dim: Optional[int] = None,
        action_horizon: Optional[int] = None,
        num_views: Optional[int] = None,
        state_key: Optional[str] = None,
        prompt_key: str = _PROMPT_KEY,
        image_keys: Optional[Sequence[str]] = None,
        image_pipeline: Optional[Pipeline] = None,
        tokenizer_path=None,
        fast_tokenizer_path=None,
        seed: int = 0,
        metadata: Optional[Mapping[str, Any]] = None,
        calibration=None,
        tactics=None,
        norm_stats=None,
        norm_key=None,
        num_flow_steps=None,
        flow_start_time=None,
        discrete_state: Optional[bool] = None,
        **kwargs,
    ) -> "Pi0FastPolicy":
        """Load a LeRobot π0-FAST checkpoint as an L2 policy.

        Every flag that reaches ``Pi05Policy`` reaches this constructor too (the
        evaluator and ``AutoPolicy`` are model-agnostic), so the ones that cannot
        mean anything here are rejected *loudly* rather than ignored: a silently
        dropped calibration file or normalization override would look like an
        accuracy result.
        """
        if kwargs:
            raise TypeError(
                f"Pi0FastPolicy.from_pretrained: unsupported options {sorted(kwargs)}"
            )
        if calibration is not None or tactics is not None:
            raise NotImplementedError(
                "Pi0FastPolicy.from_pretrained: π0-FAST is a BF16 token decoder; "
                "FP8 calibration and tactic search apply to the PI0.5 flow runtime"
            )
        if norm_stats is not None or norm_key is not None:
            raise NotImplementedError(
                "Pi0FastPolicy.from_pretrained: π0-FAST unnormalizes with the "
                "checkpoint's own policy_postprocessor_step_0_unnormalizer_processor."
                "safetensors; pass a checkpoint that carries it instead of OpenPI-style "
                "norm_stats.json"
            )
        if num_flow_steps is not None or flow_start_time is not None:
            raise NotImplementedError(
                "Pi0FastPolicy.from_pretrained: π0-FAST decodes tokens "
                "autoregressively; it has no flow-matching steps to override"
            )
        if discrete_state is False:
            raise ValueError(
                "Pi0FastPolicy.from_pretrained: π0-FAST always discretizes state "
                "into the prompt; discrete_state=False is unsupported"
            )

        model_dir = pathlib.Path(model_dir)
        config_path = model_dir / "config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Pi0FastPolicy.from_pretrained: {model_dir} has no config.json; "
                "a LeRobot π0-FAST checkpoint declares chunk_size, max_action_tokens "
                "and its tokenizer names there"
            )
        config = json.loads(config_path.read_text())

        resolved_horizon = int(
            action_horizon
            if action_horizon is not None
            else config.get("chunk_size", _DEFAULT_ACTION_HORIZON)
        )
        resolved_dim = int(
            action_dim
            if action_dim is not None
            else _config_action_dim(config, _DEFAULT_ACTION_DIM)
        )

        if model is None:
            import apxinf_py

            path = str(checkpoint) if checkpoint is not None else str(model_dir / "model.safetensors")
            model = apxinf_py.Model.load(
                "pi0_fast-cuda", path, device=device, precision=precision,
                sampling_seed=int(seed),
            )

        if num_views is not None and int(num_views) != int(model.num_views):
            raise ValueError(
                f"Pi0FastPolicy.from_pretrained: num_views={num_views} but the "
                f"checkpoint serves {model.num_views} cameras; π0-FAST takes its view "
                "count from the checkpoint's config.json"
            )

        resolved_image_keys = tuple(
            image_keys if image_keys is not None else _default_image_keys(model.num_views)
        )
        if len(resolved_image_keys) != int(model.num_views):
            raise ValueError(
                f"Pi0FastPolicy.from_pretrained: {len(resolved_image_keys)} image_keys "
                f"{list(resolved_image_keys)} for {model.num_views} model views. Supply "
                "one wire key per camera slot, in slot order."
            )
        if state_key is None:
            raise ValueError(
                "Pi0FastPolicy.from_pretrained: π0-FAST conditions on proprioception "
                "through the prompt, so state_key is required. Name the wire key your "
                "client sends (see apxinf.conventions, or a robot preset)."
            )

        normalizer = _read_safetensors_f32(
            _normalizer_state_file(model_dir, "policy_preprocessor_step_2_normalizer_processor"),
            ["observation.state.mean", "observation.state.std"],
        )
        unnormalizer = _read_safetensors_f32(
            _normalizer_state_file(model_dir, "policy_postprocessor_step_0_unnormalizer_processor"),
            ["action.mean", "action.std"],
        )
        action_mean = unnormalizer["action.mean"].reshape(-1)
        action_std = unnormalizer["action.std"].reshape(-1)
        if action_mean.size < resolved_dim:
            raise ValueError(
                f"Pi0FastPolicy.from_pretrained: action_dim={resolved_dim} exceeds the "
                f"checkpoint's {action_mean.size}-wide action statistics"
            )

        preprocessor_names = _preprocessor_tokenizer_names(model_dir)
        text_tokenizer = (
            config.get("text_tokenizer_name")
            or preprocessor_names.get("text")
            or _DEFAULT_TEXT_TOKENIZER
        )
        action_tokenizer = (
            config.get("action_tokenizer_name") or preprocessor_names.get("action")
        )
        if not action_tokenizer:
            raise ValueError(
                f"Pi0FastPolicy.from_pretrained: {config_path} names no "
                "action_tokenizer_name and the preprocessor declares no "
                "action_tokenizer_processor, so the FAST BPE detokenizer cannot be "
                "resolved; pass fast_tokenizer_path explicitly"
            )

        tokenizer = _PaligemmaTokenizer(
            _resolve_asset(
                str(tokenizer_path) if tokenizer_path is not None else str(text_tokenizer),
                "tokenizer.json",
                env="APXINF_PALIGEMMA_TOKENIZER",
                what="(text) tokenizer",
            ),
            max_length=int(config.get("tokenizer_max_length", _DEFAULT_TOKENIZER_MAX_LENGTH)),
        )
        fast_tokenizer = _FastActionTokenizer(
            _resolve_asset(
                str(fast_tokenizer_path) if fast_tokenizer_path is not None else str(action_tokenizer),
                "tokenizer.json",
                env="APXINF_FAST_TOKENIZER",
                what="FAST action tokenizer",
            ).parent
        )

        # LeRobot resizes with `resize_with_pad_torch` (torch bilinear,
        # align_corners=False), not with PIL; the two disagree by up to ~21/255 on
        # LIBERO's 256->224 downscale, which is enough to change the emitted FAST
        # tokens. Match the reference interpolator, not PI0.5's.
        image_pipeline = image_pipeline or Pipeline(
            [("parse", ParseImage()), ("resize", TorchResizeWithPad(int(model.image_size)))]
        )
        return cls(
            model,
            tokenizer=tokenizer,
            fast_tokenizer=fast_tokenizer,
            state_mean=normalizer["observation.state.mean"].reshape(-1),
            state_std=normalizer["observation.state.std"].reshape(-1),
            action_mean=action_mean,
            action_std=action_std,
            image_pipeline=Pipeline(
                [
                    (
                        "image_stack",
                        ImageStack(image_pipeline, resolved_image_keys, int(model.image_size)),
                    )
                ]
            ),
            image_keys=resolved_image_keys,
            state_key=state_key,
            prompt_key=prompt_key,
            action_dim=resolved_dim,
            action_horizon=resolved_horizon,
            fast_skip_tokens=int(config.get("fast_skip_tokens", _DEFAULT_FAST_SKIP_TOKENS)),
            max_state_dim=int(config.get("max_state_dim", _DEFAULT_MAX_STATE_DIM)),
            max_action_tokens=int(config.get("max_action_tokens", _DEFAULT_MAX_ACTION_TOKENS)),
            metadata=metadata,
        )

    # --- inference ---------------------------------------------------------

    def _require_keys(self, observation: Mapping[str, Any]) -> None:
        missing = [
            key
            for key in (*self.image_keys, self.state_key, self.prompt_key)
            if not _has_key(observation, key)
        ]
        if missing:
            raise KeyError(
                f"Pi0FastPolicy: observation is missing {missing}; this checkpoint "
                f"consumes image_keys={list(self.image_keys)}, state_key="
                f"{self.state_key!r}, prompt_key={self.prompt_key!r}"
            )

    def _prompt_ids(self, observation: Mapping[str, Any], prompt: str) -> np.ndarray:
        """Build the reference prompt text and its id vector (BOS at both ends)."""
        raw_state = np.asarray(lookup_key(observation, self.state_key), dtype=np.float32).reshape(-1)
        expected = self.state_mean.size
        if raw_state.size != expected:
            raise ValueError(
                f"Pi0FastPolicy: {self.state_key!r} has {raw_state.size} values but this "
                f"checkpoint was trained on {expected} (see metadata['state_dim']); the "
                "state is discretized into the prompt, so a wrong width is a wrong prompt"
            )
        # LeRobot normalizes in float32 with an epsilon on the divisor, then pads
        # to max_state_dim *after* normalizing. Padded zeros are part of the
        # prompt the model saw in training, so they are padded, not dropped.
        normalized = (raw_state - self.state_mean) / (self.state_std + _MEAN_STD_EPS)
        padded = np.zeros(self.max_state_dim, dtype=np.float32)
        padded[: normalized.size] = normalized
        bins = " ".join(str(int(value)) for value in discretize_state(padded))
        task = prompt.strip().replace("_", " ").replace("\n", " ")
        return np.asarray(self.tokenizer.prompt_ids(f"Task: {task}, State: {bins};\n"), dtype=np.uint32)

    def infer(self, observation: Mapping[str, Any], *, noise: Optional[np.ndarray] = None) -> dict:
        """Preprocess -> token decode -> FAST detokenization -> unnormalize.

        Returns ``actions`` (deployable ``float32`` ``[horizon, action_dim]``),
        ``normalized_actions`` (the detokenized chunk in the checkpoint's action
        space), the raw ``action_tokens``, and a ``timing`` dict. π0-FAST
        argmax-decodes, so there is no continuous latent: an explicit ``noise``
        is refused rather than quietly ignored.
        """
        if noise is not None:
            raise ValueError(
                "Pi0FastPolicy.infer: π0-FAST decodes action tokens by argmax and has "
                "no continuous latent to seed; noise/warm-start does not apply"
            )
        if not isinstance(observation, Mapping):
            raise TypeError(f"observation must be a mapping, got {type(observation)!r}")
        self._require_keys(observation)

        started = time.perf_counter()
        prompt = lookup_key(observation, self.prompt_key)
        if not isinstance(prompt, str):
            raise TypeError(f"{self.prompt_key} must be a string, got {type(prompt)!r}")

        token_ids = self._prompt_ids(observation, prompt)
        data = self.input_pipeline({OBSERVATION: observation, PROMPT: prompt})
        rgb = data[RGB]

        model_started = time.perf_counter()
        # Stop at the `|` terminator: the reference and this policy both discard
        # every token after it during detokenization, so ending the decode there
        # returns the identical chunk while skipping ~90% of the autoregressive
        # steps (the terminator lands at token 12-30 of 256 on LIBERO frames).
        tokens = np.asarray(
            self.model.infer_action_tokens_rgb(
                rgb, "nhwc", token_ids, stop_token=int(self.tokenizer.pipe_token_id)
            ),
            dtype=np.uint32,
        )
        model_ms = (time.perf_counter() - model_started) * 1000.0
        if not 0 < tokens.size <= self.max_action_tokens:
            raise RuntimeError(
                f"Pi0FastPolicy: the runtime returned {tokens.size} action tokens, "
                f"expected 1..={self.max_action_tokens}"
            )

        normalized = np.ascontiguousarray(
            detokenize_action_tokens(
                tokens,
                tokenizer=self.tokenizer,
                fast_tokenizer=self.fast_tokenizer,
                fast_skip_tokens=self.fast_skip_tokens,
                action_horizon=self._action_horizon,
                action_dim=self._action_dim,
            ),
            dtype=np.float32,
        )
        if not np.isfinite(normalized).all():
            raise FloatingPointError("Pi0FastPolicy: detokenized a non-finite action chunk")
        actions = np.ascontiguousarray(
            normalized * self.action_std + self.action_mean, dtype=np.float32
        )
        return {
            "actions": actions,
            "normalized_actions": normalized,
            "action_tokens": tokens,
            "prompt_token_ids": token_ids,
            "timing": {
                "model_ms": model_ms,
                "total_ms": (time.perf_counter() - started) * 1000.0,
            },
        }

    @property
    def action_dim(self) -> int:
        return self._action_dim

    @property
    def action_horizon(self) -> int:
        return self._action_horizon

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()


def _config_action_dim(config: Mapping[str, Any], default: int) -> int:
    """The deployable action width the checkpoint declares in ``output_features``."""
    features = config.get("output_features")
    if isinstance(features, Mapping):
        action = features.get("action")
        if isinstance(action, Mapping):
            shape = action.get("shape")
            if isinstance(shape, Sequence) and shape:
                return int(shape[0])
    return int(default)


def _has_key(observation: Mapping[str, Any], key: Optional[str]) -> bool:
    """Whether ``key`` resolves in ``observation`` (flat or nested), like lookup_key."""
    if key is None:
        return False
    try:
        lookup_key(observation, key)
    except KeyError:
        return False
    return True

"""pi05 L2 policy: raw observation dict + prompt -> unnormalized action chunk.

:class:`Pi05Policy` mirrors openpi's policy shape — ``input_pipeline`` (pre) →
``model`` → ``output_pipeline`` (post) — where each pipeline is a
:class:`~apxinf.processors.Pipeline` whose flowing value is a **data dict** and
each step is a ``dict -> dict`` :class:`~apxinf.processors.ProcessorStep` (see
:mod:`apxinf.processors.transforms`). The model inference itself is *not* a
pipeline step: it is the policy's own middle step, called directly on the
``apxinf_py`` bare-model handle, exactly as openpi calls ``sample_actions``
between its input and output transforms.

This makes the whole pre/post chain reorderable, insertable, and replaceable
through the ordinary ``Pipeline`` machinery — a custom high-performance resize or
tokenizer drops in with ``input_pipeline.replace(...)`` / ``insert_after(...)``
without forking the framework.

Domain contract: the model returns a **normalized-domain** action; this policy
returns the **unnormalized-domain** chunk. The intermediate normalized action is
also returned (``normalized_actions``) so the layering invariant
``L2 minus unnormalize == L1`` can be checked directly.

**State injection (opt-in, off by default):** ``observation/state`` is dropped by
default so the numerics match today's serving link. Enable it with
``discrete_state=True``: the raw state is first mapped to ``[-1, 1]`` by a
``state_normalizer`` (a :class:`~apxinf.processors.Normalizer` over
``norm_stats["state"]`` by default), then discretized into the prompt — matching
openpi's "normalize then discretize" order. This path does **not** assume the
incoming state is already in ``[-1, 1]``.

This module registers ``Pi05Policy`` under ``model_type="pi05"`` so
:class:`~apxinf.policies.auto.AutoPolicy` can dispatch to it.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..base import BareModel
from ...processors import (
    GaussianNoise,
    ImageStack,
    Normalizer,
    ParseImage,
    Pipeline,
    PromptTokenizer,
    ResizeWithPad,
    SampleNoise,
    SyntheticTokenizer,
    Tokenize,
    Trim,
    Unnormalizer,
)
from ...processors.transforms import (
    ACTIONS,
    NOISE,
    NORMALIZED,
    OBSERVATION,
    PROMPT,
    RGB,
    TOKEN_IDS,
    Unnormalize,
)
from ..registry import register_policy

__all__ = ["Pi05Policy"]

_DEFAULT_IMAGE_KEYS = ("observation/image", "observation/wrist_image")
_STATE_KEY = "observation/state"
_PROMPT_KEY = "prompt"


@register_policy("pi05")
class Pi05Policy:
    """Compose a pre pipeline + a bare model + a post pipeline into one call."""

    def __init__(
        self,
        model: BareModel,
        *,
        input_pipeline: Pipeline,
        output_pipeline: Pipeline,
        image_keys: Sequence[str] = _DEFAULT_IMAGE_KEYS,
        prompt_key: str = _PROMPT_KEY,
        state_key: str = _STATE_KEY,
        action_dim: Optional[int] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ):
        self.model = model
        self.input_pipeline = input_pipeline
        self.output_pipeline = output_pipeline
        self.image_keys = tuple(image_keys)
        self.prompt_key = prompt_key
        self.state_key = state_key
        self.action_dim_out = (
            int(action_dim) if action_dim is not None else self._derive_action_dim(output_pipeline)
        )

        # Introspect the optional tokenize step so ``_require_keys`` and metadata
        # know whether state is actually injected (a custom pipeline may omit it).
        tokenize = self._find_step(input_pipeline, "tokenize")
        tokenizer = getattr(tokenize, "tokenizer", None)
        self.discrete_state = bool(getattr(tokenizer, "discrete_state", False))
        state_normalized = getattr(tokenize, "state_normalizer", None) is not None

        self.metadata = {
            "model_type": "pi05",
            "action_horizon": model.action_horizon,
            "action_dim": self.action_dim_out,
            "model_action_dim": model.action_dim,
            "num_views": model.num_views,
            "image_size": [model.image_size, model.image_size],
            "image_keys": list(self.image_keys),
            "discrete_state": self.discrete_state,
            "state_normalized": state_normalized,
            "input_pipeline": input_pipeline.names,
            "output_pipeline": output_pipeline.names,
            **(dict(metadata) if metadata else {}),
        }

    # --- construction ------------------------------------------------------

    @classmethod
    def default_pipelines(
        cls,
        model: BareModel,
        *,
        tokenizer: PromptTokenizer,
        unnormalizer: Unnormalizer,
        image_pipeline: Optional[Pipeline] = None,
        noise: Optional[GaussianNoise] = None,
        state_normalizer: Optional[Normalizer] = None,
        image_keys: Sequence[str] = _DEFAULT_IMAGE_KEYS,
        state_key: str = _STATE_KEY,
    ) -> Tuple[Pipeline, Pipeline]:
        """Assemble the default ``(input_pipeline, output_pipeline)`` from parts.

        pi05 runs the exact camera set the checkpoint declares (``model.num_views``,
        parsed from the config's ``input_features``). The task's ``image_keys`` must
        name precisely those cameras — no more, no fewer. Absent cameras are never
        sent, so there is no padding: the model runs the real view shape directly.
        """
        image_keys = tuple(image_keys)
        if len(image_keys) != model.num_views:
            raise ValueError(
                f"Pi05Policy: model expects {model.num_views} camera views but "
                f"{len(image_keys)} image_keys were given: {image_keys}. Supply "
                f"exactly the checkpoint's cameras (real views only, no padding)."
            )
        image_pipeline = image_pipeline or Pipeline(
            [("parse", ParseImage()), ("resize", ResizeWithPad(model.image_size))]
        )
        input_steps = [
            ("image_stack", ImageStack(image_pipeline, image_keys, model.image_size)),
            ("tokenize", Tokenize(tokenizer, state_normalizer, state_key)),
        ]
        # The default leaves noise absent so the binding fills the stable latent
        # buffer with its backend-native generator. Supplying an explicit sampler
        # preserves the old host-generated/custom-processor path.
        if noise is not None:
            input_steps.append(("sample_noise", SampleNoise(noise)))
        input_pipeline = Pipeline(input_steps)
        output_pipeline = Pipeline(
            [
                ("trim", Trim(unnormalizer.width)),
                ("unnormalize", Unnormalize(unnormalizer)),
            ]
        )
        return input_pipeline, output_pipeline

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        model: Optional[BareModel] = None,
        model_name: str = "pi05",
        checkpoint=None,
        device: str = "cuda:0",
        precision: str = "auto",
        calibration=None,
        tactics=None,
        tokenizer_path=None,
        norm_key: str = "actions",
        action_dim: Optional[int] = None,
        action_horizon: Optional[int] = None,
        seed: int = 0,
        discrete_state: bool = False,
        state_norm_key: str = "state",
        image_pipeline: Optional[Pipeline] = None,
        image_keys: Sequence[str] = _DEFAULT_IMAGE_KEYS,
        prompt_key: str = _PROMPT_KEY,
        state_key: str = _STATE_KEY,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "Pi05Policy":
        """Build the **default** policy from a checkpoint directory.

        This is the from-disk convenience path: it loads the ``apxinf_py`` model
        (unless one is passed in), builds the SentencePiece tokenizer and the
        action quantiles from files under ``model_dir``, assembles the default
        pre/post chains via :meth:`default_pipelines`, and constructs the policy.
        Its many parameters are exactly the knobs for *building processors from
        disk* (``norm_key``/``action_dim``/``discrete_state``/``tokenizer_path``/
        ``state_norm_key``) — only this method knows ``model_dir``.

        ``action_dim`` trims the unnormalizer to the task's action width (e.g. 7
        for LIBERO); ``None`` keeps the full vector. ``action_horizon`` overrides
        the checkpoint's chunk length: ``None`` runs the native ``config.json``
        value, an explicit value outranks it (the horizon is a sequence length,
        not a weight dimension, so the same weights run at any horizon the config
        validator accepts). State injection is off by default; with
        ``discrete_state=True`` a state normalizer is built from
        ``norm_stats[state_norm_key]`` to map raw state to ``[-1, 1]`` before it is
        discretized into the prompt.

        For a **fully custom** pre/post chain, do not funnel it through here:
        build the parts yourself and use :meth:`default_pipelines` +
        :meth:`__init__` (or mutate ``policy.input_pipeline`` after construction).
        """
        model_dir = Path(model_dir)
        if model is None:
            import apxinf_py  # lazy: processor-only users never import the binding

            ckpt = str(checkpoint) if checkpoint is not None else str(model_dir / "model.safetensors")
            model = apxinf_py.Model.load(
                model_name,
                ckpt,
                device=device,
                precision=precision,
                **({"calibration": str(calibration)} if calibration else {}),
                **({"tactics": str(tactics)} if tactics else {}),
                **({"action_horizon": int(action_horizon)} if action_horizon else {}),
                sampling_seed=int(seed),
            )
        elif action_horizon is not None and int(action_horizon) != int(model.action_horizon):
            # A pre-built model carries its own horizon; silently ignoring the
            # override here would hand back a policy that disagrees with the flag.
            raise ValueError(
                f"Pi05Policy.from_pretrained: action_horizon={action_horizon} conflicts "
                f"with the supplied model's horizon {model.action_horizon}; pass the "
                f"override to the model constructor instead"
            )

        tokenizer = PromptTokenizer(
            _resolve_tokenizer(model_dir, tokenizer_path),
            max_token_len=model.max_token_len if hasattr(model, "max_token_len") else 200,
            discrete_state=discrete_state,
        )
        unnormalizer = Unnormalizer.from_norm_stats(model_dir, key=norm_key, dims=action_dim)
        state_normalizer = (
            Normalizer.from_norm_stats(model_dir, key=state_norm_key) if discrete_state else None
        )
        reset_sampling = getattr(model, "reset_sampling", None)
        if callable(reset_sampling):
            reset_sampling(int(seed))

        input_pipeline, output_pipeline = cls.default_pipelines(
            model,
            tokenizer=tokenizer,
            unnormalizer=unnormalizer,
            image_pipeline=image_pipeline,
            state_normalizer=state_normalizer,
            image_keys=image_keys,
            state_key=state_key,
        )

        return cls(
            model,
            input_pipeline=input_pipeline,
            output_pipeline=output_pipeline,
            image_keys=image_keys,
            prompt_key=prompt_key,
            state_key=state_key,
            action_dim=unnormalizer.width,
            metadata=metadata,
        )

    @classmethod
    def from_random(
        cls,
        model: BareModel,
        *,
        token_count: Optional[int] = None,
        action_dim: Optional[int] = None,
        seed: int = 0,
        image_keys: Optional[Sequence[str]] = None,
        prompt_key: str = _PROMPT_KEY,
        state_key: str = _STATE_KEY,
        metadata: Optional[Mapping[str, Any]] = None,
        warn: bool = True,
    ) -> "Pi05Policy":
        """Wrap a **checkpoint-free** random-weight model in synthetic processors.

        The full L2 (and, when served, L3) latency of the engine can be measured
        with no data files on disk: the tokenizer is a fixed-length
        :class:`~apxinf.processors.SyntheticTokenizer` and the unnormalizer is the
        identity map (``q01=-1, q99=1, eps=0``). Pair with
        ``apxinf_py.Model.random(...)`` for a fully checkpoint-free L2/L3.

        The returned actions are **latency-only and numerically meaningless** (the
        weights, tokens, and unnormalization are all synthetic); a warning says so
        unless ``warn=False``. ``token_count`` sets the synthetic prompt length
        (defaults to ``min(10, max_token_len)``); ``action_dim`` trims the output
        width (``None`` keeps the model's full action vector).
        """
        if warn:
            warnings.warn(
                "Pi05Policy.from_random: synthetic tokenizer + identity unnormalizer; "
                "actions are latency-only and numerically meaningless.",
                stacklevel=2,
            )
        max_token_len = int(getattr(model, "max_token_len", 200))
        if token_count is None:
            token_count = min(10, max_token_len)
        tokenizer = SyntheticTokenizer(token_count, max_token_len=max_token_len)

        width = int(action_dim) if action_dim is not None else int(model.action_dim)
        # Identity quantile map: with eps=0, unnormalize is (x + 1) * 1 + (-1) == x.
        unnormalizer = Unnormalizer(q01=[-1.0] * width, q99=[1.0] * width, dims=width, eps=0.0)
        reset_sampling = getattr(model, "reset_sampling", None)
        if callable(reset_sampling):
            reset_sampling(int(seed))

        if image_keys is None:
            image_keys = _synthetic_image_keys(model.num_views)

        input_pipeline, output_pipeline = cls.default_pipelines(
            model,
            tokenizer=tokenizer,
            unnormalizer=unnormalizer,
            image_keys=image_keys,
        )
        return cls(
            model,
            input_pipeline=input_pipeline,
            output_pipeline=output_pipeline,
            image_keys=image_keys,
            prompt_key=prompt_key,
            state_key=state_key,
            action_dim=width,
            metadata={"weights": "synthetic", **(dict(metadata) if metadata else {})},
        )

    # --- inference ---------------------------------------------------------

    def infer(self, observation: Mapping[str, Any], *, noise: Optional[np.ndarray] = None) -> dict:
        """Run pre-pipeline -> model -> post-pipeline on one raw observation dict.

        Returns ``actions`` (unnormalized ``float32`` ``[horizon, action_dim]``),
        ``normalized_actions`` (the model's raw output), ``token_ids``, the
        caller-provided ``noise`` (or ``None`` for internal sampling), and a
        ``timing`` dict distinguishing pure-model from end-to-end latency.

        ``noise`` is optional. An explicit keyword wins over ``observation["noise"]``
        and over a custom input-pipeline sampler. If all are absent, the bare model
        generates standard-normal noise directly in its device buffer.
        """
        started = time.perf_counter()
        if not isinstance(observation, Mapping):
            raise TypeError(f"observation must be a mapping, got {type(observation)!r}")
        self._require_keys(observation)

        prompt = observation[self.prompt_key]
        if not isinstance(prompt, str):
            raise TypeError(f"{self.prompt_key} must be a string, got {type(prompt)!r}")

        # pre: obs dict -> model inputs (rgb / token_ids / optional noise)
        data = self.input_pipeline({OBSERVATION: observation, PROMPT: prompt})
        rgb = data[RGB]
        token_ids = data[TOKEN_IDS]
        selected_noise = noise
        if selected_noise is None:
            selected_noise = observation.get(NOISE)
        if selected_noise is None:
            selected_noise = data.get(NOISE)
        if selected_noise is not None:
            selected_noise = np.ascontiguousarray(selected_noise, dtype=np.float32)
            expected_noise = (self.model.action_horizon, self.model.action_dim)
            if selected_noise.shape != expected_noise:
                raise ValueError(
                    f"noise shape {selected_noise.shape}, expected {expected_noise}"
                )
            if not np.isfinite(selected_noise).all():
                raise ValueError("noise must contain only finite values")

        # model: the policy's own middle step (not a pipeline stage)
        model_started = time.perf_counter()
        normalized = np.asarray(
            self.model.infer_rgb(rgb, "nhwc", token_ids, selected_noise), dtype=np.float32
        )
        model_ms = (time.perf_counter() - model_started) * 1000.0

        expected = (self.model.action_horizon, self.model.action_dim)
        if normalized.shape != expected:
            raise ValueError(f"model returned action shape {normalized.shape}, expected {expected}")
        if not np.isfinite(normalized).all():
            raise FloatingPointError("model returned non-finite normalized actions")

        # post: normalized action -> deployable unnormalized action. The
        # post-input-decode observation is threaded in so state-dependent output
        # steps (e.g. a robot adapter's delta->absolute) can read it; the default
        # ``trim`` / ``unnormalize`` steps ignore it. Mirrors openpi, whose output
        # transforms see the same ``state`` its input transforms produced.
        processed_obs = data.get(OBSERVATION, observation)
        out = self.output_pipeline({NORMALIZED: normalized, OBSERVATION: processed_obs})
        actions = out[ACTIONS]
        total_ms = (time.perf_counter() - started) * 1000.0

        return {
            "actions": actions,
            "normalized_actions": normalized,
            "token_ids": token_ids,
            "noise": selected_noise,
            "timing": {"model_ms": model_ms, "total_ms": total_ms},
            "metadata": self.metadata,
        }

    __call__ = infer

    @property
    def action_dim(self) -> int:
        """Width of one deployable action vector (after trim + unnormalize)."""
        return self.action_dim_out

    @property
    def action_horizon(self) -> int:
        """Number of actions in one predicted chunk."""
        return self.model.action_horizon

    def close(self) -> None:
        close = getattr(self.model, "close", None)
        if callable(close):
            close()

    # --- helpers -----------------------------------------------------------

    def _require_keys(self, observation: Mapping[str, Any]) -> None:
        required = list(self.image_keys) + [self.prompt_key]
        if self.discrete_state:
            # State is only mandatory when it is actually injected into the prompt.
            required.append(self.state_key)
        missing = [key for key in required if key not in observation]
        if missing:
            raise KeyError(f"Pi05Policy.infer: missing observation keys: {missing}")

    @staticmethod
    def _find_step(pipeline: Pipeline, name: str):
        try:
            return pipeline[name]
        except KeyError:
            return None

    @staticmethod
    def _derive_action_dim(output_pipeline: Pipeline) -> int:
        """Infer the deployable action width from the default post-chain steps.

        The default ``output_pipeline`` names its steps ``trim`` / ``unnormalize``;
        read the width from whichever is present. A custom post-chain that uses
        different names must pass ``action_dim=`` to the constructor.
        """
        trim = Pi05Policy._find_step(output_pipeline, "trim")
        if trim is not None and hasattr(trim, "action_dim"):
            return int(trim.action_dim)
        unnormalize = Pi05Policy._find_step(output_pipeline, "unnormalize")
        unnormalizer = getattr(unnormalize, "unnormalizer", None)
        if unnormalizer is not None and hasattr(unnormalizer, "width"):
            return int(unnormalizer.width)
        raise ValueError(
            "Pi05Policy: cannot infer action_dim from output_pipeline "
            f"{output_pipeline.names}; pass action_dim= explicitly"
        )


def _synthetic_image_keys(num_views: int) -> Tuple[str, ...]:
    """Generate exactly ``num_views`` camera keys for a checkpoint-free policy.

    Reuses the default ``image``/``wrist_image`` names for the first two views and
    appends ``image_2``, ``image_3``, ... beyond that, so ``default_pipelines``'
    ``len(image_keys) == model.num_views`` contract holds for any view count.
    """
    base = list(_DEFAULT_IMAGE_KEYS)
    if num_views <= len(base):
        return tuple(base[:num_views])
    extra = [f"observation/image_{i}" for i in range(len(base), num_views)]
    return tuple(base + extra)


def _resolve_tokenizer(model_dir: Path, tokenizer_path) -> Path:
    if tokenizer_path is not None:
        return Path(tokenizer_path)
    candidates = (model_dir / "tokenizer.model", model_dir / "paligemma_tokenizer.model")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"no SentencePiece tokenizer found under {model_dir}; checked {rendered}")

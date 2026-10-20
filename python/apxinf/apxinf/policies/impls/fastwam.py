"""Fast-WAM action-only policy using the native ModelRunner contract."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ...checkpoints.safetensors_state import load_state_file
from ...processors.transforms import lookup_key
from ..registry import register_policy

_DEFAULT_TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


class _Tokenizer:
    def __init__(self, path):
        import apxinf_py

        document = json.loads(Path(path).read_text())
        # The reference T5 tokenizer rebuilds this pipeline when loading UMT5;
        # the checkpoint JSON alone only splits ordinary spaces. Keep special
        # token handling inside the tokenizer, before whitespace pre-tokenization.
        document["normalizer"] = None
        document["pre_tokenizer"] = {
            "type": "Sequence",
            "pretokenizers": [
                {"type": "WhitespaceSplit"},
                {
                    "type": "Metaspace",
                    "replacement": "▁",
                    "prepend_scheme": "always",
                    "split": True,
                },
            ],
        }
        self.native = apxinf_py.HfTokenizer.from_json(json.dumps(document))
        if (
            self.native.token_to_id("<pad>") != 0
            or self.native.token_to_id("</s>") != 1
        ):
            raise ValueError("Fast-WAM requires the UMT5 pad/EOS vocabulary")

    def encode(self, text):
        # Truncation precedes EOS insertion in the official tokenizer.
        return list(self.native.encode_with_special_tokens(text, False))[:127] + [1]


def _stats(model_dir, pipeline, registry):
    document = json.loads((model_dir / f"policy_{pipeline}.json").read_text())
    candidates = [
        step for step in document["steps"] if step["registry_name"] == registry
    ]
    if len(candidates) != 1:
        raise ValueError(f"Fast-WAM requires one {registry}")
    step = candidates[0]
    config = step["config"]
    if (
        config["norm_map"].get("STATE", "MIN_MAX") != "MIN_MAX"
        or config["norm_map"].get("ACTION") != "MIN_MAX"
    ):
        raise ValueError("Fast-WAM requires MIN_MAX state and action normalization")
    name = Path(step["state_file"])
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Fast-WAM processor state must be checkpoint-relative")
    return load_state_file(model_dir / name), np.float32(config.get("eps", 1e-8))


@register_policy("fastwam")
class FastWamPolicy:
    """Two 224x224 RGB uint8 views, eight state values, and a task -> 32x7 actions.

    The fixed profile accepts already resized images. Model computation,
    including both frozen encoders and all ten flow steps, runs on CUDA.
    """

    def __init__(
        self,
        model_runner,
        *,
        tokenizer,
        state_min,
        state_max,
        action_min,
        action_max,
        image_keys,
        state_key="observation.state",
        prompt_key="task",
        prompt_template=_DEFAULT_TEMPLATE,
        toggle_dimensions=(-1,),
        state_eps=1e-8,
        action_eps=1e-8,
    ):
        self.model_runner = model_runner
        self.tokenizer = tokenizer
        self.image_keys = tuple(image_keys)
        self.state_key, self.prompt_key = state_key, prompt_key
        self.prompt_template = prompt_template
        self.state_min, self.state_max = self._pair(state_min, state_max, 8)
        self.action_min, self.action_max = self._pair(action_min, action_max, 7)
        self.state_denom = np.where(
            self.state_max == self.state_min,
            np.float32(state_eps),
            self.state_max - self.state_min,
        )
        self.action_denom = np.where(
            self.action_max == self.action_min,
            np.float32(action_eps),
            self.action_max - self.action_min,
        )
        self.toggle_dimensions = tuple(toggle_dimensions)
        if any(not -7 <= dim < 7 for dim in self.toggle_dimensions):
            raise ValueError("Fast-WAM action toggle dimension is out of bounds")
        if len(self.image_keys) != 2 or len(set(self.image_keys)) != 2:
            raise ValueError("Fast-WAM requires two distinct ordered image keys")
        if (
            model_runner.action_horizon,
            model_runner.action_dim,
            model_runner.num_views,
            model_runner.image_size,
            model_runner.max_token_len,
        ) != (32, 7, 2, 224, 128):
            raise ValueError("Fast-WAM native runner profile does not match the policy")
        self.metadata = dict(
            model_type="fastwam",
            model_variant="bf16",
            action_horizon=32,
            action_dim=7,
            image_keys=list(self.image_keys),
            state_key=state_key,
            state_dim=8,
            prompt_key=prompt_key,
            num_views=2,
            image_size=224,
            max_token_len=128,
        )

    @staticmethod
    def _pair(lower, upper, size):
        lower, upper = (
            np.asarray(lower, dtype=np.float32),
            np.asarray(upper, dtype=np.float32),
        )
        if (
            lower.shape != (size,)
            or upper.shape != (size,)
            or not np.isfinite(lower).all()
            or not np.isfinite(upper).all()
            or np.any(upper < lower)
        ):
            raise ValueError(
                "Fast-WAM normalization statistics have invalid shape or bounds"
            )
        return lower, upper

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        *,
        device="cuda:0",
        seed=0,
        model_runner=None,
        assets=None,
        tokenizer=None,
        model_variant="bf16",
        autotune=False,
        tactics=None,
        image_keys=None,
        state_key="observation.state",
        prompt_key="task",
        prompt_template=None,
    ):
        model_dir = Path(model_dir)
        config = json.loads((model_dir / "config.json").read_text())
        if config.get("type") != "fastwam" or model_variant not in ("auto", "bf16"):
            raise ValueError("Fast-WAM requires a fastwam checkpoint and BF16 variant")
        assets = {key: str(value) for key, value in (assets or {}).items()}
        if tokenizer is None:
            tokenizer_dir = Path(assets.get("tokenizer", model_dir / "tokenizer"))
            tokenizer = _Tokenizer(tokenizer_dir / "tokenizer.json")
        state, state_eps = _stats(model_dir, "preprocessor", "normalizer_processor")
        action, action_eps = _stats(
            model_dir, "postprocessor", "unnormalizer_processor"
        )
        if model_runner is None:
            import apxinf_py

            model_runner = apxinf_py.ModelRunner.load(
                "fastwam",
                model_dir,
                device=device,
                model_variant=model_variant,
                sampling_seed=int(seed),
                assets=assets,
                autotune=bool(autotune),
                tactics=None if tactics is None else str(tactics),
            )
        keys = (
            image_keys
            if image_keys is not None
            else sorted(
                key
                for key, value in config["input_features"].items()
                if value["type"] == "VISUAL"
            )
        )
        return cls(
            model_runner,
            tokenizer=tokenizer,
            image_keys=keys,
            state_key=state_key,
            prompt_key=prompt_key,
            prompt_template=prompt_template
            or config.get("prompt_template", _DEFAULT_TEMPLATE),
            state_min=state["observation.state.min"],
            state_max=state["observation.state.max"],
            action_min=action["action.min"],
            action_max=action["action.max"],
            toggle_dimensions=config.get("toggle_action_dimensions", []),
            state_eps=state_eps,
            action_eps=action_eps,
        )

    def infer(self, observation, *, noise=None):
        started = time.perf_counter()
        images = []
        for key in self.image_keys:
            image = np.asarray(lookup_key(observation, key))
            if image.shape == (3, 224, 224):
                image = image.transpose(1, 2, 0)
            if image.shape != (224, 224, 3) or image.dtype != np.uint8:
                raise ValueError(
                    f"Fast-WAM image {key!r} must be uint8 HWC/CHW at 224x224"
                )
            images.append(image)
        rgb = np.ascontiguousarray(np.stack(images))
        state = np.asarray(lookup_key(observation, self.state_key), dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("Fast-WAM requires eight finite state values")
        state = np.ascontiguousarray(
            np.float32(2) * (state - self.state_min) / self.state_denom - np.float32(1)
        )
        task = lookup_key(observation, self.prompt_key)
        if not isinstance(task, str):
            raise TypeError("Fast-WAM task must be text")
        encoded = self.tokenizer.encode(self.prompt_template.format(task=task))
        if (
            not encoded
            or len(encoded) > 128
            or encoded[-1] != 1
            or any(token <= 0 or token >= 256384 for token in encoded)
        ):
            raise ValueError(
                "Fast-WAM tokenizer must return at most 128 non-padding IDs ending in EOS"
            )
        ids = np.zeros(128, dtype=np.uint32)
        ids[: len(encoded)] = encoded
        if noise is not None:
            noise = np.ascontiguousarray(noise, dtype=np.float32)
            if noise.shape != (32, 7) or not np.isfinite(noise).all():
                raise ValueError("Fast-WAM noise must be finite [32,7]")
        model_started = time.perf_counter()
        normalized = np.asarray(
            self.model_runner.infer_rgb(rgb, "nhwc", ids, noise, state=state),
            dtype=np.float32,
        )
        model_ms = (time.perf_counter() - model_started) * 1000
        if normalized.shape != (32, 7) or not np.isfinite(normalized).all():
            raise ValueError("Fast-WAM returned invalid normalized actions")
        actions = (normalized + np.float32(1)) / np.float32(
            2
        ) * self.action_denom + self.action_min
        for dim in self.toggle_dimensions:
            actions[..., dim] = np.sign(
                -(actions[..., dim] * np.float32(2) - np.float32(1))
            )
        return dict(
            actions=np.ascontiguousarray(actions),
            normalized_actions=normalized,
            token_ids=ids,
            noise=noise,
            metadata=self.metadata,
            timing=dict(
                model_ms=model_ms, total_ms=(time.perf_counter() - started) * 1000
            ),
        )

    __call__ = infer

    @property
    def action_dim(self):
        return 7

    @property
    def action_horizon(self):
        return 32

    def close(self):
        close = getattr(self.model_runner, "close", None)
        if callable(close):
            close()

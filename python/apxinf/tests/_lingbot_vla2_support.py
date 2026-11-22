"""Shared fixtures for the LingBot-VLA 2.0 policy tests (mock runner, synthetic stats, observations)."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from apxinf.policies.impls import lingbot_vla2 as lv

# Reference inputs for the tests that compare with the upstream project. None of them ships with the
# repository: point these variables at your own checkouts (see ``doc/lingbot-vla2.md``). A test whose
# input is unset, or missing on disk, skips.
UPSTREAM = Path(os.environ.get("LINGBOT_UPSTREAM", "/nonexistent/lingbot-vla-v2"))  # robbyant/lingbot-vla-v2 checkout
MODEL_DIR = Path(os.environ.get("LINGBOT_MODEL_DIR", "/nonexistent/lingbot-vla-v2-6b"))  # checkpoint with tokenizer.json
GOLDEN = Path(os.environ.get("LINGBOT_GOLDEN_DIR", "/nonexistent/lingbot-golden"))  # golden_<tag>.safetensors + .json
QWEN3VL_CFG = Path(os.environ.get("LINGBOT_QWEN3VL_CFG", "/nonexistent/qwen3vl_cfg"))  # preprocessor_config.json, chat_template.json
ROBOTWIN_STATS = UPSTREAM / "assets/norm_stats/robotwin.json"

HORIZON, DIM = 50, 55
RAW_KEYS = {"cam_high": "observation.images.cam_high", "cam_left": "observation.images.cam_left_wrist",
            "cam_right": "observation.images.cam_right_wrist"}


class Tokenizer:
    """Deterministic stand-in: one id per character, so prompts differ and lengths vary."""

    def encode(self, text):
        return [1000 + ord(c) % 200 for c in text]


class Runner:
    action_horizon, action_dim = HORIZON, DIM
    initial_latent_shape = (HORIZON, DIM)

    def __init__(self, fn=None):
        self.calls, self.fn = [], fn

    def _infer_preprocessed(self, pixel_values, image_grid_thw, token_ids, attention_mask, state,
                            embodiment_id, noise, **options):
        self.calls.append(SimpleNamespace(pixels=pixel_values, grids=image_grid_thw, tokens=token_ids,
                                          mask=attention_mask, state=state, embodiment=embodiment_id,
                                          noise=noise, options=options))
        if self.fn is not None:
            return self.fn(noise)
        return np.zeros((HORIZON, DIM), np.float32)


def stats_dict(seed=0):
    rng = np.random.default_rng(seed)

    def one(dim):
        lo = rng.normal(size=dim)
        hi = lo + rng.uniform(0.5, 2.0, size=dim)
        return {"mean": rng.normal(size=dim), "std": rng.uniform(0.3, 1.5, size=dim),
                "q01": lo, "q99": hi, "q02": lo + 0.01, "q98": hi - 0.01,
                "min": lo - 0.1, "max": hi + 0.1}

    return {"norm_stats": {"action.arm.position": one(12), "action.effector.position": one(2),
                           "observation.state.arm.position": one(12),
                           "observation.state.effector.position": one(2)}}


def robotwin_observation(rng, prompt="stack the red block on the blue one"):
    return {
        RAW_KEYS["cam_high"]: rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
        RAW_KEYS["cam_left"]: rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
        RAW_KEYS["cam_right"]: rng.integers(0, 256, (240, 320, 3), dtype=np.uint8),
        "observation.state": rng.normal(size=14).astype(np.float32),
        "prompt": prompt,
    }


def make_policy(norm="bounds_99_woclip", robot="robotwin", stats=None, runner=None, joints=None, **kw):
    stats = stats_dict() if stats is None else stats
    return lv.LingbotVla2Policy(
        runner or Runner(), robot=robot, tokenizer=Tokenizer(), norm_stats=stats, norm_type=norm,
        joints=joints or [("arm.position", 14), ("end.position", 14), ("effector.position", 2)], **kw)




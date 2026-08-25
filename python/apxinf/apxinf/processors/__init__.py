"""Pure-numpy pre/post-processing steps for the ApxInf Python frontend.

Every step is an independently-callable :class:`ProcessorStep`; :class:`Pipeline`
chains them. Nothing here imports ``apxinf_py`` or touches CUDA, so the whole
module unit-tests offline (sentencepiece is imported lazily by the tokenizer).
"""

from __future__ import annotations

from .base import Pipeline, ProcessorStep
from .noise import GaussianNoise
from .normalize import Normalizer, Unnormalizer, load_norm_stats
from .resize import ParseImage, ResizeWithPad
from .tokenize import PromptTokenizer, SyntheticTokenizer, build_prompt, discretize_state
from .transforms import ImageStack, SampleNoise, Tokenize, Trim, has_key, lookup_key, set_key
from .transforms import Unnormalize as UnnormalizeStep

__all__ = [
    "ProcessorStep",
    "Pipeline",
    "ParseImage",
    "ResizeWithPad",
    "PromptTokenizer",
    "SyntheticTokenizer",
    "build_prompt",
    "discretize_state",
    "Normalizer",
    "Unnormalizer",
    "load_norm_stats",
    "GaussianNoise",
    # dict->dict steps for a policy's pre/post pipeline
    "ImageStack",
    "Tokenize",
    "lookup_key",
    "has_key",
    "set_key",
    "SampleNoise",
    "Trim",
    "UnnormalizeStep",
]

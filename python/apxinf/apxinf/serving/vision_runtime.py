from __future__ import annotations

import contextlib
from dataclasses import dataclass
from enum import Enum, auto
import importlib
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any, Callable

if __package__:
    from .text_worker import WorkerFailure, model_manifest
    from .vision_preparation import PreparationError, PreparationIdentity, VisionPreparation
    from .owned_media import MediaError
    from . import vision_identity as identity
else:
    from text_worker import WorkerFailure, model_manifest
    from vision_preparation import PreparationError, PreparationIdentity, VisionPreparation
    from owned_media import MediaError
    import vision_identity as identity


@dataclass(frozen=True)
class _RequestField:
    owner: Any
    name: str
    kind: str


class _StatePhase(Enum):
    IDLE = auto()
    ACTIVE = auto()
    FENCED = auto()


class _ResidentRequestState:
    def __init__(self, model: Any, backend: Any, max_context: int, fields: tuple[_RequestField, ...]):
        self.model = model
        self.backend = backend
        self.max_context = max_context
        self.fields = fields
        self.phase = _StatePhase.IDLE
        self._excluded = {(id(field.owner), field.name) for field in fields}
        self._containers = {(id(field.owner), field.name): getattr(field.owner, field.name)
                            for field in fields if field.kind == "rotary"}
        self._validate_fields(baseline=True)
        self._baseline = self._inventory()

    @classmethod
    def qualify(cls, model: Any, backend: Any, max_context: int) -> _ResidentRequestState:
        def name(value: Any) -> str:
            return f"{type(value).__module__}.{type(value).__name__}"

        if (name(model) != "mlx_vlm.models.minicpmv4_7.minicpmv4_7.Model"
                or name(model.language_model) != "mlx_vlm.models.minicpmv4_7.language.LanguageModel"):
            raise WorkerFailure("model_unavailable", "The model classes do not match the request-state profile.")
        convolutions = [module for _, module in model.named_modules()
                        if name(module) == "mlx_vlm.models.qwen3_5.language.Qwen3_5GatedDeltaNet"]
        if len(convolutions) != 18 or any(tuple(module.conv1d.weight.shape) != (6144, 4, 1)
                                          for module in convolutions):
            raise WorkerFailure("model_unavailable", "The convolution modules do not match the request-state profile.")
        rotary = [module for _, module in model.named_modules()
                  if name(module) == "mlx_vlm.models.qwen3_5.language.Qwen3_5RotaryEmbedding"]
        if len(rotary) != 6 or any(module.dim != 64 or not module.fused_apply for module in rotary):
            raise WorkerFailure("model_unavailable", "The rotary modules do not match the request-state profile.")
        fields = (_RequestField(model.language_model, "_position_ids", "positions"),
                  _RequestField(model.language_model, "_rope_deltas", "deltas"))
        fields += tuple(_RequestField(module, "_qwen3_5_decode_conv_weight", "convolution")
                        for module in convolutions)
        fields += tuple(_RequestField(module, "_compiled_apply", "rotary") for module in rotary)
        return cls(model, backend, max_context, fields)

    @property
    def fenced(self) -> bool:
        return self.phase is _StatePhase.FENCED

    def fence(self) -> None:
        self.phase = _StatePhase.FENCED

    def identity(self) -> dict[str, Any]:
        return {"revision": "minicpmv47-transient-state-v1", "max_context": self.max_context,
                "position_fields": [field.name for field in self.fields if field.kind in ("positions", "deltas")],
                "convolution_cache_field": next(field.name for field in self.fields if field.kind == "convolution"),
                "convolution_module_count": sum(field.kind == "convolution" for field in self.fields),
                "rotary_cache_field": "_compiled_apply",
                "rotary_module_count": sum(field.kind == "rotary" for field in self.fields)}

    def _validate_fields(self, *, baseline: bool) -> None:
        missing = object()
        for field in self.fields:
            state = field.owner.state
            if not isinstance(state, dict):
                raise RuntimeError("The model does not expose the qualified public state mapping.")
            value = getattr(field.owner, field.name, missing)
            if field.kind == "rotary":
                if (type(value) is not dict or state.get(field.name, missing) is not value
                        or value is not self._containers[(id(field.owner), field.name)]
                        or (baseline and value)
                        or any(type(key) is not int or key != 3 or not callable(function)
                               or type(function).__module__ != "mlx.core" or type(function).__name__ != "_gc_func"
                               for key, function in value.items())):
                    raise RuntimeError("The rotary compilation cache has an unsupported representation.")
                continue
            if field.kind == "convolution" and value is missing:
                if field.name in state:
                    raise RuntimeError("The derived-cache baseline has conflicting storage.")
                continue
            if field.kind != "convolution" and value is None:
                if field.name in state:
                    raise RuntimeError("The position baseline has conflicting storage.")
                continue
            if baseline or state.get(field.name, missing) is not value:
                raise RuntimeError("The model request-state baseline changed.")
            if field.kind == "convolution":
                if (type(value) is not tuple or len(value) != 2 or type(value[0]) is not int
                        or value[0] != id(field.owner.conv1d.weight)
                        or not isinstance(value[1], self.backend.array)
                        or value[1].dtype != self.backend.float32 or tuple(value[1].shape) != (4, 6144)):
                    raise RuntimeError("The derived convolution cache has an unsupported representation.")
            else:
                if not isinstance(value, self.backend.array) or value.dtype != self.backend.int32:
                    raise RuntimeError("The position state has an unsupported array type.")
                shape = tuple(value.shape)
                if ((field.kind == "positions" and (len(shape) != 3 or shape[:2] != (3, 1)
                                                     or not 1 <= shape[2] <= self.max_context))
                        or (field.kind == "deltas" and shape != (1, 1))):
                    raise RuntimeError("The position state has an unsupported shape.")

    def _inventory(self) -> tuple:
        entries = []
        seen = set()

        def mapping(value: Any, path: tuple, owner: Any = None) -> None:
            keys = tuple(key for key in value if (id(owner), key) not in self._excluded)
            entries.append((path, "mapping", value, keys))
            for key in keys:
                if type(key) is not str:
                    raise RuntimeError("The model state contains an unsupported mapping key.")
                visit(value[key], path + (key,))

        def visit(value: Any, path: tuple) -> None:
            if isinstance(value, self.backend.array):
                entries.append((path, "array", value, None))
                return
            if id(value) in seen:
                entries.append((path, "alias", value, type(value)))
                return
            if hasattr(value, "state"):
                seen.add(id(value))
                entries.append((path, "module", value, type(value)))
                mapping(value.state, path + ("$state",), value)
            elif type(value) is dict:
                seen.add(id(value))
                mapping(value, path)
            elif type(value) in (list, tuple):
                seen.add(id(value))
                entries.append((path, "sequence", value, (type(value), len(value))))
                for index, child in enumerate(value):
                    visit(child, path + (index,))
            elif type(value) in (str, int, float, bool, type(None)):
                entries.append((path, "scalar", value, type(value)))
            else:
                entries.append((path, "opaque", value, type(value)))

        visit(self.model, ())
        return tuple(entries)

    def _check_inventory(self) -> None:
        current = self._inventory()
        if len(current) != len(self._baseline):
            raise RuntimeError("The resident model state inventory changed.")
        for old, new in zip(self._baseline, current):
            if (old[:2] != new[:2] or old[3] != new[3]
                    or (old[2] != new[2] if old[1] == "scalar" else old[2] is not new[2])):
                raise RuntimeError("The resident model state inventory changed.")

    def begin(self) -> None:
        try:
            if self.phase is not _StatePhase.IDLE:
                raise RuntimeError("The model request-state owner is not idle.")
            self._validate_fields(baseline=True)
            self._check_inventory()
            self.phase = _StatePhase.ACTIVE
        except BaseException:
            self.fence()
            raise

    def restore(self) -> None:
        try:
            if self.fenced:
                raise RuntimeError("The model request-state owner is fenced.")
            self._validate_fields(baseline=False)
            self._check_inventory()
            for field in self.fields:
                if field.kind == "rotary":
                    getattr(field.owner, field.name).clear()
                elif field.kind == "convolution":
                    if field.name in field.owner.state:
                        delattr(field.owner, field.name)
                else:
                    setattr(field.owner, field.name, None)
            self._validate_fields(baseline=True)
            self._check_inventory()
            self.phase = _StatePhase.IDLE
        except BaseException:
            self.fence()
            raise


class VisionRuntime:
    def __init__(self, model_path: str, max_context: int, max_output_tokens: int, memory_limit_bytes: int):
        if not (1 <= max_context <= 16384 and 0 <= max_output_tokens <= 8192
                and 1 <= memory_limit_bytes <= 9007199254740991):
            raise ValueError("The vision deployment limits are invalid.")
        self.owner = threading.get_ident()
        self.model_path = str(Path(model_path).resolve(strict=True))
        packages = identity.check_packages()
        for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                     "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
            os.environ[name] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.runtime = dict(identity.PINNED_RUNTIME)
        with contextlib.redirect_stdout(sys.stderr):
            import mlx.core as mx
            import mlx_vlm
            import transformers
            import numpy
            import PIL
            import torch
            import torchvision
            import tokenizers
            from mlx.utils import tree_flatten
            from mlx_vlm import load
            from mlx_vlm.generate import generate_step, generation_stream
            from tokenizers.decoders import DecodeStream

            modules = {"mlx_core": mx, "mlx_vlm": mlx_vlm, "transformers": transformers,
                       "numpy": numpy, "pillow": PIL, "torch": torch, "torchvision": torchvision,
                       "tokenizers": tokenizers,
                       "generation": importlib.import_module("mlx_vlm.generate.ar"),
                       "processor": importlib.import_module(identity.PROCESSOR_CLASS.rsplit(".", 1)[0])}
            modules.update(identity.state_modules())
            files = identity.runtime_files(modules)
            manifest = model_manifest(Path(self.model_path), self.runtime,
                                      identity.execution(max_context, max_output_tokens, memory_limit_bytes))
            self.manifest = identity.complete_manifest(manifest, files, packages)
            mx.set_default_device(mx.gpu)
            mx.set_memory_limit(memory_limit_bytes)
            mx.set_cache_limit(256 * 1024**2)
            self.mx = mx
            self.generate_step = generate_step
            self._decode_stream_class = DecodeStream
            self._load_stream = mx.default_stream(mx.default_device())
            self._generation_stream = generation_stream
            self._request_stream = None
            self.model, self.processor = load(self.model_path, lazy=False, strict=True, trust_remote_code=False)
            self._synchronize()
            if (sum(value.size for _, value in tree_flatten(self.model.parameters())) != 1554476272
                    or any(value.dtype != mx.bfloat16 for _, value in tree_flatten(self.model.parameters()))):
                raise WorkerFailure("model_unavailable", "The loaded parameters differ from the BF16 vision profile.")
            actual_class = f"{type(self.processor).__module__}.{type(self.processor).__name__}"
            if actual_class != identity.PROCESSOR_CLASS or identity.runtime_files(modules) != files:
                raise WorkerFailure("model_unavailable", "The loaded vision processor or runtime identity changed.")
            self._request_state = _ResidentRequestState.qualify(self.model, mx, max_context)
            self.manifest["request_state"] = self._request_state.identity()
        self.max_context = max_context
        self.vocab_size = 248094
        self.eos_token_ids = [248044, 248046]
        generation = json.loads((Path(self.model_path) / "generation_config.json").read_bytes())
        if sorted(generation["eos_token_id"]) != self.eos_token_ids:
            raise WorkerFailure("model_unavailable", "The checkpoint EOS tokens differ from the vision profile.")
        self.consumed_position = None
        self.peak_host_array_bytes = 0
        self._preparation: VisionPreparation | None = None
        self._decoder = None
        self._decoded_ids: list[int] = []
        self._emitted_text = ""

    def _check_owner(self) -> None:
        if threading.get_ident() != self.owner:
            raise RuntimeError("Only the execution owner can use the vision runtime.")
        if self._request_state.fenced:
            raise RuntimeError("The vision runtime cannot reuse uncertain model state.")

    def activate(self, selected: PreparationIdentity) -> None:
        self._check_owner()
        if self._preparation is not None:
            raise RuntimeError("The vision runtime identity is already active.")
        self._preparation = VisionPreparation(self.processor, self.manifest["processor_revision"], selected,
                                              max_context=self.max_context, vocab_size=self.vocab_size)

    def prepare(self, command: dict[str, Any]) -> dict[str, Any]:
        self._check_owner()
        if self._preparation is None:
            raise WorkerFailure("model_unavailable", "The vision runtime identity is not active.")
        with contextlib.redirect_stdout(sys.stderr):
            return self._preparation.prepare(command)

    def generate(self, command: dict[str, Any], progress: Callable[[int], None]):
        self._check_owner()
        self.peak_host_array_bytes = 0
        self._decoded_ids = []
        self._emitted_text = ""
        self._decoder = self._decode_stream_class(skip_special_tokens=False)
        if self._preparation is None:
            raise WorkerFailure("model_unavailable", "The vision runtime identity is not active.")
        self._request_state.begin()
        self._request_stream = self.mx.default_stream(self.mx.default_device())
        self.mx.reset_peak_memory()
        values = device_ids = device_mask = generator = token = logprobs = pixels = None
        try:
            with contextlib.redirect_stdout(sys.stderr), self._preparation.replay(command) as host:
                self.peak_host_array_bytes = host.replay["host_array_bytes"]
                try:
                    progress(0)
                    values = dict(host.values)
                    device_ids = self.mx.array(values.pop("input_ids"))
                    device_mask = self.mx.array(values.pop("attention_mask"))
                    pixels = values.pop("pixel_values")
                    policy = command["sampling"]
                    selection = {"temperature": 0.0, "top_p": 0.95} if policy["mode"] == "greedy" else {
                        "temperature": policy["temperature"], "top_p": policy["top_p"], "seed": policy["seed"]}
                    generator = self.generate_step(
                        device_ids, self.model, pixels, device_mask, **values,
                        max_tokens=command["max_tokens"], **selection, top_k=100, min_p=0.0,
                        repetition_penalty=1.03, repetition_context_size=8192,
                        presence_penalty=0.0, frequency_penalty=0.0, prefill_step_size=256,
                    )
                    for token, logprobs in generator:
                        yield int(token.item()) if hasattr(token, "item") else int(token)
                finally:
                    try:
                        try:
                            if generator is not None:
                                generator.close()
                        finally:
                            self._synchronize()
                    except BaseException:
                        self._request_state.fence()
                        raise
                    finally:
                        values = device_ids = device_mask = generator = token = logprobs = pixels = None
        except (PreparationError, MediaError) as error:
            raise WorkerFailure(error.code, str(error)) from error
        finally:
            self.peak_host_array_bytes = max(self.peak_host_array_bytes, self._preparation.last_host_array_bytes)

    def decode(self, token: int, eos: bool = False) -> str:
        self._check_owner()
        if eos:
            return ""
        self._decoded_ids.append(token)
        part = self._decoder.step(self.processor.tokenizer.backend_tokenizer, token) or ""
        self._emitted_text += part
        return part

    def finish_text(self) -> str:
        self._check_owner()
        full = self.processor.tokenizer.decode(self._decoded_ids, skip_special_tokens=False,
                                               clean_up_tokenization_spaces=False)
        if not full.startswith(self._emitted_text):
            raise WorkerFailure("internal_error", "The final vision text differs from its streamed prefix.")
        tail = full[len(self._emitted_text):]
        self._emitted_text = full
        return tail

    def memory(self) -> dict[str, int]:
        self._check_owner()
        return {"active_bytes": int(self.mx.get_active_memory()), "peak_bytes": int(self.mx.get_peak_memory()),
                "cache_bytes": int(self.mx.get_cache_memory())}

    def _synchronize(self) -> None:
        for stream in (self._generation_stream, self._request_stream, self._load_stream):
            if stream is not None:
                self.mx.synchronize(stream)

    def settle(self) -> None:
        self._check_owner()
        try:
            self._synchronize()
            self._request_state.restore()
            self._decoder = None
            self._decoded_ids = []
            self._emitted_text = ""
            self.mx.clear_cache()
            self._synchronize()
            self._request_stream = None
        except BaseException:
            self._request_state.fence()
            raise

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import hashlib
from io import BytesIO
import math
from typing import Any, Iterator

import numpy as np
from PIL import Image

if __package__:
    from . import contracts
    from .owned_media import MediaBatch, OwnedPng
else:
    import contracts
    from owned_media import MediaBatch, OwnedPng

MAX_HOST_ARRAY_BYTES = 64 * 1024 * 1024
IMAGE_TOKEN_ID = 248056


class PreparationError(ValueError):
    def __init__(self, message: str, code: str = "invalid_request"):
        super().__init__(message)
        self.code = code


def typed_output(value: Any) -> tuple[dict[str, Any], int]:
    roots = {}

    def represent(item: Any) -> dict[str, Any]:
        if isinstance(item, np.ndarray):
            if item.dtype.kind not in "biuf" or not item.flags.c_contiguous:
                raise PreparationError("Unsupported host array representation.")
            owner = item
            while isinstance(owner.base, np.ndarray):
                owner = owner.base
            if owner.base is not None:
                raise PreparationError("Unsupported host array ownership.")
            roots[id(owner)] = owner
            if sum(root.nbytes for root in roots.values()) > MAX_HOST_ARRAY_BYTES:
                raise PreparationError("Host arrays exceed the profile limit.")
            if item.dtype.kind == "f" and not np.isfinite(item).all():
                raise PreparationError("Processor output contains a non-finite value.")
            canonical = np.ascontiguousarray(item, dtype=item.dtype.newbyteorder("<"))
            return dict(kind="array", dtype=canonical.dtype.str, shape=list(canonical.shape),
                        sha256=hashlib.sha256(canonical.tobytes(order="C")).hexdigest())
        if isinstance(item, np.generic):
            return represent(item.item())
        if type(item) is dict:
            if any(type(key) is not str or not key.isascii() for key in item):
                raise PreparationError("Processor keys must be ASCII strings.")
            return dict(kind="object", items={key: represent(child) for key, child in item.items()})
        if isinstance(item, (list, tuple)):
            return dict(kind="list", items=[represent(child) for child in item])
        if type(item) is str:
            return dict(kind="string", value=item)
        if type(item) is bool:
            return dict(kind="boolean", value=item)
        if type(item) is int:
            return dict(kind="integer", value=str(item))
        if type(item) is float and math.isfinite(item):
            return dict(kind="float", value=item.hex())
        if item is None:
            return dict(kind="null")
        raise PreparationError("Unsupported processor output value.")

    manifest = represent(value)
    return manifest, sum(root.nbytes for root in roots.values())


@dataclass
class HostPrepared:
    values: dict[str, Any] | None
    token_ids: list[int]
    replay: dict

    def close(self) -> None:
        self.values = None


@dataclass(frozen=True)
class PreparationIdentity:
    worker_epoch: str
    model_revision: str
    capability_revision: str

    def fields(self) -> dict[str, str]:
        return dict(worker_epoch=self.worker_epoch, model_revision=self.model_revision,
                    capability_revision=self.capability_revision)


class VisionPreparation:
    def __init__(self, processor: Any, processor_revision: str, identity: PreparationIdentity,
                 *, max_context: int = 16384, vocab_size: int = 248094):
        if not 1 <= max_context <= 16384 or vocab_size <= IMAGE_TOKEN_ID:
            raise ValueError("The preparation limits are invalid.")
        self.processor = processor
        self.processor_revision = processor_revision
        self.vocab_size = vocab_size
        self.identity = identity
        self.max_context = max_context
        self.last_host_array_bytes = 0

    @contextmanager
    def _process(self, rendered: str, media: list[dict], preparation_id: str) -> Iterator[HostPrepared]:
        if len(rendered.encode("utf-8")) > 131072:
            raise PreparationError("The rendered prompt exceeds the byte limit.")
        batch = MediaBatch.from_descriptors(media)
        with ExitStack() as stack:
            images = []
            for image in batch.images:
                source = stack.enter_context(Image.open(BytesIO(image.data)))
                rgb = source.convert("RGB")
                stack.callback(rgb.close)
                images.append(rgb)
            values = dict(self.processor(text=rendered, images=images, return_tensors="np",
                downsample_mode="16x", max_slice_nums=1, padding=True, add_special_tokens=False))
        ids = values["input_ids"]
        if not isinstance(ids, np.ndarray) or ids.dtype != np.dtype("int32") or ids.ndim != 2 or ids.shape[0] != 1:
            raise PreparationError("Unsupported processor token array.")
        tokens = ids[0].tolist()
        if not tokens or any(not 0 <= token < self.vocab_size for token in tokens):
            raise PreparationError("Prepared tokens exceed the vocabulary bounds.")
        if len(tokens) > self.max_context:
            raise PreparationError("Prepared tokens exceed the context limit.", "context_limit")
        for field in ("attention_mask", "mm_token_type_ids"):
            if (not isinstance(values[field], np.ndarray) or values[field].dtype != np.dtype("int32")
                    or values[field].shape != ids.shape):
                raise PreparationError("Unsupported processor position array.")
        count = len(batch.images)
        if (values["downsample_mode"] != "16x" or values["num_patches_per_image"] != [[1] * count]
                or len(values["pixel_values"]) != 1 or len(values["pixel_values"][0]) != count):
            raise PreparationError("The actual image slices exceed the profile.")
        if any(type(value) is not int for value in values["num_patches_per_image"][0]):
            raise PreparationError("The image patch counts must be integers.")
        if any(type(values[field]) is not list or len(values[field]) != 1
               for field in ("image_bound", "tgt_sizes")):
            raise PreparationError("The image metadata must contain one batch row.")
        bounds = values["image_bound"][0]
        targets = values["tgt_sizes"][0]
        if any(not isinstance(value, np.ndarray) or value.dtype != np.dtype("int32")
               or value.shape != (count, 2) for value in (bounds, targets)):
            raise PreparationError("Unsupported processor image metadata.")
        bindings = []
        covered = []
        previous_end = 0
        for index, patch in enumerate(values["pixel_values"][0]):
            start, end = map(int, bounds[index])
            height, width = map(int, targets[index])
            if (not 0 <= previous_end <= start < end <= len(tokens) or end - start > 1024
                    or not 1 <= height <= 1024 or not 1 <= width <= 1024 or height * width > 4096
                    or not isinstance(patch, np.ndarray) or patch.dtype != np.dtype("float32")
                    or patch.shape != (3, 14, 14 * height * width)
                    or any(token != IMAGE_TOKEN_ID for token in tokens[start:end])):
                raise PreparationError("Unsupported processor image position or shape.")
            covered.extend(range(start, end))
            previous_end = end
            bindings.append(dict(source_index=index, token_start=start, token_end=end,
                                 pixel_shape=list(patch.shape), target_size=[height, width]))
        if covered != [index for index, token in enumerate(tokens) if token == IMAGE_TOKEN_ID]:
            raise PreparationError("Image tokens do not match the source occurrences.")
        manifest, host_bytes = typed_output(values)
        self.last_host_array_bytes = host_bytes
        processing = dict(**self.identity.fields(), preparation_id=preparation_id,
                          processor_revision=self.processor_revision,
                          rendered_prompt_sha256=hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
                          source_sha256=[image.sha256 for image in batch.images], processor_output=manifest)
        replay = dict(**self.identity.fields(), preparation_id=preparation_id, rendered_prompt=rendered,
                      processor_revision=self.processor_revision,
                      processing_digest=contracts.canonical_identity_digest("apxinf-vision-processing-v1", processing),
                      media=[image.descriptor() for image in batch.images], image_bindings=bindings,
                      host_array_bytes=host_bytes)
        host = HostPrepared(values, tokens, replay)
        values = ids = bounds = targets = patch = None
        try:
            yield host
        finally:
            host.close()

    def prepare(self, command: dict[str, Any]) -> dict[str, Any]:
        self.last_host_array_bytes = 0
        if (command["worker_epoch"] != self.identity.worker_epoch
                or command["model_revision"] != self.identity.model_revision):
            raise PreparationError("The preparation identity changed.")
        messages = []
        media = []
        for source in command["messages"]:
            message = dict(source)
            if type(source["content"]) is list:
                parts = []
                for part in source["content"]:
                    if part["type"] == "image":
                        image = OwnedPng.from_descriptor(part["media"])
                        media.append(image.descriptor())
                        parts.append(dict(type="image"))
                    else:
                        parts.append(dict(type="text", text=part["text"]))
                message["content"] = parts
            messages.append(message)
        rendered = self.processor.tokenizer.apply_chat_template(
            messages, tools=command["tools"] or None, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        with self._process(rendered, media, command["command_id"]) as host:
            return dict(token_ids=host.token_ids, effective_input_tokens=len(host.token_ids),
                        prompt_digest=contracts.token_prefix_digest(host.token_ids), replay=host.replay)

    @contextmanager
    def replay(self, command: dict[str, Any]) -> Iterator[HostPrepared]:
        self.last_host_array_bytes = 0
        expected = command["replay"]
        if any(command[key] != value or expected[key] != value for key, value in self.identity.fields().items()):
            raise PreparationError("The replay identity changed.")
        if expected["processor_revision"] != self.processor_revision:
            raise PreparationError("The processor identity changed.")
        with self._process(expected["rendered_prompt"], expected["media"], expected["preparation_id"]) as host:
            if host.token_ids != command["token_ids"] or host.replay != expected:
                raise PreparationError("Replayed preparation differs from the admitted input.")
            yield host

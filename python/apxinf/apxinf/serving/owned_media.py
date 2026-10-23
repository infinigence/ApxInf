from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass, field
import hashlib
from io import BytesIO
import struct
import zlib
from typing import Any

from PIL import Image

MAX_IMAGE_BYTES = 131_072
MAX_BATCH_BYTES = 262_144
MAX_IMAGES = 4
MAX_DIMENSION = 1024
MAX_CHUNKS = 1024


class MediaError(ValueError):
    def __init__(self, code: str):
        super().__init__("The PNG source or descriptor is invalid." if code == "invalid_request"
                         else "The media form or limit is unsupported.")
        self.code = code


def _invalid() -> MediaError:
    return MediaError("invalid_request")


def _unsupported() -> MediaError:
    return MediaError("unsupported_feature")


def _fields(value: Any, names: set[str]) -> None:
    if type(value) is not dict or value.keys() != names:
        raise _invalid()


def _png_shape(data: bytes) -> tuple[int, int]:
    if len(data) < 33 or data[:8] != b"\x89PNG\r\n\x1a\n" or data[8:16] != b"\0\0\0\rIHDR":
        raise _invalid()
    width, height = struct.unpack_from(">II", data, 16)
    if width == 0 or height == 0:
        raise _invalid()
    if (max(width, height) > MAX_DIMENSION or data[24] != 8
            or data[25] not in (2, 6) or data[26:29] != b"\0\0\0"):
        raise _unsupported()
    offset = 8
    data_bytes = 0
    data_seen = False
    compressed = bytearray()
    ancillary_seen = set()
    for index in range(MAX_CHUNKS):
        if len(data) - offset < 12:
            raise _invalid()
        length = struct.unpack_from(">I", data, offset)[0]
        if length > len(data) - offset - 12:
            raise _invalid()
        kind = data[offset + 4:offset + 8]
        if any(not (65 <= char <= 90 or 97 <= char <= 122) for char in kind) or not 65 <= kind[2] <= 90:
            raise _invalid()
        crc_offset = offset + 8 + length
        if zlib.crc32(data[offset + 4:crc_offset]) != struct.unpack_from(">I", data, crc_offset)[0]:
            raise _invalid()
        offset += length + 12
        if kind == b"IHDR":
            if index != 0:
                raise _invalid()
        elif kind == b"IDAT":
            data_seen = True
            data_bytes += length
            compressed.extend(data[crc_offset - length:crc_offset])
        elif kind == b"IEND":
            if length != 0 or data_bytes == 0 or offset != len(data):
                raise _invalid()
            channels = 3 if data[25] == 2 else 4
            expected_bytes = height * (1 + width * channels)
            decoder = zlib.decompressobj()
            try:
                pixels = decoder.decompress(compressed, expected_bytes + 1)
            except zlib.error as error:
                raise _invalid() from error
            if (len(pixels) != expected_bytes or not decoder.eof
                    or decoder.unused_data or decoder.unconsumed_tail):
                raise _invalid()
            return width, height
        elif kind in (b"sRGB", b"eXIf", b"pHYs", b"gAMA", b"cHRM", b"tIME"):
            if data_seen or kind in ancillary_seen:
                raise _invalid()
            _check_ancillary(kind, data[crc_offset - length:crc_offset])
            ancillary_seen.add(kind)
        else:
            raise _unsupported()
    raise _unsupported()


def _check_ancillary(kind: bytes, payload: bytes) -> None:
    if kind == b"sRGB":
        valid = len(payload) == 1 and payload[0] <= 3
    elif kind == b"gAMA":
        valid = len(payload) == 4 and payload != b"\0\0\0\0"
    elif kind == b"cHRM":
        valid = len(payload) == 32
    elif kind == b"pHYs":
        valid = len(payload) == 9 and payload[8] <= 1
    elif kind == b"tIME":
        valid = (len(payload) == 7 and 1 <= payload[2] <= 12 and 1 <= payload[3] <= 31
                 and payload[4] <= 23 and payload[5] <= 59 and payload[6] <= 60)
    elif kind == b"eXIf":
        valid = len(payload) >= 8 and payload[:4] in (b"II*\0", b"MM\0*")
    else:
        valid = False
    if not valid:
        raise _invalid()


@dataclass(frozen=True)
class OwnedPng:
    data: bytes = field(repr=False)
    width: int = field(init=False)
    height: int = field(init=False)
    sha256: str = field(init=False)

    def __post_init__(self) -> None:
        if type(self.data) is not bytes:
            raise _invalid()
        if len(self.data) > MAX_IMAGE_BYTES:
            raise _unsupported()
        width, height = _png_shape(self.data)
        try:
            with Image.open(BytesIO(self.data)) as image:
                if image.format != "PNG" or image.size != (width, height) or image.mode not in ("RGB", "RGBA"):
                    raise _invalid()
                image.verify()
            with Image.open(BytesIO(self.data)) as image:
                image.load()
        except (OSError, SyntaxError, ValueError) as error:
            raise _invalid() from error
        object.__setattr__(self, "width", width)
        object.__setattr__(self, "height", height)
        object.__setattr__(self, "sha256", hashlib.sha256(self.data).hexdigest())

    @classmethod
    def _from_base64(cls, encoded: Any) -> OwnedPng:
        if type(encoded) is not str:
            raise _invalid()
        if len(encoded) > (MAX_IMAGE_BYTES + 2) // 3 * 4:
            raise _unsupported()
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as error:
            raise _invalid() from error
        if base64.b64encode(data).decode("ascii") != encoded:
            raise _invalid()
        return cls(data)

    @classmethod
    def from_source(cls, value: Any) -> OwnedPng:
        _fields(value, {"type", "media_type", "data"})
        if type(value["type"]) is not str or type(value["media_type"]) is not str:
            raise _invalid()
        if value["type"] != "base64" or value["media_type"] != "image/png":
            raise _unsupported()
        return cls._from_base64(value["data"])

    @classmethod
    def from_descriptor(cls, value: Any) -> OwnedPng:
        _fields(value, {"media_type", "data_base64", "sha256", "byte_count", "width", "height"})
        if type(value["media_type"]) is not str:
            raise _invalid()
        if value["media_type"] != "image/png":
            raise _unsupported()
        image = cls._from_base64(value["data_base64"])
        if (any(type(value[key]) is not int for key in ("byte_count", "width", "height"))
                or value["sha256"] != image.sha256 or value["byte_count"] != len(image.data)
                or value["width"] != image.width or value["height"] != image.height):
            raise _invalid()
        return image

    def descriptor(self) -> dict[str, Any]:
        return dict(media_type="image/png", data_base64=base64.b64encode(self.data).decode("ascii"),
                    sha256=self.sha256, byte_count=len(self.data), width=self.width, height=self.height)


@dataclass(frozen=True)
class MediaBatch:
    images: tuple[OwnedPng, ...]

    def __post_init__(self) -> None:
        if type(self.images) is not tuple or any(type(image) is not OwnedPng for image in self.images):
            raise _invalid()
        if len(self.images) > MAX_IMAGES or sum(len(image.data) for image in self.images) > MAX_BATCH_BYTES:
            raise _unsupported()

    @classmethod
    def from_descriptors(cls, value: Any) -> MediaBatch:
        if type(value) is not list:
            raise _invalid()
        if len(value) > MAX_IMAGES:
            raise _unsupported()
        images = []
        total = 0
        for descriptor in value:
            image = OwnedPng.from_descriptor(descriptor)
            total += len(image.data)
            if total > MAX_BATCH_BYTES:
                raise _unsupported()
            images.append(image)
        return cls(tuple(images))

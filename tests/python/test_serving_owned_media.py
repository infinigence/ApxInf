import base64
import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "owned_media_under_test", ROOT / "python/apxinf/apxinf/serving/owned_media.py")
media = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = media
SPEC.loader.exec_module(media)
CORPUS = json.loads((ROOT / "tests/fixtures/serving/owned-media-v0.1.json").read_text())


def chunk(kind, data):
    body = kind.encode("ascii") + data
    return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))


def descriptors(case):
    result = []
    for name in case["images"]:
        value = copy.deepcopy(CORPUS["images"][name])
        data = bytearray(base64.b64decode(value["data_base64"]))
        if "replace_idat_hex" in case:
            length = struct.unpack_from(">I", data, 33)[0]
            data[33:45 + length] = chunk("IDAT", bytes.fromhex(case["replace_idat_hex"]))
        if "pad_to_bytes" in case:
            payload = bytes.fromhex("49492a0008000000000000000000")
            payload += bytes(case["pad_to_bytes"] - len(data) - 12 - len(payload))
            data[33:33] = chunk("eXIf", payload)
        if "insert_chunk" in case:
            extra = case["insert_chunk"]
            insert_at = 33
            if extra.get("after_idat"):
                insert_at += 12 + struct.unpack_from(">I", data, 33)[0]
            data[insert_at:insert_at] = chunk(extra["type"], bytes.fromhex(extra["data_hex"])) * extra.get("repeat", 1)
        data[33:33] = chunk("IDAT", b"") * case.get("empty_idats", 0)
        if "xor_offset" in case:
            data[case["xor_offset"]] ^= 1
        if "set_byte" in case:
            offset, value_byte = case["set_byte"]
            data[offset] = value_byte
        if "repair_crc_offset" in case:
            start = case["repair_crc_offset"]
            length = struct.unpack_from(">I", data, start)[0]
            crc_start = start + 8 + length
            data[crc_start:crc_start + 4] = struct.pack(">I", zlib.crc32(data[start + 4:crc_start]))
        if "truncate" in case:
            del data[-case["truncate"]:]
        data.extend(bytes.fromhex(case.get("append_hex", "")))
        value.update(data_base64=base64.b64encode(data).decode(), byte_count=len(data),
                     sha256=hashlib.sha256(data).hexdigest())
        result.append(value)
    return result


def invocation(case):
    values = descriptors(case)
    operation = case["operation"]
    if operation == "batch":
        return media.MediaBatch.from_descriptors, values, values
    value = values[0]
    if operation == "source":
        value = dict(type="base64", media_type=value["media_type"], data=value["data_base64"])
        value["data"] += case.get("base64_suffix", "")
        if case.get("strip_padding"):
            value["data"] = value["data"].rstrip("=")
        if "base64_tail" in case:
            value["data"] = value["data"][:-4] + case["base64_tail"]
    value.update(case.get("set", {}))
    if "remove" in case:
        del value[case["remove"]]
    return (media.OwnedPng.from_source if operation == "source" else media.OwnedPng.from_descriptor), value, values


class OwnedMediaTests(unittest.TestCase):
    def test_shared_original_corpus(self):
        for case in CORPUS["cases"]:
            with self.subTest(case["name"]):
                parse, value, expected = invocation(case)
                before = copy.deepcopy(value)
                if case.get("valid"):
                    parsed = parse(value)
                    images = parsed.images if case["operation"] == "batch" else (parsed,)
                    self.assertEqual([image.descriptor() for image in images], expected)
                    for image, descriptor in zip(images, expected):
                        self.assertEqual(image.data, base64.b64decode(descriptor["data_base64"]))
                        self.assertEqual(media.OwnedPng.from_descriptor(image.descriptor()), image)
                else:
                    with self.assertRaises(media.MediaError) as caught:
                        parse(value)
                    self.assertEqual(caught.exception.code, case["error"])
                self.assertEqual(value, before)

    def test_literal_identity_vectors(self):
        for value in CORPUS["images"].values():
            self.assertEqual(media.OwnedPng.from_descriptor(value).descriptor(), value)

    def test_owned_values_reject_mutation(self):
        image = media.OwnedPng.from_descriptor(CORPUS["images"]["rgb"])
        with self.assertRaises(FrozenInstanceError):
            image.data = b"changed"
        with self.assertRaises(media.MediaError):
            media.OwnedPng(bytearray(image.data))
        with self.assertRaises(media.MediaError):
            media.MediaBatch([image])

    def test_nonobjects_and_nonarray_batches(self):
        for value in (None, True, 1, "image", []):
            with self.assertRaises(media.MediaError):
                media.OwnedPng.from_source(value)
        for value in (None, True, 1, "images", {}):
            with self.assertRaises(media.MediaError):
                media.MediaBatch.from_descriptors(value)


if __name__ == "__main__":
    unittest.main()

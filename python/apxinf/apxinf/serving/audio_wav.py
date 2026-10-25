"""Bounded PCM16 mono WAV inspection and encoding."""
from dataclasses import dataclass
import struct

from audio_contracts import MAX_OUTPUT, MAX_UPLOAD


@dataclass(frozen=True)
class WavInfo:
    sample_rate: int
    frames: int
    data_offset: int
    data_bytes: int


def inspect(data: bytes, asr: bool) -> WavInfo:
    cap = MAX_UPLOAD if asr else MAX_OUTPUT
    if len(data) < 44 or len(data) > cap or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("invalid WAV container")
    if struct.unpack_from("<I", data, 4)[0] != len(data) - 8:
        raise ValueError("WAV length mismatch")
    offset, metadata, chunks = 12, 0, 0
    rate = payload = None
    while offset < len(data):
        chunks += 1
        if chunks > 32 or offset + 8 > len(data):
            raise ValueError("WAV chunk limit")
        tag = data[offset:offset + 4]
        size = struct.unpack_from("<I", data, offset + 4)[0]
        start = offset + 8
        end = start + size
        padded = end + size % 2
        if padded > len(data) or (size % 2 and data[end] != 0):
            raise ValueError("invalid WAV chunk length or padding")
        if tag == b"fmt ":
            if rate is not None or size != 16:
                raise ValueError("WAV fmt must occur once with 16 bytes")
            encoding, channels, rate, byte_rate, align, bits = struct.unpack_from("<HHIIHH", data, start)
            if encoding != 1 or channels != 1 or bits != 16 or align != 2 or not 8000 <= rate <= 48000 or byte_rate != rate * 2 or (asr and rate != 16000):
                raise ValueError("unsupported WAV format")
        elif tag == b"data":
            if payload is not None or size == 0 or size % 2:
                raise ValueError("invalid WAV data")
            payload = (start, size)
        else:
            metadata += size + 8 + size % 2
            if metadata > 65536:
                raise ValueError("WAV metadata limit")
        offset = padded
    if rate is None or payload is None:
        raise ValueError("missing WAV fmt or data")
    start, size = payload
    frames = size // 2
    if frames > rate * (30 if asr else 60):
        raise ValueError("WAV duration limit")
    return WavInfo(rate, frames, start, size)


def encode_pcm16(pcm: bytes, rate: int) -> bytes:
    if not 8000 <= rate <= 48000 or not pcm or len(pcm) % 2 or len(pcm) + 44 > MAX_OUTPUT or len(pcm) // 2 > rate * 60:
        raise ValueError("output limit")
    header = b"RIFF" + struct.pack("<I", len(pcm) + 36) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    return header + b"data" + struct.pack("<I", len(pcm)) + pcm

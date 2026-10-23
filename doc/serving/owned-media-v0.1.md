# Owned PNG source contract

Status: standalone P2c input component. Date: 2026-10-23.
This component defines source media before a vision worker protocol exists.
It does not enable image requests, a new provider, or a new ready protocol.
The [main contract](contracts-v0.1.md#prepared-input) defines preparation ownership.

## Values and ownership

An owned PNG contains immutable source bytes, dimensions, and a SHA-256 digest.
The digest identifies the original file bytes, including metadata.
Equal decoded pixels do not imply equal source identities.
The value retains no decoded pixels, file reference, URL, or device array.
The request owner retains source values until preparation or attempt settlement.

A source object has exactly `type`, `media_type`, and `data`.
Their values are `base64`, `image/png`, and canonical padded standard base64.
Whitespace, missing padding, alternate alphabets, and nonzero unused bits are invalid.

A descriptor has exactly these fields:

| Field | Meaning |
| --- | --- |
| `media_type` | The literal `image/png`. |
| `data_base64` | The canonical encoding of the original bytes. |
| `sha256` | The lowercase SHA-256 digest of those bytes. |
| `byte_count` | The original file length. |
| `width`, `height` | The integer dimensions from the checked PNG. |

Descriptor parsing repeats source checks and checks every declared value.
Integer fields reject Boolean, floating-point, and string values.
Unknown and missing fields are invalid.
The outer JSON parser remains responsible for duplicate keys and transport framing.

An ordered media batch contains zero through four owned PNG values.
It retains duplicate images and their original order.
Its order follows image occurrences in normalized message history.
This source component does not define token insertion positions or a complete prepared input.
The future vision protocol must bind those positions and processor outputs separately.

## Bounds and accepted PNG form

Each file contains at most 131072 bytes.
A batch contains at most 262144 source bytes.
Each dimension is between 1 and 1024 pixels.
Each file contains at most 1024 chunks.
Each image uses eight-bit RGB or RGBA, without interlacing.
The compression and filter methods are zero.

The accepted critical chunks are `IHDR`, consecutive `IDAT` chunks, and final `IEND`.
The accepted ancillary chunks are `sRGB`, `eXIf`, `pHYs`, `gAMA`, `cHRM`, and `tIME`.
Each ancillary chunk occurs at most once, before the first `IDAT`.
The codec checks fixed payload lengths, sRGB intent, physical-unit codes, time-field ranges, and the TIFF signature.
It does not interpret EXIF directory entries or apply metadata to pixels.
The codec rejects other chunks, including animation and compressed metadata.
The codec checks chunk boundaries, CRCs, and ordering before the decoder runs.
It checks dimensions before any pixel allocation.
The codec checks the complete zlib stream against the declared scanline byte count.
It rejects missing Adler checksums, extra pixels, and compressed trailing data.
The public PNG decoder then reads the declared image pixels.
The codec rejects truncated input, invalid compressed pixels, and bytes after `IEND`.
It applies no orientation or color-profile conversion.
Decoded frame storage cannot exceed 4194304 bytes per image.
Decoder working memory is separate from that frame storage.
These limits do not establish a process memory ceiling or worker capacity.

Malformed syntax, checksums, compressed data, and inconsistent descriptors produce `invalid_request`.
Unsupported media types, source types, PNG forms, and declared limits produce `unsupported_feature`.
Decoder-specific messages do not cross the component boundary.
Validation does not mutate the supplied source or descriptor.

## Interfaces and dependencies

Rust `OwnedPng::from_source` and Python `OwnedPng.from_source` parse source objects.
Their `from_descriptor` methods check descriptors.
Their `descriptor` methods return the canonical descriptor.
`MediaBatch::from_descriptors` and `MediaBatch.from_descriptors` check ordered batches.
Both implementations consume the same original conformance corpus.

Rust base64 0.22.1 supplies `Engine::decode` and `Engine::encode`.
Rust png 0.18.0 supplies bounded `Decoder`, `Reader::next_frame`, and `Reader::finish` operations.
Rust crc32fast 1.5.0 supplies `hash` for every chunk CRC, including `IEND`.
Rust flate2 1.1.10 supplies `Decompress` with bounded output and exact stream completion checks.
These dependencies replace encoding and pixel-decoding implementations inside the media boundary.
Existing sha2 supplies SHA-256.
Python uses standard-library base64, hashlib, and zlib, plus the existing Pillow dependency.
Pillow `Image.open`, `verify`, and `load` check the PNG and decoded pixels.
This unit adds no Python dependency or MLX import.

## Later serving gate

The selected architecture recomputes host preparation after admission.
One adapter applies the chat template once and retains its rendered prompt for that replay.
Replay must reproduce authoritative tokens, source order, processor outputs, and their identities before device transfer.
Count-only and rejected requests do not replay preparation.
Preparation retains no remote media handle between commands.
The supervisor retains source bytes through operation settlement, including late replies.
This architecture adds no release command or prepared-object cache.

The new profile still needs paired protocol validators, processor bindings, memory accounting, and lifecycle checks.
It must report unsupported exact positions and cancellation boundaries honestly.
The [P2c exit checks](implementation-plan.md#p2c-serial-vision-requests) and real Claude drawing acceptance remain open.
The current text protocols and public image capabilities retain their behavior.

Structural writing checks do not establish a full dictionary review or certification.

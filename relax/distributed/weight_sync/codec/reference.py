# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Deterministic CPU reference codec over immutable canonical byte chunks."""

import struct
from dataclasses import replace

from ..limits import CodecLimits, DeltaCodecError
from ..schema import CanonicalChunk
from .format import ChunkDescriptor, Codec, EncodedChunk, content_hash


_UINT_VIEWS = {1: "B", 2: "H", 4: "I", 8: "Q"}
_PRIORITY = (Codec.COPY_BASE, Codec.SPARSE_REPLACE_V1, Codec.BITMAP_REPLACE_V1, Codec.RAW_V1)


class DeltaEncoder:
    """Encode/decode one chunk without tensor, transport, or loader imports.

    Integer views only compare bit patterns; payload values are copied from
    original bytes. Two scans avoid allocating an O(n) Python index list.
    Versions are scoped by the caller's trusted stream/epoch manifest; this
    class does not publish versions or implement consumer commit state.
    """

    def __init__(self, limits: CodecLimits = CodecLimits()) -> None:
        self.limits = limits

    def encode(self, target: CanonicalChunk, *, base: CanonicalChunk | None = None) -> EncodedChunk:
        """Use base=None for a self-contained FULL/anchor chunk."""
        target.spec.validate(self.limits)
        if base is not None:
            base.spec.validate(self.limits)
            if base.spec != target.spec or base.version >= target.version:
                raise DeltaCodecError("base identity mismatch or non-increasing version")
        size, count = target.spec.tensor.element_size, target.spec.element_count
        target_hash = content_hash(target.data)
        raw = ChunkDescriptor(
            target.spec, target.version, Codec.RAW_V1, count, len(target.data), target_hash, target_hash
        )
        raw_chunk = EncodedChunk(raw, target.data)
        raw_chunk.validate(self.limits)
        if base is None:
            return raw_chunk

        before = memoryview(base.data).cast(_UINT_VIEWS[size])
        after = memoryview(target.data).cast(_UINT_VIEWS[size])
        changed = sum(old != new for old, new in zip(before, after, strict=True))
        base_hash = content_hash(base.data)
        candidates = [raw]
        lengths = {
            Codec.SPARSE_REPLACE_V1: changed * (4 + size),
            Codec.BITMAP_REPLACE_V1: (count + 7) // 8 + changed * size,
        }
        if changed == 0:
            lengths[Codec.COPY_BASE] = 0
        for codec, length in lengths.items():
            if length > self.limits.max_chunk_bytes:
                continue
            descriptor = replace(
                raw,
                codec=codec,
                replacement_count=changed,
                encoded_length=length,
                payload_hash="0" * 64,
                base_version=base.version,
                base_hash=base_hash,
            )
            # A delta header can exceed a small metadata budget while RAW fits.
            try:
                descriptor.metadata_bytes(self.limits)
            except DeltaCodecError:
                continue
            candidates.append(descriptor)
        selected = min(
            candidates,
            key=lambda item: (EncodedChunk(item, b"").serialized_size(self.limits), _PRIORITY.index(item.codec)),
        )
        if selected.codec == Codec.RAW_V1:
            return raw_chunk
        payload = bytearray(selected.encoded_length)
        if selected.codec != Codec.COPY_BASE:
            values_start = changed * 4 if selected.codec == Codec.SPARSE_REPLACE_V1 else (count + 7) // 8
            value_cursor = values_start
            index_cursor = 0
            source = memoryview(target.data)
            for index, (old, new) in enumerate(zip(before, after, strict=True)):
                if old == new:
                    continue
                if selected.codec == Codec.SPARSE_REPLACE_V1:
                    struct.pack_into("<I", payload, index_cursor, index)
                    index_cursor += 4
                else:
                    payload[index // 8] |= 1 << (index % 8)
                payload[value_cursor : value_cursor + size] = source[index * size : (index + 1) * size]
                value_cursor += size
        data = bytes(payload)
        return EncodedChunk(replace(selected, payload_hash=content_hash(data)), data)

    def decode(
        self, encoded: EncodedChunk, *, expected: ChunkDescriptor, base: CanonicalChunk | None = None
    ) -> CanonicalChunk:
        """Verify a trusted descriptor and reconstruct into independent bytes.

        `expected` must come from the caller's trusted manifest. Hashes provide
        integrity, not source authentication. No live tensor or base is
        mutated. RAW is base-independent, even if a stale base is supplied by
        the caller.
        """
        expected.metadata_bytes(self.limits)
        if encoded.descriptor != expected:
            raise DeltaCodecError("encoded descriptor differs from trusted expectation")
        encoded.validate(self.limits)
        descriptor, payload = encoded.descriptor, encoded.payload
        count, size = descriptor.spec.element_count, descriptor.spec.tensor.element_size
        codec = descriptor.codec

        # Validate structure before allocating output (even with a valid hash).
        if codec == Codec.SPARSE_REPLACE_V1:
            previous = -1
            for offset in range(0, descriptor.replacement_count * 4, 4):
                index = struct.unpack_from("<I", payload, offset)[0]
                if not previous < index < count:
                    raise DeltaCodecError("sparse indices must be increasing, unique, and in range")
                previous = index
        elif codec == Codec.BITMAP_REPLACE_V1:
            bitmap_length = (count + 7) // 8
            if count % 8 and payload[bitmap_length - 1] >> (count % 8):
                raise DeltaCodecError("bitmap has nonzero padding bits")
            if sum(byte.bit_count() for byte in memoryview(payload)[:bitmap_length]) != descriptor.replacement_count:
                raise DeltaCodecError("bitmap population differs from replacement count")

        if codec == Codec.RAW_V1:
            data = payload
        else:
            if base is None:
                raise DeltaCodecError("base is required")
            if base.spec != descriptor.spec or base.version != descriptor.base_version:
                raise DeltaCodecError("base identity or version mismatch")
            if content_hash(base.data) != descriptor.base_hash:
                raise DeltaCodecError("base hash mismatch")
            if codec == Codec.COPY_BASE:
                data = base.data  # Sharing immutable bytes is safe.
            else:
                output = bytearray(base.data)
                if codec == Codec.SPARSE_REPLACE_V1:
                    cursor = descriptor.replacement_count * 4
                    indices = (struct.unpack_from("<I", payload, offset)[0] for offset in range(0, cursor, 4))
                else:
                    cursor = (count + 7) // 8
                    indices = (index for index in range(count) if payload[index // 8] & (1 << (index % 8)))
                source = memoryview(payload)
                for index in indices:
                    output[index * size : (index + 1) * size] = source[cursor : cursor + size]
                    cursor += size
                data = bytes(output)
        if content_hash(data) != descriptor.target_hash:
            raise DeltaCodecError("reconstructed target hash mismatch")
        return CanonicalChunk(descriptor.spec, descriptor.target_version, data)

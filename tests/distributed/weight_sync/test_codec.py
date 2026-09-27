# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Byte-oracle and malformed-input tests; no GPU or model dependencies."""

import hashlib
import json
import random
import struct
import subprocess
import sys
import textwrap
import tracemalloc
from dataclasses import replace

import pytest

from relax.distributed.weight_sync import (
    CanonicalChunk,
    ChunkDescriptor,
    ChunkSpec,
    Codec,
    CodecLimits,
    DeltaCodecError,
    DeltaEncoder,
    EncodedChunk,
    TensorSpec,
    iter_chunks,
)


SCHEMA = "a" * 64
HEADER = struct.Struct("<4sIQ")


def chunk(data: bytes, dtype: str = "bfloat16", version: int = 0) -> CanonicalChunk:
    size = TensorSpec("weight", dtype, ()).element_size
    tensor = TensorSpec("model.layers.0.weight", dtype, (len(data) // size,))
    return CanonicalChunk(ChunkSpec(SCHEMA, tensor, 0, len(data)), version, data)


def encoded_for(codec: Codec, target: bytes, payload: bytes, count: int, base: CanonicalChunk) -> EncodedChunk:
    """Independent wire producer for decoder conformance/negative tests."""
    raw = codec == Codec.RAW_V1
    return EncodedChunk(
        ChunkDescriptor(
            spec=base.spec,
            target_version=base.version + 1,
            codec=codec,
            replacement_count=count,
            encoded_length=len(payload),
            payload_hash=hashlib.sha256(payload).hexdigest(),
            target_hash=hashlib.sha256(target).hexdigest(),
            base_version=None if raw else base.version,
            base_hash=None if raw else hashlib.sha256(base.data).hexdigest(),
        ),
        payload,
    )


def independent_envelope(metadata: dict, payload: bytes = b"") -> bytes:
    data = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return HEADER.pack(b"DWC1", len(data), len(payload)) + data + payload


@pytest.mark.parametrize("codec", list(Codec))
def test_fixed_payload_vectors(codec: Codec) -> None:
    base = chunk(bytes(1024))
    target = bytearray(base.data)
    if codec == Codec.COPY_BASE:
        payload, count = b"", 0
    elif codec == Codec.SPARSE_REPLACE_V1:
        target[10:12], target[1022:1024] = b"\x01\x80", b"\xc1\x7f"
        payload, count = bytes.fromhex("05000000ff0100000180c17f"), 2
    elif codec == Codec.BITMAP_REPLACE_V1:
        target[:128] = b"\x80\x7f" * 64
        payload, count = b"\xff" * 8 + bytes(56) + b"\x80\x7f" * 64, 64
    else:
        target[:] = b"\xff\x7f" * 512
        payload, count = bytes(target), 512
    target = bytes(target)
    encoder = DeltaEncoder()
    encoded = encoder.encode(replace(base, version=1, data=target), base=base)
    independent = encoded_for(codec, target, payload, count, base)
    assert encoded == independent
    wire = independent.to_bytes()
    magic, metadata_length, payload_length = HEADER.unpack_from(wire)
    assert (magic, payload_length) == (b"DWC1", len(payload))
    assert wire[HEADER.size + metadata_length :] == payload
    parsed = EncodedChunk.from_bytes(wire)
    assert parsed == encoded
    assert encoder.decode(parsed, expected=independent.descriptor, base=base).data == target


@pytest.mark.parametrize("dtype", ["uint8", "int16", "bfloat16", "float16", "float32", "int64", "float64"])
@pytest.mark.parametrize("fraction", [0, 0.001, 0.02, 0.04, 0.25, 0.75, 1])
def test_random_bitwise_roundtrip_and_raw_bound(dtype: str, fraction: float) -> None:
    encoder = DeltaEncoder()
    rng = random.Random(6974)
    size = TensorSpec("w", dtype, ()).element_size
    for count in (1, 7, 65, 1025):
        before = rng.randbytes(count * size)
        after = bytearray(before)
        for index in rng.sample(range(count), int(count * fraction)):
            after[index * size] ^= 0x80
        base = chunk(before, dtype)
        target = replace(base, version=1, data=bytes(after))
        encoded = encoder.encode(target, base=base)
        full = encoder.encode(target)
        assert full.descriptor.codec == Codec.RAW_V1
        assert encoded.serialized_size() == len(encoded.to_bytes())
        assert encoded.serialized_size() <= full.serialized_size()
        assert encoder.encode(target, base=base).to_bytes() == encoded.to_bytes()
        assert encoder.decode(encoded, expected=encoded.descriptor, base=base) == target
        assert encoder.decode(encoded, expected=encoded.descriptor, base=base) == target
        assert base.data == before


@pytest.mark.parametrize(
    "dtype,bits",
    [
        ("bfloat16", "00000080807f80ffc17fc27f0100"),
        ("float16", "00000080007c00fc017e027e0100"),
        ("float32", "00000000000000800000807f000080ff0100c07f0200c07f01000000"),
    ],
)
def test_zero_sign_inf_and_nan_payload_changes(dtype: str, bits: str) -> None:
    # Construct bits directly: numerical casts/equality can erase this evidence.
    patterns = bytes.fromhex(bits)
    base = chunk(bytes(4096), dtype)
    target = replace(base, version=1, data=patterns + bytes(4096 - len(patterns)))
    encoded = DeltaEncoder().encode(target, base=base)
    assert encoded.descriptor.codec == Codec.SPARSE_REPLACE_V1
    assert DeltaEncoder().decode(encoded, expected=encoded.descriptor, base=base).data == target.data
    size = base.spec.tensor.element_size
    second = replace(target, version=2, data=target.data[:size] + bytes(size) + target.data[2 * size :])
    assert second.data != target.data
    encoded2 = DeltaEncoder().encode(second, base=target)
    assert DeltaEncoder().decode(encoded2, expected=encoded2.descriptor, base=target).data == second.data


def test_small_unchanged_chunk_can_be_raw_when_header_cost_wins() -> None:
    base = chunk(b"\x00\x00")
    encoded = DeltaEncoder().encode(replace(base, version=1), base=base)
    assert encoded.descriptor.codec == Codec.RAW_V1


def test_full_and_raw_ignore_stale_base() -> None:
    target = chunk(b"\x80\x7f" * 512, version=7)
    encoded = DeltaEncoder().encode(target)
    assert encoded.descriptor.base_version is None
    assert encoded.descriptor.base_hash is None
    unrelated = chunk(b"\x00", "int8", 999)
    assert DeltaEncoder().decode(encoded, expected=encoded.descriptor, base=unrelated) == target


@pytest.mark.parametrize("mutation", ["missing", "version", "bytes", "schema", "tensor", "offset"])
def test_base_is_bound_to_identity_version_and_bytes(mutation: str) -> None:
    base = chunk(bytes(1024))
    encoded = DeltaEncoder().encode(replace(base, version=1), base=base)
    bad = base
    if mutation == "missing":
        bad = None
    elif mutation == "version":
        bad = replace(base, version=1)  # Already-target data is not a new base.
    elif mutation == "bytes":
        bad = replace(base, data=b"\x01" + base.data[1:])
    elif mutation == "schema":
        bad = replace(base, spec=replace(base.spec, schema_id="b" * 64))
    elif mutation == "tensor":
        bad = replace(base, spec=replace(base.spec, tensor=replace(base.spec.tensor, name="other")))
    else:
        spec = replace(base.spec, tensor=replace(base.spec.tensor, shape=(1024,)), byte_offset=1024)
        bad = replace(base, spec=spec)
    with pytest.raises(DeltaCodecError, match="base"):
        DeltaEncoder().decode(encoded, expected=encoded.descriptor, base=bad)


@pytest.mark.parametrize("indices", [(1, 1), (2, 1), (0, 512)])
def test_sparse_rejects_invalid_indices_even_with_correct_payload_hash(indices: tuple[int, int]) -> None:
    base = chunk(bytes(1024))
    encoded = encoded_for(Codec.SPARSE_REPLACE_V1, base.data, struct.pack("<II", *indices) + bytes(4), 2, base)
    with pytest.raises(DeltaCodecError, match="indices"):
        DeltaEncoder().decode(encoded, expected=encoded.descriptor, base=base)


@pytest.mark.parametrize("bitmap,count,message", [(b"\x00\x80", 1, "padding"), (b"\x01\x00", 2, "population")])
def test_bitmap_rejects_padding_and_popcount(bitmap: bytes, count: int, message: str) -> None:
    base = chunk(bytes(18))
    encoded = encoded_for(Codec.BITMAP_REPLACE_V1, base.data, bitmap + bytes(count * 2), count, base)
    with pytest.raises(DeltaCodecError, match=message):
        DeltaEncoder().decode(encoded, expected=encoded.descriptor, base=base)


def test_bitmap_tail_vector() -> None:
    base = chunk(bytes(18))  # Nine BF16 elements: two bitmap bytes, seven unused bits.
    target = bytes(16) + b"\x00\x80"
    encoded = encoded_for(Codec.BITMAP_REPLACE_V1, target, b"\x00\x01\x00\x80", 1, base)
    assert DeltaEncoder().decode(encoded, expected=encoded.descriptor, base=base).data == target


def test_corrupt_payload_and_target_hash_are_rejected() -> None:
    base = chunk(bytes(1024))
    target = b"\x01" + bytes(1023)
    encoded = DeltaEncoder().encode(replace(base, version=1, data=target), base=base)
    corrupt = replace(encoded, payload=encoded.payload[:-1] + b"\xff")
    with pytest.raises(DeltaCodecError, match="payload hash"):
        DeltaEncoder().decode(corrupt, expected=encoded.descriptor, base=base)
    forged = replace(encoded, descriptor=replace(encoded.descriptor, target_hash="f" * 64))
    with pytest.raises(DeltaCodecError, match="target hash"):
        DeltaEncoder().decode(forged, expected=forged.descriptor, base=base)
    with pytest.raises(DeltaCodecError, match="trusted expectation"):
        DeltaEncoder().decode(forged, expected=encoded.descriptor, base=base)


@pytest.mark.parametrize(
    "change",
    ["unknown", "missing", "float", "bool", "negative", "huge", "dtype", "codec", "format", "raw_base", "shape"],
)
def test_rejects_invalid_metadata(change: str) -> None:
    encoded = DeltaEncoder().encode(chunk(bytes(1024), version=1))
    metadata = json.loads(encoded.descriptor.metadata_bytes())
    if change == "unknown":
        metadata["required_feature"] = "unknown"
    elif change == "missing":
        del metadata["target_hash"]
    elif change == "dtype":
        metadata["chunk"]["tensor"]["dtype"] = "unknown"
    elif change == "codec":
        metadata["codec"] = "XOR_UNKNOWN"
    elif change == "format":
        metadata["format_version"] = 2
    elif change == "raw_base":
        metadata["base_version"] = 0
    elif change == "shape":
        metadata["chunk"]["tensor"]["shape"] = [True]
    else:
        metadata["target_version"] = {"float": 1.0, "bool": True, "negative": -1, "huge": 1 << 64}[change]
    with pytest.raises(DeltaCodecError):
        EncodedChunk.from_bytes(independent_envelope(metadata, encoded.payload))


@pytest.mark.parametrize(
    "metadata",
    [
        b'{"x":1,"x":2}',
        b'{"x":NaN}',
        b'{"x":Infinity}',
        b'{"x":1e0}',
        b"[" * 1200 + b"]" * 1200,
        b"\xff",
        b"null",
    ],
)
def test_rejects_duplicate_noninteger_or_malformed_json(metadata: bytes) -> None:
    with pytest.raises(DeltaCodecError):
        EncodedChunk.from_bytes(HEADER.pack(b"DWC1", len(metadata), 0) + metadata)


def test_wire_lengths_magic_and_noncanonical_json() -> None:
    encoded = DeltaEncoder().encode(chunk(bytes(16), version=1))
    wire = encoded.to_bytes()
    for malformed in (wire[:5], wire[:-1], wire + b"x", b"DWC2" + wire[4:]):
        with pytest.raises(DeltaCodecError):
            EncodedChunk.from_bytes(malformed)
    data = b" " + encoded.descriptor.metadata_bytes()
    with pytest.raises(DeltaCodecError, match="canonical"):
        EncodedChunk.from_bytes(HEADER.pack(b"DWC1", len(data), len(encoded.payload)) + data + encoded.payload)


def test_limits_checked_before_decoding_or_payload_copy(monkeypatch: pytest.MonkeyPatch) -> None:
    from relax.distributed.weight_sync.codec import reference

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("unexpected output allocation")

    monkeypatch.setattr(reference, "bytearray", forbidden_allocation, raising=False)
    base = chunk(bytes(1024))
    encoded = encoded_for(Codec.SPARSE_REPLACE_V1, base.data, struct.pack("<I", 512) + bytes(2), 1, base)
    with pytest.raises(DeltaCodecError, match="indices"):
        DeltaEncoder().decode(encoded, expected=encoded.descriptor, base=base)
    with pytest.raises(DeltaCodecError, match="exceeds limit"):
        DeltaEncoder(CodecLimits(max_chunk_bytes=16)).decode(encoded, expected=encoded.descriptor, base=base)
    for metadata_size, payload_size in (((1 << 32) - 1, 0), (0, (1 << 64) - 1)):
        with pytest.raises(DeltaCodecError, match="exceeds limits"):
            EncodedChunk.from_bytes(HEADER.pack(b"DWC1", metadata_size, payload_size))


def test_schema_intervals_scalar_empty_and_tail() -> None:
    tensor = TensorSpec("visual.weight", "float32", (3, 7))
    specs = list(iter_chunks(tensor, SCHEMA, 32))
    assert [(s.byte_offset, s.byte_length) for s in specs] == [(0, 32), (32, 32), (64, 20)]
    assert list(iter_chunks(TensorSpec("empty", "float32", (7, 0)), SCHEMA, 32)) == []
    assert list(iter_chunks(TensorSpec("scalar", "int64", ()), SCHEMA, 32))[0].byte_length == 8
    raw = bytes(range(84))
    outputs = []
    for spec in specs:
        target = CanonicalChunk(spec, 0, raw[spec.byte_offset : spec.byte_offset + spec.byte_length])
        encoded = DeltaEncoder().encode(target)
        outputs.append(DeltaEncoder().decode(encoded, expected=encoded.descriptor).data)
    assert b"".join(outputs) == raw


@pytest.mark.parametrize(
    "case",
    ["dtype", "name", "shape", "empty", "unaligned", "bounds", "mutable", "length", "chunk_limit", "metadata_limit"],
)
def test_invalid_contracts(case: str) -> None:
    with pytest.raises(DeltaCodecError):
        if case == "dtype":
            TensorSpec("w", "complex128", ())
        elif case == "name":
            TensorSpec("../w", "float32", ())
        elif case == "shape":
            TensorSpec("w", "float32", (-1,))
        elif case == "empty":
            ChunkSpec(SCHEMA, TensorSpec("w", "float32", (0,)), 0, 0)
        elif case == "unaligned":
            list(iter_chunks(TensorSpec("w", "float32", (3,)), SCHEMA, 3))
        elif case == "bounds":
            ChunkSpec(SCHEMA, TensorSpec("w", "float32", (1,)), 4, 4)
        elif case == "mutable":
            replace(chunk(bytes(4)), data=bytearray(4))
        elif case == "length":
            replace(chunk(bytes(4)), data=b"x")
        elif case == "chunk_limit":
            CodecLimits(max_chunk_bytes=0)
        else:
            DeltaEncoder(CodecLimits(max_metadata_bytes=32)).encode(chunk(bytes(4)))


def test_temporary_buffers_scale_with_one_chunk() -> None:
    size = 256 * 1024
    base = chunk(bytes(size))
    target = replace(base, version=1, data=b"\x01\x00\x00\x00" * (size // 4))
    encoder = DeltaEncoder(CodecLimits(max_chunk_bytes=size))
    tracemalloc.start()
    try:
        encoded = encoder.encode(target, base=base)
        decoded = encoder.decode(encoded, expected=encoded.descriptor, base=base)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert decoded == target
    # Includes retained encoded/decoded results. Python indices must not form a
    # per-element object list; that would exceed this bound on this fixture.
    assert peak < 4 * size + 128 * 1024


def test_import_and_roundtrip_without_backend_or_network_modules() -> None:
    code = """
import importlib.abc
import sys
blocked = {"torch", "numpy", "ray", "megatron", "sglang", "socket", "requests"}
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in blocked:
            raise AssertionError("backend or network import: " + fullname)
sys.meta_path.insert(0, Block())
from relax.distributed.weight_sync import CanonicalChunk, ChunkSpec, DeltaEncoder, TensorSpec
spec = ChunkSpec("a" * 64, TensorSpec("w", "bfloat16", (512,)), 0, 1024)
base = CanonicalChunk(spec, 0, bytes(1024))
target = CanonicalChunk(spec, 1, b"\\x01" + bytes(1023))
encoder = DeltaEncoder()
encoded = encoder.encode(target, base=base)
assert encoder.decode(encoded, expected=encoded.descriptor, base=base) == target
assert not blocked.intersection(sys.modules)
"""
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_selection_matches_independently_materialized_candidates() -> None:
    priority = [Codec.COPY_BASE, Codec.SPARSE_REPLACE_V1, Codec.BITMAP_REPLACE_V1, Codec.RAW_V1]
    for count in (7, 64, 127, 512, 1024):
        for changed in sorted({0, 1, count // 32, count // 32 + 1, count // 2, count}):
            base = chunk(bytes(count * 2))
            target = b"\x01\x80" * changed + bytes((count - changed) * 2)
            bitmap = bytearray((count + 7) // 8)
            for index in range(changed):
                bitmap[index // 8] |= 1 << (index % 8)
            candidates = [
                encoded_for(Codec.RAW_V1, target, target, count, base),
                encoded_for(
                    Codec.SPARSE_REPLACE_V1,
                    target,
                    b"".join(struct.pack("<I", i) for i in range(changed)) + target[: changed * 2],
                    changed,
                    base,
                ),
                encoded_for(Codec.BITMAP_REPLACE_V1, target, bytes(bitmap) + target[: changed * 2], changed, base),
            ]
            if changed == 0:
                candidates.append(encoded_for(Codec.COPY_BASE, target, b"", 0, base))
            expected = min(
                candidates, key=lambda value: (len(value.to_bytes()), priority.index(value.descriptor.codec))
            )
            assert DeltaEncoder().encode(replace(base, version=1, data=target), base=base) == expected


def test_sparse_uint32_indices_above_uint16_range() -> None:
    base = chunk(bytes(70000 * 2))
    target = bytearray(base.data)
    target[65535 * 2 : 65537 * 2] = b"\x01\x00\x00\x80"
    encoded = DeltaEncoder().encode(replace(base, version=1, data=bytes(target)), base=base)
    assert encoded.descriptor.codec == Codec.SPARSE_REPLACE_V1
    assert encoded.payload == bytes.fromhex("ffff00000000010001000080")
    assert DeltaEncoder().decode(encoded, expected=encoded.descriptor, base=base).data == bytes(target)
    with pytest.raises(DeltaCodecError, match="uint32"):
        ChunkSpec(SCHEMA, TensorSpec("huge", "uint8", (1 << 32,)), 0, 1 << 32)


@pytest.mark.parametrize(
    "dtype,negative_zero,nan1,nan2",
    [
        ("bfloat16", "0080", "c17f", "c27f"),
        ("float16", "0080", "017e", "027e"),
        ("float32", "00000080", "0100c07f", "0200c07f"),
    ],
)
def test_equal_numeric_values_with_different_bits_are_replacements(
    dtype: str, negative_zero: str, nan1: str, nan2: str
) -> None:
    old = bytes.fromhex(negative_zero + nan1)
    new = bytes(len(bytes.fromhex(negative_zero))) + bytes.fromhex(nan2)
    base = chunk(old + bytes(4096 - len(old)), dtype)
    target = replace(base, version=1, data=new + bytes(4096 - len(new)))
    encoded = DeltaEncoder().encode(target, base=base)
    assert encoded.descriptor.replacement_count == 2
    assert DeltaEncoder().decode(encoded, expected=encoded.descriptor, base=base) == target


def test_synthetic_version_sequence_with_periodic_full() -> None:
    rng = random.Random(1623)
    encoder = DeltaEncoder()
    installed = chunk(rng.randbytes(2048), version=0)
    for version in range(1, 101):
        data = bytearray(installed.data)
        changes = [0, 2, 100, 1024][version % 4]
        for index in rng.sample(range(1024), changes):
            data[index * 2] ^= 1
        target = replace(installed, version=version, data=bytes(data))
        encoded = encoder.encode(target, base=None if version % 10 == 0 else installed)
        if version % 10 == 0:
            assert encoded.descriptor.codec == Codec.RAW_V1
        installed = encoder.decode(
            EncodedChunk.from_bytes(encoded.to_bytes()), expected=encoded.descriptor, base=installed
        )
        assert installed == target

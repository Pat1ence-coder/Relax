# CPU Delta codec

This package implements the first, transport-independent piece of the Delta
Weight Sync Engine: lossless encoding and verified reconstruction of a single
canonical chunk. It uses only the Python standard library. It does not import
Torch, Ray, Megatron, SGLang, networking, or storage backends.

## Contract

The exporter supplies immutable `bytes` in little-endian, C-contiguous logical
order. `TensorSpec` records the stable tensor name, dtype, and global shape;
`ChunkSpec` binds a trusted schema ID and an element-aligned byte interval.
`CanonicalChunk` adds the source version and bytes. `iter_chunks` yields bounded
intervals, including tensor tails. Empty tensors yield no chunks; the future
model manifest must still retain their directory entries.

The exporter owns tensor normalization, consistent snapshot acquisition, schema
construction, aliases/buffers, and layout conversion. This byte codec cannot
verify those upstream semantics. `version` is an unsigned 64-bit integer scoped
by the caller's trusted stream/epoch; it is not a global version authority.

```python
from relax.distributed.weight_sync import (
    CanonicalChunk, ChunkSpec, DeltaEncoder, EncodedChunk, TensorSpec,
)

spec = ChunkSpec("a" * 64, TensorSpec("model.weight", "bfloat16", (512,)), 0, 1024)
base = CanonicalChunk(spec, version=0, data=bytes(1024))
target = CanonicalChunk(spec, version=1, data=b"\x01\x80" + bytes(1022))
encoder = DeltaEncoder()
encoded = encoder.encode(target, base=base)

# Any transport can carry this bounded envelope. The trusted descriptor must
# be delivered through the future authenticated manifest/publication layer.
trusted_descriptor = encoded.descriptor
received = EncodedChunk.from_bytes(encoded.to_bytes())
rebuilt = encoder.decode(received, expected=trusted_descriptor, base=base)
assert rebuilt.data == target.data

# FULL/anchor chunks are independent of any base.
full = encoder.encode(target)
assert encoder.decode(full, expected=full.descriptor).data == target.data
```

## Encoding and envelope version 1

Selection minimizes the actual envelope size, including metadata and payload.
Ties prefer COPY, SPARSE, BITMAP, then RAW. Even an unchanged tiny chunk may use
RAW when its smaller base-independent descriptor makes the total smaller.
The selected representation is never larger than the corresponding FULL RAW
envelope. No floating-point subtraction, casting, or numerical equality is used.

| Codec               | Payload                                                                    | Base dependency                                                                     |
| ------------------- | -------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| `COPY_BASE`         | Empty                                                                      | Exact identity, version, and hash; base/target hashes equal                         |
| `SPARSE_REPLACE_V1` | `k` little-endian uint32 element indices, then `k` original element values | Exact base; indices strictly increasing, unique, and in range                       |
| `BITMAP_REPLACE_V1` | `ceil(n/8)` bytes followed by values in set-bit order                      | Exact base; least significant bit first; unused tail bits zero; popcount equals `k` |
| `RAW_V1`            | All target bytes                                                           | None, including when an unrelated base is supplied to decode                        |

`replacement_count` is `k` for sparse/bitmap, zero for COPY, and `n` for RAW.
For RAW it counts stored values, **not** measured differences from a prior
snapshot; it must not be used as a model change-rate metric.

`EncodedChunk.to_bytes()` emits a 16-byte `<4sIQ` header: ASCII `DWC1`, a
little-endian uint32 metadata length, and a little-endian uint64 payload length.
Canonical JSON metadata follows, then the payload, with no trailing bytes.
JSON uses UTF-8, sorted keys, `ensure_ascii=True`, and separators `(',', ':')`.
Shapes are arrays. Versions, counts, offsets, and lengths are unsigned integers;
booleans, floats, duplicate keys, unknown/missing fields, and noncanonical JSON
are rejected. Format and codec interpretation changes require a new version.

The descriptor contains `format_version`, `chunk`, `target_version`,
`base_version`, `codec`, `replacement_count`, `encoded_length`, `payload_hash`,
`target_hash`, and `base_hash`. `chunk` contains `schema_id`, `tensor`,
`byte_offset`, and `byte_length`; `tensor` contains `name`, `dtype`, and `shape`.
Hashes are lowercase SHA-256 hex over the original bytes. RAW base fields are
JSON null. This envelope is a codec object, not a TCP frame or a full model
manifest. Its descriptor and payload can also be stored separately by a later
artifact store.

## Verification and limits

Decode compares the received descriptor with the caller's trusted expectation,
checks payload size/hash and codec structure, verifies the exact base for
dependent codecs, and verifies reconstructed target bytes. It returns immutable
bytes and never mutates base or live model storage. Repeating decode against
the same valid base is safe. A partially reconstructed buffer or an already
target-version chunk cannot silently become the base. Failures raise
`DeltaCodecError`; persistent cache reuse and full-sync fallback orchestration
belong to later layers.

Default `CodecLimits` are 8 MiB per decoded/encoded chunk, 16 KiB metadata,
1,024 UTF-8 bytes per tensor name, and rank 32. Descriptor lengths, identities,
index order/range, bitmap padding, and population are checked before allocating
reconstruction output. Envelope bounds are checked before copying metadata or
payload. Transports must enforce their own limits before reading input buffers.
The encoder scans before allocating the selected payload and never builds a
per-element Python index list. Data workspace scales with one chunk; callers
must separately bound concurrency and retained inputs/outputs. This is a CPU
reference implementation, not a claim of GPU throughput or process RSS bounds.

## Validation and remaining integration

Run `python -m pytest tests/distributed/weight_sync -q`. Tests include independent
payload vectors, all four modes, arbitrary bit patterns, dtype widths, NaN
payloads, negative zero, malformed metadata/indices/bitmaps, wrong bases,
serialization sizes, temporary allocation bounds, and blocked backend/network
imports. A 100-version fixture is **synthetic**, not real-model acceptance.

Subsequent work supplies the complete schema/inventory/root and trusted model
manifest, consistent training snapshots, shared-storage/TCP transports, durable
version/apply state, and the inference Weight Loader. CPU tests do not establish
resharding correctness, actual loaded weights, fault recovery, or 50% traffic
savings.

# CPU Delta codec and canonical snapshots

This package implements transport-independent lossless chunk encoding and
verified reconstruction of complete canonical model snapshots. It uses only
the Python standard library. It does not import
Torch, Ray, Megatron, SGLang, networking, or storage backends.

## Contract

The exporter supplies immutable `bytes` in little-endian, C-contiguous logical
order. `TensorSpec` records the stable tensor name, dtype, and global shape;
`ChunkSpec` binds a trusted schema ID and an element-aligned byte interval.
`CanonicalChunk` adds the source version and bytes. `iter_chunks` yields bounded
intervals, including tensor tails. Empty tensors yield no chunks; `ModelSchema`
retains their directory entries.

The exporter owns tensor normalization, consistent snapshot acquisition,
discovering the complete tensor/alias inventory, and layout conversion. The
core validates the supplied schema but cannot verify those upstream semantics.
`version` is an unsigned 64-bit integer scoped
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
# be bound by a manifest whose identity comes from a trusted publication layer.
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
artifact store. Model indexes below store descriptors separately from payloads;
they do not embed the `DWC1` header.

## Canonical model schema and root, version 1

`ModelSchema` includes `logical_config_hash`, `converter_semantics_id`, the
fixed `chunk_bytes` rule, little-endian/C order, and the complete directory.
The exporter supplies both semantic hashes; GPU type and physical TP/PP/EP
placement do not belong in them. Every `TensorEntry` records name, dtype,
global shape, kind (`parameter` or `buffer`), and nullable `alias_of`. Entries
are sorted by UTF-8 name bytes. Duplicate names, unsupported kinds, and
unaligned chunk rules are rejected. Parsed directories must already be sorted.

An alias must point directly to a non-alias owner with identical dtype and
shape. Cycles, chains, missing owners, and partial/view aliases are rejected.
Only owners produce chunks; aliases and zero-sized tensors remain in the
directory. `logical_nbytes` counts all entries including aliases;
`canonical_nbytes` counts owner storage once. `chunk_count` counts only
nonempty owner intervals. The supplied inventory must include every weight and
buffer required by the model; the core cannot discover omitted model state.

Schema JSON has exactly `format_version: 1`, `logical_config_hash`,
`converter_semantics_id`, `byte_order: "little"`, `order: "C"`, `chunk_bytes`,
and `tensors`. Each tensor has `name`, `dtype`, `shape`, `kind`, `alias_of`.
`schema_id = SHA256(canonical schema JSON)` and
`directory_hash = SHA256(canonical tensors array JSON)`. Receiver limits are
configuration, not schema fields.

The complete content root uses SHA-256 with length-prefixed domain separation.
Define `H(tag, fields...) = SHA256(b"DWS1" || L(tag) || L(field1) || ...)`,
where `L(x) = uint64_le(len(x)) || x`. Digest fields are **32 raw bytes**,
names are UTF-8, and integer fields below are uint64 little-endian:

- Leaf: `H("chunk-leaf", schema_id, name, byte_offset, byte_length, SHA256(data))`.
- Leaves follow directory name order, then ascending chunk byte offset.
- Pair: `H("merkle-node", left, right)`. At each level, an unpaired last node
  becomes `H("merkle-odd", node)`; a single remaining node is the tree root.
- An empty tree is `H("merkle-empty")`.
- `target_root = H("target-root", schema_id, directory_hash, chunk_count, tree_root)`.

`ModelRoot` retains O(log chunk_count) hashes. It is an ordered accumulator;
the model pipeline separately enforces exact catalog coverage. Binding the
directory covers aliases and empty tensors. Version numbers, codecs, page
boundaries, writer fences, and transport details are excluded from this root.
FULL and DELTA encodings of the same canonical model have the same root.
Format changes require new versions/domains, not silent reinterpretation.

## Manifest and paged index, version 1

`SnapshotIdentity` is `(stream_id, run_epoch, version, schema_id, target_root)`.
Stream/epoch/exporter revision are bounded identifiers, not file paths or
endpoints. A `Manifest` binds its schema, target identity, FULL/DELTA kind,
optional base identity, index page references, `writer_fence`, `source_step`,
and `exporter_revision`. DELTA requires the same stream, epoch, and schema as
its strictly older base. FULL has no base. A later full anchor uses the same
build API with `base=None`; scheduling anchors belongs to the controller.

The top-level JSON fields are exactly `format_version`, `required_features`,
`schema`, `target`, `kind`, `base`, `pages`, `writer_fence`, `source_step`,
and `exporter_revision`. Required features, in canonical order, are
`canonical-le-v1`, `chunk-codecs-v1`, `model-root-v1`, `paged-index-v1`.
Unknown versions/features are rejected. `manifest_id` is SHA-256 of these
canonical bytes; it is not a self-referential JSON field. Parsing and
reconstruction require an externally trusted `expected_manifest_id`.
Hash equality alone does not authenticate a producer or select a current
version; publication/authentication and fence arbitration are separate layers.

Each `PageRef` has `first_chunk`, `chunk_count`, `byte_length`, `page_hash`,
and `object_key = indexes/{page_hash}.json`. Ranges must be contiguous and
exactly cover the derived schema chunk count. Each page contains
`format_version: 1`, `first_chunk`, and `records`; each `ChunkRecord` contains
`descriptor` (the chunk metadata above) and `object_key`. Non-COPY payloads
are standalone bytes at `chunks/{payload_hash}`. COPY has a null object key
and no stored payload. V1 does not support packed offsets or external URLs.
These are logical artifact keys, independent of local/shared storage or TCP.

Page size/hash/count and every descriptor are checked before payload reads.
Descriptors must match the next expected schema chunk and target version,
so duplicate, missing, reordered, overlapping, and incompatible chunks cannot
produce a verified snapshot. Whole manifest/index/model budgets are separate
from per-chunk limits. Pages flush on either count or byte budget.

## Snapshot pipeline and adapter obligations

The core exposes small synchronous protocols: `ChunkReader` (canonical bytes),
`ArtifactWriter`/`ArtifactReader` (index/payload objects), and `StagingSink`
(private reconstruction storage). None initializes a backend or network stack.

1. `build_manifest(schema, target_reader, writer, ..., base=None)` reads one
   owner chunk at a time, emits payloads and bounded index pages, and returns
   the manifest only after completion. The caller holds an immutable,
   cross-tensor consistent export lease throughout the call. Writes remain
   unpublished; the caller reclaims orphaned objects on failure.
2. `verify_snapshot(schema, identity, reader)` reads **all** owner chunks and
   checks the complete root before returning `VerifiedSnapshot`. A previously
   reconstructed snapshot also supplies this handle. This prevents unchecked
   storage from being used as a base merely because a referenced chunk matches.
   The caller owns the handle's immutable lease; subsequent mutation invalidates
   the contract. The constructor guard is an API check, not Python isolation.
3. `reconstruct(manifest_bytes, artifacts, staging, expected_manifest_id=..., base=...)` verifies metadata and complete base
   identity, writes decoded chunks privately, verifies the target root, then
   **reads stored staging bytes back** and verifies the complete root again.
   Only after successful `seal()` does it return a verified handle. Any error
   after `begin()` invokes `abort()`, including failed begin, readback, or seal.
   Adapter IO exceptions propagate; malformed/corrupt data raises
   `DeltaCodecError`. If abort also fails, the original error is preserved with
   the abort error chained, and adapter recovery must reclaim that generation.

`StagingSink.begin()` reserves an isolated generation and retains the complete
catalog, including zero-sized tensors and owner aliases. Its reader must read
actual stored bytes. `seal()` freezes these bytes without publishing or
activating them. `abort()` discards only that generation. Consumers must retain
the immutable lease until all dependent readers finish. A supplied DELTA base
must match every identity field; dependent chunks also check base version/hash.
An entirely RAW DELTA can recover with `base=None`. Pass `base=None` for FULL.

This API returns a canonical CPU/storage snapshot, not a live GPU model. It
does not acquire training leases, allocate device tensors, materialize aliases
on a backend, reshard, activate weights, or implement durable COMMIT/ACK state.
Repeated reconstruction into a new staging generation is deterministic;
persistent idempotence, crash recovery, retries, version catch-up, and fallback
decisions still require the version/transport/loader layers.

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

Default `SnapshotLimits`: 8 MiB catalog, 16 MiB manifest, 256 KiB index page,
256 MiB total indexes, 65,536 tensors, 1,048,576 chunks, 8,192 pages, 128 chunks
per page, and 1 TiB logical/canonical model bytes. The directory and page
references reside in bounded memory; chunk descriptors are paged and payloads
are streamed. Temporary metadata can occupy multiple bounded representations
(parsed objects plus JSON bytes); these are serialized-size budgets, not exact
Python heap limits. Readers/writers must enforce byte budgets before IO
allocation and bound queues/concurrency/storage independently. The synchronous
core does not implement network backpressure.

## Validation and remaining integration

Run `python -m pytest tests/distributed/weight_sync -q`. Tests include independent
payload vectors, all four modes, arbitrary bit patterns, dtype widths, NaN
payloads, negative zero, malformed metadata/indices/bitmaps, wrong bases,
serialization sizes, temporary allocation bounds, and blocked backend/network
imports. A 100-version fixture is **synthetic**, not real-model acceptance.

Model tests additionally cover independent list-based Merkle oracles and fixed
root vectors, aliases/buffers/empty tensors, paged metadata validation, wrong
complete base roots and epochs, FULL/DELTA root equality, staging readback,
abort paths, and 100 synthetic model versions with periodic anchors. A 64 MiB
virtual model with temporary-file staging checks bounded Python allocations;
it is not a GPU or transport performance benchmark.

Subsequent work supplies consistent real training snapshots, a trusted manifest
publication layer, shared-storage/TCP transports, durable version/apply state,
and the inference Weight Loader. CPU tests do not establish
resharding correctness, actual loaded weights, fault recovery, or 50% traffic
savings.

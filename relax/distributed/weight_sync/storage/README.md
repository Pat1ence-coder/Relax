# POSIX shared storage and offline replay

These optional adapters implement the existing `ArtifactReader`,
`ArtifactWriter`, and `StagingSink` protocols. The canonical schema, manifest,
root, and codec formats are unchanged. Import them explicitly from
`relax.distributed.weight_sync.storage`; importing the core does not initialize
filesystem, SQLite, Ray, GPU, or network resources.

## Deployment and trust

- The artifact root is an application-owned POSIX directory with a trusted
  publisher ACL. Consumers may mount it read-only. The deployment must support
  advisory locks, atomic hard links, and durable file/directory `fsync`.
  Actual shared-filesystem behavior must be verified on the target mount.
- `ProducerCatalog(local_control_dir, ...)` uses a **separate local persistent
  volume** for SQLite WAL and the authority lock. Never place this directory on
  NFS or another shared filesystem. The API rejects nested artifact/control
  roots; it cannot determine whether two unrelated paths are on a remote mount.
- One process/thread owns artifact writes; a second writer is rejected, including
  use of an inherited writer after fork. The local authority also has an
  exclusive process lock. These locks do not implement automatic cross-host
  takeover. Other clients must not write directly to the reserved roots.
- Each root is authorized for one stream/epoch and bound to one persistent
  authority ID. Reopening a different control database against the same root
  fails. Epoch transitions, authority rollback/restore, and distributed fencing
  require a separate control protocol; they are not inferred from UUIDs.
- SHA-256 detects corruption. It does not authenticate an untrusted publisher.
  Offline discovery trusts the shared root's publisher ACL; a sealed archive can
  additionally be opened using an externally trusted archive ID. Signatures and
  cross-trust-domain authentication are not provided by this POSIX adapter.

Root paths come from trusted application configuration. Child directories and
files are opened with directory descriptors and `O_NOFOLLOW`; non-regular
objects, path traversal, and oversized files are rejected. The control volume is
private to the authority, including its SQLite sidecar files.

## Artifact durability

`PosixArtifactStore(root, writable=True)` implements bounded payload/index IO and
manifest storage. Read-only instances never create objects. Files are written
to a unique temporary name in the same directory, fully written, made read-only,
and fsynced. An atomic hard link installs the final name **without overwriting**,
then the directory is fsynced and the temporary name removed. Repeating a write
requires identical existing bytes. An error after the link may leave a complete
final object; retrying checks it and completes directory durability.

The core's logical keys remain portable:

| Contents                  | Key                                         |
| ------------------------- | ------------------------------------------- |
| Encoded chunk payload     | `chunks/{payload_hash}`                     |
| Chunk index page          | `indexes/{page_hash}.json`                  |
| Canonical manifest        | `manifests/{manifest_id}.json`              |
| Exported committed record | `catalog/{version:020d}-{manifest_id}.json` |
| Sealed archive descriptor | `archives/{archive_id}.json`                |
| Publisher binding         | `authority.json`                            |

Raw object methods also enforce per-kind byte limits and content-addressed keys.
Readers check file type/length before allocating, handle short reads, and reject
truncation/trailing bytes. Payload, index, and manifest readers verify hashes and
their existing formats. Publication verifies stored artifacts, exact descriptor
coverage, base versions, and the descriptor-derived model root. Consumers still
decode and verify actual target bytes; publication is not a substitute for that
verification.

## Publication authority and recovery

Create the catalog and allow its recovery/export to finish **before starting new
artifact writes after restart**. SQLite must report WAL mode and
`synchronous=FULL`. The database has a page budget. A truncate checkpoint occurs
before each new authority transaction and after each successful decision;
automatic checkpointing is also enabled. A blocked checkpoint stops further
decisions, preventing repeated transactions from growing the WAL unchecked.

`acquire_writer()` advances a persistent fence. Build manifests using that fence.
`publish(manifest, operation_id=..., expected_head=...)` performs these steps:

1. Resolve a previously committed operation first. Identical retries return its
   original decision even after a newer writer fence has been acquired. Reusing
   an operation ID with a different request fails.
2. Verify and persist data/index/manifest objects.
3. In one SQLite transaction, check the current fence and expected head, require
   a FULL initial version, enforce increasing versions and compatible schemas,
   verify the exact committed base identity, and limit dependency depth.
4. Reserve space for the exported committed record, then atomically record the
   operation, immutable publication/dependency, version representations, and new
   head. This producer head identifies the committed comparison base; it is
   independent of consumer installation/activation state.
5. Export the same immutable publication record to shared storage.

The publication record contains `format_version: 1`, `authority_id`,
`operation_id`, `manifest_id`, the complete `target` identity, `kind`,
`writer_fence`, `expected_head`, `base_manifest_id`, and `advances_head`.
The manifest hash binds source step and all source metadata as well. Versions
use fixed-width decimal database keys, preserving the canonical uint64 range.
Writer fences are limited to SQLite's positive signed 64-bit range.

`PublicationUncertain.operation_id` means the caller must query
`resolve(operation_id)` or retry the **same request**. An exception after COMMIT
cannot revoke that decision. A missing result after a known rollback permits the
same candidate to retry; an unavailable database is not evidence of an abort.
Do not replace the candidate or advance a producer comparison cache based only
on whether the method returned. `export_pending()` replays all committed records
idempotently, including on catalog startup. Records without exported files may
temporarily be undiscoverable offline, but are still committed.

Reserved bytes/object slots remain unavailable to other artifact writes while a
record awaits export. Successful export releases the reservation; a known
transaction rollback releases it. After a crash, catalog startup completes
exports before accepting further producer work. Orphaned **final** objects remain
retained within quotas; they are not confused with published versions.

`publish(full_manifest, ..., attach_full=True)` can attach one fallback FULL to
an already committed DELTA version. Its entire target identity must match. It
does not advance head or create a new logical version. Subsequent DELTAs prefer
the committed FULL representation of their base, resetting dependency depth.
Full materialization still requires the retained original canonical bytes; this
adapter cannot reconstruct lost history from current training parameters.

## Offline discovery and sealed archives

`OfflineCatalog(store, stream_id=..., run_epoch=...).records()` reads exported
committed records without a running producer or database. This is a bounded view
of what has been exported, not an assertion that the producer has no newer
version. There is no mutable `latest` pointer acting as a publication authority.

`seal_archive(versions=(...), prefer_full=True)` requires a sorted, unique,
explicit requested version set. It selects committed representations and closes
their transitive base dependencies, including anchors outside the requested set.
It checks metadata, identities, dependency depth, and stored artifact hashes,
then writes an immutable descriptor last. Missing requested versions/dependencies
fail; the method never silently shortens the requested set.

The archive JSON contains `format_version: 1`, `authority_id`, `stream_id`,
`run_epoch`, requested manifest IDs, and ordered record references containing
`key` and `record_hash`. Its content hash is the archive ID. All referenced
records and objects are permanently retained in this first implementation.

`OfflineArchive.open(store, expected_archive_id=...)` verifies the exact
descriptor, every committed record, the complete dependency closure, and encoded
artifacts before yielding manifests in dependency order. No online producer is
needed. Consumer reconstruction then checks decoded model bytes. The current
implementation deliberately re-reads some objects for these separate checks;
it is a correctness reference, not an optimized traffic benchmark.

## Disk generations and leases

`DiskSnapshotStore(root, max_bytes=..., max_generations=...)` owns isolated
generations with one flat canonical owner-data file, a complete schema/identity
metadata file, and a seal marker. It permits only one unfinished generation per
store; completed generations remain readable while the next is built.

Use `staging()` with the core's `reconstruct`. Writes must exactly follow the
schema chunk order/version. The core reads stored bytes back and verifies the
whole target root before `seal()` fsyncs data and installs the marker. The
generation path never moves, so readers created before seal remain valid after
seal. Aliases map to owner offsets; zero-sized tensors remain in metadata.

`open_snapshot(path, expected_identity=...)` verifies the seal/metadata and
recomputes the **complete root of actual disk bytes**. A marker alone never
proves a complete snapshot. It returns `(VerifiedSnapshot, DiskSnapshotReader)`;
close the reader only after all dependent snapshot users release their leases.
For a new staging generation, `stage.reader()` provides the same explicit close
handle. Sealed generations are retained and cannot be aborted through staging.

Failure during reconstruction aborts only its private generation. Closing the
store aborts its unfinished generation. After a process crash, call
`cleanup_incomplete()` under the exclusive writer lock; it skips sealed and
currently active generations and refuses unknown contents. It does not trust a
partial weight file as either the old base or a reusable complete target.

## Budgets and remaining integration

`StorageLimits` defaults to 1 TiB final artifact bytes, 1,048,576 final objects,
4,096 publication records, 16 KiB per publication/binding record, 1 MiB per
archive descriptor, 128 DELTA links, and a 64 MiB main database. Filesystem
metadata, SQLite WAL/shared-memory files, and page cache consume additional
resources; the main database limit is not a total process/disk limit. One
temporary artifact per synchronous writer adds at most the current object size
and temporary directory entry. Failure cleanup runs before further writes; if
inventory repair fails, the writer refuses further writes until closed and
reopened successfully.

Artifact and snapshot usage is re-counted on restart; lowering a budget below
retained usage fails. Snapshot budgets count logical data/metadata file sizes
and generations; sparse-file allocation does not exempt a snapshot from quota.
Actual free-space errors propagate and preserve prior committed state. Snapshot
stores reserve metadata/seal headroom before admitting their single candidate.
Deployment volume quotas should additionally bound filesystem overhead and
unrelated users. These are not GPU/RSS limits.

This stage retains all final objects, records, archive dependencies, and sealed
generations, and stops at quota rather than deleting possibly live data. General
pin-aware GC, consumer durable COMMIT/ACK and activation, resumable partial chunk
reuse, epoch switching, TCP/backpressure, real training Exporter, Weight Loader,
resharding, and traffic accounting remain separate work. Recovery here retries
immutable writes/publications or rebuilds a discarded private generation.

Tests use real temporary files and SQLite, process exits around publication,
short IO/fsync/disk-full faults, quota exhaustion and restart, corrupted snapshots,
fallback FULL attachments, and offline replay after producer shutdown. Dense-
and MoE-shaped **synthetic** catalogs each run V0 plus 100 updates with periodic
anchors. These tests do not satisfy real-model, heterogeneous GPU/layout, shared
mount durability, or 50% total-traffic acceptance criteria.

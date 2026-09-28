# POSIX shared storage and offline replay

These optional adapters implement the existing `ArtifactReader`,
`ArtifactWriter`, and `StagingSink` protocols. The canonical schema, manifest,
root, and codec formats are unchanged. Import them explicitly from
`relax.distributed.weight_sync.storage`; importing the core does not initialize
filesystem, SQLite, Ray, GPU, or network resources.

## Deployment and trust

- The artifact root is an application-owned POSIX directory with a trusted
  publisher ACL. Consumers may mount it read-only. The deployment must support
  Linux OFD advisory locks, atomic hard links, and durable file/directory `fsync`.
  Actual shared-filesystem behavior must be verified on the target mount.
- `ProducerCatalog(local_control_dir, ...)` uses a **separate local persistent
  volume** for SQLite WAL and the authority lock. Never place this directory on
  NFS or another shared filesystem. The API rejects nested artifact/control
  roots; it cannot determine whether two unrelated paths are on a remote mount.
- One process/thread owns artifact writes; a second writer is rejected, including
  use of an inherited writer after fork. The local authority also has an
  exclusive process lock. These locks do not implement automatic cross-host
  takeover. Other clients must not write directly to the reserved roots.
- The lock adapter uses nonblocking `F_OFD_SETLK` on 64-bit Linux x86-64/AArch64.
  Unsupported ABIs, runtimes or filesystems fail without a weaker fallback.
  OFD locks remain held until the last descriptor for that open file description
  closes; opening and closing another descriptor does not release the lock.
  Cross-node conflict and process-exit recovery must still be tested on the
  target mount. Stop all old writers before upgrading: previous releases used
  `flock`, which cannot be assumed to conflict with OFD locks. Do not mix writer
  lock protocols, even when reusing a compatible format-v1 namespace.
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

### Dedicated artifact directory

Provision a dedicated directory for each stream/run epoch inside the existing
shared volume. For a volume containing other workloads, a suggested layout is:

```text
<shared-mount>/
  models/
  datasets/
  other-jobs/
  relax-delta-sync/
    <project>/
      <stream-id>/
        <run-epoch>/                 # artifact_root
          namespace.json
          authority.json
          layout-<uuid>/             # selected by namespace.json
            chunks/
            indexes/
            manifests/
            catalog/
            archives/
```

Set `artifact_root` to the final epoch directory. This naming convention is a
deployment choice; the API uses the supplied path exactly and does not append
project, stream or epoch components. `artifact_volume.mount_point` separately
identifies the containing mount. Do not pass a mixed-use directory such as the
volume root, a model directory or a common job directory as `artifact_root`.
The writer treats that directory as its owned object namespace for inventory,
quotas and temporary-file recovery. Parent and sibling directories are outside
that namespace; deployment must reserve it exclusively for this engine.

Restarting the same run reuses its epoch directory, binding and persistent
control database. A new independent run uses a new epoch directory and matching
binding; do not repurpose the old directory by replacing its marker. Versions
within one epoch share the same root and reference immutable objects. Consumers
select that same backend directory, even if their mount path differs. This
layout does not alter artifact keys or hashes.

The strict deployment entry points require the directory to be provisioned
before explicit namespace initialization. They do not choose a project path or
detect all mixed-use directory mistakes automatically. Keep local control and
consumer staging roots separately configured as described above.

## Storage contracts and deployment admission

`StorageReader` exposes bounded artifact reads and committed-record discovery.
`StorageWriter` adds immutable writes, control-placement approval, exact-record
capacity reservations and a recovery barrier. `ProducerCatalog` and offline
consumption use these public protocols; they do not access a POSIX root or
private file descriptor. Shared manifest validation lives in `repository.py`.
The producer authority itself remains a local SQLite WAL implementation.

Shared validation rechecks every returned payload's bytes type, length and hash,
including empty COPY payloads, independently of adapter-side checks. RAW target
hash and model descriptor-root checks remain separate; decoding still verifies
the reconstructed bytes.

`StoreCapabilities` describes implementation requirements and access roles. It
does not claim that a mount has passed compatibility tests. `ObjectReceipt`
binds a completed write to its key, length, content hash, configured namespace
and declared fault domain. It is neither a publication decision nor a consumer
COMMIT. The low-level, unconfigured store reports no certified fault domain.

Use `PosixDeployment` and `open_deployed_store` for deployment-aware access.
The low-level `PosixArtifactStore(root, ...)` constructor remains available for
existing callers and component fixtures. It does **not** perform full deployment
admission and must not be presented as a checked production deployment. No CLI
flags or training/service lifecycle integration are added by these APIs.

Deployment configuration supplies the following explicitly:

| Configuration                            | Meaning                                                                                                         |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `artifact_root`, `namespace`             | Process-visible root and trusted namespace/stream/epoch binding; paths may differ between nodes                 |
| `artifact_volume`                        | Expected mount point, filesystem type, source/root digests, capacity identity and evidence                      |
| `control_root`, `control_volume`         | Separate pre-existing private local persistent directory; required for a publisher                              |
| `staging_root`, `staging_volume`         | Optional pre-existing private persistent snapshot directory; admission checks placement, not snapshot lifecycle |
| `access`, `expected_uid`, `expected_gid` | Reader/publisher role and the effective identity that must actually execute it                                  |
| `required_fault_domain`                  | A fault domain covered by the operator-supplied volume evidence                                                 |
| `access_policy`                          | Private access or `PosixAccessPolicy(shared_group_id=...)` for an explicitly configured read group              |

Each `VolumeSpec` includes `volume_id`, a shared `capacity_bytes`, the allocation
`reserved_bytes`, and an `evidence_digest` referring to trusted deployment
evidence. Roots sharing capacity use the same ID and capacity. Reservations are
summed before admission; detectable aliases cannot claim independent budgets.
The operator must also identify aliases not inferable from a process's mount
table and account for other processes and filesystem overhead. These declarations
do not replace actual volume quotas or prove available free space.
Expected mount identities must come from trusted, previously established
deployment configuration, not be regenerated from whichever directory happens
to be present during startup.

Artifact allocation includes the store's final-byte limit plus one largest
temporary object. Control allocation reserves at least three main database
limits for the database and WAL/checkpoint workspace; filesystem overhead still
needs deployment headroom. Staging allocation must be passed to the caller's
`DiskSnapshotStore` budget and aggregated with other roots on that volume.
This first strict profile requires persistent staging; a disposable-cache
profile with explicit remote recovery sources remains separate work.

Admission reads a bounded Linux mount table, compares the expected mount
identity, checks the actual effective role, directory separation, access and
volume evidence, and verifies an immutable `namespace.json` marker. Raw mount
sources/options are not retained in public diagnostics. Known network/FUSE
control volumes are rejected even if incorrectly labelled local. Unknown volume
durability is not inferred from the filesystem name or `st_dev`.
The first local-control profile accepts only `ext2`, `ext3`, `ext4`, `xfs`,
`btrfs`, `zfs`, `f2fs` and `overlay`, together with explicit local/persistent
evidence. This list is a placement restriction, not a certification list;
especially an overlay or container volume must still have the required retained
backing storage. Other types require a separately assessed profile.

All configured directories must already exist. A missing shared mount, a local
empty directory with no binding, wrong namespace, or insufficient evidence fails
before opening a publisher. Admission pins directory inode/mount identities and
rechecks the opened descriptors; paths and mounts must remain stable while the
session is active. Root separation includes directory identities and filesystem
coordinates derived from mount roots, detecting overlapping bind-mount aliases.
Unresolved overlap on an identified common filesystem fails admission. External
aliases not visible in the local mount table still require deployment evidence.
Only component hashes are retained from mount roots for these comparisons.

The strict profile requires all artifact child directories to use the admitted
root mount. Existing children are checked before opening a writer lock, checked
again under that lock, and their final opened descriptors are checked before
object IO. Separately mounted children require a different deployment profile.
Namespace initialization is a separate, explicit
publisher-only call:

```python
# Trusted deployment/limits are supplied by the caller. This enrolls an
# existing, approved root; it does not create or mount a storage service.
initialize_namespace(deployment, limits=limits, storage_limits=storage_limits)
```

The marker is separate from `authority.json` and does not change schema,
manifest or payload identities. New enrollment creates a format-v2 descriptor
containing the namespace/stream/epoch binding and one relative `layout-<uuid>`
directory name. Its bytes and object slot count toward the normal artifact
quotas. A complete format-v1 namespace continues to use its original flat
directories; it is not migrated or rewritten. Missing published directories
fail for both roles, including publishers. An unbound legacy directory with
existing artifacts requires explicit migration outside this initializer;
low-level callers remain compatible with roots without a marker.

For a configured publisher, recovery precedes ordinary uploads:

```python
with open_deployed_store(deployment, limits=limits, storage_limits=storage_limits) as store:
    with ProducerCatalog(
        deployment.control_root,
        store,
        stream_id=deployment.namespace.stream_id,
        run_epoch=deployment.namespace.run_epoch,
    ) as producer:
        fence = producer.acquire_writer()
        # Build the manifest using store and fence, then publish it.
```

The writer opens in `RECOVERING`. The catalog approves and locks its local
control directory, restores the authority, exports committed records, and only
then calls `finish_recovery`. Failure leaves ordinary artifact uploads blocked;
reopening and recovering the same authority is required. Pending reservations
must be empty before the barrier opens. During recovery only binding/committed
record writes are allowed. The unconfigured constructor keeps its historical
direct-object behavior, but catalog startup still establishes the same barrier.

A configured reader opens only existing directories and never initializes a
namespace, takes a shared writer lock, or requires an online producer. A
read-only `OfflineCatalog` can discover records. Sealing an archive requires an
explicit publisher via `seal_archive(..., writer=writer)`; omitting that argument
remains compatible only when the catalog was created with a publisher. An
explicit writer must contain the same authority, committed records and verified
dependency closure, not just an equal namespace label.

Private roots are not made public automatically. Group-read mode requires the
artifact root to be pre-provisioned with the configured group and traversal/read
access, without group/other write access. Only newly created child directories
and objects receive the configured group/modes (`0750` / `0440`); existing parents
are never chmodded. Private mode uses `0700` / `0400` for newly created content.
Unconfigured stores keep their earlier `0700` / `0444` behavior. Legacy private
subdirectories are not silently migrated to group access. Actual cross-UID/ACL
behavior still requires testing with the real reader identity.

Namespace initialization holds the exclusive writer lock and creates a unique,
private layout directory. It prepares all five child directories, applies the
configured modes/groups, verifies each mount, and fsyncs the directories and
parent before publishing the descriptor. Publication uses the same immutable
file hard-link protocol as artifacts; directory rename support is not required.
Only the descriptor selects a usable layout. Readers never inspect unpublished
layouts, and namespace readiness does not replace catalog publication checks.

A retry after descriptor publication validates and reuses that layout, including
when the previous final parent fsync failed. Before publication, retries prepare
a new layout instead of modifying an ambiguous leftover. Unpublished layout
directories and root descriptor temporaries are retained and charged on restart.
Layouts may contain only the five reserved child directories; unpublished
children must be empty. Unexpected objects, symlinks or submounts stop recovery
without cleanup. Old `.tmp-dir-*` preparation objects are not adopted or deleted.
Repeated initialization with a valid descriptor creates no additional layouts.

`inspect_deployment(...)` performs read-only configuration/placement checks and
returns a bounded serializable `DeploymentReport`. It performs no challenge
writes, mount operations, remote calls, lock probes or failure injection. Public
reports contain config/evidence digests, role, namespace and stable error/check
codes, not raw paths, mount sources/options or credentials. Configuration
admission is distinct from compatibility: cross-node and infrastructure test
fields remain `NOT_RUN`, and `compatibility_verified` remains false. Operator
attestations are external inputs, not measurements generated by inspection.

The newly added contract/deployment test fixtures are not target-mount
certification. Cross-node challenges, visibility deadlines, blocked-I/O recovery,
actual reader identities, infrastructure fault guarantees and the full RFC
compatibility matrix still need implementation/integration or execution as
appropriate. No successful compatibility result follows from merely importing
these APIs or constructing configuration.

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
archive descriptor, 128 DELTA links, a 64 MiB main database, and 16 layout
directories (`max_layouts`, including the selected layout and retained attempts).
Each layout directory and child consumes one object slot and a fixed 4 KiB
logical charge from the byte quota, independent of physical directory growth.
Root descriptor temporaries count as
separate file entries even when hard-linked to the published descriptor. Scans
are bounded; exhausted budgets stop initialization before another attempt.
These logical charges do not measure all physical filesystem overhead. Filesystem
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

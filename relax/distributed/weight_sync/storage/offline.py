# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Committed-record discovery and sealed dependency closures without a DB."""

from collections.abc import Iterator
from dataclasses import dataclass

from ..codec.format import content_hash
from ..limits import DeltaCodecError, require_uint
from ..manifest import Manifest
from ..schema import require_digest
from ..serialization import canonical_json, exact_fields, parse_json, require_identifier
from .catalog import Publication
from .contracts import StorageReader, StorageWriter
from .repository import validate_artifacts


def _load_record(store: StorageReader, key: str, expected_hash: str | None = None) -> Publication:
    data = store.read_object(key, store.storage_limits.max_record_bytes)
    if expected_hash is not None:
        require_digest(expected_hash, "record hash")
        if content_hash(data) != expected_hash:
            raise DeltaCodecError("committed record hash mismatch")
    record = Publication.from_bytes(data, store.storage_limits)
    if record.key != key:
        raise DeltaCodecError("committed record key mismatch")
    return record


def _check_closure(
    store: StorageReader,
    records: tuple[Publication, ...],
    requested: tuple[str, ...],
    authority_id: str,
    stream_id: str,
    run_epoch: str,
) -> None:
    by_id: dict[str, Publication] = {}
    identities = {}
    depths: dict[str, int] = {}
    for record in records:
        if record.manifest_id in by_id:
            raise DeltaCodecError("duplicate archive record")
        if (record.authority_id, record.target.stream_id, record.target.run_epoch) != (
            authority_id,
            stream_id,
            run_epoch,
        ):
            raise DeltaCodecError("archive authority/stream/epoch mismatch")
        previous = identities.setdefault(record.target.version, record.target)
        if previous != record.target:
            raise DeltaCodecError("conflicting roots for one version")
        manifest = record.load_manifest(store)
        if record.base_manifest_id is not None:
            base = by_id.get(record.base_manifest_id)
            if base is None or base.target != manifest.base:
                raise DeltaCodecError("missing, reordered, or incompatible archive dependency")
            depth = depths[base.manifest_id] + 1
        else:
            depth = 0
        require_uint(depth, "archive chain depth", store.storage_limits.max_chain_depth)
        validate_artifacts(store, manifest)
        by_id[record.manifest_id] = record
        depths[record.manifest_id] = depth
    if not requested or len(set(requested)) != len(requested) or any(mid not in by_id for mid in requested):
        raise DeltaCodecError("archive requested set is empty, duplicate, or incomplete")
    # No extraneous records: every retained record must be requested or needed.
    needed = set()
    for manifest_id in requested:
        current = manifest_id
        while current is not None and current not in needed:
            needed.add(current)
            current = by_id[current].base_manifest_id
    if needed != set(by_id):
        raise DeltaCodecError("archive contains unrelated records")


class OfflineCatalog:
    """Trust the publisher ACL on this root; no online trainer/authority
    needed.

    Discovery is a bounded view of exported records, not a freshness oracle. A
    requested version must be explicitly present; absent versions are errors.
    """

    def __init__(self, store: StorageReader, *, stream_id: str, run_epoch: str) -> None:
        self.store = store
        require_identifier(stream_id, "stream_id")
        require_identifier(run_epoch, "run_epoch")
        capabilities = store.capabilities()
        if capabilities.stream_id is not None and (capabilities.stream_id, capabilities.run_epoch) != (
            stream_id,
            run_epoch,
        ):
            raise DeltaCodecError("archive differs from deployment namespace")
        value = exact_fields(
            parse_json(
                store.read_object("authority.json", store.storage_limits.max_record_bytes),
                store.storage_limits.max_record_bytes,
            ),
            {"format_version", "authority_id", "stream_id", "run_epoch"},
        )
        if type(value["format_version"]) is not int or value["format_version"] != 1:
            raise DeltaCodecError("unsupported authority binding")
        require_identifier(value["authority_id"], "authority_id")
        if (value["stream_id"], value["run_epoch"]) != (stream_id, run_epoch):
            raise DeltaCodecError("archive epoch is not authorized")
        self.authority_id, self.stream_id, self.run_epoch = value["authority_id"], stream_id, run_epoch

    def records(self) -> tuple[Publication, ...]:
        result = []
        identities = {}
        for key in self.store.catalog_keys():
            record = _load_record(self.store, key)
            if (record.authority_id, record.target.stream_id, record.target.run_epoch) != (
                self.authority_id,
                self.stream_id,
                self.run_epoch,
            ):
                raise DeltaCodecError("catalog authority/stream/epoch mismatch")
            previous = identities.setdefault(record.target.version, record.target)
            if previous != record.target:
                raise DeltaCodecError("catalog contains conflicting version roots")
            result.append(record)
        return tuple(result)

    def seal_archive(
        self, versions: tuple[int, ...], *, prefer_full: bool = True, writer: StorageWriter | None = None
    ) -> str:
        """Persist the exact requested set plus transitive anchor
        dependencies."""
        if writer is None:
            if not isinstance(self.store, StorageWriter) or self.store.capabilities().access != "publisher":
                raise DeltaCodecError("archive sealing requires an explicit publisher writer")
            writer = self.store
        if writer.capabilities().access != "publisher":
            raise DeltaCodecError("archive sealing requires a publisher writer")
        if not isinstance(versions, tuple) or not versions or len(versions) > self.store.storage_limits.max_records:
            raise DeltaCodecError("invalid archive version count")
        for version in versions:
            require_uint(version, "archive version")
        if tuple(sorted(set(versions))) != versions:
            raise DeltaCodecError("archive versions must be sorted and unique")
        records = self.records()
        by_id = {record.manifest_id: record for record in records}
        requested = []
        included = {}
        for version in versions:
            choices = [record for record in records if record.target.version == version]
            preferred = [record for record in choices if record.kind == "FULL"] if prefer_full else []
            preferred = preferred or [record for record in choices if record.advances_head]
            if len(preferred) != 1:
                raise DeltaCodecError("requested version is missing or ambiguous")
            selected = preferred[0]
            requested.append(selected.manifest_id)
            depth = 0
            while selected.manifest_id not in included:
                included[selected.manifest_id] = selected
                if selected.base_manifest_id is None:
                    break
                depth += 1
                require_uint(depth, "archive dependency depth", self.store.storage_limits.max_chain_depth)
                if selected.base_manifest_id not in by_id:
                    raise DeltaCodecError("archive dependency record is missing")
                selected = by_id[selected.base_manifest_id]
        ordered = tuple(sorted(included.values(), key=lambda record: (record.target.version, record.manifest_id)))
        _check_closure(self.store, ordered, tuple(requested), self.authority_id, self.stream_id, self.run_epoch)
        value = {
            "format_version": 1,
            "authority_id": self.authority_id,
            "stream_id": self.stream_id,
            "run_epoch": self.run_epoch,
            "requested": requested,
            "records": [
                {"key": record.key, "record_hash": content_hash(record.to_bytes(self.store.storage_limits))}
                for record in ordered
            ],
        }
        data = canonical_json(value, self.store.storage_limits.max_archive_bytes)
        archive_id = content_hash(data)
        if writer is not self.store:
            if writer.capabilities().namespace_id != self.store.capabilities().namespace_id:
                raise DeltaCodecError("archive writer belongs to a different namespace")
            source_binding = self.store.read_object("authority.json", self.store.storage_limits.max_record_bytes)
            if writer.read_object("authority.json", writer.storage_limits.max_record_bytes) != source_binding:
                raise DeltaCodecError("archive writer belongs to a different authority")
            for record in ordered:
                _load_record(writer, record.key, content_hash(record.to_bytes(self.store.storage_limits)))
            _check_closure(writer, ordered, tuple(requested), self.authority_id, self.stream_id, self.run_epoch)
        writer.put_immutable(f"archives/{archive_id}.json", data, expected_hash=archive_id)
        return archive_id


@dataclass(frozen=True)
class OfflineArchive:
    records: tuple[Publication, ...]
    requested: tuple[str, ...]

    @classmethod
    def open(cls, store: StorageReader, *, expected_archive_id: str) -> "OfflineArchive":
        require_digest(expected_archive_id, "archive_id")
        data = store.read_object(f"archives/{expected_archive_id}.json", store.storage_limits.max_archive_bytes)
        if content_hash(data) != expected_archive_id:
            raise DeltaCodecError("archive hash differs from trusted identity")
        value = exact_fields(
            parse_json(data, store.storage_limits.max_archive_bytes),
            {"format_version", "authority_id", "stream_id", "run_epoch", "requested", "records"},
        )
        if type(value["format_version"]) is not int or value["format_version"] != 1:
            raise DeltaCodecError("unsupported archive version")
        for name in ("authority_id", "stream_id", "run_epoch"):
            require_identifier(value[name], name)
        for name in ("requested", "records"):
            if not isinstance(value[name], list) or not 0 < len(value[name]) <= store.storage_limits.max_records:
                raise DeltaCodecError("invalid archive count")
        for manifest_id in value["requested"]:
            require_digest(manifest_id, "requested manifest")
        records = []
        for ref in value["records"]:
            exact_fields(ref, {"key", "record_hash"})
            records.append(_load_record(store, ref["key"], ref["record_hash"]))
        result = cls(tuple(records), tuple(value["requested"]))
        _check_closure(
            store, result.records, result.requested, value["authority_id"], value["stream_id"], value["run_epoch"]
        )
        return result

    def manifests(self, store: StorageReader) -> Iterator[tuple[Publication, Manifest]]:
        for record in self.records:
            yield record, record.load_manifest(store)

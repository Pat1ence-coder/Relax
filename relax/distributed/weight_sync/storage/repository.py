# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Shared artifact validation; no filesystem or publication authority IO."""

from ..codec import EncodedChunk
from ..codec.format import content_hash
from ..integrity import ModelRoot
from ..limits import DeltaCodecError
from ..manifest import IndexPage, Manifest
from ..schema import require_digest
from .contracts import StorageReader, StorageWriter


def read_manifest(store: StorageReader, manifest_id: str) -> Manifest:
    require_digest(manifest_id, "manifest_id")
    data = store.read_object(f"manifests/{manifest_id}.json", store.limits.max_manifest_bytes)
    return Manifest.from_bytes(data, expected_manifest_id=manifest_id, limits=store.limits)


def validate_artifacts(store: StorageReader, manifest: Manifest) -> None:
    """Verify descriptor coverage and stored hashes, not decoded model
    bytes."""
    manifest.validate(store.limits)
    specs = iter(manifest.schema.iter_chunks())
    root = ModelRoot(manifest.schema.schema_id, manifest.schema.directory_hash)
    for ref in manifest.pages:
        page = IndexPage.from_bytes(store.read_index(ref), ref, store.limits)
        for record in page.records:
            descriptor = record.descriptor
            if descriptor.spec != next(specs, None) or descriptor.target_version != manifest.target.version:
                raise DeltaCodecError("artifact chunk coverage or version mismatch")
            if descriptor.base_version is not None and (
                manifest.base is None or descriptor.base_version != manifest.base.version
            ):
                raise DeltaCodecError("artifact base version mismatch")
            payload = store.read_payload(record)
            EncodedChunk(descriptor, payload).validate(store.limits.codec)
            if descriptor.base_version is None and content_hash(payload) != descriptor.target_hash:
                raise DeltaCodecError("RAW target hash mismatch")
            root.add(descriptor.spec, descriptor.target_hash)
    if next(specs, None) is not None or root.hexdigest() != manifest.target.target_root:
        raise DeltaCodecError("artifact directory root mismatch")


def write_manifest(store: StorageWriter, manifest: Manifest) -> str:
    validate_artifacts(store, manifest)
    data = manifest.to_bytes(store.limits)
    manifest_id = content_hash(data)
    store.put_immutable(f"manifests/{manifest_id}.json", data, expected_hash=manifest_id)
    return manifest_id

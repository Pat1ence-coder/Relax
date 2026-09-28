# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Storage contracts independent of mount paths and filesystem primitives."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

from ..limits import DeltaCodecError, SnapshotLimits, require_uint
from ..snapshot import ArtifactReader, ArtifactWriter


@dataclass(frozen=True)
class StorageLimits:
    max_bytes: int = 1 << 40
    max_objects: int = 1 << 20
    max_records: int = 4096
    max_record_bytes: int = 16 * 1024
    max_archive_bytes: int = 1024 * 1024
    max_chain_depth: int = 128
    max_database_bytes: int = 64 * 1024 * 1024

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            require_uint(getattr(self, name), name)
            if getattr(self, name) == 0:
                raise DeltaCodecError("storage limits must be positive")


@dataclass(frozen=True)
class StoreCapabilities:
    """Implementation requirements, never a compatibility test result."""

    backend: str
    profile: str
    access: Literal["reader", "publisher"]
    namespace_id: str | None
    requirements: tuple[str, ...]
    deployment_checked: bool = False
    stream_id: str | None = None
    run_epoch: str | None = None


@dataclass(frozen=True)
class ObjectReceipt:
    """Object persistence only; neither publication nor consumer COMMIT."""

    key: str
    content_hash: str
    byte_length: int
    namespace_id: str | None
    profile: str
    fault_domain: str | None


class ControlLocationGuard(Protocol):
    """Adapter-approved placement for the separate local WAL authority."""

    def validate_opened(self, descriptor: int) -> None:
        """Bind the opened local control directory to the approved location."""
        ...


@runtime_checkable
class StorageReader(ArtifactReader, Protocol):
    limits: SnapshotLimits
    storage_limits: StorageLimits

    def capabilities(self) -> StoreCapabilities: ...

    def read_object(self, key: str, maximum: int, *, length: int | None = None) -> bytes: ...

    def catalog_keys(self) -> list[str]:
        """Bound exported records; an empty result is not a freshness proof."""
        ...


@runtime_checkable
class StorageWriter(StorageReader, ArtifactWriter, Protocol):
    def put_immutable(self, key: str, data: bytes, *, expected_hash: str) -> ObjectReceipt: ...

    def prepare_control(self, location: Path) -> ControlLocationGuard:
        """Reject overlapping or unapproved control placement before writes."""
        ...

    def begin_recovery(self) -> None:
        """Block ordinary uploads until all committed records are exported."""
        ...

    def finish_recovery(self) -> None: ...

    def reserve_record(self, key: str, data: bytes) -> None:
        """Reserve exact record bytes/hash, including its operation identity.

        UNKNOWN retains capacity. Reopening must recover exports before new
        uploads. An already durable identical record needs no reservation.
        """
        ...

    def release_record(self, key: str) -> None:
        """Release only after successful export or known authority rollback."""
        ...

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Explicit POSIX storage adapters; importing the codec does not import
these."""

from .artifacts import PosixArtifactStore
from .capture import DiskCapture, FrozenExport
from .catalog import ProducerCatalog, Publication, PublicationUncertain
from .contracts import ObjectReceipt, StorageLimits, StorageReader, StorageWriter, StoreCapabilities
from .deployment import (
    DeploymentCheck,
    DeploymentReport,
    NamespaceBinding,
    PosixAccessPolicy,
    PosixDeployment,
    VolumeSpec,
    initialize_namespace,
    inspect_deployment,
    open_deployed_store,
)
from .offline import OfflineArchive, OfflineCatalog
from .placement import MountRecord, StorageDeploymentError, parse_mountinfo
from .staging import DiskSnapshotReader, DiskSnapshotStore, DiskStaging, open_snapshot


__all__ = [
    "DeploymentCheck",
    "DeploymentReport",
    "DiskCapture",
    "DiskSnapshotReader",
    "DiskSnapshotStore",
    "DiskStaging",
    "FrozenExport",
    "MountRecord",
    "NamespaceBinding",
    "ObjectReceipt",
    "OfflineArchive",
    "OfflineCatalog",
    "PosixAccessPolicy",
    "PosixArtifactStore",
    "PosixDeployment",
    "ProducerCatalog",
    "Publication",
    "PublicationUncertain",
    "StorageDeploymentError",
    "StorageLimits",
    "StorageReader",
    "StorageWriter",
    "StoreCapabilities",
    "VolumeSpec",
    "initialize_namespace",
    "inspect_deployment",
    "open_deployed_store",
    "open_snapshot",
    "parse_mountinfo",
]

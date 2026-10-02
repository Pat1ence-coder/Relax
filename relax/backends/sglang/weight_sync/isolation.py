# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Linux process identities and pidfd-based isolation of a retained engine
group."""

import os
import select
import signal
from pathlib import Path
from typing import Any

from relax.distributed.weight_sync import DeltaCodecError
from relax.distributed.weight_sync.codec.format import content_hash
from relax.distributed.weight_sync.serialization import canonical_json


def process_identity(pid: int) -> dict[str, Any]:
    if type(pid) is not int or pid <= 0:
        raise DeltaCodecError("invalid engine process identity")
    if int(Path("/proc/self/stat").read_text().split(" ", 1)[0]) != os.getpid():
        raise DeltaCodecError("procfs does not describe the current PID namespace")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    # comm may contain spaces and parentheses; the final ')' ends field 2.
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "start_ticks": int(fields[19]),
        "boot_id": boot,
        "pid_namespace": os.stat(f"/proc/{pid}/ns/pid").st_ino,
    }


def isolate_processes(identities: list[dict[str, Any]], *, timeout: float = 5) -> tuple[dict, ...]:
    """Terminate only persisted identities, using pidfds to prevent PID reuse
    races.

    Call on a dedicated worker thread if the caller owns an asyncio loop. A
    missing/reused PID in the same boot and PID namespace proves the old
    process no longer exists. Another host/boot/namespace requires external
    isolation evidence; it never authorizes signalling an unrelated process.
    """
    if not identities or len(identities) > 4096 or len({item["pid"] for item in identities}) != len(identities):
        raise DeltaCodecError("incomplete or duplicate engine process binding")
    current_boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    namespace = os.stat("/proc/self/ns/pid").st_ino
    descriptors, evidence = [], []
    try:
        for identity in identities:
            if (
                set(identity) != {"pid", "start_ticks", "boot_id", "pid_namespace"}
                or type(identity["pid"]) is not int
                or identity["pid"] <= 0
                or type(identity["start_ticks"]) is not int
                or identity["start_ticks"] <= 0
                or not isinstance(identity["boot_id"], str)
                or not identity["boot_id"]
            ):
                raise DeltaCodecError("invalid retained process binding")
            pid = identity["pid"]
            if pid == os.getpid():
                raise DeltaCodecError("an engine binding cannot include its recovery coordinator")
            if identity["boot_id"] != current_boot or identity["pid_namespace"] != namespace:
                raise DeltaCodecError("automatic isolation requires the original boot and PID namespace")
            try:
                descriptor = os.pidfd_open(pid)
            except ProcessLookupError:
                evidence.append({"identity": identity, "outcome": "process_absent"})
                continue
            try:
                try:
                    current = process_identity(pid)
                except FileNotFoundError:
                    current = None
                if current != identity:
                    evidence.append({"identity": identity, "outcome": "original_process_absent"})
                    os.close(descriptor)
                    continue
                descriptors.append((identity, descriptor))
            except BaseException:
                os.close(descriptor)
                raise
        for _, descriptor in descriptors:
            try:
                signal.pidfd_send_signal(descriptor, signal.SIGTERM)
            except ProcessLookupError:
                pass
        for identity, descriptor in descriptors:
            if not select.select([descriptor], [], [], timeout)[0]:
                try:
                    signal.pidfd_send_signal(descriptor, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                if not select.select([descriptor], [], [], timeout)[0]:
                    raise DeltaCodecError("old engine process could not be physically isolated")
            evidence.append({"identity": identity, "outcome": "pidfd_exit_confirmed"})
        return tuple(evidence)
    finally:
        for _, descriptor in descriptors:
            os.close(descriptor)


def isolation_digest(execution_id: str, evidence: tuple[dict, ...]) -> str:
    return content_hash(canonical_json({"execution_id": execution_id, "processes": evidence}, 1024 * 1024))


def arm_parent_death(expected_parent: dict[str, Any]) -> None:
    """Close the spawn-to-registration gap before child model
    initialization."""
    import ctypes

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot arm engine parent-death isolation")
    try:
        parent = process_identity(os.getppid())
    except FileNotFoundError:
        parent = None
    if parent != expected_parent:
        # No GPU/storage work has begun. Never adopt an unrecorded orphan.
        raise DeltaCodecError("engine coordinator exited during child spawn")

# Copyright (c) 2026 Relax Authors. All Rights Reserved.
"""Real process isolation, PID identity refusal and parent-death windows."""

import multiprocessing
import os
import select
import signal
import time

import pytest

from relax.backends.sglang.weight_sync.isolation import arm_parent_death, isolate_processes, process_identity
from relax.distributed.weight_sync import DeltaCodecError


@pytest.fixture(autouse=True)
def require_matching_procfs():
    try:
        process_identity(os.getpid())
    except (DeltaCodecError, FileNotFoundError):
        pytest.skip("sandbox procfs does not expose the current PID namespace")


def _wait():
    signal.pause()


def _child(parent, ready, delay):
    time.sleep(delay)
    arm_parent_death(parent)
    ready.send("armed")
    signal.pause()


def _owner(connection, delay):
    child_ready, child_sender = multiprocessing.Pipe(duplex=False)
    process = multiprocessing.get_context("spawn").Process(
        target=_child, args=(process_identity(os.getpid()), child_sender, delay)
    )
    process.start()
    connection.send(process_identity(process.pid))
    if delay == 0:
        assert child_ready.recv() == "armed"
        connection.send("armed")
    signal.pause()


@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux pidfd isolation requires kernel support")
def test_pidfd_stops_only_retained_process_and_refuses_namespace():
    context = multiprocessing.get_context("spawn")
    owned, unrelated = context.Process(target=_wait), context.Process(target=_wait)
    owned.start()
    unrelated.start()
    try:
        identity = process_identity(owned.pid)
        with pytest.raises(DeltaCodecError, match="namespace"):
            isolate_processes([dict(identity, pid_namespace=identity["pid_namespace"] + 1)])
        assert owned.is_alive() and unrelated.is_alive()
        # A reused numeric PID cannot authorize signalling its replacement.
        proof = isolate_processes([dict(identity, start_ticks=identity["start_ticks"] + 1)])
        assert proof[0]["outcome"] == "original_process_absent" and owned.is_alive()
        proof = isolate_processes([identity])
        owned.join(5)
        assert proof[0]["outcome"] == "pidfd_exit_confirmed" and owned.exitcode is not None
        assert unrelated.is_alive()
    finally:
        for process in (owned, unrelated):
            if process.is_alive():
                process.kill()
            process.join(5)


@pytest.mark.parametrize("delay", [0, 0.5])
@pytest.mark.skipif(not hasattr(os, "pidfd_open"), reason="Linux parent-death isolation requires pidfd support")
def test_coordinator_death_before_or_after_child_arms_does_not_orphan_worker(delay):
    receive, send = multiprocessing.Pipe(duplex=False)
    owner = multiprocessing.get_context("spawn").Process(target=_owner, args=(send, delay))
    owner.start()
    descriptor = None
    try:
        assert receive.poll(15)
        identity = receive.recv()
        descriptor = os.pidfd_open(identity["pid"])
        if delay == 0:
            assert receive.poll(15) and receive.recv() == "armed"
        owner.kill()
        owner.join(5)
        # Do not signal the child: the child wrapper/kernel must isolate it.
        assert select.select([descriptor], [], [], 15)[0]
    finally:
        if owner.is_alive():
            owner.kill()
        owner.join(5)
        if descriptor is not None:
            if not select.select([descriptor], [], [], 0)[0]:
                signal.pidfd_send_signal(descriptor, signal.SIGKILL)
            os.close(descriptor)

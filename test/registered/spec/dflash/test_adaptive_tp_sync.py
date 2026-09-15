"""Unit-test DFLASH adaptive block-size decision synchronization over TP."""

from types import SimpleNamespace

import sglang.srt.speculative.dflash_worker_v2 as dflash_worker
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2


class FakeParams:
    def __init__(self):
        self.current_b = 8
        self.installed = []

    def on_verify_complete(self, *args, **kwargs):
        return self.decision

    def set_block_size_for_batch(self, batch_size, block_size):
        self.installed.append((batch_size, block_size))
        self.current_b = block_size


class FakeController:
    def __init__(self):
        self.activated = []

    def activate_step(self, block_size):
        self.activated.append(block_size)


class FakeTensor:
    def __init__(self, value=0):
        self.value = value

    def fill_(self, value):
        self.value = value

    def copy_(self, other):
        self.value = other.value

    def item(self):
        return self.value


class FakeTPGroup:
    def __init__(self, world_size):
        self.world_size = world_size
        self.root_tensor = None
        self.broadcast_src = None

    def broadcast(self, tensor, src):
        self.broadcast_src = src
        tensor.copy_(self.root_tensor)
        return tensor


def _run_tp_group(world_size, decision, monkeypatch):
    group = FakeTPGroup(world_size)
    states = []
    original_get_tp_group = dflash_worker.get_tp_group
    monkeypatch.setattr(dflash_worker, "get_tp_group", lambda: group)

    for tp_rank in range(world_size):
        params = FakeParams()
        controller = FakeController()
        worker = object.__new__(DFlashWorkerV2)
        worker.ps = SimpleNamespace(tp_rank=tp_rank)
        worker._mab_controller = params
        worker._adaptive_controller = controller
        worker._adaptive_tp_decision_buf = FakeTensor() if world_size > 1 else None
        if tp_rank == 0:
            params.decision = decision
            group.root_tensor = worker._adaptive_tp_decision_buf

        worker.on_verify_complete_cpu(
            [1, 2, 3], batch_size=27, step_time_ms=12.5
        )
        states.append((params, controller))

    monkeypatch.setattr(dflash_worker, "get_tp_group", original_get_tp_group)
    return states, group


def test_rank0_decision_is_activated_on_all_tp_ranks(monkeypatch):
    for world_size in (1, 2, 4):
        states, group = _run_tp_group(world_size, 5, monkeypatch)

        assert all(params.current_b == 5 for params, _ in states)
        assert all(controller.activated == [5] for _, controller in states)
        if world_size > 1:
            assert group.broadcast_src == 0


def test_no_change_decision_is_a_synchronized_no_op(monkeypatch):
    states, group = _run_tp_group(2, None, monkeypatch)

    assert all(params.installed == [] for params, _ in states)
    assert all(controller.activated == [] for _, controller in states)
    assert group.broadcast_src == 0

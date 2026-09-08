# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
)
from vllm.v1.worker.gpu.dp_utils import (
    DPSyncCoordinator,
    DPSyncState,
    dispatch_cg_and_sync_dp,
)

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


class _Work:
    def __init__(
        self,
        tensor: torch.Tensor,
        remote_tokens: int,
        remote_reqs: int = 2,
        generation_delta: int = 0,
        remote_need_eager: bool = False,
        remote_uniform_token_count: int = 1,
        remote_max_query_len: int = 1,
    ) -> None:
        self.tensor = tensor
        self.remote_tokens = remote_tokens
        self.remote_reqs = remote_reqs
        self.generation_delta = generation_delta
        self.remote_need_eager = remote_need_eager
        self.remote_uniform_token_count = remote_uniform_token_count
        self.remote_max_query_len = remote_max_query_len
        self.wait_calls = 0

    def wait(self) -> None:
        self.wait_calls += 1
        self.tensor[0, 1] = self.remote_tokens
        self.tensor[1, 1] = CUDAGraphMode.FULL.value
        self.tensor[2, 1] = self.remote_uniform_token_count
        self.tensor[3, 1] = self.remote_max_query_len
        self.tensor[4, 1] = self.remote_reqs
        self.tensor[5, 1] = self.tensor[5, 0] + self.generation_delta
        self.tensor[6, 1] = int(self.remote_need_eager)


def _graph_manager() -> Mock:
    manager = Mock(spec=CudaGraphManager)
    manager.dispatch.side_effect = lambda num_reqs, num_tokens, uniform, **kwargs: (
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            uniform_token_count=uniform,
            max_query_len=kwargs.get("max_query_len"),
            num_active_loras=kwargs.get("num_active_loras", 0),
        )
    )
    return manager


def test_async_dp_sync_waits_on_result_and_reuses_buffer(monkeypatch):
    tensors = []
    works = []

    def all_reduce(tensor, group, async_op):
        assert async_op
        tensors.append(tensor)
        work = _Work(tensor, remote_tokens=4)
        works.append(work)
        return work

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    coordinator = DPSyncCoordinator(2, 0, group=Mock())
    manager = _graph_manager()

    future = coordinator.start(manager, 2, 2, uniform_token_count=1)

    assert works[0].wait_calls == 0
    with pytest.raises(RuntimeError, match="already in flight"):
        coordinator.start(manager, 2, 2, uniform_token_count=1)

    batch_desc, sync = future.result(manager)

    assert works[0].wait_calls == 1
    assert batch_desc.num_tokens == 4
    assert sync is not None
    assert sync.num_tokens_across_dp.tolist() == [4, 4]
    assert sync.generation == 0
    assert sync.live_num_tokens_across_dp == (2, 4)
    assert sync.live_num_reqs_across_dp == (2, 2)
    assert future.result(manager) == (batch_desc, sync)
    assert works[0].wait_calls == 1

    future.release()
    future.release()
    next_future = coordinator.start(manager, 2, 2, uniform_token_count=1)
    assert tensors[1] is tensors[0]
    next_future.release()


def test_async_dp_sync_release_waits_for_unconsumed_work(monkeypatch):
    works = []

    def all_reduce(tensor, group, async_op):
        work = _Work(tensor, remote_tokens=2)
        works.append(work)
        return work

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    coordinator = DPSyncCoordinator(2, 0, group=Mock())
    manager = _graph_manager()

    future = coordinator.start(manager, 2, 2, uniform_token_count=1)
    future.release()

    assert works[0].wait_calls == 1
    replacement = coordinator.start(manager, 2, 2, uniform_token_count=1)
    replacement.release()


def test_async_dp_sync_does_not_wait_twice_after_resolution_error(monkeypatch):
    work = None

    def all_reduce(tensor, group, async_op):
        nonlocal work
        work = _Work(tensor, remote_tokens=2)
        return work

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    coordinator = DPSyncCoordinator(2, 0, group=Mock())
    manager = _graph_manager()
    manager.dispatch.side_effect = [
        BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=2,
            num_reqs=2,
            uniform_token_count=1,
        ),
        RuntimeError("dispatch failed"),
    ]

    future = coordinator.start(manager, 2, 2, uniform_token_count=1)
    with pytest.raises(RuntimeError, match="dispatch failed"):
        future.result(manager)
    assert work is not None
    assert work.wait_calls == 1

    future.release()
    assert work.wait_calls == 1


def test_execution_contract_ignores_inactive_rank_graph_mode(monkeypatch):
    def all_reduce(tensor, group, async_op):
        return _Work(tensor, remote_tokens=3, remote_reqs=2)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    manager = Mock(spec=CudaGraphManager)

    def dispatch(num_reqs, num_tokens, *args, **kwargs):
        if num_tokens == 0:
            return BatchExecutionDescriptor(
                cg_mode=CUDAGraphMode.NONE,
                num_tokens=0,
                num_reqs=0,
            )
        return BatchExecutionDescriptor(
            cg_mode=CUDAGraphMode.FULL,
            num_tokens=4,
            num_reqs=4,
            uniform_token_count=1,
            max_query_len=1,
        )

    manager.dispatch.side_effect = dispatch
    coordinator = DPSyncCoordinator(2, 0, group=Mock())

    future = coordinator.start(manager, 0, 0, uniform_token_count=None)
    batch_desc, sync = future.result(manager)

    assert batch_desc.cg_mode == CUDAGraphMode.FULL
    assert batch_desc.num_tokens == 4
    assert batch_desc.num_reqs == 4
    assert sync is not None
    assert sync.num_tokens_across_dp.tolist() == [4, 4]
    assert sync.live_num_tokens_across_dp == (0, 3)
    assert sync.live_num_reqs_across_dp == (0, 2)
    future.release()


def test_async_dp_sync_rejects_generation_mismatch(monkeypatch):
    work = None

    def all_reduce(tensor, group, async_op):
        nonlocal work
        work = _Work(tensor, remote_tokens=2, generation_delta=1)
        return work

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    coordinator = DPSyncCoordinator(2, 0, group=Mock())
    manager = _graph_manager()

    future = coordinator.start(manager, 2, 2, uniform_token_count=1)
    with pytest.raises(RuntimeError, match="generation mismatch"):
        future.result(manager)

    future.release()
    assert work is not None
    assert work.wait_calls == 1


def test_execution_contract_selects_global_eager_fallback(monkeypatch):
    def all_reduce(tensor, group, async_op):
        return _Work(
            tensor,
            remote_tokens=3,
            remote_reqs=2,
            remote_need_eager=True,
        )

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    coordinator = DPSyncCoordinator(2, 0, group=Mock())

    future = coordinator.start(_graph_manager(), 0, 0, uniform_token_count=None)
    batch_desc, sync = future.result(_graph_manager())

    assert batch_desc.cg_mode == CUDAGraphMode.NONE
    assert batch_desc.num_tokens == 3
    assert batch_desc.num_reqs == 2
    assert sync is not None and sync.eager
    future.release()


def test_execution_contract_returns_empty_when_all_ranks_are_idle(monkeypatch):
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, group, async_op: _Work(tensor, remote_tokens=0, remote_reqs=0),
    )
    coordinator = DPSyncCoordinator(2, 0, group=Mock())

    future = coordinator.start(_graph_manager(), 0, 0, uniform_token_count=None)
    batch_desc, sync = future.result(_graph_manager())

    assert batch_desc.num_tokens == 0
    assert batch_desc.num_reqs == 0
    assert sync is None
    future.release()


def test_reused_execution_contract_selects_global_request_capacity():
    manager = _graph_manager()
    sync = DPSyncState(
        num_tokens_across_dp=torch.tensor([4, 4]),
        uniform_token_count=1,
        eager=False,
        num_reqs=4,
        execution_num_reqs=4,
    )

    batch_desc, reused = dispatch_cg_and_sync_dp(
        manager,
        num_reqs=1,
        num_tokens=4,
        uniform_token_count=1,
        dp_size=2,
        dp_rank=0,
        dp_sync=sync,
    )

    assert batch_desc.num_reqs == 4
    assert reused is sync
    assert manager.dispatch.call_args.args[:2] == (4, 4)


def test_cached_contract_activates_then_skips_collective(monkeypatch):
    collective_calls = 0

    def all_reduce(tensor, group, async_op):
        nonlocal collective_calls
        collective_calls += 1
        return _Work(tensor, remote_tokens=2, remote_reqs=2)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    manager = _graph_manager()
    coordinator = DPSyncCoordinator(
        2,
        0,
        group=Mock(),
        cache_execution_contract=True,
        cache_stability_steps=2,
    )

    for force_refresh in (True, False):
        future = coordinator.start(
            manager,
            2,
            2,
            uniform_token_count=1,
            max_query_len=1,
            force_refresh=force_refresh,
            contract_epoch=10,
            contract_capacity_num_reqs=4,
        )
        future.result(manager)
        future.release()

    cached = coordinator.start(
        manager,
        1,
        1,
        uniform_token_count=1,
        max_query_len=1,
        contract_epoch=10,
        contract_capacity_num_reqs=4,
    )
    batch_desc, sync = cached.result(manager)

    assert collective_calls == 2
    assert batch_desc.num_tokens == 4
    assert batch_desc.num_reqs == 4
    assert sync is not None
    assert sync.num_reqs == 4
    assert sync.contract_epoch == 10
    assert not sync.live_facts_exact
    cached.release()


def test_cached_contract_idle_rank_reuses_capacity(monkeypatch):
    monkeypatch.setattr(
        torch.distributed,
        "all_reduce",
        lambda tensor, group, async_op: _Work(tensor, remote_tokens=2, remote_reqs=2),
    )
    manager = _graph_manager()
    coordinator = DPSyncCoordinator(
        2,
        0,
        group=Mock(),
        cache_execution_contract=True,
        cache_stability_steps=1,
    )
    refresh = coordinator.start(
        manager,
        2,
        2,
        uniform_token_count=1,
        max_query_len=1,
        force_refresh=True,
        contract_epoch=4,
        contract_capacity_num_reqs=4,
    )
    refresh.result(manager)
    refresh.release()

    idle = coordinator.start(
        manager,
        0,
        0,
        uniform_token_count=None,
        max_query_len=0,
        contract_epoch=4,
        contract_capacity_num_reqs=4,
    )
    batch_desc, sync = idle.result(manager)

    assert batch_desc.num_tokens == 4
    assert sync is not None and not sync.live_facts_exact
    idle.release()


def test_cached_contract_rejects_local_drift_without_collective(monkeypatch):
    collective_calls = 0

    def all_reduce(tensor, group, async_op):
        nonlocal collective_calls
        collective_calls += 1
        return _Work(tensor, remote_tokens=2, remote_reqs=2)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    manager = _graph_manager()
    coordinator = DPSyncCoordinator(
        2,
        0,
        group=Mock(),
        cache_execution_contract=True,
        cache_stability_steps=1,
    )
    refresh = coordinator.start(
        manager,
        2,
        2,
        uniform_token_count=1,
        max_query_len=1,
        force_refresh=True,
        contract_epoch=8,
        contract_capacity_num_reqs=4,
    )
    refresh.result(manager)
    refresh.release()

    with pytest.raises(RuntimeError, match="local fallback would change"):
        coordinator.start(
            manager,
            1,
            2,
            uniform_token_count=None,
            max_query_len=2,
            contract_epoch=8,
            contract_capacity_num_reqs=4,
            has_prefill=True,
        )
    with pytest.raises(RuntimeError, match="epoch=9, cached=8"):
        coordinator.start(
            manager,
            1,
            1,
            uniform_token_count=1,
            max_query_len=1,
            contract_epoch=9,
            contract_capacity_num_reqs=4,
        )

    assert collective_calls == 1
    assert coordinator._active_future is None


def test_prefill_refresh_invalidates_cached_contract(monkeypatch):
    collective_calls = 0

    def all_reduce(tensor, group, async_op):
        nonlocal collective_calls
        collective_calls += 1
        return _Work(tensor, remote_tokens=2, remote_reqs=2)

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    manager = _graph_manager()
    coordinator = DPSyncCoordinator(
        2,
        0,
        group=Mock(),
        cache_execution_contract=True,
        cache_stability_steps=1,
    )

    refresh = coordinator.start(
        manager,
        2,
        2,
        uniform_token_count=1,
        max_query_len=1,
        force_refresh=True,
        contract_epoch=3,
        contract_capacity_num_reqs=4,
    )
    refresh.result(manager)
    refresh.release()
    assert coordinator._cached_contract is not None

    prefill_refresh = coordinator.start(
        manager,
        1,
        2,
        uniform_token_count=2,
        max_query_len=2,
        force_refresh=True,
        contract_epoch=4,
        contract_capacity_num_reqs=4,
        has_prefill=True,
    )
    prefill_refresh.result(manager)
    prefill_refresh.release()
    assert coordinator._cached_contract is None

    decode = coordinator.start(
        manager,
        2,
        2,
        uniform_token_count=1,
        max_query_len=1,
        contract_epoch=4,
        contract_capacity_num_reqs=4,
    )
    decode.result(manager)
    decode.release()
    assert collective_calls == 3

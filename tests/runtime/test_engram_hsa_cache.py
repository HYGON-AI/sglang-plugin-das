# SPDX-License-Identifier: Apache-2.0
import multiprocessing
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


class SingleRankGroup:
    rank_in_group = 0
    world_size = 1

    def all_gather_object(self, value):
        return [value]

    def barrier(self):
        pass


def _embedding(rank=0):
    from hcu_sglang.environ import register_envs
    from hcu_sglang.layers.engram_host import HcuEngramEmbedding
    register_envs()
    parallel = SimpleNamespace(tp_size=2, tp_rank=rank, tp_group=SingleRankGroup())
    with patch("hcu_sglang.layers.engram_host.get_parallel", return_value=parallel):
        return HcuEngramEmbedding(12, 128, 0)


def _check_client(pipe, desc):
    try:
        with torch.device("meta"):
            client = _embedding(rank=1)
        assert client.weight.is_meta and client.host_table is None
        client.attach_ipc_host_state(desc)
        indices = torch.tensor([0, 11, 12, -1], device="cuda")
        out = client(indices).cpu()
        torch.testing.assert_close(out[:,0], torch.tensor([0, 22, 0, 0], dtype=out.dtype))
        with pytest.raises(RuntimeError, match="read-only"):
            client.weight.weight_loader(client.weight, client.weight)
        client.host_table.allocation.close()
        pipe.send(None)
    except BaseException:
        import traceback
        pipe.send(traceback.format_exc())
    finally:
        pipe.close()


def test_hsa_table_across_processes(monkeypatch):
    from hcu_sglang.utils.detection import is_hcu
    if not is_hcu() or not torch.cuda.is_available() or os.environ.get("HSA_USERPTR_FOR_PAGED_MEM") != "0":
        pytest.skip("start the test process with HSA_USERPTR_FOR_PAGED_MEM=0 on HCU")
    monkeypatch.setenv("SGLANG_HCU_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1")
    monkeypatch.setenv("SGLANG_HCU_DSV41_ENGRAM_HOST_TABLE_LAYOUT", "shared")
    daemon = _embedding()
    assert daemon.host_table.group.world_size == 1 and daemon.tp_size == 2
    source = torch.arange(12).float().unsqueeze(1).expand(-1,128).to(torch.float8_e4m3fn)
    daemon.weight.weight_loader(daemon.weight, source)
    scales = torch.full((12,4), 128, dtype=torch.uint8).view(torch.float8_e8m0fnu)
    daemon.scale.weight_loader(daemon.scale, scales)
    daemon.finish_load("IPC test")
    desc = daemon.export_ipc_host_state()
    assert desc["nbytes"] == 12 * (128 + 4)
    ctx = multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_check_client, args=(send, desc))
    try:
        proc.start()
        send.close()
        assert receive.poll(150), "timed out attaching shared Engram table"
        assert (failure := receive.recv()) is None, failure
        proc.join(20)
        assert proc.exitcode == 0
    finally:
        if proc.is_alive():
            proc.terminate()
            proc.join(20)
        receive.close()
        daemon.host_table.allocation.close()


def test_host_state_size_and_mode_mismatch(monkeypatch):
    monkeypatch.setenv("HSA_USERPTR_FOR_PAGED_MEM", "0")
    monkeypatch.setenv("SGLANG_HCU_ENABLE_DSV41_ENGRAM_HOST_TABLE", "1")
    monkeypatch.setenv("SGLANG_HCU_DSV41_ENGRAM_HOST_TABLE_LAYOUT", "shared")
    with torch.device("meta"):
        client = _embedding()
    with pytest.raises(RuntimeError, match="expects a shared host table"):
        client.attach_ipc_host_state(None)
    with pytest.raises(RuntimeError, match="size mismatch"):
        client.attach_ipc_host_state({"nbytes": 1})
    assert client.weight.is_meta


def test_host_membership_uses_local_global_ranks():
    from hcu_sglang.layers import engram_host
    from unittest.mock import Mock
    group = Mock(world_size=4, rank_in_group=3, ranks=[2,3,6,7])
    answers = [[True,False,True,False], [False,True,False,True]]
    with patch.object(engram_host, "in_the_same_node_as", side_effect=answers), patch.object(
        engram_host, "_HostTableGroup", side_effect=lambda ranks: ranks
    ):
        assert engram_host._get_host_table_group(group) == [3,7]

from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
import socket
import unittest
from unittest.mock import Mock, patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hcu_sglang.compat.aiter import bootstrap_aiter_compat
bootstrap_aiter_compat()
from hcu_sglang.environ import register_envs
register_envs()
from hcu_sglang import cp_runtime as P
from hcu_sglang.models.deepseek.current_v4.hcu_backend import compat as X
from hcu_sglang.models.deepseek.current_v4.hcu_backend import hcu_cp_utils as H
from sglang.srt.layers.utils import cp_utils as OLD
from sglang.srt.plugins.hook_registry import HookRegistry
from sglang.test.test_utils import CustomTestCase


def public_hook():
    original = (P.C.cp_materialize_global_token_order,)
    HookRegistry.reset()
    with patch.object(P, 'is_hcu', return_value=True):
        P.register_hooks()
    HookRegistry.apply_hooks()
    assert P.C.cp_materialize_global_token_order is not original[0]


def context(cp_size, gather, interleave=True):
    stack = ExitStack()
    stack.enter_context(patch.object(P, 'is_hcu', return_value=True))
    stack.enter_context(patch.object(P, 'enable_cp_v2', return_value=False))
    stack.enter_context(patch.object(P.C, 'get_parallel', return_value=SimpleNamespace(attn_cp_size=cp_size)))
    stack.enter_context(patch.object(H, 'get_parallel', return_value=SimpleNamespace(attn_cp_group=SimpleNamespace(all_gather_into_tensor=gather))))
    stack.enter_context(patch.object(H, 'attn_cp_all_gather_into_tensor', side_effect=gather))
    stack.enter_context(patch.object(H, 'use_symmetric_memory', side_effect=lambda *args, **kwargs: nullcontext()))
    stack.enter_context(patch.object(H, 'is_allocation_symmetric', return_value=False))
    stack.enter_context(patch.object(X, 'is_dsa_prefill_cp_round_robin_split', return_value=interleave))
    return stack


def gloo_worker(rank, port):
    dist.init_process_group('gloo', init_method=f'tcp://127.0.0.1:{port}', rank=rank, world_size=8)
    try:
        public_hook()
        for dims in [(4,), (2, 3)]:
            shape = (24, *dims)
            full = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
            local = torch.empty((3, *dims))
            with context(8, lambda output, value: dist.all_gather_into_tensor(output, value)):
                for multiplier in [1, 0, -3]:
                    expected = full*multiplier
                    local.copy_(expected[rank::8])
                    actual = P.C.cp_materialize_global_token_order(local, SimpleNamespace(attn_cp_metadata=SimpleNamespace(gather_index=None)))
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        dist.barrier()
    finally:
        dist.destroy_process_group()


class CPMaterializeCPU(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        public_hook()

    def test_original_native_dependency_is_a_deprecated_stub(self):
        with self.assertRaisesRegex(ValueError, 'Prefill CP on NPU/MUSA is deprecated'):
            OLD.cp_all_gather_rerange_output(torch.ones(2, 4), 8, object(), None)

    def test_public_interleave_to_existing_helper_2d_3d_and_gather_index(self):
        for cp_size in [2, 8]:
            for dims in [(4,), (2, 3)]:
                full = torch.arange(3*cp_size*torch.tensor(dims).prod().item(), dtype=torch.float32).reshape(3*cp_size, *dims)
                packed = torch.cat([full[rank::cp_size] for rank in range(cp_size)])
                local = full[1::cp_size].contiguous()
                gather = Mock(side_effect=lambda output, value: output.copy_(packed))
                with context(cp_size, gather), patch.object(H, 'cp_all_gather_rerange_output', wraps=H.cp_all_gather_rerange_output) as called:
                    for index in [None, torch.tensor([2, 0, cp_size*3-1])]:
                        batch = SimpleNamespace(attn_cp_metadata=SimpleNamespace(gather_index=index))
                        actual = P.C.cp_materialize_global_token_order(local, batch)
                        torch.testing.assert_close(actual, full if index is None else packed[index], rtol=0, atol=0)
                    self.assertEqual(called.call_count, 2)
                self.assertEqual(gather.call_count, 2)

    def test_public_zigzag_with_unequal_rank_lengths(self):
        size, length = 8, 21
        full = torch.arange(length*4, dtype=torch.float32).reshape(length, 4)
        with patch.object(X, 'is_dsa_prefill_cp_round_robin_split', return_value=False), patch('sglang.srt.layers.attention.dsa.utils.is_dsa_enable_prefill_cp', return_value=True):
            batches = [SimpleNamespace(attn_cp_metadata=H.prepare_context_parallel_metadata(length, rank, size, [length], [length], device='cpu')) for rank in range(size)]
            shards = [H.cp_split_and_rebuild_data(batch, full) for batch in batches]
        maximum = max(value.shape[0] for value in shards)
        packed = torch.cat([torch.nn.functional.pad(value, (0, 0, 0, maximum-value.shape[0])) for value in shards])
        gather = lambda output, value: output.copy_(packed)
        with context(size, gather, interleave=False):
            actual = P.C.cp_materialize_global_token_order(shards[0], batches[0])
        torch.testing.assert_close(actual, full, rtol=0, atol=0)

    def test_v2_strategy_remains_unchanged(self):
        marker, value, batch, stream = object(), object(), object(), object()
        strategy = SimpleNamespace(gather_kv_cache=Mock(return_value=marker))
        with patch.object(P, 'enable_cp_v2', return_value=True), patch.object(P, 'is_cp_active', return_value=True), patch.object(P.C, 'get_cp_strategy', return_value=strategy):
            self.assertIs(P.cp_materialize_global_token_order(value, batch, stream), marker)
        strategy.gather_kv_cache.assert_called_once_with(value, batch, stream)

    def test_non_hcu_keeps_original_return_and_exception(self):
        marker, value = object(), object()
        original = Mock(return_value=marker)
        wrapper = P._dispatch(Mock(side_effect=AssertionError('HCU implementation')))
        error = RuntimeError('original error')
        with patch.object(P, 'is_hcu', return_value=False):
            self.assertIs(wrapper(original, value), marker)
            original.assert_called_once_with(value)
            with self.assertRaises(RuntimeError) as caught:
                wrapper(Mock(side_effect=error), value)
        self.assertIs(caught.exception, error)


class CPMaterializeGloo(CustomTestCase):
    def test_eight_rank_real_collective_and_three_buffer_states(self):
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', 0))
            port = sock.getsockname()[1]
        mp.spawn(gloo_worker, args=(port,), nprocs=8, join=True)


if __name__ == '__main__':
    unittest.main()

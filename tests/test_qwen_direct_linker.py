import ctypes
import json
import os
from pathlib import Path
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from hcu_sglang.mem_cache import qwen_direct_linker as q
from hcu_sglang.mem_cache.hcu_mha_pool import HCUMHATokenToKVPool
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer, PoolHitPolicy
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool, UnquantizedKVCacheMethod
from sglang.srt.mem_cache.unified_cache.components.base import (
    ExternalLinkerLoadPhase, LinkerTransferPhase,
)


def mha(layers=2, pages=5):
    pool = object.__new__(HCUMHATokenToKVPool)
    pool.page_size = 4
    pool.quant_method = UnquantizedKVCacheMethod()
    pool.k_buffer = [torch.arange(pages * 2 * 4 * 3, dtype=torch.float32).reshape(pages, 2, 4, 3) + i * 1000 for i in range(layers)]
    pool.v_buffer = [buffer.clone() + 500 for buffer in pool.k_buffer]
    return pool


def fixture(draft=True):
    kv = object.__new__(HybridLinearKVPool)
    kv.full_kv_pool = mha()
    kv.full_attention_layer_id_mapping = {1: 0, 3: 1}
    kv.start_layer = 0
    req = NS(
        mamba_ckpt_pool=None, short_conv_pool=NS(enabled=False),
        ngram_pool=NS(enabled=False), mamba_v2p_table=None,
        mamba_map={0: 0, 2: 1},
        mamba_pool=NS(num_mamba_layers=2, mamba_cache=NS(
            temporal=torch.arange(2 * 5 * 9, dtype=torch.float32).reshape(2, 5, 3, 3),
            conv=[torch.arange(2 * 5 * 6, dtype=torch.float32).reshape(2, 5, 2, 3)],
        )),
    )
    params = NS(req_to_token_pool=req, pp_size=1, attn_cp_size=1,
                mtp_draft_device_pools=(mha(1),) if draft else (),
                mtp_draft_mamba_pools=())
    return kv, params


class QwenDirectLinkerTests(unittest.TestCase):
    def test_page_addresses_and_mixed_layer_packed_roundtrip(self):
        kv, params = fixture()
        group = q.build_qwen_pool_group(kv, params, 4)
        self.assertFalse(group.rank_replicated)
        self.assertEqual(group.num_layers, 4)
        for name, src, dst in ((PoolName.KV, torch.arange(4, 8), torch.arange(12, 16)),
                               (PoolName.MAMBA, torch.tensor([1]), torch.tensor([3]))):
            entry = group.entry_map[name]
            pointers, sizes = entry.get_page_buffer_meta(src)
            packed = b"".join(ctypes.string_at(ptr, size) for ptr, size in zip(pointers, sizes))
            locations = entry.prepare_locations(dst)
            covered = set()
            for layer in range(group.num_layers):
                meta = entry.get_prepared_layer_range_meta(locations, layer)
                if meta is None:
                    continue
                ptrs, lengths, offsets = meta
                for row_ptrs, row_sizes, row_offsets in zip(ptrs, lengths, offsets):
                    for ptr, size, offset in zip(row_ptrs, row_sizes, row_offsets):
                        ctypes.memmove(ptr, packed[offset:offset + size], size)
                        covered.update(range(offset, offset + size))
            self.assertEqual(len(covered), len(packed))
            for component in entry.components:
                for buffer in component:
                    torch.testing.assert_close(buffer[1], buffer[3])
        self.assertEqual(group.entry_map[PoolName.KV].layer_mapping[3], (1, 3, 4, 5))

    def test_hcu_page_stride_and_slot_stride(self):
        kv, params = fixture()
        group = q.build_qwen_pool_group(kv, params, 4)
        entry = group.entry_map[PoolName.KV]
        ptrs, sizes = entry.get_page_buffer_meta(torch.arange(8, 12))
        buf = kv.full_kv_pool.k_buffer[0]
        self.assertEqual(ptrs[0], buf[2].data_ptr())
        self.assertEqual(sizes[0], buf[2].nbytes)
        with self.assertRaises(ValueError):
            entry.get_page_buffer_meta(torch.arange(1, 5))
        with self.assertRaises(ValueError):
            entry.get_page_buffer_meta(torch.arange(20, 24))

    def test_hybrid_draft_wrapper_with_shared_state(self):
        kv, params = fixture()
        draft = object.__new__(HybridLinearKVPool)
        draft.full_kv_pool = params.mtp_draft_device_pools[0]
        draft.mamba_pool = params.req_to_token_pool.mamba_pool
        params.mtp_draft_device_pools = (draft,)
        group = q.build_qwen_pool_group(kv, params, 4)
        self.assertEqual(group.entry_map[PoolName.KV].layer_mapping[3], (1, 3, 4, 5))
        draft.mamba_pool = object()
        with self.assertRaises(ValueError):
            q.build_qwen_pool_group(kv, params, 4)

    def test_namespace_excludes_capacity_but_includes_draft_and_dtype(self):
        kv, params = fixture()
        schema = q.build_qwen_pool_group(kv, params, 4).hcu_qwen_schema
        kv.full_kv_pool = mha(pages=8)
        self.assertEqual(schema, q.build_qwen_pool_group(kv, params, 4).hcu_qwen_schema)
        params.mtp_draft_device_pools = ()
        self.assertNotEqual(schema, q.build_qwen_pool_group(kv, params, 4).hcu_qwen_schema)
        schema = q.build_qwen_pool_group(kv, params, 4).hcu_qwen_schema
        kv.full_kv_pool.k_buffer = [t.to(torch.bfloat16) for t in kv.full_kv_pool.k_buffer]
        self.assertNotEqual(schema, q.build_qwen_pool_group(kv, params, 4).hcu_qwen_schema)

    def test_unsupported_layout_and_topology_fail_closed(self):
        kv, params = fixture()
        params.pp_size = 2
        with self.assertRaises(ValueError):
            q.build_qwen_pool_group(kv, params, 4)
        params.pp_size = 1
        params.req_to_token_pool.short_conv_pool.enabled = True
        with self.assertRaises(ValueError):
            q.build_qwen_pool_group(kv, params, 4)
        params.req_to_token_pool.short_conv_pool.enabled = False
        params.mtp_draft_mamba_pools = (object(),)
        with self.assertRaises(ValueError):
            q.build_qwen_pool_group(kv, params, 4)
        params.mtp_draft_mamba_pools = ()
        kv.full_kv_pool.k_buffer[0] = kv.full_kv_pool.k_buffer[0].transpose(1, 2)
        with self.assertRaises(ValueError):
            q.build_qwen_pool_group(kv, params, 4)

    def test_mamba_transfer_is_checkpoint_at_last_key(self):
        component = NS(component_type=9, _alloc_mamba_slot=Mock(return_value=torch.tensor([3])))
        node = NS(hash_value=["a", "b"], component_data={9: NS(value=torch.tensor([2]))})
        for phase in (LinkerTransferPhase.OFFLOAD, LinkerTransferPhase.LOAD):
            transfer = q.around_build_transfer(None, component, phase, node, ["a", "b"])
            self.assertEqual(transfer.keys, ["b"])
            self.assertEqual(transfer.hit_policy, PoolHitPolicy.TRAILING_PAGES)
            self.assertEqual(transfer.device_indices.item(), 2 if phase == LinkerTransferPhase.OFFLOAD else 3)
        self.assertIsNone(q.around_build_transfer(None, component, LinkerTransferPhase.LOAD, None, []))

    def test_abort_preserves_old_request_state_prepare_replaces_it(self):
        allocator = NS(free=Mock())
        write_pos = torch.full((8,), 7)
        pool = NS(mamba_allocator=allocator, mamba_pool=NS(replayssm_write_pos=write_pos),
                  translate_mamba_indices=lambda ids: ids)
        component = NS(cache=NS(req_to_token_pool=pool))
        req = NS(kv=NS(holds_mamba=True, mamba_pool_idx=torch.tensor(1),
                       mamba_cow_src_index=8, mamba_needs_clear=True))
        transfer = PoolTransfer(name=PoolName.MAMBA, device_indices=torch.tensor([3]))
        q.around_update_load(None, component, ExternalLinkerLoadPhase.ABORT, req, None, transfer, 16)
        self.assertEqual(allocator.free.call_args.args[0].item(), 3)
        self.assertEqual(req.kv.mamba_pool_idx.item(), 1)
        q.around_update_load(None, component, ExternalLinkerLoadPhase.PREPARE, req, None, transfer, 16)
        self.assertEqual(allocator.free.call_args.args[0].item(), 1)
        self.assertEqual(req.kv.mamba_pool_idx.item(), 3)
        self.assertEqual(write_pos[3].item(), 0)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)

    def test_keys_and_fallback(self):
        store = NS(mem_pool_host=NS(hcu_qwen_schema="abc"), mla_suffix="tp0_schema")
        for name, suffix in ((PoolName.KV, "k"), (PoolName.MAMBA, str(PoolName.MAMBA))):
            keys, count = q.around_component_keys(None, store, ["a", "b"], PoolTransfer(name=name))
            self.assertEqual(keys, [f"a_tp0_schema_{suffix}", f"b_tp0_schema_{suffix}"])
            self.assertEqual(count, 1)
        store.mem_pool_host = None
        original = Mock(return_value=([], 2))
        self.assertEqual(q.around_component_keys(original, store, [], PoolTransfer(name=PoolName.KV)), ([], 2))
        original.assert_called_once()

    def test_draft_waits_for_packed_transfer_layer(self):
        counter = NS(wait_until=Mock())
        q._DraftLoadCounter(counter, 63).wait_until(0)
        counter.wait_until.assert_called_once_with(63)

    def test_sparse_gdn_checkpoints_bound_the_restorable_prefix(self):
        from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import MooncakeStore

        kv, params = fixture()
        group = q.build_qwen_pool_group(kv, params, 4)
        store = NS(mem_pool_host=group, mla_suffix='tp0_schema')
        store._get_hybrid_page_component_keys = lambda keys, transfer: q.around_component_keys(None, store, keys, transfer)
        store._tag_keys = lambda keys: keys
        store._batch_exist = Mock(return_value=[1, 1, 1, 1, 1, 0, 1, 0])
        transfers = [
            PoolTransfer(name=PoolName.KV, keys=['a', 'b', 'c', 'd'], hit_policy=PoolHitPolicy.ALL_PAGES),
            PoolTransfer(name=PoolName.MAMBA, keys=['d'], hit_policy=PoolHitPolicy.TRAILING_PAGES),
        ]
        result = MooncakeStore.batch_exists_v2(store, ['a', 'b', 'c', 'd'], transfers)
        self.assertEqual(result.restorable_prefix_pages, [1, 3])
        store._batch_exist.return_value = [1, 1, 1, 1, 0, 0, 0, 0]
        result = MooncakeStore.batch_exists_v2(store, ['a', 'b', 'c', 'd'], transfers)
        self.assertEqual(result.restorable_prefix_pages, [])

    def test_non_hcu_does_not_register(self):
        from sglang.srt.plugins.hook_registry import HookRegistry
        from sglang.srt.mem_cache.unified_cache import unified_cache_linker

        supported = unified_cache_linker._EXTERNAL_LINKER_SUPPORTED_COMPONENTS
        with patch('hcu_sglang.utils.detection.is_hcu', return_value=False), patch.object(HookRegistry, "register") as register:
            q.register_hooks()
            register.assert_not_called()
            self.assertIs(unified_cache_linker._EXTERNAL_LINKER_SUPPORTED_COMPONENTS, supported)

    def test_default_hook_contract_without_environment_gate(self):
        from sglang.srt.plugins.hook_registry import HookRegistry
        from sglang.srt.mem_cache.unified_cache import unified_cache_linker
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        expected = [
            'sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler.resolve_hybrid_device_pool_group',
            'sglang.srt.mem_cache.storage.mooncake_store.mooncake_direct_linker.MooncakeDirectLinker.__init__',
            'sglang.srt.mem_cache.storage.mooncake_store.mooncake_store.MooncakeStore._get_hybrid_page_component_keys',
            'sglang.srt.mem_cache.unified_cache.components.mamba.MambaComponent.build_external_linker_transfer',
            'sglang.srt.mem_cache.unified_cache.components.mamba.MambaComponent.update_external_linker_load',
        ]
        contract = json.loads((Path(__file__).parent / 'contracts/hcu_source_alignment_targets.json').read_text())
        self.assertTrue(set(expected).issubset(contract['expected_hook_targets']))
        for value in (None, '0', '1'):
            with self.subTest(legacy_environment=value), patch.dict(os.environ), patch('hcu_sglang.utils.detection.is_hcu', return_value=True), patch.object(HookRegistry, 'register') as register, patch.object(unified_cache_linker, '_EXTERNAL_LINKER_SUPPORTED_COMPONENTS', {ComponentType.FULL}):
                os.environ.pop('SGLANG_HCU_QWEN_L3_DIRECT', None)
                if value is not None:
                    os.environ['SGLANG_HCU_QWEN_L3_DIRECT'] = value
                q.register_hooks()
                self.assertEqual([call.args[0] for call in register.call_args_list], expected)
                self.assertEqual(unified_cache_linker._EXTERNAL_LINKER_SUPPORTED_COMPONENTS, {ComponentType.FULL, ComponentType.MAMBA})


if __name__ == "__main__":
    unittest.main()

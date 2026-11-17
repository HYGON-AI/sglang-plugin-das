from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, replace
import json
from enum import Enum
from pathlib import Path
from types import SimpleNamespace as S
import unittest
from unittest.mock import patch

import torch
from hcu_sglang.mem_cache import cpu_cache as H, cpu_cache_methods as M
from sglang.srt.mem_cache.layout import token_major as L
from sglang.srt.mem_cache import multimodal_cache as C
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer, PoolHitPolicy
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.plugins.hook_registry import HookRegistry


class LinkerPhase(str, Enum):
    OFFLOAD = "offload"
    LOOKUP = "lookup"
    LOAD = "load"


class LoadPhase(str, Enum):
    PREPARE = "prepare"
    ABORT = "abort"


class CpuCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cache_class = C.MultiModalStaticCache
        cls.cache_alias = C.MultiModalStaticCache
        cls.native_set = C.MultiModalStaticCache.set
        cls.native_views = L.build_mamba_entry_views
        HookRegistry.reset()
        with patch.object(H, "is_hcu", return_value=True), patch.object(H, "_GROUPS", H._GROUPS[:2]):
            H.register_hooks()
            HookRegistry.apply_hooks()
        if set(HookRegistry._hooks) != HookRegistry._patched:
            raise AssertionError("Leaf cache hooks did not apply")

    @classmethod
    def tearDownClass(cls):
        C.MultiModalStaticCache.set = cls.native_set
        L.build_mamba_entry_views = cls.native_views
        HookRegistry.reset()

    def test_oversized_video_keeps_existing_lru_and_size(self):
        with patch.object(H, "is_hcu", return_value=True):
            cache = self.cache_alias(max_size=16)
            first = C.EmbeddingResult(embedding=torch.ones(2))
            second = C.EmbeddingResult(embedding=torch.ones(2))
            self.assertTrue(cache.set(1, first))
            self.assertTrue(cache.set(2, second))
            cache.get_single(1)
            before = list(cache.mm_cache)
            self.assertFalse(cache.set(3, C.EmbeddingResult(embedding=torch.ones(5))))
            self.assertEqual(list(cache.mm_cache), before)
            self.assertEqual(cache.current_size, 16)
            self.assertIs(C.MultiModalStaticCache, self.cache_class)
            self.assertIs(cache.mm_cache[1], first)
            self.assertTrue(cache.set(4, C.EmbeddingResult(embedding=torch.ones(2))))
            self.assertEqual(list(cache.mm_cache), [1, 4])

    def test_non_hcu_uses_original_and_slice_admission_clones(self):
        with patch.object(H, "is_hcu", return_value=False):
            cache = self.cache_alias(8)
            cache.set(1, C.EmbeddingResult(embedding=torch.ones(2)))
            self.assertFalse(cache.set(2, C.EmbeddingResult(embedding=torch.ones(3))))
            self.assertEqual(list(cache.mm_cache), [])
        with patch.object(H, "is_hcu", return_value=True):
            cache = self.cache_alias(8)
            raw = torch.arange(32, dtype=torch.float32)
            self.assertTrue(cache.set(1, C.EmbeddingResult(embedding=raw[:2])))
            self.assertEqual(cache.current_size, 8)
            raw[0] = 999
            self.assertEqual(cache.get_single(1).embedding[0].item(), 0)

    def test_mamba_shared_entry_views_write_only_assigned_bytes(self):
        raw = torch.full((256,), 255, dtype=torch.uint8)
        kwargs = dict(layer_num=2, conv_state_shapes=[(2,)], conv_dtype=torch.float32,
                      temporal_state_shape=(1,), temporal_dtype=torch.float32, max_slots=3)
        with patch.object(H, "is_hcu", return_value=True):
            conv, temporal = L.build_mamba_entry_views(raw, anchor_bytes=32, entry_bytes=64, offset_bytes=16, **kwargs)
        conv[0].fill_(1)
        temporal.fill_(2)
        expected = torch.full_like(raw, 255)
        for slot in range(3):
            base = 32 + slot * 64 + 16
            expected[base:base + 16] = torch.ones(4).view(torch.uint8)
            expected[base + 16:base + 24] = torch.full((2,), 2.0).view(torch.uint8)
        self.assertTrue(torch.equal(raw, expected))
        self.assertEqual(conv[0].stride(), (2, 16, 1))
        standalone = torch.zeros(72, dtype=torch.uint8)
        with patch.object(H, "is_hcu", return_value=True):
            actual = L.build_mamba_entry_views(standalone, **kwargs)
        native = type(self).native_views(standalone, **kwargs)
        self.assertEqual(actual[0][0].stride(), native[0][0].stride())
        self.assertEqual(actual[1].stride(), native[1].stride())

    def method(self, donor, name, **namespace):
        return H._bind(vars(donor)[name], dict(torch=torch, replace=replace,
            PoolName=PoolName, PoolTransfer=PoolTransfer, PoolHitPolicy=PoolHitPolicy,
            ComponentType=ComponentType, **namespace))

    def test_storage_keys_separate_mamba_tp_but_share_mla_kv(self):
        hf = self.method(M.HiCacheHF3FS, "_pool_component_key")
        sea = self.method(M.SeaweedFSStore, "_component_key")
        left, right = S(is_mla_model=True, tp_rank=0, tp_size=2), S(is_mla_model=True, tp_rank=1, tp_size=2)
        for pool in (PoolName.MAMBA, PoolName.KV, PoolName.SWA):
            self.assertEqual(hf(left, "page", pool) == hf(right, "page", pool), pool != PoolName.MAMBA)
        left._mamba_rank_tag, right._mamba_rank_tag = ".tp0of2", ".tp1of2"
        self.assertNotEqual(sea(left, "page", PoolName.MAMBA), sea(right, "page", PoolName.MAMBA))
        self.assertEqual(sea(left, "page", PoolName.KV), "page")
        self.assertEqual(sea(left, "page", PoolName.SWA), "page.swa")
        left.is_mla_model = False
        self.assertEqual(hf(left, "page", PoolName.MAMBA), "page_mamba")

    def test_tensorcast_uses_actual_packed_layer_count(self):
        family = S(MLA=object())
        validate = self.method(M.Tensorcast, "_validate_storage_config", _SUPPORTED_FULL_LAYOUTS=("page_first",), _PoolFamily=family)
        pool = S(layout="page_first", layer_num=2, target_layer_num=2, dcp_size=1, dcp_rank=0, page_size=2, page_num=4, mtp_draft_device_pools=(object(),))
        config = S(attn_cp_size=1, attn_cp_rank=0, tp_lcm_size=None, should_split_heads=False, is_mla_model=True)
        validate(pool, family=family.MLA, storage_config=config)
        pool.layer_num = 3
        with self.assertRaisesRegex(NotImplementedError, "packed MTP"):
            validate(pool, family=family.MLA, storage_config=config)

    def test_both_window_minima_and_pending_wait_cleanup(self):
        threshold = self.method(M.StorageAttachment, "resolve_prefetch_threshold")
        cache = S(host_memory_mode="buffer_only", page_size=8, components={ComponentType.SWA:S(full_window_pages=3), ComponentType.AUXILIARY_SWA:S(full_window_pages=5)})
        self.assertEqual(threshold(S(_cache=cache), 16), 40)
        cache.host_memory_mode = "cache"
        self.assertEqual(threshold(S(_cache=cache), 16), 16)
        cache.cache_controller = None
        for name in ("ongoing_prefetch", "ongoing_backup", "_storage_prefetch_hit_remaining_by_reqid", "prefetch_loaded_tokens_by_reqid", "prefetch_loaded_storage_start_by_reqid", "storage_prefetch_retries", "_storage_prefetch_wait"):
            setattr(cache, name, {} if name in ("ongoing_prefetch", "ongoing_backup", "_storage_prefetch_hit_remaining_by_reqid") else {"req": 1})
        self.method(M.StorageAttachment, "_release_pending_storage_ops")(S(_cache=cache))
        self.assertEqual(cache._storage_prefetch_wait, {})
        self.assertEqual(cache.prefetch_loaded_tokens_by_reqid, {})

    def test_storage_constructor_initializes_wait_state_without_overwriting(self):
        hook = H._make_hook("unused", "StorageAttachment", M.StorageAttachment, "__init__", ())
        calls = []

        def original(attachment, cache):
            calls.append(cache)
            attachment._cache = cache

        attachment, cache = S(), S()
        with patch.object(H, "is_hcu", return_value=False):
            hook(original, attachment, cache)
        self.assertFalse(hasattr(cache, "_storage_prefetch_wait"))
        with patch.object(H, "is_hcu", return_value=True):
            hook(original, attachment, cache)
            waiting = cache._storage_prefetch_wait
            waiting["request"] = object()
            hook(original, attachment, cache)
        self.assertIs(cache._storage_prefetch_wait, waiting)
        self.assertIn("request", waiting)
        self.assertEqual(len(calls), 3)

    def test_linker_constructor_admits_mamba_without_changing_native_support(self):
        supported = frozenset({ComponentType.FULL, ComponentType.SWA})
        native = S(_EXTERNAL_LINKER_SUPPORTED_COMPONENTS=supported,
                   ComponentType=ComponentType, is_swa_req_ring=lambda allocator: False)
        module = "sglang.srt.mem_cache.unified_cache.unified_cache_linker"
        hook = H._make_hook(module, "UnifiedCacheLinkerWrapper", M.UnifiedCacheLinkerWrapper, "__init__", ())
        component = S()
        cache = S(tree_components=(ComponentType.FULL, ComponentType.MAMBA),
                  components={ComponentType.MAMBA: component}, _components_tuple=(component,),
                  token_to_kv_pool_allocator=S(), tree_core=S(enable_external_cache_linker=False))
        wrapper, linker = S(), S()

        def original(obj, current_cache, current_linker):
            unsupported = set(current_cache.tree_components) - native._EXTERNAL_LINKER_SUPPORTED_COMPONENTS
            if unsupported:
                raise ValueError("native rejects Mamba")
            obj.cache_linker = current_linker

        with patch.object(H, "is_hcu", return_value=True), patch.object(H, "import_module", return_value=native):
            hook(original, wrapper, cache, linker)
            self.assertIs(wrapper.cache, cache)
            self.assertIs(wrapper.cache_linker, linker)
            self.assertEqual(wrapper._components, (component,))
            self.assertEqual(wrapper.host_prefetch_hits, {})
            self.assertEqual(wrapper.pending_host_prefetch_submissions, {})
            self.assertTrue(cache.tree_core.enable_external_cache_linker)
            self.assertEqual(cache.write_through_threshold, 1)
            cache.tree_components = (ComponentType.AUXILIARY_SWA,)
            with self.assertRaisesRegex(ValueError, "unsupported"):
                hook(original, S(), cache, linker)
        self.assertIs(native._EXTERNAL_LINKER_SUPPORTED_COMPONENTS, supported)
        cache.tree_components = (ComponentType.FULL, ComponentType.MAMBA)
        with patch.object(H, "is_hcu", return_value=False):
            with self.assertRaisesRegex(ValueError, "native rejects Mamba"):
                hook(original, S(), cache, linker)
        self.assertIs(native._EXTERNAL_LINKER_SUPPORTED_COMPONENTS, supported)

    def component(self):
        freed = []
        allocator = S(alloc=lambda n: torch.tensor([2]), free=lambda indices: freed.append(indices.tolist()))
        pool = S(mamba_allocator=allocator, mamba_pool=S(replayssm_write_pos=torch.tensor([7, 7, 7, 7])), translate_mamba_indices=lambda x:x)
        config = S(mamba=S(enable_mamba_extra_buffer_lazy=True))
        namespace = dict(torch=torch, replace=replace, PoolName=PoolName, PoolTransfer=PoolTransfer,
            PoolHitPolicy=PoolHitPolicy, LinkerTransferPhase=LinkerPhase, ExternalLinkerLoadPhase=LoadPhase,
            get_exec=lambda:config, EvictParams=lambda **kw:S(**kw))
        methods = {name:H._bind(fn, namespace) for name, fn in vars(M.MambaComponent).items() if callable(fn)}
        cls = type("MambaFixture", (), methods)
        obj = cls()
        obj.cache = S(req_to_token_pool=pool)
        obj.component_type = ComponentType.MAMBA
        return obj, pool, freed

    def test_mamba_lookup_load_prepare_and_abort_lifecycle(self):
        component, pool, freed = self.component()
        node = S(hash_value=["first", "last"], component_data={ComponentType.MAMBA:S(value=torch.tensor([1]))})
        transfer = component.build_external_linker_transfer(LinkerPhase.OFFLOAD, node, None)
        self.assertEqual(transfer.keys, ["last"])
        self.assertEqual(transfer.device_indices.tolist(), [1])
        lookup = component.build_external_linker_transfer(LinkerPhase.LOOKUP, None, ["first", "last"])
        self.assertIsNone(lookup.device_indices)
        loaded = component.build_external_linker_transfer(LinkerPhase.LOAD, None, ["first", "last"])
        self.assertEqual(loaded.keys, ["last"])
        req = S(kv=S(holds_mamba=True, mamba_pool_idx=torch.tensor(1), mamba_cow_src_index=123, mamba_needs_clear=True))
        self.assertIs(component.update_external_linker_load(LoadPhase.PREPARE, req, None, loaded, 8), loaded)
        self.assertEqual(freed, [[1]])
        self.assertEqual(req.kv.mamba_pool_idx.item(), 2)
        self.assertIsNone(req.kv.mamba_cow_src_index)
        self.assertFalse(req.kv.mamba_needs_clear)
        self.assertEqual(pool.mamba_pool.replayssm_write_pos.tolist(), [7, 7, 0, 7])
        component.update_external_linker_load(LoadPhase.ABORT, req, None, loaded, 8)
        self.assertEqual(freed, [[1], [2]])
        self.assertIsNone(component.build_external_linker_transfer(LinkerPhase.LOAD, None, []))

    def test_lazy_mamba_allocation_releases_only_cow_lock_once(self):
        component, pool, _ = self.component()
        results = iter([None, torch.tensor([2]), None, torch.tensor([3])])
        pool.mamba_allocator.alloc = lambda n:next(results)
        events = []
        component.cache.tree_components = (ComponentType.FULL, ComponentType.SWA, ComponentType.MAMBA)
        component.cache.dec_lock_ref = lambda node,receipt:events.append((node,receipt.skipped_lock_components))
        component.cache.evict_for_alloc = lambda params:events.append((params.num_tokens,params.mamba_num))
        @dataclass
        class Receipt:
            skipped_lock_components: tuple = ()
        req = S(lock=S(node=7, swa_released=False, receipt=Receipt()))
        self.assertEqual(component._alloc_mamba_slot(req).item(), 2)
        self.assertEqual(events[0], (7, (ComponentType.FULL, ComponentType.SWA)))
        self.assertEqual(req.lock.receipt.skipped_lock_components, (ComponentType.MAMBA,))
        component._alloc_mamba_slot(req)
        self.assertEqual(len([e for e in events if e[0] == 7]), 1)

    def test_mamba_hook_non_hcu_preserves_effective_original(self):
        hook = H._make_hook("unused", "MambaComponent", M.MambaComponent, "_alloc_mamba_slot", ("_adopt_loaded_state",))
        receiver, request, result = object(), object(), object()
        calls = []

        def original(obj, req=None):
            calls.append((obj, req))
            return result

        with patch.object(H, "is_hcu", return_value=False), patch.object(H, "import_module") as importer:
            self.assertIs(hook(original, receiver, req=request), result)
            importer.assert_not_called()
        self.assertEqual(calls, [(receiver, request)])

    def test_mamba_helper_is_installed_before_original_constructor(self):
        class NativeComponent:
            pass

        native = S(MambaComponent=NativeComponent)
        hook = H._make_hook("namespace_fixture", "MambaComponent", M.MambaComponent, None, ("_adopt_loaded_state",))
        pool = S(mamba_pool=S(replayssm_write_pos=torch.tensor([5, 6])), translate_mamba_indices=lambda value:value)
        request = S(kv=S(mamba_pool_idx=torch.tensor(1), mamba_cow_src_index=9, mamba_needs_clear=True))
        component = NativeComponent()
        result = object()

        def original(obj, current_pool):
            obj.cache = S(req_to_token_pool=current_pool)
            obj._adopt_loaded_state(request)
            return result

        with patch.object(H, "is_hcu", return_value=True), patch.object(H, "import_module", return_value=native), patch.dict("sys.modules", {"sglang.srt.mem_cache.unified_cache.components.base": S(ExternalLinkerLoadPhase=LoadPhase)}):
            self.assertIs(hook(original, component, pool), result)
            helper = NativeComponent._adopt_loaded_state
            hook(original, component, pool)
            self.assertIs(NativeComponent._adopt_loaded_state, helper)
        self.assertEqual(pool.mamba_pool.replayssm_write_pos.tolist(), [5, 0])
        self.assertIsNone(request.kv.mamba_cow_src_index)
        self.assertFalse(request.kv.mamba_needs_clear)

    def test_tensorcast_missing_dependency_does_not_register_unresolvable_target(self):
        group = tuple(row for row in H._GROUPS if row[2] == "Tensorcast")
        with patch.object(H, "is_hcu", return_value=True), patch.object(H, "_GROUPS", group), patch.object(H, "find_spec", return_value=None) as lookup, patch.object(HookRegistry, "register") as register:
            H.register_hooks()
            lookup.assert_called_once_with("tensorcast")
            register.assert_not_called()

    def test_tensorcast_available_dependency_keeps_public_target(self):
        group = tuple(row for row in H._GROUPS if row[2] == "Tensorcast")
        target = "sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store._validate_storage_config"
        with patch.object(H, "is_hcu", return_value=True), patch.object(H, "_GROUPS", group), patch.object(H, "find_spec", return_value=object()), patch.object(HookRegistry, "register") as register:
            H.register_hooks()
            self.assertEqual(register.call_count, 1)
            self.assertEqual(register.call_args.args[0], target)

    def test_tensorcast_probe_errors_are_not_swallowed(self):
        group = tuple(row for row in H._GROUPS if row[2] == "Tensorcast")
        with patch.object(H, "is_hcu", return_value=True), patch.object(H, "_GROUPS", group), patch.object(H, "find_spec", side_effect=ImportError("broken installed package")):
            with self.assertRaisesRegex(ImportError, "broken installed package"):
                H.register_hooks()

    def mooncake(self, name, **namespace):
        return self.method(M.MooncakeStore, name, **namespace)

    def test_mooncake_client_http_port_follows_process_rank(self):
        envs = S(MOONCAKE_CLIENT_METRICS_PORT_BASE=S(get=lambda: 12000))
        port = self.method(M.MooncakeGlue, "_get_mooncake_client_http_port", envs=envs)
        dist = S(is_available=lambda: True, is_initialized=lambda: False)
        with patch.object(torch, "distributed", dist):
            self.assertEqual(port(), 12000)
            dist.is_initialized = lambda: True
            dist.get_rank = lambda: 3
            dist.get_world_size = lambda: 2
            self.assertEqual(port(), 12003)
            self.assertEqual(port(1), 12005)
        kwargs = self.method(M.MooncakeGlue, "_get_mooncake_client_http_setup_kwargs", envs=envs, _get_mooncake_client_http_port=port)
        with patch.object(torch, "distributed", dist):
            self.assertEqual(kwargs(False), {})
            self.assertEqual(kwargs(True), {"enable_client_http_server": True, "client_http_port": 12003})

    def test_mooncake_config_protocol_falls_back_to_environment(self):
        captured = {}

        class Config:
            @staticmethod
            def _resolve_local_hostname(config):
                return "host"

            def __init__(self, **kwargs):
                captured.update(kwargs)

        settings = S(
            SGLANG_HICACHE_MOONCAKE_CONFIG_PATH=S(is_set=lambda: True, get=lambda: "/tmp/mooncake.json"),
            MOONCAKE_TE_META_DATA_SERVER=S(default="meta"), MOONCAKE_GLOBAL_SEGMENT_SIZE=S(default="1GB"),
            MOONCAKE_PROTOCOL=S(get=lambda: "efa", default="tcp"),
            MOONCAKE_DEVICE=S(default=""), MOONCAKE_MASTER=S(default="m"), MOONCAKE_MASTER_METRICS_PORT=S(default=9003),
            MOONCAKE_CHECK_SERVER=S(default=False), MOONCAKE_STANDALONE_STORAGE=S(default=False),
            MOONCAKE_CLIENT=S(default=""), MOONCAKE_ENABLE_SSD_OFFLOAD=S(default=False),
            MOONCAKE_OFFLOAD_FILE_STORAGE_PATH=S(default=""), MOONCAKE_TENANT_ID=S(default="default"),
        )
        load = self.method(M.MooncakeStoreConfig, "from_file", envs=settings, json=json, MooncakeStoreConfig=Config,
            _parse_global_segment_size=lambda value: 1, _normalize_tenant_id=lambda value: value)
        Path("/tmp/mooncake.json").write_text('{"master_server_address": "10.0.0.1"}')
        load()
        self.assertEqual(captured["protocol"], "efa")
        Path("/tmp/mooncake.json").write_text('{"master_server_address": "10.0.0.1", "protocol": "rdma"}')
        load()
        self.assertEqual(captured["protocol"], "rdma")

    def test_mooncake_register_buffer_expands_tuples_and_skips_empty(self):
        calls = []

        class Store:
            def register_buffer(self, ptr, size):
                calls.append((ptr, size))
                return 0

        register = self.method(M.MooncakeBaseStore, "register_buffer", logger=S(error=lambda *a, **k: None))
        owner = S(store=Store())
        owner.register_buffer = lambda tensor: register(owner, tensor)
        register(owner, torch.zeros(4, dtype=torch.int32))
        tensor = torch.zeros(2, dtype=torch.float32)
        register(owner, (torch.zeros(0, dtype=torch.float32), [tensor]))
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], 16)
        self.assertEqual(calls[1][1], 8)
        with self.assertRaisesRegex(RuntimeError, "not initialized"):
            register(S(store=None), tensor)

    def test_mooncake_hybrid_keys_use_one_object_per_direct_linker_page(self):
        class Entry:
            pass

        keys = self.method(M.MooncakeStore, "_get_hybrid_page_component_keys", DevicePoolEntry=Entry)
        owner = S(registered_pools={PoolName.KV: Entry()}, mla_suffix="r0")
        component_keys, multiplier = keys(owner, ["p1", "p2"], S(name=PoolName.KV))
        self.assertEqual(multiplier, 1)
        self.assertEqual(component_keys, ["p1_r0_k", "p2_r0_k"])
        owner.registered_pools = {}
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            keys(owner, ["p1"], S(name=PoolName.KV))

    def test_mooncake_module_helpers_are_injected_not_hook_targets(self):
        module_name = "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store"
        envs = S(MOONCAKE_CLIENT_METRICS_PORT_BASE=S(get=lambda: 12000))
        namespace = {"envs": envs, "torch": torch}
        H._inject_module_helpers(namespace, module_name)
        self.assertIn("_get_mooncake_client_http_port", namespace)
        self.assertIn("_get_mooncake_client_http_setup_kwargs", namespace)
        setup_kwargs = namespace["_get_mooncake_client_http_setup_kwargs"]
        with patch.object(torch, "distributed", S(is_available=lambda: False)):
            self.assertEqual(setup_kwargs(False), {})
            self.assertEqual(
                setup_kwargs(True),
                {"enable_client_http_server": True, "client_http_port": 12000},
            )
        sentinel = object()
        namespace["_get_mooncake_client_http_port"] = sentinel
        H._inject_module_helpers(namespace, module_name)
        self.assertIs(namespace["_get_mooncake_client_http_port"], sentinel)

    def test_mooncake_glue_is_not_registered_as_module_target(self):
        module_name = "sglang.srt.mem_cache.storage.mooncake_store.mooncake_store"
        HookRegistry.reset()
        try:
            with patch.object(H, "is_hcu", return_value=True):
                H.register_hooks()
            registered = set(HookRegistry._hooks)
        finally:
            HookRegistry.reset()
        self.assertNotIn(module_name + "._get_mooncake_client_http_port", registered)
        self.assertNotIn(module_name + "._get_mooncake_client_http_setup_kwargs", registered)
        self.assertNotIn(module_name + ".__init__", registered)
        self.assertIn(module_name + ".MooncakeStore.__init__", registered)

    def test_gpu_was_not_initialized(self):
        self.assertFalse(torch.cuda.is_initialized())


if __name__ == "__main__":
    unittest.main(verbosity=2)

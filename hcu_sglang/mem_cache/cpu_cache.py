# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import replace
from importlib import import_module
from importlib.util import find_spec
from types import FunctionType

from hcu_sglang.mem_cache import cpu_cache_methods
from hcu_sglang.utils.detection import is_hcu


_GROUPS = (('sglang.srt.mem_cache.layout.token_major',
  '',
  'TokenMajor',
  ('build_mamba_entry_views',),
  ()),
 ('sglang.srt.mem_cache.multimodal_cache',
  'MultiModalStaticCache',
  'MultiModalStaticCache',
  ('set',),
  ()),
 ('sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs',
  'HiCacheHF3FS',
  'HiCacheHF3FS',
  ('__init__', 'from_env_config', 'batch_exists_v2', '_pool_batch_get', '_pool_batch_set'),
  ('_pool_component_key',)),
 ('sglang.srt.mem_cache.storage.seaweedfs.seaweedfs_store',
  'SeaweedFSStore',
  'SeaweedFSStore',
  ('__init__', '_component_key'),
  ()),
 ('sglang.srt.mem_cache.storage.tensorcast_store.tensorcast_store',
  '',
  'Tensorcast',
  ('_validate_storage_config',),
  ()),
 ('sglang.srt.mem_cache.unified_cache.storage_attachment',
  'StorageAttachment',
  'StorageAttachment',
  ('__init__', 'attach', 'resolve_prefetch_threshold', '_release_pending_storage_ops'),
  ()),
 ('sglang.srt.mem_cache.unified_cache.components.mamba',
  'MambaComponent',
  'MambaComponent',
  ('_alloc_mamba_slot',
   'prepare_for_caching_req',
   'build_external_linker_transfer',
   'update_external_linker_load'),
  ('_adopt_loaded_state',)),
 ('sglang.srt.mem_cache.unified_cache.components.mamba',
  '_MambaBufferLoadBackContext',
  'MambaBufferLoadBackContext',
  ('finalize_allocation',),
  ()),
 ('sglang.srt.mem_cache.storage.mooncake_store.mooncake_store',
  '',
  'MooncakeGlue',
  (),
  ('_get_mooncake_client_http_port', '_get_mooncake_client_http_setup_kwargs')),
 ('sglang.srt.mem_cache.storage.mooncake_store.mooncake_store',
  'MooncakeStoreConfig',
  'MooncakeStoreConfig',
  ('from_file', 'load_from_extra_config'),
  ()),
 ('sglang.srt.mem_cache.storage.mooncake_store.mooncake_store',
  'MooncakeBaseStore',
  'MooncakeBaseStore',
  ('register_buffer',),
  ()),
 ('sglang.srt.mem_cache.storage.mooncake_store.mooncake_store',
  'MooncakeStore',
  'MooncakeStore',
  ('__init__', '_get_hybrid_page_component_keys'),
  ()),
 ('sglang.srt.mem_cache.unified_cache.unified_cache_linker',
  'UnifiedCacheLinkerWrapper',
  'UnifiedCacheLinkerWrapper',
  ('__init__',),
  ()))


_MODULE_HELPERS = {
    'sglang.srt.mem_cache.storage.mooncake_store.mooncake_store': (
        'MooncakeGlue',
        ('_get_mooncake_client_http_port', '_get_mooncake_client_http_setup_kwargs'),
    ),
}


def _inject_module_helpers(namespace, module_name):
    spec = _MODULE_HELPERS.get(module_name)
    if spec is None:
        return
    glue_name, names = spec
    glue = getattr(cpu_cache_methods, glue_name)
    for name in names:
        namespace.setdefault(name, _bind(vars(glue)[name], namespace))


def _bind(source, namespace):
    if isinstance(source, (staticmethod, classmethod)):
        source = source.__func__
    bound = FunctionType(
        source.__code__, namespace, source.__name__, source.__defaults__, source.__closure__
    )
    bound.__kwdefaults__ = source.__kwdefaults__
    bound.__annotations__ = source.__annotations__
    return bound


def _make_hook(module_name, class_name, donor, method, helpers):
    if donor is cpu_cache_methods.StorageAttachment and method == "__init__":
        return _storage_attachment_init
    bound = None

    def around(original, *args, **kwargs):
        nonlocal bound
        if not is_hcu():
            return original(*args, **kwargs)
        native = import_module(module_name)
        namespace = vars(native)
        if donor is cpu_cache_methods.UnifiedCacheLinkerWrapper:
            namespace = dict(namespace)
            namespace["_EXTERNAL_LINKER_SUPPORTED_COMPONENTS"] = frozenset(
                native._EXTERNAL_LINKER_SUPPORTED_COMPONENTS | {native.ComponentType.MAMBA}
            )
        namespace.setdefault("replace", replace)
        _inject_module_helpers(namespace, module_name)
        if donor is cpu_cache_methods.MambaComponent:
            from sglang.srt.mem_cache.unified_cache.components.base import ExternalLinkerLoadPhase
            namespace.setdefault("ExternalLinkerLoadPhase", ExternalLinkerLoadPhase)
        if helpers:
            cls = getattr(native, class_name)
            for name in helpers:
                source = vars(donor)[name]
                current = vars(cls).get(name)
                if getattr(current, "__code__", None) is not source.__code__:
                    setattr(cls, name, _bind(source, namespace))
        if method is None:
            return original(*args, **kwargs)
        if bound is None:
            bound = _bind(vars(donor)[method], namespace)
        return bound(*args, **kwargs)

    return around


def _storage_attachment_init(original, self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if is_hcu() and not hasattr(self._cache, "_storage_prefetch_wait"):
        self._cache._storage_prefetch_wait = {}
    return result


def register_hooks():
    if not is_hcu():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for module_name, class_name, donor_name, methods, helpers in _GROUPS:
        if donor_name == "Tensorcast" and find_spec("tensorcast") is None:
            continue
        donor = getattr(cpu_cache_methods, donor_name)
        parent = module_name + ("." + class_name if class_name else "")
        for name in methods:
            HookRegistry.register(
                parent + "." + name,
                _make_hook(module_name, class_name, donor, name, helpers),
                HookType.AROUND,
            )
        if helpers and class_name and "__init__" not in methods:
            HookRegistry.register(
                parent + ".__init__",
                _make_hook(module_name, class_name, donor, None, helpers),
                HookType.AROUND,
            )

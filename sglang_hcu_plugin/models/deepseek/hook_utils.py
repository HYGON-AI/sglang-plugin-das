"""Shared helpers for DeepSeek V2 method-level plugin hooks."""

from __future__ import annotations

import logging
import sys
import types
from typing import Iterable

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

logger = logging.getLogger(__name__)

_POST_APPLY_REGISTERED = False


def register_method_replaces_from_module(
    base_module: str,
    class_name: str,
    method_names: Iterable[str],
    source_module: types.ModuleType,
    *,
    source_method_map: dict[str, str] | None = None,
) -> int:
    """Register REPLACE hooks using functions defined in *source_module*."""
    source_method_map = source_method_map or {}
    count = 0
    for name in method_names:
        src_name = source_method_map.get(name, name)
        HookRegistry.register(
            f"{base_module}.{class_name}.{name}",
            getattr(source_module, src_name),
            HookType.REPLACE,
        )
        count += 1
    return count


def register_mixin_method_replaces_from_module(
    mixin_module: str,
    mixin_class: str,
    method_specs: Iterable[tuple[str, str]],
    source_module: types.ModuleType,
) -> int:
    """Register REPLACE hooks on mixin methods from a plain module."""
    count = 0
    base = f"{mixin_module}.{mixin_class}"
    for target_name, source_name in method_specs:
        HookRegistry.register(
            f"{base}.{target_name}",
            getattr(source_module, source_name),
            HookType.REPLACE,
        )
        count += 1
    return count


def register_method_replaces(
    base_module: str,
    class_name: str,
    method_names: Iterable[str],
    source_cls: type,
    *,
    source_method_map: dict[str, str] | None = None,
) -> int:
    """Register REPLACE hooks for unbound methods taken from *source_cls*.

    *source_method_map* maps target method name -> source method name when they
    differ (e.g. upstream ``_get_mla_kv_buffer_from_fp8_for_nsa``).
    """
    source_method_map = source_method_map or {}
    count = 0
    for name in method_names:
        src_name = source_method_map.get(name, name)
        HookRegistry.register(
            f"{base_module}.{class_name}.{name}",
            getattr(source_cls, src_name),
            HookType.REPLACE,
        )
        count += 1
    return count


def register_mixin_method_replaces(
    mixin_module: str,
    mixin_class: str,
    method_specs: Iterable[tuple[str, str]],
    source_cls: type,
) -> int:
    """Register REPLACE hooks on mixin methods.

    Each spec is ``(target_method, source_method)``. Names may differ when fork
    renamed a helper (same tuple with identical strings otherwise).
    """
    count = 0
    base = f"{mixin_module}.{mixin_class}"
    for target_name, source_name in method_specs:
        HookRegistry.register(
            f"{base}.{target_name}",
            getattr(source_cls, source_name),
            HookType.REPLACE,
        )
        count += 1
    return count


def register_function_replaces(
    base_module: str,
    func_names: Iterable[str],
    source_module: types.ModuleType,
) -> int:
    count = 0
    for name in func_names:
        HookRegistry.register(
            f"{base_module}.{name}",
            getattr(source_module, name),
            HookType.REPLACE,
        )
        count += 1
    return count


def _propagate_module_attr_patch(
    base_module: types.ModuleType,
    attr_name: str,
    new_value: object,
) -> int:
    old_value = getattr(base_module, attr_name, None)
    if old_value is new_value:
        return 0
    patched_count = 0
    for mod in list(sys.modules.values()):
        if mod is base_module or mod is None:
            continue
        if not isinstance(mod, types.ModuleType):
            continue
        try:
            mod_vars = vars(mod)
        except TypeError:
            continue
        if mod_vars.get(attr_name) is old_value:
            try:
                setattr(mod, attr_name, new_value)
                patched_count += 1
            except (AttributeError, TypeError):
                pass
    return patched_count


def patch_deepseek_common_module_attrs() -> None:
    """Patch fork module-level constants onto upstream deepseek_common.utils."""
    import sglang.srt.models.deepseek_common.utils as base
    from sglang_hcu_plugin.models.deepseek.deepseek_common import constants as C

    for name in C.MODULE_PATCH_ATTRS:
        setattr(base, name, getattr(C, name))
        extra = _propagate_module_attr_patch(base, name, getattr(C, name))
        if extra:
            logger.debug(
                "[sglang_hcu_plugin] Propagated deepseek_common.utils.%s to %d module(s)",
                name,
                extra,
            )


def patch_deepseek_common_attention_backend() -> None:
    """Register HCU MLA attention backend handler."""
    from sglang_hcu_plugin.models.deepseek.deepseek_common.hooks import (
        attention_backend_handler,
    )

    attention_backend_handler.register_attention_hcu_mla_backend()


def patch_deepseek_v2_module_attrs() -> None:
    """Patch fork module-level constants onto upstream deepseek_v2."""
    import sglang.srt.models.deepseek_v2 as base
    from sglang_hcu_plugin.models.deepseek.deepseek_v2 import constants as C

    for name in C.MODULE_PATCH_ATTRS:
        setattr(base, name, getattr(C, name))
        extra = _propagate_module_attr_patch(base, name, getattr(C, name))
        if extra:
            logger.debug(
                "[sglang_hcu_plugin] Propagated deepseek_v2.%s to %d module(s)",
                name,
                extra,
            )

    # Ensure patched list object is shared (mutations via add_forward_absorb_*).
    base.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS = (
        C.FORWARD_ABSORB_CORE_ATTENTION_BACKENDS
    )


def patch_deepseek_v4_module_attrs() -> None:
    """Patch fork module-level constants onto upstream deepseek_v4."""
    import sglang.srt.models.deepseek_v4 as base
    from sglang_hcu_plugin.models.deepseek.deepseek_v4 import constants as C

    for name in C.MODULE_PATCH_ATTRS:
        setattr(base, name, getattr(C, name))
        extra = _propagate_module_attr_patch(base, name, getattr(C, name))
        if extra:
            logger.debug(
                "[sglang_hcu_plugin] Propagated deepseek_v4.%s to %d module(s)",
                name,
                extra,
            )


def ensure_post_apply_module_patch() -> None:
    """Wrap HookRegistry.apply_hooks once to patch module attrs after hooks."""
    global _POST_APPLY_REGISTERED
    if _POST_APPLY_REGISTERED:
        return

    original_apply = HookRegistry.apply_hooks.__func__

    @classmethod
    def apply_hooks_with_module_patch(cls) -> None:
        original_apply(cls)
        patch_deepseek_common_module_attrs()
        patch_deepseek_common_attention_backend()
        patch_deepseek_v2_module_attrs()
        patch_deepseek_v4_module_attrs()

    HookRegistry.apply_hooks = apply_hooks_with_module_patch  # type: ignore[method-assign]
    _POST_APPLY_REGISTERED = True

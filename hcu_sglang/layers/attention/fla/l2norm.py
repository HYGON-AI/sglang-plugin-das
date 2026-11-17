# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""REPLACE hook implementation for the fla L2 normalization kernel."""

from sglang.srt.plugins.hook_registry import HookRegistry, HookType

from hcu_sglang.utils.detection import is_hcu
from hcu_sglang.utils.module_patch import get_original_attr

L2NORM_MODULE = "sglang.kernels.ops.attention.fla.l2norm"


def register() -> None:
    HookRegistry.register(
        "sglang.kernels.ops.attention.fla.l2norm.l2norm_fwd",
        _make_replacement(L2NORM_MODULE, "l2norm_fwd"),
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.kernels.ops.attention.fla.l2norm.l2norm",
        _make_replacement(L2NORM_MODULE, "l2norm"),
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.kernels.ops.attention.fla.l2norm.l2_norm",
        _make_replacement(L2NORM_MODULE, "l2_norm"),
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.kernels.ops.attention.fla.chunk.l2norm_fwd",
        _make_replacement("sglang.kernels.ops.attention.fla.chunk", "l2norm_fwd"),
        HookType.REPLACE,
    )
    HookRegistry.register(
        "sglang.kernels.ops.attention.fla.kda.l2norm_fwd",
        _make_replacement("sglang.kernels.ops.attention.fla.kda", "l2norm_fwd"),
        HookType.REPLACE,
    )


def _make_replacement(module, name):
    def replacement(*args, **kwargs):
        if not is_hcu():
            return get_original_attr(module, name)(*args, **kwargs)
        from hcu_sglang.layers.attention.fla import l2norm_source

        return getattr(l2norm_source, name)(*args, **kwargs)

    replacement.__name__ = f"replacement_{name}"
    return replacement

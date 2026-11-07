# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single forward_cuda dispatcher for the unquantized fused-MoE method."""

from __future__ import annotations

from typing import Any, Callable


def around_forward_cuda(
    original: Callable[..., Any], method: Any, layer: Any, dispatch_output: Any
) -> Any:
    from hcu_sglang.layers.quantization import unquantized_aiter, unquantized_marlin

    if unquantized_marlin._enabled():
        return unquantized_marlin.around_forward_cuda(
            original, method, layer, dispatch_output
        )
    return unquantized_aiter.around_forward_cuda(
        original, method, layer, dispatch_output
    )


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        "sglang.srt.layers.quantization.unquant."
        "UnquantizedFusedMoEMethod.forward_cuda",
        around_forward_cuda,
        HookType.AROUND,
    )

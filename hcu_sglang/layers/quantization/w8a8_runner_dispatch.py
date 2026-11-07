# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Single create_moe_runner AROUND dispatcher for the W8A8 MoE methods."""

from __future__ import annotations

from typing import Any, Callable


def around_create_moe_runner(
    original: Callable[..., Any], quant_method, layer, moe_runner_config
) -> None:
    from hcu_sglang.layers.quantization import w8a8_aiter, w8a8_lightop

    if w8a8_lightop._requested_lightop():
        return w8a8_lightop.around_create_moe_runner(
            original, quant_method, layer, moe_runner_config
        )
    return w8a8_aiter.around_create_moe_runner(
        original, quant_method, layer, moe_runner_config
    )


def register_hooks() -> None:
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    for module, class_name in (
        ("sglang.srt.layers.quantization.w8a8_int8", "W8A8Int8MoEMethod"),
        ("sglang.srt.layers.quantization.w8a8_fp8", "W8A8FP8MoEMethod"),
    ):
        HookRegistry.register(
            f"{module}.{class_name}.create_moe_runner",
            around_create_moe_runner,
            HookType.AROUND,
        )
